//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/registry.cpp
//
//
//===----------------------------------------------------------------------===//

#include "registry.hpp"

#include <utility>

// The Arrow C data and stream interface structs, under their standard guards.
#include "duckdb_v2.h"

namespace duckdb_python {
namespace {

const char *const kScanFunction = "python_object_scan";
const char *const kStreamCapsule = "arrow_array_stream";
const char *const kSchemaCapsule = "arrow_schema";
/// The engine's standard vector size, which is what the output chunk handed to the exec callback is allocated for.
constexpr cxx::idx_t kBatchRows = 2048;

/// ASCII letters only, as the engine and the Python side fold identifiers; a locale-aware fold would rewrite the
/// bytes of a non-ASCII name.
std::string Fold(const std::string &name) {
	std::string folded = name;
	for (auto &c : folded) {
		if (c >= 'A' && c <= 'Z') {
			c = static_cast<char>(c - 'A' + 'a');
		}
	}
	return folded;
}

std::string StreamError(ArrowArrayStream &stream) {
	const char *text = stream.get_last_error ? stream.get_last_error(&stream) : nullptr;
	return text ? text : "no error message";
}

/// The stream a capsule carries, checked to be one that has not been released.
ArrowArrayStream &StreamOf(nb::handle capsule, const std::string &name) {
	if (!PyCapsule_IsValid(capsule.ptr(), kStreamCapsule)) {
		const char *found = PyCapsule_CheckExact(capsule.ptr()) ? PyCapsule_GetName(capsule.ptr()) : nullptr;
		throw cxx::InvalidInputException("the object registered as '" + name + "' did not export an '" +
		                                 kStreamCapsule + "' capsule but " +
		                                 (found ? "a '" + std::string(found) + "' capsule" : "something else"));
	}
	auto *stream = static_cast<ArrowArrayStream *>(PyCapsule_GetPointer(capsule.ptr(), kStreamCapsule));
	if (stream == nullptr || stream->release == nullptr) {
		throw cxx::InvalidInputException("the stream registered as '" + name + "' is released already");
	}
	return *stream;
}

/// Export a fresh stream from the registered object, or the capsule itself when that is what was registered.
nb::object ExportStream(Registered &entry) {
	if (PyCapsule_CheckExact(entry.object.ptr())) {
		return entry.object;
	}
	return entry.object.attr("__arrow_c_stream__")();
}

/// Reads the registered object's schema into `out` without consuming a stream: from `__arrow_c_schema__`, from a
/// `schema` attribute that has it, or from the stream itself, which for a capsule is a peek and for any other object
/// is a fresh export.
void SchemaOf(Registered &entry, ArrowSchema &out) {
	nb::object object = entry.object;
	nb::object exporter;
	if (!PyCapsule_CheckExact(object.ptr())) {
		if (nb::hasattr(object, "__arrow_c_schema__")) {
			exporter = object;
		} else if (nb::hasattr(object, "schema") && nb::hasattr(object.attr("schema"), "__arrow_c_schema__")) {
			exporter = object.attr("schema");
		}
	}
	if (exporter.is_valid()) {
		nb::object capsule = exporter.attr("__arrow_c_schema__")();
		if (!PyCapsule_IsValid(capsule.ptr(), kSchemaCapsule)) {
			throw cxx::InvalidInputException("the object registered as '" + entry.name +
			                                 "' did not export an Arrow schema capsule");
		}
		auto *schema = static_cast<ArrowSchema *>(PyCapsule_GetPointer(capsule.ptr(), kSchemaCapsule));
		if (schema == nullptr || schema->release == nullptr) {
			throw cxx::InvalidInputException("the object registered as '" + entry.name +
			                                 "' exported a released Arrow schema");
		}
		// Moved out of the capsule, which then has nothing left to release.
		out = *schema;
		schema->release = nullptr;
		return;
	}
	nb::object capsule = ExportStream(entry);
	auto &stream = StreamOf(capsule, entry.name);
	if (stream.get_schema(&stream, &out) != 0) {
		throw cxx::InvalidInputException("reading the schema of the stream registered as '" + entry.name +
		                                 "' failed: " + StreamError(stream));
	}
}

std::string ReadAlready(const Registered &entry) {
	return "the stream registered as '" + entry.name +
	       "' has been read already, by an earlier query or by another reference to it in this one; a stream can be "
	       "read once, so register it again, read it through one CTE, or load it into a table first";
}

struct ScanUserData {
	std::shared_ptr<Registry> registry;
};

/// The entry a query bound over and its schema as read then, which the scan's importer is built from.
struct ScanBind {
	ScanBind(std::shared_ptr<Registered> entry, ArrowSchema schema) : entry(std::move(entry)), schema(schema) {
	}
	ScanBind(const ScanBind &) = delete;
	ScanBind &operator=(const ScanBind &) = delete;
	~ScanBind() {
		if (schema.release != nullptr) {
			schema.release(&schema);
		}
	}

	std::shared_ptr<Registered> entry;
	ArrowSchema schema;
};

/// One scan's stream and the importer turning its arrays into chunks.
struct ScanState {
	ScanState(nb::object capsule, ArrowArrayStream &stream, cxx::ArrowImporter importer)
	    : capsule(std::move(capsule)), stream(stream), importer(std::move(importer)) {
	}

	/// Torn down from an engine thread, so the stream is released and the capsule dropped under the GIL.
	~ScanState() {
		nb::gil_scoped_acquire gil;
		if (stream.release != nullptr) {
			stream.release(&stream);
		}
		current.reset();
		capsule.reset();
	}

	nb::object capsule;
	ArrowArrayStream &stream;
	cxx::ArrowImporter importer;
	/// The chunk the output vectors reference, kept until the next batch replaces it.
	std::optional<cxx::DataChunk> current;
	bool exhausted = false;
};

/// Resolved by name at bind time, so a query sees the object registered under the name when it binds, as it would
/// see a table.
void PyScanBind(cxx::TableFunction::BindInput &input) {
	auto &registry = *input.GetUserData<ScanUserData>().registry;
	const auto name = std::string(input.GetArgument(0).Get<cxx::varchar_t>());
	auto entry = registry.ByName(name);
	if (!entry) {
		throw cxx::InvalidInputException("nothing is registered as '" + name + "'");
	}
	if (entry->one_shot) {
		std::lock_guard<std::mutex> guard(entry->read_lock);
		if (entry->read) {
			throw cxx::InvalidInputException(ReadAlready(*entry));
		}
	}
	ArrowSchema schema {};
	{
		nb::gil_scoped_acquire gil;
		try {
			SchemaOf(*entry, schema);
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("reading the schema of the object registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		}
	}
	try {
		cxx::ArrowImporter importer(input.GetContext(), schema, kBatchRows);
		auto resolved = importer.GetSchema();
		for (cxx::idx_t i = 0; i < resolved.GetFieldCount(); i++) {
			input.AddResultColumn(std::string(resolved.GetFieldName(i)), resolved.GetFieldType(i));
		}
	} catch (...) {
		schema.release(&schema);
		throw;
	}
	// The bind data owns the schema from here on.
	input.SetBindData<ScanBind>(std::move(entry), schema);
}

/// Exports the stream and builds the importer over the schema read at bind; an array of another shape is
/// refused by the importer when it is appended.
void OpenStream(const ScanBind &bound, cxx::TableFunction::InitGlobalInput &input) {
	auto &entry = *bound.entry;
	nb::gil_scoped_acquire gil;
	nb::object capsule;
	try {
		capsule = ExportStream(entry);
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("exporting a stream from the object registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
	auto &stream = StreamOf(capsule, entry.name);
	// The importer reads the schema without consuming it, so the bind data keeps ownership.
	cxx::ArrowImporter importer(input.GetContext(), const_cast<ArrowSchema &>(bound.schema), kBatchRows);
	input.SetGlobalState<ScanState>(std::move(capsule), stream, std::move(importer));
}

void PyScanInitGlobal(cxx::TableFunction::InitGlobalInput &input) {
	const auto &bound = input.GetBindData<ScanBind>();
	auto &entry = *bound.entry;
	if (!entry.one_shot) {
		OpenStream(bound, input);
		return;
	}
	// Claimed before the export so two scans cannot both take the one stream, and given back when opening fails
	// before a row was read, so a failed export does not poison the entry.
	{
		std::lock_guard<std::mutex> guard(entry.read_lock);
		if (entry.read) {
			throw cxx::InvalidInputException(ReadAlready(entry));
		}
		entry.read = true;
	}
	try {
		OpenStream(bound, input);
	} catch (...) {
		std::lock_guard<std::mutex> guard(entry.read_lock);
		entry.read = false;
		throw;
	}
}

/// Hands the next chunk to the engine: the importer's chunk is referenced, not copied, and stays alive here.
void HandOver(ScanState &state, cxx::DataChunk chunk, cxx::TableFunction::ExecInput &input) {
	auto output = input.GetOutputChunk();
	const auto columns = output.GetVectorCount();
	if (chunk.GetVectorCount() != columns) {
		throw cxx::InvalidInputException("an Arrow batch carried " + std::to_string(chunk.GetVectorCount()) +
		                                 " columns where the schema declared " + std::to_string(columns));
	}
	for (cxx::idx_t i = 0; i < columns; i++) {
		output.GetVector(i).Reference(chunk.GetVector(i));
	}
	output.GetVector(0).SetSize(chunk.GetRowCount());
	state.current = std::move(chunk);
}

void PyScanExec(cxx::TableFunction::ExecInput &input) {
	auto &state = input.GetGlobalState<ScanState>();
	const auto &entry = *input.GetBindData<ScanBind>().entry;
	for (;;) {
		auto chunk = state.importer.NextChunk();
		if (chunk && chunk.GetRowCount() > 0) {
			HandOver(state, std::move(chunk), input);
			return;
		}
		if (state.exhausted) {
			return;
		}
		ArrowArray array {};
		{
			// A stream backed by Python code runs Python in get_next, so the GIL is held for every pull.
			nb::gil_scoped_acquire gil;
			if (state.stream.get_next(&state.stream, &array) != 0) {
				throw cxx::InvalidInputException("reading the stream registered as '" + entry.name +
				                                 "' failed: " + StreamError(state.stream));
			}
		}
		if (array.release == nullptr) {
			state.exhausted = true;
			state.importer.Flush();
			continue;
		}
		try {
			state.importer.Append(array, true, false);
		} catch (...) {
			if (array.release != nullptr) {
				array.release(&array);
			}
			throw;
		}
	}
}

void ReplaceRegisteredName(cxx::ReplacementScan::Input &input) {
	auto &registry = *input.GetUserData<ScanUserData>().registry;
	const auto name = input.GetName();
	if (name.GetPartCount() != 1) {
		return;
	}
	auto entry = registry.ByName(std::string(name.GetPart(0)));
	if (!entry) {
		return;
	}
	input.SetFunctionName(kScanFunction);
	auto context = input.GetContext();
	input.AddArgument(context.CreateValue(cxx::varchar_t(name.GetPart(0))));
}

} // namespace

Registered::Registered(std::string name, nb::object object, bool one_shot)
    : name(std::move(name)), object(std::move(object)), one_shot(one_shot) {
}

Registered::~Registered() {
	nb::gil_scoped_acquire gil;
	object.reset();
}

void Registry::Add(const std::string &name, nb::object object, bool one_shot) {
	// The replaced entry, if any, is dropped outside the lock: its destructor takes the GIL, which a thread waiting
	// for this lock may hold.
	std::shared_ptr<Registered> replaced;
	{
		std::lock_guard<std::mutex> guard(lock);
		auto &slot = by_name[Fold(name)];
		replaced = std::move(slot);
		slot = std::make_shared<Registered>(name, std::move(object), one_shot);
	}
}

bool Registry::Remove(const std::string &name) {
	std::shared_ptr<Registered> removed;
	{
		std::lock_guard<std::mutex> guard(lock);
		auto it = by_name.find(Fold(name));
		if (it == by_name.end()) {
			return false;
		}
		removed = std::move(it->second);
		by_name.erase(it);
	}
	return true;
}

std::shared_ptr<Registered> Registry::ByName(const std::string &name) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = by_name.find(Fold(name));
	return it == by_name.end() ? nullptr : it->second;
}

std::vector<nb::handle> Registry::Objects() {
	std::lock_guard<std::mutex> guard(lock);
	std::vector<nb::handle> objects;
	objects.reserve(by_name.size());
	for (const auto &[_, entry] : by_name) {
		objects.push_back(entry->object);
	}
	return objects;
}

void Registry::Clear() {
	std::vector<std::shared_ptr<Registered>> dropped;
	{
		std::lock_guard<std::mutex> guard(lock);
		dropped.reserve(by_name.size());
		for (auto &[_, entry] : by_name) {
			dropped.push_back(std::move(entry));
		}
		by_name.clear();
	}
}

void InstallRegistryScan(cxx::Instance &instance, std::shared_ptr<Registry> registry) {
	auto connection = instance.Connect();
	auto function = cxx::TableFunction::Create(connection);
	function.SetName(kScanFunction);
	function.WithSignature(
	    [&](cxx::FunctionSignature &signature) { signature.AddParameter("name", connection.ParseType("VARCHAR")); });
	function.SetUserData<ScanUserData>(ScanUserData {registry});
	function.SetBindCallback(&PyScanBind);
	function.SetInitGlobalCallback(&PyScanInitGlobal);
	function.SetExecCallback(&PyScanExec);
	function.Register();

	auto scan = cxx::ReplacementScan::Create(instance);
	scan.SetUserData<ScanUserData>(ScanUserData {std::move(registry)});
	scan.SetCallback(&ReplaceRegisteredName);
	scan.Register();
}

} // namespace duckdb_python
