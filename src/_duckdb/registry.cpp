//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/registry.cpp
//
//
//===----------------------------------------------------------------------===//

#include "registry.hpp"

#include <cstring>
#include <utility>

// The Arrow C data and stream interface structs, under their standard guards.
#include "duckdb_v2.h"

namespace duckdb_python {
namespace {

const char *const kScanFunction = "python_object_scan";
const char *const kStreamCapsule = "arrow_array_stream";
const char *const kSchemaCapsule = "arrow_schema";
const char *const kArrayCapsule = "arrow_array";
/// A struct array with no validity buffer still declares one buffer slot, holding null.
const void *kNoBuffers[1] = {nullptr};
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

bool IsBatch(const ArrowSchema &schema) {
	return schema.format != nullptr && std::strcmp(schema.format, "+s") == 0;
}

/// The moved-in child a wrapper owns, released and freed with the wrapper.
template <class T>
struct Wrapped {
	T *child;
	T **children;
};

template <class T>
void ReleaseWrapped(T *wrapper) {
	auto *owned = static_cast<Wrapped<T> *>(wrapper->private_data);
	if (owned->child->release != nullptr) {
		owned->child->release(owned->child);
	}
	delete owned->child;
	delete[] owned->children;
	delete owned;
	wrapper->release = nullptr;
}

/// Turns a schema that is not a batch into a batch of one column: a struct with the schema as its only child.
void WrapAsBatch(ArrowSchema &schema) {
	auto *child = new ArrowSchema(schema);
	auto **children = new ArrowSchema *[1] {child};
	schema = ArrowSchema {};
	schema.format = "+s";
	schema.name = "";
	schema.n_children = 1;
	schema.children = children;
	schema.release = &ReleaseWrapped<ArrowSchema>;
	schema.private_data = new Wrapped<ArrowSchema> {child, children};
}

/// The array counterpart: a struct array with no validity buffer whose only child is the array.
void WrapAsBatch(ArrowArray &array) {
	auto *child = new ArrowArray(array);
	auto **children = new ArrowArray *[1] {child};
	array = ArrowArray {};
	array.length = child->length;
	array.null_count = 0;
	array.offset = 0;
	array.n_buffers = 1;
	array.buffers = kNoBuffers;
	array.n_children = 1;
	array.children = children;
	array.release = &ReleaseWrapped<ArrowArray>;
	array.private_data = new Wrapped<ArrowArray> {child, children};
}

/// Moves the struct out of a capsule, leaving the capsule nothing to release.
template <class T>
T TakeFromCapsule(nb::handle capsule, const char *kind, const std::string &name) {
	if (!PyCapsule_IsValid(capsule.ptr(), kind)) {
		throw cxx::InvalidInputException("the object registered as '" + name + "' did not hand out an '" + kind +
		                                 "' capsule");
	}
	auto *held = static_cast<T *>(PyCapsule_GetPointer(capsule.ptr(), kind));
	if (held == nullptr || held->release == nullptr) {
		throw cxx::InvalidInputException("the object registered as '" + name + "' handed out a released '" + kind +
		                                 "' capsule");
	}
	T taken = *held;
	held->release = nullptr;
	return taken;
}

/// What a source answered: a stream capsule, or the schema and array capsules of one array, which is one batch.
struct Exported {
	nb::object stream;
	nb::object schema;
	nb::object array;
	bool projected = false;

	bool IsArray() const {
		return array.is_valid();
	}
};

Exported ExportStream(Registered &entry, const std::vector<cxx::idx_t> *columns) {
	nb::object request = nb::none();
	if (columns != nullptr) {
		nb::list wanted;
		for (const auto column : *columns) {
			wanted.append(nb::int_(static_cast<uint64_t>(column)));
		}
		request = std::move(wanted);
	}
	nb::object answer = entry.object.attr("stream")(request);
	if (!nb::isinstance<nb::tuple>(answer) || nb::len(answer) != 2) {
		throw cxx::InvalidInputException("the source registered as '" + entry.name +
		                                 "' did not answer stream() with a (capsule, projected) pair");
	}
	auto pair = nb::cast<nb::tuple>(answer);
	Exported exported;
	exported.projected = nb::cast<bool>(pair[1]);
	nb::object data = nb::borrow(pair[0]);
	if (nb::isinstance<nb::tuple>(data) && nb::len(data) == 2) {
		auto both = nb::cast<nb::tuple>(data);
		exported.schema = nb::borrow(both[0]);
		exported.array = nb::borrow(both[1]);
	} else {
		exported.stream = std::move(data);
	}
	return exported;
}

/// Reads the source's schema into `out` without consuming data: from `__arrow_c_schema__` when the source has
/// it, else by peeking the schema of a full stream, which a raw capsule answers without being read.
void SchemaOf(Registered &entry, ArrowSchema &out) {
	nb::object object = entry.object;
	if (nb::hasattr(object, "__arrow_c_schema__")) {
		nb::object capsule = object.attr("__arrow_c_schema__")();
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
	auto exported = ExportStream(entry, nullptr);
	if (exported.IsArray()) {
		// The array capsule is dropped unread, which releases it.
		out = TakeFromCapsule<ArrowSchema>(exported.schema, kSchemaCapsule, entry.name);
		return;
	}
	auto &stream = StreamOf(exported.stream, entry.name);
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

/// The entry a query bound over and the Arrow names of the columns it declared, in order.
struct ScanBind {
	std::shared_ptr<Registered> entry;
	std::vector<std::string> names;
};

std::string NameOf(const ArrowSchema &schema, cxx::idx_t index) {
	const auto *child = schema.children[index];
	return child->name ? child->name : "";
}

/// One scan's stream and the importer turning its arrays into chunks.
struct ScanState {
	ScanState(nb::object capsule, ArrowArrayStream *stream, ArrowArray single, bool wrap, cxx::ArrowImporter importer,
	          std::vector<cxx::idx_t> picks)
	    : capsule(std::move(capsule)), stream(stream), single(single), wrap(wrap), importer(std::move(importer)),
	      picks(std::move(picks)) {
	}

	/// Torn down from an engine thread, so the stream is released and the capsule dropped under the GIL.
	~ScanState() {
		nb::gil_scoped_acquire gil;
		if (stream != nullptr && stream->release != nullptr) {
			stream->release(stream);
		}
		if (single.release != nullptr) {
			single.release(&single);
		}
		current.reset();
		capsule.reset();
	}

	nb::object capsule;
	/// The stream the arrays come from, or null when the source handed over one array.
	ArrowArrayStream *stream;
	/// The one array, until it is appended.
	ArrowArray single;
	/// Whether the source's arrays are plain values rather than batches, to be wrapped as a one-column batch.
	bool wrap;
	cxx::ArrowImporter importer;
	/// Which of the stream's columns each output vector takes; empty when the stream holds exactly the output.
	std::vector<cxx::idx_t> picks;
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
	if (!IsBatch(schema)) {
		WrapAsBatch(schema);
	}
	std::vector<std::string> names;
	try {
		cxx::ArrowImporter importer(input.GetContext(), schema, kBatchRows);
		auto resolved = importer.GetSchema();
		for (cxx::idx_t i = 0; i < resolved.GetFieldCount(); i++) {
			// A plain array has no column name of its own; the importer would call it v0.
			const auto raw = NameOf(schema, i);
			input.AddResultColumn(raw.empty() ? std::string("value") : std::string(resolved.GetFieldName(i)),
			                      resolved.GetFieldType(i));
			names.push_back(raw);
		}
	} catch (...) {
		schema.release(&schema);
		throw;
	}
	schema.release(&schema);
	input.SetBindData<ScanBind>(ScanBind {std::move(entry), std::move(names)});
}

/// Exports the stream, narrowed to the columns the query uses when the source can, and builds the importer over
/// the stream's own schema. The scan takes columns by position, so a source that answers with other columns
/// than it said, in width or in order as far as the names tell, is refused rather than read wrongly.
void OpenStream(const ScanBind &bound, cxx::TableFunction::InitGlobalInput &input) {
	auto &entry = *bound.entry;
	const auto declared = static_cast<cxx::idx_t>(bound.names.size());
	std::vector<cxx::idx_t> requested;
	bool identity = input.GetColumnCount() == declared;
	for (cxx::idx_t i = 0; i < input.GetColumnCount(); i++) {
		requested.push_back(input.GetColumnIndex(i));
		identity = identity && requested.back() == i;
	}
	nb::gil_scoped_acquire gil;
	Exported exported;
	try {
		exported = ExportStream(entry, identity ? nullptr : &requested);
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("exporting a stream from the object registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
	ArrowArrayStream *stream = nullptr;
	ArrowArray single {};
	ArrowSchema schema {};
	if (exported.IsArray()) {
		schema = TakeFromCapsule<ArrowSchema>(exported.schema, kSchemaCapsule, entry.name);
		try {
			single = TakeFromCapsule<ArrowArray>(exported.array, kArrayCapsule, entry.name);
		} catch (...) {
			schema.release(&schema);
			throw;
		}
	} else {
		stream = &StreamOf(exported.stream, entry.name);
		if (stream->get_schema(stream, &schema) != 0) {
			throw cxx::InvalidInputException("reading the schema of the stream registered as '" + entry.name +
			                                 "' failed: " + StreamError(*stream));
		}
	}
	const bool wrap = !IsBatch(schema);
	if (wrap) {
		WrapAsBatch(schema);
		if (exported.IsArray()) {
			WrapAsBatch(single);
		}
	}
	try {
		const auto fields = static_cast<cxx::idx_t>(schema.n_children);
		std::vector<cxx::idx_t> picks;
		if (exported.projected) {
			if (fields != requested.size()) {
				throw cxx::InvalidInputException("the source registered as '" + entry.name + "' answered with " +
				                                 std::to_string(fields) + " columns where " +
				                                 std::to_string(requested.size()) + " were requested");
			}
		} else {
			if (fields != declared) {
				throw cxx::InvalidInputException("the source registered as '" + entry.name + "' answered with " +
				                                 std::to_string(fields) + " columns where it declared " +
				                                 std::to_string(declared));
			}
			if (!identity) {
				picks = requested;
			}
		}
		for (cxx::idx_t i = 0; i < fields; i++) {
			const auto expected = exported.projected ? requested[i] : i;
			if (NameOf(schema, i) != bound.names[expected]) {
				throw cxx::InvalidInputException("the source registered as '" + entry.name + "' answered with column '" +
				                                 NameOf(schema, i) + "' at position " + std::to_string(i) + " where '" +
				                                 bound.names[expected] + "' was expected");
			}
		}
		cxx::ArrowImporter importer(input.GetContext(), schema, kBatchRows);
		schema.release(&schema);
		nb::object keep = exported.IsArray() ? std::move(exported.array) : std::move(exported.stream);
		input.SetGlobalState<ScanState>(std::move(keep), stream, single, wrap, std::move(importer), std::move(picks));
	} catch (...) {
		if (schema.release != nullptr) {
			schema.release(&schema);
		}
		if (single.release != nullptr) {
			single.release(&single);
		}
		throw;
	}
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
	// The engine sizes a batch from its first output vector, so it never asks for zero columns; a count over the
	// scan still asks for one. Should that change, an empty output would need another way to carry the row count.
	if (columns == 0) {
		throw cxx::InvalidInputException("the scan of '" + input.GetBindData<ScanBind>().entry->name +
		                                 "' was asked for no columns, which it cannot report rows through");
	}
	const auto needed = state.picks.empty() ? columns : state.picks.size();
	if (chunk.GetVectorCount() < needed) {
		throw cxx::InvalidInputException("an Arrow batch carried " + std::to_string(chunk.GetVectorCount()) +
		                                 " columns where " + std::to_string(needed) + " were expected");
	}
	for (cxx::idx_t i = 0; i < columns; i++) {
		output.GetVector(i).Reference(chunk.GetVector(state.picks.empty() ? i : state.picks[i]));
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
		if (state.stream == nullptr) {
			array = state.single;
			state.single.release = nullptr;
		} else {
			// A stream backed by Python code runs Python in get_next, so the GIL is held for every pull.
			nb::gil_scoped_acquire gil;
			if (state.stream->get_next(state.stream, &array) != 0) {
				throw cxx::InvalidInputException("reading the stream registered as '" + entry.name +
				                                 "' failed: " + StreamError(*state.stream));
			}
			if (array.release != nullptr && state.wrap) {
				WrapAsBatch(array);
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
	function.SetProjectionPushdown(true);
	function.Register();

	auto scan = cxx::ReplacementScan::Create(instance);
	scan.SetUserData<ScanUserData>(ScanUserData {std::move(registry)});
	scan.SetCallback(&ReplaceRegisteredName);
	scan.Register();
}

} // namespace duckdb_python
