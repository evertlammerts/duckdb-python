//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/scan.cpp
//
//
//===----------------------------------------------------------------------===//

#include "scan.hpp"

#include "arrowc.hpp"
#include "predicate.hpp"

#include <utility>

namespace duckdb_python {
namespace {

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

Exported ExportStream(Registered &entry, const std::vector<cxx::idx_t> *columns,
                      const std::vector<nb::object> &filters) {
	nb::object request = nb::none();
	if (columns != nullptr) {
		nb::list wanted;
		for (const auto column : *columns) {
			wanted.append(nb::int_(static_cast<uint64_t>(column)));
		}
		request = std::move(wanted);
	}
	nb::list promised;
	for (const auto &filter : filters) {
		promised.append(filter);
	}
	nb::object answer = entry.object.attr("stream")(request, nb::tuple(promised));
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
	auto exported = ExportStream(entry, nullptr, {});
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
	std::shared_ptr<ModuleState> module;
	/// The engine's standard vector size, which is what the output chunk handed to the exec callback is allocated
	/// for, so the importer produces chunks of at most that many rows.
	cxx::idx_t batch_rows;
};

/// The entry a query bound over, the Arrow names and types of the columns it declared, in order, and the predicates
/// the source promised to apply, as frame expressions.
struct ScanBind {
	ScanBind(std::shared_ptr<Registered> entry, std::vector<std::string> names, std::vector<cxx::LogicalTypeId> types)
	    : entry(std::move(entry)), names(std::move(names)), types(std::move(types)) {
	}

	/// Freed from engine threads too, so the predicates are dropped under the GIL.
	~ScanBind() {
		nb::gil_scoped_acquire gil;
		filters.clear();
		entry.reset();
	}

	std::shared_ptr<Registered> entry;
	std::vector<std::string> names;
	std::vector<cxx::LogicalTypeId> types;
	std::vector<nb::object> filters;
};

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
	std::vector<cxx::LogicalTypeId> types;
	try {
		cxx::ArrowImporter importer(input.GetContext(), schema, input.GetUserData<ScanUserData>().batch_rows);
		auto resolved = importer.GetSchema();
		for (cxx::idx_t i = 0; i < resolved.GetFieldCount(); i++) {
			// A plain array has no column name of its own; the importer would call it v0.
			const auto raw = NameOf(schema, i);
			auto type = resolved.GetFieldType(i);
			types.push_back(type.GetTypeId());
			input.AddResultColumn(raw.empty() ? std::string("value") : std::string(resolved.GetFieldName(i)), type);
			names.push_back(raw);
		}
	} catch (...) {
		schema.release(&schema);
		throw;
	}
	schema.release(&schema);
	{
		nb::gil_scoped_acquire gil;
		try {
			nb::object rows = entry->object.attr("rows")();
			// A bool is an int to Python, and a truth value is never a row count.
			if (nb::isinstance<nb::bool_>(rows)) {
				throw nb::cast_error();
			}
			if (!rows.is_none()) {
				input.SetCardinality(nb::cast<cxx::idx_t>(rows), true);
			}
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("counting the rows of the object registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		} catch (const nb::cast_error &) {
			throw cxx::InvalidInputException("the source registered as '" + entry->name +
			                                 "' answered rows() with something other than a count or None");
		}
	}
	input.SetBindData<ScanBind>(std::move(entry), std::move(names), std::move(types));
}

/// Offers each predicate the source can be told about; one it accepts is kept for the scan and dropped from the
/// plan by the engine, which is why only a True answer accepts.
void PyScanFilterPushdown(cxx::TableFunction::FilterPushdownInput &input) {
	auto &bound = input.GetBindData<ScanBind>();
	auto &conversion = input.GetUserData<ScanUserData>().module->conversion;
	auto &entry = *bound.entry;
	nb::gil_scoped_acquire gil;
	try {
		const auto resolve = [&](cxx::idx_t reference) {
			const auto declared = input.GetColumnIndex(reference);
			return PredicateColumn {bound.names.at(declared), bound.types.at(declared)};
		};
		for (cxx::idx_t i = 0; i < input.GetFilterCount(); i++) {
			nb::object predicate;
			try {
				predicate = TranslatePredicate(input.GetFilter(i), resolve, conversion);
			} catch (const Refused &) {
				continue;
			}
			nb::object answer = entry.object.attr("accepts")(predicate);
			if (!nb::isinstance<nb::bool_>(answer)) {
				throw cxx::InvalidInputException("the source registered as '" + entry.name +
				                                 "' answered accepts() with something other than True or False");
			}
			if (nb::cast<bool>(answer)) {
				input.Accept(i);
				bound.filters.push_back(std::move(predicate));
			}
		}
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("offering a filter to the source registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
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
		exported = ExportStream(entry, identity ? nullptr : &requested, bound.filters);
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
		cxx::ArrowImporter importer(input.GetContext(), schema, input.GetUserData<ScanUserData>().batch_rows);
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

} // namespace

void RegisterObjectScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                        std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows) {
	auto function = cxx::TableFunction::Create(connection);
	function.SetName(kScanFunction);
	function.WithSignature(
	    [&](cxx::FunctionSignature &signature) { signature.AddParameter("name", connection.ParseType("VARCHAR")); });
	function.SetUserData<ScanUserData>(ScanUserData {std::move(registry), std::move(module), batch_rows});
	function.SetBindCallback(&PyScanBind);
	function.SetInitGlobalCallback(&PyScanInitGlobal);
	function.SetExecCallback(&PyScanExec);
	function.SetFilterPushdownCallback(&PyScanFilterPushdown);
	function.SetProjectionPushdown(true);
	function.Register();
}

} // namespace duckdb_python
