//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrow_scan.cpp
//
//
//===----------------------------------------------------------------------===//

#include "arrow_scan.hpp"

#include "arrowc.hpp"
#include "predicate.hpp"

#include <nanobind/stl/pair.h>

#include <utility>

// The table function that reads a registered object, and one run of it over a query is a scan. The registered
// object is always a source, the adapter in duckdb/_sources that wraps the caller's data and answers stream(),
// accepts() and rows(); "the source" below names that adapter. A predicate offered to it is a tree of
// duckdb._expressions nodes, built by predicate.cpp.

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
	auto [data, projected] =
	    nb::cast<std::pair<nb::object, bool>>(entry.object.attr("stream")(request, nb::tuple(promised)));
	Exported exported;
	exported.projected = projected;
	if (nb::isinstance<nb::tuple>(data) && nb::len(data) == 2) {
		auto both = nb::cast<nb::tuple>(data);
		exported.schema = nb::borrow(both[0]);
		exported.array = nb::borrow(both[1]);
	} else {
		exported.stream = std::move(data);
	}
	return exported;
}

std::string RegisteredObject(const Registered &entry) {
	return "the object registered as '" + entry.name + "'";
}

/// The source's schema, read without consuming data: from `__arrow_c_schema__` when the source has it, else from a
/// full export, whose stream is only borrowed, since a raw capsule is read once and must still be there to scan.
ArrowOwned<ArrowSchema> SchemaOf(Registered &entry) {
	if (nb::hasattr(entry.object, "__arrow_c_schema__")) {
		return TakeFromCapsule<ArrowSchema>(entry.object.attr("__arrow_c_schema__")(), kSchemaCapsule,
		                                    RegisteredObject(entry));
	}
	auto exported = ExportStream(entry, nullptr, {});
	if (exported.IsArray()) {
		// The array capsule is dropped unread, which releases it.
		return TakeFromCapsule<ArrowSchema>(exported.schema, kSchemaCapsule, RegisteredObject(entry));
	}
	auto &stream = InCapsule<ArrowArrayStream>(exported.stream, kStreamCapsule, RegisteredObject(entry));
	ArrowOwned<ArrowSchema> schema;
	if (stream.get_schema(&stream, &schema.value) != 0) {
		throw cxx::InvalidInputException("reading the schema of the stream registered as '" + entry.name +
		                                 "' failed: " + StreamError(stream));
	}
	return schema;
}

std::string ReadAlready(const Registered &entry) {
	return "the stream registered as '" + entry.name +
	       "' has been read already, by an earlier query or by another reference to it in this one; a stream can be "
	       "read once, so register it again, read it through one CTE, or load it into a table first";
}

struct ArrowScanUserData {
	std::shared_ptr<Registry> registry;
	std::shared_ptr<ModuleState> module;
	/// The engine's standard vector size, which is what the output chunk handed to the exec callback is allocated
	/// for, so the importer produces chunks of at most that many rows.
	cxx::idx_t batch_rows;
};

/// The entry a query bound over, the Arrow names and types of the columns it declared, in order, and the predicates
/// the source promised to apply, as `duckdb._expressions` nodes.
struct ArrowScanBindData {
	ArrowScanBindData(std::shared_ptr<Registered> entry, std::vector<std::string> names,
	                  std::vector<cxx::LogicalTypeId> types)
	    : entry(std::move(entry)), names(std::move(names)), types(std::move(types)) {
	}

	/// Freed from engine threads too, so the predicates are dropped under the GIL.
	~ArrowScanBindData() {
		FencedGil gil;
		filters.clear();
		entry.reset();
	}

	std::shared_ptr<Registered> entry;
	std::vector<std::string> names;
	std::vector<cxx::LogicalTypeId> types;
	std::vector<nb::object> filters;
};

/// One scan's source, shared by every thread pulling from it: either a stream or the single array a source handed
/// over through `__arrow_c_array__`, and the schema every thread's own importer is built from.
struct ArrowScanState {
	/// `stream_capsule` is the source's stream capsule, or empty when it handed over one array. The stream is taken
	/// out of it here, once the state exists, so nothing that can fail comes between emptying a capsule the source
	/// may keep and handing its stream to the scan.
	ArrowScanState(nb::handle stream_capsule, const std::string &source, ArrowOwned<ArrowArray> single,
	               ArrowOwned<ArrowSchema> schema, bool wrap, std::vector<cxx::idx_t> picks, bool pull_under_gil)
	    : single(std::move(single)), schema(std::move(schema)), wrap(wrap), picks(std::move(picks)),
	      pull_under_gil(pull_under_gil) {
		if (stream_capsule.is_valid()) {
			stream = TakeFromCapsule<ArrowArrayStream>(stream_capsule, kStreamCapsule, source);
		}
	}

	/// Torn down from an engine thread; a source's release may run Python, so everything is released here, under
	/// the GIL, rather than by the members' own destructors after it.
	~ArrowScanState() {
		FencedGil gil;
		stream.Release();
		single.Release();
		schema.Release();
	}

	/// The stream the arrays come from; empty when the source handed over one array.
	ArrowOwned<ArrowArrayStream> stream;
	/// The one array, until some thread claims it.
	ArrowOwned<ArrowArray> single;
	/// The schema every thread's own importer is resolved against; not consumed by building an importer from it.
	ArrowOwned<ArrowSchema> schema;
	/// Whether the source's arrays are plain values rather than batches, to be wrapped as a one-column batch.
	bool wrap;
	/// Which of the stream's columns each output vector takes; empty when the stream holds exactly the output.
	std::vector<cxx::idx_t> picks;
	/// Whether the stream needs the GIL held while its next array is pulled, because pulling may run Python.
	bool pull_under_gil;

	/// Guards the pull, the claim of a batch index and the claim of the single array; the fields above never change.
	std::mutex pull_lock;
	cxx::idx_t next_batch = 0;
	bool exhausted = false;
};

/// One scanning thread's own importer, which the Arrow C data contract requires to be single threaded, and the
/// chunk it is currently emitting.
struct ArrowScanLocalState {
	explicit ArrowScanLocalState(cxx::ArrowImporter importer) : importer(std::move(importer)) {
	}

	cxx::ArrowImporter importer;
	/// The chunk the output vectors reference, kept until the next batch replaces it.
	std::optional<cxx::DataChunk> current;
	/// The ordering position of the array this thread is currently emitting rows from.
	cxx::idx_t batch_index = 0;
};

/// Resolved by name at bind time, so a query sees the object registered under the name when it binds, as it would
/// see a table.
void ArrowScanBind(cxx::TableFunction::BindInput &input) {
	auto &registry = *input.GetUserData<ArrowScanUserData>().registry;
	const auto name = std::string(input.GetConstantArgument(0).Get<cxx::varchar_t>());
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
	ArrowOwned<ArrowSchema> schema;
	{
		FencedGil gil;
		try {
			schema = SchemaOf(*entry);
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("reading the schema of the object registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		}
	}
	if (!IsBatch(schema.value)) {
		WrapAsBatch(schema.value);
	}
	std::vector<std::string> names;
	std::vector<cxx::LogicalTypeId> types;
	cxx::ArrowImporter importer(input.GetContext(), schema.value, input.GetUserData<ArrowScanUserData>().batch_rows);
	auto resolved = importer.GetSchema();
	for (cxx::idx_t i = 0; i < resolved.GetFieldCount(); i++) {
		// A plain array has no column name of its own; the importer would call it v0.
		const auto raw = NameOf(schema.value, i);
		auto type = resolved.GetFieldType(i);
		types.push_back(type.GetTypeId());
		input.AddResultColumn(raw.empty() ? std::string("value") : std::string(resolved.GetFieldName(i)), type);
		names.push_back(raw);
	}
	{
		FencedGil gil;
		try {
			nb::object rows = entry->object.attr("rows")();
			if (!rows.is_none()) {
				input.SetCardinality(nb::cast<cxx::idx_t>(rows), true);
			}
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("counting the rows of the object registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		}
	}
	input.SetBindData<ArrowScanBindData>(std::move(entry), std::move(names), std::move(types));
}

/// Offers each predicate the source can be told about; one it accepts is kept for the scan and dropped from the
/// plan by the engine, which is why only a True answer accepts.
void ArrowScanFilterPushdown(cxx::TableFunction::FilterPushdownInput &input) {
	auto &bound = input.GetBindData<ArrowScanBindData>();
	auto &conversion = input.GetUserData<ArrowScanUserData>().module->conversion;
	auto &entry = *bound.entry;
	FencedGil gil;
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
			if (nb::cast<bool>(entry.object.attr("accepts")(predicate))) {
				input.Accept(i);
				bound.filters.push_back(std::move(predicate));
			}
		}
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("offering a filter to the source registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
}

/// Exports the stream, narrowed to the columns the query uses when the source can, and validates it against the
/// declared columns. The scan takes columns by position, so a source that answers with other columns than it
/// said, in width or in order as far as the names tell, is refused rather than read wrongly.
void OpenStream(const ArrowScanBindData &bound, cxx::TableFunction::InitGlobalInput &input) {
	auto &entry = *bound.entry;
	const auto declared = static_cast<cxx::idx_t>(bound.names.size());
	std::vector<cxx::idx_t> requested;
	bool identity = input.GetColumnCount() == declared;
	for (cxx::idx_t i = 0; i < input.GetColumnCount(); i++) {
		requested.push_back(input.GetColumnIndex(i));
		identity = identity && requested.back() == i;
	}
	FencedGil gil;
	Exported exported;
	try {
		exported = ExportStream(entry, identity ? nullptr : &requested, bound.filters);
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("exporting a stream from the object registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
	const bool pull_under_gil = nb::cast<bool>(entry.object.attr("pull_under_gil"));
	ArrowOwned<ArrowArray> single;
	ArrowOwned<ArrowSchema> schema;
	if (exported.IsArray()) {
		schema = TakeFromCapsule<ArrowSchema>(exported.schema, kSchemaCapsule, RegisteredObject(entry));
		single = TakeFromCapsule<ArrowArray>(exported.array, kArrayCapsule, RegisteredObject(entry));
	} else {
		// Only borrowed here; the scan state takes it. A source may hand over a stream it keeps, and a scan that fails
		// before reading a row must leave that stream in place for the next query to try again.
		auto &borrowed = InCapsule<ArrowArrayStream>(exported.stream, kStreamCapsule, RegisteredObject(entry));
		if (borrowed.get_schema(&borrowed, &schema.value) != 0) {
			throw cxx::InvalidInputException("reading the schema of the stream registered as '" + entry.name +
			                                 "' failed: " + StreamError(borrowed));
		}
	}
	const bool wrap = !IsBatch(schema.value);
	if (wrap) {
		WrapAsBatch(schema.value);
		if (single) {
			WrapAsBatch(single.value);
		}
	}
	const auto fields = static_cast<cxx::idx_t>(schema.value.n_children);
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
		if (NameOf(schema.value, i) != bound.names[expected]) {
			throw cxx::InvalidInputException("the source registered as '" + entry.name + "' answered with column '" +
			                                 NameOf(schema.value, i) + "' at position " + std::to_string(i) +
			                                 " where '" + bound.names[expected] + "' was expected");
		}
	}
	// The engine clamps the cap to its scheduler's thread count, so a large one asks for every thread it has;
	// a single array is one batch, which only one thread can ever work on.
	input.SetMaxThreads(exported.IsArray() ? 1 : 1024);
	input.SetGlobalState<ArrowScanState>(exported.stream, RegisteredObject(entry), std::move(single), std::move(schema),
	                                     wrap, std::move(picks), pull_under_gil);
}

void ArrowScanInitGlobal(cxx::TableFunction::InitGlobalInput &input) {
	const auto &bound = input.GetBindData<ArrowScanBindData>();
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

/// Builds this thread's own importer over the global schema; the Arrow importer is single threaded by contract,
/// which is why every scanning thread gets one of its own. Runs on an engine thread: no Python here.
void ArrowScanInitLocal(cxx::TableFunction::InitLocalInput &input) {
	auto &global = input.GetGlobalState<ArrowScanState>();
	const auto batch_rows = input.GetUserData<ArrowScanUserData>().batch_rows;
	input.SetLocalState<ArrowScanLocalState>(cxx::ArrowImporter(input.GetContext(), global.schema.value, batch_rows));
}

/// Hands the next chunk to the engine: the importer's chunk is referenced, not copied, and stays alive here.
void HandOver(ArrowScanState &global, ArrowScanLocalState &local, cxx::DataChunk chunk,
              cxx::TableFunction::ExecInput &input) {
	auto output = input.GetOutputChunk();
	const auto columns = output.GetVectorCount();
	// The engine sizes a batch from its first output vector, so it never asks for zero columns; a count over the
	// scan still asks for one. Should that change, an empty output would need another way to carry the row count.
	if (columns == 0) {
		throw cxx::InvalidInputException("the scan of '" + input.GetBindData<ArrowScanBindData>().entry->name +
		                                 "' was asked for no columns, which it cannot report rows through");
	}
	const auto needed = global.picks.empty() ? columns : global.picks.size();
	if (chunk.GetVectorCount() < needed) {
		throw cxx::InvalidInputException("an Arrow batch carried " + std::to_string(chunk.GetVectorCount()) +
		                                 " columns where " + std::to_string(needed) + " were expected");
	}
	for (cxx::idx_t i = 0; i < columns; i++) {
		output.GetVector(i).Reference(chunk.GetVector(global.picks.empty() ? i : global.picks[i]));
	}
	output.GetVector(0).SetSize(chunk.GetRowCount());
	local.current = std::move(chunk);
}

/// Pulls the stream's next array, under the GIL when the source said pulling may run Python.
void PullNext(ArrowArrayStream &stream, bool under_gil, const std::string &name, ArrowArray &out) {
	std::optional<FencedGil> gil;
	if (under_gil) {
		gil.emplace();
	}
	if (stream.get_next(&stream, &out) != 0) {
		throw cxx::InvalidInputException("reading the stream registered as '" + name +
		                                 "' failed: " + StreamError(stream));
	}
}

void ArrowScanExec(cxx::TableFunction::ExecInput &input) {
	auto &global = input.GetGlobalState<ArrowScanState>();
	auto &local = input.GetLocalState<ArrowScanLocalState>();
	const auto &entry = *input.GetBindData<ArrowScanBindData>().entry;
	for (;;) {
		auto chunk = local.importer.NextChunk();
		if (chunk && chunk.GetRowCount() > 0) {
			HandOver(global, local, std::move(chunk), input);
			return;
		}
		ArrowOwned<ArrowArray> array;
		{
			std::lock_guard<std::mutex> guard(global.pull_lock);
			if (global.exhausted) {
				return;
			}
			if (!global.stream) {
				array = std::move(global.single);
				global.exhausted = true;
			} else {
				try {
					PullNext(global.stream.value, global.pull_under_gil, entry.name, array.value);
				} catch (...) {
					// A stream that failed is not pulled again by another thread while the query is being cancelled.
					global.exhausted = true;
					throw;
				}
				if (array && global.wrap) {
					WrapAsBatch(array.value);
				}
				if (!array) {
					global.exhausted = true;
				}
			}
			if (array) {
				local.batch_index = global.next_batch++;
			}
		}
		if (!array) {
			continue;
		}
		// Flushed on every array, not only at the end: otherwise the importer would carry a held-back tail of rows
		// into the next array it is given, and with two threads pulling, that tail would emit rows of one array
		// under a later array's batch index once both arrays had come out. Taking the array clears its release.
		local.importer.Append(array.value, true, true);
	}
}

/// Reports the ordering position of the batch the exec callback just produced. The engine requires this on every
/// call and never sees it decrease within one thread, which holds because a thread only ever claims increasing
/// indices from the global counter.
void ArrowScanPartitionData(cxx::TableFunction::PartitionDataInput &input) {
	if (input.GetPartitionColumnCount() != 0) {
		// Without a partitioning callback the engine never asks for partition values; a silent answer would be wrong.
		throw cxx::InvalidInputException("the scan of a registered object was asked for partition values, which it "
		                                 "does not report");
	}
	auto &local = input.GetLocalState<ArrowScanLocalState>();
	input.SetBatchIndex(local.batch_index);
}

} // namespace

void RegisterArrowScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                       std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows) {
	auto function = cxx::TableFunction::Create(connection);
	function.SetName(kArrowScanFunction);
	function.WithSignature(
	    [&](cxx::FunctionSignature &signature) { signature.AddParameter("name", connection.ParseType("VARCHAR")); });
	function.SetUserData<ArrowScanUserData>(ArrowScanUserData {std::move(registry), std::move(module), batch_rows});
	function.SetBindCallback(&ArrowScanBind);
	function.SetInitGlobalCallback(&ArrowScanInitGlobal);
	function.SetInitLocalCallback(&ArrowScanInitLocal);
	function.SetExecCallback(&ArrowScanExec);
	function.SetFilterPushdownCallback(&ArrowScanFilterPushdown);
	function.SetPartitionDataCallback(&ArrowScanPartitionData);
	function.SetProjectionPushdown(true);
	function.Register();
}

} // namespace duckdb_python
