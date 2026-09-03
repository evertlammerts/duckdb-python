//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/module.cpp
//
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "duckdb_cpp.hpp"
#include "pyconv.hpp"

namespace nb = nanobind;
namespace cxx = duckdb::cxx;

namespace duckdb_python {
namespace {

// Step participates in executing the query: WAITING means "call again", never
// "wait for someone else". Sleeping between calls only serializes the work,
// so the step loops below run hot, releasing the GIL for one quantum of
// stepping at a time and retaking it for the Ctrl-C check in between.
constexpr std::chrono::milliseconds kStepQuantum {20};

// An engine error carries a numeric code; duckdb.exceptions owns the mapping
// from code to class. One catch clause covers every engine exception, including
// the two the C++ API gives a dedicated type, since all of them carry the code.
void TranslateException(const std::exception_ptr &captured, void * /*payload*/) {
	try {
		std::rethrow_exception(captured);
	} catch (const cxx::Exception &e) {
		try {
			nb::object module = nb::module_::import_("duckdb.exceptions");
			nb::object cls = module.attr("class_for_code")(e.GetCode());
			PyErr_SetString(cls.ptr(), e.what());
		} catch (...) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
}

/// Everything this module owns for the lifetime of its interpreter.
///
/// There is exactly one Environment, and every database is opened through it.
/// That is the whole point of the Environment: it is what notices a second
/// attempt to open a database already open in this process. Giving each
/// Database its own Environment would silently remove that guard.
///
/// This is per module instance, not a C++ static. nanobind initialises modules
/// in two phases and the exec function runs once per interpreter that imports
/// us, so a state created there and captured by the bindings registered
/// alongside it belongs to that interpreter alone.
struct ModuleState {
	cxx::Environment environment;
	ConversionContext conversion;
};

/// One open database, and the module state that must outlive it.
struct DatabaseState {
	DatabaseState(std::shared_ptr<ModuleState> module, const std::string &path,
	              const std::vector<std::pair<std::string, std::string>> &options)
	    : module(std::move(module)) {
		if (options.empty()) {
			database = std::make_unique<cxx::Database>(this->module->environment.Open(path));
			return;
		}
		std::vector<cxx::DatabaseOption> opts;
		opts.reserve(options.size());
		for (const auto &[name, value] : options) {
			opts.emplace_back(name, value);
		}
		database = std::make_unique<cxx::Database>(this->module->environment.Open(path, opts));
	}

	std::shared_ptr<ModuleState> module;
	std::unique_ptr<cxx::Database> database;
};

/// The engine result and the lock guarding its lifetime.
///
/// Held behind a shared_ptr rather than inline, because a std::mutex member
/// would make Result non-movable, and nanobind constructs Result by value from
/// Connection::execute.
struct ResultState {
	std::mutex lifetime;
	std::shared_ptr<cxx::QueryResult> result;
};

/// One fetched chunk, column-wise, for the numpy converter in duckdb/_numpy.py.
///
/// Fixed-width columns hand out zero-copy memoryviews over the flattened
/// vector data; everything else falls back to per-cell objects. The views
/// borrow the chunk's memory, so they are valid only while this object
/// lives: the converter copies out of them within one loop iteration and
/// never keeps one.
class ChunkView {
public:
	// The types ride along because this facade keeps them on the result's
	// schema, not on the vectors. `row_offset` is how many leading rows a
	// prior row fetch already consumed; the buffers still cover the whole
	// chunk, so the converter slices them by it.
	ChunkView(cxx::DataChunk chunk, std::vector<cxx::LogicalType> types, cxx::idx_t row_offset = 0)
	    : chunk(std::move(chunk)), types(std::move(types)), row_offset(row_offset) {
		const auto count = this->chunk.GetVectorCount();
		vectors.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			cxx::Vector vector = this->chunk.GetVector(i);
			// Dictionary, constant and other encodings become flat data
			// plus validity, the one layout the buffer views can serve.
			vector.Flatten();
			vectors.push_back(std::move(vector));
		}
	}

	cxx::idx_t RowCount() const {
		return chunk.GetRowCount();
	}

	cxx::idx_t RowOffset() const {
		return row_offset;
	}

	cxx::idx_t ColumnCount() const {
		return static_cast<cxx::idx_t>(vectors.size());
	}

	int TypeId(cxx::idx_t column) const {
		return static_cast<int>(Type(column).GetTypeId());
	}

	std::string TypeText(cxx::idx_t column) const {
		return Type(column).ToText();
	}

	/// Zero-copy view over the flattened data, or None without a fixed-width layout.
	nb::object Data(cxx::idx_t column) {
		const size_t element = ElementSize(column);
		if (element == 0) {
			return nb::none();
		}
		const auto view = vectors.at(column).GetView();
		return Memoryview(view.data, element * chunk.GetRowCount());
	}

	/// Validity bitmask as 64-bit words, LSB first, or None when all rows are valid.
	nb::object Validity(cxx::idx_t column) {
		const auto view = vectors.at(column).GetView();
		if (!view.validity) {
			return nb::none();
		}
		const size_t words = (chunk.GetRowCount() + 63) / 64;
		return Memoryview(view.validity, words * sizeof(uint64_t));
	}

	/// A DECIMAL column's scale, so the converter never parses type text.
	int DecimalScale(cxx::idx_t column) const {
		return static_cast<int>(Type(column).GetDecimalScale());
	}

	/// The ENUM dictionary, index to string, for categorical assembly.
	std::vector<std::string> EnumValues(cxx::idx_t column) const {
		const auto &type = Type(column);
		std::vector<std::string> out;
		const auto count = type.GetEnumSize();
		out.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			out.push_back(type.GetEnumValue(i));
		}
		return out;
	}

	/// Per-cell object fallback for the columns Data() cannot serve.
	nb::list Values(cxx::idx_t column, ConversionContext &ctx) {
		auto &vector = vectors.at(column);
		nb::list out;
		const auto count = chunk.GetRowCount();
		for (cxx::idx_t row = 0; row < count; row++) {
			out.append(ValueToPython(vector.GetValue(row), ctx));
		}
		return out;
	}

private:
	const cxx::LogicalType &Type(cxx::idx_t column) const {
		return types.at(column);
	}

	/// Bytes per element for the fixed-width layouts, 0 for everything else.
	size_t ElementSize(cxx::idx_t column) const {
		using Id = cxx::LogicalTypeId;
		const auto &type = Type(column);
		switch (type.GetTypeId()) {
		case Id::BOOLEAN:
		case Id::TINYINT:
		case Id::UTINYINT:
			return 1;
		case Id::SMALLINT:
		case Id::USMALLINT:
			return 2;
		case Id::INTEGER:
		case Id::UINTEGER:
		case Id::DATE:
		case Id::FLOAT:
			return 4;
		case Id::BIGINT:
		case Id::UBIGINT:
		case Id::DOUBLE:
		case Id::TIMESTAMP:
		case Id::TIMESTAMP_SEC:
		case Id::TIMESTAMP_MS:
		case Id::TIMESTAMP_NS:
		case Id::TIMESTAMP_TZ:
			return 8;
		case Id::INTERVAL:
		case Id::HUGEINT:
		case Id::UHUGEINT:
			return 16;
		case Id::DECIMAL: {
			// The storage tier follows the width; the widest tier is int128,
			// which the converter reads as two 64-bit limbs.
			const auto width = type.GetDecimalWidth();
			return width <= 4 ? 2 : width <= 9 ? 4 : width <= 18 ? 8 : 16;
		}
		case Id::ENUM: {
			const auto size = type.GetEnumSize();
			return size < 256 ? 1 : size < 65536 ? 2 : 4;
		}
		default:
			return 0;
		}
	}

	static nb::object Memoryview(const void *data, size_t bytes) {
		PyObject *view = PyMemoryView_FromMemory(const_cast<char *>(static_cast<const char *>(data)),
		                                         static_cast<Py_ssize_t>(bytes), PyBUF_READ);
		if (!view) {
			throw nb::python_error();
		}
		return nb::steal(view);
	}

	cxx::DataChunk chunk;
	std::vector<cxx::LogicalType> types;
	std::vector<cxx::Vector> vectors;
	cxx::idx_t row_offset;
};

class Result {
public:
	Result(std::shared_ptr<DatabaseState> owner, cxx::QueryResult result)
	    : owner(std::move(owner)), state(std::make_shared<ResultState>()) {
		state->result = std::make_shared<cxx::QueryResult>(std::move(result));
	}

	/// Column names paired with the text form of their type.
	std::vector<std::pair<std::string, std::string>> Schema() {
		const auto schema = Live()->GetSchema();
		std::vector<std::pair<std::string, std::string>> out;
		const auto count = schema.GetFieldCount();
		out.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			out.emplace_back(std::string(schema.GetFieldName(i)), schema.GetFieldType(i).ToText());
		}
		return out;
	}

	/// A reference to the live result, or a clear error once it is closed.
	///
	/// Returns a shared_ptr rather than a raw reference on purpose: the caller
	/// keeps the engine result alive for as long as it is using it, even if
	/// another thread closes this Result meanwhile.
	std::shared_ptr<cxx::QueryResult> Live() {
		std::lock_guard<std::mutex> guard(state->lifetime);
		if (!state->result) {
			nb::object module = nb::module_::import_("duckdb.exceptions");
			PyErr_SetString(module.attr("InterfaceError").ptr(), "result is closed");
			throw nb::python_error();
		}
		return state->result;
	}

	/// Whether this result carries rows, a changed-row count, or nothing.
	std::string ResultType() {
		switch (Live()->GetResultType()) {
		case cxx::QueryResult::ResultType::QUERY_RESULT:
			return "rows";
		case cxx::QueryResult::ResultType::CHANGED_ROWS:
			return "changed_rows";
		default:
			return "nothing";
		}
	}

	/// The kind of SQL statement this result came from, as lowercase text.
	std::string StatementTypeName() {
		using Kind = cxx::QueryResult::StatementType;
		switch (Live()->GetStatementType()) {
		case Kind::SELECT:
			return "select";
		case Kind::INSERT:
			return "insert";
		case Kind::UPDATE:
			return "update";
		case Kind::CREATE:
			return "create";
		case Kind::DELETE:
			return "delete";
		case Kind::PREPARE:
			return "prepare";
		case Kind::EXECUTE:
			return "execute";
		case Kind::ALTER:
			return "alter";
		case Kind::TRANSACTION:
			return "transaction";
		case Kind::COPY:
			return "copy";
		case Kind::ANALYZE:
			return "analyze";
		case Kind::VARIABLE_SET:
			return "variable_set";
		case Kind::CREATE_FUNC:
			return "create_func";
		case Kind::EXPLAIN:
			return "explain";
		case Kind::DROP:
			return "drop";
		case Kind::EXPORT:
			return "export";
		case Kind::PRAGMA:
			return "pragma";
		case Kind::VACUUM:
			return "vacuum";
		case Kind::CALL:
			return "call";
		case Kind::SET:
			return "set";
		case Kind::LOAD:
			return "load";
		case Kind::RELATION:
			return "relation";
		case Kind::EXTENSION:
			return "extension";
		case Kind::LOGICAL_PLAN:
			return "logical_plan";
		case Kind::ATTACH:
			return "attach";
		case Kind::DETACH:
			return "detach";
		case Kind::MULTI:
			return "multi";
		case Kind::COPY_DATABASE:
			return "copy_database";
		case Kind::UPDATE_EXTENSIONS:
			return "update_extensions";
		case Kind::MERGE_INTO:
			return "merge_into";
		default:
			return "unknown";
		}
	}

	/// Run the statement to completion and report how many rows it changed.
	///
	/// Side effects land when a result is drained, so a statement whose result
	/// is dropped without draining never takes effect at all -- except one
	/// carrying RETURNING, which the engine applies at execute. Stepped here
	/// rather than by the engine's blocking drain, so a Ctrl-C can land midway.
	/// The GIL stays released for whole batches of stepping and is retaken a
	/// few times a second for the signal check: this loop does no Python work,
	/// and per-chunk GIL churn starves every other Python thread, the one
	/// delivering the Ctrl-C included. The count of a changed-rows result
	/// travels as a chunk the stepping consumes, so it is harvested here.
	cxx::idx_t Drain() {
		pending.reset();
		cxx::idx_t changed = 0;
		while (true) {
			auto live = Live();
			bool done = false;
			bool cancelled = false;
			{
				nb::gil_scoped_release release;
				const auto deadline = std::chrono::steady_clock::now() + kStepQuantum;
				while (std::chrono::steady_clock::now() < deadline) {
					auto step = live->Step();
					if (step.status == cxx::QueryResult::StepStatus::CHUNK) {
						if (step.chunk.GetRowCount() > 0 &&
						    live->GetResultType() == cxx::QueryResult::ResultType::CHANGED_ROWS) {
							changed += static_cast<cxx::idx_t>(
							    step.chunk.GetVector(0).GetValue(0).Get<int64_t>());
						}
						continue;
					}
					if (step.status == cxx::QueryResult::StepStatus::FINISHED) {
						done = true;
					} else if (step.status == cxx::QueryResult::StepStatus::CANCELLED) {
						cancelled = true;
					} else {
						// WAITING: stepping is the progress; go again.
						continue;
					}
					break;
				}
			}
			if (cancelled) {
				nb::object module = nb::module_::import_("duckdb.exceptions");
				PyErr_SetString(module.attr("InterruptError").ptr(), "query was cancelled");
				throw nb::python_error();
			}
			if (done) {
				// The statement completed and its side effects landed; a
				// signal pending this very batch must not turn that into an
				// exception a caller would read as "did not happen". It is
				// delivered at the interpreter's own next check instead.
				finished = true;
				return changed;
			}
			if (PyErr_CheckSignals() != 0) {
				throw nb::python_error();
			}
		}
	}

	/// Release the result so the connection can run another query.
	///
	/// The engine allows one live result per connection, so a caller that
	/// stops reading early needs a way to say so. Relying on the Python object
	/// being collected would make the moment of release depend on refcount
	/// timing, which is exactly the wrong thing for an exclusive resource.
	///
	/// Only the owning pointer is cleared here. A thread already inside
	/// Step/Wait/Drain holds its own reference and keeps the engine result
	/// alive until it returns, so closing never frees it underneath that
	/// thread. Both sides drop the GIL, so the GIL cannot serialise this.
	void Close() {
		std::shared_ptr<cxx::QueryResult> released;
		std::shared_ptr<DatabaseState> dropped;
		{
			std::lock_guard<std::mutex> guard(state->lifetime);
			pending.reset();
			finished = true;
			released = std::move(state->result);
			state->result.reset();
			// A closed result must pin nothing: holding the database here
			// kept the file "in use" past a close that had returned, until
			// garbage collection happened to run.
			dropped = std::move(owner);
		}
		// Drop the last references, if they are the last, outside the lock
		// and without the GIL: destruction talks to the engine.
		nb::gil_scoped_release unlock;
		released.reset();
		dropped.reset();
	}

	/// Up to `count` more rows, or every remaining row when `count` is zero.
	///
	/// Streaming is the only execution model here, so rows are taken a chunk
	/// at a time and a partly-read chunk is carried across calls. Buffering
	/// the whole result to serve fetchone() would defeat the point.
	nb::list FetchRows(ConversionContext &ctx, size_t count) {
		nb::list rows;
		Live(); // raises if the result was closed
		while (count == 0 || nb::len(rows) < count) {
			// Anything left over from the previous call comes first.
			if (pending) {
				const auto available = pending->GetRowCount();
				if (offset < available) {
					auto want = available - offset;
					if (count != 0) {
						const auto remaining = count - nb::len(rows);
						if (remaining < want) {
							want = remaining;
						}
					}
					AppendChunkRows(*pending, RowTypes(), offset, offset + want, ctx, rows);
					// Advanced only on success, so a failed conversion can
					// be retried from the same row.
					offset += want;
				}
				if (offset < available) {
					return rows; // count reached mid-chunk
				}
				pending.reset();
				offset = 0;
			}
			if (finished) {
				return rows;
			}
			if (!Advance()) {
				return rows;
			}
		}
		return rows;
	}

	/// Drain the result into a list of row tuples.
	nb::list FetchAll(ConversionContext &ctx) {
		return FetchRows(ctx, 0);
	}

	/// The next chunk column-wise for the numpy converter, or None at the end.
	///
	/// A chunk partly consumed by a prior row fetch is handed out whole with
	/// its row offset, so the converter delivers only the remaining rows.
	nb::object FetchChunkView() {
		auto live = Live();
		if (!pending && (finished || !Advance())) {
			return nb::none();
		}
		auto chunk = std::move(*pending);
		const auto consumed = offset;
		pending.reset();
		offset = 0;
		const auto schema = live->GetSchema();
		std::vector<cxx::LogicalType> types;
		const auto count = schema.GetFieldCount();
		types.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			types.push_back(schema.GetFieldType(i));
		}
		return nb::cast(ChunkView(std::move(chunk), std::move(types), consumed));
	}

	/// Per-column (type id, decimal scale, enum dictionary or None), from the
	/// schema alone, so an empty result still assembles with the dtypes its
	/// rows would have had.
	std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> SchemaTypes() {
		const auto schema = Live()->GetSchema();
		std::vector<std::tuple<int, int, std::optional<std::vector<std::string>>>> out;
		const auto count = schema.GetFieldCount();
		out.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			const auto type = schema.GetFieldType(i);
			const auto id = static_cast<int>(type.GetTypeId());
			int scale = 0;
			std::optional<std::vector<std::string>> dictionary;
			if (type.GetTypeId() == cxx::LogicalTypeId::DECIMAL) {
				scale = static_cast<int>(type.GetDecimalScale());
			} else if (type.GetTypeId() == cxx::LogicalTypeId::ENUM) {
				std::vector<std::string> values;
				const auto size = type.GetEnumSize();
				values.reserve(size);
				for (cxx::idx_t v = 0; v < size; v++) {
					values.push_back(type.GetEnumValue(v));
				}
				dictionary = std::move(values);
			}
			out.emplace_back(id, scale, std::move(dictionary));
		}
		return out;
	}

private:
	/// Step until a chunk arrives or the result ends. False once it has ended.
	bool Advance() {
		while (true) {
			// The reference is taken before the GIL is dropped and held for
			// the whole quantum, so a concurrent Close cannot free it here.
			auto live = Live();
			bool got_chunk = false;
			bool cancelled = false;
			{
				// The engine may run this thread's share of the query inside
				// Step; never hold the GIL across it, or a callback into
				// Python deadlocks.
				nb::gil_scoped_release release;
				const auto deadline = std::chrono::steady_clock::now() + kStepQuantum;
				while (std::chrono::steady_clock::now() < deadline) {
					auto step = live->Step();
					if (step.status == cxx::QueryResult::StepStatus::CHUNK) {
						pending = std::move(step.chunk);
						offset = 0;
						got_chunk = true;
					} else if (step.status == cxx::QueryResult::StepStatus::FINISHED) {
						finished = true;
					} else if (step.status == cxx::QueryResult::StepStatus::CANCELLED) {
						cancelled = true;
					} else {
						// WAITING: stepping is the progress; go again.
						continue;
					}
					break;
				}
			}
			if (cancelled) {
				nb::object module = nb::module_::import_("duckdb.exceptions");
				PyErr_SetString(module.attr("InterruptError").ptr(), "query was cancelled");
				throw nb::python_error();
			}
			if (got_chunk) {
				// The chunk was parked before this check, so a Ctrl-C on a
				// chunk boundary loses no rows: a caller that catches it
				// finds the chunk still pending and can keep fetching. A
				// no-op off the main thread, where Python runs no signal
				// handlers.
				if (PyErr_CheckSignals() != 0) {
					throw nb::python_error();
				}
				return true;
			}
			if (finished) {
				// End of stream: a pending signal is delivered at the
				// interpreter's own next check, never as a raise that hides
				// the completion.
				return false;
			}
			if (PyErr_CheckSignals() != 0) {
				throw nb::python_error();
			}
		}
	}

	/// The columns' logical types, cached: the row path dispatches on them
	/// per chunk, and this facade keeps them on the schema, not the vectors.
	const std::vector<cxx::LogicalType> &RowTypes() {
		if (!row_types) {
			const auto schema = Live()->GetSchema();
			std::vector<cxx::LogicalType> types;
			const auto count = schema.GetFieldCount();
			types.reserve(count);
			for (cxx::idx_t i = 0; i < count; i++) {
				types.push_back(schema.GetFieldType(i));
			}
			row_types = std::move(types);
		}
		return *row_types;
	}

	std::shared_ptr<DatabaseState> owner;
	std::shared_ptr<ResultState> state;
	std::optional<cxx::DataChunk> pending;
	std::optional<std::vector<cxx::LogicalType>> row_types;
	cxx::idx_t offset = 0;
	bool finished = false;
};

/// Carried by a registered Python scalar function and read by every exec
/// call. The engine frees it at teardown, which can run without the GIL, so
/// dropping the callable takes it first; with the interpreter already gone
/// the reference is deliberately leaked rather than decref'd into a corpse.
struct PyFunctionData {
	PyFunctionData(nb::object callable, std::string name, std::vector<cxx::LogicalType> parameter_types,
	               bool skip_nulls, std::shared_ptr<ModuleState> module)
	    : callable(std::move(callable)), name(std::move(name)), parameter_types(std::move(parameter_types)),
	      skip_nulls(skip_nulls), module(std::move(module)) {
	}

	~PyFunctionData() {
		if (!Py_IsInitialized()) {
			static_cast<void>(callable.release());
			return;
		}
		nb::gil_scoped_acquire gil;
		callable.reset();
	}

	nb::object callable;
	std::string name;
	std::vector<cxx::LogicalType> parameter_types;
	bool skip_nulls;
	std::shared_ptr<ModuleState> module;
};

bool IsPandasNA(nb::handle object) {
	return nb::cast<std::string>(nb::handle(Py_TYPE(object.ptr())).attr("__name__")) == "NAType";
}

/// The engine runs this on its own threads, so the GIL is taken here, once
/// per batch. The arguments are trusted to match the declared parameter
/// types: the signature refuses ANY, so the binder has cast them.
void PyScalarExec(cxx::ScalarFunction::ExecInput &input) {
	auto &data = input.GetUserData<PyFunctionData>();
	const auto rows = input.GetRowCount();
	const auto count = input.GetArgCount();
	auto context = input.GetContext();
	auto result = input.GetResult();

	nb::gil_scoped_acquire gil;
	try {
		std::vector<nb::list> columns;
		columns.reserve(count);
		for (cxx::idx_t a = 0; a < count; a++) {
			auto argument = input.GetArg(a);
			columns.push_back(
			    VectorElements(argument, data.parameter_types.at(a), 0, rows, data.module->conversion));
		}
		result.SetSize(rows);
		for (cxx::idx_t r = 0; r < rows; r++) {
			PyObject *raw = PyTuple_New(static_cast<Py_ssize_t>(count));
			if (raw == nullptr) {
				throw nb::python_error();
			}
			auto arguments = nb::steal<nb::tuple>(raw);
			bool any_null = false;
			for (cxx::idx_t a = 0; a < count; a++) {
				PyObject *item = PyList_GetItem(columns[a].ptr(), static_cast<Py_ssize_t>(r));
				any_null = any_null || item == Py_None;
				// SetItem steals the new reference whatever it returns.
				if (PyTuple_SetItem(raw, static_cast<Py_ssize_t>(a), Py_NewRef(item)) != 0) {
					throw nb::python_error();
				}
			}
			// DEFAULT null handling promises NULL in, NULL out without a
			// call, but the engine still runs the batch over such rows, so
			// the row skip lives here.
			if (any_null && data.skip_nulls) {
				result.SetNull(r);
				continue;
			}
			PyObject *returned = PyObject_CallObject(data.callable.ptr(), raw);
			if (returned == nullptr) {
				throw nb::python_error();
			}
			const auto object = nb::steal(returned);
			if (object.is_none()) {
				result.SetNull(r);
				continue;
			}
			try {
				// SetValue casts to the vector's type, so the declared
				// return type is enforced right here.
				result.SetValue(r, PythonToValue(context, object, data.module->conversion));
			} catch (const cxx::Exception &) {
				// pandas' NA is unbindable but means NULL, as the old
				// client treated it.
				if (IsPandasNA(object)) {
					result.SetNull(r);
					continue;
				}
				throw;
			}
		}
	} catch (nb::python_error &error) {
		// Rendered while the GIL is still held; what() needs it. Summary
		// before traceback, in the old client's words, which callers and
		// adopted tests match on. The engine prefixes callback errors
		// itself, so no "Invalid Input Error:" here.
		std::string summary;
		try {
			summary = nb::cast<std::string>(nb::handle(error.type()).attr("__name__"));
			const auto text = nb::cast<std::string>(nb::str(nb::handle(error.value())));
			if (!text.empty()) {
				summary += ": " + text;
			}
		} catch (...) {
			summary.clear();
		}
		throw cxx::InvalidInputException("Python exception occurred while executing the UDF '" + data.name +
		                                 "': " + summary + "\n" + error.what());
	}
}

class Connection {
public:
	Connection(std::shared_ptr<DatabaseState> owner, cxx::Connection connection)
	    : owner(std::move(owner)), connection(std::move(connection)) {
	}

	/// Run one statement, optionally binding parameters.
	///
	/// `parameters` is either a sequence, binding $1, $2, ... in order, or a
	/// mapping, binding $name. NamedParam carries both shapes: an empty name
	/// means positional. A statement cannot mix the two, which the engine
	/// enforces at bind time.
	Result Execute(const std::string &sql, nb::handle parameters, ConversionContext &ctx) {
		if (parameters.is_none()) {
			nb::gil_scoped_release release;
			return Result(owner, connection.Execute(sql));
		}

		std::vector<cxx::NamedParam> bound;
		if (nb::isinstance<nb::dict>(parameters)) {
			for (auto entry : nb::cast<nb::dict>(parameters)) {
				// Checked, because nanobind's failed cast would surface as a
				// raw std::bad_cast instead of an error naming the mistake.
				if (!nb::isinstance<nb::str>(entry.first)) {
					throw cxx::InvalidInputException(
					    "Invalid Input Error: parameter names must be strings");
				}
				bound.push_back({nb::cast<std::string>(entry.first),
				                 PythonToValue(connection, entry.second, ctx)});
			}
		} else {
			for (nb::handle item : parameters) {
				bound.push_back({std::string(), PythonToValue(connection, item, ctx)});
			}
		}

		// Parameters need a parsed statement; the string overload takes none.
		// Exactly one statement, so a caller cannot smuggle a second past the
		// parameter binding.
		auto statements = connection.ParseSQL(sql);
		auto statement = statements.Next();
		if (!statement) {
			throw cxx::InvalidInputException("Invalid Input Error: no statement to execute");
		}
		if (statements.Next()) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: execute takes exactly one statement when binding parameters");
		}
		nb::gil_scoped_release release;
		return Result(owner, connection.Execute(statement, bound));
	}

	/// What a statement would produce, without running it.
	///
	/// The binder is the only authority on schema. Predicting types in Python
	/// would mean reimplementing DuckDB's type resolution and drifting from it.
	/// Returns the output columns and the parameters the statement expects.
	std::pair<std::vector<std::pair<std::string, std::string>>,
	          std::vector<std::pair<std::string, std::string>>>
	Bind(const std::string &sql) {
		auto statements = connection.ParseSQL(sql);
		auto statement = statements.Next();
		if (!statement) {
			throw cxx::InvalidInputException("Invalid Input Error: no statement to bind");
		}
		if (statements.Next()) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: bind takes exactly one statement");
		}
		nb::gil_scoped_release release;
		const auto signature = connection.Bind(statement);
		return {Fields(signature.output), Fields(signature.parameters)};
	}

	/// Register a Python callable as a scalar SQL function on this database.
	void CreateScalarFunction(const std::string &name, nb::object callable,
	                          const std::vector<std::string> &parameters, const std::string &returns,
	                          const std::string &null_handling, const std::string &stability,
	                          std::shared_ptr<ModuleState> module) {
		cxx::FunctionNullHandling nulls;
		if (null_handling == "default") {
			nulls = cxx::FunctionNullHandling::DEFAULT;
		} else if (null_handling == "special") {
			nulls = cxx::FunctionNullHandling::SPECIAL;
		} else {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: null_handling must be 'default' or 'special'");
		}
		cxx::FunctionStability level;
		if (stability == "consistent") {
			level = cxx::FunctionStability::CONSISTENT;
		} else if (stability == "volatile") {
			level = cxx::FunctionStability::VOLATILE;
		} else if (stability == "consistent_within_query") {
			level = cxx::FunctionStability::CONSISTENT_WITHIN_QUERY;
		} else {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: stability must be 'consistent', 'volatile' or "
			    "'consistent_within_query'");
		}

		std::vector<cxx::LogicalType> parameter_types;
		parameter_types.reserve(parameters.size());
		for (const auto &text : parameters) {
			parameter_types.push_back(connection.ParseType(text));
		}
		auto return_type = connection.ParseType(returns);
		// ANY leaves arguments un-cast, so the exec callback could no longer
		// trust the declared types it reads the vectors by. Supporting it
		// needs a bind callback that captures the bound argument types.
		for (const auto &type : parameter_types) {
			if (type.GetTypeId() == cxx::LogicalTypeId::ANY) {
				throw cxx::InvalidInputException(
				    "Invalid Input Error: ANY parameters are not supported yet");
			}
		}
		if (return_type.GetTypeId() == cxx::LogicalTypeId::ANY) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: an ANY return type is not supported yet");
		}

		auto function = cxx::ScalarFunction::Create(connection);
		function.SetName(name);
		function.WithSignature([&](cxx::FunctionSignature &signature) {
			for (size_t i = 0; i < parameter_types.size(); i++) {
				signature.AddParameter("arg" + std::to_string(i), parameter_types[i]);
			}
			signature.SetReturnType(return_type);
		});
		function.SetUserData<PyFunctionData>(std::move(callable), name, std::move(parameter_types),
		                                     nulls == cxx::FunctionNullHandling::DEFAULT, std::move(module));
		function.SetExecCallback(&PyScalarExec);
		function.SetNullHandling(nulls);
		function.SetStability(level);
		nb::gil_scoped_release release;
		function.Register();
	}

	void Interrupt() {
		connection.Interrupt();
	}

	std::string GetOption(const std::string &name) {
		return std::string(connection.GetOption(name).GetValue());
	}

	void SetOption(const std::string &name, const std::string &value) {
		connection.SetOption(cxx::DatabaseOption(name, value));
	}

private:
	static std::vector<std::pair<std::string, std::string>> Fields(const cxx::Schema &schema) {
		std::vector<std::pair<std::string, std::string>> fields;
		const auto count = schema.GetFieldCount();
		fields.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			fields.emplace_back(std::string(schema.GetFieldName(i)), schema.GetFieldType(i).ToText());
		}
		return fields;
	}

	std::shared_ptr<DatabaseState> owner;
	cxx::Connection connection;
};

class Database {
public:
	explicit Database(std::shared_ptr<DatabaseState> state) : state(std::move(state)) {
	}

	Connection Connect() {
		nb::gil_scoped_release release;
		return Connection(state, state->database->Connect());
	}

private:
	std::shared_ptr<DatabaseState> state;
};

} // namespace
} // namespace duckdb_python

NB_MODULE(_duckdb, m) {
	using namespace duckdb_python;

	m.doc() = "DuckDB Python extension module.";
	nb::register_exception_translator(&TranslateException);

	// Created here, in the module's exec function, so it belongs to this
	// interpreter and is reachable only through the bindings registered below.
	auto state = std::make_shared<ModuleState>();

	nb::class_<Database>(m, "Database")
	    .def("__init__",
	         // options is taken as a handle so None is accepted, matching the
	         // stub and the way Connection::execute takes its parameters.
	         [state](Database *self, const std::string &path, nb::handle options) {
		         std::vector<std::pair<std::string, std::string>> settings;
		         if (!options.is_none()) {
			         settings = nb::cast<std::vector<std::pair<std::string, std::string>>>(options);
		         }
		         new (self) Database(std::make_shared<DatabaseState>(state, path, settings));
	         },
	         nb::arg("path") = std::string(":memory:"), nb::arg("options") = nb::none())
	    .def("connect", &Database::Connect);

	nb::class_<Connection>(m, "Connection")
	    .def("execute",
	         [state](Connection &self, const std::string &sql, nb::handle parameters) {
		         return self.Execute(sql, parameters, state->conversion);
	         },
	         nb::arg("sql"), nb::arg("parameters") = nb::none())
	    .def("bind", &Connection::Bind, nb::arg("sql"))
	    .def("create_scalar_function",
	         [state](Connection &self, const std::string &name, nb::object callable,
	                 const std::vector<std::string> &parameters, const std::string &returns,
	                 const std::string &null_handling, const std::string &stability) {
		         self.CreateScalarFunction(name, std::move(callable), parameters, returns, null_handling,
		                                   stability, state);
	         },
	         nb::arg("name"), nb::arg("callable"), nb::arg("parameters"), nb::arg("returns"),
	         nb::arg("null_handling"), nb::arg("stability"))
	    .def("interrupt", &Connection::Interrupt)
	    .def("get_option", &Connection::GetOption, nb::arg("name"))
	    .def("set_option", &Connection::SetOption, nb::arg("name"), nb::arg("value"));

	nb::class_<Result>(m, "Result")
	    .def_prop_ro("schema", &Result::Schema)
	    .def("fetch_all", [state](Result &self) { return self.FetchAll(state->conversion); })
	    .def("close", &Result::Close)
	    .def("drain", &Result::Drain)
	    .def_prop_ro("result_type", &Result::ResultType)
	    .def("fetch_rows",
	         [state](Result &self, size_t count) { return self.FetchRows(state->conversion, count); },
	         nb::arg("count"))
	    .def("fetch_chunk_view", &Result::FetchChunkView)
	    .def_prop_ro("schema_types", &Result::SchemaTypes)
	    .def_prop_ro("statement_type", &Result::StatementTypeName);

	nb::class_<ChunkView>(m, "ChunkView")
	    .def_prop_ro("row_count", &ChunkView::RowCount)
	    .def_prop_ro("row_offset", &ChunkView::RowOffset)
	    .def_prop_ro("column_count", &ChunkView::ColumnCount)
	    .def("type_id", &ChunkView::TypeId, nb::arg("column"))
	    .def("type_text", &ChunkView::TypeText, nb::arg("column"))
	    // keep_alive: the memoryviews borrow the chunk's memory, so the view
	    // object must outlive them whatever the caller drops.
	    .def("data", &ChunkView::Data, nb::arg("column"), nb::keep_alive<0, 1>())
	    .def("validity", &ChunkView::Validity, nb::arg("column"), nb::keep_alive<0, 1>())
	    .def("decimal_scale", &ChunkView::DecimalScale, nb::arg("column"))
	    .def("enum_values", &ChunkView::EnumValues, nb::arg("column"))
	    .def("values",
	         [state](ChunkView &self, cxx::idx_t column) { return self.Values(column, state->conversion); },
	         nb::arg("column"));

	m.def("library_version", []() { return cxx::LibraryVersion(); },
	      "The version of the DuckDB engine this extension is linked against.");
}
