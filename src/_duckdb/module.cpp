//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/module.cpp
//
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "duckdb_cpp.hpp"
#include "pyconv.hpp"

namespace nb = nanobind;
namespace cxx = duckdb::cxx;

namespace duckdb_python {
namespace {

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

	/// Run the statement to completion and report how many rows it changed.
	///
	/// Side effects land when a result is drained, so a statement whose result
	/// is dropped without draining never takes effect at all.
	cxx::idx_t Drain() {
		auto live = Live();
		pending.reset();
		finished = true;
		nb::gil_scoped_release release;
		return live->Drain();
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
		{
			std::lock_guard<std::mutex> guard(state->lifetime);
			pending.reset();
			finished = true;
			released = std::move(state->result);
			state->result.reset();
		}
		// Drop the last reference, if it is the last, outside the lock and
		// without the GIL: destruction talks to the engine.
		nb::gil_scoped_release unlock;
		released.reset();
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
				while (offset < available && (count == 0 || nb::len(rows) < count)) {
					rows.append(RowAt(*pending, offset++, ctx));
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

private:
	/// Step until a chunk arrives or the result ends. False once it has ended.
	bool Advance() {
		while (true) {
			// StepResult carries a DataChunk and so is not default
			// constructible; build it in place from the call.
			// The reference is taken before the GIL is dropped and held for
			// the whole call, so a concurrent Close cannot free it here.
			auto live = Live();
			auto step = [&live] {
				// The engine runs on its own threads and may block; never hold
				// the GIL across it, or a callback into Python deadlocks.
				nb::gil_scoped_release release;
				return live->Step();
			}();
			switch (step.status) {
			case cxx::QueryResult::StepStatus::CHUNK:
				pending = std::move(step.chunk);
				offset = 0;
				return true;
			case cxx::QueryResult::StepStatus::FINISHED:
				finished = true;
				return false;
			case cxx::QueryResult::StepStatus::CANCELLED: {
				nb::object module = nb::module_::import_("duckdb.exceptions");
				PyErr_SetString(module.attr("InterruptError").ptr(), "query was cancelled");
				throw nb::python_error();
			}
			case cxx::QueryResult::StepStatus::WAITING: {
				nb::gil_scoped_release release;
				live->Wait();
				break;
			}
			}
		}
	}

	static nb::tuple RowAt(const cxx::DataChunk &chunk, cxx::idx_t row, ConversionContext &ctx) {
		const auto columns = chunk.GetVectorCount();
		nb::list values;
		for (cxx::idx_t c = 0; c < columns; c++) {
			values.append(ValueToPython(chunk.GetVector(c).GetValue(row), ctx));
		}
		return nb::tuple(values);
	}

	std::shared_ptr<DatabaseState> owner;
	std::shared_ptr<ResultState> state;
	std::optional<cxx::DataChunk> pending;
	cxx::idx_t offset = 0;
	bool finished = false;
};

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
	         nb::arg("count"));

	m.def("library_version", []() { return cxx::LibraryVersion(); },
	      "The version of the DuckDB engine this extension is linked against.");
}
