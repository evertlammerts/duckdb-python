//===----------------------------------------------------------------------===//
//                         DuckDB
//
// module.cpp
//
// Entry point of the duckdb Python extension module (_duckdb).
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
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

// The environment owns the databases opened through it, so it must outlive
// every connection. Holding both together keeps that ordering without asking
// the Python layer to care.
struct DatabaseState {
	DatabaseState(const std::string &path,
	              const std::vector<std::pair<std::string, std::string>> &options)
	    : environment() {
		if (options.empty()) {
			database = std::make_unique<cxx::Database>(environment.Open(path));
			return;
		}
		std::vector<cxx::DatabaseOption> opts;
		opts.reserve(options.size());
		for (const auto &[name, value] : options) {
			opts.emplace_back(name, value);
		}
		database = std::make_unique<cxx::Database>(environment.Open(path, opts));
	}

	cxx::Environment environment;
	std::unique_ptr<cxx::Database> database;
};

class Result {
public:
	Result(std::shared_ptr<DatabaseState> owner, cxx::QueryResult result)
	    : owner(std::move(owner)), result(std::move(result)) {
	}

	/// Column names paired with the text form of their type.
	std::vector<std::pair<std::string, std::string>> Schema() {
		const auto schema = result.GetSchema();
		std::vector<std::pair<std::string, std::string>> out;
		const auto count = schema.GetFieldCount();
		out.reserve(count);
		for (cxx::idx_t i = 0; i < count; i++) {
			out.emplace_back(std::string(schema.GetFieldName(i)), schema.GetFieldType(i).ToText());
		}
		return out;
	}

	/// Drain the result into a list of row tuples.
	nb::list FetchAll(ConversionContext &ctx) {
		nb::list rows;
		while (true) {
			// StepResult carries a DataChunk and so is not default
			// constructible; build it in place from the call.
			auto step = [this] {
				// The engine runs on its own threads and may block; never hold
				// the GIL across it, or a callback into Python deadlocks.
				nb::gil_scoped_release release;
				return result.Step();
			}();
			switch (step.status) {
			case cxx::QueryResult::StepStatus::CHUNK:
				AppendChunk(rows, step.chunk, ctx);
				break;
			case cxx::QueryResult::StepStatus::FINISHED:
				return rows;
			case cxx::QueryResult::StepStatus::CANCELLED: {
				nb::object module = nb::module_::import_("duckdb.exceptions");
				PyErr_SetString(module.attr("InterruptError").ptr(), "query was cancelled");
				throw nb::python_error();
			}
			case cxx::QueryResult::StepStatus::WAITING: {
				nb::gil_scoped_release release;
				result.Wait();
				break;
			}
			}
		}
	}

private:
	static void AppendChunk(nb::list &rows, const cxx::DataChunk &chunk, ConversionContext &ctx) {
		const auto columns = chunk.GetVectorCount();
		const auto row_count = chunk.GetRowCount();
		std::vector<cxx::Vector> vectors;
		vectors.reserve(columns);
		for (cxx::idx_t c = 0; c < columns; c++) {
			vectors.push_back(chunk.GetVector(c));
		}
		for (cxx::idx_t r = 0; r < row_count; r++) {
			nb::list row;
			for (cxx::idx_t c = 0; c < columns; c++) {
				row.append(ValueToPython(vectors[c].GetValue(r), ctx));
			}
			rows.append(nb::tuple(row));
		}
	}

	std::shared_ptr<DatabaseState> owner;
	cxx::QueryResult result;
};

class Connection {
public:
	Connection(std::shared_ptr<DatabaseState> owner, cxx::Connection connection)
	    : owner(std::move(owner)), connection(std::move(connection)) {
	}

	Result Execute(const std::string &sql) {
		nb::gil_scoped_release release;
		return Result(owner, connection.Execute(sql));
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

	// One conversion context per module: the Python constructors it holds are
	// looked up once rather than per value.
	auto conversion = std::make_shared<ConversionContext>();

	nb::class_<Database>(m, "Database")
	    .def("__init__",
	         [](Database *self, const std::string &path,
	            const std::vector<std::pair<std::string, std::string>> &options) {
		         new (self) Database(std::make_shared<DatabaseState>(path, options));
	         },
	         nb::arg("path") = std::string(":memory:"),
	         nb::arg("options") = std::vector<std::pair<std::string, std::string>>())
	    .def("connect", &Database::Connect);

	nb::class_<Connection>(m, "Connection")
	    .def("execute", &Connection::Execute, nb::arg("sql"))
	    .def("interrupt", &Connection::Interrupt)
	    .def("get_option", &Connection::GetOption, nb::arg("name"))
	    .def("set_option", &Connection::SetOption, nb::arg("name"), nb::arg("value"));

	nb::class_<Result>(m, "Result")
	    .def_prop_ro("schema", &Result::Schema)
	    .def("fetch_all", [conversion](Result &self) { return self.FetchAll(*conversion); });

	m.def("library_version", []() { return cxx::LibraryVersion(); },
	      "The version of the DuckDB engine this extension is linked against.");
}
