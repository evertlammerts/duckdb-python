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
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "chunkview.hpp"
#include "lifetime.hpp"
#include "result.hpp"
#include "udf.hpp"

namespace nb = nanobind;
namespace cxx = duckdb::cxx;

namespace duckdb_python {
namespace {

// An engine error carries a numeric code; duckdb.exceptions owns the mapping
// from code to class. One catch clause covers every engine exception, including
// the two the C++ API gives a dedicated type, since all of them carry the code.
void TranslateException(const std::exception_ptr &captured, void *payload) {
	auto &state = *static_cast<ModuleState *>(payload);
	try {
		std::rethrow_exception(captured);
	} catch (const cxx::Exception &e) {
		try {
			PyErr_SetString(state.ClassForCode(e.GetCode()).ptr(), e.what());
		} catch (...) {
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
	}
}

class Connection;

/// One open database: the engine instance, the module state it was opened
/// through, and the callables registered on it.
///
/// Every Connection and Result on this database holds a reference to it, so
/// the ownership graph the collector sees is the one that exists: children
/// keep the database alive and die first, and the engine instance goes with
/// the last of them. The callables are held here and only here; see
/// PyFunctionData for why the engine borrows them.
class Database {
public:
	Database(std::shared_ptr<ModuleState> module, const std::string &path,
	         const std::vector<std::pair<std::string, std::string>> &options)
	    : module(std::move(module)), database(Open(this->module->environment, path, options)) {
	}

	std::unique_ptr<Connection> Connect();

	std::vector<nb::object> &Callables() {
		return callables;
	}

	/// The Database behind a reference a child holds.
	static Database &From(nb::handle object) {
		return *nb::inst_ptr<Database>(object);
	}

private:
	static cxx::Database Open(cxx::Environment &environment, const std::string &path,
	                          const std::vector<std::pair<std::string, std::string>> &options) {
		if (options.empty()) {
			return environment.Open(path);
		}
		std::vector<cxx::DatabaseOption> opts;
		opts.reserve(options.size());
		for (const auto &[name, value] : options) {
			opts.emplace_back(name, value);
		}
		return environment.Open(path, opts);
	}

	std::shared_ptr<ModuleState> module;
	// The registry outlives the engine instance, in destruction as in the
	// collector's clear.
	std::vector<nb::object> callables;
	cxx::Database database;
};

class Connection {
public:
	Connection(nb::object database, std::shared_ptr<ModuleState> module, cxx::Connection connection)
	    : connection(std::move(module), std::move(database), std::move(connection)) {
	}

	nb::handle Parent() const {
		return connection.Parent();
	}

	/// Run one statement, optionally binding parameters.
	///
	/// `parameters` is either a sequence, binding $1, $2, ... in order, or a
	/// mapping, binding $name. NamedParam carries both shapes: an empty name
	/// means positional. A statement cannot mix the two, which the engine
	/// enforces at bind time.
	std::unique_ptr<Result> Execute(const std::string &sql, nb::handle parameters) {
		auto held = Live();
		auto &live = *held.engine;
		if (parameters.is_none()) {
			auto result = WithoutGil([&] { return live.Execute(sql); });
			return std::make_unique<Result>(std::move(held.database), connection.Module(), std::move(result));
		}

		auto &ctx = connection.Module()->conversion;
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
				                 PythonToValue(live, entry.second, ctx)});
			}
		} else {
			for (nb::handle item : parameters) {
				bound.push_back({std::string(), PythonToValue(live, item, ctx)});
			}
		}

		// Parameters need a parsed statement; the string overload takes none.
		// Exactly one statement, so a caller cannot smuggle a second past the
		// parameter binding.
		auto statements = live.ParseSQL(sql);
		auto statement = statements.Next();
		if (!statement) {
			throw cxx::InvalidInputException("Invalid Input Error: no statement to execute");
		}
		if (statements.Next()) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: execute takes exactly one statement when binding parameters");
		}
		auto result = WithoutGil([&] { return live.Execute(statement, bound); });
		return std::make_unique<Result>(std::move(held.database), connection.Module(), std::move(result));
	}

	/// What a statement would produce, without running it.
	///
	/// The binder is the only authority on schema. Predicting types in Python
	/// would mean reimplementing DuckDB's type resolution and drifting from it.
	/// Returns the output columns and the parameters the statement expects.
	std::pair<std::vector<std::pair<std::string, std::string>>,
	          std::vector<std::pair<std::string, std::string>>>
	Bind(const std::string &sql) {
		auto held = Live();
		auto &live = *held.engine;
		auto statements = live.ParseSQL(sql);
		auto statement = statements.Next();
		if (!statement) {
			throw cxx::InvalidInputException("Invalid Input Error: no statement to bind");
		}
		if (statements.Next()) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: bind takes exactly one statement");
		}
		nb::gil_scoped_release release;
		const auto signature = live.Bind(statement);
		return {FieldTexts(signature.output), FieldTexts(signature.parameters)};
	}

	/// Register a Python callable as a scalar SQL function on this database.
	void CreateScalarFunction(const std::string &name, nb::object callable,
	                          const std::vector<std::string> &parameters, const std::string &returns,
	                          cxx::FunctionNullHandling nulls, cxx::FunctionStability level) {
		auto held = Live();
		auto &owner = Database::From(held.database);
		RegisterScalarFunction(*held.engine, name, callable, parameters, returns, nulls, level,
		                       connection.Module());
		// Registered, so the engine now borrows the callable; the registry
		// keeps it alive from here on, and a failed registration left nothing
		// behind to keep.
		owner.Callables().push_back(std::move(callable));
	}

	void Interrupt() {
		Live().engine->Interrupt();
	}

	std::string GetOption(const std::string &name) {
		return std::string(Live().engine->GetOption(name).GetValue());
	}

	void SetOption(const std::string &name, const std::string &value) {
		Live().engine->SetOption(cxx::DatabaseOption(name, value));
	}

	/// Release the engine connection now rather than when this object is
	/// collected. Idempotent; every other method refuses afterwards, and a
	/// closed connection pins nothing.
	void Close() {
		connection.Release();
	}

private:
	Pinned<cxx::Connection> Live() {
		return connection.Acquire("connection is closed");
	}

	Owned<cxx::Connection> connection;
};

std::unique_ptr<Connection> Database::Connect() {
	auto connection = WithoutGil([&] { return database.Connect(); });
	return std::make_unique<Connection>(nb::find(*this), module, std::move(connection));
}

/// GC slots. An instance of a heap type holds its type, so every traverse
/// visits it. A child visits the one reference it holds to its Database and
/// a Database visits each callable it registered, so the collector's count
/// of every holder comes out right.
int TraverseDatabase(PyObject *self, visitproc visit, void *arg) {
	Py_VISIT(Py_TYPE(self));
	// Not constructed yet when the constructor raised.
	if (!nb::inst_ready(self)) {
		return 0;
	}
	for (const auto &callable : nb::inst_ptr<Database>(self)->Callables()) {
		Py_VISIT(callable.ptr());
	}
	return 0;
}

/// Drops the callables and nothing else. The engine instance must survive a
/// clear: a garbage Connection may still hold its engine connection until
/// its own clear or dealloc runs, and the collector orders those arbitrarily.
int ClearDatabase(PyObject *self) {
	if (nb::inst_ready(self)) {
		nb::inst_ptr<Database>(self)->Callables().clear();
	}
	return 0;
}

template <class T>
int TraverseChild(PyObject *self, visitproc visit, void *arg) {
	Py_VISIT(Py_TYPE(self));
	if (!nb::inst_ready(self)) {
		return 0;
	}
	Py_VISIT(nb::inst_ptr<T>(self)->Parent().ptr());
	return 0;
}

template <class T>
int ClearChild(PyObject *self) {
	if (nb::inst_ready(self)) {
		nb::inst_ptr<T>(self)->Close();
	}
	return 0;
}

const PyType_Slot kDatabaseSlots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(&TraverseDatabase)},
    {Py_tp_clear, reinterpret_cast<void *>(&ClearDatabase)},
    {0, nullptr},
};

template <class T>
const PyType_Slot *ChildSlots() {
	static const PyType_Slot slots[] = {
	    {Py_tp_traverse, reinterpret_cast<void *>(&TraverseChild<T>)},
	    {Py_tp_clear, reinterpret_cast<void *>(&ClearChild<T>)},
	    {0, nullptr},
	};
	return slots;
}

} // namespace
} // namespace duckdb_python

NB_MODULE(_duckdb, m) {
	using namespace duckdb_python;

	m.doc() = "DuckDB Python extension module.";

	// Created here, in the module's exec function, so it belongs to this
	// interpreter and is reachable only through the bindings registered below.
	auto state = std::make_shared<ModuleState>();
	nb::register_exception_translator(&TranslateException, state.get());

	nb::enum_<cxx::FunctionNullHandling>(m, "FunctionNullHandling")
	    .value("DEFAULT", cxx::FunctionNullHandling::DEFAULT)
	    .value("SPECIAL", cxx::FunctionNullHandling::SPECIAL);

	nb::enum_<cxx::FunctionStability>(m, "FunctionStability")
	    .value("CONSISTENT", cxx::FunctionStability::CONSISTENT)
	    .value("VOLATILE", cxx::FunctionStability::VOLATILE)
	    .value("CONSISTENT_WITHIN_QUERY", cxx::FunctionStability::CONSISTENT_WITHIN_QUERY);

	nb::class_<Database>(m, "Database", nb::type_slots(kDatabaseSlots))
	    .def("__init__",
	         // options is taken as a handle so None is accepted, matching the
	         // stub and the way Connection::execute takes its parameters.
	         [state](Database *self, const std::string &path, nb::handle options) {
		         std::vector<std::pair<std::string, std::string>> settings;
		         if (!options.is_none()) {
			         settings = nb::cast<std::vector<std::pair<std::string, std::string>>>(options);
		         }
		         new (self) Database(state, path, settings);
	         },
	         nb::arg("path") = std::string(":memory:"), nb::arg("options") = nb::none())
	    .def("connect", &Database::Connect);

	nb::class_<Connection>(m, "Connection", nb::type_slots(ChildSlots<Connection>()))
	    .def("execute", &Connection::Execute, nb::arg("sql"), nb::arg("parameters") = nb::none())
	    .def("bind", &Connection::Bind, nb::arg("sql"))
	    .def("create_scalar_function", &Connection::CreateScalarFunction, nb::arg("name"), nb::arg("callable"),
	         nb::arg("parameters"), nb::arg("returns"), nb::arg("null_handling"), nb::arg("stability"))
	    .def("interrupt", &Connection::Interrupt)
	    .def("get_option", &Connection::GetOption, nb::arg("name"))
	    .def("set_option", &Connection::SetOption, nb::arg("name"), nb::arg("value"))
	    .def("close", &Connection::Close);

	nb::class_<Result>(m, "Result", nb::type_slots(ChildSlots<Result>()))
	    .def_prop_ro("schema", &Result::Schema)
	    .def("fetch_all", &Result::FetchAll)
	    .def("close", &Result::Close)
	    .def("drain", &Result::Drain)
	    .def_prop_ro("result_type", &Result::ResultType)
	    .def("fetch_rows", &Result::FetchRows, nb::arg("count"))
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
