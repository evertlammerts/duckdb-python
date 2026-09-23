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

#include "arrowc.hpp"
#include "chunkview.hpp"
#include "lifetime.hpp"
#include "registry.hpp"
#include "result.hpp"
#include "udf.hpp"

namespace nb = nanobind;
namespace cxx = duckdb::cxx;

namespace duckdb_python {
namespace {

// Every DuckDB error carries a numeric code and duckdb.exceptions maps it to a class, so one catch suffices.
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

/// One open database, together with the Python functions and objects registered on it.
///
/// Every Connection and Result holds a reference to its Database, so the garbage collector sees the real
/// ownership and the database outlives them. The registered callables and objects are owned here and nowhere else.
class Database {
public:
	Database(std::shared_ptr<ModuleState> module, const std::string &path,
	         const std::vector<std::pair<std::string, std::string>> &options)
	    : module(std::move(module)), registry(std::make_shared<Registry>()),
	      database(Open(this->module->environment, path, options)) {
		WithoutGil([&] { InstallRegistryScan(database, registry, this->module); });
	}

	std::unique_ptr<Connection> Connect();

	std::vector<nb::object> &Callables() {
		return callables;
	}

	Registry &Objects() {
		return *registry;
	}

	/// The Database behind a reference a child holds.
	static Database &From(nb::handle object) {
		return *nb::inst_ptr<Database>(object);
	}

private:
	static cxx::Instance Open(cxx::Environment &environment, const std::string &path,
	                          const std::vector<std::pair<std::string, std::string>> &options) {
		auto instance = environment.CreateInstance();
		// A startup-only setting such as access_mode is accepted only before the first attach.
		for (const auto &[name, value] : options) {
			instance.SetOption(name, value);
		}
		instance.Attach(path, true);
		return instance;
	}

	std::shared_ptr<ModuleState> module;
	// Declared before the database so they are destroyed after it, since DuckDB only borrows these callables.
	std::vector<nb::object> callables;
	std::shared_ptr<Registry> registry;
	cxx::Instance database;
};

class Connection {
public:
	Connection(nb::object database, std::shared_ptr<ModuleState> module, cxx::Connection connection)
	    : connection(std::move(module), std::move(database), std::move(connection)) {
	}

	nb::handle Parent() const {
		return connection.Parent();
	}

	/// Run one statement, with `parameters` either a sequence filling $1, $2, ... in order or a mapping by name.
	///
	/// An empty name in the list handed to DuckDB means positional, and a statement cannot mix the two forms.
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
				// Checked here because a failed nanobind cast surfaces as std::bad_cast, which names nothing.
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

		// Parameters need a parsed statement, and exactly one, so a second cannot slip past unparameterised.
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

	/// The columns a statement would produce and the parameters it expects, asked of DuckDB rather than guessed.
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
		// DuckDB borrows the callable only once registration succeeds, so only then must the registry keep it.
		owner.Callables().push_back(std::move(callable));
	}

	/// Register a Python object as the table `name`; a one-shot object is a stream, readable once.
	void RegisterObject(const std::string &name, nb::object object, bool one_shot) {
		auto held = Live();
		Database::From(held.database).Objects().Add(name, std::move(object), one_shot);
	}

	bool UnregisterObject(const std::string &name) {
		auto held = Live();
		return Database::From(held.database).Objects().Remove(name);
	}

	/// "stream" or "object" for a registered name, None otherwise.
	std::optional<std::string> RegisteredKind(const std::string &name) {
		auto held = Live();
		auto entry = Database::From(held.database).Objects().ByName(name);
		if (!entry) {
			return std::nullopt;
		}
		return std::string(entry->one_shot ? "stream" : "object");
	}

	void Interrupt() {
		Live().engine->Interrupt();
	}

	std::string GetOption(const std::string &name) {
		return std::string(Live().engine->GetOption(name).GetValue());
	}

	void SetOption(const std::string &name, const std::string &value) {
		Live().engine->SetOption(name, value);
	}

	/// Disconnect now rather than at collection; repeatable, and every other method refuses afterwards.
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

/// Garbage collector hooks: each object reports its type and every Python reference it holds, or cycles leak.
int TraverseDatabase(PyObject *self, visitproc visit, void *arg) {
	Py_VISIT(Py_TYPE(self));
	// Not constructed yet when the constructor raised.
	if (!nb::inst_ready(self)) {
		return 0;
	}
	auto &database = *nb::inst_ptr<Database>(self);
	for (const auto &callable : database.Callables()) {
		Py_VISIT(callable.ptr());
	}
	for (const auto &object : database.Objects().Objects()) {
		Py_VISIT(object.ptr());
	}
	return 0;
}

/// Drops only the callables and objects, since a Connection collected in the same pass may still use the database.
int ClearDatabase(PyObject *self) {
	if (nb::inst_ready(self)) {
		auto &database = *nb::inst_ptr<Database>(self);
		database.Callables().clear();
		database.Objects().Clear();
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

	// Created per import, so each interpreter gets its own and reaches it only through the bindings below.
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
	         // A handle so None is accepted, matching the type stub and how Connection::execute takes parameters.
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
	    .def("register_object", &Connection::RegisterObject, nb::arg("name"), nb::arg("obj"), nb::arg("one_shot"))
	    .def("unregister_object", &Connection::UnregisterObject, nb::arg("name"))
	    .def("registered_kind", &Connection::RegisteredKind, nb::arg("name"))
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
	    // keep_alive: the memoryviews borrow this object's memory, so it stays alive for as long as they do.
	    .def("data", &ChunkView::Data, nb::arg("column"), nb::keep_alive<0, 1>())
	    .def("validity", &ChunkView::Validity, nb::arg("column"), nb::keep_alive<0, 1>())
	    .def("decimal_scale", &ChunkView::DecimalScale, nb::arg("column"))
	    .def("enum_values", &ChunkView::EnumValues, nb::arg("column"))
	    .def("values",
	         [state](ChunkView &self, cxx::idx_t column) { return self.Values(column, state->conversion); },
	         nb::arg("column"));

	m.def("library_version", []() { return cxx::LibraryVersion(); },
	      "The DuckDB version this extension module is linked against.");
	m.def(
	    "capsule_name",
	    [](nb::handle object) -> std::optional<std::string> {
		    if (!PyCapsule_CheckExact(object.ptr())) {
			    return std::nullopt;
		    }
		    const char *name = PyCapsule_GetName(object.ptr());
		    return std::string(name ? name : "");
	    },
	    nb::arg("object"), "The name a capsule carries, which for Arrow data says what it holds; None for anything else.");
	m.def("chain_streams", &ChainStreams, nb::arg("schema"), nb::arg("parts"),
	      "Several exports read as one stream, each part with the schema of `schema`, which the caller guarantees; "
	      "the chain does not check it.");
}
