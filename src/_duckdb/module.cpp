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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow_export.hpp"
#include "arrowc.hpp"
#include "chunkview.hpp"
#include "lifetime.hpp"
#include "conversion/python_to_value.hpp"
#include "conversion/sql_types.hpp"
#include "conversion/untyped.hpp"
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
	      database(IsUntypedMarker(this->module->environment, path, options)) {
		WithoutGil([&] { InstallRegistryScan(database, registry, this->module); });
	}

	std::unique_ptr<Connection> Connect();

	/// Keep `callable` alive for as long as DuckDB may call it; duplicate connections register concurrently.
	void KeepCallable(nb::object callable) {
		nb::ft_lock_guard guard(callables_lock);
		callables.push_back(std::move(callable));
	}

	/// Hand the kept callables to the caller, which drops them outside the lock.
	std::vector<nb::object> TakeCallables() {
		nb::ft_lock_guard guard(callables_lock);
		return std::exchange(callables, {});
	}

	/// A copy for a stream to pin: its query may call these long after this Database is gone.
	std::vector<nb::object> CallablesSnapshot() {
		nb::ft_lock_guard guard(callables_lock);
		return callables;
	}

	/// Undo a KeepCallable whose registration then failed: the pin must not outlive the publication attempt.
	void AbandonCallable(nb::handle callable) {
		nb::object dropped;
		{
			nb::ft_lock_guard guard(callables_lock);
			for (auto it = callables.rbegin(); it != callables.rend(); ++it) {
				if (it->ptr() == callable.ptr()) {
					dropped = std::move(*it);
					callables.erase(std::next(it).base());
					break;
				}
			}
		}
	}

	/// Only for the garbage collector's visit, which runs with every other thread stopped.
	const std::vector<nb::object> &CallablesForTraversal() const {
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
	static cxx::Instance IsUntypedMarker(cxx::Environment &environment, const std::string &path,
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
	nb::ft_mutex callables_lock;
	// Declared before the database so they are destroyed after it, since DuckDB only borrows these callables.
	std::vector<nb::object> callables;
	std::shared_ptr<Registry> registry;
	cxx::Instance database;
};

// A None or an empty container is an untyped value: it converts to the marker type, since the engine refuses a
// value without one. Only when such a value survives conversion is the statement bound once more, and the parameter
// takes the type the binder expects at its position; a value with its own type is never cast, and the engine matches
// $name without case, so the pairing here does too. The expectation is only a hint: an unresolved parameter leaves
// the binder silent about every one after it, and some expectations hold ANY or pair no field, so what the hint
// cannot type cleanly goes to the engine as it is. Refused is only an untyped value at the statement's first free
// position, where nothing could type it.
void FillUntypedFromStatement(cxx::Connection &live, const cxx::SqlStatement &statement,
                              std::vector<cxx::NamedParam> &bound, const std::vector<bool> &untyped) {
	if (std::find(untyped.begin(), untyped.end(), true) == untyped.end()) {
		return;
	}
	const auto signature = WithoutGil([&] { return live.Bind(statement); });
	const auto &wanted = signature.parameters;
	const auto count = wanted.GetFieldCount();
	std::optional<cxx::idx_t> first_free;
	for (cxx::idx_t j = 0; j < count; j++) {
		if (wanted.GetFieldType(j).GetTypeId() == cxx::LogicalTypeId::UNKNOWN) {
			first_free = j;
			break;
		}
	}
	for (std::size_t i = 0; i < bound.size(); i++) {
		auto &param = bound[i];
		if (!untyped[i]) {
			continue;
		}
		const auto key = param.name.empty() ? std::to_string(i + 1) : param.name;
		std::optional<cxx::idx_t> at;
		for (cxx::idx_t j = 0; j < count; j++) {
			if (EqualIgnoringCase(wanted.GetFieldName(j), key)) {
				at = j;
				break;
			}
		}
		if (!at) {
			// The engine's own count or name mismatch says more than a refusal here would.
			continue;
		}
		const auto expected = wanted.GetFieldType(*at);
		if (expected.GetTypeId() == cxx::LogicalTypeId::UNKNOWN) {
			// A bare NULL means NULL whatever its type, so it binds as it is.
			if (at == first_free && !param.value.IsNull()) {
				throw cxx::InvalidInputException(
				    "Invalid Input Error: the parameter $" + key +
				    " holds an untyped value (None, or an empty list, tuple or dict) and "
				    "the statement does not say its type; cast it to say its type, like $" +
				    key + "::VARCHAR[]");
			}
			continue;
		}
		if (ContainsUnknownOrAny(expected)) {
			continue;
		}
		if (param.value.IsNull()) {
			param.value = cxx::Value::CreateNull(live, expected);
			continue;
		}
		FillUntypedFromExpected(live, param.value, expected);
	}
}

/// `parameters`, a sequence filling $1, $2, ... in order or a mapping by name, converted and bound by name where
/// one is given; `untyped` marks the values whose type the statement must still say.
///
/// An empty name in the list handed to DuckDB means positional, and a statement cannot mix the two forms.
void ConvertParameters(cxx::Connection &live, nb::handle parameters, ConversionContext &ctx,
                       std::vector<cxx::NamedParam> &bound, std::vector<bool> &untyped) {
	const auto convert = [&](nb::handle value) {
		bool still = false;
		auto converted = PythonToValue(live, value, ctx, TimestampPrecision::MICROSECONDS, &still);
		untyped.push_back(still);
		return converted;
	};
	const auto bind_named = [&](nb::handle name, nb::handle value) {
		// Checked here because a failed nanobind cast surfaces as std::bad_cast, which names nothing.
		if (!nb::isinstance<nb::str>(name)) {
			throw cxx::InvalidInputException("Invalid Input Error: parameter names must be strings");
		}
		bound.push_back({nb::cast<std::string>(name), convert(value)});
	};
	// Converting a value can run Python code that mutates the container passed in, so every shape is snapshotted
	// before any value converts: what binds is what was passed.
	if (nb::isinstance<nb::dict>(parameters)) {
		std::vector<std::pair<nb::object, nb::object>> entries;
		for (auto entry : nb::cast<nb::dict>(parameters)) {
			entries.emplace_back(nb::borrow(entry.first), nb::borrow(entry.second));
		}
		for (const auto &[name, value] : entries) {
			bind_named(name, value);
		}
	} else if (nb::isinstance(parameters, ctx.mapping_cls)) {
		std::vector<std::pair<nb::object, nb::object>> entries;
		for (nb::handle entry : parameters.attr("items")()) {
			entries.emplace_back(nb::borrow(entry[0]), nb::borrow(entry[1]));
		}
		for (const auto &[name, value] : entries) {
			bind_named(name, value);
		}
	} else {
		std::vector<nb::object> items;
		for (nb::handle item : parameters) {
			items.push_back(nb::borrow(item));
		}
		for (const nb::object &item : items) {
			bound.push_back({std::string(), convert(item)});
		}
	}
}

/// Parameters need a parsed statement, and exactly one, so a second cannot slip past unparameterised.
cxx::SqlStatement ParseExactlyOne(cxx::Connection &live, const std::string &sql) {
	auto statements = live.ParseSQL(sql);
	auto statement = statements.Next();
	if (!statement) {
		throw cxx::InvalidInputException("Invalid Input Error: no statement to execute");
	}
	if (statements.Next()) {
		throw cxx::InvalidInputException(
		    "Invalid Input Error: execute takes exactly one statement when binding parameters");
	}
	return statement;
}

class Connection {
public:
	Connection(nb::object database, std::shared_ptr<ModuleState> module, cxx::Connection connection)
	    : connection(std::move(module), std::move(database), std::move(connection)) {
	}

	nb::handle Parent() const {
		return connection.Parent();
	}

	/// Run one statement, with `parameters` either a sequence filling $1, $2, ... in order or a mapping by name.
	std::unique_ptr<Result> Execute(const std::string &sql, nb::handle parameters) {
		auto held = Live();
		auto &live = *held.engine;
		if (parameters.is_none()) {
			auto result = WithoutGil([&] { return live.Execute(sql); });
			return std::make_unique<Result>(std::move(held.database), connection.Module(), std::move(result));
		}
		std::vector<cxx::NamedParam> bound;
		std::vector<bool> untyped;
		ConvertParameters(live, parameters, connection.Module()->conversion, bound, untyped);
		auto statement = ParseExactlyOne(live, sql);
		FillUntypedFromStatement(live, statement, bound, untyped);
		auto result = WithoutGil([&] { return live.Execute(statement, bound); });
		return std::make_unique<Result>(std::move(held.database), connection.Module(), std::move(result));
	}

	/// Run one statement like `Execute`, into an Arrow stream of at most `batch_size` rows per array.
	std::unique_ptr<ArrowStream> ExecuteArrow(const std::string &sql, nb::handle parameters, cxx::idx_t batch_size) {
		auto held = Live();
		auto &live = *held.engine;
		const cxx::ArrowFormat format {batch_size};
		std::optional<cxx::ArrowResult> result;
		if (parameters.is_none()) {
			result.emplace(WithoutGil([&] { return live.Execute(sql, format); }));
		} else {
			std::vector<cxx::NamedParam> bound;
			std::vector<bool> untyped;
			ConvertParameters(live, parameters, connection.Module()->conversion, bound, untyped);
			auto statement = ParseExactlyOne(live, sql);
			FillUntypedFromStatement(live, statement, bound, untyped);
			result.emplace(WithoutGil([&] { return live.Execute(statement, bound, format); }));
		}
		// An expanding statement knows its shape only once stepping reaches it; the stream reports those late.
		bool rows = true;
		try {
			rows = WithoutGil([&] { return result->GetResultType(); }) == cxx::ResultType::QUERY_RESULT;
		} catch (const cxx::Exception &error) {
			if (error.GetCode() != DUCKDB_V2_ERROR_INPUT_INVALID) {
				throw;
			}
		}
		if (!rows) {
			throw cxx::InvalidInputException(
			    "Invalid Input Error: the statement returns no rows, so there is no Arrow stream to read");
		}
		auto kept = Database::From(held.database).CallablesSnapshot();
		return std::make_unique<ArrowStream>(std::move(held.database), connection.Module(),
		                                     std::weak_ptr<cxx::Connection>(held.engine), std::move(*result),
		                                     std::move(kept));
	}

	/// The columns a statement would produce and the parameters it expects, asked of DuckDB rather than guessed.
	std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>>
	Bind(const std::string &sql) {
		auto held = Live();
		auto &live = *held.engine;
		auto statements = live.ParseSQL(sql);
		auto statement = statements.Next();
		if (!statement) {
			throw cxx::InvalidInputException("Invalid Input Error: no statement to bind");
		}
		if (statements.Next()) {
			throw cxx::InvalidInputException("Invalid Input Error: bind takes exactly one statement");
		}
		nb::gil_scoped_release release;
		const auto signature = live.Bind(statement);
		return {FieldTexts(signature.output), FieldTexts(signature.parameters)};
	}

	/// Register a Python callable as a scalar SQL function on this database.
	void CreateScalarFunction(const std::string &name, nb::object callable, const std::vector<std::string> &parameters,
	                          const std::string &returns, cxx::FunctionNullHandling nulls,
	                          cxx::FunctionStability level) {
		auto held = Live();
		auto &owner = Database::From(held.database);
		// Kept BEFORE the engine can hand the name to anyone: a stream on another connection snapshots the
		// kept callables when it starts, so a function must never be visible without its lifetime pin.
		owner.KeepCallable(callable);
		try {
			RegisterScalarFunction(*held.engine, name, callable, parameters, returns, nulls, level,
			                       connection.Module());
		} catch (...) {
			owner.AbandonCallable(callable);
			throw;
		}
	}

	/// Register a Python object as the table `name`; a one-shot object is a stream, readable once, and `numpy_scan`
	/// says the numpy scan reads it rather than the Arrow scan.
	void RegisterObject(const std::string &name, nb::object object, bool one_shot, bool numpy_scan) {
		auto held = Live();
		Database::From(held.database).Objects().Add(name, std::move(object), one_shot, numpy_scan);
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
	for (const auto &callable : database.CallablesForTraversal()) {
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
		auto released = database.TakeCallables();
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

int ClearArrowStream(PyObject *self) {
	if (nb::inst_ready(self)) {
		nb::inst_ptr<ArrowStream>(self)->GcClear();
	}
	return 0;
}

int TraverseArrowStream(PyObject *self, visitproc visit, void *arg) {
	Py_VISIT(Py_TYPE(self));
	if (!nb::inst_ready(self)) {
		return 0;
	}
	return nb::inst_ptr<ArrowStream>(self)->Traverse(visit, arg);
}

const PyType_Slot kArrowStreamSlots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(&TraverseArrowStream)},
    {Py_tp_clear, reinterpret_cast<void *>(&ClearArrowStream)},
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
	    .def(
	        "__init__",
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

	nb::class_<ArrowStream>(m, "ArrowStream", nb::type_slots(kArrowStreamSlots))
	    .def("__arrow_c_stream__", &ArrowStream::Capsule, nb::arg("requested_schema") = nb::none())
	    .def_prop_ro("error", &ArrowStream::Error)
	    .def_prop_ro("live", &ArrowStream::Live)
	    .def_prop_ro("close_pending", &ArrowStream::ClosePending)
	    .def("close", &ArrowStream::Close);

	nb::class_<Connection>(m, "Connection", nb::type_slots(ChildSlots<Connection>()))
	    .def("execute", &Connection::Execute, nb::arg("sql"), nb::arg("parameters") = nb::none())
	    .def("execute_arrow", &Connection::ExecuteArrow, nb::arg("sql"), nb::arg("parameters") = nb::none(),
	         nb::arg("batch_size") = cxx::idx_t(0))
	    .def("bind", &Connection::Bind, nb::arg("sql"))
	    .def("create_scalar_function", &Connection::CreateScalarFunction, nb::arg("name"), nb::arg("callable"),
	         nb::arg("parameters"), nb::arg("returns"), nb::arg("null_handling"), nb::arg("stability"))
	    .def("register_object", &Connection::RegisterObject, nb::arg("name"), nb::arg("obj"), nb::arg("one_shot"),
	         nb::arg("numpy_scan"))
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
	    .def(
	        "values", [state](ChunkView &self, cxx::idx_t column) { return self.Values(column, state->conversion); },
	        nb::arg("column"));

	m.def(
	    "library_version", []() { return cxx::LibraryVersion(); },
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
	    nb::arg("object"),
	    "The name a capsule carries, which for Arrow data says what it holds; None for anything else.");
	m.def(
	    "temporal_literal",
	    [state](nb::handle value) {
		    ModuleState::LiteralLease connection(*state);
		    const auto bound = PythonToValue(*connection, value, state->conversion, TimestampPrecision::MICROSECONDS);
		    if (bound.IsNull()) {
			    return std::string("NULL");
		    }
		    const auto type = bound.GetLogicalType();
		    if (type.GetTypeId() == cxx::LogicalTypeId::INTERVAL) {
			    // The engine's own text for an interval of two billion hours or more does not parse back.
			    const auto interval = bound.Get<cxx::interval_t>();
			    return "INTERVAL '" + std::to_string(interval.months) + " months " + std::to_string(interval.days) +
			           " days " + std::to_string(interval.micros) + " microseconds'";
		    }
		    std::string quoted;
		    for (const char c : bound.ToText()) {
			    quoted += c;
			    if (c == '\'') {
				    quoted += c;
			    }
		    }
		    return type.ToText() + " '" + quoted + "'";
	    },
	    nb::arg("value"),
	    "A date, time or duration as SQL text: converted as a query parameter is, then written as a literal of its "
	    "engine type in the engine's own text for it; NULL for a missing value.");
	m.def(
	    "literal_type",
	    [state](nb::handle value) {
		    ModuleState::LiteralLease connection(*state);
		    return PythonToValue(*connection, value, state->conversion, TimestampPrecision::MICROSECONDS)
		        .GetLogicalType()
		        .ToText();
	    },
	    nb::arg("value"),
	    "The engine type `value` binds as when it is a query parameter, refusing what a parameter refuses.");
	m.def("chain_streams", &ChainStreams, nb::arg("schema"), nb::arg("parts"),
	      "Several exports read as one stream, each part with the schema of `schema`; the caller guarantees that, "
	      "nothing checks it.");
}
