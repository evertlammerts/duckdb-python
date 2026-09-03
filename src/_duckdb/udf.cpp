//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/udf.cpp
//
//
//===----------------------------------------------------------------------===//

#include "udf.hpp"

#include <utility>

namespace duckdb_python {
namespace {

/// Carried by a registered Python scalar function and read by every exec
/// call.
///
/// The callable is borrowed, not owned. An owning reference here would be
/// invisible to Python's cycle collector, so a callable that reaches its own
/// connection (a bound method, a closure over a module global) would pin the
/// database forever. The owner is instead the Database's registry, which its
/// traverse slot visits. Its clear slot drops the registry and nothing else,
/// and by then no query can run: finalizers and weakref callbacks have all
/// run before any clear slot, and a garbage Database is reachable from
/// nothing that could call into the engine.
struct PyFunctionData {
	PyFunctionData(nb::handle callable, std::string name, std::vector<cxx::LogicalType> parameter_types,
	               bool skip_nulls, std::shared_ptr<ModuleState> module)
	    : callable(callable), name(std::move(name)), parameter_types(std::move(parameter_types)),
	      skip_nulls(skip_nulls), module(std::move(module)) {
	}

	nb::handle callable;
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
/// types: no signature carries ANY, since create_function refuses it and
/// ParseType would too, so the binder has cast them.
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
			} catch (const UnsupportedTypeException &error) {
				// pandas' NA is unbindable but means NULL, as the old
				// client treated it.
				if (IsPandasNA(object)) {
					result.SetNull(r);
					continue;
				}
				throw cxx::InvalidInputException("the UDF '" + data.name + "' returned a value of type " +
				                                 error.TypeName() +
				                                 ", which cannot be converted to a DuckDB value");
			}
		}
	} catch (const cxx::Exception &error) {
		// The engine prefixes what a callback reports with its own error
		// class, so it gets the body alone, which the facade keeps apart from
		// the prefixed what() of an engine error.
		const auto &body = error.GetRawMessage();
		throw cxx::InvalidInputException(body.empty() ? error.what() : body);
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

} // namespace

void RegisterScalarFunction(cxx::Connection &connection, const std::string &name, nb::handle callable,
                            const std::vector<std::string> &parameters, const std::string &returns,
                            cxx::FunctionNullHandling nulls, cxx::FunctionStability level,
                            std::shared_ptr<ModuleState> module) {
	std::vector<cxx::LogicalType> parameter_types;
	parameter_types.reserve(parameters.size());
	for (const auto &text : parameters) {
		parameter_types.push_back(connection.ParseType(text));
	}
	auto return_type = connection.ParseType(returns);

	auto function = cxx::ScalarFunction::Create(connection);
	function.SetName(name);
	function.WithSignature([&](cxx::FunctionSignature &signature) {
		for (size_t i = 0; i < parameter_types.size(); i++) {
			signature.AddParameter("arg" + std::to_string(i), parameter_types[i]);
		}
		signature.SetReturnType(return_type);
	});
	function.SetUserData<PyFunctionData>(callable, name, std::move(parameter_types),
	                                     nulls == cxx::FunctionNullHandling::DEFAULT, std::move(module));
	function.SetExecCallback(&PyScalarExec);
	function.SetNullHandling(nulls);
	function.SetStability(level);
	nb::gil_scoped_release release;
	function.Register();
}

} // namespace duckdb_python
