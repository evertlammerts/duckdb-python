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

/// What a registered Python scalar function carries into every call.
///
/// The callable is borrowed, not owned: an owning reference here would be invisible to Python's cycle
/// collector, so a callable that reaches its own connection would keep the database alive forever. The
/// Database's registry owns it instead, and by the time that is dropped no query can still run.
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

/// Called from DuckDB's own threads, so the GIL is taken here once per batch; the arguments arrive already cast.
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
			// NULL in means NULL out, but DuckDB still runs the batch over those rows, so the skip is here.
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
				// SetValue casts to the column's type, so the declared return type is enforced here.
				result.SetValue(r, PythonToValue(context, object, data.module->conversion));
			} catch (const UnsupportedTypeException &error) {
				throw cxx::InvalidInputException("the UDF '" + data.name + "' returned a value of type " +
				                                 error.TypeName() +
				                                 ", which cannot be converted to a DuckDB value");
			}
		}
	} catch (const cxx::Exception &error) {
		// DuckDB prefixes a callback's error with its own class name, so hand it the message body alone.
		const auto &body = error.GetRawMessage();
		throw cxx::InvalidInputException(body.empty() ? error.what() : body);
	} catch (nb::python_error &error) {
		// Rendered while the GIL is still held, and worded as the previous duckdb package did; tests match it.
		throw cxx::InvalidInputException("Python exception occurred while executing the UDF '" + data.name +
		                                 "': " + DescribePythonError(error));
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
