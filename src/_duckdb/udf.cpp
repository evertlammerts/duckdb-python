//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/udf.cpp
//
//
//===----------------------------------------------------------------------===//

#include "udf.hpp"

#include "conversion/python_to_declared_type.hpp"
#include "conversion/python_to_value.hpp"
#include "conversion/value_to_python.hpp"

#include <optional>
#include <utility>

namespace duckdb_python {
namespace {

/// What a registered Python scalar function carries into every call.
///
/// The callable is borrowed, not owned: an owning reference here would be invisible to Python's cycle
/// collector, so a callable that reaches its own connection would keep the database alive forever. The
/// database that registered the function owns it instead, and by the time that is dropped no query can still run.
struct PyFunctionData {
	PyFunctionData(nb::handle callable, std::string name, TypeTree returns, bool skip_nulls,
	               std::shared_ptr<ModuleState> module)
	    : callable(callable), name(std::move(name)), returns(std::move(returns)), skip_nulls(skip_nulls),
	      module(std::move(module)) {
	}

	nb::handle callable;
	std::string name;
	// Described once here; engine threads only read it, and reading the type's children through the C API per row
	// would serialize the type each time.
	TypeTree returns;
	bool skip_nulls;
	std::shared_ptr<ModuleState> module;
};

/// What only the bind callback can read about a call: the type each argument arrives in, since the C++ API's Vector
/// exposes none, and the names of the arguments passed by name, which follow the positional ones. The engine merges
/// two identical calls only when their bind data compares equal.
struct BoundCall {
	std::vector<cxx::LogicalType> types;
	std::vector<std::string> keywords;

	bool operator==(const BoundCall &other) const {
		return types == other.types && keywords == other.keywords;
	}
};

void PyScalarBind(cxx::ScalarFunction::BindInput &input) {
	BoundCall call;
	const auto count = input.GetArgCount();
	const auto named = input.GetArgumentCounts().NamedOffset();
	call.types.reserve(count);
	for (cxx::idx_t a = 0; a < count; a++) {
		call.types.push_back(input.GetArgType(a));
		if (a >= named) {
			call.keywords.push_back(input.GetArgName(a));
		}
	}
	input.SetBindData<BoundCall>(std::move(call));
}

/// Why a Python value did not become a value of the declared type, for the conversion failure being handled, told
/// after `subject` ("the UDF 'f' returned"); `declared` is null for ANY. Any other exception propagates.
std::string ConversionFailure(const std::string &subject, const cxx::LogicalType *declared) {
	try {
		throw;
	} catch (const DeclaredTypeRefusal &refusal) {
		return subject + " " + refusal.GetRawMessage();
	} catch (const UnsupportedTypeException &error) {
		return subject + " a value of type " + error.TypeName() + ", which cannot be converted to a DuckDB value";
	} catch (const cxx::Exception &error) {
		const auto &body = error.GetRawMessage();
		const auto into = declared ? " to its declared type " + declared->ToText() : std::string();
		return subject + " a value that cannot be converted" + into + ": " + (body.empty() ? error.what() : body);
	}
}

/// A Python default as the engine keeps it, converted as a returned value is, or for ANY as a query parameter is, so
/// a default its parameter cannot hold is refused once here rather than by every call that leaves it out.
cxx::Value DefaultValue(cxx::Factory &factory, const ParameterSpec &parameter, const TypeTree &type,
                        ConversionContext &conversion) {
	const auto subject = "the default of parameter '" + parameter.name + "'";
	const bool any = type.id == cxx::LogicalTypeId::ANY;
	try {
		if (any) {
			return PythonToValue(factory, parameter.default_value, conversion, TimestampPrecision::MICROSECONDS);
		}
		return PythonToDeclaredType(factory, parameter.default_value, type, conversion);
	} catch (const cxx::Exception &) {
		ThrowInvalidInput(ConversionFailure(subject + " is", any ? nullptr : &type.type));
	} catch (nb::python_error &error) {
		// Ctrl-C and SystemExit stop the program, as they do when a query parameter converts.
		if (!error.matches(PyExc_Exception)) {
			throw;
		}
		ThrowInvalidInput(subject + " cannot be converted: " + DescribePythonError(error));
	} catch (const std::exception &error) {
		ThrowInvalidInput(subject + " cannot be converted: " + error.what());
	}
}

/// Called from DuckDB's own threads, so the GIL is taken here once per batch.
void PyScalarExec(cxx::ScalarFunction::ExecInput &input) {
	auto &data = input.GetUserData<PyFunctionData>();
	const auto &bound = input.GetBindData<BoundCall>();
	const auto rows = input.GetRowCount();
	const auto count = input.GetArgCount();
	const auto positional = count - bound.keywords.size();
	auto &context = input.GetContext();
	auto result = input.GetResult();

	// Fenced: the function may close an Arrow stream of this very query, which must not wait on this thread.
	FencedGil gil;
	try {
		std::vector<nb::list> columns;
		columns.reserve(count);
		for (cxx::idx_t a = 0; a < count; a++) {
			auto argument = input.GetArg(a);
			columns.push_back(VectorElements(argument, bound.types.at(a), 0, rows, data.module->conversion));
		}
		nb::object keywords;
		if (!bound.keywords.empty()) {
			keywords = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(bound.keywords.size())));
			if (!keywords.is_valid()) {
				throw nb::python_error();
			}
			for (size_t k = 0; k < bound.keywords.size(); k++) {
				const auto &keyword = bound.keywords[k];
				PyObject *name = PyUnicode_FromStringAndSize(keyword.data(), static_cast<Py_ssize_t>(keyword.size()));
				if (name == nullptr) {
					throw nb::python_error();
				}
				// SetItem steals the new reference whatever it returns.
				if (PyTuple_SetItem(keywords.ptr(), static_cast<Py_ssize_t>(k), name) != 0) {
					throw nb::python_error();
				}
			}
		}
		result.SetSize(rows);
		// Borrowed from the columns, which outlive every call.
		std::vector<PyObject *> arguments(count);
		for (cxx::idx_t r = 0; r < rows; r++) {
			bool any_null = false;
			for (cxx::idx_t a = 0; a < count; a++) {
				arguments[a] = PyList_GetItem(columns[a].ptr(), static_cast<Py_ssize_t>(r));
				any_null = any_null || arguments[a] == Py_None;
			}
			// NULL in means NULL out, but DuckDB still runs the batch over those rows, so the skip is here.
			if (any_null && data.skip_nulls) {
				result.SetNull(r);
				continue;
			}
			PyObject *returned = PyObject_Vectorcall(data.callable.ptr(), arguments.data(), positional, keywords.ptr());
			if (returned == nullptr) {
				throw nb::python_error();
			}
			const auto object = nb::steal(returned);
			if (object.is_none()) {
				result.SetNull(r);
				continue;
			}
			try {
				result.SetValue(
				    r, PythonToDeclaredType(context.GetFactory(), object, data.returns, data.module->conversion));
			} catch (const cxx::Exception &) {
				throw cxx::InvalidInputException(
				    ConversionFailure("the UDF '" + data.name + "' returned", &data.returns.type));
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
                            const std::vector<ParameterSpec> &parameters, const std::string &returns,
                            cxx::FunctionNullHandling nulls, cxx::FunctionStability level,
                            std::shared_ptr<ModuleState> module) {
	auto &factory = connection.GetFactory();
	std::vector<TypeTree> types;
	std::vector<std::optional<cxx::Value>> defaults;
	types.reserve(parameters.size());
	defaults.reserve(parameters.size());
	for (const auto &parameter : parameters) {
		// The type parser refuses ANY, which only a signature may hold.
		types.push_back(TypeTreeOf(parameter.type == "ANY"
		                               ? factory.CreateType(cxx::LogicalTypeId::ANY)
		                               : WithoutGil([&] { return factory.ParseType(parameter.type); })));
		defaults.push_back(parameter.has_default
		                       ? std::optional(DefaultValue(factory, parameter, types.back(), module->conversion))
		                       : std::nullopt);
	}
	auto return_type = WithoutGil([&] { return factory.ParseType(returns); });

	auto function = cxx::ScalarFunction::Create(factory);
	function.SetName(name);
	function.WithSignature([&](cxx::FunctionSignature &signature) {
		for (size_t i = 0; i < parameters.size(); i++) {
			if (defaults[i]) {
				signature.AddParameter(parameters[i].name, types[i].type, *defaults[i], parameters[i].kind);
			} else {
				signature.AddParameter(parameters[i].name, types[i].type, parameters[i].kind);
			}
		}
		signature.SetReturnType(return_type);
	});
	function.SetUserData<PyFunctionData>(callable, name, TypeTreeOf(std::move(return_type)),
	                                     nulls == cxx::FunctionNullHandling::DEFAULT, std::move(module));
	function.SetBindCallback(&PyScalarBind);
	function.SetExecCallback(&PyScalarExec);
	function.SetNullHandling(nulls);
	function.SetStability(level);
	nb::gil_scoped_release release;
	connection.Register(function);
}

} // namespace duckdb_python
