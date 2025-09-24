#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

void InitializeStaticMethods(py::module_ &m) {
	const char *docs;

	// Constant Expression
	docs = "Create a constant expression - e.g. SELECT <constant> - from the provided value.";
	m.def("ConstantExpression", &DuckDBPyExpression::ConstantExpression, py::arg("value"), docs);

	// ColumnRef Expression
	docs = "Create a column expression - e.g. SELECT column - from the optional column name(s).";
	m.def("ColumnExpression", &DuckDBPyExpression::ColumnExpression, docs);

	// Default Expression
	docs = "Create a default expression - e.g. INSERT a=DEFAULT WHERE b=42.";
	m.def("DefaultExpression", &DuckDBPyExpression::DefaultExpression, docs);

	// Case Expression
	docs = "Create a case expression - e.g. SELECT CASE WHEN b = 42 THEN 'universe' ELSE 'nothing' END as answer.";
	m.def("CaseExpression", &DuckDBPyExpression::CaseExpression, py::arg("condition"), py::arg("value"), docs);

	// Star Expression
	docs = "Create a star expression - e.g. SELECT * - with an optional list of columns to be excluded.";
	m.def("StarExpression", &DuckDBPyExpression::StarExpression, py::kw_only(), py::arg("exclude") = py::none(), docs);
	m.def("StarExpression", []() { return DuckDBPyExpression::StarExpression(); }, docs);

	// Function Expression
	docs = "Create a function expression - e.g. list_filter - on optional input columns.";
	m.def("FunctionExpression", &DuckDBPyExpression::FunctionExpression, py::arg("function_name"), docs);

	// Coalesce Operator
	docs = "Create a coalesce operator expression - e.g. COLEASCE(expr1, expr2, ...) - given one or more expressions.";
	m.def("CoalesceOperator", &DuckDBPyExpression::Coalesce, docs);

	// Lambda Expression
	docs = "Create a lambda expression - e.g. lambda param : param > 1 - given lhs (the function's params) and rhs "
	       "(some expression).";
	m.def("LambdaExpression", &DuckDBPyExpression::LambdaExpression, py::arg("lhs"), py::arg("rhs"), docs);

	// SQL Expression
	docs = "Create a SQL expression.";
	m.def("SQLExpression", &DuckDBPyExpression::SQLExpression, docs, py::arg("expression"));
}

static void InitializeDunderMethods(py::class_<DuckDBPyExpression, shared_ptr<DuckDBPyExpression>> &m) {
	const char *docs;

	docs = R"(
        Add other to self

        Parameters:
            other: The expression to add together with

        Returns:
            FunctionExpression: self '+' other
	)";

	m.def("__add__", &DuckDBPyExpression::Add, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__radd__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Add(a); }, py::arg("other"),
	    docs, py::is_operator());

	docs = R"(
        Negate the expression.

        Returns:
            FunctionExpression: -self
	)";
	m.def("__neg__", &DuckDBPyExpression::Negate, docs, py::is_operator());

	docs = R"(
        Subtract other from self

        Parameters:
            other: The expression to subtract from

        Returns:
		    FunctionExpression: self '-' other
	)";
	m.def("__sub__", &DuckDBPyExpression::Subtract, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rsub__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Subtract(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		Multiply self by other

		Parameters:
			other: The expression to multiply by

		Returns:
			FunctionExpression: self '*' other
	)";
	m.def("__mul__", &DuckDBPyExpression::Multiply, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rmul__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Multiply(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		Divide self by other

		Parameters:
			other: The expression to divide by

		Returns:
			FunctionExpression: self '/' other
	)";
	m.def("__div__", &DuckDBPyExpression::Division, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rdiv__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Division(a); },
	    py::arg("other"), docs, py::is_operator());

	m.def("__truediv__", &DuckDBPyExpression::Division, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rtruediv__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Division(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		(Floor) Divide self by other

		Parameters:
			other: The expression to (floor) divide by

		Returns:
			FunctionExpression: self '//' other
	)";
	m.def("__floordiv__", &DuckDBPyExpression::FloorDivision, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rfloordiv__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.FloorDivision(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		Modulo self by other

		Parameters:
			other: The expression to modulo by

		Returns:
			FunctionExpression: self '%' other
	)";
	m.def("__mod__", &DuckDBPyExpression::Modulo, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rmod__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Modulo(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		Power self by other

		Parameters:
			other: The expression to power by

		Returns:
			FunctionExpression: self '**' other
	)";
	m.def("__pow__", &DuckDBPyExpression::Power, py::arg("other"), docs, py::is_operator());
	m.def(
	    "__rpow__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Power(a); },
	    py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create an equality expression between two expressions

		Parameters:
			other: The expression to check equality with

		Returns:
			FunctionExpression: self '=' other
	)";
	m.def("__eq__", &DuckDBPyExpression::Equality, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create an inequality expression between two expressions

		Parameters:
			other: The expression to check inequality with

		Returns:
			FunctionExpression: self '!=' other
	)";
	m.def("__ne__", &DuckDBPyExpression::Inequality, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create a greater than expression between two expressions

		Parameters:
			other: The expression to check

		Returns:
			FunctionExpression: self '>' other
	)";
	m.def("__gt__", &DuckDBPyExpression::GreaterThan, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create a greater than or equal expression between two expressions

		Parameters:
			other: The expression to check

		Returns:
			FunctionExpression: self '>=' other
	)";
	m.def("__ge__", &DuckDBPyExpression::GreaterThanOrEqual, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create a less than expression between two expressions

		Parameters:
			other: The expression to check

		Returns:
			FunctionExpression: self '<' other
	)";
	m.def("__lt__", &DuckDBPyExpression::LessThan, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create a less than or equal expression between two expressions

		Parameters:
			other: The expression to check

		Returns:
			FunctionExpression: self '<=' other
	)";
	m.def("__le__", &DuckDBPyExpression::LessThanOrEqual, py::arg("other"), docs, py::is_operator());

	// AND, NOT and OR

	docs = R"(
		Binary-and self together with expr

		Parameters:
			other: The expression to AND together with self

		Returns:
			FunctionExpression: self '&' other
	)";
	m.def("__and__", &DuckDBPyExpression::And, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Binary-or self together with other

		Parameters:
			other: The expression to OR together with self

		Returns:
			FunctionExpression: self '|' other
	)";
	m.def("__or__", &DuckDBPyExpression::Or, py::arg("other"), docs, py::is_operator());

	docs = R"(
		Create a binary-not expression from self

		Returns:
			FunctionExpression: ~self
	)";
	m.def("__invert__", &DuckDBPyExpression::Not, docs, py::is_operator());

	docs = R"(
		Binary-and self together with other

		Parameters:
			other: The expression to AND together with self

		Returns:
			FunctionExpression: other '&' self
	)";
	m.def(
	    "__rand__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.And(a); }, py::arg("other"),
	    docs, py::is_operator());

	docs = R"(
		Binary-or self together with other

		Parameters:
			other: The expression to OR together with self

		Returns:
			FunctionExpression: other '|' self
	)";
	m.def(
	    "__ror__", [](const DuckDBPyExpression &a, const DuckDBPyExpression &b) { return b.Or(a); }, py::arg("other"),
	    docs, py::is_operator());
}

static void InitializeImplicitConversion(py::class_<DuckDBPyExpression, shared_ptr<DuckDBPyExpression>> &m) {
	m.def(py::init<>([](const string &name) {
		auto names = py::make_tuple(py::str(name));
		return DuckDBPyExpression::ColumnExpression(names);
	}));
	m.def(py::init<>([](const py::object &obj) {
		auto val = TransformPythonValue(obj);
		return DuckDBPyExpression::InternalConstantExpression(std::move(val));
	}));
	py::implicitly_convertible<py::str, DuckDBPyExpression>();
	py::implicitly_convertible<py::object, DuckDBPyExpression>();
}

void DuckDBPyExpression::Initialize(py::module_ &m) {
	auto expression =
	    py::class_<DuckDBPyExpression, shared_ptr<DuckDBPyExpression>>(m, "Expression", py::module_local());

	InitializeStaticMethods(m);
	InitializeDunderMethods(expression);
	InitializeImplicitConversion(expression);

	const char *docs;

	docs = R"(
		Print the stringified version of the expression.
	)";
	expression.def("show", &DuckDBPyExpression::Print, docs);

	docs = R"(
		Set the order by modifier to ASCENDING.
	)";
	expression.def("asc", &DuckDBPyExpression::Ascending, docs);

	docs = R"(
		Set the order by modifier to DESCENDING.
	)";
	expression.def("desc", &DuckDBPyExpression::Descending, docs);

	docs = R"(
		Set the NULL order by modifier to NULLS FIRST.
	)";
	expression.def("nulls_first", &DuckDBPyExpression::NullsFirst, docs);

	docs = R"(
		Set the NULL order by modifier to NULLS LAST.
	)";
	expression.def("nulls_last", &DuckDBPyExpression::NullsLast, docs);

	docs = R"(
		Create a binary IS NULL expression from self

		Returns:
			DuckDBPyExpression: self IS NULL
	)";
	expression.def("isnull", &DuckDBPyExpression::IsNull, docs);

	docs = R"(
		Create a binary IS NOT NULL expression from self

		Returns:
			DuckDBPyExpression: self IS NOT NULL
	)";
	expression.def("isnotnull", &DuckDBPyExpression::IsNotNull, docs);

	docs = R"(
		Return an IN expression comparing self to the input arguments.

		Returns:
			DuckDBPyExpression: The compare IN expression
	)";
	expression.def("isin", &DuckDBPyExpression::In, docs);

	docs = R"(
		Return a NOT IN expression comparing self to the input arguments.

		Returns:
			DuckDBPyExpression: The compare NOT IN expression
	)";
	expression.def("isnotin", &DuckDBPyExpression::NotIn, docs);

	docs = R"(
		Return the stringified version of the expression.

		Returns:
			str: The string representation.
	)";
	expression.def("__repr__", &DuckDBPyExpression::ToString, docs);

	expression.def("get_name", &DuckDBPyExpression::GetName, docs);

	docs = R"(
		Create a copy of this expression with the given alias.

		Parameters:
			name: The alias to use for the expression, this will affect how it can be referenced.

		Returns:
			Expression: self with an alias.
	)";
	expression.def("alias", &DuckDBPyExpression::SetAlias, py::arg("name"), docs);

	docs = R"(
		Add an additional WHEN <condition> THEN <value> clause to the CaseExpression.

		Parameters:
			condition: The condition that must be met.
			value: The value to use if the condition is met.

		Returns:
			CaseExpression: self with an additional WHEN clause.
	)";
	expression.def("when", &DuckDBPyExpression::When, py::arg("condition"), py::arg("value"), docs);

	docs = R"(
		Add an ELSE <value> clause to the CaseExpression.

		Parameters:
			value: The value to use if none of the WHEN conditions are met.

		Returns:
			CaseExpression: self with an ELSE clause.
	)";
	expression.def("otherwise", &DuckDBPyExpression::Else, py::arg("value"), docs);

	docs = R"(
		Create a CastExpression to type from self

		Parameters:
			type: The type to cast to

		Returns:
			CastExpression: self::type
	)";
	expression.def("cast", &DuckDBPyExpression::Cast, py::arg("type"), docs);

	docs = "";
	expression.def("between", &DuckDBPyExpression::Between, py::arg("lower"), py::arg("upper"), docs);

	docs = "";
	expression.def("collate", &DuckDBPyExpression::Collate, py::arg("collation"), docs);
}

} // namespace duckdb
