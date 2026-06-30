//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/expression/pyexpression.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb_python/python_conversion.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb/common/enums/order_type.hpp"

namespace duckdb {

//! Value-semantic wrapper around a parsed expression. Every combinator deep-copies its operands into a fresh
//! tree, so two wrappers never alias the same expression -- there is no shared ownership to model. Bound to
//! Python by value (returned as std::unique_ptr); implicit str/scalar/None -> Expression conversions are handled
//! by nanobind's value caster + the registered implicitly_convertible<>() rules (no custom shared_ptr caster).
struct DuckDBPyExpression {
public:
	explicit DuckDBPyExpression(unique_ptr<ParsedExpression> expr, OrderType order_type = OrderType::ORDER_DEFAULT,
	                            OrderByNullType null_order = OrderByNullType::ORDER_DEFAULT);

public:
	static void Initialize(nb::module_ &m);

	//! Convert an arbitrary Python object into an owned expression, applying the same implicit conversions as a
	//! by-value Expression parameter: an existing Expression is copied, a str becomes a column reference, and
	//! anything else (including None) becomes a constant. Used by the variadic (*args / list) call-sites which
	//! iterate handles manually and so cannot lean on nanobind's automatic argument conversion. Throws a generic
	//! "arguments of type Expression" error if the object cannot be converted.
	static std::unique_ptr<DuckDBPyExpression> ToExpression(nb::handle obj);
	//! Non-throwing variant: returns false (clearing any pending Python error) if `obj` cannot be converted, so a
	//! caller can raise a context-specific message. This reproduces the old try_cast<>() control flow without a caster.
	static bool TryToExpression(nb::handle obj, std::unique_ptr<DuckDBPyExpression> &result);

	string Type() const;

	string ToString() const;
	string GetName() const;
	void Print() const;
	std::unique_ptr<DuckDBPyExpression> Add(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Subtract(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Multiply(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Division(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> FloorDivision(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Modulo(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Power(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Negate();

	// Equality operations

	std::unique_ptr<DuckDBPyExpression> Equality(const DuckDBPyExpression &other);
	std::unique_ptr<DuckDBPyExpression> Inequality(const DuckDBPyExpression &other);
	std::unique_ptr<DuckDBPyExpression> GreaterThan(const DuckDBPyExpression &other);
	std::unique_ptr<DuckDBPyExpression> GreaterThanOrEqual(const DuckDBPyExpression &other);
	std::unique_ptr<DuckDBPyExpression> LessThan(const DuckDBPyExpression &other);
	std::unique_ptr<DuckDBPyExpression> LessThanOrEqual(const DuckDBPyExpression &other);

	std::unique_ptr<DuckDBPyExpression> SetAlias(const string &alias) const;
	// `value` is nb::object (not Expression) so it accepts None: nanobind rejects None for bound-type params
	// before implicit conversion runs, so None->NULL-constant has to go through ToExpression explicitly.
	std::unique_ptr<DuckDBPyExpression> When(const DuckDBPyExpression &condition, const nb::object &value);
	std::unique_ptr<DuckDBPyExpression> Else(const nb::object &value);

	std::unique_ptr<DuckDBPyExpression> Cast(const DuckDBPyType &type) const;
	std::unique_ptr<DuckDBPyExpression> Between(const DuckDBPyExpression &lower, const DuckDBPyExpression &upper);
	std::unique_ptr<DuckDBPyExpression> Collate(const string &collation);

	// AND, OR and NOT

	std::unique_ptr<DuckDBPyExpression> Not();
	std::unique_ptr<DuckDBPyExpression> And(const DuckDBPyExpression &other) const;
	std::unique_ptr<DuckDBPyExpression> Or(const DuckDBPyExpression &other) const;

	// IS NULL / IS NOT NULL

	std::unique_ptr<DuckDBPyExpression> IsNull();
	std::unique_ptr<DuckDBPyExpression> IsNotNull();

	// IN / NOT IN

	std::unique_ptr<DuckDBPyExpression> CreateCompareExpression(ExpressionType compare_type, const nb::args &args);
	std::unique_ptr<DuckDBPyExpression> In(const nb::args &args);
	std::unique_ptr<DuckDBPyExpression> NotIn(const nb::args &args);

	// Order modifiers

	std::unique_ptr<DuckDBPyExpression> Ascending();
	std::unique_ptr<DuckDBPyExpression> Descending();

	// Null order modifiers

	std::unique_ptr<DuckDBPyExpression> NullsFirst();
	std::unique_ptr<DuckDBPyExpression> NullsLast();

public:
	const ParsedExpression &GetExpression() const;
	std::unique_ptr<DuckDBPyExpression> Copy() const;

public:
	static std::unique_ptr<DuckDBPyExpression> StarExpression(nb::object exclude = nb::none());
	static std::unique_ptr<DuckDBPyExpression> ColumnExpression(const nb::args &column_name);
	static std::unique_ptr<DuckDBPyExpression> DefaultExpression();
	static std::unique_ptr<DuckDBPyExpression> ConstantExpression(const nb::object &value);
	static std::unique_ptr<DuckDBPyExpression> LambdaExpression(const nb::object &lhs, const DuckDBPyExpression &rhs);
	static std::unique_ptr<DuckDBPyExpression> CaseExpression(const DuckDBPyExpression &condition,
	                                                          const nb::object &value);
	static std::unique_ptr<DuckDBPyExpression> FunctionExpression(const string &function_name, const nb::args &args);
	static std::unique_ptr<DuckDBPyExpression> Coalesce(const nb::args &args);
	static std::unique_ptr<DuckDBPyExpression> SQLExpression(string sql);

public:
	// Internal functions (not exposed to Python)
	static std::unique_ptr<DuckDBPyExpression> InternalFunctionExpression(const string &function_name,
	                                                                      vector<unique_ptr<ParsedExpression>> children,
	                                                                      bool is_operator = false);

	static std::unique_ptr<DuckDBPyExpression> InternalUnaryOperator(ExpressionType type,
	                                                                 const DuckDBPyExpression &arg);
	static std::unique_ptr<DuckDBPyExpression> InternalConjunction(ExpressionType type, const DuckDBPyExpression &arg,
	                                                               const DuckDBPyExpression &other);
	static std::unique_ptr<DuckDBPyExpression> InternalConstantExpression(Value value);
	static std::unique_ptr<DuckDBPyExpression>
	BinaryOperator(const string &function_name, const DuckDBPyExpression &arg_one, const DuckDBPyExpression &arg_two);
	static std::unique_ptr<DuckDBPyExpression> ComparisonExpression(ExpressionType type, const DuckDBPyExpression &left,
	                                                                const DuckDBPyExpression &right);
	static std::unique_ptr<DuckDBPyExpression> InternalWhen(unique_ptr<duckdb::CaseExpression> expr,
	                                                        const DuckDBPyExpression &condition,
	                                                        const DuckDBPyExpression &value);
	void AssertCaseExpression() const;

private:
	unique_ptr<ParsedExpression> expression;

public:
	OrderByNullType null_order = OrderByNullType::ORDER_DEFAULT;
	OrderType order_type = OrderType::ORDER_DEFAULT;
};

} // namespace duckdb
