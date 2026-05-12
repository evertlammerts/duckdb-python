#include "duckdb_python/arrow/filter_pushdown_visitor.hpp"

#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"

namespace duckdb {

namespace {

bool ValueIsNan(const Value &value) {
	if (value.type().id() == LogicalTypeId::FLOAT) {
		return Value::IsNan(value.GetValue<float>());
	}
	if (value.type().id() == LogicalTypeId::DOUBLE) {
		return Value::IsNan(value.GetValue<double>());
	}
	return false;
}

// ResolveColumn walks a column-side expression to extract the (full path, leaf
// ArrowType) pair. Accepts a bare BoundReferenceExpression and (nested)
// `struct_extract` chains. Anything else throws NotImplementedException —
// that gives the OPTIONAL_FILTER catch point a chance to swallow it.
struct ResolvedColumn {
	vector<string> path;
	const ArrowType *leaf_type;
};

ResolvedColumn ResolveColumn(const Expression &expr, const vector<string> &root_path, const ArrowType *root_type) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		return {root_path, root_type};
	}
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		throw NotImplementedException("Cannot push down arrow scan filter on column-side expression: %s",
		                              ExpressionClassToString(expr.GetExpressionClass()));
	}
	auto &func = expr.Cast<BoundFunctionExpression>();
	idx_t child_idx;
	if (!TryGetStructExtractChildIndex(func, child_idx)) {
		throw NotImplementedException("Cannot push down arrow scan filter on column-side function: %s",
		                              ExpressionTypeToString(expr.GetExpressionType()));
	}
	// Recurse innermost-first so names accumulate root → leaf.
	auto inner = ResolveColumn(*func.children[0], root_path, root_type);
	inner.path.push_back(StructType::GetChildName(func.children[0]->GetReturnType(), child_idx));
	if (inner.leaf_type) {
		inner.leaf_type = &inner.leaf_type->GetTypeInfo<ArrowStructInfo>().GetChild(child_idx);
	}
	return inner;
}

py::object EmitCompare(FilterBackend &backend, ExpressionType op, py::object col, const Value &constant,
                       const ArrowType *arrow_type, const string &timezone_config) {
	if (ValueIsNan(constant)) {
		return backend.NaNCompare(op, std::move(col));
	}
	auto scalar = backend.MakeScalar(constant, arrow_type, timezone_config);
	return backend.Compare(op, std::move(col), std::move(scalar));
}

} // anonymous namespace

py::object TransformExpression(const Expression &expression, const vector<string> &column_path, FilterBackend &backend,
                               const ArrowType *arrow_type, const string &timezone_config) {
	auto expression_class = expression.GetExpressionClass();
	auto expression_type = expression.GetExpressionType();

	if (expression_class == ExpressionClass::BOUND_FUNCTION) {
		auto &bound_function_expression = expression.Cast<BoundFunctionExpression>();
		if (BoundComparisonExpression::IsComparison(expression_type)) {
			auto &left = BoundComparisonExpression::Left(bound_function_expression);
			auto &right = BoundComparisonExpression::Right(bound_function_expression);

			optional_ptr<const Expression> column_side;
			optional_ptr<const BoundConstantExpression> constant_side;

			if (right.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
				column_side = &left;
				constant_side = &right.Cast<BoundConstantExpression>();
			} else if (left.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
				column_side = &right;
				constant_side = &left.Cast<BoundConstantExpression>();
				expression_type = FlipComparisonExpression(expression_type);
			} else {
				throw NotImplementedException("Can only push down constant comparisons.");
			}

			auto resolved = ResolveColumn(*column_side, column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return EmitCompare(backend, expression_type, std::move(col), constant_side->value, resolved.leaf_type,
			                   timezone_config);
		}
	}

	if (expression_class == ExpressionClass::BOUND_OPERATOR) {
		auto &op_expr = expression.Cast<BoundOperatorExpression>();
		if (expression_type == ExpressionType::OPERATOR_IS_NULL) {
			auto resolved = ResolveColumn(*op_expr.children[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return backend.IsNull(std::move(col));
		}
		if (expression_type == ExpressionType::OPERATOR_IS_NOT_NULL) {
			auto resolved = ResolveColumn(*op_expr.children[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			return backend.IsNotNull(std::move(col));
		}
		if (expression_type == ExpressionType::COMPARE_IN) {
			auto resolved = ResolveColumn(*op_expr.children[0], column_path, arrow_type);
			auto col = backend.MakeColumnRef(resolved.path);
			vector<Value> values;
			for (idx_t i = 1; i < op_expr.children.size(); i++) {
				auto &const_expr = op_expr.children[i]->Cast<BoundConstantExpression>();
				values.push_back(const_expr.value);
			}
			auto col_type = op_expr.children[0]->GetReturnType();
			return backend.IsIn(std::move(col), values, col_type, timezone_config);
		}
	}

	if (expression_class == ExpressionClass::BOUND_CONJUNCTION) {
		if (expression_type == ExpressionType::CONJUNCTION_OR || expression_type == ExpressionType::CONJUNCTION_AND) {
			auto &conj_expr = expression.Cast<BoundConjunctionExpression>();
			py::object result = py::none();
			for (idx_t i = 0; i < conj_expr.children.size(); i++) {
				py::object child_expression =
				    TransformExpression(*conj_expr.children[i], column_path, backend, arrow_type, timezone_config);
				if (child_expression.is(py::none())) {
					continue;
				}
				if (result.is(py::none())) {
					result = std::move(child_expression);
				} else if (expression_type == ExpressionType::CONJUNCTION_AND) {
					result = backend.And(std::move(result), std::move(child_expression));
				} else {
					result = backend.Or(std::move(result), std::move(child_expression));
				}
			}
			return result;
		}
	}

	throw NotImplementedException("Pushdown Filter Type %s is not currently supported in arrow scans",
	                              ExpressionClassToString(expression_class));
}

py::object TransformFilter(const TableFilter &filter, vector<string> column_path, FilterBackend &backend,
                           const ArrowType *arrow_type, const string &timezone_config) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto col = backend.MakeColumnRef(column_path);
		return EmitCompare(backend, constant_filter.comparison_type, std::move(col), constant_filter.constant,
		                   arrow_type, timezone_config);
	}
	case TableFilterType::IS_NULL: {
		auto col = backend.MakeColumnRef(column_path);
		return backend.IsNull(std::move(col));
	}
	case TableFilterType::IS_NOT_NULL: {
		auto col = backend.MakeColumnRef(column_path);
		return backend.IsNotNull(std::move(col));
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		py::object result = py::none();
		for (idx_t i = 0; i < and_filter.child_filters.size(); i++) {
			py::object child_expression =
			    TransformFilter(*and_filter.child_filters[i], column_path, backend, arrow_type, timezone_config);
			if (child_expression.is(py::none())) {
				continue;
			}
			if (result.is(py::none())) {
				result = std::move(child_expression);
			} else {
				result = backend.And(std::move(result), std::move(child_expression));
			}
		}
		return result;
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &or_filter = filter.Cast<ConjunctionOrFilter>();
		py::object result = py::none();
		for (idx_t i = 0; i < or_filter.child_filters.size(); i++) {
			py::object child_expression =
			    TransformFilter(*or_filter.child_filters[i], column_path, backend, arrow_type, timezone_config);
			if (child_expression.is(py::none())) {
				continue;
			}
			if (result.is(py::none())) {
				result = std::move(child_expression);
			} else {
				result = backend.Or(std::move(result), std::move(child_expression));
			}
		}
		return result;
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		column_path.push_back(struct_filter.child_name);
		const ArrowType *child_type = nullptr;
		if (arrow_type) {
			child_type = &arrow_type->GetTypeInfo<ArrowStructInfo>().GetChild(struct_filter.child_idx);
		}
		return TransformFilter(*struct_filter.child_filter, std::move(column_path), backend, child_type,
		                       timezone_config);
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (!optional_filter.child_filter) {
			return py::none();
		}
		try {
			return TransformFilter(*optional_filter.child_filter, column_path, backend, arrow_type, timezone_config);
		} catch (const NotImplementedException &) {
			return py::none();
		}
	}
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		auto col = backend.MakeColumnRef(column_path);
		// The column's logical type for IN comes from the values themselves
		// (they share the comparison type). Empty IN lists are not produced
		// by the optimizer so we can safely index values[0].
		LogicalType col_logical_type =
		    in_filter.values.empty() ? LogicalType::SQLNULL : in_filter.values.front().type();
		return backend.IsIn(std::move(col), in_filter.values, col_logical_type, timezone_config);
	}
	case TableFilterType::DYNAMIC_FILTER:
		return py::none();
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expression_filter = filter.Cast<ExpressionFilter>();
		return TransformExpression(*expression_filter.expr, column_path, backend, arrow_type, timezone_config);
	}
	default:
		throw NotImplementedException("Pushdown Filter Type %s is not currently supported in arrow scans",
		                              EnumUtil::ToString(filter.filter_type));
	}
}

} // namespace duckdb
