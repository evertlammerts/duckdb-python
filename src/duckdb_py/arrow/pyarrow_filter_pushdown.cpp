#include "duckdb_python/arrow/pyarrow_filter_pushdown.hpp"

#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"

namespace duckdb {

string ConvertTimestampUnit(ArrowDateTimeType unit) {
	switch (unit) {
	case ArrowDateTimeType::MICROSECONDS:
		return "us";
	case ArrowDateTimeType::MILLISECONDS:
		return "ms";
	case ArrowDateTimeType::NANOSECONDS:
		return "ns";
	case ArrowDateTimeType::SECONDS:
		return "s";
	default:
		throw NotImplementedException("DatetimeType not recognized in ConvertTimestampUnit: %d",
		                              static_cast<int>(unit));
	}
}

int64_t ConvertTimestampTZValue(int64_t base_value, ArrowDateTimeType datetime_type) {
	auto input = timestamp_t(base_value);
	if (!Value::IsFinite(input)) {
		return base_value;
	}

	switch (datetime_type) {
	case ArrowDateTimeType::MICROSECONDS:
		return Timestamp::GetEpochMicroSeconds(input);
	case ArrowDateTimeType::MILLISECONDS:
		return Timestamp::GetEpochMs(input);
	case ArrowDateTimeType::NANOSECONDS:
		return Timestamp::GetEpochNanoSeconds(input);
	case ArrowDateTimeType::SECONDS:
		return Timestamp::GetEpochSeconds(input);
	default:
		throw NotImplementedException("DatetimeType not recognized in ConvertTimestampTZValue");
	}
}

py::object GetScalar(const Value &constant, const string &timezone_config, const ArrowType &type) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto scalar = import_cache.pyarrow.scalar();
	py::handle dataset_scalar = import_cache.pyarrow.dataset().attr("scalar");

	switch (constant.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return dataset_scalar(constant.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return dataset_scalar(constant.GetValue<int8_t>());
	case LogicalTypeId::SMALLINT:
		return dataset_scalar(constant.GetValue<int16_t>());
	case LogicalTypeId::INTEGER:
		return dataset_scalar(constant.GetValue<int32_t>());
	case LogicalTypeId::BIGINT:
		return dataset_scalar(constant.GetValue<int64_t>());
	case LogicalTypeId::DATE: {
		py::handle date_type = import_cache.pyarrow.date32();
		return dataset_scalar(scalar(constant.GetValue<int32_t>(), date_type()));
	}
	case LogicalTypeId::TIME: {
		py::handle date_type = import_cache.pyarrow.time64();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIMESTAMP: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIMESTAMP_MS: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ms")));
	}
	case LogicalTypeId::TIMESTAMP_NS: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ns")));
	}
	case LogicalTypeId::TIMESTAMP_SEC: {
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("s")));
	}
	case LogicalTypeId::TIMESTAMP_TZ: {
		auto &datetime_info = type.GetTypeInfo<ArrowDateTimeInfo>();
		auto base_value = constant.GetValue<int64_t>();
		auto arrow_datetime_type = datetime_info.GetDateTimeType();
		auto time_unit_string = ConvertTimestampUnit(arrow_datetime_type);
		auto converted_value = ConvertTimestampTZValue(base_value, arrow_datetime_type);
		py::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(converted_value, date_type(time_unit_string, py::arg("tz") = timezone_config)));
	}
	case LogicalTypeId::UTINYINT: {
		py::handle integer_type = import_cache.pyarrow.uint8();
		return dataset_scalar(scalar(constant.GetValue<uint8_t>(), integer_type()));
	}
	case LogicalTypeId::USMALLINT: {
		py::handle integer_type = import_cache.pyarrow.uint16();
		return dataset_scalar(scalar(constant.GetValue<uint16_t>(), integer_type()));
	}
	case LogicalTypeId::UINTEGER: {
		py::handle integer_type = import_cache.pyarrow.uint32();
		return dataset_scalar(scalar(constant.GetValue<uint32_t>(), integer_type()));
	}
	case LogicalTypeId::UBIGINT: {
		py::handle integer_type = import_cache.pyarrow.uint64();
		return dataset_scalar(scalar(constant.GetValue<uint64_t>(), integer_type()));
	}
	case LogicalTypeId::FLOAT:
		return dataset_scalar(constant.GetValue<float>());
	case LogicalTypeId::DOUBLE:
		return dataset_scalar(constant.GetValue<double>());
	case LogicalTypeId::VARCHAR:
		return dataset_scalar(constant.ToString());
	case LogicalTypeId::BLOB: {
		if (type.GetTypeInfo<ArrowStringInfo>().GetSizeType() == ArrowVariableSizeType::VIEW) {
			py::handle binary_view_type = import_cache.pyarrow.binary_view();
			return dataset_scalar(scalar(py::bytes(constant.GetValueUnsafe<string>()), binary_view_type()));
		}
		return dataset_scalar(py::bytes(constant.GetValueUnsafe<string>()));
	}
	case LogicalTypeId::DECIMAL: {
		py::handle decimal_type;
		auto &datetime_info = type.GetTypeInfo<ArrowDecimalInfo>();
		auto bit_width = datetime_info.GetBitWidth();
		switch (bit_width) {
		case DecimalBitWidth::DECIMAL_32:
			decimal_type = import_cache.pyarrow.decimal32();
			break;
		case DecimalBitWidth::DECIMAL_64:
			decimal_type = import_cache.pyarrow.decimal64();
			break;
		case DecimalBitWidth::DECIMAL_128:
			decimal_type = import_cache.pyarrow.decimal128();
			break;
		default:
			throw NotImplementedException("Unsupported precision for Arrow Decimal Type.");
		}

		uint8_t width;
		uint8_t scale;
		constant.type().GetDecimalProperties(width, scale);
		// pyarrow only allows 'decimal.Decimal' to be used to construct decimal scalars such as 0.05
		auto val = import_cache.decimal.Decimal()(constant.ToString());
		return dataset_scalar(
		    scalar(std::move(val), decimal_type(py::arg("precision") = width, py::arg("scale") = scale)));
	}
	default:
		throw NotImplementedException("Unimplemented type \"%s\" for Arrow Filter Pushdown",
		                              constant.type().ToString());
	}
}

static py::list TransformInList(const InFilter &in) {
	py::list res;
	ClientProperties default_properties;
	for (auto &val : in.values) {
		res.append(PythonObject::FromValue(val, val.type(), default_properties));
	}
	return res;
}

struct ResolvedColumn {
	vector<string> path;
	reference<const ArrowType> leaf_type;
};

// Resolves a column-side expression to (full path, leaf ArrowType). Handles bare BoundReferenceExpression and (nested)
// struct_extract chains. Throws NotImplementedException for any other input.
ResolvedColumn ResolveColumn(const Expression &expr, const vector<string> &root_path, const ArrowType &root_type) {
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
		throw InternalException("Cannot push down arrow scan filter on column-side function: %s\n",
		                        ExpressionTypeToString(expr.GetExpressionType()));
	}
	// Recurse innermost-first so names accumulate root -> leaf.
	auto inner = ResolveColumn(*func.children[0], root_path, root_type);
	inner.path.push_back(StructType::GetChildName(func.children[0]->GetReturnType(), child_idx));
	inner.leaf_type = inner.leaf_type.get().GetTypeInfo<ArrowStructInfo>().GetChild(child_idx);
	return inner;
}

py::object TransformExpressionRecursive(const Expression &expression, const vector<string> &column_ref,
                                        const string &timezone_config, const ArrowType &type) {
	fprintf(stderr, "!!!! EXPRESSION CLASS = %s\n", ExpressionClassToString(expression.GetExpressionClass()).c_str());
	fprintf(stderr, "!!!! EXPRESSION TYPE = %s\n", ExpressionTypeToString(expression.GetExpressionType()).c_str());

	auto &import_cache = *DuckDBPyConnection::ImportCache();
	const py::object field = import_cache.pyarrow.dataset().attr("field");
	auto expression_type = expression.GetExpressionType();

	if (expression.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		auto &bound_function_expression = expression.Cast<BoundFunctionExpression>();

		// ExpressionType::COMPARE_*
		if (BoundComparisonExpression::IsComparison(expression_type)) {
			// Comparisons have a column-side and a constant-side. We first resolve them.
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
				fprintf(stderr, "!!!! NOT A CONSTANT COMPARISON\n");
				throw NotImplementedException("Can only push down constant comparisons.");
			}

			// Get the column-side path and arrow type and the matching reference field and constant value
			auto [path, leaf_type] = ResolveColumn(*column_side, column_ref, type);
			const auto reference_field = field(py::tuple(py::cast(path)));
			const auto constant_py_value = GetScalar(constant_side->value, timezone_config, leaf_type);

			// And finally, apply the comparison:
			// 1. Special handling for NaN comparisons (to explicitly violate IEEE-754)
			auto is_nan = false;
			if (constant_side->value.type() == LogicalTypeId::FLOAT) {
				is_nan = Value::IsNan(constant_side->value.GetValue<float>());
			} else if (constant_side->value.type() == LogicalTypeId::DOUBLE) {
				is_nan = Value::IsNan(constant_side->value.GetValue<double>());
			}
			if (is_nan) {
				switch (expression_type) {
				case ExpressionType::COMPARE_EQUAL:
				case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
					return reference_field.attr("is_nan")();
				case ExpressionType::COMPARE_LESSTHAN:
				case ExpressionType::COMPARE_NOTEQUAL:
					return reference_field.attr("is_nan")().attr("__invert__")();
				case ExpressionType::COMPARE_GREATERTHAN:
					// Nothing is greater than NaN
					return import_cache.pyarrow.dataset().attr("scalar")(false);
				case ExpressionType::COMPARE_LESSTHANOREQUALTO:
					// Everything is less than or equal to NaN
					return import_cache.pyarrow.dataset().attr("scalar")(true);
				default:
					throw NotImplementedException("Unsupported comparison type (%s) for NaN values",
					                              ExpressionTypeToString(expression_type));
				}
			}
			// 2. Regular handling for non-NaN comparisons
			switch (expression_type) {
			case ExpressionType::COMPARE_EQUAL:
				return reference_field.attr("__eq__")(constant_py_value);
			case ExpressionType::COMPARE_NOTEQUAL:
				return reference_field.attr("__ne__")(constant_py_value);
			case ExpressionType::COMPARE_LESSTHAN:
				return reference_field.attr("__lt__")(constant_py_value);
			case ExpressionType::COMPARE_GREATERTHAN:
				return reference_field.attr("__gt__")(constant_py_value);
			case ExpressionType::COMPARE_LESSTHANOREQUALTO:
				return reference_field.attr("__le__")(constant_py_value);
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				return reference_field.attr("__ge__")(constant_py_value);
			default:
				throw NotImplementedException("Comparison Type %s can't be an Arrow Scan Pushdown Filter",
				                              ExpressionTypeToString(expression_type));
			}
		}
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		// ExpressionType::OPERATOR_IS_NULL
		if (expression_type == ExpressionType::OPERATOR_IS_NULL) {
			auto &column_side = expression.Cast<BoundOperatorExpression>().children[0];
			auto [path, leaf_type] = ResolveColumn(*column_side, column_ref, type);
			const auto reference_field = field(py::tuple(py::cast(path)));
			return reference_field.attr("is_null")();
		}
		// ExpressionType::OPERATOR_IS_NOT_NULL
		if (expression_type == ExpressionType::OPERATOR_IS_NOT_NULL) {
			auto &column_side = expression.Cast<BoundOperatorExpression>().children[0];
			auto [path, leaf_type] = ResolveColumn(*column_side, column_ref, type);
			const auto reference_field = field(py::tuple(py::cast(path)));
			return reference_field.attr("is_valid")();
		}
		// ExpressionType::COMPARE_IN
		if (expression_type == ExpressionType::COMPARE_IN) {
			auto &op_expr = expression.Cast<BoundOperatorExpression>();
			auto &column_side = op_expr.children[0];
			auto [path, leaf_type] = ResolveColumn(*column_side, column_ref, type);
			const auto duck_type = leaf_type.get().GetDuckType();
			const auto reference_field = field(py::tuple(py::cast(path)));
			py::list in_list;
			for (idx_t i = 1; i < op_expr.children.size(); i++) {
				ClientProperties default_properties;
				auto &const_expr = op_expr.children[i]->Cast<BoundConstantExpression>();
				in_list.append(PythonObject::FromValue(const_expr.value, duck_type, default_properties));
			}
			return reference_field.attr("isin")(std::move(in_list));
		}
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
		// ExpressionType::CONJUNCTION_OR || ExpressionType::CONJUNCTION_AND
		if (expression_type == ExpressionType::CONJUNCTION_OR || expression_type == ExpressionType::CONJUNCTION_AND) {
			const auto pyarrow_function = expression_type == ExpressionType::CONJUNCTION_OR ? "__or__" : "__and__";
			auto &or_expr = expression.Cast<BoundConjunctionExpression>();
			py::object pyarrow_expression = py::none();
			for (idx_t i = 0; i < or_expr.children.size(); i++) {
				const auto &child_expr = or_expr.children[i];
				py::object pyarrow_child_expression =
				    TransformExpressionRecursive(*child_expr, column_ref, timezone_config, type);
				if (pyarrow_child_expression.is(py::none())) {
					continue;
				}
				if (pyarrow_expression.is(py::none())) {
					pyarrow_expression = std::move(pyarrow_child_expression);
				} else {
					pyarrow_expression = pyarrow_expression.attr(pyarrow_function)(pyarrow_child_expression);
				}
			}
			return pyarrow_expression;
		}
	}
	fprintf(stderr, "!!!! EXPRESSION NOT PUSHED DOWN!\n");
	throw NotImplementedException("Pushdown Filter Type %s is not currently supported in PyArrow Scans",
	                              ExpressionClassToString(expression.GetExpressionClass()));
}

py::object TransformFilterRecursive(TableFilter &filter, vector<string> column_ref, const string &timezone_config,
                                    const ArrowType &type) {
	fprintf(stderr, "!!!! FILTER TYPE = %s\n", EnumUtil::ToString(filter.filter_type).c_str());
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	py::object field = import_cache.pyarrow.dataset().attr("field");
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto constant_field = field(py::tuple(py::cast(column_ref)));
		auto constant_value = GetScalar(constant_filter.constant, timezone_config, type);

		bool is_nan = false;
		auto &constant = constant_filter.constant;
		auto &constant_type = constant.type();
		if (constant_type.id() == LogicalTypeId::FLOAT) {
			is_nan = Value::IsNan(constant.GetValue<float>());
		} else if (constant_type.id() == LogicalTypeId::DOUBLE) {
			is_nan = Value::IsNan(constant.GetValue<double>());
		}

		// Special handling for NaN comparisons (to explicitly violate IEEE-754)
		if (is_nan) {
			switch (constant_filter.comparison_type) {
			case ExpressionType::COMPARE_EQUAL:
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				return constant_field.attr("is_nan")();
			case ExpressionType::COMPARE_LESSTHAN:
			case ExpressionType::COMPARE_NOTEQUAL:
				return constant_field.attr("is_nan")().attr("__invert__")();
			case ExpressionType::COMPARE_GREATERTHAN:
				// Nothing is greater than NaN
				return import_cache.pyarrow.dataset().attr("scalar")(false);
			case ExpressionType::COMPARE_LESSTHANOREQUALTO:
				// Everything is less than or equal to NaN
				return import_cache.pyarrow.dataset().attr("scalar")(true);
			default:
				throw NotImplementedException("Unsupported comparison type (%s) for NaN values",
				                              EnumUtil::ToString(constant_filter.comparison_type));
			}
		}

		switch (constant_filter.comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			return constant_field.attr("__eq__")(constant_value);
		case ExpressionType::COMPARE_LESSTHAN:
			return constant_field.attr("__lt__")(constant_value);
		case ExpressionType::COMPARE_GREATERTHAN:
			return constant_field.attr("__gt__")(constant_value);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return constant_field.attr("__le__")(constant_value);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return constant_field.attr("__ge__")(constant_value);
		case ExpressionType::COMPARE_NOTEQUAL:
			return constant_field.attr("__ne__")(constant_value);
		default:
			throw NotImplementedException("Comparison Type %s can't be an Arrow Scan Pushdown Filter",
			                              EnumUtil::ToString(constant_filter.comparison_type));
		}
	}
	//! We do not pushdown is null yet
	case TableFilterType::IS_NULL: {
		auto constant_field = field(py::tuple(py::cast(column_ref)));
		return constant_field.attr("is_null")();
	}
	case TableFilterType::IS_NOT_NULL: {
		auto constant_field = field(py::tuple(py::cast(column_ref)));
		return constant_field.attr("is_valid")();
	}
	//! We do not pushdown or conjunctions yet
	case TableFilterType::CONJUNCTION_OR: {
		auto &or_filter = filter.Cast<ConjunctionOrFilter>();
		py::object expression = py::none();
		for (idx_t i = 0; i < or_filter.child_filters.size(); i++) {
			auto &child_filter = *or_filter.child_filters[i];
			py::object child_expression = TransformFilterRecursive(child_filter, column_ref, timezone_config, type);
			if (child_expression.is(py::none())) {
				continue;
			}
			if (expression.is(py::none())) {
				expression = std::move(child_expression);
			} else {
				expression = expression.attr("__or__")(child_expression);
			}
		}
		return expression;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		py::object expression = py::none();
		for (idx_t i = 0; i < and_filter.child_filters.size(); i++) {
			auto &child_filter = *and_filter.child_filters[i];
			py::object child_expression = TransformFilterRecursive(child_filter, column_ref, timezone_config, type);
			if (child_expression.is(py::none())) {
				continue;
			}
			if (expression.is(py::none())) {
				expression = std::move(child_expression);
			} else {
				expression = expression.attr("__and__")(child_expression);
			}
		}
		return expression;
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		auto &child_name = struct_filter.child_name;
		auto &struct_type_info = type.GetTypeInfo<ArrowStructInfo>();
		auto &struct_child_type = struct_type_info.GetChild(struct_filter.child_idx);

		column_ref.push_back(child_name);
		auto child_expr = TransformFilterRecursive(*struct_filter.child_filter, std::move(column_ref), timezone_config,
		                                           struct_child_type);
		return child_expr;
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (!optional_filter.child_filter) {
			return py::none();
		}
		try {
			return TransformFilterRecursive(*optional_filter.child_filter, column_ref, timezone_config, type);
		} catch (const NotImplementedException &) {
			return py::none();
		}
	}
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		auto constant_field = field(py::tuple(py::cast(column_ref)));
		auto in_list = TransformInList(in_filter);
		return constant_field.attr("isin")(std::move(in_list));
	}
	case TableFilterType::DYNAMIC_FILTER: {
		//! Ignore dynamic filters for now, not necessary for correctness
		return py::none();
	}
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expression_filter = filter.Cast<ExpressionFilter>();
		return TransformExpressionRecursive(*expression_filter.expr, column_ref, timezone_config, type);
	}
	default:
		throw NotImplementedException("Pushdown Filter Type %s is not currently supported in PyArrow Scans",
		                              EnumUtil::ToString(filter.filter_type));
	}
}

py::object PyArrowFilterPushdown::TransformFilter(TableFilterSet &filter_collection,
                                                  unordered_map<idx_t, string> &columns,
                                                  unordered_map<idx_t, idx_t> filter_to_col,
                                                  const ClientProperties &config, const ArrowTableSchema &arrow_table) {

	py::object expression = py::none();
	for (auto &entry : filter_collection) {
		auto column_idx = entry.GetIndex();
		auto &column_name = columns[column_idx];

		vector<string> column_ref;
		column_ref.push_back(column_name);

		D_ASSERT(columns.find(column_idx) != columns.end());

		auto &arrow_type = arrow_table.GetColumns().at(filter_to_col.at(column_idx));
		py::object child_expression =
		    TransformFilterRecursive(entry.Filter(), column_ref, config.time_zone, *arrow_type);
		if (child_expression.is(py::none())) {
			continue;
		} else if (expression.is(py::none())) {
			expression = std::move(child_expression);
		} else {
			expression = expression.attr("__and__")(child_expression);
		}
	}
	return expression;
}

} // namespace duckdb
