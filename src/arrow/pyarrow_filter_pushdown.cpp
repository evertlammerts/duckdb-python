#include "duckdb_python/arrow/pyarrow_filter_pushdown.hpp"

#include "duckdb_python/arrow/filter_pushdown_visitor.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb/function/table/arrow.hpp"
#include <cmath>

namespace duckdb {

namespace {

string ColumnTimeZone(const ArrowSchema *schema) {
	const string format = schema && schema->format ? schema->format : "";
	const auto colon = format.find(':');
	if (format.rfind("ts", 0) != 0 || colon == string::npos || colon + 1 == format.size()) {
		// A TIMESTAMP_TZ constant only meets a zoned timestamp column, so any other schema node is one the client
		// cannot read.
		throw NotImplementedException("Cannot read the time zone of column \"%s\" from the Arrow schema",
		                              schema && schema->name ? schema->name : "");
	}
	return format.substr(colon + 1);
}

// Build a pyarrow.dataset scalar matching the given DuckDB Value and (optionally) ArrowType.
// The ArrowType is needed for decimal precision / blob-view disambiguation; the
// DuckDB Value alone is not sufficient.
nb::object MakePyArrowScalar(const Value &constant, const string &timezone_config, const ArrowType *arrow_type,
                             const ArrowSchema *arrow_schema) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto scalar = import_cache.pyarrow.scalar();
	nb::handle dataset_scalar = import_cache.pyarrow.dataset().attr("scalar");

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
		nb::handle date_type = import_cache.pyarrow.date32();
		return dataset_scalar(scalar(constant.GetValue<int32_t>(), date_type()));
	}
	case LogicalTypeId::TIME: {
		nb::handle date_type = import_cache.pyarrow.time64();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIME_NS: {
		// Polars TIME columns round-trip through arrow as time64("ns").
		// `Value::GetValue<int64_t>()` has a hand-rolled fast-path switch for TIME but not
		// TIME_NS — it falls through to GetValueInternal, which then tries
		// Cast::Operation<dtime_ns_t, int64_t> for which no specialization exists, and
		// throws "Unimplemented type for cast (INT64 -> INT64)". Use the type-strong
		// GetValueUnsafe<dtime_ns_t>() which reads `value_.time_ns` from the union
		// directly. dtime_ns_t.value holds nanoseconds (see arrow_conversion.cpp:432).
		nb::handle date_type = import_cache.pyarrow.time64();
		return dataset_scalar(scalar(constant.GetValueUnsafe<dtime_ns_t>().value, date_type("ns")));
	}
	case LogicalTypeId::TIMESTAMP: {
		nb::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us")));
	}
	case LogicalTypeId::TIMESTAMP_MS: {
		nb::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ms")));
	}
	case LogicalTypeId::TIMESTAMP_NS: {
		nb::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("ns")));
	}
	case LogicalTypeId::TIMESTAMP_SEC: {
		nb::handle date_type = import_cache.pyarrow.timestamp();
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("s")));
	}
	case LogicalTypeId::TIMESTAMP_TZ: {
		// The microsecond scalar keeps its UTC instant under the column's own zone, and pyarrow has no kernel for a
		// comparison that differs in both unit and zone.
		nb::handle date_type = import_cache.pyarrow.timestamp();
		auto zone = ColumnTimeZone(arrow_schema);
		return dataset_scalar(scalar(constant.GetValue<int64_t>(), date_type("us", nb::arg("tz") = zone)));
	}
	case LogicalTypeId::TIMESTAMP_TZ_NS: {
		nb::handle date_type = import_cache.pyarrow.timestamp();
		auto converted_value = Timestamp::GetEpochNanoSeconds(timestamp_t(constant.GetValue<int64_t>()));
		return dataset_scalar(scalar(converted_value, date_type("ns", nb::arg("tz") = timezone_config)));
	}
	case LogicalTypeId::UTINYINT: {
		nb::handle integer_type = import_cache.pyarrow.uint8();
		return dataset_scalar(scalar(constant.GetValue<uint8_t>(), integer_type()));
	}
	case LogicalTypeId::USMALLINT: {
		nb::handle integer_type = import_cache.pyarrow.uint16();
		return dataset_scalar(scalar(constant.GetValue<uint16_t>(), integer_type()));
	}
	case LogicalTypeId::UINTEGER: {
		nb::handle integer_type = import_cache.pyarrow.uint32();
		return dataset_scalar(scalar(constant.GetValue<uint32_t>(), integer_type()));
	}
	case LogicalTypeId::UBIGINT: {
		nb::handle integer_type = import_cache.pyarrow.uint64();
		return dataset_scalar(scalar(constant.GetValue<uint64_t>(), integer_type()));
	}
	case LogicalTypeId::FLOAT:
		return dataset_scalar(constant.GetValue<float>());
	case LogicalTypeId::DOUBLE:
		return dataset_scalar(constant.GetValue<double>());
	case LogicalTypeId::VARCHAR:
		return dataset_scalar(constant.ToString());
	case LogicalTypeId::BLOB: {
		if (arrow_type && arrow_type->GetTypeInfo<ArrowStringInfo>().GetSizeType() == ArrowVariableSizeType::VIEW) {
			nb::handle binary_view_type = import_cache.pyarrow.binary_view();
			{
				auto blob = constant.GetValueUnsafe<string>();
				return dataset_scalar(scalar(nb::bytes(blob.data(), blob.size()), binary_view_type()));
			}
		}
		{
			auto blob = constant.GetValueUnsafe<string>();
			return dataset_scalar(nb::bytes(blob.data(), blob.size()));
		}
	}
	case LogicalTypeId::DECIMAL: {
		if (!arrow_type) {
			throw NotImplementedException("Cannot push down DECIMAL filter without an arrow type");
		}
		nb::handle decimal_type;
		auto &decimal_info = arrow_type->GetTypeInfo<ArrowDecimalInfo>();
		auto bit_width = decimal_info.GetBitWidth();
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
		auto val = import_cache.decimal.Decimal()(constant.ToString());
		return dataset_scalar(
		    scalar(std::move(val), decimal_type(nb::arg("precision") = width, nb::arg("scale") = scale)));
	}
	default:
		throw NotImplementedException("Unimplemented type \"%s\" for Arrow Filter Pushdown",
		                              constant.type().ToString());
	}
}

struct PyArrowBackend : public FilterBackend {
	explicit PyArrowBackend(const ClientProperties &client_properties_p) : client_properties(client_properties_p) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		field_factory = import_cache.pyarrow.dataset().attr("field");
		dataset_scalar = import_cache.pyarrow.dataset().attr("scalar");
	}

	nb::object MakeColumnRef(const vector<Identifier> &path) override {
		vector<string> str_path;
		std::transform(path.begin(), path.end(), std::back_inserter(str_path),
		               [](const Identifier &segment) { return segment.GetIdentifierName(); });
		return field_factory(nb::tuple(nb::cast(str_path)));
	}

	nb::object MakeScalar(const Value &v, const ArrowType *arrow_type, const ArrowSchema *arrow_schema,
	                      const string &timezone_config) override {
		return MakePyArrowScalar(v, timezone_config, arrow_type, arrow_schema);
	}

	nb::object Compare(ExpressionType op, nb::object col, nb::object scalar) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
			return col.attr("__eq__")(scalar);
		case ExpressionType::COMPARE_NOTEQUAL:
			return col.attr("__ne__")(scalar);
		case ExpressionType::COMPARE_LESSTHAN:
			return col.attr("__lt__")(scalar);
		case ExpressionType::COMPARE_GREATERTHAN:
			return col.attr("__gt__")(scalar);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return col.attr("__le__")(scalar);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return col.attr("__ge__")(scalar);
		default:
			throw NotImplementedException("Comparison Type %s can't be an Arrow Scan Pushdown Filter",
			                              ExpressionTypeToString(op));
		}
	}

	nb::object NaNCompare(ExpressionType op, nb::object col) override {
		switch (op) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return IsNaN(std::move(col));
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_NOTEQUAL:
			// Every value but NaN equals itself, and a NULL row stays NULL.
			return col.attr("__eq__")(col);
		case ExpressionType::COMPARE_GREATERTHAN:
			// Nothing is greater than NaN.
			return dataset_scalar(false);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			// NaN sorts above every other value, so only NULL rows fail.
			return IsNotNull(std::move(col));
		default:
			throw NotImplementedException("Unsupported comparison type (%s) for NaN values",
			                              ExpressionTypeToString(op));
		}
	}

	nb::object IsNaN(nb::object col) override {
		// NaN is the only value unequal to itself, and pyarrow has no is_nan kernel for a dictionary column.
		return col.attr("__ne__")(col);
	}

	nb::object IsNull(nb::object col) override {
		return col.attr("is_null")();
	}

	nb::object IsNotNull(nb::object col) override {
		return col.attr("is_valid")();
	}

	nb::object IsIn(nb::object col, const vector<Value> &values, const LogicalType &col_logical_type,
	                const string &timezone_config) override {
		(void)timezone_config;
		const auto col_type_id = col_logical_type.id();
		const bool is_float = col_type_id == LogicalTypeId::FLOAT || col_type_id == LogicalTypeId::DOUBLE;

		// isin compares float bits (missing -0.0 and NaN payloads), and is_nan has no dictionary kernel.
		if (is_float) {
			bool has_zero = false;
			nb::list non_zero_values;

			for (auto &val : values) {
				if (val.IsNull()) {
					non_zero_values.append(PythonObject::FromValue(val, val.type(), client_properties));
					continue;
				}

				double dbl = col_type_id == LogicalTypeId::FLOAT ? static_cast<double>(val.GetValue<float>())
				                                                 : val.GetValue<double>();

				if (std::isnan(dbl)) {
					throw NotImplementedException(
					    "Arrow filter pushdown: IN list with NaN over float/double is not pushed");
				}

				if (dbl == 0.0) {
					has_zero = true;
				} else {
					non_zero_values.append(PythonObject::FromValue(val, val.type(), client_properties));
				}
			}

			if (has_zero) {
				auto zero_expr = Compare(ExpressionType::COMPARE_EQUAL, col, dataset_scalar(0.0));
				if (nb::len(non_zero_values) > 0) {
					auto isin_expr = col.attr("isin")(non_zero_values);
					return Or(zero_expr, isin_expr);
				}
				return zero_expr;
			}

			if (nb::len(non_zero_values) > 0) {
				return col.attr("isin")(non_zero_values);
			}

			return dataset_scalar(false);
		}

		nb::list py_values;
		for (auto &val : values) {
			py_values.append(PythonObject::FromValue(val, val.type(), client_properties));
		}
		return col.attr("isin")(std::move(py_values));
	}

	nb::object And(nb::object a, nb::object b) override {
		return a.attr("__and__")(b);
	}

	nb::object Or(nb::object a, nb::object b) override {
		return a.attr("__or__")(b);
	}

private:
	const ClientProperties &client_properties;
	nb::object field_factory;
	nb::object dataset_scalar;
};

} // anonymous namespace

nb::object PyArrowFilterPushdown::TransformFilter(TableFilterSet &filter_collection,
                                                  unordered_map<idx_t, string> &columns,
                                                  unordered_map<idx_t, idx_t> filter_to_col,
                                                  const ClientProperties &config, const ArrowTableSchema &arrow_table,
                                                  const ArrowSchema &arrow_schema) {
	PyArrowBackend backend(config);
	nb::object expression = nb::none();
	for (auto &entry : filter_collection) {
		auto column_idx = entry.GetIndex();
		auto &column_name = columns[column_idx];
		D_ASSERT(columns.find(column_idx) != columns.end());

		vector<Identifier> column_path = {Identifier(column_name)};
		auto arrow_column_idx = filter_to_col.at(column_idx);
		auto &arrow_type = arrow_table.GetColumns().at(arrow_column_idx);
		D_ASSERT(arrow_column_idx < NumericCast<idx_t>(arrow_schema.n_children));
		auto column_schema = arrow_schema.children[arrow_column_idx];
		nb::object child_expression = duckdb::TransformFilter(entry.Filter(), std::move(column_path), backend,
		                                                      arrow_type.get(), column_schema, config.time_zone);
		if (child_expression.is(nb::none())) {
			continue;
		}
		if (expression.is(nb::none())) {
			expression = std::move(child_expression);
		} else {
			expression = expression.attr("__and__")(child_expression);
		}
	}
	return expression;
}

} // namespace duckdb
