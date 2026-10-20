#include "duckdb_python/numpy/numpy_bind.hpp"
#include "duckdb_python/numpy/array_wrapper.hpp"
#include "duckdb_python/numpy/numpy_array.hpp"
#include "duckdb_python/pandas/pandas_analyzer.hpp"
#include "duckdb_python/pandas/column/pandas_numpy_column.hpp"
#include "duckdb_python/pandas/pandas_bind.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_conversion.hpp"

namespace duckdb {

idx_t NumpyBind::Bind(ClientContext &context, nb::handle df, vector<PandasColumnBindData> &bind_columns,
                      vector<LogicalType> &return_types, vector<Identifier> &names) {
	// The scan, which runs without the GIL, only recognizes pandas.NaT and pandas.NA once the import cache holds them
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (PandasIsImported(import_cache)) {
		(void)import_cache.pandas.NaT();
		(void)import_cache.pandas.NA();
	}

	auto keys = nb::list(df.attr("keys")());
	if (nb::len(keys) == 0) {
		throw InvalidInputException("Need a DataFrame with at least one column");
	}
	auto column_names = ColumnNamesFromLabels(keys);
	auto get_fun = df.attr("__getitem__");
	idx_t row_count = 0;
	string counted_from;
	for (idx_t col_idx = 0; col_idx < nb::len(keys); col_idx++) {
		LogicalType duckdb_col_type;
		PandasColumnBindData bind_data;
		bind_data.name = column_names[col_idx];
		names.emplace_back(bind_data.name);

		auto column = get_fun(keys[col_idx]);
		// Checked before any conversion, because numpy below 2 flattens the codes np.unique returns.
		if (col_idx == 0) {
			row_count = ColumnArrayRows(bind_data.name, column);
			counted_from = StringUtil::Format("'%s'", bind_data.name);
		} else {
			CheckColumnArray(bind_data.name, column, row_count, counted_from);
		}
		auto dtype = column.attr("dtype");
		if (nb::cast<std::string>(nb::str(nb::object(dtype.attr("char")))) == "U") {
			bind_data.numpy_type = ConvertNumpyType(nb::str("string"));
		} else {
			bind_data.numpy_type = ConvertNumpyType(dtype);
		}

		nb::object data;
		if (bind_data.numpy_type.type == NumpyNullableType::FLOAT_16) {
			data = column.attr("astype")("float32");
			bind_data.numpy_type.type = NumpyNullableType::FLOAT_32;
			duckdb_col_type = NumpyToLogicalType(bind_data.numpy_type);
		} else if (bind_data.numpy_type.type == NumpyNullableType::STRING) {
			bind_data.numpy_type.type = NumpyNullableType::CATEGORY;
			// here we call numpy.unique
			// this function call will return the unique values of a given array
			// together with the indices to reconstruct the given array
			auto uniq = nb::cast<nb::tuple>(nb::module_::import_("numpy").attr("unique")(column, false, true));
			vector<string> enum_entries;
			for (auto entry : uniq.attr("__getitem__")(0)) {
				enum_entries.push_back(TransformPythonString(nb::str(entry)));
			}
			idx_t size = enum_entries.size();
			Vector enum_entries_vec(LogicalType::VARCHAR, size);
			auto enum_entries_ptr = FlatVector::GetDataMutable<string_t>(enum_entries_vec);
			for (idx_t i = 0; i < size; i++) {
				enum_entries_ptr[i] = StringVector::AddStringOrBlob(enum_entries_vec, enum_entries[i]);
			}
			duckdb_col_type = LogicalType::ENUM(enum_entries_vec, size);
			data = uniq.attr("__getitem__")(1);
			bind_data.internal_categorical_type = nb::cast<std::string>(nb::str(nb::object(data.attr("dtype"))));
		} else {
			data = column;
			duckdb_col_type = NumpyToLogicalType(bind_data.numpy_type);
		}

		bind_data.pandas_col = std::make_unique<PandasNumpyColumn>(NumpyArray(std::move(data)));

		if (bind_data.numpy_type.type == NumpyNullableType::OBJECT) {
			PandasAnalyzer analyzer(context);
			if (analyzer.Analyze(column)) {
				duckdb_col_type = analyzer.AnalyzedType();
			}
		}

		return_types.push_back(duckdb_col_type);
		bind_columns.push_back(std::move(bind_data));
	}
	return row_count;
}

} // namespace duckdb
