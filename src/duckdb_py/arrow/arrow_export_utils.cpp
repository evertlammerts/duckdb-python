#include "duckdb_python/arrow/arrow_array_stream.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyresult.hpp"

namespace duckdb {

namespace pyarrow {

py::object ToPyArrowSchema(ArrowSchema &schema) {
	py::gil_scoped_acquire acquire;

	auto pyarrow_lib_module = py::module::import("pyarrow").attr("lib");
	auto schema_import_func = pyarrow_lib_module.attr("Schema").attr("_import_from_c");
	return schema_import_func(reinterpret_cast<uint64_t>(&schema));
}

py::object ToArrowTable(const py::list &batches, py::object pyarrow_schema) {
	py::gil_scoped_acquire acquire;

	auto pyarrow_lib_module = py::module::import("pyarrow").attr("lib");
	auto from_batches_func = pyarrow_lib_module.attr("Table").attr("from_batches");

	return py::cast<duckdb::pyarrow::Table>(from_batches_func(batches, pyarrow_schema));
}

py::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const py::list &batches,
                        ClientProperties &options) {
	ArrowSchema schema;
	ArrowConverter::ToArrowSchema(&schema, types, names, options);
	return ToArrowTable(batches, ToPyArrowSchema(schema));
}

} // namespace pyarrow

} // namespace duckdb
