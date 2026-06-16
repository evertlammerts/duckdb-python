#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

namespace pyarrow {

py::object ToPyArrowSchema(const ArrowSchema &schema);

py::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const py::list &batches,
                        ClientProperties &options);

py::object ToArrowTable(const py::list &batches, py::object pyarrow_schema);

} // namespace pyarrow

} // namespace duckdb
