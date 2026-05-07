#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

namespace pyarrow {

py::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const py::list &batches,
                        ClientContext &client_context);

} // namespace pyarrow

} // namespace duckdb
