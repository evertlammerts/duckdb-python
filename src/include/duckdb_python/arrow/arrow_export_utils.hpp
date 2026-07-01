#pragma once

#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

namespace pyarrow {

nb::object ToPyArrowSchema(const ArrowSchema &schema);

nb::object ToArrowTable(const vector<LogicalType> &types, const vector<string> &names, const nb::list &batches,
                        ClientProperties &options);

nb::object ToArrowTable(const nb::list &batches, nb::object pyarrow_schema);

} // namespace pyarrow

} // namespace duckdb
