#pragma once

#include "duckdb_python/nb/casters.hpp"

namespace nb = nanobind;

namespace duckdb {

void RegisterExceptions(const nb::module_ &m);

} // namespace duckdb
