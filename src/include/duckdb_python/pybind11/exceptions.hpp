#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace nb = nanobind;

namespace duckdb {

void RegisterExceptions(const nb::module_ &m);

} // namespace duckdb
