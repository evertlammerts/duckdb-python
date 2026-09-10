//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/udf.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <string>
#include <vector>

#include "lifetime.hpp"

namespace duckdb_python {

/// Register `callable` as the scalar SQL function `name`; DuckDB only borrows it, so the caller keeps it alive.
void RegisterScalarFunction(cxx::Connection &connection, const std::string &name, nb::handle callable,
                            const std::vector<std::string> &parameters, const std::string &returns,
                            cxx::FunctionNullHandling nulls, cxx::FunctionStability level,
                            std::shared_ptr<ModuleState> module);

} // namespace duckdb_python
