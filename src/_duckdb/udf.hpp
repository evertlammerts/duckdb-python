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

/// Register `callable` as the scalar SQL function `name` on `connection`.
///
/// The engine borrows the callable from the moment this returns; the caller
/// keeps it alive for as long as the database lives. The type texts go to
/// the engine's parser as they are.
void RegisterScalarFunction(cxx::Connection &connection, const std::string &name, nb::handle callable,
                            const std::vector<std::string> &parameters, const std::string &returns,
                            cxx::FunctionNullHandling nulls, cxx::FunctionStability level,
                            std::shared_ptr<ModuleState> module);

} // namespace duckdb_python
