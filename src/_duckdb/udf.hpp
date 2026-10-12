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

/// One parameter of a Python function as the engine declares it; `has_default` is separate since None is a default.
struct ParameterSpec {
	std::string name;
	cxx::FunctionParameterKind kind;
	std::string type;
	bool has_default;
	nb::object default_value;
};

/// Register `callable` as the scalar SQL function `name`; DuckDB only borrows it, so the caller keeps it alive.
void RegisterScalarFunction(cxx::Connection &connection, const std::string &name, nb::handle callable,
                            const std::vector<ParameterSpec> &parameters, const std::string &returns,
                            cxx::FunctionNullHandling nulls, cxx::FunctionStability level,
                            std::shared_ptr<ModuleState> module);

} // namespace duckdb_python
