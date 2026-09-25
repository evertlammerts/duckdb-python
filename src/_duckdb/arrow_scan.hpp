//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrow_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "registry.hpp"

namespace duckdb_python {

/// The table function that reads a registered object, named by the replacement scan that resolves a name to it.
inline constexpr const char *kArrowScanFunction = "python_arrow_scan";

/// Registers the table function on the connection's database: bind resolves the name against `registry`, and the
/// scan reads the object's Arrow export with projection and filter pushdown, in batches of at most `batch_rows`,
/// the engine's standard vector size.
void RegisterArrowScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                       std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows);

} // namespace duckdb_python
