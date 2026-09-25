//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/numpy_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "registry.hpp"

namespace duckdb_python {

/// The table function that reads a registered object natively, named by the replacement scan for an entry whose
/// source answered native.
inline constexpr const char *kNumpyScanFunction = "python_numpy_scan";

/// Registers the table function on the connection's database: bind resolves the name against `registry` and reads
/// the source's `describe()`, and the scan reads its `columns()` through the buffer protocol, in batches of at
/// most `batch_rows`, the engine's standard vector size.
void RegisterNumpyScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                       std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows);

} // namespace duckdb_python
