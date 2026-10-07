//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/value_to_python.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <vector>

#include "conversion.hpp"
#include "duckdb_cpp.hpp"

namespace duckdb_python {

/// One DuckDB value as a Python object. NULL becomes None.
nb::object ValueToPython(const duckdb::cxx::Value &value, ConversionContext &ctx);

/// Whether `ValueToPython` keeps every value of the type exactly, so the Python object stands for the engine's
/// value and not an approximation. Nanosecond timestamps and times floor to microseconds, an interval folds its
/// months, and a list or dict carries no element type, so those answer false.
bool ConvertsLossless(duckdb::cxx::LogicalTypeId type);

/// Rows [start, end) of a batch appended to `out` as tuples; `types` comes from the schema, not the data.
void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<duckdb::cxx::LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out);

/// Elements [first, last) of one column's data as a new list. NULL becomes None.
nb::list VectorElements(duckdb::cxx::Vector &vector, const duckdb::cxx::LogicalType &type, duckdb::cxx::idx_t first,
                        duckdb::cxx::idx_t last, ConversionContext &ctx);

} // namespace duckdb_python
