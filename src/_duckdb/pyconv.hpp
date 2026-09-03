//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pyconv.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <vector>

#include "duckdb_cpp.hpp"

namespace duckdb_python {

namespace nb = nanobind;

/// Python constructors held for the lifetime of the module, so the hot loop
/// does not re-import them per value.
struct ConversionContext {
	ConversionContext();

	nb::object date_cls;
	nb::object time_cls;
	nb::object datetime_cls;
	nb::object timedelta_cls;
	nb::object timezone_cls;
	nb::object timezone_utc;
	nb::object decimal_cls;
	nb::object uuid_cls;
	nb::object int_cls;

	/// Context for exact scaleb on the int128 decimal tier: its precision
	/// clears the 39 digits an int128 can carry, so it never rounds.
	nb::object decimal_context;
	/// 2^64 as a Python int, for combining 128-bit limbs exactly.
	nb::object two_pow_64;

	/// Epochs and the unit, cached because both directions offset from them
	/// per value.
	nb::object epoch_date;

	/// Epochs held for the parameter direction, so datetimes convert by
	/// subtraction rather than by reimplementing the calendar.
	nb::object epoch_naive;
	nb::object epoch_aware;
	nb::object one_microsecond;
};

/// One DuckDB value as a Python object. NULL becomes None.
nb::object ValueToPython(const duckdb::cxx::Value &value, ConversionContext &ctx);

/// Rows [start, end) of a chunk appended to `out` as tuples, converted
/// column-at-a-time from the flattened vector data. `types` carries the
/// columns' logical types, which this facade keeps on the result's schema,
/// not on the vectors.
void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<duckdb::cxx::LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out);

/// Elements [first, last) of one vector as a new list. NULL becomes None.
nb::list VectorElements(duckdb::cxx::Vector &vector, const duckdb::cxx::LogicalType &type,
                        duckdb::cxx::idx_t first, duckdb::cxx::idx_t last, ConversionContext &ctx);

/// One Python object as a DuckDB value, for binding as a parameter or writing
/// a function result. `scope` is the Connection outside engine callbacks and
/// the callback's Context inside one; the facade's value factories take both.
template <class SCOPE>
duckdb::cxx::Value PythonToValue(SCOPE &scope, nb::handle object, ConversionContext &ctx);

extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Connection>(duckdb::cxx::Connection &,
                                                                          nb::handle, ConversionContext &);
extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle,
                                                                       ConversionContext &);

} // namespace duckdb_python
