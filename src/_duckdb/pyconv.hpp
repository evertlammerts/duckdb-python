//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pyconv.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

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

/// One Python object as a DuckDB value, for binding as a parameter.
duckdb::cxx::Value PythonToValue(duckdb::cxx::Connection &connection, nb::handle object,
                                 ConversionContext &ctx);

} // namespace duckdb_python
