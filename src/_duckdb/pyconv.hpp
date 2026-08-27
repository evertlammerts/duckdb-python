//===----------------------------------------------------------------------===//
//                         DuckDB
//
// pyconv.hpp
//
// Conversion between DuckDB values and Python objects.
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
	nb::object timezone_utc;
	nb::object decimal_cls;
	nb::object uuid_cls;
};

/// One DuckDB value as a Python object. NULL becomes None.
nb::object ValueToPython(const duckdb::cxx::Value &value, ConversionContext &ctx);

} // namespace duckdb_python
