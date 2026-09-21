//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pyconv.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "duckdb_cpp.hpp"

namespace duckdb_python {

namespace nb = nanobind;

/// Python constructors held for the module's lifetime, so converting a value never re-imports them.
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

	/// Wide enough for the 39 digits a 128-bit integer can carry, so rescaling a decimal never rounds.
	nb::object decimal_context;
	/// 2^64 as a Python int, for combining the two halves of a 128-bit integer exactly.
	nb::object two_pow_64;

	/// 1970-01-01, cached because every date conversion offsets from it.
	nb::object epoch_date;

	/// The same day as a datetime, so values convert by subtraction rather than by redoing the calendar.
	nb::object epoch_naive;
	nb::object epoch_aware;
	nb::object one_microsecond;
};

/// One DuckDB value as a Python object. NULL becomes None.
nb::object ValueToPython(const duckdb::cxx::Value &value, ConversionContext &ctx);

/// "ExceptionType: message" followed by the traceback; rendered while the GIL is held.
std::string DescribePythonError(nb::python_error &error);

/// Rows [start, end) of a batch appended to `out` as tuples; `types` comes from the schema, not the data.
void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<duckdb::cxx::LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out);

/// Elements [first, last) of one column's data as a new list. NULL becomes None.
nb::list VectorElements(duckdb::cxx::Vector &vector, const duckdb::cxx::LogicalType &type,
                        duckdb::cxx::idx_t first, duckdb::cxx::idx_t last, ConversionContext &ctx);

/// Raised by PythonToValue for an object it cannot convert; worded for query parameters, so others reword it.
class UnsupportedTypeException : public duckdb::cxx::InvalidInputException {
public:
	explicit UnsupportedTypeException(std::string type_name);

	/// The offending object's Python type name.
	const std::string &TypeName() const {
		return type_name;
	}

private:
	std::string type_name;
};

/// One Python object as a DuckDB value; `scope` is a Connection, or the Context inside a function callback.
template <class SCOPE>
duckdb::cxx::Value PythonToValue(SCOPE &scope, nb::handle object, ConversionContext &ctx);

extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Connection>(duckdb::cxx::Connection &,
                                                                          nb::handle, ConversionContext &);
extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle,
                                                                       ConversionContext &);

} // namespace duckdb_python
