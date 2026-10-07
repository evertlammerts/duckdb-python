//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/conversion.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <string>

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
	nb::object mapping_cls;

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

/// How finely `PythonToValue` keeps a timestamp: in its own unit, or in microseconds at most, as a query parameter
/// does while the engine cannot compare TIMESTAMPTZ_NS with TIMESTAMPTZ; a finer value is then refused, not truncated.
enum class TimestampPrecision : uint8_t { OWN_UNIT, MICROSECONDS };

/// Raised by `PythonToDeclaredType` for a date or time its declared type cannot take as it is; the message names the
/// value, the part of the declared type it met and why, and its readers prefix whose value it was.
class DeclaredTypeRefusal : public duckdb::cxx::InvalidInputException {
public:
	using duckdb::cxx::InvalidInputException::InvalidInputException;
};

/// Raised by `PythonToValue` for a date or time past what its engine type holds as a finite value.
class BeyondRangeException : public duckdb::cxx::InvalidInputException {
public:
	using duckdb::cxx::InvalidInputException::InvalidInputException;
};

/// Raised by `PythonToValue` for an object it cannot convert; worded for query parameters, so others reword it.
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

[[noreturn]] inline void ThrowInvalidInput(const std::string &body) {
	throw duckdb::cxx::InvalidInputException("Invalid Input Error: " + body, body);
}

[[noreturn]] inline void ThrowBeyondRange(const std::string &text) {
	const auto body = "the value " + text + " is beyond the range of its engine type";
	throw BeyondRangeException("Invalid Input Error: " + body, body);
}

/// DuckDB dates reach far past Python's year 9999, so name the offending value; OverflowError names a day count.
[[noreturn]] inline void ThrowUnrepresentable(const std::string &what, const std::string &rendered,
                                              const char *python_type = "datetime") {
	throw duckdb::cxx::Exception(4001 /* TYPE_CONVERSION */, "Conversion Error: " + what + " " + rendered +
	                                                             " is outside the range Python's " + python_type +
	                                                             " can represent");
}

/// For types whose text form is exact but whose binary form is wider than any C integer nanobind converts.
inline nb::object IntFromText(ConversionContext &ctx, const std::string &text) {
	return ctx.int_cls(text);
}

/// "ExceptionType: message" followed by the traceback; rendered while the GIL is held.
std::string DescribePythonError(nb::python_error &error);

} // namespace duckdb_python
