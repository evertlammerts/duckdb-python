//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pyconv.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
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

/// How a count in one time unit becomes a count in another: multiplied by `step` and `unit`, then divided by
/// `denominator`, truncating toward zero. The factors stay apart so a unit too large for any nonzero count to fit
/// still converts zero; `denominator` is above 1 only when `unit` is 1, and is below 2^40.
struct UnitConversion {
	uint64_t step;
	uint64_t unit;
	uint64_t denominator;
};

inline bool MultiplyWithin(uint64_t a, uint64_t b, uint64_t limit, uint64_t &out) {
	if (a != 0 && b > limit / a) {
		return false;
	}
	out = a * b;
	return true;
}

inline bool AddWithin(uint64_t a, uint64_t b, uint64_t limit, uint64_t &out) {
	if (b > limit || a > limit - b) {
		return false;
	}
	out = a + b;
	return true;
}

/// `raw` converted, false when the result does not fit an int64 or, with `exact`, when truncation would drop a
/// remainder. Inline because the scan calls it for every row.
inline bool ScaleCount(int64_t raw, const UnitConversion &conversion, bool exact, int64_t &out) {
	if (conversion.step == 1 && conversion.unit == 1) {
		// The identity, or a division with nothing to multiply, such as nanoseconds to microseconds: nothing overflows.
		const auto d = static_cast<int64_t>(conversion.denominator);
		if (exact && raw % d != 0) {
			return false;
		}
		out = raw / d;
		return true;
	}
	const bool negative = raw < 0;
	// |raw| without negating INT64_MIN, and a negative result may reach -2^63.
	const uint64_t magnitude = negative ? static_cast<uint64_t>(-(raw + 1)) + 1 : static_cast<uint64_t>(raw);
	const uint64_t limit = negative ? uint64_t(1) << 63 : (uint64_t(1) << 63) - 1;
	uint64_t scaled = 0;
	if (conversion.denominator == 1) {
		if (!MultiplyWithin(magnitude, conversion.step, limit, scaled) ||
		    !MultiplyWithin(scaled, conversion.unit, limit, scaled)) {
			return false;
		}
	} else {
		// With `unit` 1, magnitude * step / d is a * step + b * q + (b * r) / d. The first two terms never exceed the
		// result, so their overflow is real. b * r itself may pass 64 bits; splitting r into 20-bit halves keeps every
		// partial product used for its quotient and remainder below 2^61, which needs d below 2^40.
		const uint64_t d = conversion.denominator;
		const uint64_t a = magnitude / d;
		const uint64_t b = magnitude % d;
		const uint64_t q = conversion.step / d;
		const uint64_t r = conversion.step % d;
		uint64_t part = 0;
		if (!MultiplyWithin(a, conversion.step, limit, scaled) || !MultiplyWithin(b, q, limit, part) ||
		    !AddWithin(scaled, part, limit, scaled)) {
			return false;
		}
		const uint64_t x = b * (r >> 20);
		const uint64_t y = ((x % d) << 20) + b * (r & ((uint64_t(1) << 20) - 1));
		if (exact && y % d != 0) {
			return false;
		}
		if (!AddWithin(scaled, ((x / d) << 20) + y / d, limit, scaled)) {
			return false;
		}
	}
	if (!negative || scaled == 0) {
		out = static_cast<int64_t>(scaled);
	} else {
		out = -static_cast<int64_t>(scaled - 1) - 1;
	}
	return true;
}

/// One DuckDB value as a Python object. NULL becomes None.
nb::object ValueToPython(const duckdb::cxx::Value &value, ConversionContext &ctx);

/// Whether `ValueToPython` keeps every value of the type exactly, so the Python object stands for the engine's
/// value and not an approximation. Nanosecond timestamps and times floor to microseconds, an interval folds its
/// months, and a list or dict carries no element type, so those answer false.
bool ConvertsLossless(duckdb::cxx::LogicalTypeId type);

/// "ExceptionType: message" followed by the traceback; rendered while the GIL is held.
std::string DescribePythonError(nb::python_error &error);

/// Rows [start, end) of a batch appended to `out` as tuples; `types` comes from the schema, not the data.
void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<duckdb::cxx::LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out);

/// Elements [first, last) of one column's data as a new list. NULL becomes None.
nb::list VectorElements(duckdb::cxx::Vector &vector, const duckdb::cxx::LogicalType &type, duckdb::cxx::idx_t first,
                        duckdb::cxx::idx_t last, ConversionContext &ctx);

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

extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Connection>(duckdb::cxx::Connection &, nb::handle,
                                                                          ConversionContext &);
extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle,
                                                                       ConversionContext &);

} // namespace duckdb_python
