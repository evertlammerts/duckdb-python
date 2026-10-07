//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/temporal.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "conversion.hpp"
#include "duckdb_cpp.hpp"

namespace duckdb_python {

// DuckDB reserves the extremes of each storage type for infinite dates and timestamps, which Python cannot
// represent, so they convert to Python's own extremes instead.
constexpr int32_t DATE_POSITIVE_INFINITY = 2147483647;
constexpr int32_t DATE_NEGATIVE_INFINITY = -2147483647;
constexpr int64_t TIMESTAMP_POSITIVE_INFINITY = 9223372036854775807LL;
constexpr int64_t TIMESTAMP_NEGATIVE_INFINITY = -9223372036854775807LL;

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

/// A unit numpy spells after a datetime64 or timedelta64 step, as `nanos` nanoseconds or, for the last three, one
/// `per_nano`-th of a nanosecond. Every unit of a nanosecond or more is a whole multiple, or a power-of-ten fraction,
/// of every target unit, which is what keeps `UnitConversion`'s denominator above 1 only when its unit factor is 1.
struct TimeUnit {
	std::string_view code;
	uint64_t nanos;
	uint64_t per_nano;
};

/// The unit `code` names; null for a calendar or unknown code, and for a unit finer than a nanosecond unless
/// `finer_than_nanos`.
const TimeUnit *FindTimeUnit(std::string_view code, bool finer_than_nanos);

/// How a count of `step` times `unit` becomes a count of `target_nanos` nanoseconds.
UnitConversion ConversionOf(const TimeUnit &unit, uint64_t step, uint64_t target_nanos);

/// A unit as a datetime64 or timedelta64 dtype spells it inside its brackets, split into its step and its code: an
/// optional step, from 0 to numpy's largest, 2147483647, with no sign or leading zero, then the code.
bool SplitTimeUnit(std::string_view text, uint64_t &step, std::string_view &code);

/// As `SplitTimeUnit`, for a code of a nanosecond or more.
bool ParseTimeUnit(std::string_view text, const TimeUnit *&unit, uint64_t &step);

/// The first and last counts of a timestamp type that the engine holds as finite instants.
struct CountRange {
	int64_t first;
	int64_t last;
};

/// The counts of the timestamp type `id` the engine can use. The extremes of int64 are its infinities; past the
/// rest, its own casts of a stored count fail with an error that closes the database: a coarse count converts to
/// microseconds, and any count to a calendar date, which overflows before 290309-12-22 BC in microseconds and before
/// 1677-09-22 in nanoseconds. A count outside is refused instead, and the range can widen when the engine's casts
/// cover all of int64.
CountRange FiniteTimestampRange(duckdb::cxx::LogicalTypeId id);

/// Whether converting a date or time of type `from` to type `to` needs a time zone the value does not name: between a
/// type with a time zone and one without, or from a timestamp with a time zone to a time of day, which it has only in
/// some time zone chosen for it.
bool AssumesTimeZone(duckdb::cxx::LogicalTypeId from, duckdb::cxx::LogicalTypeId to);

/// `value` cast to `target` when the cast loses nothing, and nullopt otherwise; the engine's own cast rounds rather
/// than refuses.
std::optional<duckdb::cxx::Value> TryCastTemporalExactly(duckdb::cxx::Context &context, const duckdb::cxx::Value &value,
                                                         const duckdb::cxx::LogicalType &target);

/// Which kind of date or time a type is: an instant without or with a time zone, a time of day without or with
/// one, or none of these.
enum class TemporalKind : uint8_t { NONE, NAIVE_INSTANT, ZONED_INSTANT, NAIVE_TIME, ZONED_TIME };

TemporalKind KindOf(duckdb::cxx::LogicalTypeId id);

/// Whether the type is a timestamp or date, an instant on the timeline rather than a time of day.
bool IsInstant(duckdb::cxx::LogicalTypeId id);

/// Whether the type holds a date or time anywhere inside it.
bool ContainsTemporal(const duckdb::cxx::LogicalType &type);

/// Whether the engine's cast from `from` to `to` would assume a time zone for some date or time inside.
bool CastAssumesTimeZone(const duckdb::cxx::LogicalType &from, const duckdb::cxx::LogicalType &to);

/// A count in a coarser unit as microseconds, the infinity markers passed through unscaled.
int64_t MicrosFromUnit(int64_t raw, uint64_t multiply, uint64_t divide);

/// A numpy scalar as numpy prints it, or as its raw count and dtype when printing it fails.
std::string ScalarText(nanobind::handle scalar, int64_t raw, const std::string &dtype);

/// Whether `object` is the numpy scalar `name`, recognised by its type's name and module.
bool IsNumpyScalar(nanobind::handle object, const char *name);

/// A numpy datetime64 or timedelta64 scalar as an engine value in its own unit; `temporal.cpp` states the rules.
template <class SCOPE>
duckdb::cxx::Value NumpyTemporalToValue(SCOPE &scope, nanobind::handle scalar, bool is_datetime, bool zoned,
                                        TimestampPrecision precision);

extern template duckdb::cxx::Value NumpyTemporalToValue<duckdb::cxx::Connection>(duckdb::cxx::Connection &,
                                                                                 nanobind::handle, bool, bool,
                                                                                 TimestampPrecision);
extern template duckdb::cxx::Value NumpyTemporalToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nanobind::handle,
                                                                              bool, bool, TimestampPrecision);

} // namespace duckdb_python
