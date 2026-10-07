//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/temporal.cpp
//
//
//===----------------------------------------------------------------------===//

#include "temporal.hpp"

#include "conversion.hpp"
#include "sql_types.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <cstdint>
#include <string>
#include <string_view>

namespace duckdb_python {

using duckdb::cxx::LogicalType;
using duckdb::cxx::LogicalTypeId;
using duckdb::cxx::Value;

using duckdb::cxx::Connection;

// Scaling the infinity markers would overflow and destroy them, so they pass through in their own unit. A coarse
// unit can hold an instant no microsecond count can, which is unrepresentable rather than a wrapped value; the
// diagnostic names the raw count, since rendering the value through DuckDB fails on the same conversion. A finer
// unit floors where ScaleCount truncates, so an instant before the epoch stays in its own microsecond.
int64_t MicrosFromUnit(int64_t raw, uint64_t multiply, uint64_t divide) {
	if (raw == TIMESTAMP_POSITIVE_INFINITY || raw == TIMESTAMP_NEGATIVE_INFINITY) {
		return raw;
	}
	int64_t micros;
	if (!ScaleCount(raw, UnitConversion {multiply, 1, divide}, false, micros)) {
		const char *unit = multiply == 1'000'000 ? " seconds" : " milliseconds";
		ThrowUnrepresentable("timestamp", std::to_string(raw) + unit + " since the epoch");
	}
	if (divide > 1 && raw < 0 && raw % static_cast<int64_t>(divide) != 0) {
		micros -= 1;
	}
	return micros;
}

TemporalKind KindOf(LogicalTypeId id) {
	switch (id) {
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
		return TemporalKind::NAIVE_INSTANT;
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return TemporalKind::ZONED_INSTANT;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_NS:
		return TemporalKind::NAIVE_TIME;
	case LogicalTypeId::TIME_TZ:
		return TemporalKind::ZONED_TIME;
	default:
		return TemporalKind::NONE;
	}
}

namespace {

constexpr TimeUnit kTimeUnits[] = {{"W", 604'800'000'000'000, 1},
                                   {"D", 86'400'000'000'000, 1},
                                   {"h", 3'600'000'000'000, 1},
                                   {"m", 60'000'000'000, 1},
                                   {"s", 1'000'000'000, 1},
                                   {"ms", 1'000'000, 1},
                                   {"us", 1'000, 1},
                                   {"ns", 1, 1},
                                   {"ps", 1, 1'000},
                                   {"fs", 1, 1'000'000},
                                   {"as", 1, 1'000'000'000}};

// Which kind of date or time a type is: an instant without or with a time zone, a time of day without or with one, or

int TimestampFineness(LogicalTypeId id) {
	switch (id) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return 0;
	case LogicalTypeId::TIMESTAMP_MS:
		return 1;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return 2;
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return 3;
	default:
		return -1;
	}
}

// The nanoseconds one count of the timestamp type `id` stands for.
uint64_t NanosPerCount(LogicalTypeId id) {
	switch (id) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return 1'000'000'000;
	case LogicalTypeId::TIMESTAMP_MS:
		return 1'000'000;
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return 1;
	default:
		return 1'000;
	}
}

template <class SCOPE>
Value TimestampValue(SCOPE &scope, LogicalTypeId id, int64_t count) {
	switch (id) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return Value::Create(scope, duckdb::cxx::timestamp_s_t {count});
	case LogicalTypeId::TIMESTAMP_MS:
		return Value::Create(scope, duckdb::cxx::timestamp_ms_t {count});
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::Create(scope, duckdb::cxx::timestamp_ns_t {count});
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::Create(scope, duckdb::cxx::timestamp_tz_t {count});
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return Value::Create(scope, duckdb::cxx::timestamp_tz_ns_t {count});
	default:
		return Value::Create(scope, duckdb::cxx::timestamp_t {count});
	}
}

// The count a timestamp value stores, in its own unit.
int64_t TimestampCount(const Value &value) {
	switch (value.GetLogicalType().GetTypeId()) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return value.Get<duckdb::cxx::timestamp_s_t>().seconds;
	case LogicalTypeId::TIMESTAMP_MS:
		return value.Get<duckdb::cxx::timestamp_ms_t>().millis;
	case LogicalTypeId::TIMESTAMP_NS:
		return value.Get<duckdb::cxx::timestamp_ns_t>().nanos;
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return value.Get<duckdb::cxx::timestamp_tz_ns_t>().nanos;
	case LogicalTypeId::TIMESTAMP_TZ:
		return value.Get<duckdb::cxx::timestamp_tz_t>().micros;
	default:
		return value.Get<duckdb::cxx::timestamp_t>().micros;
	}
}

} // namespace

const TimeUnit *FindTimeUnit(std::string_view code, bool finer_than_nanos) {
	for (const auto &unit : kTimeUnits) {
		if (unit.code == code) {
			return unit.per_nano == 1 || finer_than_nanos ? &unit : nullptr;
		}
	}
	return nullptr;
}

UnitConversion ConversionOf(const TimeUnit &unit, uint64_t step, uint64_t target_nanos) {
	const uint64_t whole = unit.per_nano * target_nanos;
	const uint64_t common = std::gcd(unit.nanos, whole);
	const uint64_t reduced = whole / common;
	const uint64_t shared = std::gcd(step, reduced);
	const UnitConversion conversion {step / shared, unit.nanos / common, reduced / shared};
	// ScaleCount relies on both; a unit added to the table that broke them would convert wrongly without a sound.
	if (conversion.denominator >= (uint64_t(1) << 40) || (conversion.denominator > 1 && conversion.unit != 1)) {
		throw duckdb::cxx::InvalidInputException("the time unit '" + std::string(unit.code) +
		                                         "' has no exact conversion");
	}
	return conversion;
}

bool SplitTimeUnit(std::string_view text, uint64_t &step, std::string_view &code) {
	const auto digits = text.find_first_not_of("0123456789");
	if (digits == std::string_view::npos || (digits > 1 && text[0] == '0') || digits > 10) {
		return false;
	}
	step = 1;
	if (digits > 0) {
		step = 0;
		for (std::size_t i = 0; i < digits; i++) {
			step = step * 10 + static_cast<uint64_t>(text[i] - '0');
		}
		if (step > 2'147'483'647) {
			return false;
		}
	}
	code = text.substr(digits);
	return true;
}

bool ParseTimeUnit(std::string_view text, const TimeUnit *&unit, uint64_t &step) {
	std::string_view code;
	if (!SplitTimeUnit(text, step, code)) {
		return false;
	}
	unit = FindTimeUnit(code, false);
	return unit != nullptr;
}

CountRange FiniteTimestampRange(LogicalTypeId id) {
	constexpr int64_t last_micros = std::numeric_limits<int64_t>::max() - 1;
	constexpr int64_t first_micros = -106'751'991LL * 86'400'000'000LL;
	switch (id) {
	case LogicalTypeId::TIMESTAMP_SEC:
		return {first_micros / 1'000'000, last_micros / 1'000'000};
	case LogicalTypeId::TIMESTAMP_MS:
		return {first_micros / 1'000, last_micros / 1'000};
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return {-106'751LL * 86'400'000'000'000LL, last_micros};
	default:
		return {first_micros, last_micros};
	}
}

bool IsInstant(LogicalTypeId id) {
	const auto kind = KindOf(id);
	return kind == TemporalKind::NAIVE_INSTANT || kind == TemporalKind::ZONED_INSTANT;
}

bool AssumesTimeZone(LogicalTypeId from, LogicalTypeId to) {
	const auto kind_from = KindOf(from);
	const auto kind_to = KindOf(to);
	if (kind_from == TemporalKind::NONE || kind_to == TemporalKind::NONE) {
		return false;
	}
	const auto zoned = [](TemporalKind kind) {
		return kind == TemporalKind::ZONED_INSTANT || kind == TemporalKind::ZONED_TIME;
	};
	return zoned(kind_from) != zoned(kind_to) ||
	       (kind_from == TemporalKind::ZONED_INSTANT && kind_to == TemporalKind::ZONED_TIME);
}

std::optional<Value> TryCastTemporalExactly(duckdb::cxx::Context &context, const Value &value,
                                            const LogicalType &target) {
	const auto from = value.GetLogicalType().GetTypeId();
	const auto to = target.GetTypeId();
	if (AssumesTimeZone(from, to)) {
		return std::nullopt;
	}
	if (TimestampFineness(from) >= 0 && TimestampFineness(to) >= 0 && !value.IsNull()) {
		// Between timestamps a count only scales, and the engine casts some pairs in neither direction.
		const auto from_nanos = NanosPerCount(from);
		const auto to_nanos = NanosPerCount(to);
		const UnitConversion conversion = from_nanos >= to_nanos ? UnitConversion {from_nanos / to_nanos, 1, 1}
		                                                         : UnitConversion {1, 1, to_nanos / from_nanos};
		int64_t count = 0;
		const auto range = FiniteTimestampRange(to);
		if (!ScaleCount(TimestampCount(value), conversion, true, count) || count < range.first || count > range.last) {
			return std::nullopt;
		}
		return TimestampValue(context, to, count);
	}
	if (value.GetLogicalType() == target) {
		return value.Cast(context, target);
	}
	try {
		auto cast = value.Cast(context, target);
		if (cast.Cast(context, value.GetLogicalType()).ToText() != value.ToText()) {
			return std::nullopt;
		}
		return cast;
	} catch (...) {
		return std::nullopt;
	}
}

// Where numpy's scaling of a count before printing it overflows int64, numpy refuses on some platforms and wraps to a
// wrong value on others, so such a count is never handed to it.
std::string ScalarText(nb::handle scalar, int64_t raw, uint64_t scale, const std::string &dtype) {
	const auto magnitude = raw < 0 ? 0 - static_cast<uint64_t>(raw) : static_cast<uint64_t>(raw);
	if (scale == 0 || magnitude <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / scale) {
		nb::object text = nb::steal(PyObject_Str(scalar.ptr()));
		if (text.is_valid()) {
			return nb::cast<std::string>(text);
		}
		PyErr_Clear();
	}
	return std::to_string(raw) + " of " + dtype;
}

// numpy's scalar types, matched by name, so numpy is never imported for the check.
bool IsNumpyScalar(nb::handle object, const char *name) {
	nb::handle type(reinterpret_cast<PyObject *>(Py_TYPE(object.ptr())));
	return type.attr("__module__").equal(nb::str("numpy")) && type.attr("__name__").equal(nb::str(name));
}

namespace {

constexpr int64_t DATE_LIMIT = 2'147'483'646;
// numpy converts months and years to days with wrapping arithmetic, so such a count is bounded first, far past DATE.
constexpr int64_t CALENDAR_LIMIT = int64_t(1) << 40;

// A numpy datetime64 or timedelta64 in its own unit. A day or coarser is a DATE; a finer datetime is the timestamp
// type storing its unit, hours and minutes in seconds, zoned for the UTC instant of an aware pandas Timestamp; a

} // namespace

template <class SCOPE>
Value NumpyTemporalToValue(SCOPE &scope, nb::handle scalar, bool is_datetime, bool zoned,
                           TimestampPrecision precision) {
	using duckdb::cxx::LogicalTypeId;
	const auto raw = nb::cast<int64_t>(nb::int_(scalar.attr("view")("i8")));
	if (raw == std::numeric_limits<int64_t>::min()) {
		// numpy's NaT.
		return Value::CreateNull(scope, scope.CreateType(LogicalTypeId::INTEGER));
	}
	const char *kind = is_datetime ? "datetime64" : "timedelta64";
	const auto spelled = nb::cast<std::string>(scalar.attr("dtype").attr("str"));
	const auto open = spelled.find('[');
	uint64_t step = 1;
	std::string_view code;
	if (open == std::string::npos ||
	    !SplitTimeUnit(std::string_view(spelled).substr(open + 1, spelled.size() - open - 2), step, code)) {
		ThrowInvalidInput(std::string("a ") + kind + " without a unit has no engine type");
	}
	// numpy prints a datetime of weeks as the days it counts.
	const uint64_t print_scale = is_datetime && code == "W" ? step * 7 : step;
	const TimeUnit *unit = FindTimeUnit(code, true);
	if (unit == nullptr) {
		if (!is_datetime || (code != "M" && code != "Y")) {
			ThrowInvalidInput(std::string("a ") + kind + " counting months or years has no fixed length");
		}
		// A step of zero makes every count the epoch, as it does for a timestamp.
		int64_t days = 0;
		if (step != 0) {
			const int64_t bound = CALENDAR_LIMIT / static_cast<int64_t>(step);
			if (raw > bound || raw < -bound) {
				ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
			}
			days = nb::cast<int64_t>(nb::int_(scalar.attr("astype")("<M8[D]").attr("view")("i8")));
		}
		if (days > DATE_LIMIT || days < -DATE_LIMIT) {
			ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
		}
		return Value::Create(scope, duckdb::cxx::date_t {static_cast<int32_t>(days)});
	}
	if (!is_datetime) {
		int64_t micros = 0;
		if (!ScaleCount(raw, ConversionOf(*unit, step, 1'000), false, micros)) {
			ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
		}
		return Value::Create(scope, duckdb::cxx::interval_t {0, 0, micros});
	}
	if (unit->code == "D" || unit->code == "W") {
		int64_t days = 0;
		if (!ScaleCount(raw, ConversionOf(*unit, step, 86'400'000'000'000), true, days) || days > DATE_LIMIT ||
		    days < -DATE_LIMIT) {
			ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
		}
		return Value::Create(scope, duckdb::cxx::date_t {static_cast<int32_t>(days)});
	}
	uint64_t target_nanos = unit->nanos >= 1'000'000'000 ? 1'000'000'000 : unit->nanos;
	if (zoned && target_nanos > 1) {
		target_nanos = 1'000;
	}
	if (precision == TimestampPrecision::MICROSECONDS && target_nanos == 1) {
		target_nanos = 1'000;
	}
	// Past the span nanoseconds hold, a value is held in microseconds, as an object column sampling it is typed. A
	// zoned value at a whole microsecond too: the engine neither combines TIMESTAMPTZ_NS with TIMESTAMPTZ nor casts
	// between them implicitly.
	if (target_nanos == 1) {
		const auto span = FiniteTimestampRange(LogicalTypeId::TIMESTAMP_NS);
		int64_t nanos = 0;
		if (!ScaleCount(raw, ConversionOf(*unit, step, 1), false, nanos) || nanos < span.first || nanos > span.last ||
		    (zoned && nanos % 1'000 == 0)) {
			target_nanos = 1'000;
		}
	}
	const auto target = target_nanos == 1       ? (zoned ? LogicalTypeId::TIMESTAMP_TZ_NS : LogicalTypeId::TIMESTAMP_NS)
	                    : target_nanos == 1'000 ? (zoned ? LogicalTypeId::TIMESTAMP_TZ : LogicalTypeId::TIMESTAMP)
	                    : target_nanos == 1'000'000 ? LogicalTypeId::TIMESTAMP_MS
	                                                : LogicalTypeId::TIMESTAMP_SEC;
	const auto conversion = ConversionOf(*unit, step, target_nanos);
	int64_t count = 0;
	if (!ScaleCount(raw, conversion, false, count)) {
		ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
	}
	if (conversion.denominator > 1 && !ScaleCount(raw, conversion, true, count)) {
		const auto text = ScalarText(scalar, raw, print_scale, spelled);
		const auto to_nanos = ConversionOf(*unit, step, 1);
		if (to_nanos.denominator > 1 && raw % static_cast<int64_t>(to_nanos.denominator) != 0) {
			ThrowInvalidInput("the value " + text + " is finer than the engine's nanoseconds");
		}
		if (precision == TimestampPrecision::MICROSECONDS) {
			ThrowInvalidInput("the value " + text +
			                  " has nanoseconds, and a query parameter holds microseconds at most");
		}
		// Only nanoseconds past the span TIMESTAMP_NS holds reach here.
		ThrowBeyondRange(text);
	}
	const auto range = FiniteTimestampRange(target);
	if (count < range.first || count > range.last) {
		ThrowBeyondRange(ScalarText(scalar, raw, print_scale, spelled));
	}
	return TimestampValue(scope, target, count);
}

template Value NumpyTemporalToValue<Connection>(Connection &, nb::handle, bool, bool, TimestampPrecision);
template Value NumpyTemporalToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle, bool, bool,
                                                          TimestampPrecision);

bool ContainsTemporal(const LogicalType &type) {
	return Contains(type, [](LogicalTypeId id) { return KindOf(id) != TemporalKind::NONE; });
}

// Whether the engine's cast from `from` to `to` takes a date or time anywhere inside across a time zone.
bool CastAssumesTimeZone(const LogicalType &from, const LogicalType &to) {
	const auto from_id = from.GetTypeId();
	const auto to_id = to.GetTypeId();
	if (from_id == LogicalTypeId::LIST && to_id == LogicalTypeId::LIST) {
		return CastAssumesTimeZone(from.GetListChildType(), to.GetListChildType());
	}
	if (from_id == LogicalTypeId::MAP && to_id == LogicalTypeId::MAP) {
		return CastAssumesTimeZone(from.GetMapKeyType(), to.GetMapKeyType()) ||
		       CastAssumesTimeZone(from.GetMapValueType(), to.GetMapValueType());
	}
	if (IsStruct(from_id) && IsStruct(to_id)) {
		const auto pairing = CastPairing(StructNames(from), StructNames(to));
		for (std::size_t i = 0; i < pairing.size(); i++) {
			if (pairing[i] && CastAssumesTimeZone(from.GetStructChildType(i), to.GetStructChildType(*pairing[i]))) {
				return true;
			}
		}
		return false;
	}
	return AssumesTimeZone(from_id, to_id);
}

} // namespace duckdb_python
