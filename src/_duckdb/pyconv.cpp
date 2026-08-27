//===----------------------------------------------------------------------===//
//                         DuckDB
//
// pyconv.cpp
//
// Conversion between DuckDB values and Python objects.
//===----------------------------------------------------------------------===//

#include "pyconv.hpp"

#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

namespace duckdb_python {

using duckdb::cxx::LogicalTypeId;
using duckdb::cxx::Value;

namespace {

// DuckDB counts days and microseconds from the Unix epoch; Python's date and
// datetime count from year 1. Convert by offsetting from the epoch rather than
// reimplementing the calendar.
nb::object EpochDate(ConversionContext &ctx, int32_t days) {
	return ctx.date_cls(1970, 1, 1) + ctx.timedelta_cls(days, 0, 0);
}

nb::object EpochDateTime(ConversionContext &ctx, int64_t micros, bool utc) {
	nb::object epoch = utc ? ctx.datetime_cls(1970, 1, 1, 0, 0, 0, 0, ctx.timezone_utc)
	                       : ctx.datetime_cls(1970, 1, 1);
	return epoch + ctx.timedelta_cls(0, 0, micros);
}

// A leading integer, for the types whose text form is exact but whose binary
// form has no Python counterpart (HUGEINT is wider than any C integer nanobind
// converts).
nb::object IntFromText(const std::string &text) {
	return nb::module_::import_("builtins").attr("int")(text);
}

} // namespace

ConversionContext::ConversionContext() {
	nb::object datetime = nb::module_::import_("datetime");
	date_cls = datetime.attr("date");
	time_cls = datetime.attr("time");
	datetime_cls = datetime.attr("datetime");
	timedelta_cls = datetime.attr("timedelta");
	timezone_utc = datetime.attr("timezone").attr("utc");
	decimal_cls = nb::module_::import_("decimal").attr("Decimal");
	uuid_cls = nb::module_::import_("uuid").attr("UUID");
}

nb::object ValueToPython(const Value &value, ConversionContext &ctx) {
	if (value.IsNull()) {
		return nb::none();
	}
	const auto type = value.GetLogicalType();
	switch (type.GetTypeId()) {
	case LogicalTypeId::BOOLEAN:
		return nb::cast(value.Get<bool>());
	case LogicalTypeId::TINYINT:
		return nb::cast(value.Get<int8_t>());
	case LogicalTypeId::SMALLINT:
		return nb::cast(value.Get<int16_t>());
	case LogicalTypeId::INTEGER:
		return nb::cast(value.Get<int32_t>());
	case LogicalTypeId::BIGINT:
		return nb::cast(value.Get<int64_t>());
	case LogicalTypeId::UTINYINT:
		return nb::cast(value.Get<uint8_t>());
	case LogicalTypeId::USMALLINT:
		return nb::cast(value.Get<uint16_t>());
	case LogicalTypeId::UINTEGER:
		return nb::cast(value.Get<uint32_t>());
	case LogicalTypeId::UBIGINT:
		return nb::cast(value.Get<uint64_t>());
	case LogicalTypeId::FLOAT:
		return nb::cast(value.Get<float>());
	case LogicalTypeId::DOUBLE:
		return nb::cast(value.Get<double>());
	case LogicalTypeId::VARCHAR:
		return nb::cast(std::string(value.Get<duckdb::cxx::varchar_t>()));
	case LogicalTypeId::BLOB: {
		const auto blob = value.Get<duckdb::cxx::blob_t>();
		return nb::bytes(blob.data(), blob.size());
	}
	case LogicalTypeId::DATE:
		return EpochDate(ctx, value.Get<duckdb::cxx::date_t>().days);
	case LogicalTypeId::TIME: {
		// A time of day is a microsecond offset with no date, so build it by
		// offsetting midnight and dropping the date part.
		const auto micros = value.Get<duckdb::cxx::dtime_t>().micros;
		return (ctx.datetime_cls(1970, 1, 1) + ctx.timedelta_cls(0, 0, micros)).attr("time")();
	}
	case LogicalTypeId::TIMESTAMP:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_t>().micros, false);
	case LogicalTypeId::TIMESTAMP_TZ:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_tz_t>().micros, true);
	case LogicalTypeId::INTERVAL: {
		const auto interval = value.Get<duckdb::cxx::interval_t>();
		// A month is not a fixed duration, so this mapping is lossy for any
		// interval carrying months. Recorded as a documented divergence; the
		// previous client folded months at 30 days and we match it.
		return ctx.timedelta_cls(static_cast<int64_t>(interval.months) * 30 + interval.days, 0,
		                         interval.micros);
	}
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
		// Exact: the text form of an integer loses nothing, and Python ints are
		// arbitrary precision.
		return IntFromText(value.ToText());
	case LogicalTypeId::DECIMAL:
		// Exact, and deliberately not float: Decimal(str) preserves the scale.
		return ctx.decimal_cls(value.ToText());
	case LogicalTypeId::UUID:
		return ctx.uuid_cls(value.ToText());
	case LogicalTypeId::ENUM:
		return nb::cast(value.ToText());
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY: {
		nb::list out;
		const auto count = value.GetChildCount();
		for (duckdb::cxx::idx_t i = 0; i < count; i++) {
			out.append(ValueToPython(value.GetChild(i), ctx));
		}
		return out;
	}
	case LogicalTypeId::STRUCT: {
		nb::dict out;
		const auto count = value.GetChildCount();
		for (duckdb::cxx::idx_t i = 0; i < count; i++) {
			out[nb::cast(std::string(type.GetStructChildName(i)))] = ValueToPython(value.GetChild(i), ctx);
		}
		return out;
	}
	case LogicalTypeId::MAP: {
		nb::dict out;
		const auto count = value.GetChildCount();
		// Children alternate key, value.
		for (duckdb::cxx::idx_t i = 0; i + 1 < count; i += 2) {
			out[ValueToPython(value.GetChild(i), ctx)] = ValueToPython(value.GetChild(i + 1), ctx);
		}
		return out;
	}
	case LogicalTypeId::UNION:
		// Child 0 is the tag, child 1 the active member.
		return ValueToPython(value.GetChild(1), ctx);
	default:
		// BIT, BIGNUM, VARIANT, GEOMETRY, and anything a newer engine adds.
		// Degrading to the SQL text keeps unknown types readable instead of
		// failing the whole fetch.
		return nb::cast(value.ToText());
	}
}

} // namespace duckdb_python
