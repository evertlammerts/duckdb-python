//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/python_to_value.cpp
//
//
//===----------------------------------------------------------------------===//

#include "python_to_value.hpp"

#include "conversion.hpp"
#include "sql_types.hpp"
#include "temporal.hpp"
#include "untyped.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <limits>
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

namespace {

constexpr int64_t EPOCH_ORDINAL = 719163;

// Whole microseconds between two datetimes; total_seconds() is a float and loses digits far from the epoch.
int64_t MicrosSince(const nb::object &epoch, nb::handle moment, ConversionContext &ctx) {
	nb::object delta = nb::steal(PyNumber_Subtract(moment.ptr(), epoch.ptr()));
	if (!delta.is_valid()) {
		throw nb::python_error();
	}
	nb::object count = nb::steal(PyNumber_FloorDivide(delta.ptr(), ctx.one_microsecond.ptr()));
	if (!count.is_valid()) {
		throw nb::python_error();
	}
	return nb::cast<int64_t>(count);
}

// Through text, which is exact for integers of any width and for decimals, neither having a C++ type here.
template <class SCOPE>
Value FromText(SCOPE &scope, const std::string &text, const LogicalType &target) {
	const duckdb::cxx::varchar_t borrowed(text);
	return Value::Create(scope, borrowed).Cast(scope, target);
}

[[noreturn]] void ThrowUnsupported(nb::handle object) {
	throw UnsupportedTypeException(nb::cast<std::string>(nb::handle(Py_TYPE(object.ptr())).attr("__name__")));
}

template <class SCOPE>
Value ScalarValue(SCOPE &scope, nb::handle object, ConversionContext &ctx, TimestampPrecision precision) {
	using duckdb::cxx::LogicalTypeId;

	// A NULL still needs a type; SQLNULL is rejected, and INTEGER works because a NULL casts to anything.
	if (object.is_none()) {
		return Value::CreateNull(scope, scope.CreateType(LogicalTypeId::INTEGER));
	}
	// Before the int branch: a Python bool is an int, so testing int first would silently pass True as 1.
	if (nb::isinstance<nb::bool_>(object)) {
		return Value::Create(scope, nb::cast<bool>(object));
	}
	if (nb::isinstance<nb::int_>(object)) {
		// The widest type that holds the value, since narrowing here would reject values the column can hold.
		int64_t narrow = 0;
		if (nb::try_cast<int64_t>(object, narrow)) {
			return Value::Create(scope, narrow);
		}
		const auto text = nb::cast<std::string>(nb::str(object));
		// The C++ API pins C++17, so no starts_with here.
		const bool negative = !text.empty() && text[0] == '-';
		const auto id = negative ? LogicalTypeId::HUGEINT : LogicalTypeId::UHUGEINT;
		return FromText(scope, text, scope.CreateType(id));
	}
	if (nb::isinstance<nb::float_>(object)) {
		return Value::Create(scope, nb::cast<double>(object));
	}
	if (nb::isinstance<nb::str>(object)) {
		const auto text = nb::cast<std::string>(object);
		return Value::Create(scope, duckdb::cxx::varchar_t(text));
	}
	if (nb::isinstance<nb::bytes>(object)) {
		const auto bytes = nb::cast<nb::bytes>(object);
		// A BLOB length is 32 bits, so anything larger would wrap and be passed silently truncated.
		if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
			ThrowInvalidInput("bytes value is larger than a BLOB can hold");
		}
		return Value::Create(scope, duckdb::cxx::blob_t(bytes.c_str(), static_cast<uint32_t>(bytes.size())));
	}
	// datetime before date: datetime subclasses date, so order decides.
	if (nb::isinstance(object, ctx.datetime_cls)) {
		// A pandas Timestamp keeps its own unit, down to nanoseconds, in `asm8`, a numpy scalar of its UTC instant;
		// pandas' NaT is a datetime whose `asm8` is numpy's NaT. Only a subclass can carry it, so the exact type
		// skips the lookup, whose miss costs an exception inside hasattr.
		const bool exact = Py_TYPE(object.ptr()) == reinterpret_cast<PyTypeObject *>(ctx.datetime_cls.ptr());
		if (!exact && nb::hasattr(object, "asm8")) {
			const nb::object instant = object.attr("asm8");
			if (IsNumpyScalar(instant, "datetime64")) {
				return NumpyTemporalToValue(scope, instant, true, !object.attr("tzinfo").is_none(), precision);
			}
		}
		// Aware only when its time zone gives an offset, as Python defines it; a datetime.timezone always gives one,
		// so it is not asked, and a missing tzinfo asks nothing.
		const nb::object tz = object.attr("tzinfo");
		const bool aware =
		    !tz.is_none() && (Py_TYPE(tz.ptr()) == reinterpret_cast<PyTypeObject *>(ctx.timezone_cls.ptr()) ||
		                      !object.attr("utcoffset")().is_none());
		const int64_t micros = MicrosSince(aware ? ctx.epoch_aware : ctx.epoch_naive, object, ctx);
		if (aware) {
			return Value::Create(scope, duckdb::cxx::timestamp_tz_t {micros});
		}
		return Value::Create(scope, duckdb::cxx::timestamp_t {micros});
	}
	if (nb::isinstance(object, ctx.date_cls)) {
		const auto days = nb::cast<int64_t>(object.attr("toordinal")()) - EPOCH_ORDINAL;
		return Value::Create(scope, duckdb::cxx::date_t {static_cast<int32_t>(days)});
	}
	if (nb::isinstance(object, ctx.time_cls)) {
		nb::object naive = object.attr("replace")(nb::arg("tzinfo") = nb::none());
		nb::object combined = ctx.datetime_cls.attr("combine")(ctx.epoch_date, naive);
		const int64_t micros = MicrosSince(ctx.epoch_naive, combined, ctx);

		// A time with a time zone becomes TIME_TZ; dropping the offset would silently break the round trip.
		nb::object offset = object.attr("utcoffset")();
		if (offset.is_none() && !object.attr("tzinfo").is_none()) {
			ThrowInvalidInput("the time " + nb::cast<std::string>(nb::str(object)) +
			                  " has a time zone whose offset depends on a date, which a time does not have");
		}
		if (!offset.is_none()) {
			const auto seconds = nb::cast<int64_t>(offset.attr("total_seconds")().attr("__int__")());
			if (seconds > duckdb::cxx::dtime_tz_t::MAX_OFFSET || seconds < -duckdb::cxx::dtime_tz_t::MAX_OFFSET) {
				ThrowInvalidInput("time zone offset is outside the range TIME_TZ can hold");
			}
			return Value::Create(scope, duckdb::cxx::dtime_tz_t(micros, static_cast<int32_t>(seconds)));
		}
		return Value::Create(scope, duckdb::cxx::dtime_t {micros});
	}
	if (nb::isinstance(object, ctx.timedelta_cls)) {
		// timedelta carries no months, so this direction is lossless where the reverse folds months at 30 days. Its
		// days are separate from a rest that is never negative and under a day, so they stay in the days field.
		int64_t days = nb::cast<int64_t>(object.attr("days"));
		int64_t micros =
		    nb::cast<int64_t>(object.attr("seconds")) * 1'000'000 + nb::cast<int64_t>(object.attr("microseconds"));
		// A pandas Timedelta also carries nanoseconds, dropped toward zero as a timedelta64 column drops them,
		// which for a negative duration rounds the rest up.
		int64_t nanos = 0;
		if (days < 0 && nb::hasattr(object, "nanoseconds") && nb::try_cast(object.attr("nanoseconds"), nanos) &&
		    nanos > 0) {
			micros += 1;
			if (micros == 86'400'000'000) {
				days += 1;
				micros = 0;
			}
		}
		if (days > std::numeric_limits<int32_t>::max() || days < std::numeric_limits<int32_t>::min()) {
			ThrowBeyondRange(nb::cast<std::string>(nb::str(object)));
		}
		return Value::Create(scope, duckdb::cxx::interval_t {0, static_cast<int32_t>(days), micros});
	}
	if (nb::isinstance(object, ctx.decimal_cls)) {
		// Through text, since a double loses the scale; the width comes from the value, so nothing is repadded.
		nb::object parts = object.attr("as_tuple")();
		nb::object exponent = parts.attr("exponent");
		if (!nb::isinstance<nb::int_>(exponent)) {
			// NaN and the infinities carry a string exponent and have no DECIMAL counterpart.
			ThrowInvalidInput("cannot bind a non-finite Decimal");
		}
		// A Decimal is digits x 10^exponent, so a positive exponent adds integer places the digit tuple omits.
		const auto power = nb::cast<int>(exponent);
		const auto digits = static_cast<int>(nb::cast<nb::tuple>(parts.attr("digits")).size());
		const auto scale = std::max(0, -power);
		const auto integer_places = digits + std::max(0, power);
		const auto width = std::min(38, std::max(std::max(integer_places, scale), 1));
		if (scale > width) {
			ThrowInvalidInput("Decimal has more fractional digits than DECIMAL can hold");
		}
		return FromText(scope, nb::cast<std::string>(nb::str(object)),
		                scope.ParseType("DECIMAL(" + std::to_string(width) + "," + std::to_string(scale) + ")"));
	}
	if (nb::isinstance(object, ctx.uuid_cls)) {
		return FromText(scope, nb::cast<std::string>(nb::str(object)), scope.CreateType(LogicalTypeId::UUID));
	}
	if (IsNumpyScalar(object, "datetime64")) {
		return NumpyTemporalToValue(scope, object, true, false, precision);
	}
	if (IsNumpyScalar(object, "timedelta64")) {
		return NumpyTemporalToValue(scope, object, false, false, precision);
	}
	ThrowUnsupported(object);
}

template <class SCOPE>
struct Builder {
	SCOPE &scope;
	ConversionContext &ctx;
	TimestampPrecision precision;
};

template <class SCOPE>
Value TakeScalar(Builder<SCOPE> &builder, nb::handle object) {
	return ScalarValue(builder.scope, object, builder.ctx, builder.precision);
}

// The marker type an untyped value takes: a NULL or empty container carries no type, the C API builds no value
// typed as nothing, and no Python value converts to INTEGER, a Python int being a BIGINT or wider, so an INTEGER

struct ConvertedValue {
	Value value;
	bool contains_untyped;
};

// Each untyped place in one of `values` takes the type the first sibling saying anything there gives it; a place a
// sibling has already typed is never changed. Whether a place every sibling leaves untyped remains.
template <class SCOPE>
bool FillUntypedFromSiblings(SCOPE &scope, std::vector<Value> &values, const std::vector<bool> &untyped) {
	if (std::find(untyped.begin(), untyped.end(), true) == untyped.end()) {
		return false;
	}
	// The common case, a NULL beside complete siblings, takes the first one's type without walking it.
	const auto said = std::find(untyped.begin(), untyped.end(), false);
	bool bare = true;
	for (std::size_t i = 0; i < values.size() && bare; i++) {
		bare = !untyped[i] || values[i].IsNull();
	}
	if (said != untyped.end() && bare) {
		const auto type = values[static_cast<std::size_t>(said - untyped.begin())].GetLogicalType();
		for (std::size_t i = 0; i < values.size(); i++) {
			if (untyped[i]) {
				values[i] = Value::CreateNull(scope, type);
			}
		}
		return false;
	}
	// A type already folded in adds nothing, and a long list repeats a few types.
	auto merged = values.front().GetLogicalType();
	std::vector<LogicalType> seen;
	seen.push_back(values.front().GetLogicalType());
	for (std::size_t i = 1; i < values.size(); i++) {
		auto type = values[i].GetLogicalType();
		if (std::any_of(seen.begin(), seen.end(), [&](const LogicalType &known) { return known == type; })) {
			continue;
		}
		if (auto widened = FilledFromSibling(scope, merged, type, true)) {
			merged = std::move(*widened);
		}
		if (seen.size() < 4) {
			seen.push_back(std::move(type));
		}
	}
	// Untyped siblings of one type, the usual case, share their target, which only the untyped places' types decide.
	std::optional<LogicalType> own;
	std::optional<LogicalType> target;
	for (std::size_t i = 0; i < values.size(); i++) {
		if (untyped[i]) {
			auto type = values[i].GetLogicalType();
			if (!own || !(type == *own)) {
				target = FilledFromSibling(scope, type, merged, false);
				own = std::move(type);
			}
			if (target) {
				values[i] = RelabelUntyped(scope, std::move(values[i]), *target);
			}
		}
	}
	return ContainsUntyped(merged);
}

[[noreturn]] void ThrowZoneMix(const char *holder) {
	ThrowInvalidInput(std::string(holder) +
	                  " with and without a time zone, which no one type holds without assuming a time zone");
}

// Siblings that are themselves dates or times, some with a time zone and some without, are refused: the engine
// combines some such pairs by assuming the session's time zone and refuses the rest with an error that does not say
// why. The count of siblings that are not NULL.
template <class PART>
std::size_t RefuseMixedScalars(std::size_t count, const PART &part, const char *holder) {
	std::size_t present = 0;
	bool naive = false;
	bool zoned = false;
	for (std::size_t i = 0; i < count; i++) {
		const Value &value = part(i);
		if (value.IsNull()) {
			continue;
		}
		present++;
		const auto kind = KindOf(value.GetLogicalType().GetTypeId());
		naive = naive || kind == TemporalKind::NAIVE_INSTANT || kind == TemporalKind::NAIVE_TIME;
		zoned = zoned || kind == TemporalKind::ZONED_INSTANT || kind == TemporalKind::ZONED_TIME;
	}
	if (naive && zoned) {
		ThrowZoneMix(holder);
	}
	return present;
}

// The engine cast each of `count` siblings, `part(i)`, to `combined` when it combined them, so none may have taken a
// date or time across a time zone; their own types still say which were without one. With fewer than two siblings
// that are not NULL, each NULL typed as the other, `combined` is the other's own type and nothing is cast.
template <class PART>
void RefuseCrossing(std::size_t count, const PART &part, const LogicalType &combined, const char *holder) {
	if (!ContainsTemporal(combined)) {
		return;
	}
	for (std::size_t i = 0; i < count; i++) {
		const Value &value = part(i);
		if (!value.IsNull() && CastAssumesTimeZone(value.GetLogicalType(), combined)) {
			ThrowZoneMix(holder);
		}
	}
}

// `node` as a query parameter binds it.
template <class SCOPE>
ConvertedValue Build(Builder<SCOPE> &builder, nb::handle node) {
	auto &scope = builder.scope;
	switch (FormOf(node)) {
	case Form::SCALAR: {
		auto value = TakeScalar(builder, node);
		const bool untyped = value.IsNull() && IsUntypedMarker(value.GetLogicalType());
		return {std::move(value), untyped};
	}
	case Form::LIST: {
		std::vector<Value> values;
		std::vector<bool> untyped;
		ForEachElement(node, [&](auto &part) {
			auto built = Build(builder, part);
			values.push_back(std::move(built.value));
			untyped.push_back(built.contains_untyped);
		});
		if (values.empty()) {
			// An empty list still needs an element type and nothing says which, so INTEGER stands in, as for a NULL.
			return {Value::CreateList(scope, scope.CreateType(LogicalTypeId::INTEGER)), true};
		}
		const bool still_untyped = FillUntypedFromSiblings(scope, values, untyped);
		const auto value_at = [&](std::size_t i) -> const Value & {
			return values[i];
		};
		const auto present = RefuseMixedScalars(values.size(), value_at, "a list holds values");
		auto list = Value::CreateList(scope, values);
		if (present > 1) {
			RefuseCrossing(values.size(), value_at, list.GetLogicalType().GetListChildType(), "a list holds values");
		}
		return {std::move(list), still_untyped};
	}
	case Form::STRUCT: {
		std::vector<std::pair<std::string, Value>> named;
		bool untyped = false;
		ForEachField(node, [&](const std::string &name, auto &part) {
			auto built = Build(builder, part);
			untyped = untyped || built.contains_untyped;
			named.emplace_back(name, std::move(built.value));
		});
		return {Value::CreateStruct(scope, named), untyped};
	}
	case Form::MAP:
		break;
	}
	std::vector<Value> keys;
	std::vector<Value> items;
	std::vector<bool> keys_untyped;
	std::vector<bool> items_untyped;
	ForEachEntry(node, [&](auto &key, auto &item) {
		auto built_key = Build(builder, key);
		keys.push_back(std::move(built_key.value));
		keys_untyped.push_back(built_key.contains_untyped);
		auto built_item = Build(builder, item);
		items.push_back(std::move(built_item.value));
		items_untyped.push_back(built_item.contains_untyped);
	});
	// Bitwise, so the items fill even when the keys leave a place untyped.
	const bool untyped =
	    FillUntypedFromSiblings(scope, keys, keys_untyped) | FillUntypedFromSiblings(scope, items, items_untyped);
	std::vector<std::pair<Value, Value>> entries;
	for (std::size_t i = 0; i < keys.size(); i++) {
		entries.emplace_back(std::move(keys[i]), std::move(items[i]));
	}
	const auto key_at = [&](std::size_t i) -> const Value & {
		return entries[i].first;
	};
	const auto item_at = [&](std::size_t i) -> const Value & {
		return entries[i].second;
	};
	const auto keys_present = RefuseMixedScalars(entries.size(), key_at, "a dict holds keys");
	const auto items_present = RefuseMixedScalars(entries.size(), item_at, "a dict holds values");
	auto map = Value::CreateMap(scope, entries);
	const auto type = map.GetLogicalType();
	if (keys_present > 1) {
		RefuseCrossing(entries.size(), key_at, type.GetMapKeyType(), "a dict holds keys");
	}
	if (items_present > 1) {
		RefuseCrossing(entries.size(), item_at, type.GetMapValueType(), "a dict holds values");
	}
	return {std::move(map), untyped};
}

} // namespace

template <class SCOPE>
Value PythonToValue(SCOPE &scope, nb::handle object, ConversionContext &ctx, TimestampPrecision precision,
                    bool *contains_untyped) {
	if (!nb::isinstance<nb::list>(object) && !nb::isinstance<nb::tuple>(object) && !nb::isinstance<nb::dict>(object)) {
		auto value = ScalarValue(scope, object, ctx, precision);
		if (contains_untyped != nullptr) {
			*contains_untyped = value.IsNull() && IsUntypedMarker(value.GetLogicalType());
		}
		return value;
	}
	Builder<SCOPE> builder {scope, ctx, precision};
	auto built = Build(builder, object);
	if (contains_untyped != nullptr) {
		*contains_untyped = built.contains_untyped;
	}
	return std::move(built.value);
}

template Value PythonToValue<Connection>(Connection &, nb::handle, ConversionContext &, TimestampPrecision, bool *);
template Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle, ConversionContext &,
                                                   TimestampPrecision, bool *);

} // namespace duckdb_python
