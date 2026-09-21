//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pyconv.cpp
//
//
//===----------------------------------------------------------------------===//

#include "pyconv.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <cstdint>
#include <string>

namespace duckdb_python {

using duckdb::cxx::LogicalType;
using duckdb::cxx::LogicalTypeId;
using duckdb::cxx::Value;

namespace {

// DuckDB counts from the Unix epoch, Python from year 1, so convert by offsetting instead of by calendar math.

// DuckDB reserves the extremes of each storage type for infinite dates, which Python cannot represent, so they
// clamp to date and datetime min and max as the previous duckdb package did; such a value no longer round-trips.
constexpr int32_t DATE_POSITIVE_INFINITY = 2147483647;
constexpr int32_t DATE_NEGATIVE_INFINITY = -2147483647;
constexpr int64_t TIMESTAMP_POSITIVE_INFINITY = 9223372036854775807LL;
constexpr int64_t TIMESTAMP_NEGATIVE_INFINITY = -9223372036854775807LL;

// DuckDB dates reach far past Python's year 9999, so name the offending value; OverflowError names a day count.
[[noreturn]] void ThrowUnrepresentable(const std::string &what, const std::string &rendered) {
	throw duckdb::cxx::Exception(4001 /* TYPE_CONVERSION */,
	                             "Conversion Error: " + what + " " + rendered +
	                                 " is outside the range Python's datetime can represent");
}

// Scaling the infinity markers would overflow and destroy them, so they pass through in their own unit. A coarse
// unit can hold an instant no microsecond count can, which is unrepresentable rather than a wrapped value.
template <class TEXT>
int64_t MicrosFromUnit(int64_t raw, int64_t multiply, int64_t divide, TEXT &&text) {
	if (raw == TIMESTAMP_POSITIVE_INFINITY || raw == TIMESTAMP_NEGATIVE_INFINITY) {
		return raw;
	}
	if (multiply == 1) {
		return raw / divide;
	}
	int64_t micros;
	if (__builtin_mul_overflow(raw, multiply, &micros)) {
		ThrowUnrepresentable("timestamp", text());
	}
	return micros;
}
// `text` runs only on the failure path, so the bulk converter builds a Value only when one actually fails.
template <class TEXT>
nb::object EpochDate(ConversionContext &ctx, int32_t days, TEXT &&text) {
	if (days == DATE_POSITIVE_INFINITY) {
		return ctx.date_cls.attr("max");
	}
	if (days == DATE_NEGATIVE_INFINITY) {
		return ctx.date_cls.attr("min");
	}
	try {
		return ctx.epoch_date + ctx.timedelta_cls(days, 0, 0);
	} catch (const nb::python_error &) {
		ThrowUnrepresentable("date", text());
	}
}

// A time of day carries no date, so build it by offsetting midnight and dropping the date part.
nb::object TimeFromMicros(ConversionContext &ctx, int64_t micros) {
	return (ctx.epoch_naive + ctx.timedelta_cls(0, 0, micros)).attr("time")();
}

template <class TEXT>
nb::object EpochDateTime(ConversionContext &ctx, int64_t micros, bool utc, TEXT &&text) {
	// The clamped limits keep the column's time zone, since comparing an aware value with a naive one raises.
	if (micros == TIMESTAMP_POSITIVE_INFINITY) {
		nb::object limit = ctx.datetime_cls.attr("max");
		return utc ? limit.attr("replace")(nb::arg("tzinfo") = ctx.timezone_utc) : limit;
	}
	if (micros == TIMESTAMP_NEGATIVE_INFINITY) {
		nb::object limit = ctx.datetime_cls.attr("min");
		return utc ? limit.attr("replace")(nb::arg("tzinfo") = ctx.timezone_utc) : limit;
	}
	const nb::object &epoch = utc ? ctx.epoch_aware : ctx.epoch_naive;
	try {
		return epoch + ctx.timedelta_cls(0, 0, micros);
	} catch (const nb::python_error &) {
		ThrowUnrepresentable("timestamp", text());
	}
}

// For types whose text form is exact but whose binary form is wider than any C integer nanobind converts.
nb::object IntFromText(ConversionContext &ctx, const std::string &text) {
	return ctx.int_cls(text);
}

} // namespace

ConversionContext::ConversionContext() {
	nb::object datetime = nb::module_::import_("datetime");
	date_cls = datetime.attr("date");
	time_cls = datetime.attr("time");
	datetime_cls = datetime.attr("datetime");
	timedelta_cls = datetime.attr("timedelta");
	timezone_cls = datetime.attr("timezone");
	timezone_utc = timezone_cls.attr("utc");
	nb::object decimal = nb::module_::import_("decimal");
	decimal_cls = decimal.attr("Decimal");
	decimal_context = decimal.attr("Context")(nb::arg("prec") = 45);
	uuid_cls = nb::module_::import_("uuid").attr("UUID");
	int_cls = nb::module_::import_("builtins").attr("int");
	two_pow_64 = int_cls("18446744073709551616");
	epoch_date = date_cls(1970, 1, 1);
	epoch_naive = datetime_cls(1970, 1, 1);
	epoch_aware = datetime_cls(1970, 1, 1, 0, 0, 0, 0, timezone_utc);
	one_microsecond = timedelta_cls(0, 0, 1);
}

namespace {

// Whether values of this type land on something Python can hash; lists and dicts cannot be dictionary keys.
bool KeysHashable(const LogicalType &type) {
	switch (type.GetTypeId()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY:
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::MAP:
		return false;
	case LogicalTypeId::UNION: {
		const auto members = type.GetUnionMemberCount();
		for (duckdb::cxx::idx_t i = 0; i < members; i++) {
			if (!KeysHashable(type.GetUnionMemberType(i))) {
				return false;
			}
		}
		return true;
	}
	default:
		return true;
	}
}

} // namespace

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
		return EpochDate(ctx, value.Get<duckdb::cxx::date_t>().days, [&value] { return value.ToText(); });
	case LogicalTypeId::TIME:
		return TimeFromMicros(ctx, value.Get<duckdb::cxx::dtime_t>().micros);
	case LogicalTypeId::TIMESTAMP:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_t>().micros, false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_TZ:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_tz_t>().micros, true, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_SEC:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_s_t>().seconds, 1'000'000, 1, [&value] { return value.ToText(); }), false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_MS:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_ms_t>().millis, 1'000, 1, [&value] { return value.ToText(); }), false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_NS:
		// Python datetime stops at microseconds, so finer digits are dropped on purpose.
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_ns_t>().nanos, 1, 1'000, [&value] { return value.ToText(); }), false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_tz_ns_t>().nanos, 1, 1'000, [&value] { return value.ToText(); }), true, [&value] { return value.ToText(); });
	case LogicalTypeId::TIME_NS:
		// Same microsecond floor as TIMESTAMP_NS.
		return TimeFromMicros(ctx, value.Get<duckdb::cxx::dtime_ns_t>().nanos / 1'000);
	case LogicalTypeId::TIME_TZ: {
		const auto value_tz = value.Get<duckdb::cxx::dtime_tz_t>();
		nb::object tz = ctx.timezone_cls(ctx.timedelta_cls(0, value_tz.GetOffset(), 0));
		return TimeFromMicros(ctx, value_tz.GetMicros()).attr("replace")(nb::arg("tzinfo") = tz);
	}
	case LogicalTypeId::INTERVAL: {
		const auto interval = value.Get<duckdb::cxx::interval_t>();
		// A month has no fixed length, so months fold at 30 days as the previous package did. Lossy on purpose.
		return ctx.timedelta_cls(static_cast<int64_t>(interval.months) * 30 + interval.days, 0,
		                         interval.micros);
	}
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
		// Exact: an integer's text form loses nothing and a Python int has no width limit.
		return IntFromText(ctx, value.ToText());
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
		// A MAP with unhashable keys becomes (key, value) pairs rather than failing, decided per column not row.
		const auto count = value.GetChildCount();
		if (!KeysHashable(type.GetMapKeyType())) {
			nb::list pairs;
			for (duckdb::cxx::idx_t i = 0; i + 1 < count; i += 2) {
				pairs.append(nb::make_tuple(ValueToPython(value.GetChild(i), ctx),
				                            ValueToPython(value.GetChild(i + 1), ctx)));
			}
			return pairs;
		}
		nb::dict out;
		for (duckdb::cxx::idx_t i = 0; i + 1 < count; i += 2) {
			out[ValueToPython(value.GetChild(i), ctx)] = ValueToPython(value.GetChild(i + 1), ctx);
		}
		return out;
	}
	case LogicalTypeId::UNION:
		// Child 0 is the tag, child 1 the active member.
		return ValueToPython(value.GetChild(1), ctx);
	default:
		// BIT, BIGNUM, VARIANT, GEOMETRY and whatever a later DuckDB adds: the SQL text beats failing the fetch.
		return nb::cast(value.ToText());
	}
}

namespace {

namespace cxx = duckdb::cxx;

// A decimal's exact text, point placed by the scale, so Decimal(text) keeps both the value and the scale.
std::string DecimalText(int64_t raw, uint8_t scale) {
	const bool negative = raw < 0;
	const auto magnitude = negative ? ~static_cast<uint64_t>(raw) + 1 : static_cast<uint64_t>(raw);
	std::string digits = std::to_string(magnitude);
	if (scale > 0) {
		if (digits.size() <= scale) {
			digits.insert(0, scale + 1 - digits.size(), '0');
		}
		digits.insert(digits.size() - scale, ".");
	}
	return negative ? "-" + digits : digits;
}

// A 128-bit value as an exact Python int, upper * 2^64 + lower, the caller having converted the upper half.
nb::object CombineLimbs(ConversionContext &ctx, nb::object upper, uint64_t lower) {
	if (!upper.is_valid()) {
		throw nb::python_error();
	}
	nb::object shifted = nb::steal(PyNumber_Multiply(upper.ptr(), ctx.two_pow_64.ptr()));
	if (!shifted.is_valid()) {
		throw nb::python_error();
	}
	nb::object low = nb::steal(PyLong_FromUnsignedLongLong(lower));
	if (!low.is_valid()) {
		throw nb::python_error();
	}
	nb::object combined = nb::steal(PyNumber_Add(shifted.ptr(), low.ptr()));
	if (!combined.is_valid()) {
		throw nb::python_error();
	}
	return combined;
}

/// Stores converted elements for a parent to assemble rows from.
struct VectorSink {
	std::vector<nb::object> &out;

	void operator()(size_t position, PyObject *object) const {
		out[position] = nb::steal(object);
	}
};

// Elements [first, last) of one column, each handed to `sink(position, object)` as a new reference with
// `position` relative to `first`. Limited-API calls only, since the extension builds against the stable ABI.
template <class SINK>
void EmitElements(cxx::Vector &vector, const LogicalType &type, cxx::idx_t first, cxx::idx_t last,
                  ConversionContext &ctx, SINK &&sink) {
	using Id = LogicalTypeId;
	// Compact encodings expand to plain values plus a NULL mask, the one layout the reads below can serve, and
	// element indices then equal row indices, which the nested cases rely on for their child ranges.
	vector.Flatten();
	const auto view = vector.GetView();
	const auto typed = [&](auto convert) {
		for (cxx::idx_t e = first; e < last; e++) {
			const auto element = view.SelAt(e);
			PyObject *object = nullptr;
			if (!view.RowIsValid(element)) {
				object = Py_NewRef(Py_None);
			} else {
				object = convert(element);
				if (object == nullptr) {
					throw nb::python_error();
				}
			}
			sink(static_cast<size_t>(e - first), object);
		}
	};
	switch (type.GetTypeId()) {
	case Id::BOOLEAN:
		typed([&](cxx::idx_t i) { return Py_NewRef(view.Data<bool>()[i] ? Py_True : Py_False); });
		break;
	case Id::TINYINT:
		typed([&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int8_t>()[i]); });
		break;
	case Id::SMALLINT:
		typed([&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int16_t>()[i]); });
		break;
	case Id::INTEGER:
		typed([&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int32_t>()[i]); });
		break;
	case Id::BIGINT:
		typed([&](cxx::idx_t i) { return PyLong_FromLongLong(view.Data<int64_t>()[i]); });
		break;
	case Id::UTINYINT:
		typed([&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint8_t>()[i]); });
		break;
	case Id::USMALLINT:
		typed([&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint16_t>()[i]); });
		break;
	case Id::UINTEGER:
		typed([&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint32_t>()[i]); });
		break;
	case Id::UBIGINT:
		typed([&](cxx::idx_t i) { return PyLong_FromUnsignedLongLong(view.Data<uint64_t>()[i]); });
		break;
	case Id::FLOAT:
		typed([&](cxx::idx_t i) { return PyFloat_FromDouble(view.Data<float>()[i]); });
		break;
	case Id::DOUBLE:
		typed([&](cxx::idx_t i) { return PyFloat_FromDouble(view.Data<double>()[i]); });
		break;
	case Id::VARCHAR:
		typed([&](cxx::idx_t i) {
			const auto &text = view.Data<cxx::varchar_t>()[i];
			return PyUnicode_FromStringAndSize(text.data(), text.size());
		});
		break;
	case Id::BLOB:
		typed([&](cxx::idx_t i) {
			const auto &blob = view.Data<cxx::blob_t>()[i];
			return PyBytes_FromStringAndSize(blob.data(), blob.size());
		});
		break;
	case Id::DATE:
		typed([&](cxx::idx_t i) {
			return EpochDate(ctx, view.Data<cxx::date_t>()[i].days, [&] { return vector.GetValue(i).ToText(); })
			    .release()
			    .ptr();
		});
		break;
	case Id::TIME:
		typed([&](cxx::idx_t i) {
			return TimeFromMicros(ctx, view.Data<cxx::dtime_t>()[i].micros).release().ptr();
		});
		break;
	case Id::TIMESTAMP:
	case Id::TIMESTAMP_TZ:
	case Id::TIMESTAMP_SEC:
	case Id::TIMESTAMP_MS:
	case Id::TIMESTAMP_NS:
	case Id::TIMESTAMP_TZ_NS: {
		const auto id = type.GetTypeId();
		const bool utc = id == Id::TIMESTAMP_TZ || id == Id::TIMESTAMP_TZ_NS;
		typed([&](cxx::idx_t i) {
			const auto raw = view.Data<int64_t>()[i];
			const auto text = [&] { return vector.GetValue(i).ToText(); };
			// Nanosecond columns floor to microseconds as the per-value path does; markers keep their unit.
			const auto micros = id == Id::TIMESTAMP_SEC ? MicrosFromUnit(raw, 1'000'000, 1, text)
			                    : id == Id::TIMESTAMP_MS ? MicrosFromUnit(raw, 1'000, 1, text)
			                    : id == Id::TIMESTAMP_NS || id == Id::TIMESTAMP_TZ_NS
			                        ? MicrosFromUnit(raw, 1, 1'000, text)
			                        : raw;
			return EpochDateTime(ctx, micros, utc, text).release().ptr();
		});
		break;
	}
	case Id::HUGEINT:
		typed([&](cxx::idx_t i) {
			const auto &limbs = view.Data<cxx::int128_t>()[i];
			return CombineLimbs(ctx, nb::steal(PyLong_FromLongLong(limbs.upper)), limbs.lower).release().ptr();
		});
		break;
	case Id::UHUGEINT:
		typed([&](cxx::idx_t i) {
			const auto &limbs = view.Data<cxx::uint128_t>()[i];
			return CombineLimbs(ctx, nb::steal(PyLong_FromUnsignedLongLong(limbs.upper)), limbs.lower)
			    .release()
			    .ptr();
		});
		break;
	case Id::UUID:
		typed([&](cxx::idx_t i) {
			const auto decoded = view.Data<cxx::uuid_t>()[i].Decode();
			nb::bytes canonical(reinterpret_cast<const char *>(decoded.bytes), sizeof(decoded.bytes));
			return ctx.uuid_cls(nb::arg("bytes") = canonical).release().ptr();
		});
		break;
	case Id::DECIMAL: {
		const auto width = type.GetDecimalWidth();
		const auto scale = static_cast<uint8_t>(type.GetDecimalScale());
		if (width > 18) {
			typed([&](cxx::idx_t i) {
				const auto &limbs = view.Data<cxx::int128_t>()[i];
				nb::object unscaled = CombineLimbs(ctx, nb::steal(PyLong_FromLongLong(limbs.upper)), limbs.lower);
				// Exact: Decimal(int) never rounds and the wide context covers every digit 128 bits can hold.
				return ctx.decimal_cls(unscaled)
				    .attr("scaleb")(-static_cast<int>(scale), ctx.decimal_context)
				    .release()
				    .ptr();
			});
			break;
		}
		typed([&](cxx::idx_t i) {
			const int64_t raw = width <= 4   ? view.Data<int16_t>()[i]
			                    : width <= 9 ? view.Data<int32_t>()[i]
			                                 : view.Data<int64_t>()[i];
			return ctx.decimal_cls(DecimalText(raw, scale)).release().ptr();
		});
		break;
	}
	case Id::ENUM: {
		// The labels become Python strings once; each row is then one new reference into that list.
		const auto size = type.GetEnumSize();
		std::vector<nb::object> dictionary;
		dictionary.reserve(size);
		for (cxx::idx_t v = 0; v < size; v++) {
			dictionary.push_back(nb::cast(type.GetEnumValue(v)));
		}
		typed([&](cxx::idx_t i) {
			const auto code = size < 256     ? static_cast<cxx::idx_t>(view.Data<uint8_t>()[i])
			                  : size < 65536 ? static_cast<cxx::idx_t>(view.Data<uint16_t>()[i])
			                                 : static_cast<cxx::idx_t>(view.Data<uint32_t>()[i]);
			return Py_NewRef(dictionary.at(code).ptr());
		});
		break;
	}
	case Id::LIST:
	case Id::MAP: {
		// Both index child columns by offset and length; only the slice the served rows reference is converted.
		const auto *entries = view.Data<cxx::list_entry_t>();
		auto lo = std::numeric_limits<uint64_t>::max();
		uint64_t hi = 0;
		for (cxx::idx_t e = first; e < last; e++) {
			const auto element = view.SelAt(e);
			if (view.RowIsValid(element)) {
				lo = std::min(lo, entries[element].offset);
				hi = std::max(hi, entries[element].offset + entries[element].length);
			}
		}
		if (lo > hi) {
			lo = hi = 0;
		}
		const bool is_map = type.GetTypeId() == Id::MAP;
		std::vector<nb::object> keys(is_map ? hi - lo : 0);
		std::vector<nb::object> values(hi - lo);
		if (hi > lo) {
			if (is_map) {
				auto key_vector = vector.GetChild(0);
				auto value_vector = vector.GetChild(1);
				const auto key_type = type.GetMapKeyType();
				const auto value_type = type.GetMapValueType();
				EmitElements(key_vector, key_type, lo, hi, ctx, VectorSink {keys});
				EmitElements(value_vector, value_type, lo, hi, ctx, VectorSink {values});
			} else {
				auto child = vector.GetChild(0);
				const auto child_type = type.GetListChildType();
				EmitElements(child, child_type, lo, hi, ctx, VectorSink {values});
			}
		}
		// Unhashable keys become (key, value) pairs, decided from the type so every row of the column matches.
		const bool hashable = is_map && KeysHashable(type.GetMapKeyType());
		for (cxx::idx_t e = first; e < last; e++) {
			const auto element = view.SelAt(e);
			if (!view.RowIsValid(element)) {
				sink(static_cast<size_t>(e - first), Py_NewRef(Py_None));
				continue;
			}
			const auto &entry = entries[element];
			const auto base = entry.offset - lo;
			nb::object row;
			if (!is_map) {
				row = nb::steal(PyList_New(static_cast<Py_ssize_t>(entry.length)));
				if (!row.is_valid()) {
					throw nb::python_error();
				}
				for (uint64_t j = 0; j < entry.length; j++) {
					if (PyList_SetItem(row.ptr(), static_cast<Py_ssize_t>(j),
					                   Py_NewRef(values[base + j].ptr())) != 0) {
						throw nb::python_error();
					}
				}
			} else if (hashable) {
				row = nb::steal(PyDict_New());
				if (!row.is_valid()) {
					throw nb::python_error();
				}
				for (uint64_t j = 0; j < entry.length; j++) {
					if (PyDict_SetItem(row.ptr(), keys[base + j].ptr(), values[base + j].ptr()) != 0) {
						throw nb::python_error();
					}
				}
			} else {
				row = nb::steal(PyList_New(static_cast<Py_ssize_t>(entry.length)));
				if (!row.is_valid()) {
					throw nb::python_error();
				}
				for (uint64_t j = 0; j < entry.length; j++) {
					nb::object pair = nb::steal(PyTuple_New(2));
					if (!pair.is_valid()) {
						throw nb::python_error();
					}
					if (PyTuple_SetItem(pair.ptr(), 0, Py_NewRef(keys[base + j].ptr())) != 0 ||
					    PyTuple_SetItem(pair.ptr(), 1, Py_NewRef(values[base + j].ptr())) != 0) {
						throw nb::python_error();
					}
					if (PyList_SetItem(row.ptr(), static_cast<Py_ssize_t>(j), pair.release().ptr()) != 0) {
						throw nb::python_error();
					}
				}
			}
			sink(static_cast<size_t>(e - first), row.release().ptr());
		}
		break;
	}
	case Id::ARRAY: {
		const auto size = type.GetArraySize();
		auto child = vector.GetChild(0);
		const auto child_type = type.GetArrayChildType();
		std::vector<nb::object> elements(static_cast<size_t>(last - first) * size);
		if (!elements.empty()) {
			EmitElements(child, child_type, first * size, last * size, ctx, VectorSink {elements});
		}
		for (cxx::idx_t e = first; e < last; e++) {
			const auto element = view.SelAt(e);
			if (!view.RowIsValid(element)) {
				sink(static_cast<size_t>(e - first), Py_NewRef(Py_None));
				continue;
			}
			nb::object row = nb::steal(PyList_New(static_cast<Py_ssize_t>(size)));
			if (!row.is_valid()) {
				throw nb::python_error();
			}
			const auto base = static_cast<size_t>(e - first) * size;
			for (cxx::idx_t j = 0; j < size; j++) {
				if (PyList_SetItem(row.ptr(), static_cast<Py_ssize_t>(j),
				                   Py_NewRef(elements[base + j].ptr())) != 0) {
					throw nb::python_error();
				}
			}
			sink(static_cast<size_t>(e - first), row.release().ptr());
		}
		break;
	}
	case Id::STRUCT: {
		const auto fields = type.GetStructChildCount();
		std::vector<std::vector<nb::object>> columns(fields);
		std::vector<nb::object> names(fields);
		for (cxx::idx_t f = 0; f < fields; f++) {
			columns[f].resize(static_cast<size_t>(last - first));
			auto child = vector.GetChild(f);
			const auto field_type = type.GetStructChildType(f);
			EmitElements(child, field_type, first, last, ctx, VectorSink {columns[f]});
			names[f] = nb::cast(type.GetStructChildName(f));
		}
		for (cxx::idx_t e = first; e < last; e++) {
			const auto element = view.SelAt(e);
			if (!view.RowIsValid(element)) {
				sink(static_cast<size_t>(e - first), Py_NewRef(Py_None));
				continue;
			}
			nb::object row = nb::steal(PyDict_New());
			if (!row.is_valid()) {
				throw nb::python_error();
			}
			for (cxx::idx_t f = 0; f < fields; f++) {
				if (PyDict_SetItem(row.ptr(), names[f].ptr(), columns[f][e - first].ptr()) != 0) {
					throw nb::python_error();
				}
			}
			sink(static_cast<size_t>(e - first), row.release().ptr());
		}
		break;
	}
	default:
		// UNION, BIT, BIGNUM, VARIANT, TIME_TZ, TIME_NS and later additions go one value at a time.
		typed([&](cxx::idx_t i) { return ValueToPython(vector.GetValue(i), ctx).release().ptr(); });
		break;
	}
}

} // namespace

void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out) {
	const auto columns = chunk.GetVectorCount();
	const auto rows = end - start;
	// Held here so an exception part way through releases them; a tuple with empty slots deallocates fine.
	std::vector<nb::object> tuples;
	tuples.reserve(rows);
	for (cxx::idx_t r = 0; r < rows; r++) {
		PyObject *row = PyTuple_New(static_cast<Py_ssize_t>(columns));
		if (row == nullptr) {
			throw nb::python_error();
		}
		tuples.emplace_back(nb::steal(row));
	}
	for (cxx::idx_t c = 0; c < columns; c++) {
		auto vector = chunk.GetVector(c);
		EmitElements(vector, types.at(c), start, end, ctx, [&](size_t position, PyObject *object) {
			// SetItem steals `object` whatever it returns.
			if (PyTuple_SetItem(tuples[position].ptr(), static_cast<Py_ssize_t>(c), object) != 0) {
				throw nb::python_error();
			}
		});
	}
	for (auto &row : tuples) {
		out.append(row);
	}
}

namespace {

using duckdb::cxx::Connection;

// date(1970, 1, 1).toordinal(), since Python counts days from year 1 and DuckDB from the epoch.
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
	throw UnsupportedTypeException(
	    nb::cast<std::string>(nb::handle(Py_TYPE(object.ptr())).attr("__name__")));
}

// The message is carried twice, with and without the prefix, since DuckDB adds its own inside a callback.
[[noreturn]] void ThrowInvalidInput(const std::string &body) {
	throw duckdb::cxx::InvalidInputException("Invalid Input Error: " + body, body);
}

} // namespace

UnsupportedTypeException::UnsupportedTypeException(std::string type_name)
    : duckdb::cxx::InvalidInputException("Invalid Input Error: cannot bind a parameter of type " + type_name,
                                         "cannot bind a parameter of type " + type_name),
      type_name(std::move(type_name)) {
}

template <class SCOPE>
Value PythonToValue(SCOPE &scope, nb::handle object, ConversionContext &ctx) {
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
		return Value::Create(scope, duckdb::cxx::blob_t(bytes.c_str(),
		                                                    static_cast<uint32_t>(bytes.size())));
	}
	// datetime before date: datetime subclasses date, so order decides.
	if (nb::isinstance(object, ctx.datetime_cls)) {
		const bool aware = !object.attr("tzinfo").is_none();
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
		if (!offset.is_none()) {
			const auto seconds =
			    nb::cast<int64_t>(offset.attr("total_seconds")().attr("__int__")());
			if (seconds > duckdb::cxx::dtime_tz_t::MAX_OFFSET ||
			    seconds < -duckdb::cxx::dtime_tz_t::MAX_OFFSET) {
				ThrowInvalidInput("time zone offset is outside the range TIME_TZ can hold");
			}
			return Value::Create(scope,
			                     duckdb::cxx::dtime_tz_t(micros, static_cast<int32_t>(seconds)));
		}
		return Value::Create(scope, duckdb::cxx::dtime_t {micros});
	}
	if (nb::isinstance(object, ctx.timedelta_cls)) {
		// timedelta carries no months, so this direction is lossless where the reverse folds months at 30 days.
		duckdb::cxx::interval_t interval {};
		interval.months = 0;
		interval.days = nb::cast<int32_t>(object.attr("days"));
		interval.micros = nb::cast<int64_t>(object.attr("seconds")) * 1'000'000 +
		                  nb::cast<int64_t>(object.attr("microseconds"));
		return Value::Create(scope, interval);
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
		                scope.ParseType("DECIMAL(" + std::to_string(width) + "," +
		                                     std::to_string(scale) + ")"));
	}
	if (nb::isinstance(object, ctx.uuid_cls)) {
		return FromText(scope, nb::cast<std::string>(nb::str(object)),
		                scope.CreateType(LogicalTypeId::UUID));
	}
	if (nb::isinstance<nb::list>(object) || nb::isinstance<nb::tuple>(object)) {
		std::vector<Value> children;
		for (nb::handle item : object) {
			children.push_back(PythonToValue(scope, item, ctx));
		}
		if (children.empty()) {
			// An empty list still needs an element type and nothing says which, so INTEGER stands in again.
			return Value::CreateList(scope, scope.CreateType(LogicalTypeId::INTEGER));
		}
		return Value::CreateList(scope, children);
	}
	if (nb::isinstance<nb::dict>(object)) {
		// A dict maps onto two DuckDB types, so the rule is fixed: string keys make a STRUCT, anything else a MAP.
		auto mapping = nb::cast<nb::dict>(object);
		bool all_strings = true;
		for (auto entry : mapping) {
			if (!nb::isinstance<nb::str>(entry.first)) {
				all_strings = false;
				break;
			}
		}
		if (all_strings) {
			std::vector<std::pair<std::string, Value>> fields;
			for (auto entry : mapping) {
				fields.emplace_back(nb::cast<std::string>(entry.first),
				                    PythonToValue(scope, entry.second, ctx));
			}
			return Value::CreateStruct(scope, fields);
		}
		std::vector<std::pair<Value, Value>> entries;
		for (auto entry : mapping) {
			entries.emplace_back(PythonToValue(scope, entry.first, ctx),
			                     PythonToValue(scope, entry.second, ctx));
		}
		return Value::CreateMap(scope, entries);
	}
	ThrowUnsupported(object);
}

template Value PythonToValue<Connection>(Connection &, nb::handle, ConversionContext &);
template Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle, ConversionContext &);

nb::list VectorElements(duckdb::cxx::Vector &vector, const LogicalType &type, duckdb::cxx::idx_t first,
                        duckdb::cxx::idx_t last, ConversionContext &ctx) {
	PyObject *list = PyList_New(static_cast<Py_ssize_t>(last - first));
	if (list == nullptr) {
		throw nb::python_error();
	}
	// Empty slots are fine for list dealloc, so an exception part way through still releases what was made.
	auto out = nb::steal<nb::list>(list);
	EmitElements(vector, type, first, last, ctx, [&](size_t position, PyObject *object) {
		// SetItem steals `object` whatever it returns.
		if (PyList_SetItem(list, static_cast<Py_ssize_t>(position), object) != 0) {
			throw nb::python_error();
		}
	});
	return out;
}

} // namespace duckdb_python
