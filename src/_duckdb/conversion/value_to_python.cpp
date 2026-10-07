//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/value_to_python.cpp
//
//
//===----------------------------------------------------------------------===//

#include "value_to_python.hpp"

#include "conversion.hpp"
#include "sql_types.hpp"
#include "temporal.hpp"

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

namespace {

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

bool ConvertsLossless(LogicalTypeId type) {
	switch (type) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::UUID:
	case LogicalTypeId::ENUM:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_TZ:
		return true;
	default:
		return false;
	}
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
		return EpochDate(ctx, value.Get<duckdb::cxx::date_t>().days, [&value] { return value.ToText(); });
	case LogicalTypeId::TIME:
		return TimeFromMicros(ctx, value.Get<duckdb::cxx::dtime_t>().micros);
	case LogicalTypeId::TIMESTAMP:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_t>().micros, false,
		                     [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_TZ:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_tz_t>().micros, true,
		                     [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_SEC:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_s_t>().seconds, 1'000'000, 1), false,
		                     [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_MS:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_ms_t>().millis, 1'000, 1), false,
		                     [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_NS:
		// Python datetime stops at microseconds, so finer digits are dropped on purpose.
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_ns_t>().nanos, 1, 1'000), false,
		                     [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return EpochDateTime(ctx, MicrosFromUnit(value.Get<duckdb::cxx::timestamp_tz_ns_t>().nanos, 1, 1'000), true,
		                     [&value] { return value.ToText(); });
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
		try {
			return ctx.timedelta_cls(static_cast<int64_t>(interval.months) * 30 + interval.days, 0, interval.micros);
		} catch (const nb::python_error &) {
			ThrowUnrepresentable("interval", value.ToText(), "timedelta");
		}
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
				pairs.append(
				    nb::make_tuple(ValueToPython(value.GetChild(i), ctx), ValueToPython(value.GetChild(i + 1), ctx)));
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
		typed([&](cxx::idx_t i) { return TimeFromMicros(ctx, view.Data<cxx::dtime_t>()[i].micros).release().ptr(); });
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
			const auto text = [&] {
				return vector.GetValue(i).ToText();
			};
			// Nanosecond columns floor to microseconds as the per-value path does; markers keep their unit.
			const auto micros = id == Id::TIMESTAMP_SEC  ? MicrosFromUnit(raw, 1'000'000, 1)
			                    : id == Id::TIMESTAMP_MS ? MicrosFromUnit(raw, 1'000, 1)
			                    : id == Id::TIMESTAMP_NS || id == Id::TIMESTAMP_TZ_NS ? MicrosFromUnit(raw, 1, 1'000)
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
			return CombineLimbs(ctx, nb::steal(PyLong_FromUnsignedLongLong(limbs.upper)), limbs.lower).release().ptr();
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
				// A MAP's one child is its entries, a STRUCT(key, value); key and value sit under it.
				auto map_entries_vector = vector.GetChild(0);
				auto key_vector = map_entries_vector.GetChild(0);
				auto value_vector = map_entries_vector.GetChild(1);
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
					if (PyList_SetItem(row.ptr(), static_cast<Py_ssize_t>(j), Py_NewRef(values[base + j].ptr())) != 0) {
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
				if (PyList_SetItem(row.ptr(), static_cast<Py_ssize_t>(j), Py_NewRef(elements[base + j].ptr())) != 0) {
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
