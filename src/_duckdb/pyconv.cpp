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

// DuckDB counts days and microseconds from the Unix epoch; Python's date and
// datetime count from year 1. Convert by offsetting from the epoch rather than
// reimplementing the calendar.

// DuckDB reserves the extremes of the storage type for the infinite dates,
// which have no Python counterpart. The previous client clamped them to
// date/datetime min and max, and the adopted test suite expects that, so the
// behaviour deliberately carries over: an infinite date returned this way no
// longer round-trips as infinite.
constexpr int32_t DATE_POSITIVE_INFINITY = 2147483647;
constexpr int32_t DATE_NEGATIVE_INFINITY = -2147483647;
constexpr int64_t TIMESTAMP_POSITIVE_INFINITY = 9223372036854775807LL;
constexpr int64_t TIMESTAMP_NEGATIVE_INFINITY = -9223372036854775807LL;

// Python's date stops at year 9999 while DuckDB reaches year 5874897, so a
// value can be perfectly valid in the engine and unrepresentable here. Report
// that as a conversion error rather than letting datetime raise OverflowError,
// whose message names an internal day count and not the column.
[[noreturn]] void ThrowUnrepresentable(const std::string &what, const std::string &rendered) {
	throw duckdb::cxx::Exception(4001 /* TYPE_CONVERSION */,
	                             "Conversion Error: " + what + " " + rendered +
	                                 " is outside the range Python's datetime can represent");
}
// `text` renders the offending value for the error message and runs only on
// that path, so the bulk row converter can defer building a Value until a
// value actually fails.
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

// A time of day is a microsecond offset with no date, so build it by offsetting
// midnight and dropping the date part.
nb::object TimeFromMicros(ConversionContext &ctx, int64_t micros) {
	return (ctx.epoch_naive + ctx.timedelta_cls(0, 0, micros)).attr("time")();
}

template <class TEXT>
nb::object EpochDateTime(ConversionContext &ctx, int64_t micros, bool utc, TEXT &&text) {
	// Keep the infinities as aware as the column they came from: mixing an
	// aware value with a naive one raises TypeError on comparison.
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

// For the types whose text form is exact but whose binary form has no Python
// counterpart: HUGEINT is wider than any C integer nanobind converts.
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
	decimal_cls = nb::module_::import_("decimal").attr("Decimal");
	uuid_cls = nb::module_::import_("uuid").attr("UUID");
	int_cls = nb::module_::import_("builtins").attr("int");
	epoch_date = date_cls(1970, 1, 1);
	epoch_naive = datetime_cls(1970, 1, 1);
	epoch_aware = datetime_cls(1970, 1, 1, 0, 0, 0, 0, timezone_utc);
	one_microsecond = timedelta_cls(0, 0, 1);
}

namespace {

// Whether values of this type convert to something Python can hash. A LIST
// or ARRAY becomes a list, a STRUCT or MAP a dict, and a UNION whatever its
// active member is.
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
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_s_t>().seconds * 1'000'000, false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_MS:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_ms_t>().millis * 1'000, false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_NS:
		// Python datetime resolves to microseconds, so sub-microsecond digits
		// are dropped. A deliberate divergence, not a rounding bug.
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_ns_t>().nanos / 1'000, false, [&value] { return value.ToText(); });
	case LogicalTypeId::TIMESTAMP_TZ_NS:
		return EpochDateTime(ctx, value.Get<duckdb::cxx::timestamp_tz_ns_t>().nanos / 1'000, true, [&value] { return value.ToText(); });
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
		// A month is not a fixed duration, so this mapping is lossy for any
		// interval carrying months. A deliberate divergence: the previous
		// client folded months at 30 days and this matches it.
		return ctx.timedelta_cls(static_cast<int64_t>(interval.months) * 30 + interval.days, 0,
		                         interval.micros);
	}
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UHUGEINT:
		// Exact: the text form of an integer loses nothing, and Python ints are
		// arbitrary precision.
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
		// A dict, when the key type allows it. A MAP keyed by a LIST or a
		// STRUCT has keys that arrive as lists and dicts, which cannot be
		// hashed; such a map becomes a list of (key, value) pairs instead of
		// failing the whole fetch. Decided from the type, so every map in a
		// column, the empty ones included, comes back in the same shape.
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
		// BIT, BIGNUM, VARIANT, GEOMETRY, and anything a newer engine adds.
		// Degrading to the SQL text keeps unknown types readable instead of
		// failing the whole fetch.
		return nb::cast(value.ToText());
	}
}

namespace {

// Writes rows [start, start + tuples.size()) of one column into the pre-made
// tuples, dispatching on the type once and reading the flattened view
// directly. `convert` returns a NEW reference, or nullptr with a Python error
// set. Stable-ABI calls only: this extension builds under Py_LIMITED_API.
template <class CONVERT>
void FillColumn(std::vector<nb::object> &tuples, const duckdb::cxx::VectorView &view, duckdb::cxx::idx_t start,
                duckdb::cxx::idx_t column, CONVERT &&convert) {
	const auto rows = tuples.size();
	for (size_t r = 0; r < rows; r++) {
		const auto element = view.SelAt(start + r);
		PyObject *object = nullptr;
		if (!view.RowIsValid(element)) {
			object = Py_NewRef(Py_None);
		} else {
			object = convert(element);
			if (object == nullptr) {
				throw nb::python_error();
			}
		}
		// SetItem steals `object` whatever it returns.
		if (PyTuple_SetItem(tuples[r].ptr(), static_cast<Py_ssize_t>(column), object) != 0) {
			throw nb::python_error();
		}
	}
}

// The exact text of an int64-tier decimal: digits with the point placed by
// the scale, so Decimal(text) preserves both value and scale.
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

} // namespace

void AppendChunkRows(const duckdb::cxx::DataChunk &chunk, const std::vector<LogicalType> &types,
                     duckdb::cxx::idx_t start, duckdb::cxx::idx_t end, ConversionContext &ctx, nb::list &out) {
	namespace cxx = duckdb::cxx;
	using Id = LogicalTypeId;
	const auto columns = chunk.GetVectorCount();
	const auto rows = end - start;
	// The tuples are built first and filled column by column, held here so an
	// exception mid-fill releases them (tuple dealloc accepts empty slots).
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
		// Dictionary, constant and other encodings become flat data plus
		// validity, the one layout the typed reads below can serve.
		vector.Flatten();
		const auto view = vector.GetView();
		const auto &type = types.at(c);
		// Nested, HUGEINT, UUID, wide DECIMAL, the exotic strings: one cell
		// at a time through the per-value path.
		const auto fallback = [&](cxx::idx_t i) { return ValueToPython(vector.GetValue(i), ctx).release().ptr(); };
		switch (type.GetTypeId()) {
		case Id::BOOLEAN:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return Py_NewRef(view.Data<bool>()[i] ? Py_True : Py_False); });
			break;
		case Id::TINYINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int8_t>()[i]); });
			break;
		case Id::SMALLINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int16_t>()[i]); });
			break;
		case Id::INTEGER:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromLong(view.Data<int32_t>()[i]); });
			break;
		case Id::BIGINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromLongLong(view.Data<int64_t>()[i]); });
			break;
		case Id::UTINYINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint8_t>()[i]); });
			break;
		case Id::USMALLINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint16_t>()[i]); });
			break;
		case Id::UINTEGER:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromUnsignedLong(view.Data<uint32_t>()[i]); });
			break;
		case Id::UBIGINT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyLong_FromUnsignedLongLong(view.Data<uint64_t>()[i]); });
			break;
		case Id::FLOAT:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyFloat_FromDouble(view.Data<float>()[i]); });
			break;
		case Id::DOUBLE:
			FillColumn(tuples, view, start, c,
			           [&](cxx::idx_t i) { return PyFloat_FromDouble(view.Data<double>()[i]); });
			break;
		case Id::VARCHAR:
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				const auto &text = view.Data<cxx::varchar_t>()[i];
				return PyUnicode_FromStringAndSize(text.data(), text.size());
			});
			break;
		case Id::BLOB:
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				const auto &blob = view.Data<cxx::blob_t>()[i];
				return PyBytes_FromStringAndSize(blob.data(), blob.size());
			});
			break;
		case Id::DATE:
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				return EpochDate(ctx, view.Data<cxx::date_t>()[i].days,
				                 [&] { return vector.GetValue(i).ToText(); })
				    .release()
				    .ptr();
			});
			break;
		case Id::TIME:
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
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
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				const auto raw = view.Data<int64_t>()[i];
				// The sub-microsecond floor of the ns variants matches the
				// per-value path above.
				const auto micros = id == Id::TIMESTAMP_SEC                            ? raw * 1'000'000
				                    : id == Id::TIMESTAMP_MS                           ? raw * 1'000
				                    : id == Id::TIMESTAMP_NS || id == Id::TIMESTAMP_TZ_NS ? raw / 1'000
				                                                                          : raw;
				return EpochDateTime(ctx, micros, utc, [&] { return vector.GetValue(i).ToText(); })
				    .release()
				    .ptr();
			});
			break;
		}
		case Id::DECIMAL: {
			const auto width = type.GetDecimalWidth();
			if (width > 18) {
				// The int128 tier's text form needs 128-bit division.
				FillColumn(tuples, view, start, c, fallback);
				break;
			}
			const auto scale = static_cast<uint8_t>(type.GetDecimalScale());
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				const int64_t raw = width <= 4   ? view.Data<int16_t>()[i]
				                    : width <= 9 ? view.Data<int32_t>()[i]
				                                 : view.Data<int64_t>()[i];
				return ctx.decimal_cls(DecimalText(raw, scale)).release().ptr();
			});
			break;
		}
		case Id::ENUM: {
			// The dictionary becomes Python strings once; every row is then
			// one new reference into it.
			const auto size = type.GetEnumSize();
			std::vector<nb::object> dictionary;
			dictionary.reserve(size);
			for (cxx::idx_t v = 0; v < size; v++) {
				dictionary.push_back(nb::cast(type.GetEnumValue(v)));
			}
			FillColumn(tuples, view, start, c, [&](cxx::idx_t i) {
				const auto code = size < 256     ? static_cast<cxx::idx_t>(view.Data<uint8_t>()[i])
				                  : size < 65536 ? static_cast<cxx::idx_t>(view.Data<uint16_t>()[i])
				                                 : static_cast<cxx::idx_t>(view.Data<uint32_t>()[i]);
				return Py_NewRef(dictionary.at(code).ptr());
			});
			break;
		}
		default:
			FillColumn(tuples, view, start, c, fallback);
			break;
		}
	}
	for (auto &row : tuples) {
		out.append(row);
	}
}

namespace {

using duckdb::cxx::Connection;

// date(1970, 1, 1).toordinal(); Python counts days from year 1, DuckDB from
// the epoch.
constexpr int64_t EPOCH_ORDINAL = 719163;

// Whole microseconds between two Python datetimes, using Python's own
// arithmetic rather than total_seconds(), which is a float and loses precision
// on timestamps far from the epoch.
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

// Round-trip a value through its text form and let the engine parse it. Exact
// for integers of any width and for decimals, neither of which has a runtime
// C++ type here to build directly.
Value FromText(Connection &connection, const std::string &text, const LogicalType &target) {
	const duckdb::cxx::varchar_t borrowed(text);
	return Value::Create(connection, borrowed).Cast(connection, target);
}

[[noreturn]] void ThrowUnsupported(nb::handle object) {
	const auto name = nb::cast<std::string>(nb::handle(Py_TYPE(object.ptr())).attr("__name__"));
	throw duckdb::cxx::InvalidInputException("Invalid Input Error: cannot bind a parameter of type " +
	                                         name);
}

} // namespace

Value PythonToValue(Connection &connection, nb::handle object, ConversionContext &ctx) {
	using duckdb::cxx::LogicalTypeId;

	// A NULL still needs a type to carry it. SQLNULL would be the honest
	// choice but create_type_from_id rejects it, so INTEGER stands in: NULL
	// casts from any type to any other, so the binder still lands on whatever
	// the statement wants.
	if (object.is_none()) {
		return Value::CreateNull(connection, connection.CreateType(LogicalTypeId::INTEGER));
	}
	// Before the int branch: a Python bool IS an int, so testing int first
	// would silently bind True as 1.
	if (nb::isinstance<nb::bool_>(object)) {
		return Value::Create(connection, nb::cast<bool>(object));
	}
	if (nb::isinstance<nb::int_>(object)) {
		// Widest type that holds the value; the binder narrows from there.
		// Choosing narrowly here would reject values the column can hold.
		int64_t narrow = 0;
		if (nb::try_cast<int64_t>(object, narrow)) {
			return Value::Create(connection, narrow);
		}
		const auto text = nb::cast<std::string>(nb::str(object));
		// The C++ API pins C++17, so no starts_with here.
		const bool negative = !text.empty() && text[0] == '-';
		const auto id = negative ? LogicalTypeId::HUGEINT : LogicalTypeId::UHUGEINT;
		return FromText(connection, text, connection.CreateType(id));
	}
	if (nb::isinstance<nb::float_>(object)) {
		return Value::Create(connection, nb::cast<double>(object));
	}
	if (nb::isinstance<nb::str>(object)) {
		const auto text = nb::cast<std::string>(object);
		return Value::Create(connection, duckdb::cxx::varchar_t(text));
	}
	if (nb::isinstance<nb::bytes>(object)) {
		const auto bytes = nb::cast<nb::bytes>(object);
		// blob_t carries a uint32 length, so anything larger would wrap and
		// bind silently truncated. Every other overflow here throws.
		if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
			throw duckdb::cxx::InvalidInputException(
			    "Invalid Input Error: bytes value is larger than a BLOB can hold");
		}
		return Value::Create(connection, duckdb::cxx::blob_t(bytes.c_str(),
		                                                    static_cast<uint32_t>(bytes.size())));
	}
	// datetime before date: datetime subclasses date, so order decides.
	if (nb::isinstance(object, ctx.datetime_cls)) {
		const bool aware = !object.attr("tzinfo").is_none();
		const int64_t micros = MicrosSince(aware ? ctx.epoch_aware : ctx.epoch_naive, object, ctx);
		if (aware) {
			return Value::Create(connection, duckdb::cxx::timestamp_tz_t {micros});
		}
		return Value::Create(connection, duckdb::cxx::timestamp_t {micros});
	}
	if (nb::isinstance(object, ctx.date_cls)) {
		const auto days = nb::cast<int64_t>(object.attr("toordinal")()) - EPOCH_ORDINAL;
		return Value::Create(connection, duckdb::cxx::date_t {static_cast<int32_t>(days)});
	}
	if (nb::isinstance(object, ctx.time_cls)) {
		nb::object naive = object.attr("replace")(nb::arg("tzinfo") = nb::none());
		nb::object combined = ctx.datetime_cls.attr("combine")(ctx.epoch_date, naive);
		const int64_t micros = MicrosSince(ctx.epoch_naive, combined, ctx);

		// An aware time binds as TIME_TZ. Dropping the offset here would be a
		// silent loss, and the read direction already returns TIME_TZ aware, so
		// the round trip has to keep it.
		nb::object offset = object.attr("utcoffset")();
		if (!offset.is_none()) {
			const auto seconds =
			    nb::cast<int64_t>(offset.attr("total_seconds")().attr("__int__")());
			if (seconds > duckdb::cxx::dtime_tz_t::MAX_OFFSET ||
			    seconds < -duckdb::cxx::dtime_tz_t::MAX_OFFSET) {
				throw duckdb::cxx::InvalidInputException(
				    "Invalid Input Error: time zone offset is outside the range TIME_TZ can hold");
			}
			return Value::Create(connection,
			                     duckdb::cxx::dtime_tz_t(micros, static_cast<int32_t>(seconds)));
		}
		return Value::Create(connection, duckdb::cxx::dtime_t {micros});
	}
	if (nb::isinstance(object, ctx.timedelta_cls)) {
		// timedelta carries no months, so this direction is lossless; the
		// reverse is not, which is why months are folded at 30 days there.
		duckdb::cxx::interval_t interval {};
		interval.months = 0;
		interval.days = nb::cast<int32_t>(object.attr("days"));
		interval.micros = nb::cast<int64_t>(object.attr("seconds")) * 1'000'000 +
		                  nb::cast<int64_t>(object.attr("microseconds"));
		return Value::Create(connection, interval);
	}
	if (nb::isinstance(object, ctx.decimal_cls)) {
		// Via text: there is no runtime decimal type to build, and going
		// through double would lose the scale that makes it a Decimal.
		//
		// The width and scale come from the value itself. A fixed DECIMAL(38,10)
		// would silently repad, turning Decimal("123.456") into
		// Decimal("123.4560000000") and losing the scale the caller chose.
		nb::object parts = object.attr("as_tuple")();
		nb::object exponent = parts.attr("exponent");
		if (!nb::isinstance<nb::int_>(exponent)) {
			// NaN and the infinities carry a string exponent and have no
			// DECIMAL counterpart at all.
			throw duckdb::cxx::InvalidInputException(
			    "Invalid Input Error: cannot bind a non-finite Decimal");
		}
		// A Decimal is digits x 10^exponent. A negative exponent is the scale;
		// a positive one adds trailing zeroes the digit tuple does not carry,
		// so Decimal("1E+2") needs three integer places, not one.
		const auto power = nb::cast<int>(exponent);
		const auto digits = static_cast<int>(nb::cast<nb::tuple>(parts.attr("digits")).size());
		const auto scale = std::max(0, -power);
		const auto integer_places = digits + std::max(0, power);
		const auto width = std::min(38, std::max(std::max(integer_places, scale), 1));
		if (scale > width) {
			throw duckdb::cxx::InvalidInputException(
			    "Invalid Input Error: Decimal has more fractional digits than DECIMAL can hold");
		}
		return FromText(connection, nb::cast<std::string>(nb::str(object)),
		                connection.ParseType("DECIMAL(" + std::to_string(width) + "," +
		                                     std::to_string(scale) + ")"));
	}
	if (nb::isinstance(object, ctx.uuid_cls)) {
		return FromText(connection, nb::cast<std::string>(nb::str(object)),
		                connection.CreateType(LogicalTypeId::UUID));
	}
	if (nb::isinstance<nb::list>(object) || nb::isinstance<nb::tuple>(object)) {
		std::vector<Value> children;
		for (nb::handle item : object) {
			children.push_back(PythonToValue(connection, item, ctx));
		}
		if (children.empty()) {
			// An empty list still needs a child type, and nothing in the value
			// says which. INTEGER for the same reason a bare NULL takes it.
			return Value::CreateList(connection, connection.CreateType(LogicalTypeId::INTEGER));
		}
		return Value::CreateList(connection, children);
	}
	if (nb::isinstance<nb::dict>(object)) {
		// A dict is the only Python type that maps onto two DuckDB types, so
		// the rule is stated rather than guessed per call: string keys read as
		// a STRUCT, anything else as a MAP.
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
				                    PythonToValue(connection, entry.second, ctx));
			}
			return Value::CreateStruct(connection, fields);
		}
		std::vector<std::pair<Value, Value>> entries;
		for (auto entry : mapping) {
			entries.emplace_back(PythonToValue(connection, entry.first, ctx),
			                     PythonToValue(connection, entry.second, ctx));
		}
		return Value::CreateMap(connection, entries);
	}
	ThrowUnsupported(object);
}

} // namespace duckdb_python
