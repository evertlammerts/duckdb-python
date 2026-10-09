//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/numpy_scan.cpp
//
//
//===----------------------------------------------------------------------===//

#include "numpy_scan.hpp"

#include "arrowc.hpp"
#include "conversion/conversion.hpp"
#include "conversion/python_to_value.hpp"
#include "conversion/temporal.hpp"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A registered object is a source from duckdb/_sources that answers two calls: describe() names its columns and their
// engine types, and columns() answers each requested column as an encoding, an engine type, a numpy data array and a
// numpy mask array or None. A pandas DataFrame and a numpy array are such sources. The encoding says how a row's bytes
// turn into a vector element below. A column held as Arrow is the exception: describe() gives an Arrow schema capsule
// in place of its type, and columns() an Arrow stream capsule in place of its arrays, which core's Arrow importer
// reads, so core decides its engine type as it does for any Arrow data. One run of this table function over a query is
// a scan, and its threads read the object in ranges of rows they claim in turn. An object array can hold pandas' NA
// and NaT singletons, so they stay among the missing markers this file checks for.

namespace duckdb_python {
namespace {

struct NumpyScanUserData {
	std::shared_ptr<Registry> registry;
	std::shared_ptr<ModuleState> module;
	/// The engine's standard vector size, which bounds how many rows one exec call fills.
	cxx::idx_t batch_rows;
};

/// The entry a query bound over, and the column names and engine types `describe()` gave, in declared order: the
/// types declared when the query was bound, which every scan of it must fill.
struct NumpyScanBindData {
	NumpyScanBindData(std::shared_ptr<Registered> entry, std::vector<std::string> names,
	                  std::vector<cxx::LogicalType> types, cxx::idx_t rows)
	    : entry(std::move(entry)), names(std::move(names)), types(std::move(types)), rows(rows) {
	}

	/// Freed from engine threads too, so the Python reference is dropped under the GIL.
	~NumpyScanBindData() {
		FencedGil gil;
		entry.reset();
	}

	std::shared_ptr<Registered> entry;
	std::vector<std::string> names;
	std::vector<cxx::LogicalType> types;
	cxx::idx_t rows;
};

/// A column's encoding: how the scan turns its data array's bytes into the column's engine type. `TIMESTAMP` and
/// `INTERVAL` data are int64 counts in a unit the encoding names.
enum class NumpyEncoding : uint8_t {
	FIXED,
	TIMESTAMP,
	INTERVAL,
	ENUM_CODES,
	TEXT,
	OBJECTS,
	UCS4,
	BYTES,
	ARROW,
};

/// Releases a buffer protocol view and frees the structure holding it; runs under the GIL, when opening a later
/// column fails at global init and at the scan's teardown.
struct ReleaseBuffer {
	void operator()(Py_buffer *view) const {
		PyBuffer_Release(view);
		delete view;
	}
};

/// A buffer protocol view of an array, holding a reference to the array until released. The view lives on the heap
/// and only the pointer moves, since an exporter such as `bytes` points the view's shape and strides into the view
/// itself.
using BufferView = std::unique_ptr<Py_buffer, ReleaseBuffer>;

/// An Arrow array shared by the scan and every chunk imported from a view of it; the last share releases it, from
/// whichever thread drops it, since the importer's chunks may outlive the scan. The arrays come from pyarrow, whose
/// buffers over Python objects take the GIL themselves when freed.
using SharedArray = std::shared_ptr<ArrowOwned<ArrowArray>>;

/// Where an Arrow type records its nulls: in a validity bitmap, the first buffer; nowhere, for a union or a run-end
/// encoded array, whose first buffer if any is not a bitmap; or in the type itself, the null type's every row.
enum class ArrowNulls : uint8_t { BITMAP, NONE, EVERY_ROW };

ArrowNulls NullsOfFormat(const char *format) {
	const std::string_view spelled(format != nullptr ? format : "");
	if (spelled == "n") {
		return ArrowNulls::EVERY_ROW;
	}
	if (spelled == "+r" || spelled.rfind("+u", 0) == 0) {
		return ArrowNulls::NONE;
	}
	return ArrowNulls::BITMAP;
}

/// An ARROW column's data: the one-column batch schema each thread's importer is resolved against, which building an
/// importer only reads, where its type records nulls, and the stream's arrays, empty ones left out, with the first row
/// of each and the row count after the last in `starts`.
struct ArrowColumn {
	ArrowOwned<ArrowSchema> schema;
	ArrowNulls nulls = ArrowNulls::BITMAP;
	std::vector<SharedArray> arrays;
	std::vector<cxx::idx_t> starts;
};

constexpr CountRange kEveryCount {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};

/// One requested column, kept for the scan's life: the rule its bytes are read by, the engine type it fills, and
/// views of its data array and, when it has one, its mask array, or for an ARROW column its Arrow data.
struct NumpyColumn {
	std::string name;
	NumpyEncoding encoding;
	/// How a TIMESTAMP or INTERVAL column's counts become the engine's; unused otherwise.
	UnitConversion conversion;
	/// The counts a TIMESTAMP column's converted values must lie in; unused otherwise.
	CountRange range = kEveryCount;
	cxx::LogicalType type;
	BufferView data;
	BufferView mask;
	ArrowColumn arrow;
};

/// The Python objects an object column's cells are recognised by, looked up once per scan under the GIL.
struct ScalarMarkers {
	/// pandas' NA and NaT singletons; null handles, matching no cell, when pandas is not loaded.
	nb::object na;
	nb::object nat;
	/// numpy's scalar base class and the scalar classes whose missing value is not equal to itself.
	nb::object numpy_generic;
	nb::object numpy_floating;
	nb::object numpy_datetime64;
	nb::object numpy_timedelta64;
};

/// One scan's columns, the row count they cover, and the claim counter every thread divides the object with.
struct NumpyScanState {
	NumpyScanState(std::vector<NumpyColumn> columns, cxx::idx_t rows, cxx::idx_t range_rows, ScalarMarkers markers)
	    : columns(std::move(columns)), rows(rows), range_rows(range_rows), markers(std::move(markers)) {
	}

	/// Torn down from an engine thread, so every buffer and Python reference is released under the GIL.
	~NumpyScanState() {
		FencedGil gil;
		columns.clear();
		markers = ScalarMarkers();
	}

	std::vector<NumpyColumn> columns;
	cxx::idx_t rows;
	/// Rows claimed together by one thread: 50 batches' worth, so a thread does not pay the claim's cost every batch.
	cxx::idx_t range_rows;
	std::atomic<cxx::idx_t> next {0};
	ScalarMarkers markers;
};

/// One thread's claimed range within the object, and the ordering position that range stands for. Each ARROW column
/// has this thread's own importer, since an importer is single threaded, and the chunk its output vector references,
/// kept until the next batch replaces it; both are empty for every other column.
struct NumpyScanLocalState {
	explicit NumpyScanLocalState(std::vector<std::optional<cxx::ArrowImporter>> importers)
	    : importers(std::move(importers)), held(this->importers.size()) {
	}

	cxx::idx_t start = 0;
	cxx::idx_t end = 0;
	cxx::idx_t batch_index = 0;
	std::vector<std::optional<cxx::ArrowImporter>> importers;
	std::vector<std::optional<cxx::DataChunk>> held;
};

/// The engine type core imports the one Arrow column `schema` describes as, `schema` rewrapped as the one-column batch
/// an importer reads. Always wrapped, since the column may itself be a struct.
cxx::LogicalType ArrowColumnType(cxx::Context &context, ArrowSchema &schema, cxx::idx_t batch_rows) {
	WrapAsBatch(schema);
	cxx::ArrowImporter importer(context, schema, batch_rows);
	return importer.GetSchema().GetFieldType(0);
}

/// A column's engine type as `describe()` gave it: its text, or an Arrow schema capsule for a column held as Arrow.
cxx::LogicalType DescribedType(cxx::Context &context, const Registered &entry, nb::handle described,
                               cxx::idx_t batch_rows) {
	if (nb::isinstance<nb::str>(described)) {
		return context.ParseType(nb::cast<std::string>(described));
	}
	auto schema =
	    TakeFromCapsule<ArrowSchema>(described, kSchemaCapsule, "the object registered as '" + entry.name + "'");
	return ArrowColumnType(context, schema.value, batch_rows);
}

/// Resolved once per query while bound, so a scan sees the object registered under the name when it binds.
void NumpyScanBind(cxx::TableFunction::BindInput &input) {
	auto &registry = *input.GetUserData<NumpyScanUserData>().registry;
	const auto name = std::string(input.GetConstantArgument(0).Get<cxx::varchar_t>());
	auto entry = registry.ByName(name);
	if (!entry) {
		throw cxx::InvalidInputException("nothing is registered as '" + name + "'");
	}
	std::vector<std::string> names;
	std::vector<cxx::LogicalType> types;
	cxx::idx_t rows = 0;
	const auto batch_rows = input.GetUserData<NumpyScanUserData>().batch_rows;
	auto context = input.GetContext();
	{
		FencedGil gil;
		try {
			nb::object description = entry->object.attr("describe")();
			for (nb::handle item : description) {
				auto [column_name, described] = nb::cast<std::pair<std::string, nb::object>>(item);
				auto type = DescribedType(context, *entry, described, batch_rows);
				input.AddResultColumn(column_name, type);
				names.push_back(std::move(column_name));
				types.push_back(std::move(type));
			}
			rows = nb::cast<cxx::idx_t>(entry->object.attr("rows")());
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("describing the object registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		}
	}
	input.SetCardinality(rows, true);
	input.SetBindData<NumpyScanBindData>(std::move(entry), std::move(names), std::move(types), rows);
}

/// The encoding `columns()` named and, for `timestamp:<unit>` or `interval:<unit>`, that unit and its step; `unit`
/// stays null for every other encoding.
NumpyEncoding ParseEncoding(const std::string &registered_name, const std::string &column_name,
                            const std::string &spelling, const TimeUnit *&unit, uint64_t &step) {
	const std::string_view spelled(spelling);
	unit = nullptr;
	step = 1;
	if (spelling == "fixed") {
		return NumpyEncoding::FIXED;
	}
	if (spelled.rfind("timestamp:", 0) == 0 && ParseTimeUnit(spelled.substr(10), unit, step)) {
		return NumpyEncoding::TIMESTAMP;
	}
	if (spelled.rfind("interval:", 0) == 0 && ParseTimeUnit(spelled.substr(9), unit, step)) {
		return NumpyEncoding::INTERVAL;
	}
	if (spelling == "enum") {
		return NumpyEncoding::ENUM_CODES;
	}
	if (spelling == "text") {
		return NumpyEncoding::TEXT;
	}
	if (spelling == "objects") {
		return NumpyEncoding::OBJECTS;
	}
	if (spelling == "ucs4") {
		return NumpyEncoding::UCS4;
	}
	if (spelling == "bytes") {
		return NumpyEncoding::BYTES;
	}
	if (spelling == "arrow") {
		return NumpyEncoding::ARROW;
	}
	throw cxx::InvalidInputException("the object registered as '" + registered_name + "' answered columns() for '" +
	                                 column_name + "' with the unknown encoding '" + spelling + "'");
}

/// How a TIMESTAMP or INTERVAL column's counts in `unit` become counts of the declared `type`: seconds, milliseconds,
/// microseconds or nanoseconds for a naive timestamp, microseconds or nanoseconds for a zoned one, microseconds for
/// an interval. A timestamp only ever multiplies, since the source chose a type at least as fine as its unit; only an
/// interval may drop precision, the engine having none finer than microseconds.
UnitConversion TimeConversion(const std::string &registered_name, const std::string &column_name,
                              const std::string &encoding_text, const TimeUnit &unit, uint64_t step,
                              const cxx::LogicalType &type) {
	uint64_t target_nanos = 1'000;
	bool exact = true;
	switch (type.GetTypeId()) {
	case cxx::LogicalTypeId::TIMESTAMP_SEC:
		target_nanos = 1'000'000'000;
		break;
	case cxx::LogicalTypeId::TIMESTAMP_MS:
		target_nanos = 1'000'000;
		break;
	case cxx::LogicalTypeId::TIMESTAMP_NS:
	case cxx::LogicalTypeId::TIMESTAMP_TZ_NS:
		target_nanos = 1;
		break;
	case cxx::LogicalTypeId::TIMESTAMP:
	case cxx::LogicalTypeId::TIMESTAMP_TZ:
		break;
	default:
		exact = false;
		break;
	}
	const auto refuse = [&](const std::string &why) {
		return cxx::InvalidInputException("the object registered as '" + registered_name +
		                                  "' answered columns() for '" + column_name + "' with the encoding '" +
		                                  encoding_text + "', " + why + " " + type.ToText());
	};
	const auto conversion = ConversionOf(unit, step, target_nanos);
	if (exact && conversion.denominator != 1) {
		throw refuse("whose unit is finer than");
	}
	return conversion;
}

/// What a buffer's elements are, as far as the scan needs to know.
enum class ElementClass : uint8_t { SIGNED, UNSIGNED, FLOAT, BOOLEAN, OBJECT, UCS4, BYTES, OTHER };

/// The class of a buffer's elements, from the struct module format the buffer protocol reports: an optional
/// byte-order prefix, an optional repeat count, and one code. A repeat count belongs only to numpy's fixed-width
/// strings, whose element is the whole string; on any other code it makes the element an array, which is not read.
ElementClass ClassOf(const char *format) {
	std::string_view text = format != nullptr ? format : "B";
	if (!text.empty() && std::string_view("@=<>!").find(text.front()) != std::string_view::npos) {
		text.remove_prefix(1);
	}
	const auto code = text.find_first_not_of("0123456789");
	if (code == std::string_view::npos || code + 1 != text.size()) {
		return ElementClass::OTHER;
	}
	switch (text[code]) {
	case 'w':
		return ElementClass::UCS4;
	case 's':
		return ElementClass::BYTES;
	default:
		break;
	}
	if (code > 0) {
		return ElementClass::OTHER;
	}
	switch (text[code]) {
	case 'b':
	case 'h':
	case 'i':
	case 'l':
	case 'q':
	case 'n':
		return ElementClass::SIGNED;
	case 'B':
	case 'H':
	case 'I':
	case 'L':
	case 'Q':
	case 'N':
		return ElementClass::UNSIGNED;
	case 'e':
	case 'f':
	case 'd':
	case 'g':
		return ElementClass::FLOAT;
	case '?':
		return ElementClass::BOOLEAN;
	case 'O':
		return ElementClass::OBJECT;
	default:
		return ElementClass::OTHER;
	}
}

/// The element a `"fixed"` column of `type` is copied from, class and width; OTHER for a type not stored as a plain
/// number.
std::pair<ElementClass, cxx::idx_t> FixedElement(const cxx::LogicalType &type) {
	switch (type.GetTypeId()) {
	case cxx::LogicalTypeId::BOOLEAN:
		return {ElementClass::BOOLEAN, 1};
	case cxx::LogicalTypeId::TINYINT:
		return {ElementClass::SIGNED, 1};
	case cxx::LogicalTypeId::SMALLINT:
		return {ElementClass::SIGNED, 2};
	case cxx::LogicalTypeId::INTEGER:
	case cxx::LogicalTypeId::DATE:
		return {ElementClass::SIGNED, 4};
	case cxx::LogicalTypeId::BIGINT:
		return {ElementClass::SIGNED, 8};
	case cxx::LogicalTypeId::UTINYINT:
		return {ElementClass::UNSIGNED, 1};
	case cxx::LogicalTypeId::USMALLINT:
		return {ElementClass::UNSIGNED, 2};
	case cxx::LogicalTypeId::UINTEGER:
		return {ElementClass::UNSIGNED, 4};
	case cxx::LogicalTypeId::UBIGINT:
		return {ElementClass::UNSIGNED, 8};
	case cxx::LogicalTypeId::FLOAT:
		return {ElementClass::FLOAT, 4};
	case cxx::LogicalTypeId::DOUBLE:
		return {ElementClass::FLOAT, 8};
	default:
		return {ElementClass::OTHER, 0};
	}
}

/// Whether a data buffer of `element`s `itemsize` bytes wide is what `encoding` reads into a column of `type`, the
/// type declared when the query was bound. Each fill function below writes `type`'s own layout and reads its buffer
/// as this allows, so this one check keeps both the reads and the writes inside their memory.
bool BufferFits(NumpyEncoding encoding, const cxx::LogicalType &type, ElementClass element, cxx::idx_t itemsize) {
	const auto id = type.GetTypeId();
	const bool int64 = element == ElementClass::SIGNED && itemsize == sizeof(int64_t);
	switch (encoding) {
	case NumpyEncoding::FIXED: {
		const auto fixed = FixedElement(type);
		return fixed.first != ElementClass::OTHER && element == fixed.first && itemsize == fixed.second;
	}
	case NumpyEncoding::TIMESTAMP:
		return int64 && (id == cxx::LogicalTypeId::TIMESTAMP_SEC || id == cxx::LogicalTypeId::TIMESTAMP_MS ||
		                 id == cxx::LogicalTypeId::TIMESTAMP || id == cxx::LogicalTypeId::TIMESTAMP_NS ||
		                 id == cxx::LogicalTypeId::TIMESTAMP_TZ || id == cxx::LogicalTypeId::TIMESTAMP_TZ_NS);
	case NumpyEncoding::INTERVAL:
		return int64 && id == cxx::LogicalTypeId::INTERVAL;
	case NumpyEncoding::ENUM_CODES:
		return id == cxx::LogicalTypeId::ENUM && element == ElementClass::SIGNED &&
		       (itemsize == 1 || itemsize == 2 || itemsize == 4);
	case NumpyEncoding::TEXT:
		return id == cxx::LogicalTypeId::VARCHAR && element == ElementClass::OBJECT && itemsize == sizeof(void *);
	case NumpyEncoding::OBJECTS:
		return element == ElementClass::OBJECT && itemsize == sizeof(void *);
	case NumpyEncoding::UCS4:
		return id == cxx::LogicalTypeId::VARCHAR && element == ElementClass::UCS4 && itemsize > 0 && itemsize % 4 == 0;
	case NumpyEncoding::BYTES:
		return id == cxx::LogicalTypeId::BLOB && element == ElementClass::BYTES && itemsize > 0;
	case NumpyEncoding::ARROW:
		return false;
	}
	return false;
}

bool HostIsLittleEndian() {
	const uint16_t probe = 1;
	return *reinterpret_cast<const uint8_t *>(&probe) == 1;
}

/// A strided buffer view of `obj`, refusing one that is not one-dimensional, the wrong length, in the other byte order,
/// or whose elements `fits` refuses; `role` names what `obj` is and `refusal` ends the message for the last case.
template <class FITS>
BufferView OpenBuffer(const std::string &registered_name, const std::string &column_name, const char *role,
                      nb::handle obj, cxx::idx_t rows, FITS fits, const std::string &refusal) {
	const auto refuse = [&](const std::string &what) {
		return cxx::InvalidInputException("the object registered as '" + registered_name +
		                                  "' answered columns() for '" + column_name + "' with a " + role + " array " +
		                                  what);
	};
	auto acquired = std::make_unique<Py_buffer>();
	if (PyObject_GetBuffer(obj.ptr(), acquired.get(), PyBUF_FORMAT | PyBUF_STRIDES) != 0) {
		PyErr_Clear();
		throw refuse("that exports no strided buffer");
	}
	BufferView buffer(acquired.release());
	const auto &view = *buffer;
	if (view.ndim != 1) {
		throw refuse("that is not one-dimensional");
	}
	if (static_cast<cxx::idx_t>(view.shape[0]) != rows) {
		throw refuse("of " + std::to_string(view.shape[0]) + " rows where " + std::to_string(rows) + " were expected");
	}
	// The bytes are copied as they are, so a buffer in the other byte order would read as garbage.
	const char *format = view.format != nullptr ? view.format : "";
	if (HostIsLittleEndian() ? (format[0] == '>' || format[0] == '!') : format[0] == '<') {
		throw refuse("in the other byte order, which is not read");
	}
	if (!fits(ClassOf(view.format), static_cast<cxx::idx_t>(view.itemsize))) {
		throw refuse("of elements in the buffer format '" + std::string(format) + "', " +
		             std::to_string(view.itemsize) + " bytes wide, " + refusal);
	}
	return buffer;
}

/// Row `row` of a one-dimensional strided view. numpy's `buf` addresses element 0 whatever the stride's sign, and a
/// stride of 0 (a broadcast array) repeats that element.
const uint8_t *Element(const BufferView &buffer, cxx::idx_t row) {
	return static_cast<const uint8_t *>(buffer->buf) + static_cast<Py_ssize_t>(row) * buffer->strides[0];
}

/// Read through memcpy, since a strided view over a packed record array need not be aligned for `T`.
template <class T>
T Load(const BufferView &buffer, cxx::idx_t row) {
	T value;
	std::memcpy(&value, Element(buffer, row), sizeof(T));
	return value;
}

/// `count` elements from row `start` of `view`, packed into `dest`: one copy for a contiguous run, one per element
/// otherwise.
void CopyRows(const BufferView &buffer, cxx::idx_t start, cxx::idx_t count, void *dest) {
	const auto width = static_cast<size_t>(buffer->itemsize);
	auto *out = static_cast<uint8_t *>(dest);
	if (buffer->strides[0] == buffer->itemsize) {
		std::memcpy(out, Element(buffer, start), count * width);
		return;
	}
	for (cxx::idx_t i = 0; i < count; i++) {
		std::memcpy(out + i * width, Element(buffer, start + i), width);
	}
}

/// Whether the mask marks row `row` of `column` missing; false for a column without a mask.
bool Masked(const NumpyColumn &column, cxx::idx_t row) {
	return column.mask && *Element(column.mask, row) != 0;
}

/// An ARROW column's stream capsule, drained into `arrow` and refused unless it holds `rows` rows; answers the engine
/// type core imports the stream's schema as.
cxx::LogicalType OpenArrow(const std::string &registered_name, const std::string &column_name, cxx::Context &context,
                           nb::handle capsule, cxx::idx_t rows, cxx::idx_t batch_rows, ArrowColumn &arrow) {
	const auto refuse = [&](const std::string &what) {
		return cxx::InvalidInputException("the object registered as '" + registered_name +
		                                  "' answered columns() for '" + column_name + "' with an Arrow stream " +
		                                  what);
	};
	auto stream = TakeFromCapsule<ArrowArrayStream>(capsule, kStreamCapsule,
	                                                "the object registered as '" + registered_name + "'");
	if (stream.value.get_schema(&stream.value, &arrow.schema.value) != 0) {
		throw refuse("whose schema could not be read: " + StreamError(stream.value));
	}
	auto type = ArrowColumnType(context, arrow.schema.value, batch_rows);
	arrow.nulls = NullsOfFormat(arrow.schema.value.children[0]->format);
	arrow.starts.push_back(0);
	for (;;) {
		ArrowOwned<ArrowArray> array;
		if (stream.value.get_next(&stream.value, &array.value) != 0) {
			throw refuse("that failed: " + StreamError(stream.value));
		}
		if (!array) {
			break;
		}
		if (array.value.length == 0) {
			continue;
		}
		arrow.starts.push_back(arrow.starts.back() + static_cast<cxx::idx_t>(array.value.length));
		arrow.arrays.push_back(std::make_shared<ArrowOwned<ArrowArray>>(std::move(array)));
	}
	if (arrow.starts.back() != rows) {
		throw refuse("of " + std::to_string(arrow.starts.back()) + " rows where " + std::to_string(rows) +
		             " were expected");
	}
	return type;
}

/// `columns(requested)`'s answer, checked against the query's bound names and types and opened into `NumpyColumn`s;
/// every view already opened is released before an error escapes, so a later column's refusal never leaks an earlier
/// one's.
std::vector<NumpyColumn> OpenColumns(const Registered &entry, const NumpyScanBindData &bound, cxx::Context &context,
                                     const std::vector<cxx::idx_t> &requested, nb::handle answer,
                                     cxx::idx_t batch_rows) {
	if (static_cast<cxx::idx_t>(nb::len(answer)) != requested.size()) {
		throw cxx::InvalidInputException("the object registered as '" + entry.name + "' answered columns() with " +
		                                 std::to_string(nb::len(answer)) + " columns where " +
		                                 std::to_string(requested.size()) + " were requested");
	}
	std::vector<NumpyColumn> columns;
	columns.reserve(requested.size());
	for (cxx::idx_t i = 0; i < requested.size(); i++) {
		auto [column_name, encoding_text, type_text, data_array, mask_array] =
		    nb::cast<std::tuple<std::string, std::string, nb::object, nb::object, nb::object>>(answer[i]);
		const auto declared = requested[i];
		if (column_name != bound.names.at(declared)) {
			throw cxx::InvalidInputException(
			    "the object registered as '" + entry.name + "' answered columns() with column '" + column_name +
			    "' at position " + std::to_string(i) + " where '" + bound.names.at(declared) + "' was expected");
		}
		const TimeUnit *unit = nullptr;
		uint64_t step = 1;
		const auto encoding = ParseEncoding(entry.name, column_name, encoding_text, unit, step);
		ArrowColumn arrow;
		// The object is read afresh for every scan, so it may have changed since the query was bound.
		auto type = encoding == NumpyEncoding::ARROW
		                ? OpenArrow(entry.name, column_name, context, data_array, bound.rows, batch_rows, arrow)
		                : context.ParseType(nb::cast<std::string>(type_text));
		if (type != bound.types.at(declared)) {
			throw cxx::InvalidInputException("the object registered as '" + entry.name + "' answered columns() for '" +
			                                 column_name + "' with the type " + type.ToText() +
			                                 ", but the query was bound when it was " +
			                                 bound.types.at(declared).ToText());
		}
		if (encoding == NumpyEncoding::ARROW) {
			columns.push_back(NumpyColumn {std::move(column_name), encoding, UnitConversion {1, 1, 1}, kEveryCount,
			                               std::move(type), BufferView(), BufferView(), std::move(arrow)});
			continue;
		}
		const auto data_fits = [&](ElementClass element, cxx::idx_t width) {
			return BufferFits(encoding, type, element, width);
		};
		auto data = OpenBuffer(entry.name, column_name, "data", data_array, bound.rows, data_fits,
		                       "which the encoding '" + encoding_text + "' does not read into " + type.ToText());
		const auto mask_fits = [](ElementClass element, cxx::idx_t width) {
			return width == 1 && (element == ElementClass::BOOLEAN || element == ElementClass::SIGNED ||
			                      element == ElementClass::UNSIGNED);
		};
		auto mask = mask_array.is_none() ? BufferView()
		                                 : OpenBuffer(entry.name, column_name, "mask", mask_array, bound.rows,
		                                              mask_fits, "where a mask holds one byte per row");
		UnitConversion conversion {1, 1, 1};
		CountRange range = kEveryCount;
		if (encoding == NumpyEncoding::TIMESTAMP || encoding == NumpyEncoding::INTERVAL) {
			conversion = TimeConversion(entry.name, column_name, encoding_text, *unit, step, type);
		}
		if (encoding == NumpyEncoding::TIMESTAMP) {
			range = FiniteTimestampRange(type.GetTypeId());
		}
		columns.push_back(NumpyColumn {std::move(column_name), encoding, conversion, range, std::move(type),
		                               std::move(data), std::move(mask), ArrowColumn()});
	}
	return columns;
}

void NumpyScanInitGlobal(cxx::TableFunction::InitGlobalInput &input) {
	const auto &bound = input.GetBindData<NumpyScanBindData>();
	auto &entry = *bound.entry;
	const auto declared = static_cast<cxx::idx_t>(bound.names.size());
	std::vector<cxx::idx_t> requested;
	bool identity = input.GetColumnCount() == declared;
	for (cxx::idx_t i = 0; i < input.GetColumnCount(); i++) {
		requested.push_back(input.GetColumnIndex(i));
		identity = identity && requested.back() == i;
	}

	const auto batch_rows = input.GetUserData<NumpyScanUserData>().batch_rows;
	const cxx::idx_t range_rows = std::max<cxx::idx_t>(1, batch_rows) * 50;
	input.SetMaxThreads(std::max<cxx::idx_t>(1, (bound.rows + range_rows - 1) / range_rows));

	// One GIL scope up to the hand-over, so views and markers dropped by a failure are released under the GIL.
	FencedGil gil;
	nb::object answer;
	try {
		nb::object request = nb::none();
		if (!identity) {
			nb::list wanted;
			for (const auto column : requested) {
				wanted.append(nb::int_(static_cast<uint64_t>(column)));
			}
			request = std::move(wanted);
		}
		answer = entry.object.attr("columns")(request);
	} catch (nb::python_error &error) {
		throw cxx::InvalidInputException("reading the columns of the object registered as '" + entry.name +
		                                 "' failed: " + DescribePythonError(error));
	}
	auto context = input.GetContext();
	auto columns = OpenColumns(entry, bound, context, requested, answer, batch_rows);
	ScalarMarkers markers;
	// An array can hold pandas' singletons only once pandas is loaded, so an absent pandas leaves no marker to match
	// and is never imported for one.
	nb::object pandas = nb::module_::import_("sys").attr("modules").attr("get")("pandas");
	if (!pandas.is_none()) {
		markers.na = pandas.attr("NA");
		markers.nat = pandas.attr("NaT");
	}
	nb::module_ numpy = nb::module_::import_("numpy");
	markers.numpy_generic = numpy.attr("generic");
	markers.numpy_floating = numpy.attr("floating");
	markers.numpy_datetime64 = numpy.attr("datetime64");
	markers.numpy_timedelta64 = numpy.attr("timedelta64");
	input.SetGlobalState<NumpyScanState>(std::move(columns), bound.rows, range_rows, std::move(markers));
}

/// Runs on an engine thread: no Python here.
void NumpyScanInitLocal(cxx::TableFunction::InitLocalInput &input) {
	auto &global = input.GetGlobalState<NumpyScanState>();
	const auto batch_rows = input.GetUserData<NumpyScanUserData>().batch_rows;
	auto context = input.GetContext();
	std::vector<std::optional<cxx::ArrowImporter>> importers(global.columns.size());
	for (cxx::idx_t i = 0; i < global.columns.size(); i++) {
		auto &column = global.columns[i];
		if (column.encoding == NumpyEncoding::ARROW) {
			importers[i].emplace(context, column.arrow.schema.value, batch_rows);
		}
	}
	input.SetLocalState<NumpyScanLocalState>(std::move(importers));
}

/// A run of fixed-width elements, validity from the mask when there is one, else for FLOAT and DOUBLE from a raw
/// NaN, the only way a plain float column marks a missing value.
void FillFixed(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count) {
	auto *dest = static_cast<uint8_t *>(vector.GetDataMutable());
	CopyRows(column.data, start, count, dest);

	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	const auto id = column.type.GetTypeId();
	const bool is_float = id == cxx::LogicalTypeId::FLOAT;
	const bool is_double = id == cxx::LogicalTypeId::DOUBLE;
	for (cxx::idx_t i = 0; i < count; i++) {
		bool invalid = false;
		if (column.mask) {
			invalid = Masked(column, start + i);
		} else if (is_float) {
			invalid = std::isnan(reinterpret_cast<float *>(dest)[i]);
		} else if (is_double) {
			invalid = std::isnan(reinterpret_cast<double *>(dest)[i]);
		}
		if (invalid) {
			validity.SetInvalid(i);
		}
	}
}

void FillCounts(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count,
                const std::string &registered_name) {
	const auto &conversion = column.conversion;
	const bool interval = column.type.GetTypeId() == cxx::LogicalTypeId::INTERVAL;
	const auto beyond_range = [&](cxx::idx_t row) {
		return cxx::InvalidInputException("the object registered as '" + registered_name + "' failed: column '" +
		                                  column.name + "' holds a value at row " + std::to_string(row) +
		                                  " beyond the range of its engine type");
	};
	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	if (!interval && conversion.step == 1 && conversion.unit == 1 && conversion.denominator == 1) {
		auto *dest = vector.GetDataMutable<int64_t>();
		CopyRows(column.data, start, count, dest);
		for (cxx::idx_t i = 0; i < count; i++) {
			if (dest[i] == std::numeric_limits<int64_t>::min() || Masked(column, start + i)) {
				validity.SetInvalid(i);
			} else if (dest[i] < column.range.first || dest[i] > column.range.last) {
				throw beyond_range(start + i);
			}
		}
		return;
	}
	auto *counts = interval ? nullptr : vector.GetDataMutable<int64_t>();
	auto *intervals = interval ? vector.GetDataMutable<cxx::interval_t>() : nullptr;
	for (cxx::idx_t i = 0; i < count; i++) {
		const auto raw = Load<int64_t>(column.data, start + i);
		int64_t scaled = 0;
		if (raw == std::numeric_limits<int64_t>::min() || Masked(column, start + i)) {
			validity.SetInvalid(i);
		} else if (!ScaleCount(raw, conversion, false, scaled) ||
		           (!interval && (scaled < column.range.first || scaled > column.range.last))) {
			throw beyond_range(start + i);
		}
		if (interval) {
			intervals[i] = cxx::interval_t {0, 0, scaled};
		} else {
			counts[i] = scaled;
		}
	}
}

int64_t ReadCode(const NumpyColumn &column, cxx::idx_t row) {
	switch (column.data->itemsize) {
	case 1:
		return Load<int8_t>(column.data, row);
	case 2:
		return Load<int16_t>(column.data, row);
	default:
		return Load<int32_t>(column.data, row);
	}
}

/// Category codes into an ENUM whose codes the engine stores as `CODE`; a code past the ENUM's labels is refused,
/// since the engine would look its label up out of range.
template <class CODE>
void FillEnumCodes(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count,
                   const std::string &registered_name) {
	auto *dest = vector.GetDataMutable<CODE>();
	const auto labels = column.type.GetEnumSize();
	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		const auto code = ReadCode(column, start + i);
		if (code < 0 || Masked(column, start + i)) {
			validity.SetInvalid(i);
			dest[i] = 0;
			continue;
		}
		if (static_cast<cxx::idx_t>(code) >= labels) {
			throw cxx::InvalidInputException("the object registered as '" + registered_name + "' failed: column '" +
			                                 column.name + "' holds a category code at row " +
			                                 std::to_string(start + i) + " that its ENUM has no label for");
		}
		dest[i] = static_cast<CODE>(code);
	}
}

/// Categorical codes, narrowed or widened from pandas' own signed width into the ENUM's unsigned physical width;
/// the two widths need not match, since pandas and DuckDB each size a dictionary's codes by a different rule.
void FillEnum(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count,
              const std::string &registered_name) {
	switch (column.type.GetEnumInternalTypeId()) {
	case cxx::LogicalTypeId::UTINYINT:
		FillEnumCodes<uint8_t>(vector, column, start, count, registered_name);
		break;
	case cxx::LogicalTypeId::USMALLINT:
		FillEnumCodes<uint16_t>(vector, column, start, count, registered_name);
		break;
	default:
		FillEnumCodes<uint32_t>(vector, column, start, count, registered_name);
		break;
	}
}

/// Whether a cell of an object column stands for SQL NULL: Python's None, pandas' `NA` or `NaT` singletons
/// (compared by identity, since neither is equal to itself under `==`), a float NaN of any width, the marker a
/// plain float column uses, or numpy's datetime64 or timedelta64 NaT; the sampling rule must count the same cells.
bool IsNoneLike(const ScalarMarkers &markers, PyObject *value) {
	if (value == Py_None || value == markers.na.ptr() || value == markers.nat.ptr()) {
		return true;
	}
	if (PyFloat_Check(value)) {
		return std::isnan(PyFloat_AsDouble(value));
	}
	nb::handle cell(value);
	if (!nb::isinstance(cell, markers.numpy_generic)) {
		return false;
	}
	if (nb::isinstance(cell, markers.numpy_floating)) {
		return std::isnan(nb::cast<double>(cell));
	}
	if (nb::isinstance(cell, markers.numpy_datetime64) || nb::isinstance(cell, markers.numpy_timedelta64)) {
		// Not PyObject_RichCompareBool, which reports any object equal to itself without asking it.
		nb::object differs = nb::steal(PyObject_RichCompare(value, value, Py_NE));
		const int truth = differs.is_valid() ? PyObject_IsTrue(differs.ptr()) : -1;
		if (truth < 0) {
			// Left to the conversion, which reports a failing value with its row and column.
			PyErr_Clear();
			return false;
		}
		return truth == 1;
	}
	return false;
}

/// A Python-object array of `str`, `None`, `pd.NA` or a float NaN: each string read as UTF-8 directly, and any
/// other value, which only reaches this encoding through the sampling rule's mixed-type fallback, through `str()`.
void FillText(const NumpyScanState &global, cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start,
              cxx::idx_t count, const std::string &registered_name) {
	FencedGil gil;
	for (cxx::idx_t i = 0; i < count; i++) {
		PyObject *value = Load<PyObject *>(column.data, start + i);
		if (Masked(column, start + i) || IsNoneLike(global.markers, value)) {
			vector.SetNull(i);
			continue;
		}
		try {
			if (PyUnicode_Check(value)) {
				Py_ssize_t size = 0;
				// CPython hands out valid UTF-8 for any str it can encode; a lone surrogate is the one it cannot.
				const char *utf8 = PyUnicode_AsUTF8AndSize(value, &size);
				if (utf8 == nullptr) {
					throw nb::python_error();
				}
				vector.AssignStringUnsafe(i, std::string_view(utf8, static_cast<size_t>(size)));
				continue;
			}
			const auto rendered = nb::cast<std::string>(nb::str(nb::handle(value)));
			vector.AssignStringUnsafe(i, rendered);
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("the object registered as '" + registered_name + "' failed: column '" +
			                                 column.name + "' holds a value at row " + std::to_string(start + i) +
			                                 " that cannot be read as text: " + DescribePythonError(error));
		}
	}
}

/// An object column sampled to one engine type other than VARCHAR: each value converted with `PythonToValue` and
/// cast to that type only when the cast loses nothing, since the engine's own cast rounds rather than refuses a
/// value such as a fractional double read into an integer column.
void FillObjects(const NumpyScanState &global, cxx::Context &context, cxx::Vector &vector, const NumpyColumn &column,
                 cxx::idx_t start, cxx::idx_t count, ConversionContext &conversion,
                 const std::string &registered_name) {
	FencedGil gil;
	for (cxx::idx_t i = 0; i < count; i++) {
		PyObject *value = Load<PyObject *>(column.data, start + i);
		if (Masked(column, start + i) || IsNoneLike(global.markers, value)) {
			vector.SetNull(i);
			continue;
		}
		const auto failed = [&](const std::string &why) {
			return cxx::InvalidInputException("the object registered as '" + registered_name + "' failed: column '" +
			                                  column.name + "' holds a value " + why);
		};
		// Lazy, so the common row that converts never pays for its own error text.
		const auto row = [&] {
			return std::to_string(start + i);
		};
		std::optional<cxx::Value> converted;
		try {
			nb::handle cell(value);
			// PythonToValue reads numpy's datetime64 and timedelta64 in their own unit, and only Python's own types
			// otherwise, so any other numpy scalar is unwrapped first.
			const bool temporal = nb::isinstance(cell, global.markers.numpy_datetime64) ||
			                      nb::isinstance(cell, global.markers.numpy_timedelta64);
			nb::object item = nb::isinstance(cell, global.markers.numpy_generic) && !temporal ? cell.attr("item")()
			                                                                                  : nb::borrow(cell);
			converted = PythonToValue(context, item, conversion);
		} catch (const BeyondRangeException &) {
			throw failed("at row " + row() + " beyond the range of its engine type");
		} catch (const UnsupportedTypeException &) {
			converted.reset();
		} catch (const cxx::Exception &error) {
			throw failed("at row " + row() + " that cannot be read: " + error.GetRawMessage());
		} catch (nb::python_error &error) {
			throw failed("at row " + row() + " that cannot be read: " + DescribePythonError(error));
		} catch (...) {
			converted.reset();
		}
		if (converted && AssumesTimeZone(converted->GetLogicalType().GetTypeId(), column.type.GetTypeId())) {
			throw failed("of type " + converted->GetLogicalType().ToText() + " at row " + row() +
			             ", and converting it to its sampled type " + column.type.ToText() +
			             " would assume a time zone");
		}
		const auto cast = converted ? TryCastTemporalExactly(context, *converted, column.type) : std::nullopt;
		if (!cast) {
			throw failed("at row " + row() + " that its sampled type " + column.type.ToText() + " cannot hold exactly");
		}
		vector.SetValue(i, *cast);
	}
}

/// A fixed-width UCS-4 string column, as numpy's `U` dtype stores it, encoded to UTF-8 without the GIL; numpy pads a
/// shorter string with NUL code points and drops them on read, so trailing NULs are not part of the value.
void FillUcs4(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count,
              const std::string &registered_name) {
	const auto width = static_cast<cxx::idx_t>(column.data->itemsize) / 4;
	std::string utf8;
	for (cxx::idx_t i = 0; i < count; i++) {
		if (Masked(column, start + i)) {
			vector.SetNull(i);
			continue;
		}
		const auto *element = Element(column.data, start + i);
		const auto code_point = [&](cxx::idx_t k) {
			uint32_t value;
			std::memcpy(&value, element + k * 4, 4);
			return value;
		};
		auto length = width;
		while (length > 0 && code_point(length - 1) == 0) {
			length--;
		}
		utf8.clear();
		for (cxx::idx_t k = 0; k < length; k++) {
			const auto value = code_point(k);
			if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
				throw cxx::InvalidInputException("the object registered as '" + registered_name + "' failed: column '" +
				                                 column.name + "' holds a value at row " + std::to_string(start + i) +
				                                 " that is not valid Unicode");
			}
			if (value < 0x80) {
				utf8.push_back(static_cast<char>(value));
			} else if (value < 0x800) {
				utf8.push_back(static_cast<char>(0xC0 | (value >> 6)));
				utf8.push_back(static_cast<char>(0x80 | (value & 0x3F)));
			} else if (value < 0x10000) {
				utf8.push_back(static_cast<char>(0xE0 | (value >> 12)));
				utf8.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
				utf8.push_back(static_cast<char>(0x80 | (value & 0x3F)));
			} else {
				utf8.push_back(static_cast<char>(0xF0 | (value >> 18)));
				utf8.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
				utf8.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
				utf8.push_back(static_cast<char>(0x80 | (value & 0x3F)));
			}
		}
		vector.AssignStringUnsafe(i, utf8);
	}
}

/// A fixed-width bytes column, as numpy's `S` dtype stores it, read as BLOB without the GIL; trailing NUL bytes are
/// padding, as for `FillUcs4`.
void FillBytes(cxx::Vector &vector, const NumpyColumn &column, cxx::idx_t start, cxx::idx_t count) {
	const auto width = static_cast<size_t>(column.data->itemsize);
	for (cxx::idx_t i = 0; i < count; i++) {
		if (Masked(column, start + i)) {
			vector.SetNull(i);
			continue;
		}
		const auto *element = reinterpret_cast<const char *>(Element(column.data, start + i));
		auto length = width;
		while (length > 0 && element[length - 1] == 0) {
			length--;
		}
		vector.AssignString(i, std::string_view(element, length));
	}
}

void ReleaseView(ArrowArray *view) {
	delete static_cast<SharedArray *>(view->private_data);
	view->release = nullptr;
}

/// How many of `count` rows from row `from` of `array`, whose type records nulls as `nulls` says, are null. Exact,
/// never the -1 the Arrow C data interface allows for an uncounted array, since the engine's dictionary import reads -1
/// as no nulls at all.
int64_t NullsIn(const ArrowArray &array, ArrowNulls nulls, cxx::idx_t from, cxx::idx_t count) {
	if (nulls == ArrowNulls::EVERY_ROW) {
		return static_cast<int64_t>(count);
	}
	if (nulls == ArrowNulls::NONE || array.null_count == 0 || array.n_buffers == 0 || array.buffers[0] == nullptr) {
		return 0;
	}
	const auto *bits = static_cast<const uint8_t *>(array.buffers[0]);
	const auto first = static_cast<cxx::idx_t>(array.offset) + from;
	cxx::idx_t valid = 0;
	for (cxx::idx_t bit = first; bit < first + count; bit++) {
		valid += (bits[bit / 8] >> (bit % 8)) & 1;
	}
	return static_cast<int64_t>(count - valid);
}

/// `count` rows from row `from` of `shared`'s array, as a one-column batch whose column shares the array's buffers,
/// children and dictionary and holds a share of the array until released.
ArrowOwned<ArrowArray> ViewOf(const SharedArray &shared, ArrowNulls nulls, cxx::idx_t from, cxx::idx_t count) {
	const auto &array = shared->value;
	ArrowOwned<ArrowArray> view;
	view.value = array;
	view.value.offset = array.offset + static_cast<int64_t>(from);
	view.value.length = static_cast<int64_t>(count);
	view.value.null_count = NullsIn(array, nulls, from, count);
	view.value.private_data = new SharedArray(shared);
	view.value.release = &ReleaseView;
	WrapAsBatch(view.value);
	return view;
}

/// Rows `start` to `start + count` of an ARROW column, imported by this thread's importer from views of the arrays
/// holding them and referenced by `vector` without a copy. Rows spanning two arrays come out as one chunk, which the
/// importer copies them into.
void FillArrow(cxx::Vector &vector, NumpyScanLocalState &local, cxx::idx_t index, const NumpyColumn &column,
               cxx::idx_t start, cxx::idx_t count, const std::string &registered_name) {
	const auto &arrow = column.arrow;
	auto &importer = *local.importers.at(index);
	auto array = static_cast<cxx::idx_t>(std::upper_bound(arrow.starts.begin(), arrow.starts.end(), start) -
	                                     arrow.starts.begin() - 1);
	for (cxx::idx_t done = 0; done < count; array++) {
		const auto from = start + done - arrow.starts[array];
		const auto take = std::min(count - done, arrow.starts[array + 1] - arrow.starts[array] - from);
		done += take;
		auto view = ViewOf(arrow.arrays[array], arrow.nulls, from, take);
		// Every piece is shorter than a batch, so the importer holds each back until the flush on the last.
		importer.Append(view.value, true, done == count);
		if (done < count && importer.NextChunk()) {
			throw cxx::InvalidInputException("the Arrow column '" + column.name + "' of the object registered as '" +
			                                 registered_name + "' was imported in more chunks than its batch");
		}
	}
	auto chunk = importer.NextChunk();
	if (!chunk || chunk.GetRowCount() != count) {
		throw cxx::InvalidInputException("the Arrow column '" + column.name + "' of the object registered as '" +
		                                 registered_name + "' was imported short of its batch of " +
		                                 std::to_string(count) + " rows");
	}
	vector.Reference(chunk.GetVector(0));
	vector.SetSize(count);
	local.held.at(index) = std::move(chunk);
}

void NumpyScanExec(cxx::TableFunction::ExecInput &input) {
	auto &global = input.GetGlobalState<NumpyScanState>();
	auto &local = input.GetLocalState<NumpyScanLocalState>();
	auto &entry = *input.GetBindData<NumpyScanBindData>().entry;

	if (local.start >= local.end) {
		const auto claimed = global.next.fetch_add(global.range_rows);
		if (claimed >= global.rows) {
			return;
		}
		local.start = claimed;
		local.end = std::min(claimed + global.range_rows, global.rows);
		local.batch_index = claimed / global.range_rows;
	}

	auto output = input.GetOutputChunk();
	const auto out_columns = output.GetVectorCount();
	if (out_columns == 0) {
		throw cxx::InvalidInputException("the object registered as '" + entry.name +
		                                 "' was asked for no columns, which it cannot report rows through");
	}
	const auto batch_rows = input.GetUserData<NumpyScanUserData>().batch_rows;
	const auto emit = std::min(batch_rows, local.end - local.start);
	auto context = input.GetContext();
	auto &conversion = input.GetUserData<NumpyScanUserData>().module->conversion;

	for (cxx::idx_t i = 0; i < out_columns; i++) {
		auto &column = global.columns.at(i);
		auto vector = output.GetVector(i);
		vector.SetSize(emit);
		switch (column.encoding) {
		case NumpyEncoding::FIXED:
			FillFixed(vector, column, local.start, emit);
			break;
		case NumpyEncoding::TIMESTAMP:
		case NumpyEncoding::INTERVAL:
			FillCounts(vector, column, local.start, emit, entry.name);
			break;
		case NumpyEncoding::ENUM_CODES:
			FillEnum(vector, column, local.start, emit, entry.name);
			break;
		case NumpyEncoding::TEXT:
			FillText(global, vector, column, local.start, emit, entry.name);
			break;
		case NumpyEncoding::OBJECTS:
			FillObjects(global, context, vector, column, local.start, emit, conversion, entry.name);
			break;
		case NumpyEncoding::UCS4:
			FillUcs4(vector, column, local.start, emit, entry.name);
			break;
		case NumpyEncoding::BYTES:
			FillBytes(vector, column, local.start, emit);
			break;
		case NumpyEncoding::ARROW:
			FillArrow(vector, local, i, column, local.start, emit, entry.name);
			break;
		}
	}
	local.start += emit;
}

/// Reports the ordering position of the range this thread is emitting from; ranges are fixed-size slabs claimed in
/// order, so their index needs no separate counter, unlike a stream whose batches vary in size.
void NumpyScanPartitionData(cxx::TableFunction::PartitionDataInput &input) {
	if (input.GetPartitionColumnCount() != 0) {
		throw cxx::InvalidInputException(
		    "the scan of a registered object was asked for partition values, which it does not report");
	}
	auto &local = input.GetLocalState<NumpyScanLocalState>();
	input.SetBatchIndex(local.batch_index);
}

void NumpyScanProgress(cxx::TableFunction::ProgressInput &input) {
	auto &global = input.GetGlobalState<NumpyScanState>();
	if (global.rows == 0) {
		input.SetProgress(1.0);
		return;
	}
	const auto claimed = std::min(global.next.load(), global.rows);
	input.SetProgress(static_cast<double>(claimed) / static_cast<double>(global.rows));
}

} // namespace

void RegisterNumpyScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                       std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows) {
	auto function = cxx::TableFunction::Create(connection);
	function.SetName(kNumpyScanFunction);
	function.WithSignature(
	    [&](cxx::FunctionSignature &signature) { signature.AddParameter("name", connection.ParseType("VARCHAR")); });
	function.SetUserData<NumpyScanUserData>(NumpyScanUserData {std::move(registry), std::move(module), batch_rows});
	function.SetBindCallback(&NumpyScanBind);
	function.SetInitGlobalCallback(&NumpyScanInitGlobal);
	function.SetInitLocalCallback(&NumpyScanInitLocal);
	function.SetExecCallback(&NumpyScanExec);
	function.SetProgressCallback(&NumpyScanProgress);
	function.SetPartitionDataCallback(&NumpyScanPartitionData);
	function.SetProjectionPushdown(true);
	function.Register();
}

} // namespace duckdb_python
