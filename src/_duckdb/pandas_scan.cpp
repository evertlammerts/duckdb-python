//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/pandas_scan.cpp
//
//
//===----------------------------------------------------------------------===//

#include "pandas_scan.hpp"

#include "pyconv.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// A registered pandas frame is a pandas DataFrame that PandasSource, in duckdb/_sources/pandas.py, has judged fit
// to read without pyarrow: every column numpy- or Python-object-backed. Its describe() names the columns and their
// engine types; its columns() answers each requested column as a kind, an engine type, a data array and a mask
// array or None. A kind picks how a row's bytes turn into a vector element below, and the scan reads the pandas
// frame in ranges of rows that its threads claim in turn.

namespace duckdb_python {
namespace {

struct PandasScanUserData {
	std::shared_ptr<Registry> registry;
	std::shared_ptr<ModuleState> module;
	/// The engine's standard vector size, which bounds how many rows one exec call fills.
	cxx::idx_t batch_rows;
};

/// The entry a query bound over, and the column names and engine type texts `describe()` gave, in declared order.
struct PandasScanBindData {
	PandasScanBindData(std::shared_ptr<Registered> entry, std::vector<std::string> names, std::vector<std::string> type_texts,
	         cxx::idx_t rows)
	    : entry(std::move(entry)), names(std::move(names)), type_texts(std::move(type_texts)), rows(rows) {
	}

	/// Freed from engine threads too, so the Python reference is dropped under the GIL.
	~PandasScanBindData() {
		nb::gil_scoped_acquire gil;
		entry.reset();
	}

	std::shared_ptr<Registered> entry;
	std::vector<std::string> names;
	std::vector<std::string> type_texts;
	cxx::idx_t rows;
};

/// Which write rule a column's bytes follow. `TIMESTAMP_TZ` and `INTERVAL` carry the source unit of their int64
/// data, everything else needs none.
enum class PandasColumnKind : uint8_t {
	FIXED,
	TIMESTAMP,
	TIMESTAMP_TZ,
	INTERVAL,
	ENUM_CODES,
	TEXT,
	OBJECTS,
};

/// One requested column's buffers, kept for the scan's life: the arrays `columns()` handed over, and the buffer
/// protocol views taken on them at global init.
struct PandasColumn {
	PandasColumn(std::string name, PandasColumnKind column_kind, char unit, cxx::LogicalType type)
	    : name(std::move(name)), column_kind(column_kind), unit(unit), type(std::move(type)) {
	}

	PandasColumn(PandasColumn &&other) noexcept
	    : name(std::move(other.name)), column_kind(other.column_kind), unit(other.unit), type(std::move(other.type)),
	      data_obj(std::move(other.data_obj)), mask_obj(std::move(other.mask_obj)), data(other.data),
	      has_data_buffer(other.has_data_buffer), mask(other.mask), has_mask_buffer(other.has_mask_buffer) {
		other.has_data_buffer = false;
		other.has_mask_buffer = false;
	}
	PandasColumn(const PandasColumn &) = delete;
	PandasColumn &operator=(const PandasColumn &) = delete;
	PandasColumn &operator=(PandasColumn &&) = delete;

	/// Runs under the GIL: at global init when a later column is refused, and at the scan's teardown.
	~PandasColumn() {
		if (has_data_buffer) {
			PyBuffer_Release(&data);
		}
		if (has_mask_buffer) {
			PyBuffer_Release(&mask);
		}
	}

	std::string name;
	PandasColumnKind column_kind;
	/// The unit of a TIMESTAMP_TZ or INTERVAL column's int64 data: 's', 'm' (milli), 'u' (micro) or 'n' (nano).
	char unit;
	cxx::LogicalType type;
	nb::object data_obj;
	nb::object mask_obj;
	Py_buffer data {};
	bool has_data_buffer = false;
	Py_buffer mask {};
	bool has_mask_buffer = false;
};

/// One scan's columns, the row count they cover, and the claim counter every thread divides the pandas frame with.
struct PandasScanState {
	PandasScanState(std::vector<PandasColumn> columns, cxx::idx_t rows, cxx::idx_t range_rows, nb::object na,
	                 nb::object nat, nb::object numpy_generic)
	    : columns(std::move(columns)), rows(rows), range_rows(range_rows), na(std::move(na)), nat(std::move(nat)),
	      numpy_generic(std::move(numpy_generic)) {
	}

	/// Torn down from an engine thread, so every buffer and Python reference is released under the GIL.
	~PandasScanState() {
		nb::gil_scoped_acquire gil;
		columns.clear();
		na = nb::object();
		nat = nb::object();
		numpy_generic = nb::object();
	}

	std::vector<PandasColumn> columns;
	cxx::idx_t rows;
	/// Rows claimed together by one thread: 50 batches' worth, so a thread does not pay the claim's cost every batch.
	cxx::idx_t range_rows;
	std::atomic<cxx::idx_t> next {0};
	/// pandas' NA and NaT singletons and numpy's scalar base class, looked up once per scan under the GIL.
	nb::object na;
	nb::object nat;
	nb::object numpy_generic;
};

/// One thread's claimed range within the pandas frame, and the ordering position that range stands for.
struct PandasScanLocalState {
	cxx::idx_t start = 0;
	cxx::idx_t end = 0;
	cxx::idx_t batch_index = 0;
};

/// Resolved once per query while bound, so a scan sees the pandas frame registered under the name when it binds.
void PandasScanBind(cxx::TableFunction::BindInput &input) {
	auto &registry = *input.GetUserData<PandasScanUserData>().registry;
	const auto name = std::string(input.GetArgument(0).Get<cxx::varchar_t>());
	auto entry = registry.ByName(name);
	if (!entry) {
		throw cxx::InvalidInputException("nothing is registered as '" + name + "'");
	}
	std::vector<std::string> names;
	std::vector<std::string> type_texts;
	cxx::idx_t rows = 0;
	{
		nb::gil_scoped_acquire gil;
		try {
			nb::object described = entry->object.attr("describe")();
			for (nb::handle item : described) {
				auto pair = nb::cast<nb::tuple>(item);
				auto column_name = nb::cast<std::string>(pair[0]);
				auto type_text = nb::cast<std::string>(pair[1]);
				input.AddResultColumn(column_name, input.GetContext().ParseType(type_text));
				names.push_back(std::move(column_name));
				type_texts.push_back(std::move(type_text));
			}
			nb::object count = entry->object.attr("rows")();
			// A bool is an int to Python, and a truth value is never a row count.
			if (count.is_none() || nb::isinstance<nb::bool_>(count)) {
				throw nb::cast_error();
			}
			rows = nb::cast<cxx::idx_t>(count);
		} catch (nb::python_error &error) {
			throw cxx::InvalidInputException("describing the pandas frame registered as '" + entry->name +
			                                 "' failed: " + DescribePythonError(error));
		} catch (const nb::cast_error &) {
			throw cxx::InvalidInputException("the pandas frame registered as '" + entry->name +
			                                 "' answered describe() or rows() with something other than (name, type) "
			                                 "pairs and a row count");
		}
	}
	input.SetCardinality(rows, true);
	input.SetBindData<PandasScanBindData>(std::move(entry), std::move(names), std::move(type_texts), rows);
}

/// `kind` as `columns()` spelled it, split into the write rule and, for a TIMESTAMP_TZ or INTERVAL column, the
/// source unit its int64 data carries.
PandasColumnKind ParseKind(const std::string &registered_name, const std::string &column_name, const std::string &kind,
                    char &unit) {
	unit = 0;
	if (kind == "fixed") {
		return PandasColumnKind::FIXED;
	}
	if (kind == "timestamp") {
		return PandasColumnKind::TIMESTAMP;
	}
	// The unit is "s", "ms", "us" or "ns"; the first letter alone tells the two apart, so the rest is not checked.
	if (kind.rfind("timestamp:", 0) == 0 && kind.size() > 10) {
		unit = kind[10];
		return PandasColumnKind::TIMESTAMP_TZ;
	}
	if (kind.rfind("interval:", 0) == 0 && kind.size() > 9) {
		unit = kind[9];
		return PandasColumnKind::INTERVAL;
	}
	if (kind == "enum") {
		return PandasColumnKind::ENUM_CODES;
	}
	if (kind == "text") {
		return PandasColumnKind::TEXT;
	}
	if (kind == "objects") {
		return PandasColumnKind::OBJECTS;
	}
	throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
	                                 column_name + "' with the unknown kind '" + kind + "'");
}

/// The byte width of a `"fixed"` column's engine type; 0 for a type this kind never carries.
cxx::idx_t FixedWidth(const cxx::LogicalType &type) {
	switch (type.GetTypeId()) {
	case cxx::LogicalTypeId::BOOLEAN:
	case cxx::LogicalTypeId::TINYINT:
	case cxx::LogicalTypeId::UTINYINT:
		return 1;
	case cxx::LogicalTypeId::SMALLINT:
	case cxx::LogicalTypeId::USMALLINT:
		return 2;
	case cxx::LogicalTypeId::INTEGER:
	case cxx::LogicalTypeId::UINTEGER:
	case cxx::LogicalTypeId::FLOAT:
		return 4;
	case cxx::LogicalTypeId::BIGINT:
	case cxx::LogicalTypeId::UBIGINT:
	case cxx::LogicalTypeId::DOUBLE:
		return 8;
	default:
		return 0;
	}
}

/// Whether a data buffer of `itemsize` bytes is the width `column_kind` and `type` need: the engine type's own width
/// for a fixed column, 8 bytes (the int64 view Python took) for a timestamp or interval, a signed code of 1, 2 or 4
/// bytes for an enum, and a pointer width for text or objects.
bool ValidDataWidth(PandasColumnKind column_kind, const cxx::LogicalType &type, cxx::idx_t itemsize) {
	switch (column_kind) {
	case PandasColumnKind::FIXED:
		return itemsize == FixedWidth(type);
	case PandasColumnKind::ENUM_CODES:
		return itemsize == 1 || itemsize == 2 || itemsize == 4;
	case PandasColumnKind::TEXT:
	case PandasColumnKind::OBJECTS:
		return itemsize == sizeof(void *);
	default: // TIMESTAMP, TIMESTAMP_TZ, INTERVAL
		return itemsize == sizeof(int64_t);
	}
}

bool HostIsLittleEndian() {
	const uint16_t probe = 1;
	return *reinterpret_cast<const uint8_t *>(&probe) == 1;
}

/// Takes a buffer on `obj`, refusing one that is not one-dimensional, not contiguous, the wrong length, or the
/// wrong element width for `column_kind` and `type`; `kind_of` names what `obj` is, for the message.
void OpenBuffer(const std::string &registered_name, const std::string &column_name, const char *kind_of, nb::object &obj,
                PandasColumnKind column_kind, const cxx::LogicalType &type, cxx::idx_t rows, Py_buffer &view) {
	if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) != 0) {
		PyErr_Clear();
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
		                                 column_name + "' with a " + kind_of +
		                                 " array that is not one-dimensional and contiguous");
	}
	if (view.ndim != 1) {
		PyBuffer_Release(&view);
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
		                                 column_name + "' with a " + kind_of + " array that is not one-dimensional");
	}
	if (static_cast<cxx::idx_t>(view.shape[0]) != rows) {
		const auto length = std::to_string(view.shape[0]);
		PyBuffer_Release(&view);
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
		                                 column_name + "' with a " + kind_of + " array of " + length +
		                                 " rows where " + std::to_string(rows) + " were expected");
	}
	if (!ValidDataWidth(column_kind, type, static_cast<cxx::idx_t>(view.itemsize))) {
		PyBuffer_Release(&view);
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
		                                 column_name + "' with a " + kind_of +
		                                 " array whose element size does not match its engine type");
	}
	// The bytes are copied as they are, so a buffer in the other byte order would read as garbage.
	const char *format = view.format != nullptr ? view.format : "";
	const bool swapped = HostIsLittleEndian() ? (format[0] == '>' || format[0] == '!') : format[0] == '<';
	if (swapped) {
		PyBuffer_Release(&view);
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' answered columns() for '" +
		                                 column_name + "' with a " + kind_of +
		                                 " array in the other byte order, which is not read");
	}
}

/// `columns(requested)`'s reply, opened into buffered `PandasColumn`s; every buffer already opened is released
/// before an error escapes, so a later column's refusal never leaks an earlier one's.
std::vector<PandasColumn> OpenColumns(const Registered &entry, const PandasScanBindData &bound, cxx::Context &context,
                                     const std::vector<cxx::idx_t> &requested, nb::object answer) {
	if (!nb::isinstance<nb::list>(answer) && !nb::isinstance<nb::tuple>(answer)) {
		throw cxx::InvalidInputException("the pandas frame registered as '" + entry.name +
		                                 "' answered columns() with something other than a list");
	}
	if (static_cast<cxx::idx_t>(nb::len(answer)) != requested.size()) {
		throw cxx::InvalidInputException("the pandas frame registered as '" + entry.name + "' answered columns() with " +
		                                 std::to_string(nb::len(answer)) + " columns where " +
		                                 std::to_string(requested.size()) + " were requested");
	}
	std::vector<PandasColumn> columns;
	columns.reserve(requested.size());
	try {
		for (cxx::idx_t i = 0; i < requested.size(); i++) {
			auto item = nb::cast<nb::tuple>(answer[i]);
			auto column_name = nb::cast<std::string>(item[0]);
			const auto declared = requested[i];
			if (column_name != bound.names.at(declared)) {
				throw cxx::InvalidInputException("the pandas frame registered as '" + entry.name +
				                                 "' answered columns() with column '" + column_name +
				                                 "' at position " + std::to_string(i) + " where '" +
				                                 bound.names.at(declared) + "' was expected");
			}
			const auto kind_text = nb::cast<std::string>(item[1]);
			char unit = 0;
			const auto column_kind = ParseKind(entry.name, column_name, kind_text, unit);
			auto type = context.ParseType(bound.type_texts.at(declared));

			PandasColumn column(std::move(column_name), column_kind, unit, std::move(type));
			column.data_obj = nb::borrow(item[3]);
			OpenBuffer(entry.name, column.name, "data", column.data_obj, column_kind, column.type, bound.rows,
			          column.data);
			column.has_data_buffer = true;

			nb::object mask = nb::borrow(item[4]);
			if (!mask.is_none()) {
				column.mask_obj = mask;
				OpenBuffer(entry.name, column.name, "mask", column.mask_obj, PandasColumnKind::FIXED,
				          context.ParseType("BOOLEAN"), bound.rows, column.mask);
				column.has_mask_buffer = true;
			}
			columns.push_back(std::move(column));
		}
	} catch (const nb::cast_error &) {
		throw cxx::InvalidInputException("the pandas frame registered as '" + entry.name +
		                                 "' answered columns() with something other than (name, kind, type, data, "
		                                 "mask) tuples");
	}
	return columns;
}

void PandasScanInitGlobal(cxx::TableFunction::InitGlobalInput &input) {
	const auto &bound = input.GetBindData<PandasScanBindData>();
	auto &entry = *bound.entry;
	const auto declared = static_cast<cxx::idx_t>(bound.names.size());
	std::vector<cxx::idx_t> requested;
	bool identity = input.GetColumnCount() == declared;
	for (cxx::idx_t i = 0; i < input.GetColumnCount(); i++) {
		requested.push_back(input.GetColumnIndex(i));
		identity = identity && requested.back() == i;
	}

	std::vector<PandasColumn> columns;
	{
		nb::gil_scoped_acquire gil;
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
			throw cxx::InvalidInputException("reading the columns of the pandas frame registered as '" + entry.name +
			                                 "' failed: " + DescribePythonError(error));
		}
		auto context = input.GetContext();
		columns = OpenColumns(entry, bound, context, requested, std::move(answer));
	}

	const auto batch_rows = input.GetUserData<PandasScanUserData>().batch_rows;
	const cxx::idx_t range_rows = std::max<cxx::idx_t>(1, batch_rows) * 50;
	const cxx::idx_t max_threads = std::max<cxx::idx_t>(1, (bound.rows + range_rows - 1) / range_rows);
	input.SetMaxThreads(max_threads);
	nb::object na;
	nb::object nat;
	nb::object numpy_generic;
	{
		nb::gil_scoped_acquire gil;
		nb::module_ pandas = nb::module_::import_("pandas");
		na = pandas.attr("NA");
		nat = pandas.attr("NaT");
		numpy_generic = nb::module_::import_("numpy").attr("generic");
	}
	input.SetGlobalState<PandasScanState>(std::move(columns), bound.rows, range_rows, std::move(na), std::move(nat),
	                                       std::move(numpy_generic));
}

void PandasScanInitLocal(cxx::TableFunction::InitLocalInput &input) {
	input.SetLocalState<PandasScanLocalState>();
}

/// A row's worth of raw bytes into microseconds, from the unit `columns()` carried in the kind string; the caller
/// checks the NaT sentinel first, since scaling it would overflow.
int64_t ScaleToMicros(int64_t raw, char unit, const std::string &registered_name, const std::string &column_name) {
	int64_t scaled = raw;
	switch (unit) {
	case 's':
		if (__builtin_mul_overflow(raw, static_cast<int64_t>(1'000'000), &scaled)) {
			throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' has a value in column '" +
			                                 column_name + "' whose instant overflows microseconds");
		}
		return scaled;
	case 'm':
		if (__builtin_mul_overflow(raw, static_cast<int64_t>(1'000), &scaled)) {
			throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' has a value in column '" +
			                                 column_name + "' whose instant overflows microseconds");
		}
		return scaled;
	case 'u':
		return raw;
	case 'n':
		return raw / 1000;
	default:
		throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' has an unrecognised unit '" +
		                                 std::string(1, unit) + "' for column '" + column_name + "'");
	}
}

/// A contiguous run of `width`-byte elements, validity from the mask when there is one, else for FLOAT and DOUBLE
/// from a raw NaN, the only way a plain float column marks a missing value.
void FillFixed(cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start, cxx::idx_t count) {
	const auto width = FixedWidth(column.type);
	auto *dest = static_cast<uint8_t *>(vector.GetDataMutable());
	const auto *src = static_cast<const uint8_t *>(column.data.buf) + start * width;
	std::memcpy(dest, src, static_cast<size_t>(count * width));

	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	const auto id = column.type.GetTypeId();
	const bool is_float = id == cxx::LogicalTypeId::FLOAT;
	const bool is_double = id == cxx::LogicalTypeId::DOUBLE;
	const auto *mask = column.has_mask_buffer ? static_cast<const uint8_t *>(column.mask.buf) + start : nullptr;
	for (cxx::idx_t i = 0; i < count; i++) {
		bool invalid = false;
		if (mask != nullptr) {
			invalid = mask[i] != 0;
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

/// A naive timestamp's int64 view, copied straight through: DuckDB's own TIMESTAMP_S/MS/TIMESTAMP/TIMESTAMP_NS
/// storage counts in the same unit numpy does, so no per-element conversion is needed, only the NaT sentinel.
void FillTimestampNaive(cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start, cxx::idx_t count) {
	auto *dest = vector.GetDataMutable<int64_t>();
	const auto *src = static_cast<const int64_t *>(column.data.buf) + start;
	std::memcpy(dest, src, count * sizeof(int64_t));
	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		if (dest[i] == std::numeric_limits<int64_t>::min()) {
			validity.SetInvalid(i);
		}
	}
}

/// A UTC-normalized aware timestamp, one element at a time: the engine's TIMESTAMP_TZ is always microseconds, so a
/// column of another unit is scaled per row.
void FillTimestampAware(cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start, cxx::idx_t count,
                        const std::string &registered_name) {
	auto *dest = vector.GetDataMutable<int64_t>();
	const auto *src = static_cast<const int64_t *>(column.data.buf) + start;
	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		const auto raw = src[i];
		if (raw == std::numeric_limits<int64_t>::min()) {
			validity.SetInvalid(i);
			dest[i] = 0;
			continue;
		}
		dest[i] = ScaleToMicros(raw, column.unit, registered_name, column.name);
	}
}

/// A timedelta64 column as INTERVAL, the whole span in microseconds and no months or days, since a timedelta
/// carries no calendar component to split out.
void FillInterval(cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start, cxx::idx_t count,
                  const std::string &registered_name) {
	auto *dest = vector.GetDataMutable<cxx::interval_t>();
	const auto *src = static_cast<const int64_t *>(column.data.buf) + start;
	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		const auto raw = src[i];
		if (raw == std::numeric_limits<int64_t>::min()) {
			validity.SetInvalid(i);
			dest[i] = cxx::interval_t {0, 0, 0};
			continue;
		}
		dest[i] = cxx::interval_t {0, 0, ScaleToMicros(raw, column.unit, registered_name, column.name)};
	}
}

/// Categorical codes, narrowed or widened from pandas' own signed width into the ENUM's unsigned physical width;
/// the two widths need not match, since pandas and DuckDB each size a dictionary's codes by a different rule.
void FillEnum(cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start, cxx::idx_t count) {
	const auto src_width = static_cast<cxx::idx_t>(column.data.itemsize);
	const auto *base = static_cast<const uint8_t *>(column.data.buf) + start * src_width;
	const auto read_code = [&](cxx::idx_t i) -> int64_t {
		const auto *element = base + i * src_width;
		switch (src_width) {
		case 1:
			return *reinterpret_cast<const int8_t *>(element);
		case 2:
			return *reinterpret_cast<const int16_t *>(element);
		default:
			return *reinterpret_cast<const int32_t *>(element);
		}
	};

	auto validity = vector.GetValidityMutable();
	validity.SetAllValid(count);
	switch (column.type.GetEnumInternalTypeId()) {
	case cxx::LogicalTypeId::UTINYINT: {
		auto *dest = vector.GetDataMutable<uint8_t>();
		for (cxx::idx_t i = 0; i < count; i++) {
			const auto code = read_code(i);
			if (code < 0) {
				validity.SetInvalid(i);
				dest[i] = 0;
			} else {
				dest[i] = static_cast<uint8_t>(code);
			}
		}
		break;
	}
	case cxx::LogicalTypeId::USMALLINT: {
		auto *dest = vector.GetDataMutable<uint16_t>();
		for (cxx::idx_t i = 0; i < count; i++) {
			const auto code = read_code(i);
			if (code < 0) {
				validity.SetInvalid(i);
				dest[i] = 0;
			} else {
				dest[i] = static_cast<uint16_t>(code);
			}
		}
		break;
	}
	default: {
		auto *dest = vector.GetDataMutable<uint32_t>();
		for (cxx::idx_t i = 0; i < count; i++) {
			const auto code = read_code(i);
			if (code < 0) {
				validity.SetInvalid(i);
				dest[i] = 0;
			} else {
				dest[i] = static_cast<uint32_t>(code);
			}
		}
		break;
	}
	}
}

/// Whether a cell of an object column stands for SQL NULL: Python's None, pandas' `NA` or `NaT` singletons
/// (compared by identity, since neither is equal to itself under `==`), or a float NaN, the marker a plain
/// float64 column uses.
bool IsNoneLike(const PandasScanState &global, PyObject *value) {
	if (value == Py_None || value == global.na.ptr() || value == global.nat.ptr()) {
		return true;
	}
	return PyFloat_Check(value) && std::isnan(PyFloat_AsDouble(value));
}

/// A numpy scalar as the plain Python value it wraps, since `PythonToValue` understands only Python's own types.
nb::object UnwrapNumpyScalar(const PandasScanState &global, nb::handle value) {
	if (PyObject_IsInstance(value.ptr(), global.numpy_generic.ptr()) != 0) {
		return value.attr("item")();
	}
	return nb::borrow(value);
}

/// A Python-object array of `str`, `None`, `pd.NA` or a float NaN: each string read as UTF-8 directly, and any
/// other value, which only reaches this kind through the sampling rule's mixed-type fallback, through `str()`.
void FillText(const PandasScanState &global, cxx::Vector &vector, const PandasColumn &column, cxx::idx_t start,
              cxx::idx_t count, const std::string &registered_name) {
	nb::gil_scoped_acquire gil;
	auto *const *values = static_cast<PyObject *const *>(column.data.buf) + start;
	for (cxx::idx_t i = 0; i < count; i++) {
		PyObject *value = values[i];
		if (IsNoneLike(global, value)) {
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
			throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' failed: column '" +
			                                 column.name + "' holds a value at row " + std::to_string(start + i) +
			                                 " that cannot be read as text: " + DescribePythonError(error));
		}
	}
}

/// An object column sampled to one engine type other than VARCHAR: each value converted with `PythonToValue` and
/// cast to that type, with the cast checked to be lossless by casting back and comparing text, since the engine's
/// own cast rounds rather than refuses a value such as a fractional double read into an integer column.
void FillObjects(const PandasScanState &global, cxx::Context &context, cxx::Vector &vector,
                 const PandasColumn &column, cxx::idx_t start, cxx::idx_t count, ConversionContext &conversion,
                 const std::string &registered_name) {
	nb::gil_scoped_acquire gil;
	auto *const *values = static_cast<PyObject *const *>(column.data.buf) + start;
	for (cxx::idx_t i = 0; i < count; i++) {
		PyObject *value = values[i];
		if (IsNoneLike(global, value)) {
			vector.SetNull(i);
			continue;
		}
		std::optional<cxx::Value> cast;
		try {
			nb::object unwrapped = UnwrapNumpyScalar(global, nb::handle(value));
			cxx::Value converted = PythonToValue(context, unwrapped, conversion);
			cast = converted.Cast(context, column.type);
			bool exact = false;
			try {
				exact = cast->Cast(context, converted.GetLogicalType()).ToText() == converted.ToText();
			} catch (...) {
				exact = false;
			}
			if (!exact) {
				cast.reset();
			}
		} catch (...) {
			cast.reset();
		}
		if (!cast) {
			throw cxx::InvalidInputException("the pandas frame registered as '" + registered_name + "' failed: column '" +
			                                 column.name + "' holds a value at row " + std::to_string(start + i) +
			                                 " that its sampled type " + column.type.ToText() +
			                                 " cannot hold exactly");
		}
		vector.SetValue(i, *cast);
	}
}

void PandasScanExec(cxx::TableFunction::ExecInput &input) {
	auto &global = input.GetGlobalState<PandasScanState>();
	auto &local = input.GetLocalState<PandasScanLocalState>();
	auto &entry = *input.GetBindData<PandasScanBindData>().entry;

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
		throw cxx::InvalidInputException("the pandas frame registered as '" + entry.name +
		                                 "' was asked for no columns, which it cannot report rows through");
	}
	const auto batch_rows = input.GetUserData<PandasScanUserData>().batch_rows;
	const auto emit = std::min(batch_rows, local.end - local.start);
	auto context = input.GetContext();
	auto &conversion = input.GetUserData<PandasScanUserData>().module->conversion;

	for (cxx::idx_t i = 0; i < out_columns; i++) {
		auto &column = global.columns.at(i);
		auto vector = output.GetVector(i);
		vector.SetSize(emit);
		switch (column.column_kind) {
		case PandasColumnKind::FIXED:
			FillFixed(vector, column, local.start, emit);
			break;
		case PandasColumnKind::TIMESTAMP:
			FillTimestampNaive(vector, column, local.start, emit);
			break;
		case PandasColumnKind::TIMESTAMP_TZ:
			FillTimestampAware(vector, column, local.start, emit, entry.name);
			break;
		case PandasColumnKind::INTERVAL:
			FillInterval(vector, column, local.start, emit, entry.name);
			break;
		case PandasColumnKind::ENUM_CODES:
			FillEnum(vector, column, local.start, emit);
			break;
		case PandasColumnKind::TEXT:
			FillText(global, vector, column, local.start, emit, entry.name);
			break;
		case PandasColumnKind::OBJECTS:
			FillObjects(global, context, vector, column, local.start, emit, conversion, entry.name);
			break;
		}
	}
	local.start += emit;
}

/// Reports the ordering position of the range this thread is emitting from; ranges are fixed-size slabs claimed in
/// order, so their index needs no separate counter, unlike a stream whose batches vary in size.
void PandasScanPartitionData(cxx::TableFunction::PartitionDataInput &input) {
	if (input.GetPartitionColumnCount() != 0) {
		throw cxx::InvalidInputException(
		    "the scan of a registered pandas frame was asked for partition values, which it does not report");
	}
	auto &local = input.GetLocalState<PandasScanLocalState>();
	input.SetBatchIndex(local.batch_index);
}

void PandasScanProgress(cxx::TableFunction::ProgressInput &input) {
	auto &global = input.GetGlobalState<PandasScanState>();
	if (global.rows == 0) {
		input.SetProgress(1.0);
		return;
	}
	const auto claimed = std::min(global.next.load(), global.rows);
	input.SetProgress(static_cast<double>(claimed) / static_cast<double>(global.rows));
}

} // namespace

void RegisterPandasScan(cxx::Connection &connection, std::shared_ptr<Registry> registry,
                        std::shared_ptr<ModuleState> module, cxx::idx_t batch_rows) {
	auto function = cxx::TableFunction::Create(connection);
	function.SetName(kPandasScanFunction);
	function.WithSignature(
	    [&](cxx::FunctionSignature &signature) { signature.AddParameter("name", connection.ParseType("VARCHAR")); });
	function.SetUserData<PandasScanUserData>(PandasScanUserData {std::move(registry), std::move(module), batch_rows});
	function.SetBindCallback(&PandasScanBind);
	function.SetInitGlobalCallback(&PandasScanInitGlobal);
	function.SetInitLocalCallback(&PandasScanInitLocal);
	function.SetExecCallback(&PandasScanExec);
	function.SetProgressCallback(&PandasScanProgress);
	function.SetPartitionDataCallback(&PandasScanPartitionData);
	function.SetProjectionPushdown(true);
	function.Register();
}

} // namespace duckdb_python
