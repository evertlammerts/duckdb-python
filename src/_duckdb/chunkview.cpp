//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/chunkview.cpp
//
//
//===----------------------------------------------------------------------===//

#include "chunkview.hpp"

#include <utility>

namespace duckdb_python {

std::vector<std::string> EnumValues(const cxx::LogicalType &type) {
	std::vector<std::string> out;
	const auto count = type.GetEnumSize();
	out.reserve(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		out.push_back(type.GetEnumValue(i));
	}
	return out;
}

ChunkView::ChunkView(cxx::DataChunk chunk, ColumnTypes types, cxx::idx_t row_offset)
    : chunk(std::move(chunk)), types(std::move(types)), row_offset(row_offset) {
	const auto count = this->chunk.GetVectorCount();
	vectors.reserve(count);
	for (cxx::idx_t i = 0; i < count; i++) {
		cxx::Vector vector = this->chunk.GetVector(i);
		// Dictionary, constant and other encodings become flat data
		// plus validity, the one layout the buffer views can serve.
		vector.Flatten();
		vectors.push_back(std::move(vector));
	}
}

nb::object ChunkView::Data(cxx::idx_t column) {
	const size_t element = ElementSize(column);
	if (element == 0) {
		return nb::none();
	}
	const auto view = vectors.at(column).GetView();
	return Memoryview(view.data, element * chunk.GetRowCount());
}

nb::object ChunkView::Validity(cxx::idx_t column) {
	const auto view = vectors.at(column).GetView();
	if (!view.validity) {
		return nb::none();
	}
	const size_t words = (chunk.GetRowCount() + 63) / 64;
	return Memoryview(view.validity, words * sizeof(uint64_t));
}

nb::list ChunkView::Values(cxx::idx_t column, ConversionContext &ctx) {
	auto &vector = vectors.at(column);
	nb::list out;
	const auto count = chunk.GetRowCount();
	for (cxx::idx_t row = 0; row < count; row++) {
		out.append(ValueToPython(vector.GetValue(row), ctx));
	}
	return out;
}

size_t ChunkView::ElementSize(cxx::idx_t column) const {
	using Id = cxx::LogicalTypeId;
	const auto &type = Type(column);
	switch (type.GetTypeId()) {
	case Id::BOOLEAN:
	case Id::TINYINT:
	case Id::UTINYINT:
		return 1;
	case Id::SMALLINT:
	case Id::USMALLINT:
		return 2;
	case Id::INTEGER:
	case Id::UINTEGER:
	case Id::DATE:
	case Id::FLOAT:
		return 4;
	case Id::BIGINT:
	case Id::UBIGINT:
	case Id::DOUBLE:
	case Id::TIMESTAMP:
	case Id::TIMESTAMP_SEC:
	case Id::TIMESTAMP_MS:
	case Id::TIMESTAMP_NS:
	case Id::TIMESTAMP_TZ:
		return 8;
	case Id::INTERVAL:
	case Id::HUGEINT:
	case Id::UHUGEINT:
		return 16;
	case Id::DECIMAL: {
		// The storage tier follows the width; the widest tier is int128,
		// which the converter reads as two 64-bit limbs.
		const auto width = type.GetDecimalWidth();
		return width <= 4 ? 2 : width <= 9 ? 4 : width <= 18 ? 8 : 16;
	}
	case Id::ENUM: {
		const auto size = type.GetEnumSize();
		return size < 256 ? 1 : size < 65536 ? 2 : 4;
	}
	default:
		return 0;
	}
}

nb::object ChunkView::Memoryview(const void *data, size_t bytes) {
	PyObject *view = PyMemoryView_FromMemory(const_cast<char *>(static_cast<const char *>(data)),
	                                         static_cast<Py_ssize_t>(bytes), PyBUF_READ);
	if (!view) {
		throw nb::python_error();
	}
	return nb::steal(view);
}

} // namespace duckdb_python
