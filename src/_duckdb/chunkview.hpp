//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/chunkview.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <string>
#include <vector>

#include "lifetime.hpp"

namespace duckdb_python {

/// The ENUM dictionary, index to string.
std::vector<std::string> EnumValues(const cxx::LogicalType &type);

/// A result's column types, shared between the result and the chunk views it
/// hands out.
using ColumnTypes = std::shared_ptr<const std::vector<cxx::LogicalType>>;

/// One fetched chunk, column-wise, for the numpy converter in duckdb/_numpy.py.
///
/// Fixed-width columns hand out zero-copy memoryviews over the flattened
/// vector data; everything else falls back to per-cell objects. The views
/// borrow the chunk's memory, so they are valid only while this object
/// lives: the converter copies out of them within one loop iteration and
/// never keeps one.
class ChunkView {
public:
	// `row_offset` is how many leading rows a prior row fetch already
	// consumed; the buffers still cover the whole chunk, so the converter
	// slices them by it.
	ChunkView(cxx::DataChunk chunk, ColumnTypes types, cxx::idx_t row_offset = 0);

	cxx::idx_t RowCount() const {
		return chunk.GetRowCount();
	}

	cxx::idx_t RowOffset() const {
		return row_offset;
	}

	cxx::idx_t ColumnCount() const {
		return static_cast<cxx::idx_t>(vectors.size());
	}

	int TypeId(cxx::idx_t column) const {
		return static_cast<int>(Type(column).GetTypeId());
	}

	std::string TypeText(cxx::idx_t column) const {
		return Type(column).ToText();
	}

	/// Zero-copy view over the flattened data, or None without a fixed-width layout.
	nb::object Data(cxx::idx_t column);

	/// Validity bitmask as 64-bit words, LSB first, or None when all rows are valid.
	nb::object Validity(cxx::idx_t column);

	/// A DECIMAL column's scale, so the converter never parses type text.
	int DecimalScale(cxx::idx_t column) const {
		return static_cast<int>(Type(column).GetDecimalScale());
	}

	/// The ENUM dictionary, index to string, for categorical assembly.
	std::vector<std::string> EnumValues(cxx::idx_t column) const {
		return duckdb_python::EnumValues(Type(column));
	}

	/// Per-cell object fallback for the columns Data() cannot serve.
	nb::list Values(cxx::idx_t column, ConversionContext &ctx);

private:
	const cxx::LogicalType &Type(cxx::idx_t column) const {
		return types->at(column);
	}

	/// Bytes per element for the fixed-width layouts, 0 for everything else.
	size_t ElementSize(cxx::idx_t column) const;

	static nb::object Memoryview(const void *data, size_t bytes);

	cxx::DataChunk chunk;
	ColumnTypes types;
	std::vector<cxx::Vector> vectors;
	cxx::idx_t row_offset;
};

} // namespace duckdb_python
