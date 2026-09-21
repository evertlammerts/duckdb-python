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

/// An ENUM type's labels, in index order.
std::vector<std::string> EnumValues(const cxx::LogicalType &type);

/// A result's column types, shared between the result and every ChunkView it hands out.
using ColumnTypes = std::shared_ptr<const std::vector<cxx::LogicalType>>;

/// One batch of rows, column by column, for the numpy converter; its memoryviews last only as long as it does.
class ChunkView {
public:
	// `row_offset` counts rows an earlier fetch took; the buffers still cover every row, so the caller slices.
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

	/// A memoryview onto the column's values without copying, or None when the type has no fixed width.
	nb::object Data(cxx::idx_t column);

	/// One bit per row, set when the value is not NULL, in 64-bit words lowest bit first; None when no row is.
	nb::object Validity(cxx::idx_t column);

	/// A DECIMAL column's scale, so the converter never parses type text.
	int DecimalScale(cxx::idx_t column) const {
		return static_cast<int>(Type(column).GetDecimalScale());
	}

	/// An ENUM column's labels in index order; the data buffer holds only the codes.
	std::vector<std::string> EnumValues(cxx::idx_t column) const {
		return duckdb_python::EnumValues(Type(column));
	}

	/// One Python object per row, for the columns Data() cannot serve.
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
