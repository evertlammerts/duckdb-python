#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;

struct NumpyScan {
	static void Scan(ClientContext &context, PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);
	//! masked holds one bool per row of the whole column, or is null when the column has no mask
	static void ScanObjectColumn(ClientContext &context, PyObject **col, idx_t stride, const bool *masked, idx_t count,
	                             idx_t offset, Vector &out);
};

} // namespace duckdb
