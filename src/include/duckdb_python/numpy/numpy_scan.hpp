#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;

struct NumpyScan {
	static void Scan(ClientContext &context, PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);
	static void ScanObjectColumn(ClientContext &context, PyObject **col, idx_t stride, idx_t count, idx_t offset,
	                             Vector &out);
};

} // namespace duckdb
