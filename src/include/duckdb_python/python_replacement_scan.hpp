#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace duckdb {

struct PythonReplacementScan {
public:
	static unique_ptr<TableRef> Replace(ClientContext &context, ReplacementScanInput &input,
	                                    optional_ptr<ReplacementScanData> data);
	//! Try to perform a replacement, returns NULL on error
	static unique_ptr<TableRef> TryReplacementObject(const nb::object &entry, const string &name,
	                                                 ClientContext &context, bool relation = false);
	//! Perform a replacement or throw if it failed
	static unique_ptr<TableRef> ReplacementObject(const nb::object &entry, const string &name, ClientContext &context,
	                                              bool relation = false);
};

} // namespace duckdb
