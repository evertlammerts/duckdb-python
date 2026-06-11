//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/python_log_storage.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/logging/log_storage.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/unique_ptr.hpp"

namespace duckdb {

class ColumnDataCollection;
class DatabaseInstance;

//! Scan state backing PythonLogStorage's in-memory buffers (so `duckdb_logs` can read them).
//! We define our own rather than reuse the engine's InMemoryLogStorageScanState to avoid
//! depending on whether that type's symbols are exported across platforms.
class PythonLogStorageScanState : public LogStorageScanState {
public:
	explicit PythonLogStorageScanState(LoggingTargetTable table) : LogStorageScanState(table) {
	}
	~PythonLogStorageScanState() override = default;

	ColumnDataScanState scan_state;
};

//! A composite log storage that does two things for every engine log entry:
//!   1. forwards it to Python's standard `logging` module (logging.getLogger("duckdb")), and
//!   2. retains it in-memory so `SELECT * FROM duckdb_logs` keeps working.
//!
//! It subclasses BufferingLogStorage with a buffer size of 1 so each entry is flushed (and
//! therefore forwarded to Python) immediately, rather than batched until a 2048-entry buffer
//! fills — engine WARNINGs are sparse and must surface inline to be useful.
class PythonLogStorage : public BufferingLogStorage {
public:
	explicit PythonLogStorage(DatabaseInstance &db);
	~PythonLogStorage() override;

	const string GetStorageName() override {
		return "python_log_storage";
	}

	//! Single-threaded scan interface — mirrors InMemoryLogStorage so duckdb_logs can read us.
	bool CanScan(LoggingTargetTable table) override;
	unique_ptr<LogStorageScanState> CreateScanState(LoggingTargetTable table) const override;
	bool Scan(LogStorageScanState &state, DataChunk &result) const override;
	void InitializeScan(LogStorageScanState &state) const override;

protected:
	//! Stores the chunk for duckdb_logs and (for LOG_ENTRIES) forwards it to Python.
	void FlushChunk(LoggingTargetTable table, DataChunk &chunk) override;
	//! Clears the in-memory buffers.
	void ResetAllBuffers() override;

private:
	ColumnDataCollection &GetBuffer(LoggingTargetTable table) const;
	//! Forwards each row of a LOG_ENTRIES chunk to logging.getLogger("duckdb"). Never throws.
	void ForwardEntriesToPython(DataChunk &chunk);

	map<LoggingTargetTable, unique_ptr<ColumnDataCollection>> log_storage_buffers;
};

} // namespace duckdb
