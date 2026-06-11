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
//! It subclasses BufferingLogStorage with a buffer size of 1 so each entry is flushed
//! immediately, rather than batched until a 2048-entry buffer fills — engine WARNINGs are
//! sparse and must surface promptly to be useful.
//!
//! Forwarding to Python is ASYNCHRONOUS. The engine calls FlushChunk while holding
//! LogManager::lock (a non-recursive mutex also taken by CreateLogger/WriteLogEntry). Acquiring
//! the GIL there would deadlock against any other thread that holds the GIL and then enters one
//! of those LogManager methods (i.e. ordinary concurrent queries). So FlushChunk only copies
//! (level, message) into a process-global queue, and a single background thread — which holds
//! no engine lock — drains it and forwards to `logging`. See python_log_storage.cpp.
class PythonLogStorage : public BufferingLogStorage {
public:
	explicit PythonLogStorage(DatabaseInstance &db);
	~PythonLogStorage() override;

	const string GetStorageName() override {
		return "python_log_storage";
	}

	//! Starts the process-global forwarder thread (idempotent). MUST be called with the GIL held
	//! and no engine lock held — i.e. from Connect(), never from the engine log-write path.
	static void EnsureForwarderStarted();

	//! Blocks (releasing the GIL) until every queued entry has been forwarded to `logging`.
	//! Forwarding is asynchronous, so callers that need to observe a just-emitted warning on the
	//! Python side must drain first. Exposed to Python as `_duckdb._drain_log_forwarding`
	//! for deterministic tests; harmless if the forwarder was never started.
	static void DrainForwarder();

	//! Single-threaded scan interface — mirrors InMemoryLogStorage so duckdb_logs can read us.
	bool CanScan(LoggingTargetTable table) override;
	unique_ptr<LogStorageScanState> CreateScanState(LoggingTargetTable table) const override;
	bool Scan(LogStorageScanState &state, DataChunk &result) const override;
	void InitializeScan(LogStorageScanState &state) const override;

protected:
	//! Stores the chunk for duckdb_logs and (for LOG_ENTRIES) queues it for async forwarding.
	void FlushChunk(LoggingTargetTable table, DataChunk &chunk) override;
	//! Clears the in-memory buffers.
	void ResetAllBuffers() override;

private:
	ColumnDataCollection &GetBuffer(LoggingTargetTable table) const;
	//! Copies each row of a LOG_ENTRIES chunk into the global forward queue. Never touches the
	//! GIL or calls Python (it runs under LogManager::lock). Never throws.
	void EnqueueEntriesForPython(DataChunk &chunk);

	map<LoggingTargetTable, unique_ptr<ColumnDataCollection>> log_storage_buffers;
};

} // namespace duckdb
