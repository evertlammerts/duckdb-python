#include "duckdb_python/python_log_storage.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

// Maps the engine's textual log level (stored as VARCHAR in the LOG_ENTRIES chunk) to the
// integer levels of Python's logging module.
static int LevelStringToPython(const string &level_str) {
	if (level_str == "TRACE" || level_str == "DEBUG") {
		return 10; // logging.DEBUG
	}
	if (level_str == "INFO") {
		return 20; // logging.INFO
	}
	if (level_str == "WARNING") {
		return 30; // logging.WARNING
	}
	if (level_str == "ERROR") {
		return 40; // logging.ERROR
	}
	if (level_str == "FATAL") {
		return 50; // logging.CRITICAL
	}
	return 30;
}

PythonLogStorage::PythonLogStorage(DatabaseInstance &db) : BufferingLogStorage(db, 1, true) {
	log_storage_buffers[LoggingTargetTable::LOG_ENTRIES] =
	    make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), GetSchema(LoggingTargetTable::LOG_ENTRIES));
	log_storage_buffers[LoggingTargetTable::LOG_CONTEXTS] =
	    make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), GetSchema(LoggingTargetTable::LOG_CONTEXTS));
}

PythonLogStorage::~PythonLogStorage() {
}

ColumnDataCollection &PythonLogStorage::GetBuffer(LoggingTargetTable table) const {
	auto res = log_storage_buffers.find(table);
	if (res == log_storage_buffers.end()) {
		throw InternalException("PythonLogStorage: failed to find buffer for logging target table");
	}
	return *res->second;
}

void PythonLogStorage::ForwardEntriesToPython(DataChunk &chunk) {
	// This fires from engine worker threads with the GIL released, and from within both the
	// LogManager lock and this storage's lock. It runs arbitrary user Python (logging
	// handlers) and MUST NOT let an exception escape: the engine calls the write path with no
	// try/catch, directly from query binding/execution, so a raising handler would otherwise
	// fail the user's query. Hence we swallow everything here.
	//
	// Caveat: because a lock is held across this call, a handler that re-enters DuckDB on the
	// same thread and emits another log entry can self-deadlock on the non-recursive lock.
	// That is outside our control (and matches the engine's own contract for log storages).
	if (!Py_IsInitialized()) {
		return; // interpreter is finalizing — acquiring the GIL would crash
	}
	try {
		py::gil_scoped_acquire gil;
		auto logging = py::module::import("logging");
		auto logger = logging.attr("getLogger")("duckdb");
		// LOG_ENTRIES schema: context_id, timestamp, type, log_level (idx 3), message (idx 4).
		// log_level and message are both VARCHAR; the buffer chunk is flat.
		auto level_data = FlatVector::GetData<string_t>(chunk.data[3]);
		auto message_data = FlatVector::GetData<string_t>(chunk.data[4]);
		for (idx_t i = 0; i < chunk.size(); i++) {
			logger.attr("log")(LevelStringToPython(level_data[i].GetString()), message_data[i].GetString());
		}
	} catch (...) {
		// Logging must never disrupt query execution.
	}
}

void PythonLogStorage::FlushChunk(LoggingTargetTable table, DataChunk &chunk) {
	D_ASSERT(table == LoggingTargetTable::LOG_ENTRIES || table == LoggingTargetTable::LOG_CONTEXTS);
	// Retain the entry for duckdb_logs FIRST, so a misbehaving Python handler can never cost
	// us a stored row.
	log_storage_buffers[table]->Append(chunk);
	// Forward only real log entries (not context metadata) to Python's logging module.
	if (table == LoggingTargetTable::LOG_ENTRIES) {
		ForwardEntriesToPython(chunk);
	}
}

void PythonLogStorage::ResetAllBuffers() {
	BufferingLogStorage::ResetAllBuffers();
	for (const auto &buffer : log_storage_buffers) {
		buffer.second->Reset();
	}
}

bool PythonLogStorage::CanScan(LoggingTargetTable table) {
	unique_lock<mutex> lck(lock);
	return IsEnabledInternal(table);
}

unique_ptr<LogStorageScanState> PythonLogStorage::CreateScanState(LoggingTargetTable table) const {
	return make_uniq<PythonLogStorageScanState>(table);
}

bool PythonLogStorage::Scan(LogStorageScanState &state, DataChunk &result) const {
	unique_lock<mutex> lck(lock);
	auto &python_scan_state = state.Cast<PythonLogStorageScanState>();
	return GetBuffer(python_scan_state.table).Scan(python_scan_state.scan_state, result);
}

void PythonLogStorage::InitializeScan(LogStorageScanState &state) const {
	unique_lock<mutex> lck(lock);
	auto &python_scan_state = state.Cast<PythonLogStorageScanState>();
	GetBuffer(python_scan_state.table)
	    .InitializeScan(python_scan_state.scan_state, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
}

} // namespace duckdb
