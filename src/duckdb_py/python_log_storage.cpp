#include "duckdb_python/python_log_storage.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/logging/logging.hpp"

namespace duckdb {

static int LogLevelToPython(LogLevel level) {
	switch (level) {
	case LogLevel::LOG_TRACE:
	case LogLevel::LOG_DEBUG:
		return 10; // logging.DEBUG
	case LogLevel::LOG_INFO:
		return 20; // logging.INFO
	case LogLevel::LOG_WARNING:
		return 30; // logging.WARNING
	case LogLevel::LOG_ERROR:
		return 40; // logging.ERROR
	case LogLevel::LOG_FATAL:
		return 50; // logging.CRITICAL
	default:
		return 30;
	}
}

static int LevelStringToPython(const string &level_str) {
	if (level_str == "TRACE" || level_str == "DEBUG") {
		return 10;
	}
	if (level_str == "INFO") {
		return 20;
	}
	if (level_str == "WARNING") {
		return 30;
	}
	if (level_str == "ERROR") {
		return 40;
	}
	if (level_str == "FATAL") {
		return 50;
	}
	return 30;
}

// Both write methods run on engine worker threads and invoke arbitrary user Python (the
// handlers installed on the "duckdb" logger). The engine calls these directly from query
// binding/execution with NO surrounding try/catch (see LogManager::WriteLogEntry), so an
// exception escaping here would fail the user's query. Logging is a side effect — it must
// never do that. Hence every body swallows all exceptions.
//
// Note also that the engine holds LogManager::lock (a non-recursive mutex) across this call.
// A handler that re-enters DuckDB on the same thread and emits another log entry would
// self-deadlock on that lock — outside our control, but worth knowing.

void PythonLogStorage::WriteLogEntry(timestamp_t, LogLevel level, const string &, const string &log_message,
                                     const RegisteredLoggingContext &) {
	if (!Py_IsInitialized()) {
		return; // interpreter is finalizing — acquiring the GIL would crash
	}
	try {
		py::gil_scoped_acquire gil;
		auto logging = py::module::import("logging");
		auto logger = logging.attr("getLogger")("duckdb");
		logger.attr("log")(LogLevelToPython(level), log_message);
	} catch (...) {
		// Logging must not disrupt query execution.
	}
}

void PythonLogStorage::WriteLogEntries(DataChunk &chunk, const RegisteredLoggingContext &) {
	if (!Py_IsInitialized()) {
		return; // interpreter is finalizing — acquiring the GIL would crash
	}
	try {
		py::gil_scoped_acquire gil;
		auto logging = py::module::import("logging");
		auto logger = logging.attr("getLogger")("duckdb");
		// DataChunk is in LOG_ENTRIES format: context_id, timestamp, type, log_level, message.
		// log_level (idx 3) and message (idx 4) are both VARCHAR; the chunk is freshly
		// allocated by the engine so the vectors are flat.
		auto level_data = FlatVector::GetData<string_t>(chunk.data[3]);
		auto message_data = FlatVector::GetData<string_t>(chunk.data[4]);
		for (idx_t i = 0; i < chunk.size(); i++) {
			logger.attr("log")(LevelStringToPython(level_data[i].GetString()), message_data[i].GetString());
		}
	} catch (...) {
		// Logging must not disrupt query execution.
	}
}

} // namespace duckdb
