//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/python_log_storage.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/logging/log_storage.hpp"
#include "duckdb/logging/logging.hpp"

namespace duckdb {

class PythonLogStorage : public LogStorage {
public:
	PythonLogStorage() = default;
	~PythonLogStorage() override = default;

	const string GetStorageName() override {
		return "python_log_storage";
	}

	void WriteLogEntry(timestamp_t timestamp, LogLevel level, const string &log_type, const string &log_message,
	                   const RegisteredLoggingContext &context) override;
	void WriteLogEntries(DataChunk &chunk, const RegisteredLoggingContext &context) override;
	void FlushAll() override {
	}
	void Flush(LoggingTargetTable table) override {
	}
	bool IsEnabled(LoggingTargetTable table) override {
		return true;
	}
};

} // namespace duckdb
