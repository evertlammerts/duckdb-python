#include "duckdb_python/python_log_storage.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

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

//===--------------------------------------------------------------------===//
// Asynchronous forwarder
//
// The engine invokes FlushChunk while holding LogManager::lock — a non-recursive mutex that is
// also taken by LogManager::CreateLogger / WriteLogEntry / Flush. Acquiring the GIL from inside
// that lock deadlocks: a worker thread holding the lock blocks on the GIL, while another thread
// holding the GIL blocks on the lock (e.g. via CreateLogger at the start of a concurrent query).
// We observed exactly this with two threads each running execute() on one database.
//
// So forwarding is decoupled. FlushChunk only copies plain (level, message) data into this
// process-global queue (no GIL, no Python). A single background thread drains the queue and
// forwards to logging.getLogger("duckdb") with the GIL held but NO engine lock held — breaking
// the lock-ordering cycle. One global thread (not one per DatabaseInstance) avoids spawning a
// thread per connection. The queue holds owned copies, so it is independent of any storage's
// lifetime.
//===--------------------------------------------------------------------===//
namespace {

struct PendingLogEntry {
	int level;
	string message;
};

struct LogForwarder {
	std::mutex mutex;                // guards the fields below; NEVER held while acquiring the GIL
	std::condition_variable cv;      // forwarder waits here for work
	std::condition_variable idle_cv; // drainers wait here for the queue to empty
	vector<PendingLogEntry> queue;
	bool stop = false;
	bool started = false;
	bool busy = false; // a batch has been dequeued but not yet forwarded
	std::thread thread;
};

LogForwarder &GetForwarder() {
	static LogForwarder forwarder;
	return forwarder;
}

void ForwarderLoop() {
	auto &fwd = GetForwarder();
	while (true) {
		vector<PendingLogEntry> batch;
		{
			std::unique_lock<std::mutex> lck(fwd.mutex);
			fwd.cv.wait(lck, [&fwd] { return fwd.stop || !fwd.queue.empty(); });
			if (fwd.stop && fwd.queue.empty()) {
				return;
			}
			batch.swap(fwd.queue);
			fwd.busy = true; // queue is empty again, but this batch isn't delivered yet
		}
		// No engine lock and no forwarder lock held here, so acquiring the GIL cannot deadlock.
		if (Py_IsInitialized()) { // else interpreter is finalizing — acquiring the GIL would crash
			try {
				py::gil_scoped_acquire gil;
				auto logging = py::module::import("logging");
				auto logger = logging.attr("getLogger")("duckdb");
				for (auto &entry : batch) {
					logger.attr("log")(entry.level, entry.message);
				}
			} catch (...) {
				// Logging must never disrupt anything.
			}
		}
		{
			std::unique_lock<std::mutex> lck(fwd.mutex);
			fwd.busy = false;
			fwd.idle_cv.notify_all(); // wake any DrainForwarder() waiters
		}
	}
}

// atexit callback: stop and join the forwarder while the interpreter is still alive. Runs on the
// main thread with the GIL held; the GIL is released around join() because the forwarder may be
// parked in take_gil and could not otherwise wake to observe `stop`.
void StopForwarder() {
	auto &fwd = GetForwarder();
	{
		std::unique_lock<std::mutex> lck(fwd.mutex);
		if (!fwd.started) {
			return;
		}
		fwd.stop = true;
	}
	fwd.cv.notify_all();
	if (fwd.thread.joinable()) {
		py::gil_scoped_release release;
		fwd.thread.join();
	}
}

} // namespace

void PythonLogStorage::EnsureForwarderStarted() {
	// Called from Connect() with the GIL held and no engine lock held.
	auto &fwd = GetForwarder();
	{
		std::unique_lock<std::mutex> lck(fwd.mutex);
		if (fwd.started) {
			return;
		}
		fwd.started = true;
		fwd.thread = std::thread(ForwarderLoop);
	}
	// Stop+join before interpreter finalization. Joining a GIL-blocked thread after Py_Finalize
	// would crash, so we hook atexit (which runs while the interpreter is still valid).
	try {
		auto atexit = py::module::import("atexit");
		atexit.attr("register")(py::cpp_function([]() { StopForwarder(); }));
	} catch (...) {
	}
}

void PythonLogStorage::DrainForwarder() {
	auto &fwd = GetForwarder();
	// Release the GIL while waiting: the forwarder thread needs it to finish its in-flight batch
	// and signal idle. Holding it here would deadlock the very thread we're waiting on.
	py::gil_scoped_release release;
	std::unique_lock<std::mutex> lck(fwd.mutex);
	fwd.idle_cv.wait(lck, [&fwd] { return fwd.queue.empty() && !fwd.busy; });
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

void PythonLogStorage::EnqueueEntriesForPython(DataChunk &chunk) {
	// Runs under LogManager::lock (and our scan lock). It MUST NOT touch the GIL or call Python:
	// doing so here would deadlock against any thread that holds the GIL and then enters a
	// LogManager method that needs the same lock (CreateLogger / WriteLogEntry / Flush). So we
	// only copy plain data into the global queue; the forwarder thread does the Python work
	// lock-free. The strings are deep-copied (GetString), so they outlive this chunk.
	//
	// A side benefit of decoupling: a user logging handler that raises now runs on the forwarder
	// thread, where the exception is swallowed — it can never reach the engine's query path.
	//
	// LOG_ENTRIES schema: context_id, timestamp, type, log_level (idx 3), message (idx 4).
	// log_level and message are both VARCHAR; the buffer chunk is flat.
	auto level_data = FlatVector::GetData<string_t>(chunk.data[3]);
	auto message_data = FlatVector::GetData<string_t>(chunk.data[4]);
	auto &fwd = GetForwarder();
	{
		std::unique_lock<std::mutex> lck(fwd.mutex);
		for (idx_t i = 0; i < chunk.size(); i++) {
			fwd.queue.push_back({LevelStringToPython(level_data[i].GetString()), message_data[i].GetString()});
		}
	}
	fwd.cv.notify_one();
}

void PythonLogStorage::FlushChunk(LoggingTargetTable table, DataChunk &chunk) {
	D_ASSERT(table == LoggingTargetTable::LOG_ENTRIES || table == LoggingTargetTable::LOG_CONTEXTS);
	// Retain the entry for duckdb_logs FIRST, so a misbehaving Python handler can never cost
	// us a stored row.
	log_storage_buffers[table]->Append(chunk);
	// Queue only real log entries (not context metadata) for async forwarding to logging.
	if (table == LoggingTargetTable::LOG_ENTRIES) {
		EnqueueEntriesForPython(chunk);
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
