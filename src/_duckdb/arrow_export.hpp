//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrow_export.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lifetime.hpp"

namespace duckdb_python {

/// A query result leaving as an Arrow C stream, handed out once as an "arrow_array_stream" capsule.
///
/// The capsule's callbacks hold the result; the engine's worker threads produce into its buffer, and a read
/// pops, lends the calling thread, or parks. Reads never hold the GIL across engine work, so a Python scalar
/// function inside the query can finish. One reader at a time, as the Arrow stream interface requires; Close
/// is the one safe cross-thread call.
class ArrowStream {
public:
	/// Error codes `Error()` reports beside the engine's own: the stream was closed under the consumer, a
	/// pending Ctrl-C ended the read and was re-armed for Python's next check, or Python raised during one.
	static constexpr int kClosed = -1;
	static constexpr int kKeyboardInterrupt = -2;
	static constexpr int kPythonError = -3;

	/// The capsule callbacks live outside the class, so the state they share is spelled here.
	struct State;

	/// `kept` is what the running query may call back into, the database's registered scalar functions,
	/// pinned by the stream so they outlive their Database for as long as the consumer holds the capsule.
	ArrowStream(nb::object database, std::shared_ptr<ModuleState> module, std::weak_ptr<cxx::Connection> connection,
	            cxx::ArrowResult result, std::vector<nb::object> kept);
	~ArrowStream();
	ArrowStream(const ArrowStream &) = delete;
	ArrowStream &operator=(const ArrowStream &) = delete;

	nb::handle Parent() const {
		return database;
	}

	/// The capsule, once; a stream reads front to back exactly once, so a second take would lose rows silently.
	nb::object Capsule(nb::handle requested_schema);

	/// What ended the stream, as (code, message), or None; codes above zero are the engine's.
	std::optional<std::pair<int, std::string>> Error() const;

	/// Whether the query is still open: false once read out, released by the consumer, or closed. What lets a
	/// tracker drop finished streams instead of keeping every handle until the connection closes.
	bool Live() const;

	/// Whether a close was requested but the stream is still open: a close from an engine thread may only
	/// cancel and defer, so its connection completes the teardown before its next statement.
	bool ClosePending() const;

	/// End the query rather than wait for collection; repeatable, and a later read reports a closed stream.
	/// Cancels a read in flight and waits for its teardown like a join, so the connection is free on return:
	/// unbounded, since a silent bound would break that promise, but polling for Ctrl-C, which raises here
	/// with the request left standing. On a thread the read may itself be waiting for, the wait and the
	/// teardown are skipped entirely and the request stands; a standing request is completed by the read it
	/// cancelled, by whoever touches the stream next, or by the connection before its next statement. Also
	/// drops this handle's own Database reference, so a closed stream kept around pins nothing.
	void Close();

	/// The garbage collector's visit: this handle's Database reference, and the shared state's as well while
	/// no capsule is out, so a cycle through an unconsumed stream stays collectable.
	int Traverse(visitproc visit, void *arg) const;

	/// The garbage collector's clear: a capsule already handed out keeps reading, an unconsumed stream ends,
	/// and the database reference is dropped either way to break the cycle.
	void GcClear();

private:
	void CloseInternal() noexcept;

	nb::ft_mutex handle_lock;
	nb::object database;
	std::shared_ptr<ModuleState> module;
	std::shared_ptr<State> state;
};

} // namespace duckdb_python
