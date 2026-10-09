//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrow_export.cpp
//
//
//===----------------------------------------------------------------------===//

#include "arrow_export.hpp"

#include "arrowc.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace duckdb_python {

struct ArrowStream::State {
	/// Serialises reads against Close from another thread. Never acquired while holding the GIL: an engine call
	/// under it may wait for a Python scalar function, which needs the GIL to finish. Never waited on by a
	/// close either: that function may itself be blocked on this very close.
	std::mutex lock;
	/// The thread holding `lock`, so Python code reached from under it, a signal handler or a finalizer, can
	/// recognise reentrancy instead of deadlocking on its own lock.
	std::atomic<std::thread::id> owner {};
	std::optional<cxx::ArrowResult> result;
	/// For ending the read in flight instead of waiting for it; the first batch of a heavy pipeline may
	/// otherwise be hours away.
	std::weak_ptr<cxx::Connection> connection;
	/// True while a read runs engine work under the lock. Close interrupts the query only when this is set:
	/// contention from an error or liveness poll must not cancel whatever the connection runs next.
	std::atomic<bool> reading {false};
	/// A close that could not take the lock leaves its request here; the read in flight acts on it at its next
	/// pace point and finishes the teardown itself.
	std::atomic<bool> close_requested {false};
	/// Callbacks currently anywhere inside the stream, teardown included: a release must know frames remain
	/// even when the mutex is free, since the teardown's own object drops can run finalizers that come right
	/// back into the stream.
	std::atomic<int> callbacks {0};
	/// The Python scalar functions registered when the query started, pinned so they survive their Database
	/// however long the consumer holds the capsule, and dropped the moment the stream ends. The engine only
	/// borrows them, and the sources a bound scan reads pin themselves. Only ever moved with the GIL held, so
	/// the garbage collector's traversal can read it.
	std::vector<nb::object> kept;
	/// The schema, captured before the result is destroyed at the end of the stream: the Arrow contract keeps
	/// get_schema answerable until the consumer releases the stream.
	ArrowSchema schema_cache {};
	/// Signals that the engine result was destroyed, under its own mutex so a close can wait for the reader's
	/// teardown without touching the state lock that reader holds.
	std::mutex done_lock;
	std::condition_variable done_cv;
	bool torn_down = false;
	int code = 0;
	/// Owned here so get_last_error's pointer outlives the callback that failed.
	std::string message;
	bool capsule_taken = false;
	/// Set the moment the engine reports the query over, BEFORE any work that follows: the engine frees the
	/// connection's slot before reporting, so from then on no path may interrupt the connection, where the
	/// next statement may already run. Also what keeps a read-out stream answering with a clean end.
	std::atomic<bool> exhausted {false};

	/// The first cause wins: a read after a Ctrl-C or an engine error must not relabel it as "closed".
	void Record(int what, std::string text) {
		if (code == 0) {
			code = what;
			message = std::move(text);
		}
	}

	bool OwnedByThisThread() const {
		return owner.load() == std::this_thread::get_id();
	}

	~State() {
		if (schema_cache.release != nullptr) {
			schema_cache.release(&schema_cache);
		}
	}
};

namespace {

/// What the capsule's private_data points at. The Python objects the stream must keep alive live in the shared
/// state, so ending the stream can drop them before the consumer lets go of the capsule. The module reference
/// is where a holder goes when it cannot be freed; it is dropped on the way in, or the quarantine would keep
/// its own keeper alive.
struct Holder {
	std::shared_ptr<ArrowStream::State> state;
	std::shared_ptr<ModuleState> module;
};

/// A copy, not a reference: a release reached from Python running inside a callback deletes the holder while
/// that callback is still on the stack.
std::shared_ptr<ArrowStream::State> StateOf(ArrowArrayStream &stream) {
	return static_cast<Holder *>(stream.private_data)->state;
}

/// Takes the state lock with the GIL released, so a close waiting out an engine call cannot starve the Python
/// scalar function that call waits for.
std::unique_lock<std::mutex> LockWithoutGil(ArrowStream::State &state) {
	nb::gil_scoped_release no_gil;
	return std::unique_lock<std::mutex>(state.lock);
}

/// LockWithoutGil plus the owner mark that lets reentrant Python recognise this thread already holds the lock.
class StateLock {
public:
	explicit StateLock(ArrowStream::State &state) : guard(LockWithoutGil(state)), state(state) {
		state.owner.store(std::this_thread::get_id());
	}
	~StateLock() {
		state.owner.store(std::thread::id {});
	}
	StateLock(const StateLock &) = delete;
	StateLock &operator=(const StateLock &) = delete;

private:
	// Declared first so the owner mark above is cleared before the mutex is released.
	std::unique_lock<std::mutex> guard;
	ArrowStream::State &state;
};

/// Marks a read for the span it runs engine work under the lock, so a close can tell an unbounded read from
/// brief bookkeeping contention.
struct ReadingGuard {
	explicit ReadingGuard(ArrowStream::State &state) : state(state) {
		state.reading.store(true);
	}
	~ReadingGuard() {
		state.reading.store(false);
	}
	ArrowStream::State &state;
};

/// What ending a stream still has to do once the lock falls: the engine teardown may wait for a Python scalar
/// function, and dropping the pins runs arbitrary finalizers, so neither belongs under the mutex.
struct Teardown {
	std::optional<cxx::ArrowResult> result;
	std::vector<nb::object> pins;
	std::weak_ptr<cxx::Connection> connection;
	bool interrupt = false;
};

/// Counts a callback for its whole stay. Decremented last thing before the callback returns.
struct CallbackGuard {
	explicit CallbackGuard(ArrowStream::State &state) : state(state) {
		state.callbacks.fetch_add(1);
	}
	~CallbackGuard() {
		state.callbacks.fetch_sub(1);
	}
	ArrowStream::State &state;
};

/// Empties the stream under the caller's lock and GIL; FinishTeardown completes the job after the lock falls.
/// Whether the query gets cancelled first is decided here, in one place: only while it still owns the
/// connection's slot. `cancel` is false only when the engine itself ended the query with an error, where the
/// slot is already free and an interrupt could hit whatever statement comes next; the exhausted mark covers
/// the same hazard for a stream the engine finished.
Teardown Take(ArrowStream::State &state, bool cancel) {
	Teardown teardown;
	teardown.result = std::move(state.result);
	state.result.reset();
	teardown.pins = std::move(state.kept);
	state.kept.clear();
	teardown.connection = state.connection;
	// KNOWN RESIDUAL: the engine releases the connection's slot inside Step, before the exhausted mark or a
	// thrown error reaches us, and a cancel decided in that gap can hit the connection's next statement; the
	// interrupt is connection-wide, so only a result-scoped engine interrupt could close this.
	teardown.interrupt = cancel && teardown.result.has_value() && !state.exhausted.load();
	return teardown;
}

/// Caller holds the GIL and not the lock. Destroying a running result waits for its in-flight tasks to end on
/// their own, which for a pipeline breaker is the rest of the query, so the interrupt comes first; the pins
/// die last, under the GIL and free of the lock, since dropping them may run arbitrary finalizers. The fence
/// keeps a close those finalizers reach from waiting on this very thread.
void FinishTeardown(ArrowStream::State &state, Teardown teardown) {
	BusyFence fence;
	if (teardown.result) {
		{
			nb::gil_scoped_release no_gil;
			if (teardown.interrupt) {
				if (auto connection = teardown.connection.lock()) {
					connection->Interrupt();
				}
			}
			teardown.result.reset();
		}
		{
			std::lock_guard<std::mutex> mark(state.done_lock);
			state.torn_down = true;
		}
		state.done_cv.notify_all();
	}
}

/// What ShutdownQuery established, for the caller to act on.
struct Shutdown {
	/// The lock came at once with no callback anywhere inside: nothing can be using the capsule's struct.
	bool quiet = false;
	/// The teardown is known finished. False only on a thread the close may not wait on; the request then
	/// stands, Live() keeps answering true, and the frame this thread interrupted completes it.
	bool complete = false;
};

/// The one shutdown path for a stream's query, shared by every close and release. Sets the request, cancels
/// the query while it still owns the connection's slot, and tears down when the lock can be had without
/// waiting on a thread that may be waiting on this one; otherwise the read in flight finishes the job at its
/// next pace point. With `wait`, blocks until that teardown is done, so the connection is free on return;
/// the wait is skipped only where it would deadlock, on a thread the read may itself be waiting for: the
/// lock's owner, a thread inside a user scalar function, or one inside a stream teardown. The caller holds
/// the GIL.
Shutdown ShutdownQuery(ArrowStream::State &state, bool wait) {
	Shutdown shutdown;
	state.close_requested.store(true);
	if (state.OwnedByThisThread()) {
		// Reached from Python running under this stream's lock. The frame holding the lock finishes the job
		// at its next pace point; the query is cancelled now, while the connection reference is still good,
		// since that frame's own connection may be closed before it resumes. This thread holds the lock, so
		// reading the result is safe.
		if (state.result && !state.exhausted.load()) {
			if (auto connection = state.connection.lock()) {
				connection->Interrupt();
			}
		}
		return shutdown;
	}
	if (BusyFence::EngineTask()) {
		// An engine task must never destroy a result: destroying one waits for the engine's in-flight tasks,
		// and this thread is one, so it would wait on itself. Cancel only; the interrupted read, or a later
		// closer off the engine, completes the job. With the lock in hand the interrupt fires under it, where
		// a present result proves the slot is this stream's; Interrupt does not block.
		nb::gil_scoped_release no_gil;
		std::unique_lock<std::mutex> guard(state.lock, std::try_to_lock);
		if (guard.owns_lock()) {
			if (state.result.has_value() && !state.exhausted.load()) {
				if (auto connection = state.connection.lock()) {
					connection->Interrupt();
				}
			}
		} else if (state.reading.load() && !state.exhausted.load()) {
			if (auto connection = state.connection.lock()) {
				connection->Interrupt();
			}
		}
		return shutdown;
	}
	std::unique_lock<std::mutex> guard;
	{
		nb::gil_scoped_release no_gil;
		guard = std::unique_lock<std::mutex>(state.lock, std::try_to_lock);
		shutdown.quiet = guard.owns_lock() && state.callbacks.load() == 0;
		while (!guard.owns_lock()) {
			// A read in flight holds the lock for as long as its next piece of work takes: cancel its query
			// and leave the teardown to it, rather than wait for a thread that may be waiting for us.
			// Anything else holding the lock is brief, and interrupting on mere contention could cancel
			// whatever the connection runs after this stream, so without the reading mark this polls. A read
			// that finishes between the load and the interrupt leaves the interrupt on an idle connection,
			// where the next statement's own setup clears it; only a statement already past that setup in the
			// same instant could see it, a window the engine's API cannot close.
			if (state.reading.load()) {
				if (!state.exhausted.load()) {
					if (auto connection = state.connection.lock()) {
						connection->Interrupt();
					}
				}
				break;
			}
			std::this_thread::sleep_for(std::chrono::microseconds(100));
			guard.try_lock();
		}
	}
	if (guard.owns_lock()) {
		Teardown teardown = Take(state, true);
		guard.unlock();
		shutdown.complete = teardown.result.has_value();
		FinishTeardown(state, std::move(teardown));
		if (shutdown.complete) {
			return shutdown;
		}
		// The result was already taken: someone else is mid-teardown; their finish is what `wait` waits for.
	}
	if (wait && !BusyFence::Here()) {
		// Unbounded, like joining the read: a silent bound would break the free-connection promise. A cycle
		// the user builds, a callback waiting on this very thread, hangs here as any join would; the slices
		// poll for Ctrl-C so the main thread can break out loudly, with the request left standing for the
		// connection to complete.
		for (;;) {
			{
				nb::gil_scoped_release no_gil;
				std::unique_lock<std::mutex> waiter(state.done_lock);
				if (state.done_cv.wait_for(waiter, std::chrono::milliseconds(100), [&] { return state.torn_down; })) {
					shutdown.complete = true;
					break;
				}
			}
			if (PyErr_CheckSignals() != 0) {
				throw nb::python_error();
			}
		}
	}
	return shutdown;
}

/// One look at Python's pending signals, taken between engine steps so a Ctrl-C can end a read whose next
/// array is still far away. Returns 0 when nothing is pending, or EINTR after a Ctrl-C, with the stream ended
/// and the signal re-armed for Python's next check; anything else a handler raised leaves as nb::python_error.
/// An array already produced is released first: its ownership transfers only on a 0 return.
int HandlePendingSignals(ArrowStream::State &state, ArrowArray *array, Teardown &teardown) {
	if (PyErr_CheckSignals() == 0) {
		return 0;
	}
	if (array != nullptr && array->release != nullptr) {
		array->release(array);
	}
	if (PyErr_ExceptionMatches(PyExc_KeyboardInterrupt) == 0) {
		throw nb::python_error();
	}
	PyErr_Clear();
	PyErr_SetInterrupt();
	state.Record(ArrowStream::kKeyboardInterrupt, "interrupted");
	teardown = Take(state, true);
	return EINTR;
}

/// Acts on a close requested while this read held the lock; the array produced meanwhile goes back.
int HandleCloseRequest(ArrowStream::State &state, ArrowArray *array, Teardown &teardown) {
	if (!state.close_requested.load()) {
		return 0;
	}
	if (array != nullptr && array->release != nullptr) {
		array->release(array);
	}
	state.Record(ArrowStream::kClosed, "result is closed");
	teardown = Take(state, true);
	return EIO;
}

/// Runs a callback body under the GIL and turns every exception into the stream's error return, so nothing
/// crosses the C boundary; the body releases the GIL itself around engine work. An error ends the stream: the
/// engine's are sticky, and a dead stream must not keep its connection busy or its database pinned. The
/// teardown itself runs after the lock falls. `gone` answers for a stream that already ended.
template <class G, class F>
int Guarded(ArrowStream::State &state, G &&gone, F &&body) {
	nb::gil_scoped_acquire gil;
	// Fenced: Python reached from in here, a signal handler or a finalizer, may close ANOTHER stream whose
	// reader could in turn be waiting on work behind this very callback; such a close must not wait.
	BusyFence fence;
	CallbackGuard counted(state);
	try {
		if (state.OwnedByThisThread()) {
			// Reentered from Python running inside this stream's own callback; a second read cannot be
			// served, and recording would mislabel a stream that is otherwise fine.
			return EIO;
		}
		Teardown teardown;
		int status = EIO;
		try {
			StateLock guard(state);
			if (!state.result) {
				return gone(state);
			}
			if (const int closed = HandleCloseRequest(state, nullptr, teardown); closed != 0) {
				status = closed;
			} else {
				ReadingGuard reading(state);
				status = body(state, teardown);
				// A close that raced in after the body's last look leaves its request here; honoring it under
				// the same hold of the lock means no request is ever stranded with a live query. The array a
				// successful body handed out stays valid: arrays outlive their result.
				if (!teardown.result && state.result && state.close_requested.load()) {
					state.Record(ArrowStream::kClosed, "result is closed");
					teardown = Take(state, true);
				}
			}
		} catch (nb::python_error &error) {
			// Described under the GIL first; the lock fell with the unwinding, so recording re-takes it.
			auto described = DescribePythonError(error);
			StateLock guard(state);
			state.Record(ArrowStream::kPythonError, std::move(described));
			teardown = Take(state, true);
		} catch (const cxx::Exception &error) {
			StateLock guard(state);
			if (state.close_requested.load()) {
				// A close's cancel ended the read; "closed" is the cause the closer promised, whatever the
				// engine's exact code, so the label does not depend on which check saw the close first.
				state.Record(ArrowStream::kClosed, "result is closed");
			} else {
				state.Record(error.GetCode(), error.what());
			}
			// The engine ended this query itself, so no interrupt: the connection's slot may already be free
			// and a new statement running in it.
			teardown = Take(state, false);
		} catch (const std::exception &error) {
			StateLock guard(state);
			state.Record(ArrowStream::kPythonError, error.what());
			teardown = Take(state, true);
		}
		FinishTeardown(state, std::move(teardown));
		return status;
	} catch (...) {
		return EINVAL;
	}
}

int RecordClosed(ArrowStream::State &state) {
	state.Record(ArrowStream::kClosed, "result is closed");
	return EIO;
}

/// One bounded piece of engine work, with the GIL dropped. WAITING alone does not say whether the step ran
/// anything: a step that returns at once found nothing to claim, one that took real time did work and
/// deserves an immediate successor, or a consumer driving the whole query alone would crawl.
cxx::StepStatus TimedStep(ArrowStream::State &state, ArrowArray &out, bool &idle) {
	nb::gil_scoped_release no_gil;
	const auto before = std::chrono::steady_clock::now();
	const auto status = state.result->Step(out);
	idle = std::chrono::steady_clock::now() - before < std::chrono::microseconds(20);
	return status;
}

/// After a WAITING step: a look at Python's signals about once a millisecond, so Ctrl-C keeps its cadence
/// however busy the steps are, and a nap only when the step found nothing to run. The result's own Wait would
/// park until the workers deliver the next array, which for a pipeline breaker is the whole query, beyond any
/// signal's reach. Returns 0 to keep stepping.
int PaceWaiting(ArrowStream::State &state, bool idle, std::chrono::steady_clock::time_point &last_signals,
                Teardown &teardown) {
	if (const int closed = HandleCloseRequest(state, nullptr, teardown); closed != 0) {
		return closed;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - last_signals >= std::chrono::milliseconds(1)) {
		if (const int signalled = HandlePendingSignals(state, nullptr, teardown); signalled != 0) {
			return signalled;
		}
		last_signals = now;
	}
	if (idle) {
		nb::gil_scoped_release no_gil;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return 0;
}

/// A deep copy's backing storage; the node's strings and children die with it, in its release callback.
struct OwnedSchema {
	std::string format;
	std::optional<std::string> name;
	std::optional<std::string> metadata;
	std::vector<ArrowSchema *> children;
	ArrowSchema *dictionary = nullptr;

	~OwnedSchema() {
		for (auto *child : children) {
			if (child != nullptr) {
				if (child->release != nullptr) {
					child->release(child);
				}
				delete child;
			}
		}
		if (dictionary != nullptr) {
			if (dictionary->release != nullptr) {
				dictionary->release(dictionary);
			}
			delete dictionary;
		}
	}
};

void ReleaseCopiedSchema(ArrowSchema *schema) {
	delete static_cast<OwnedSchema *>(schema->private_data);
	schema->release = nullptr;
}

/// The byte length of Arrow C metadata: a pair count, then length-prefixed keys and values.
std::size_t MetadataLength(const char *metadata) {
	const char *cursor = metadata;
	std::int32_t pairs = 0;
	std::memcpy(&pairs, cursor, sizeof(pairs));
	cursor += sizeof(pairs);
	for (std::int32_t pair = 0; pair < pairs * 2; pair++) {
		std::int32_t length = 0;
		std::memcpy(&length, cursor, sizeof(length));
		cursor += sizeof(length) + length;
	}
	return static_cast<std::size_t>(cursor - metadata);
}

/// A deep copy whose lifetime is independent of the result that produced the source.
void CopySchema(const ArrowSchema &source, ArrowSchema &out) {
	auto owned = std::make_unique<OwnedSchema>();
	owned->format = source.format == nullptr ? "" : source.format;
	if (source.name != nullptr) {
		owned->name.emplace(source.name);
	}
	if (source.metadata != nullptr) {
		owned->metadata.emplace(source.metadata, MetadataLength(source.metadata));
	}
	owned->children.reserve(static_cast<std::size_t>(source.n_children));
	for (std::int64_t child = 0; child < source.n_children; child++) {
		owned->children.push_back(nullptr);
		auto copy = std::make_unique<ArrowSchema>();
		*copy = ArrowSchema {};
		CopySchema(*source.children[child], *copy);
		owned->children.back() = copy.release();
	}
	if (source.dictionary != nullptr) {
		auto copy = std::make_unique<ArrowSchema>();
		*copy = ArrowSchema {};
		CopySchema(*source.dictionary, *copy);
		owned->dictionary = copy.release();
	}
	out = ArrowSchema {};
	out.flags = source.flags;
	out.n_children = source.n_children;
	out.format = owned->format.c_str();
	out.name = owned->name ? owned->name->c_str() : nullptr;
	out.metadata = owned->metadata ? owned->metadata->data() : nullptr;
	out.children = owned->children.empty() ? nullptr : owned->children.data();
	out.dictionary = owned->dictionary;
	out.private_data = owned.release();
	out.release = &ReleaseCopiedSchema;
}

/// After a clean end of stream the schema stays answerable from the capture, as the Arrow contract requires;
/// only a closed or failed stream reports closed.
int GoneGetSchema(ArrowStream::State &state, ArrowSchema *out) {
	if (state.exhausted.load() && state.schema_cache.release != nullptr) {
		CopySchema(state.schema_cache, *out);
		return 0;
	}
	return RecordClosed(state);
}

int ExportGetSchema(ArrowArrayStream *self, ArrowSchema *out) {
	auto gone = [&](ArrowStream::State &state) {
		return GoneGetSchema(state, out);
	};
	return Guarded(*StateOf(*self), gone, [&](ArrowStream::State &state, Teardown &teardown) {
		auto last_signals = std::chrono::steady_clock::now();
		for (;;) {
			try {
				nb::gil_scoped_release no_gil;
				state.result->GetSchema(*out);
				return 0;
			} catch (const cxx::Exception &error) {
				if (error.GetCode() != DUCKDB_V2_ERROR_INPUT_INVALID) {
					throw;
				}
			}
			// Metadata is pending: a statement that expands into several, such as PIVOT, runs the earlier
			// ones here, and no array can precede the metadata.
			ArrowArray scratch {};
			bool idle = false;
			const auto status = TimedStep(state, scratch, idle);
			if (scratch.release != nullptr) {
				scratch.release(&scratch);
			}
			switch (status) {
			case cxx::StepStatus::WAITING: {
				if (const int paced = PaceWaiting(state, idle, last_signals, teardown); paced != 0) {
					return paced;
				}
				break;
			}
			case cxx::StepStatus::CHUNK:
				// The array just released held rows: a schema handed out now would quietly lose them.
				state.Record(DUCKDB_V2_ERROR_INPUT_INVALID,
				             "Invalid Input Error: the engine produced an array before the schema was available");
				teardown = Take(state, true);
				return EIO;
			case cxx::StepStatus::CANCELLED:
				throw cxx::InterruptException("query was cancelled");
			default: {
				// FINISHED: the metadata is as known as it will get; surface what GetSchema says now. The
				// engine freed the connection's slot before reporting, hence the mark, so no close cancels
				// whatever the connection runs next.
				state.exhausted.store(true);
				nb::gil_scoped_release no_gil;
				state.result->GetSchema(*out);
				return 0;
			}
			}
		}
	});
}

/// A stream read to its end answers later reads with a clean end of stream, as the result itself would.
int GoneGetNext(ArrowStream::State &state, ArrowArray *out) {
	if (state.exhausted.load()) {
		*out = ArrowArray {};
		return 0;
	}
	return RecordClosed(state);
}

int ExportGetNext(ArrowArrayStream *self, ArrowArray *out) {
	auto gone = [&](ArrowStream::State &state) {
		return GoneGetNext(state, out);
	};
	// Stepping rather than a blocking fetch, so a Ctrl-C gets a look between bounded pieces of engine work:
	// the first array of a heavy pipeline may be minutes away, with this thread doing its share of the work.
	return Guarded(*StateOf(*self), gone, [&](ArrowStream::State &state, Teardown &teardown) {
		auto last_signals = std::chrono::steady_clock::now();
		for (;;) {
			bool idle = false;
			const auto status = TimedStep(state, *out, idle);
			switch (status) {
			case cxx::StepStatus::CHUNK: {
				if (const int signalled = HandlePendingSignals(state, out, teardown); signalled != 0) {
					return signalled;
				}
				return HandleCloseRequest(state, out, teardown);
			}
			case cxx::StepStatus::FINISHED:
				// Marked before the schema capture: the engine freed the connection's slot before reporting,
				// so from here on a concurrent close must not cancel whatever the connection runs next.
				state.exhausted.store(true);
				if (state.schema_cache.release == nullptr) {
					nb::gil_scoped_release no_gil;
					try {
						state.result->GetSchema(state.schema_cache);
					} catch (...) {
						// Without metadata there is no schema to keep; get_schema then reports closed.
					}
				}
				teardown = Take(state, true);
				*out = ArrowArray {};
				return 0;
			case cxx::StepStatus::CANCELLED:
				throw cxx::InterruptException("query was cancelled");
			case cxx::StepStatus::WAITING: {
				if (const int paced = PaceWaiting(state, idle, last_signals, teardown); paced != 0) {
					return paced;
				}
				break;
			}
			}
		}
	});
}

const char *ExportGetLastError(ArrowArrayStream *self) {
	auto state = StateOf(*self);
	CallbackGuard counted(*state);
	// This thread already under the lock means a handler asking mid-callback; the lock is ours, read away.
	if (state->OwnedByThisThread()) {
		return state->message.empty() ? nullptr : state->message.c_str();
	}
	// Never blocks, and the caller's GIL state is unknown, so no GIL dance either: any error this call may
	// see was recorded before the failing callback returned, on this same thread.
	std::unique_lock<std::mutex> guard(state->lock, std::try_to_lock);
	if (!guard.owns_lock()) {
		return nullptr;
	}
	return state->message.empty() ? nullptr : state->message.c_str();
}

void ExportRelease(ArrowArrayStream *self) {
	nb::gil_scoped_acquire gil;
	auto *holder = static_cast<Holder *>(self->private_data);
	auto state = holder->state;
	const Shutdown shutdown = ShutdownQuery(*state, false);
	if (shutdown.quiet) {
		// Nothing was anywhere near the struct, so this call is the consumer's last touch of it.
		delete holder;
		self->private_data = nullptr;
		self->get_schema = nullptr;
		self->get_next = nullptr;
		self->get_last_error = nullptr;
		self->release = nullptr;
		return;
	}
	// A release while callbacks are in or around the struct breaks the Arrow contract, yet it is reachable
	// from ordinary Python (pyarrow's reader.close() in a finalizer or another thread) and must not crash: a
	// frame unwinding toward the consumer may still call get_last_error through the struct. So everything but
	// the released mark stays whole for those trailing calls, and since no event marks the last of them, the
	// holder is never freed: it goes to the module's quarantine as containment, reachable until the
	// interpreter tears the module down.
	auto module = std::move(holder->module);
	module->Quarantine(std::shared_ptr<void>(holder, [](void *parked) { delete static_cast<Holder *>(parked); }));
	self->release = nullptr;
}

} // namespace

ArrowStream::ArrowStream(nb::object database, std::shared_ptr<ModuleState> module,
                         std::weak_ptr<cxx::Connection> connection, cxx::ArrowResult result,
                         std::vector<nb::object> kept)
    : database(std::move(database)), module(std::move(module)), state(std::make_shared<State>()) {
	state->result.emplace(std::move(result));
	state->connection = std::move(connection);
	state->kept = std::move(kept);
}

ArrowStream::~ArrowStream() {
	// A capsule already handed out is the consumer's: its reader keeps reading after this object is collected,
	// and the capsule's release ends the query. Only a stream never consumed ends here.
	if (state && !state->capsule_taken) {
		CloseInternal();
	}
}

nb::object ArrowStream::Capsule(nb::handle) {
	if (state->OwnedByThisThread()) {
		// Reached from Python running inside this stream's own callback; taking the lock would deadlock.
		Raise(module->InterfaceError(), "the stream is being read; it cannot be exported from its own callbacks");
	}
	std::unique_ptr<Holder> holder;
	{
		StateLock guard(*state);
		if (state->capsule_taken) {
			Raise(module->InterfaceError(), "the stream was already consumed; run the plan again for another");
		}
		state->capsule_taken = true;
		holder = std::make_unique<Holder>(Holder {state, module});
	}
	auto stream = std::make_unique<ArrowArrayStream>();
	stream->get_schema = &ExportGetSchema;
	stream->get_next = &ExportGetNext;
	stream->get_last_error = &ExportGetLastError;
	stream->release = &ExportRelease;
	stream->private_data = holder.release();
	return WrapStream(std::move(stream));
}

std::optional<std::pair<int, std::string>> ArrowStream::Error() const {
	// This thread already under the lock means a handler asking mid-callback; the lock is ours, read away.
	if (state->OwnedByThisThread()) {
		if (state->code == 0) {
			return std::nullopt;
		}
		return std::make_pair(state->code, state->message);
	}
	// Never blocks behind a read in flight, whose next batch may be hours away; while a read runs there is no
	// error yet, since recording happens before the lock falls. Any other holder is brief, and a recorded
	// cause must not be missed over a moment's contention: the typed re-raise reads it right here.
	std::unique_lock<std::mutex> guard(state->lock, std::try_to_lock);
	for (int attempt = 0; !guard.owns_lock() && attempt < 20; attempt++) {
		if (state->reading.load()) {
			return std::nullopt;
		}
		{
			nb::gil_scoped_release no_gil;
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
		guard.try_lock();
	}
	if (!guard.owns_lock() || state->code == 0) {
		return std::nullopt;
	}
	return std::make_pair(state->code, state->message);
}

bool ArrowStream::Live() const {
	if (state->OwnedByThisThread()) {
		return state->result.has_value();
	}
	// Never blocks: whoever holds the lock is a read or close still running, so the stream counts as live and
	// a tracker simply asks again later.
	std::unique_lock<std::mutex> guard(state->lock, std::try_to_lock);
	return !guard.owns_lock() || state->result.has_value();
}

bool ArrowStream::ClosePending() const {
	if (!state->close_requested.load()) {
		return false;
	}
	return Live();
}

void ArrowStream::Close() {
	// Not CloseInternal: a Ctrl-C that breaks the wait must reach the caller, not be swallowed.
	ShutdownQuery(*state, true);
	// This handle's own hold on the Database goes too: a closed stream kept around must pin nothing. Moved
	// under the lock and dropped outside it, as Owned::Release does: concurrent closes are supported, and two
	// unsynchronised assignments to one reference are a race on a build without the GIL.
	nb::object dropped;
	{
		nb::ft_lock_guard guard(handle_lock);
		dropped = std::move(database);
	}
}

int ArrowStream::Traverse(visitproc visit, void *arg) const {
	Py_VISIT(database.ptr());
	// The state's pins are this object's to report only while the capsule is still here: once handed out the
	// consumer owns them, and claiming them would let the collector free callables the capsule still needs.
	if (state && !state->capsule_taken) {
		for (const auto &kept : state->kept) {
			Py_VISIT(kept.ptr());
		}
	}
	return 0;
}

void ArrowStream::GcClear() {
	if (state && !state->capsule_taken) {
		CloseInternal();
	}
	nb::object dropped;
	{
		nb::ft_lock_guard guard(handle_lock);
		dropped = std::move(database);
	}
}

void ArrowStream::CloseInternal() noexcept {
	try {
		ShutdownQuery(*state, true);
	} catch (nb::python_error &error) {
		// Only a Ctrl-C escapes the wait, and nothing can raise from here (a destructor or the collector's
		// clear); re-arm it so Python's next check still sees it rather than eating the user's interrupt.
		if (error.matches(PyExc_KeyboardInterrupt)) {
			PyErr_SetInterrupt();
		}
	} catch (...) {
	}
}

} // namespace duckdb_python
