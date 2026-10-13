//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/lifetime.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "duckdb_cpp.hpp"
#include "duckdb_v2.h"
#include "conversion/conversion.hpp"

namespace duckdb::cxx::detail {
// The C++ API specializes this in its source file, not its header; the same specialization here reaches the
// environment's C handle, which `CreateInstance` below needs.
template <>
struct HandleTraits<Environment> {
	using handle = duckdb_v2_environment_handle;
};
} // namespace duckdb::cxx::detail

namespace duckdb_python {

namespace cxx = duckdb::cxx;

[[noreturn]] inline void Raise(nb::handle cls, const char *message) {
	PyErr_SetString(cls.ptr(), message);
	throw nb::python_error();
}

/// A database instance in `environment` with every setting in `options` chosen before it starts, each a name and a
/// value in the text SQL `SET` takes. The C++ API's `StartupOptions` can choose only five of the settings fixed at
/// startup, access_mode not among them, so this calls the C function it wraps, which takes any setting.
inline cxx::Instance CreateInstance(cxx::Environment &environment,
                                    const std::vector<std::pair<std::string, std::string>> &options) {
	std::vector<duckdb_v2_str> names;
	std::vector<duckdb_v2_str> values;
	for (const auto &[name, value] : options) {
		names.push_back({name.data(), name.size()});
		values.push_back({value.data(), value.size()});
	}
	duckdb_v2_instance_handle instance = nullptr;
	duckdb_v2_error_info_handle error = nullptr;
	const auto code = duckdb_v2_instance_create_with_options(environment.handle(), names.data(), values.data(),
	                                                         names.size(), &instance, &error);
	if (code != DUCKDB_V2_ERROR_NONE) {
		duckdb_v2_str text {nullptr, 0};
		duckdb_v2_str body {nullptr, 0};
		if (error != nullptr) {
			duckdb_v2_error_info_get_text(error, &text);
			duckdb_v2_error_info_get_raw_text(error, &body);
		}
		std::string message = text.ptr ? std::string(text.ptr, text.len) : "unknown error";
		std::string raw = body.ptr ? std::string(body.ptr, body.len) : "";
		duckdb_v2_error_info_destroy(&error);
		// The exception the C++ API's own calls raise for the same code.
		if (code == DUCKDB_V2_ERROR_INPUT_INVALID) {
			throw cxx::InvalidInputException(message, std::move(raw));
		}
		if (code == DUCKDB_V2_ERROR_RUNTIME_INTERRUPT) {
			throw cxx::InterruptException(message, std::move(raw));
		}
		throw cxx::Exception(code, message, std::move(raw));
	}
	return cxx::detail::HandleFactory::Make<cxx::Instance>(static_cast<void *>(instance));
}

/// Everything the extension owns for as long as its interpreter lives.
///
/// The single shared Environment is what notices a second attempt to open a database already open in this
/// process, and it is created per import rather than in a C++ static so every interpreter gets its own.
class ModuleState {
public:
	cxx::Environment environment;
	ConversionContext conversion;

	/// Quarantine for an Arrow capsule holder released while frames still used its stream: no event marks the
	/// consumer's last trailing call, so the holder is never freed while the process can still make one. It
	/// stays reachable here until the interpreter tears the module down, which is deliberate, not a leak.
	void Quarantine(std::shared_ptr<void> holder) {
		nb::ft_lock_guard guard(orphan_lock);
		orphans.push_back(std::move(holder));
	}

	/// A connection to a private one-thread database, opened on first use, that values written into SQL text are
	/// converted on by the same rules as query parameters. Its session is in UTC, so a zoned value's text names its
	/// offset the same way whatever zone the process runs in.
	///
	/// Each conversion holds one of its own until its lease ends, however that ends, and no lock is held meanwhile:
	/// converting calls Python code, which may convert another value on this thread or wait for another thread to.
	class LiteralLease {
	public:
		explicit LiteralLease(ModuleState &state) : state(state) {
			nb::ft_lock_guard guard(state.literal_lock);
			if (!state.literal_instance) {
				// Returning a connection then never allocates, so it cannot throw from a destructor.
				state.literal_idle.reserve(LITERAL_IDLE_KEPT);
				auto instance = CreateInstance(state.environment, {{"threads", "1"}});
				instance.Attach(":memory:", true);
				state.literal_instance.emplace(std::move(instance));
			}
			if (state.literal_idle.empty()) {
				connection.emplace(state.literal_instance->Connect());
				connection->GetConfig().SetOption("TimeZone", "UTC");
			} else {
				connection.emplace(std::move(state.literal_idle.back()));
				state.literal_idle.pop_back();
			}
		}

		~LiteralLease() {
			nb::ft_lock_guard guard(state.literal_lock);
			if (state.literal_idle.size() < LITERAL_IDLE_KEPT) {
				state.literal_idle.push_back(std::move(*connection));
			}
		}

		LiteralLease(const LiteralLease &) = delete;
		LiteralLease &operator=(const LiteralLease &) = delete;

		cxx::Connection &operator*() {
			return *connection;
		}

	private:
		ModuleState &state;
		std::optional<cxx::Connection> connection;
	};

	nb::handle InterfaceError() {
		return Exceptions().interface_error;
	}

	nb::handle InterruptError() {
		return Exceptions().interrupt_error;
	}

	/// The exception class duckdb.exceptions maps a DuckDB error code to.
	nb::object ClassForCode(int code) {
		return Exceptions().class_for_code(code);
	}

private:
	struct ExceptionClasses {
		nb::object class_for_code;
		nb::object interface_error;
		nb::object interrupt_error;
	};

	/// Looked up on first use, not at import, so the extension can be loaded without the duckdb package.
	const ExceptionClasses &Exceptions() {
		nb::ft_lock_guard guard(exceptions_lock);
		if (!exceptions) {
			nb::object module = nb::module_::import_("duckdb.exceptions");
			exceptions = ExceptionClasses {module.attr("class_for_code"), module.attr("InterfaceError"),
			                               module.attr("InterruptError")};
		}
		return *exceptions;
	}

	nb::ft_mutex orphan_lock;
	std::vector<std::shared_ptr<void>> orphans;
	nb::ft_mutex exceptions_lock;
	std::optional<ExceptionClasses> exceptions;
	// As many idle connections as nested or parallel conversions commonly need; more are opened when wanted.
	static constexpr std::size_t LITERAL_IDLE_KEPT = 4;
	nb::ft_mutex literal_lock;
	// Declared after the environment, so they are closed before it, the connections before their database.
	std::optional<cxx::Instance> literal_instance;
	std::vector<cxx::Connection> literal_idle;
};

/// Marks this thread as one a stream close must never wait on: it is doing work the close could be waiting
/// for, user Python reached from an engine thread or a stream callback, or a stream teardown running
/// finalizers. Python reached from such work can close streams; the close then leaves its request standing,
/// completed by the read it interrupted, instead of deadlocking.
class BusyFence {
public:
	enum class Kind {
		/// Work a close may be waiting for; a close here only skips its wait.
		kWork,
		/// An engine task, a scalar function or a scan callback mid-call. A close here must also never
		/// destroy a result: destroying one waits for the engine's in-flight tasks, and this thread is one.
		kEngineTask,
	};

	explicit BusyFence(Kind kind = Kind::kWork) : kind(kind) {
		depth++;
		if (kind == Kind::kEngineTask) {
			engine_depth++;
		}
	}
	~BusyFence() {
		depth--;
		if (kind == Kind::kEngineTask) {
			engine_depth--;
		}
	}
	BusyFence(const BusyFence &) = delete;
	BusyFence &operator=(const BusyFence &) = delete;

	static bool Here() {
		return depth > 0;
	}

	static bool EngineTask() {
		return engine_depth > 0;
	}

private:
	Kind kind;
	inline static thread_local int depth = 0;
	inline static thread_local int engine_depth = 0;
};

/// The GIL plus the engine-task mark, as one word: what every engine-thread entry into user Python takes.
struct FencedGil {
	nb::gil_scoped_acquire gil;
	BusyFence fence {BusyFence::Kind::kEngineTask};
};

/// References a running call holds itself, so a close on another thread cannot free what it is still using.
template <class T>
struct Pinned {
	nb::object database;
	std::shared_ptr<T> engine;
};

/// A DuckDB object and the Database it belongs to, held so a close cannot race a call on another thread.
///
/// Acquire hands a call its own references and Release drops the owner's. The global interpreter lock already
/// serialises those two and the mutex costs nothing there; on a build without that lock, the mutex is what does.
template <class T>
class Owned {
public:
	Owned(std::shared_ptr<ModuleState> module, nb::object database, T value)
	    : module(std::move(module)), database(std::move(database)), held(std::make_shared<T>(std::move(value))) {
	}

	const std::shared_ptr<ModuleState> &Module() const {
		return module;
	}

	/// The Database reference, for the garbage collector's visit, which runs with every other thread stopped.
	nb::handle Parent() const {
		return database;
	}

	/// Own references to both, or InterfaceError with `closed_message` once Release has run.
	Pinned<T> Acquire(const char *closed_message) {
		Pinned<T> live;
		{
			nb::ft_lock_guard guard(lifetime);
			live.database = database;
			live.engine = held;
		}
		if (!live.engine) {
			Raise(module->InterfaceError(), closed_message);
		}
		return live;
	}

	/// Drop this owner's references, repeatably; the last one calls into DuckDB, so the GIL is dropped first.
	void Release() {
		std::shared_ptr<T> released;
		nb::object parent;
		{
			nb::ft_lock_guard guard(lifetime);
			released = std::move(held);
			parent = std::move(database);
		}
		{
			nb::gil_scoped_release unlock;
			released.reset();
		}
		parent.reset();
	}

private:
	std::shared_ptr<ModuleState> module;
	nb::ft_mutex lifetime;
	nb::object database;
	std::shared_ptr<T> held;
};

/// How a connection is in use. It runs one query at a time: the query's result holds the connection from the
/// statement's submission until it is read to its end, fails, or is destroyed, also where the engine built the result
/// in full and no longer needs the connection, and the connection is running while a call on it, the reading of its
/// result, or a task of a held result's query is in the engine. Any other use of a held or running connection is
/// refused at once: the engine would wait for a lock the running query holds, forever when the use comes from a
/// Python function inside that query.
///
/// The mutex guards the fields below and is never held across an engine call that can wait or a GIL acquisition.
class ConnectionUse {
public:
	/// What a call needs of the connection: free, or only not running, for parsing and binding, which the engine
	/// allows beside an open result but not while a query holds the client context's lock.
	enum class Needs : uint8_t { FREE, NOT_RUNNING };

	class Reading;

	/// A call's claim for as long as it lives; raises OperationalError when the connection is in use.
	class Call {
	public:
		Call(ConnectionUse &use, Needs needs) : use(use) {
			std::lock_guard<std::mutex> guard(use.lock);
			// The engine runs a held result's query on its workers with no call or read of ours in progress, so an
			// engine task reaching a held connection is taken to come from inside that query.
			if (use.running > 0 || (use.held && BusyFence::EngineTask())) {
				Refuse(kRunning);
			}
			if (needs == Needs::FREE && use.held) {
				Refuse(kHeld);
			}
			use.running += 1;
		}
		~Call() {
			std::lock_guard<std::mutex> guard(use.lock);
			use.running -= 1;
		}
		Call(const Call &) = delete;
		Call &operator=(const Call &) = delete;

	private:
		ConnectionUse &use;
	};

	/// One particular hold, named after it is taken, so another thread can act while that hold, and no later one,
	/// still holds the connection.
	class HoldTag {
	public:
		HoldTag() = default;
		HoldTag(std::shared_ptr<ConnectionUse> use, uint64_t id) : use(std::move(use)), id(id) {
		}

		/// Runs `action` under the lock while the hold still holds the connection, so no new call can start meanwhile;
		/// `action` must neither wait nor take the GIL.
		template <class ACTION>
		void WhileHeld(ACTION &&action) const {
			if (!use) {
				return;
			}
			std::lock_guard<std::mutex> guard(use->lock);
			if (use->held && use->holder == id) {
				action();
			}
		}

	private:
		std::shared_ptr<ConnectionUse> use;
		uint64_t id = 0;
	};

	/// A result's hold on the connection, taken while the call that made the result still has its claim, and
	/// given up by `Release` or with the hold.
	class Hold {
	public:
		Hold() = default;
		explicit Hold(std::shared_ptr<ConnectionUse> taken) : use(std::move(taken)) {
			std::lock_guard<std::mutex> guard(use->lock);
			use->held = true;
			id = ++use->holds;
			use->holder = id;
		}
		Hold(Hold &&other) noexcept : use(std::move(other.use)), id(other.id) {
		}
		Hold &operator=(Hold &&other) noexcept {
			if (this != &other) {
				Release();
				use = std::move(other.use);
				id = other.id;
			}
			return *this;
		}
		~Hold() {
			Release();
		}
		Hold(const Hold &) = delete;
		Hold &operator=(const Hold &) = delete;

		void Release() {
			if (use) {
				{
					std::lock_guard<std::mutex> guard(use->lock);
					use->held = false;
					use->holder = 0;
				}
				use.reset();
			}
		}

		/// Who this hold is, for acting later only while it still holds the connection; nothing once released.
		HoldTag Tag() const {
			return HoldTag(use, id);
		}

	private:
		friend class ConnectionUse::Reading;
		std::shared_ptr<ConnectionUse> use;
		uint64_t id = 0;
	};

	/// Marks the connection running while its holding result reads, which no claim of its own can refuse.
	class Reading {
	public:
		explicit Reading(const Hold &hold) : use(hold.use) {
			if (use) {
				std::lock_guard<std::mutex> guard(use->lock);
				use->running += 1;
			}
		}
		~Reading() {
			if (use) {
				std::lock_guard<std::mutex> guard(use->lock);
				use->running -= 1;
			}
		}
		Reading(const Reading &) = delete;
		Reading &operator=(const Reading &) = delete;

	private:
		std::shared_ptr<ConnectionUse> use;
	};

private:
	static constexpr const char *kHeld =
	    "connection has a live result; finish reading it or close it before using the connection again";
	static constexpr const char *kRunning =
	    "connection has a live result; it cannot be used while its query runs, from a function inside that query or "
	    "from another thread: use another connection";

	[[noreturn]] static void Refuse(const char *body) {
		throw cxx::Exception(DUCKDB_V2_ERROR_RESOURCE_IN_USE, std::string("Resource In Use Error: ") + body, body);
	}

	std::mutex lock;
	int running = 0;
	bool held = false;
	/// The hold that has the connection, 0 for none, and how many holds there have been, so each has its own number.
	uint64_t holder = 0;
	uint64_t holds = 0;
};

/// Run a DuckDB call with the GIL released and hand back what it returned.
template <class CALL>
auto WithoutGil(CALL &&call) {
	nb::gil_scoped_release release;
	return call();
}

} // namespace duckdb_python
