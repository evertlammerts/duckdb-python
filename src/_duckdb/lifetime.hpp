//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/lifetime.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <utility>

#include "duckdb_cpp.hpp"
#include "pyconv.hpp"

namespace duckdb_python {

namespace cxx = duckdb::cxx;

[[noreturn]] inline void Raise(nb::handle cls, const char *message) {
	PyErr_SetString(cls.ptr(), message);
	throw nb::python_error();
}

/// Everything this module owns for the lifetime of its interpreter.
///
/// There is exactly one Environment, and every database is opened through it.
/// That is the whole point of the Environment: it is what notices a second
/// attempt to open a database already open in this process. Giving each
/// Database its own Environment would silently remove that guard.
///
/// This is per module instance, not a C++ static. nanobind initialises modules
/// in two phases and the exec function runs once per interpreter that imports
/// us, so a state created there and captured by the bindings registered
/// alongside it belongs to that interpreter alone.
class ModuleState {
public:
	cxx::Environment environment;
	ConversionContext conversion;

	nb::handle InterfaceError() {
		return Exceptions().interface_error;
	}

	nb::handle InterruptError() {
		return Exceptions().interrupt_error;
	}

	/// The class duckdb.exceptions assigns an engine error code.
	nb::object ClassForCode(int code) {
		return Exceptions().class_for_code(code);
	}

private:
	struct ExceptionClasses {
		nb::object class_for_code;
		nb::object interface_error;
		nb::object interrupt_error;
	};

	/// Resolved on first use rather than at module init, so the extension can
	/// be loaded on its own before the package is, as the sanitizer staging
	/// does.
	const ExceptionClasses &Exceptions() {
		nb::ft_lock_guard guard(exceptions_lock);
		if (!exceptions) {
			nb::object module = nb::module_::import_("duckdb.exceptions");
			exceptions = ExceptionClasses {module.attr("class_for_code"), module.attr("InterfaceError"),
			                               module.attr("InterruptError")};
		}
		return *exceptions;
	}

	nb::ft_mutex exceptions_lock;
	std::optional<ExceptionClasses> exceptions;
};

/// What a running call pins: the engine object it uses and the Database it
/// runs on. Both are owning copies, so a close on another thread releases
/// only its own references and this call's stay valid until it returns.
template <class T>
struct Pinned {
	nb::object database;
	std::shared_ptr<T> engine;
};

/// What a wrapper owns: its engine object and the Database that object runs
/// on, guarded against a close racing a call.
///
/// A call takes its own owning copies through Acquire and keeps both alive
/// until it returns; Release moves the owner's copies out and lets the last
/// ones go. On a GIL build both run with the GIL held, which serialises
/// them, and the mutex compiles to nothing; on the free-threaded build it is
/// a PyMutex and is what serialises them.
template <class T>
class Owned {
public:
	Owned(std::shared_ptr<ModuleState> module, nb::object database, T value)
	    : module(std::move(module)), database(std::move(database)), held(std::make_shared<T>(std::move(value))) {
	}

	const std::shared_ptr<ModuleState> &Module() const {
		return module;
	}

	/// The Database reference, for the traverse slot, which runs with every
	/// other thread stopped.
	nb::handle Parent() const {
		return database;
	}

	/// Owning copies, or InterfaceError with `closed_message` once released.
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

	/// Drop this owner's copies. Idempotent. Destroying the engine object
	/// talks to the engine, so the last copy, if this is it, goes without the
	/// GIL; the Database reference goes with it.
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

/// Run an engine call with the GIL released and hand back what it produced.
template <class CALL>
auto WithoutGil(CALL &&call) {
	nb::gil_scoped_release release;
	return call();
}

} // namespace duckdb_python
