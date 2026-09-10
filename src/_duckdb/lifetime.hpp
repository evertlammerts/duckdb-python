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

/// Everything the extension owns for as long as its interpreter lives.
///
/// The single shared Environment is what notices a second attempt to open a database already open in this
/// process, and it is created per import rather than in a C++ static so every interpreter gets its own.
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

	nb::ft_mutex exceptions_lock;
	std::optional<ExceptionClasses> exceptions;
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

/// Run a DuckDB call with the GIL released and hand back what it returned.
template <class CALL>
auto WithoutGil(CALL &&call) {
	nb::gil_scoped_release release;
	return call();
}

} // namespace duckdb_python
