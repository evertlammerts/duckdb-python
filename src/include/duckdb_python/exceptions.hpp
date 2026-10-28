#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb_python/nb/casters.hpp"

namespace nb = nanobind;

namespace duckdb {

void RegisterExceptions(const nb::module_ &m);

//! Python text as UTF-8, a lone surrogate escaped the way Python prints it in a traceback; call with the GIL held
string ToUTF8(const nb::object &text);
//! "TypeName: message" for a caught Python error, without its traceback; call with the GIL held
string FormatUDFPythonError(nb::python_error &error);
//! The engine error for a Python error raised inside a query, carrying its traceback. Make it where the error was
//! raised, with the GIL held and inside any callback scope: the engine formats and frees an error it catches later, on
//! a thread that may no longer enter Python.
Exception AsDuckDBError(const nb::python_error &error);

} // namespace duckdb
