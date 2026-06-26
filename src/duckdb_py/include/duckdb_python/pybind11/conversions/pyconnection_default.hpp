#pragma once

#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/helper.hpp"

// NANOBIND PORTING NOTE (default-connection / None handling):
//
// pybind11 mapped a Python None (or an omitted `connection=None` argument) to the module-level default
// connection via a custom `copyable_holder_caster` specialization here. nanobind has no
// `copyable_holder_caster`, and -- more importantly -- the cutover already moved the None->DefaultConnection()
// decision OUT of the caster and INTO every binding lambda (each `connection`-taking function now does
// `if (!conn) { conn = DuckDBPyConnection::DefaultConnection(); }`). See duckdb_python.cpp and
// typing/pytype.cpp::FromString.
//
// Because of that refactor we rely on nanobind's built-in `std::shared_ptr<T>` type caster
// (from <nanobind/stl/shared_ptr.h>, pulled in by the umbrella) instead of a custom one:
//   * a passed Python connection -> the corresponding shared_ptr<DuckDBPyConnection>, and
//   * None -> a null shared_ptr, which the lambda's null-check turns into DefaultConnection().
//
// nanobind rejects None for bound-type arguments unless the argument is annotated `.none()`, so every
// `connection` argument is declared `py::arg("connection").none() = py::none()` (see NANOBIND_NONE_AUDIT.md).
// No custom caster is required; this header intentionally only forwards the connection type so existing
// includes keep resolving.
