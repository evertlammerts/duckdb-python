//===----------------------------------------------------------------------===//
//                         DuckDB
//
// module.cpp
//
// Entry point of the duckdb Python extension module (_duckdb).
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "duckdb_cpp.hpp"

namespace nb = nanobind;

NB_MODULE(_duckdb, m) {
	m.doc() = "DuckDB Python extension module.";

	m.def("library_version", []() { return duckdb::cxx::LibraryVersion(); },
	      "The version of the DuckDB engine this extension is linked against.");
}
