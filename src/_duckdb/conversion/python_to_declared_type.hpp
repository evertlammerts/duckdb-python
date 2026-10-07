//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/python_to_declared_type.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "conversion.hpp"
#include "duckdb_cpp.hpp"

namespace duckdb_python {

/// A declared type with everything per-row conversion asks of it read upfront, since the C API serializes the
/// type on each child read. Built once at registration and only read after, so engine threads share it.
struct TypeTree {
	duckdb::cxx::LogicalType type;
	duckdb::cxx::LogicalTypeId id;
	/// Holds a date, time, struct or union anywhere, so a value converts part by part rather than whole.
	bool converts_per_element;
	std::vector<std::string> names;
	std::vector<TypeTree> parts;
};

TypeTree TypeTreeOf(duckdb::cxx::LogicalType type);

/// One Python object as a value of the declared type `target`, each part converted against the part of `target`
/// the engine's cast pairs it with, before anything is combined: a date or timestamp anywhere in it is held
/// exactly and never meets a time zone it does not name, or a `DeclaredTypeRefusal` is raised; every other part
/// is cast as CAST would.
duckdb::cxx::Value PythonToDeclaredType(duckdb::cxx::Context &scope, nb::handle object, const TypeTree &target,
                                        ConversionContext &ctx);

} // namespace duckdb_python
