//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pybind11//pybind_wrapper.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/detail/nb_list.h>
#include <nanobind/operators.h>
#include <cassert>

// Custom type_caster specializations must be visible in every TU that converts the type (otherwise it is
// UB); keep ALL of them here, in this universally-included umbrella, never in scattered per-feature headers.
#include "duckdb_python/pybind11/conversions/identifier.hpp"
#include "duckdb_python/pybind11/conversions/python_udf_type_enum.hpp"
#include "duckdb_python/pybind11/conversions/null_handling_enum.hpp"
#include "duckdb_python/pybind11/conversions/exception_handling_enum.hpp"
#include "duckdb_python/pybind11/conversions/explain_enum.hpp"
#include "duckdb_python/pybind11/conversions/render_mode_enum.hpp"
#include "duckdb_python/pybind11/conversions/python_csv_line_terminator_enum.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/helper.hpp"
#include <memory>
#include <type_traits>

// nanobind has no holder-type declaration macros; std::shared_ptr / std::unique_ptr support is
// provided by the <nanobind/stl/shared_ptr.h> / <nanobind/stl/unique_ptr.h> includes above.

// Python interop helpers (raw CPython accessors, guarded isinstance, string coercion, tuple builder, GIL/collection).
#include "duckdb_python/pyutil.hpp"

namespace nanobind {

namespace detail {

// duckdb::vector behaves like a Python list on the boundary; reuse nanobind's list_caster.
template <typename Type, bool SAFE>
struct type_caster<duckdb::vector<Type, SAFE>> : list_caster<duckdb::vector<Type, SAFE>, Type> {};
} // namespace detail
} // namespace nanobind

namespace duckdb {
namespace py {

// We include everything from nanobind
using namespace nanobind;

// But we have the option to override certain functions
template <typename T, std::enable_if_t<std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) {
	return T::check_(obj);
}

template <typename T, std::enable_if_t<!std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) {
	return nanobind::isinstance<T>(obj);
}

template <class T>
bool try_cast(const handle &object, T &result) {
	try {
		result = cast<T>(object);
	} catch (cast_error &) {
		return false;
	}
	return true;
}

} // namespace py

template <class T, typename... ARGS>
void DefineMethod(std::vector<const char *> aliases, T &mod, ARGS &&...args) {
	for (auto &alias : aliases) {
		mod.def(alias, args...);
	}
}

} // namespace duckdb
