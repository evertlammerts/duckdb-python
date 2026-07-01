//===----------------------------------------------------------------------===//
//                         DuckDB
//
// numpy_array.cpp
//
// Out-of-line definitions for the NumpyArray facade (numpy_array.hpp). This is the
// ONLY translation unit that uses the numpy C API, so it does not need
// PY_ARRAY_UNIQUE_SYMBOL / NO_IMPORT_ARRAY (those coordinate the C-API function
// pointer table across multiple TUs).
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "duckdb_python/numpy/numpy_array.hpp"

#include <stdexcept>
#include <unordered_map>

namespace duckdb {
namespace numpy_internal {

namespace {

//! Lazy, guarded one-time init of the numpy C-API function pointer table. numpy is always
//! already imported by the time we allocate a result array, so import_array should succeed;
//! if it does not, the returned value is false and the caller raises. Runs exactly once
//! (function-local static initializer, GIL held on the result path).
bool EnsureNumpyCApi() {
	static bool ok = []() -> bool {
		// import_array1(ret) expands to `return ret;` on failure, so wrap it in a lambda that
		// returns int and surface success via the return value.
		auto do_import = []() -> int {
			import_array1(-1);
			return 0;
		};
		return do_import() == 0;
	}();
	return ok;
}

} // namespace

nb::object NumpyEmpty(idx_t count, const string &dtype) {
	// Process-lifetime cache of parsed np.dtype objects, keyed by dtype string. The parse is
	// otherwise repeated per call; a LIST/ARRAY column allocates one array per row. Leaked on
	// purpose (numpy is never unloaded; no Python destructor runs after finalization). Only ever
	// touched on the single-threaded, GIL-held result path.
	static auto &dtype_cache = *new std::unordered_map<string, PyObject *>();
	PyObject *&descr = dtype_cache[dtype];
	if (!descr) {
		nb::object d = nb::module_::import_("numpy").attr("dtype")(dtype);
		descr = d.release().ptr();
	}

	if (!EnsureNumpyCApi()) {
		throw std::runtime_error("Failed to initialize the numpy C API (import_array failed)");
	}

	npy_intp dims[1] = {static_cast<npy_intp>(count)};
	// PyArray_Empty STEALS a reference to descr. descr is a single cached np.dtype reused across
	// every allocation, so hand PyArray_Empty its own reference to consume.
	Py_INCREF(descr);
	PyObject *arr = PyArray_Empty(1, dims, reinterpret_cast<PyArray_Descr *>(descr), 0 /* C order */);
	if (!arr) {
		// PyArray_Empty consumed the stolen reference even on failure; balance the INCREF above so
		// the cached descr is not leaked, then surface the numpy error.
		Py_DECREF(descr);
		throw nb::python_error();
	}
	// PyArray_Empty returns a NEW reference; hand ownership to nanobind via steal.
	return nb::steal<nb::object>(arr);
}

} // namespace numpy_internal
} // namespace duckdb
