//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/numpy/numpy_array.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"

namespace duckdb {

//! Thin façade over the numpy array representation.
//!
//! This class is the SINGLE place in the codebase that owns the underlying numpy-array
//! object. Under nanobind there is no `py::array` (and no `py::dtype`); the array is held
//! as a plain `nb::object` and the few buffer operations go through numpy directly.
//!
//! Performance note: `Data()`/`MutableData()` are on the HOT path — the numpy scan calls
//! `Data()` once per column per 2048-row chunk (see numpy_scan.cpp), and DuckDB drives that
//! scan from multiple threads WITHOUT holding the GIL. Fetching the buffer address via
//! `arr.ctypes.data` is ~1-5µs, allocates a numpy `_ctypes` object, and *requires the GIL*,
//! so doing it per chunk would be both a scaling regression and a correctness hazard under a
//! parallel scan. We therefore compute the pointer ONCE, eagerly, in the constructor (always
//! invoked single-threaded with the GIL held at bind/result time) and cache it; `Data()` then
//! becomes a plain pointer read with no Python call and no GIL — matching pybind11's
//! `py::array.data()`. The cache is invalidated (and recomputed) by `Resize()`, the only
//! operation that reallocates the buffer. `ctypes.data` is also dtype-agnostic (works for the
//! `object` dtype that DLPack/`nb::ndarray` cannot represent).
//!
//! Ownership is move-only-when-asked: the ctor takes by value and moves, GetArray() hands
//! back a reference, and no method copies the array buffer. The raw `cached_data_` member uses
//! default copy/move: a copy shares the same underlying numpy buffer (so the pointer stays
//! valid), and a move transfers array + pointer together.
class NumpyArray {
public:
	NumpyArray() = default;
	//! Wrap an existing numpy array object (no copy; the object is moved in). The buffer pointer is
	//! computed eagerly here (GIL held) so the hot scan path never makes a Python call.
	explicit NumpyArray(py::object arr) : array(std::move(arr)) {
		EnsurePointer();
	}

	NumpyArray(NumpyArray &&) = default;
	NumpyArray &operator=(NumpyArray &&) = default;
	NumpyArray(const NumpyArray &) = default;
	NumpyArray &operator=(const NumpyArray &) = default;

public:
	//! Allocate a fresh, contiguous 1-D numpy array of `count` elements with the given numpy
	//! dtype string (e.g. "int64", "float32", "object", "datetime64[us]"). Uninitialized —
	//! callers fill it immediately, matching the previous `py::array(py::dtype(d), count)`.
	static NumpyArray Allocate(const string &dtype, idx_t count) {
		auto numpy = py::module_::import_("numpy");
		return NumpyArray(numpy.attr("empty")(count, dtype));
	}

	//! Produce a numpy array from an arbitrary Python object (np.asarray semantics: no copy
	//! when `obj` already is an ndarray). The object is moved into the call.
	static NumpyArray FromObject(py::object obj) {
		auto numpy = py::module_::import_("numpy");
		return NumpyArray(numpy.attr("asarray")(std::move(obj)));
	}

	//! Read-only pointer to the underlying data buffer (hot path: plain cached read, no GIL).
	const void *Data() const {
		return cached_data_;
	}

	//! Mutable pointer to the underlying data buffer (hot path: plain cached read, no GIL).
	void *MutableData() {
		return cached_data_;
	}

	//! Resize the underlying numpy buffer in place. This REALLOCATES the buffer, so the cached
	//! pointer is invalidated and recomputed (GIL is held -- this only runs on the single-threaded
	//! result-materialization path).
	void Resize(idx_t count) {
		array.attr("resize")(count, py::arg("refcheck") = false);
		cached_data_ = nullptr;
		EnsurePointer();
	}

	//! Access the underlying array, e.g. for `.attr(...)` calls, iteration, or to hand it
	//! back to Python. Returned by reference -- never copied.
	py::object &GetArray() {
		return array;
	}
	const py::object &GetArray() const {
		return array;
	}

private:
	//! Compute and cache the buffer start address of the underlying numpy array, if not already
	//! cached and an array is held. `ctypes.data` is dtype-agnostic (works for the `object` dtype
	//! too). Only ever called with the GIL held (construction / Resize).
	void EnsurePointer() {
		// Only numpy ndarrays expose `ctypes`; some NumpyArray wrappers hold other objects (e.g. a pandas Index)
		// whose buffer pointer is never read. Guard the eager compute so constructing such a wrapper doesn't raise
		// (the original lazy code only touched `ctypes` if Data()/MutableData() was actually called).
		if (!cached_data_ && array.ptr() != nullptr && py::hasattr(array, "ctypes")) {
			cached_data_ = reinterpret_cast<void *>(py::cast<uintptr_t>(array.attr("ctypes").attr("data")));
		}
	}

	//! The owned numpy array (formerly `py::array`).
	py::object array;
	//! Cached buffer start address; see the class-level performance note.
	void *cached_data_ = nullptr;
};

} // namespace duckdb
