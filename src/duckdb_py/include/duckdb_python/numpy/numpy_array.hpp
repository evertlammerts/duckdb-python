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
//! Performance note: `Data()`/`MutableData()` are COLD — every caller fetches the pointer
//! once and then loops over it (see RawArrayWrapper::data / numpy scan helpers), so reading
//! the buffer address via `arr.ctypes.data` (which works for every dtype, including the
//! `object` dtype that DLPack/`nb::ndarray` cannot represent) costs nothing in the hot path.
//! Ownership is move-only-when-asked: the ctor takes by value and moves, GetArray() hands
//! back a reference, and no method copies the array buffer.
class NumpyArray {
public:
	NumpyArray() = default;
	//! Wrap an existing numpy array object (no copy; the object is moved in).
	explicit NumpyArray(py::object arr) : array(std::move(arr)) {
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

	//! Read-only pointer to the underlying data buffer (cold; see class note).
	const void *Data() const {
		return BufferPointer();
	}

	//! Mutable pointer to the underlying data buffer (cold; see class note).
	void *MutableData() {
		return BufferPointer();
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
	//! Buffer start address of the underlying numpy array. `ctypes.data` is dtype-agnostic
	//! (works for the `object` dtype too) and only ever called on the cold path.
	void *BufferPointer() const {
		return reinterpret_cast<void *>(py::cast<uintptr_t>(array.attr("ctypes").attr("data")));
	}

	//! The single data member -- the owned numpy array (formerly `py::array`).
	py::object array;
};

} // namespace duckdb
