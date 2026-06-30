#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "duckdb/common/types.hpp"
#include "duckdb/common/helper.hpp"
#include <cassert>
#include <string>

namespace nb = nanobind;

namespace duckdb {

// Python interop helpers: raw CPython accessors plus duckdb extensions over nanobind (guarded isinstance,
// lenient string coercion, immutable-tuple builder, GIL and collection predicates). Self-contained on
// nanobind so the umbrella can include it; do not pull the umbrella back in here.
struct PyUtil {
	static idx_t PyByteArrayGetSize(nb::handle &obj) {
		return PyByteArray_GET_SIZE(obj.ptr()); // NOLINT
	}

	static Py_buffer *PyMemoryViewGetBuffer(nb::handle &obj) {
		return PyMemoryView_GET_BUFFER(obj.ptr());
	}

	static bool PyUnicodeIsCompactASCII(nb::handle &obj) {
		return PyUnicode_IS_COMPACT_ASCII(obj.ptr());
	}

	static const char *PyUnicodeData(nb::handle &obj) {
		return const_char_ptr_cast(PyUnicode_DATA(obj.ptr()));
	}

	static char *PyUnicodeDataMutable(nb::handle &obj) {
		return char_ptr_cast(PyUnicode_DATA(obj.ptr()));
	}

	static idx_t PyUnicodeGetLength(nb::handle &obj) {
		return PyUnicode_GET_LENGTH(obj.ptr());
	}

	static bool PyUnicodeIsCompact(PyCompactUnicodeObject *obj) {
		return PyUnicode_IS_COMPACT(obj);
	}

	static bool PyUnicodeIsASCII(PyCompactUnicodeObject *obj) {
		return PyUnicode_IS_ASCII(obj);
	}

	static int PyUnicodeKind(nb::handle &obj) {
		return PyUnicode_KIND(obj.ptr());
	}

	static Py_UCS1 *PyUnicode1ByteData(nb::handle &obj) {
		return PyUnicode_1BYTE_DATA(obj.ptr());
	}

	static Py_UCS2 *PyUnicode2ByteData(nb::handle &obj) {
		return PyUnicode_2BYTE_DATA(obj.ptr());
	}

	static Py_UCS4 *PyUnicode4ByteData(nb::handle &obj) {
		return PyUnicode_4BYTE_DATA(obj.ptr());
	}

	// isinstance(obj, type) with a null-type guard: an un-imported optional module yields a null type handle,
	// for which we return false. nanobind's isinstance(obj, type) would raise instead.
	static bool IsInstance(nb::handle obj, nb::handle type) {
		if (type.ptr() == nullptr) {
			return false;
		}
		const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
		if (result == -1) {
			throw nb::python_error();
		}
		return result != 0;
	}

	// Lenient string conversion: str as is, bytes UTF-8 decoded, anything else via str().
	// nanobind's cast<std::string> rejects bytes/scalars. For identifier/param-key/separator sites.
	static std::string CastToString(nb::handle obj) {
		if (nb::bytes::check_(obj)) {
			return nb::cast<std::string>(obj.attr("decode")("utf-8"));
		}
		if (nb::str::check_(obj)) {
			return nb::cast<std::string>(obj);
		}
		return nb::cast<std::string>(nb::str(obj));
	}

	// GIL state checks.
	static bool GilCheck();
	static void GilAssert();

	// Collection predicates consulting the connection's ImportCache (collections.abc Iterable/Mapping).
	static bool IsListLike(nb::handle obj);
	static bool IsDictLike(nb::handle obj);

	// Fills a fixed-size immutable nb::tuple via PyTuple_SET_ITEM (cheaper than a list then a copy).
	// Fill every slot with append()/set_item(), then take().
	class TupleBuilder {
	public:
		explicit TupleBuilder(size_t size)
		    : tuple_(nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)))), size_(size) {
		}
		void append(nb::object item) {
			assert(index_ < size_);
			PyTuple_SET_ITEM(tuple_.ptr(), static_cast<Py_ssize_t>(index_++), item.release().ptr());
		}
		void set_item(size_t index, nb::object item) {
			assert(index < size_);
			PyTuple_SET_ITEM(tuple_.ptr(), static_cast<Py_ssize_t>(index), item.release().ptr());
		}
		size_t size() const {
			return size_;
		}
		nb::tuple take() {
			return std::move(tuple_);
		}

	private:
		nb::tuple tuple_;
		size_t size_;
		size_t index_ = 0;
	};
};

} // namespace duckdb
