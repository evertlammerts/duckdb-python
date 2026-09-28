//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrowc.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "lifetime.hpp"

// The Arrow C data and stream interface structs, under their standard guards.
#include "duckdb_v2.h"

namespace duckdb_python {

/// The capsule names the Arrow PyCapsule interface reserves.
inline constexpr const char *kStreamCapsule = "arrow_array_stream";
inline constexpr const char *kSchemaCapsule = "arrow_schema";
inline constexpr const char *kArrayCapsule = "arrow_array";

std::string StreamError(ArrowArrayStream &stream);

/// Custody of one Arrow C struct (a schema, an array or a stream), released when the owner goes unless it was
/// handed on first. The Arrow C interfaces let a struct move by copying it and clearing the source's `release`,
/// which is what moving the owner does; nothing may point into the struct itself.
template <class T>
struct ArrowOwned {
	ArrowOwned() = default;
	ArrowOwned(ArrowOwned &&other) noexcept : value(other.value) {
		other.value.release = nullptr;
	}
	ArrowOwned &operator=(ArrowOwned &&other) noexcept {
		if (this != &other) {
			Release();
			value = other.value;
			other.value.release = nullptr;
		}
		return *this;
	}
	ArrowOwned(const ArrowOwned &) = delete;
	ArrowOwned &operator=(const ArrowOwned &) = delete;
	~ArrowOwned() {
		Release();
	}

	/// A release callback that throws is broken; the struct is then abandoned rather than released a second time,
	/// and the exception does not escape into the C code or the destructor that called this.
	void Release() noexcept {
		if (value.release == nullptr) {
			return;
		}
		try {
			value.release(&value);
		} catch (...) {
		}
		value.release = nullptr;
	}
	explicit operator bool() const {
		return value.release != nullptr;
	}

	T value {};
};

bool IsBatch(const ArrowSchema &schema);

/// Turns a schema that is not a batch into a batch of one column: a struct with the schema as its only child.
void WrapAsBatch(ArrowSchema &schema);
/// The array counterpart: a struct array with no validity buffer whose only child is the array.
void WrapAsBatch(ArrowArray &array);

/// Several Arrow exports read as one stream: `parts`, an iterable of objects with `__arrow_c_stream__`, are read
/// in order, each part expected to carry the schema `schema` (itself anything with `__arrow_c_stream__`) exports;
/// the caller guarantees that, nothing checks it. Pulled from one thread at a time, as the Arrow stream
/// interface requires of every stream.
nb::capsule ChainStreams(nb::object schema, nb::handle parts);

/// The name of a batch schema's child, empty when it has none.
std::string NameOf(const ArrowSchema &schema, cxx::idx_t index);

/// The struct a capsule named `kind` carries, still owned by the capsule, refused when the capsule is of another
/// kind or already released; `source` says where the capsule came from, for the error, such as "the object
/// registered as 't'".
template <class T>
T &InCapsule(nb::handle capsule, const char *kind, const std::string &source) {
	if (!PyCapsule_IsValid(capsule.ptr(), kind)) {
		const char *found = PyCapsule_CheckExact(capsule.ptr()) ? PyCapsule_GetName(capsule.ptr()) : nullptr;
		throw cxx::InvalidInputException(source + " did not export an '" + kind + "' capsule but " +
		                                 (found ? "a '" + std::string(found) + "' capsule" : "something else"));
	}
	auto *held = static_cast<T *>(PyCapsule_GetPointer(capsule.ptr(), kind));
	if (held == nullptr || held->release == nullptr) {
		throw cxx::InvalidInputException("the '" + std::string(kind) + "' capsule of " + source +
		                                 " is released already");
	}
	return *held;
}

template <class T>
ArrowOwned<T> TakeFromCapsule(nb::handle capsule, const char *kind, const std::string &source) {
	auto &held = InCapsule<T>(capsule, kind, source);
	ArrowOwned<T> taken;
	taken.value = held;
	held.release = nullptr;
	return taken;
}

} // namespace duckdb_python
