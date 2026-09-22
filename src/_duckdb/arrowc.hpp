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

/// The stream a capsule carries, checked to be one that has not been released; `name` is the registered name
/// for the error.
ArrowArrayStream &StreamOf(nb::handle capsule, const std::string &name);

bool IsBatch(const ArrowSchema &schema);

/// Turns a schema that is not a batch into a batch of one column: a struct with the schema as its only child.
void WrapAsBatch(ArrowSchema &schema);
/// The array counterpart: a struct array with no validity buffer whose only child is the array.
void WrapAsBatch(ArrowArray &array);

/// The name of a batch schema's child, empty when it has none.
std::string NameOf(const ArrowSchema &schema, cxx::idx_t index);

/// Moves the struct out of a capsule, leaving the capsule nothing to release.
template <class T>
T TakeFromCapsule(nb::handle capsule, const char *kind, const std::string &name) {
	if (!PyCapsule_IsValid(capsule.ptr(), kind)) {
		throw cxx::InvalidInputException("the object registered as '" + name + "' did not hand out an '" + kind +
		                                 "' capsule");
	}
	auto *held = static_cast<T *>(PyCapsule_GetPointer(capsule.ptr(), kind));
	if (held == nullptr || held->release == nullptr) {
		throw cxx::InvalidInputException("the object registered as '" + name + "' handed out a released '" + kind +
		                                 "' capsule");
	}
	T taken = *held;
	held->release = nullptr;
	return taken;
}

} // namespace duckdb_python
