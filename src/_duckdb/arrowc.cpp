//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrowc.cpp
//
//
//===----------------------------------------------------------------------===//

#include "arrowc.hpp"

#include <cstring>

namespace duckdb_python {
namespace {

/// A struct array with no validity buffer still declares one buffer slot, holding null.
const void *kNoBuffers[1] = {nullptr};

/// The moved-in child a wrapper owns, released and freed with the wrapper.
template <class T>
struct Wrapped {
	T *child;
	T **children;
};

template <class T>
void ReleaseWrapped(T *wrapper) {
	auto *owned = static_cast<Wrapped<T> *>(wrapper->private_data);
	if (owned->child->release != nullptr) {
		owned->child->release(owned->child);
	}
	delete owned->child;
	delete[] owned->children;
	delete owned;
	wrapper->release = nullptr;
}

} // namespace

std::string StreamError(ArrowArrayStream &stream) {
	const char *text = stream.get_last_error ? stream.get_last_error(&stream) : nullptr;
	return text ? text : "no error message";
}

ArrowArrayStream &StreamOf(nb::handle capsule, const std::string &name) {
	if (!PyCapsule_IsValid(capsule.ptr(), kStreamCapsule)) {
		const char *found = PyCapsule_CheckExact(capsule.ptr()) ? PyCapsule_GetName(capsule.ptr()) : nullptr;
		throw cxx::InvalidInputException("the object registered as '" + name + "' did not export an '" +
		                                 kStreamCapsule + "' capsule but " +
		                                 (found ? "a '" + std::string(found) + "' capsule" : "something else"));
	}
	auto *stream = static_cast<ArrowArrayStream *>(PyCapsule_GetPointer(capsule.ptr(), kStreamCapsule));
	if (stream == nullptr || stream->release == nullptr) {
		throw cxx::InvalidInputException("the stream registered as '" + name + "' is released already");
	}
	return *stream;
}

bool IsBatch(const ArrowSchema &schema) {
	return schema.format != nullptr && std::strcmp(schema.format, "+s") == 0;
}

void WrapAsBatch(ArrowSchema &schema) {
	auto *child = new ArrowSchema(schema);
	auto **children = new ArrowSchema *[1] {child};
	schema = ArrowSchema {};
	schema.format = "+s";
	schema.name = "";
	schema.n_children = 1;
	schema.children = children;
	schema.release = &ReleaseWrapped<ArrowSchema>;
	schema.private_data = new Wrapped<ArrowSchema> {child, children};
}

void WrapAsBatch(ArrowArray &array) {
	auto *child = new ArrowArray(array);
	auto **children = new ArrowArray *[1] {child};
	array = ArrowArray {};
	array.length = child->length;
	array.null_count = 0;
	array.offset = 0;
	array.n_buffers = 1;
	array.buffers = kNoBuffers;
	array.n_children = 1;
	array.children = children;
	array.release = &ReleaseWrapped<ArrowArray>;
	array.private_data = new Wrapped<ArrowArray> {child, children};
}

std::string NameOf(const ArrowSchema &schema, cxx::idx_t index) {
	const auto *child = schema.children[index];
	return child->name ? child->name : "";
}

} // namespace duckdb_python
