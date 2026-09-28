//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/arrowc.cpp
//
//
//===----------------------------------------------------------------------===//

#include "arrowc.hpp"

#include <cerrno>
#include <cstring>
#include <memory>
#include <utility>

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

bool IsBatch(const ArrowSchema &schema) {
	return schema.format != nullptr && std::strcmp(schema.format, "+s") == 0;
}

void WrapAsBatch(ArrowSchema &schema) {
	auto *child = new ArrowSchema(schema);
	auto **children = new ArrowSchema *[1] {child};
	// Allocated before the wrapper takes over, so a failed allocation leaves the caller's schema releasable.
	auto *wrapped = new Wrapped<ArrowSchema> {child, children};
	schema = ArrowSchema {};
	schema.format = "+s";
	schema.name = "";
	schema.n_children = 1;
	schema.children = children;
	schema.release = &ReleaseWrapped<ArrowSchema>;
	schema.private_data = wrapped;
}

void WrapAsBatch(ArrowArray &array) {
	auto *child = new ArrowArray(array);
	auto **children = new ArrowArray *[1] {child};
	auto *wrapped = new Wrapped<ArrowArray> {child, children};
	array = ArrowArray {};
	array.length = child->length;
	array.null_count = 0;
	array.offset = 0;
	array.n_buffers = 1;
	array.buffers = kNoBuffers;
	array.n_children = 1;
	array.children = children;
	array.release = &ReleaseWrapped<ArrowArray>;
	array.private_data = wrapped;
}

std::string NameOf(const ArrowSchema &schema, cxx::idx_t index) {
	const auto *child = schema.children[index];
	return child->name ? child->name : "";
}

namespace {

/// What `ChainStreams` below reads from: several Arrow exports, the parts, read one after another as one stream.
/// Holds the object whose export gives the schema, the iterator of parts still to read, and the part being read.
struct ChainedStream {
	nb::object schema_source;
	/// A Python iterator, so a plain exhaustion is `next(parts)` raising StopIteration.
	nb::object parts;
	ArrowOwned<ArrowArrayStream> current;
	std::string last_error;
};

ChainedStream &PrivateOf(ArrowArrayStream &stream) {
	return *static_cast<ChainedStream *>(stream.private_data);
}

ArrowOwned<ArrowArrayStream> TakeExport(nb::handle object, const std::string &what) {
	return TakeFromCapsule<ArrowArrayStream>(object.attr("__arrow_c_stream__")(), kStreamCapsule, what);
}

/// Reads the schema off a fresh export of the schema source; throws what the export throws.
int ReadSchema(ChainedStream &chain, ArrowSchema *out) {
	auto stream = TakeExport(chain.schema_source, "the chained stream's schema source");
	const int rc = stream.value.get_schema(&stream.value, out);
	if (rc != 0) {
		chain.last_error = StreamError(stream.value);
	}
	return rc;
}

/// The next array: the next of the part being read, or the first of the part after it; throws what a part throws.
int ReadNext(ChainedStream &chain, ArrowArray *out) {
	for (;;) {
		if (chain.current) {
			auto &stream = chain.current.value;
			int rc = 0;
			{
				// Reading a part's next array is native work, or takes the GIL itself where it runs Python.
				nb::gil_scoped_release released;
				rc = stream.get_next(&stream, out);
			}
			if (rc != 0) {
				chain.last_error = StreamError(stream);
				return rc;
			}
			if (out->release != nullptr) {
				return 0;
			}
			chain.current.Release();
		}
		// PyIter_Next clears a StopIteration itself, so a null with no error set is the end of the parts.
		PyObject *raw = PyIter_Next(chain.parts.ptr());
		if (raw == nullptr) {
			if (PyErr_Occurred()) {
				throw nb::python_error();
			}
			out->release = nullptr;
			return 0;
		}
		nb::object part = nb::steal(raw);
		chain.current = TakeExport(part, "a part of the chained stream");
	}
}

/// Runs a callback body under the GIL and turns every exception into the stream's error return: a Python or
/// engine error is described for the caller, and anything else, such as a failed allocation while describing
/// one, still comes back as a code rather than crossing the C boundary.
template <class F>
int Guarded(ChainedStream &chain, F &&body) {
	nb::gil_scoped_acquire gil;
	try {
		try {
			return body();
		} catch (nb::python_error &error) {
			chain.last_error = DescribePythonError(error);
		} catch (const cxx::Exception &error) {
			chain.last_error = error.what();
		}
		return EINVAL;
	} catch (...) {
		return EINVAL;
	}
}

int ChainGetSchema(ArrowArrayStream *self, ArrowSchema *out) {
	auto &chain = PrivateOf(*self);
	return Guarded(chain, [&] { return ReadSchema(chain, out); });
}

int ChainGetNext(ArrowArrayStream *self, ArrowArray *out) {
	auto &chain = PrivateOf(*self);
	return Guarded(chain, [&] { return ReadNext(chain, out); });
}

const char *ChainGetLastError(ArrowArrayStream *self) {
	return PrivateOf(*self).last_error.c_str();
}

void ChainRelease(ArrowArrayStream *self) {
	nb::gil_scoped_acquire gil;
	delete static_cast<ChainedStream *>(self->private_data);
	self->private_data = nullptr;
	self->get_schema = nullptr;
	self->get_next = nullptr;
	self->get_last_error = nullptr;
	self->release = nullptr;
}

void ReleaseStreamCapsule(void *ptr) noexcept {
	auto *stream = static_cast<ArrowArrayStream *>(ptr);
	if (stream->release != nullptr) {
		stream->release(stream);
	}
	delete stream;
}

} // namespace

nb::capsule ChainStreams(nb::object schema, nb::handle parts) {
	std::unique_ptr<ChainedStream> chain(new ChainedStream {std::move(schema), nb::iter(parts)});
	std::unique_ptr<ArrowArrayStream> stream(new ArrowArrayStream {});
	stream->get_schema = &ChainGetSchema;
	stream->get_next = &ChainGetNext;
	stream->get_last_error = &ChainGetLastError;
	stream->release = &ChainRelease;
	stream->private_data = chain.get();
	nb::capsule capsule(stream.get(), kStreamCapsule, &ReleaseStreamCapsule);
	chain.release();
	stream.release();
	return capsule;
}

} // namespace duckdb_python
