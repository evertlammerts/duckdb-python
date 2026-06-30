#include "duckdb_python/pyutil.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

bool PyUtil::GilCheck() {
	return (bool)PyGILState_Check();
}

void PyUtil::GilAssert() {
	if (!GilCheck()) {
		throw InternalException("The GIL should be held for this operation, but it's not!");
	}
}

bool PyUtil::IsListLike(nb::handle obj) {
	if (nb::isinstance<nb::str>(obj) || nb::isinstance<nb::bytes>(obj)) {
		return false;
	}
	if (IsDictLike(obj)) {
		return false;
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto iterable = import_cache.collections.abc.Iterable();
	return IsInstance(obj, iterable);
}

bool PyUtil::IsDictLike(nb::handle obj) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto mapping = import_cache.collections.abc.Mapping();
	return IsInstance(obj, mapping);
}

} // namespace duckdb
