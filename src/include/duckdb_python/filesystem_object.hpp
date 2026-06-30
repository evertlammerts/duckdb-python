//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/filesystem_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/registered_py_object.hpp"
#include "duckdb_python/pyfilesystem.hpp"

namespace duckdb {

class FileSystemObject : public RegisteredObject {
public:
	explicit FileSystemObject(nb::object fs, vector<string> filenames_p)
	    : RegisteredObject(std::move(fs)), filenames(std::move(filenames_p)) {
	}
	~FileSystemObject() override {
		nb::gil_scoped_acquire acquire;
		// Assert that the 'obj' is a filesystem
		D_ASSERT(duckdb::PyUtil::IsInstance(
		    obj, DuckDBPyConnection::ImportCache()->duckdb.filesystem.ModifiedMemoryFileSystem()));
		for (auto &file : filenames) {
			obj.attr("delete")(file);
		}
	}

	vector<string> filenames;
};

} // namespace duckdb
