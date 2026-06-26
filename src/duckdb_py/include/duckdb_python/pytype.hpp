#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

class PyGenericAlias : public py::object {
public:
	using py::object::object;

public:
	static bool check_(const py::handle &object);
};

class PyUnionType : public py::object {
public:
	using py::object::object;

public:
	static bool check_(const py::handle &object);
};

class DuckDBPyType : public std::enable_shared_from_this<DuckDBPyType> {
public:
	explicit DuckDBPyType(LogicalType type);

public:
	static void Initialize(py::handle &m);

	//! Convert a Python object (an existing DuckDBPyType, a type string, a Python type object such as `int`, or a
	//! dict describing a struct) into a DuckDBPyType. nanobind's shared_ptr type caster strips the implicit-convert
	//! flag, so a plain try_cast<shared_ptr<DuckDBPyType>> no longer triggers DuckDBPyType's registered implicit
	//! conversion; this routes non-DuckDBPyType objects through the registered Python constructor. Returns false
	//! (without throwing) when the object can't be converted.
	static bool TryConvert(const py::object &object, std::shared_ptr<DuckDBPyType> &result);

public:
	bool Equals(const std::shared_ptr<DuckDBPyType> &other) const;
	bool EqualsString(const string &type_str) const;
	std::shared_ptr<DuckDBPyType> GetAttribute(const string &name) const;
	py::list Children() const;
	string ToString() const;
	const LogicalType &Type() const;
	string GetId() const;

private:
private:
	LogicalType type;
};

} // namespace duckdb
