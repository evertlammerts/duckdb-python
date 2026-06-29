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

//! Value-semantic wrapper around a LogicalType. There is no shared ownership to model -- every factory returns a
//! brand-new, fully-owned type. Bound to Python by value (returned as std::unique_ptr); implicit
//! str/type-object/dict -> DuckDBPyType conversions are handled by nanobind's value caster + the registered
//! implicitly_convertible<>() rules (no custom shared_ptr caster).
class DuckDBPyType {
public:
	explicit DuckDBPyType(LogicalType type);

public:
	static void Initialize(py::handle &m);

	//! Convert a Python object (an existing DuckDBPyType, a type string, a Python type object such as `int`, or a
	//! dict describing a struct) into an owned DuckDBPyType. An existing DuckDBPyType is copied (value semantics);
	//! anything else is routed through the registered Python constructor, which drives the same factories as the
	//! registered implicit conversions. Returns false (clearing any pending Python error) when the object can't be
	//! converted, so a caller can raise a context-specific message.
	static bool TryConvert(const py::object &object, std::unique_ptr<DuckDBPyType> &result);

public:
	bool Equals(const DuckDBPyType &other) const;
	bool EqualsString(const string &type_str) const;
	std::unique_ptr<DuckDBPyType> GetAttribute(const string &name) const;
	py::list Children() const;
	string ToString() const;
	const LogicalType &Type() const;
	string GetId() const;

private:
private:
	LogicalType type;
};

} // namespace duckdb
