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

namespace nanobind {
namespace detail {

// Custom type caster for std::shared_ptr<duckdb::DuckDBPyType>.
//
// nanobind's default std::shared_ptr<T> caster strips cast_flags::convert before delegating to the inner caster,
// which disables implicit conversions for shared_ptr-typed arguments. DuckDBPyType, however, is routinely passed
// as a string ("VARCHAR"), a Python type object (int), a typing generic, or a dict, relying on its registered
// implicit conversions (as it did under pybind11). Those conversions construct brand-new, fully-owned
// DuckDBPyType objects, so they carry no dangling risk -- we therefore mirror nanobind's shared_ptr caster but
// KEEP the convert flag. (This specialization is visible in every TU that converts the type, since such TUs use
// DuckDBPyType and thus include this header.)
template <>
struct type_caster<std::shared_ptr<duckdb::DuckDBPyType>> {
	using T = duckdb::DuckDBPyType;
	using Caster = make_caster<T>;
	NB_TYPE_CASTER(std::shared_ptr<T>, Caster::Name)

	bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
		// NOTE: deliberately do NOT clear cast_flags::convert (see header comment).
		Caster caster;
		if (!caster.from_python(src, flags, cleanup)) {
			return false;
		}
		T *ptr = caster.operator T *();
		if (ptr) {
			ft_object_guard guard(src);
			if (auto sp = ptr->weak_from_this().lock()) {
				value = std::static_pointer_cast<T>(std::move(sp));
				return true;
			}
			value = shared_from_python(ptr, src);
			return true;
		}
		value = shared_from_python(ptr, src);
		return true;
	}

	static handle from_cpp(const std::shared_ptr<T> &value, rv_policy, cleanup_list *cleanup) noexcept {
		// DuckDBPyType is non-polymorphic and registers no type_hook, so this is a simplified version of
		// nanobind's shared_ptr from_cpp.
		bool is_new = false;
		T *ptr = value.get();
		handle result = nb_type_put(&typeid(T), ptr, rv_policy::reference, cleanup, &is_new);
		if (is_new) {
			auto pp = std::static_pointer_cast<void>(value);
			shared_from_cpp(std::move(pp), result.ptr());
		}
		return result;
	}
};

} // namespace detail
} // namespace nanobind
