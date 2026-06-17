#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

enum class PythonUDFType : uint8_t { NATIVE, ARROW };

inline PythonUDFType PythonUDFTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default" || ltype == "native") {
		return PythonUDFType::NATIVE;
	}
	if (ltype == "arrow") {
		return PythonUDFType::ARROW;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'udf_type'", type);
}

inline PythonUDFType PythonUDFTypeFromInteger(int64_t value) {
	if (value == 0) {
		return PythonUDFType::NATIVE;
	}
	if (value == 1) {
		return PythonUDFType::ARROW;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'udf_type'", value);
}

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! Accepts the registered PythonUDFType enum, or a string / integer naming one. Unlike the previous version,
//! this does NOT inherit type_caster_base: it owns its value (PYBIND11_TYPE_CASTER) and delegates only the
//! enum case to a *local* base caster. Inheriting the base while also writing custom branches is what made
//! the old version accept str XOR the enum depending on include visibility. This specialization must be
//! visible in every TU that converts PythonUDFType (it is included from pybind_wrapper.hpp), otherwise it is
//! UB. Keeping the binding parameter typed as the enum preserves the type + default in help()/stubs.
template <>
struct type_caster<duckdb::PythonUDFType> {
	PYBIND11_TYPE_CASTER(duckdb::PythonUDFType, const_name("PythonUDFType"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::PythonUDFTypeFromString(src.cast<std::string>());
			return true;
		}
		if (isinstance<int_>(src)) {
			value = duckdb::PythonUDFTypeFromInteger(src.cast<int64_t>());
			return true;
		}
		// Otherwise it must be an actual (registered) PythonUDFType instance.
		type_caster_base<duckdb::PythonUDFType> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::PythonUDFType *>(base);
		return true;
	}

	static handle cast(duckdb::PythonUDFType src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::PythonUDFType>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
