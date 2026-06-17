#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

enum class PythonExceptionHandling : uint8_t { FORWARD_ERROR, RETURN_NULL };

inline PythonExceptionHandling PythonExceptionHandlingFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default") {
		return PythonExceptionHandling::FORWARD_ERROR;
	}
	if (ltype == "return_null") {
		return PythonExceptionHandling::RETURN_NULL;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'exception_handling'", type);
}

inline PythonExceptionHandling PythonExceptionHandlingFromInteger(int64_t value) {
	if (value == 0) {
		return PythonExceptionHandling::FORWARD_ERROR;
	}
	if (value == 1) {
		return PythonExceptionHandling::RETURN_NULL;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'exception_handling'", value);
}

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! See python_udf_type_enum.hpp for the rationale (composition over inheritance, umbrella visibility).
template <>
struct type_caster<duckdb::PythonExceptionHandling> {
	PYBIND11_TYPE_CASTER(duckdb::PythonExceptionHandling, const_name("PythonExceptionHandling"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::PythonExceptionHandlingFromString(src.cast<std::string>());
			return true;
		}
		if (isinstance<int_>(src)) {
			value = duckdb::PythonExceptionHandlingFromInteger(src.cast<int64_t>());
			return true;
		}
		type_caster_base<duckdb::PythonExceptionHandling> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::PythonExceptionHandling *>(base);
		return true;
	}

	static handle cast(duckdb::PythonExceptionHandling src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::PythonExceptionHandling>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
