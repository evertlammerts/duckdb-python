#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

inline FunctionNullHandling FunctionNullHandlingFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "default") {
		return FunctionNullHandling::DEFAULT_NULL_HANDLING;
	}
	if (ltype == "special") {
		return FunctionNullHandling::SPECIAL_HANDLING;
	}
	throw InvalidInputException("'%s' is not a recognized type for 'null_handling'", type);
}

inline FunctionNullHandling FunctionNullHandlingFromInteger(int64_t value) {
	if (value == 0) {
		return FunctionNullHandling::DEFAULT_NULL_HANDLING;
	}
	if (value == 1) {
		return FunctionNullHandling::SPECIAL_HANDLING;
	}
	throw InvalidInputException("'%d' is not a recognized type for 'null_handling'", value);
}

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! See python_udf_type_enum.hpp for why this owns its value and delegates the enum case to a local base
//! caster instead of inheriting type_caster_base. Must stay visible in every TU (included from
//! pybind_wrapper.hpp).
template <>
struct type_caster<duckdb::FunctionNullHandling> {
	PYBIND11_TYPE_CASTER(duckdb::FunctionNullHandling, const_name("FunctionNullHandling"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::FunctionNullHandlingFromString(src.cast<std::string>());
			return true;
		}
		if (isinstance<int_>(src)) {
			value = duckdb::FunctionNullHandlingFromInteger(src.cast<int64_t>());
			return true;
		}
		type_caster_base<duckdb::FunctionNullHandling> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::FunctionNullHandling *>(base);
		return true;
	}

	static handle cast(duckdb::FunctionNullHandling src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::FunctionNullHandling>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
