#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/enum_util.hpp"

namespace duckdb {

inline RenderMode RenderModeFromString(const string &value) {
	return EnumUtil::FromString<RenderMode>(value.empty() ? "ROWS" : value);
}

inline RenderMode RenderModeFromInteger(int64_t value) {
	if (value == 0) {
		return RenderMode::ROWS;
	}
	if (value == 1) {
		return RenderMode::COLUMNS;
	}
	throw InvalidInputException("Unrecognized type for 'render_mode'");
}

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! See python_udf_type_enum.hpp for the rationale (composition over inheritance, umbrella visibility).
template <>
struct type_caster<duckdb::RenderMode> {
	PYBIND11_TYPE_CASTER(duckdb::RenderMode, const_name("RenderMode"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::RenderModeFromString(src.cast<std::string>());
			return true;
		}
		if (isinstance<int_>(src)) {
			value = duckdb::RenderModeFromInteger(src.cast<int64_t>());
			return true;
		}
		type_caster_base<duckdb::RenderMode> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::RenderMode *>(base);
		return true;
	}

	static handle cast(duckdb::RenderMode src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::RenderMode>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
