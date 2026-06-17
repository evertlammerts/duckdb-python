#pragma once

#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

inline ExplainType ExplainTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "standard") {
		return ExplainType::EXPLAIN_STANDARD;
	}
	if (ltype == "analyze") {
		return ExplainType::EXPLAIN_ANALYZE;
	}
	throw InvalidInputException("Unrecognized type for 'explain'");
}

inline ExplainType ExplainTypeFromInteger(int64_t value) {
	if (value == 0) {
		return ExplainType::EXPLAIN_STANDARD;
	}
	if (value == 1) {
		return ExplainType::EXPLAIN_ANALYZE;
	}
	throw InvalidInputException("Unrecognized type for 'explain'");
}

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! See python_udf_type_enum.hpp for the rationale (composition over inheritance, umbrella visibility).
template <>
struct type_caster<duckdb::ExplainType> {
	PYBIND11_TYPE_CASTER(duckdb::ExplainType, const_name("ExplainType"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::ExplainTypeFromString(src.cast<std::string>());
			return true;
		}
		if (isinstance<int_>(src)) {
			value = duckdb::ExplainTypeFromInteger(src.cast<int64_t>());
			return true;
		}
		type_caster_base<duckdb::ExplainType> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::ExplainType *>(base);
		return true;
	}

	static handle cast(duckdb::ExplainType src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::ExplainType>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
