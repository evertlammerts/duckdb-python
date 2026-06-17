#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

struct PythonCSVLineTerminator {
public:
	PythonCSVLineTerminator() = delete;
	enum class Type { LINE_FEED, CARRIAGE_RETURN_LINE_FEED };

public:
	static Type FromString(const string &str) {
		if (str == "\\n") {
			return Type::LINE_FEED;
		}
		if (str == "\\r\\n") {
			return Type::CARRIAGE_RETURN_LINE_FEED;
		}
		if (str == "LINE_FEED") {
			return Type::LINE_FEED;
		}
		if (str == "CARRIAGE_RETURN_LINE_FEED") {
			return Type::CARRIAGE_RETURN_LINE_FEED;
		}
		throw InvalidInputException("'%s' is not a recognized type for 'lineterminator'", str);
	}
	static string ToString(Type type) {
		switch (type) {
		case Type::LINE_FEED:
			return "\\n";
		case Type::CARRIAGE_RETURN_LINE_FEED:
			return "\\r\\n";
		default:
			throw NotImplementedException("No conversion for PythonCSVLineTerminator enum to string");
		}
	}
};

} // namespace duckdb

namespace PYBIND11_NAMESPACE {
namespace detail {

//! See python_udf_type_enum.hpp for the rationale (composition over inheritance, umbrella visibility).
//! Only a string or the enum itself are accepted (no integer form).
template <>
struct type_caster<duckdb::PythonCSVLineTerminator::Type> {
	PYBIND11_TYPE_CASTER(duckdb::PythonCSVLineTerminator::Type, const_name("CSVLineTerminator"));

	bool load(handle src, bool convert) {
		if (isinstance<str>(src)) {
			value = duckdb::PythonCSVLineTerminator::FromString(src.cast<std::string>());
			return true;
		}
		type_caster_base<duckdb::PythonCSVLineTerminator::Type> base;
		if (!base.load(src, convert)) {
			return false;
		}
		value = *static_cast<duckdb::PythonCSVLineTerminator::Type *>(base);
		return true;
	}

	static handle cast(duckdb::PythonCSVLineTerminator::Type src, return_value_policy policy, handle parent) {
		return type_caster_base<duckdb::PythonCSVLineTerminator::Type>::cast(src, policy, parent);
	}
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
