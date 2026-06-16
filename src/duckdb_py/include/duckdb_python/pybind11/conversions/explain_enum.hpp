#pragma once

#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

using duckdb::ExplainType;
using duckdb::InvalidInputException;
using duckdb::string;
using duckdb::StringUtil;

namespace py = pybind11;

static ExplainType ExplainTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "standard") {
		return ExplainType::EXPLAIN_STANDARD;
	} else if (ltype == "analyze") {
		return ExplainType::EXPLAIN_ANALYZE;
	} else {
		throw InvalidInputException("Unrecognized type for 'explain'");
	}
}

static ExplainType ExplainTypeFromInteger(int64_t value) {
	if (value == 0) {
		return ExplainType::EXPLAIN_STANDARD;
	} else if (value == 1) {
		return ExplainType::EXPLAIN_ANALYZE;
	} else {
		throw InvalidInputException("Unrecognized type for 'explain'");
	}
}

//! Resolve a Python explain-type argument (ExplainType enum, str, or int) to an ExplainType.
//! NOTE: deliberately NOT a pybind type_caster. A custom caster inheriting type_caster_base shadows the
//! registered py::enum_ inconsistently across translation units - it ends up accepting str/int XOR the enum
//! instance, never both, depending on which TU sees the specialization. Explicit dispatch at the call site is
//! robust regardless of include order.
static ExplainType ExplainTypeFromPython(const py::object &obj) {
	if (py::isinstance<py::str>(obj)) {
		return ExplainTypeFromString(py::str(obj));
	}
	if (py::isinstance<py::int_>(obj)) {
		return ExplainTypeFromInteger(obj.cast<int64_t>());
	}
	// Fall through to the registered py::enum_ caster (handles an actual ExplainType, throws otherwise).
	return obj.cast<ExplainType>();
}
