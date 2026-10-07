//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/sql_types.cpp
//
//
//===----------------------------------------------------------------------===//

#include "sql_types.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace duckdb_python {

using duckdb::cxx::LogicalType;
using duckdb::cxx::LogicalTypeId;
using duckdb::cxx::Value;

// Whether two names are one, as the engine pairs struct fields and matches $name parameters: ignoring ASCII case.
bool EqualIgnoringCase(std::string_view a, std::string_view b) {
	const auto lower = [](char c) {
		return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
	};
	return a.size() == b.size() &&
	       std::equal(a.begin(), a.end(), b.begin(), [&](char x, char y) { return lower(x) == lower(y); });
}

bool IsStruct(LogicalTypeId id) {
	return id == LogicalTypeId::STRUCT || id == LogicalTypeId::TUPLE;
}

std::vector<std::string> StructNames(const LogicalType &type) {
	std::vector<std::string> names;
	for (duckdb::cxx::idx_t i = 0; i < type.GetStructChildCount(); i++) {
		names.push_back(type.GetStructChildName(i));
	}
	return names;
}

bool AllUnnamed(const std::vector<std::string> &names) {
	return std::all_of(names.begin(), names.end(), [](const std::string &name) { return name.empty(); });
}

// Which field of a target struct each field of a source struct fills when the engine casts one to the other: by
// position when either is unnamed, and then only between as many fields; else by name ignoring ASCII case, each
// target field taken by the first source field naming it. A source field paired with none is dropped.
std::vector<std::optional<std::size_t>> CastPairing(const std::vector<std::string> &source,
                                                    const std::vector<std::string> &target) {
	std::vector<std::optional<std::size_t>> pairing(source.size());
	if (AllUnnamed(source) || AllUnnamed(target)) {
		if (source.size() == target.size()) {
			for (std::size_t i = 0; i < source.size(); i++) {
				pairing[i] = i;
			}
		}
		return pairing;
	}
	std::vector<bool> taken(target.size(), false);
	for (std::size_t i = 0; i < source.size(); i++) {
		for (std::size_t j = 0; j < target.size(); j++) {
			if (!taken[j] && EqualIgnoringCase(source[i], target[j])) {
				taken[j] = true;
				pairing[i] = j;
				break;
			}
		}
	}
	return pairing;
}

} // namespace duckdb_python
