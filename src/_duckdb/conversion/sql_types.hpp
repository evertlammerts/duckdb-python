//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/sql_types.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "duckdb_cpp.hpp"

namespace duckdb_python {

/// Whether the id is one of the engine's struct kinds.
bool IsStruct(duckdb::cxx::LogicalTypeId id);

/// The field names of a struct type, in order.
std::vector<std::string> StructNames(const duckdb::cxx::LogicalType &type);

/// For each source field, the target field the engine's cast pairs it with, or nullopt for one it drops: by name
/// ignoring ASCII case, or by position when either struct is unnamed.
std::vector<std::optional<std::size_t>> CastPairing(const std::vector<std::string> &source,
                                                    const std::vector<std::string> &target);

/// Whether two names are one, as the engine pairs struct fields and matches $name parameters: ignoring ASCII case.
bool EqualIgnoringCase(std::string_view a, std::string_view b);

/// Whether `match(id)` holds for the type or any part nested anywhere inside it.
template <class MATCH>
bool Contains(const duckdb::cxx::LogicalType &type, const MATCH &match) {
	using duckdb::cxx::LogicalTypeId;
	const auto id = type.GetTypeId();
	if (match(id)) {
		return true;
	}
	switch (id) {
	case LogicalTypeId::LIST:
		return Contains(type.GetListChildType(), match);
	case LogicalTypeId::ARRAY:
		return Contains(type.GetArrayChildType(), match);
	case LogicalTypeId::MAP:
		return Contains(type.GetMapKeyType(), match) || Contains(type.GetMapValueType(), match);
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE:
		for (duckdb::cxx::idx_t i = 0; i < type.GetStructChildCount(); i++) {
			if (Contains(type.GetStructChildType(i), match)) {
				return true;
			}
		}
		return false;
	case LogicalTypeId::UNION:
		for (duckdb::cxx::idx_t i = 0; i < type.GetUnionMemberCount(); i++) {
			if (Contains(type.GetUnionMemberType(i), match)) {
				return true;
			}
		}
		return false;
	default:
		return false;
	}
}

} // namespace duckdb_python
