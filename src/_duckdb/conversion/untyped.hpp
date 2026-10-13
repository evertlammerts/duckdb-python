//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/untyped.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <optional>
#include <vector>

#include "duckdb_cpp.hpp"

namespace duckdb_python {

// An untyped value is a None or an empty container: it carries no type, and the engine refuses a value without
// one. Each converts to a NULL or empty value of the marker type INTEGER, which no genuine converted value ever
// has, so the place stays recognizable until a real type is learned: first from a sibling at the same place,
// then from what the statement expects, and a parameter that stays untyped where the statement expects nothing
// is refused with a cast hint.

/// The marker type an untyped value takes.
inline duckdb::cxx::LogicalType UntypedMarkerType(duckdb::cxx::Factory &scope) {
	return scope.CreateType(duckdb::cxx::LogicalTypeId::INTEGER);
}

/// Whether this place's type is the marker itself.
bool IsUntypedMarker(const duckdb::cxx::LogicalType &type);

/// Whether the type holds the marker anywhere inside it.
bool ContainsUntyped(const duckdb::cxx::LogicalType &type);

/// Whether a type holds a place no value can be built of: INVALID, SQLNULL, UNKNOWN or ANY, which the binder
/// reports where a statement leaves a parameter free or takes any type.
bool ContainsUnknownOrAny(const duckdb::cxx::LogicalType &type);

/// `type` with each untyped place typed as `filler` has it, struct fields meeting by exact name, or nullopt when
/// that changes nothing. With `widen`, the fields only `filler` has are added, so one type collects what every
/// sibling says.
std::optional<duckdb::cxx::LogicalType> FilledFromSibling(duckdb::cxx::Factory &scope,
                                                          const duckdb::cxx::LogicalType &type,
                                                          const duckdb::cxx::LogicalType &filler, bool widen);

/// `value` carrying `type`, which may differ from its own type only at untyped places, so nothing else changes
/// and no cast runs.
duckdb::cxx::Value RelabelUntyped(duckdb::cxx::Factory &scope, duckdb::cxx::Value value,
                                  const duckdb::cxx::LogicalType &type);

/// `value` with each untyped place typed as `expected`, the type the binder expects at the value's position;
/// struct fields pair by exact name and a place the value types itself is never cast. False, with `value`
/// unchanged, when an untyped place gets nothing: the expectation's form differs or it does not name the field.
bool FillUntypedFromExpected(duckdb::cxx::Factory &scope, duckdb::cxx::Value &value,
                             const duckdb::cxx::LogicalType &expected);

} // namespace duckdb_python
