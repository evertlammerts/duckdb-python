//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/predicate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <functional>
#include <string>

#include "lifetime.hpp"

namespace duckdb_python {

/// A predicate the optimizer offers, or a part of one, that the translation does not model; the engine keeps
/// applying the whole predicate.
struct Refused {};

/// A column a predicate refers to, as the scan declared it.
struct PredicateColumn {
	std::string name;
	cxx::LogicalTypeId type;
};

/// The declared column behind a column reference. A reference indexes the columns the query reads, not the
/// columns the scan declared, and only the pushdown input knows the mapping; the walker gets it through this.
using ColumnResolver = std::function<PredicateColumn(cxx::idx_t)>;

/// Turns a predicate the optimizer offers into frame expression nodes a source can read: comparisons of a column
/// with a constant, with the column on the left, `IN`, `IS NULL`, `AND`, `OR` and `NOT`. A constant is only
/// carried when it has the column's type, since the engine compares after casting and a source would not, and when
/// a Python value holds it without loss; whether a library then compares the type as the engine does is for the
/// source to decide in `accepts`.
/// @throws Refused For a predicate with any node outside that set, including a NULL or NaN constant.
nb::object TranslatePredicate(const cxx::Expression &node, const ColumnResolver &resolve,
                              ConversionContext &conversion);

} // namespace duckdb_python
