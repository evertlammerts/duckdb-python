//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/untyped.cpp
//
//
//===----------------------------------------------------------------------===//

#include "untyped.hpp"

#include "sql_types.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <cstdint>
#include <string>
#include <string_view>

namespace duckdb_python {

using duckdb::cxx::LogicalType;
using duckdb::cxx::LogicalTypeId;
using duckdb::cxx::Value;

using duckdb::cxx::Connection;

bool IsUntypedMarker(const LogicalType &type) {
	return type.GetTypeId() == LogicalTypeId::INTEGER;
}

bool ContainsUntyped(const LogicalType &type) {
	return Contains(type, [](LogicalTypeId id) { return id == LogicalTypeId::INTEGER; });
}

namespace {

bool Unnamed(const LogicalType &type) {
	return type.GetTypeId() != LogicalTypeId::STRUCT || AllUnnamed(StructNames(type));
}

template <class SCOPE>
LogicalType Copy(SCOPE &scope, const LogicalType &type) {
	return Value::CreateNull(scope, type).GetLogicalType();
}

// `type` with each untyped place typed as `filler` types it, struct fields meeting by exact name, or nullopt when that

} // namespace

template <class SCOPE>
std::optional<LogicalType> FilledFromSibling(SCOPE &scope, const LogicalType &type, const LogicalType &filler,
                                             bool widen) {
	if (IsUntypedMarker(type)) {
		return IsUntypedMarker(filler) ? std::nullopt : std::optional<LogicalType>(Copy(scope, filler));
	}
	const auto id = type.GetTypeId();
	if (id != filler.GetTypeId()) {
		return std::nullopt;
	}
	if (id == LogicalTypeId::LIST) {
		auto child = FilledFromSibling(scope, type.GetListChildType(), filler.GetListChildType(), widen);
		return child ? std::optional<LogicalType>(Value::CreateList(scope, *child).GetLogicalType()) : std::nullopt;
	}
	if (id == LogicalTypeId::MAP) {
		auto key = FilledFromSibling(scope, type.GetMapKeyType(), filler.GetMapKeyType(), widen);
		auto item = FilledFromSibling(scope, type.GetMapValueType(), filler.GetMapValueType(), widen);
		if (!key && !item) {
			return std::nullopt;
		}
		const auto own_key = type.GetMapKeyType();
		const auto own_item = type.GetMapValueType();
		return Value::CreateMap(scope, key ? *key : own_key, item ? *item : own_item).GetLogicalType();
	}
	if (Unnamed(type) || Unnamed(filler)) {
		return std::nullopt;
	}
	const auto names = StructNames(type);
	const auto others = StructNames(filler);
	bool changed = false;
	std::vector<std::optional<LogicalType>> filled(names.size());
	for (std::size_t i = 0; i < names.size(); i++) {
		const auto found = std::find(others.begin(), others.end(), names[i]);
		if (found != others.end()) {
			filled[i] = FilledFromSibling(
			    scope, type.GetStructChildType(i),
			    filler.GetStructChildType(static_cast<duckdb::cxx::idx_t>(found - others.begin())), widen);
			changed = changed || filled[i].has_value();
		}
	}
	std::vector<std::size_t> added;
	for (std::size_t j = 0; widen && j < others.size(); j++) {
		if (std::find(names.begin(), names.end(), others[j]) == names.end()) {
			added.push_back(j);
		}
	}
	if (!changed && added.empty()) {
		return std::nullopt;
	}
	std::vector<std::pair<std::string, Value>> fields;
	for (std::size_t i = 0; i < names.size(); i++) {
		const auto own_field = type.GetStructChildType(i);
		fields.emplace_back(names[i], Value::CreateNull(scope, filled[i] ? *filled[i] : own_field));
	}
	for (const auto j : added) {
		fields.emplace_back(others[j], Value::CreateNull(scope, filler.GetStructChildType(j)));
	}
	return Value::CreateStruct(scope, fields).GetLogicalType();
}

// `value` as a value of `type`, which differs from its own type only at untyped places, where only NULLs and empty
// lists stand, so nothing else changes and no cast runs.
template <class SCOPE>
Value RelabelUntyped(SCOPE &scope, Value value, const LogicalType &type) {
	if (value.IsNull()) {
		return Value::CreateNull(scope, type);
	}
	const auto id = type.GetTypeId();
	const auto count = value.GetChildCount();
	if (id == LogicalTypeId::LIST) {
		const auto child = type.GetListChildType();
		if (count == 0) {
			return Value::CreateList(scope, child);
		}
		std::vector<Value> children;
		for (duckdb::cxx::idx_t i = 0; i < count; i++) {
			children.push_back(RelabelUntyped(scope, value.GetChild(i), child));
		}
		return Value::CreateList(scope, children);
	}
	if (id == LogicalTypeId::MAP) {
		const auto key = type.GetMapKeyType();
		const auto item = type.GetMapValueType();
		if (count == 0) {
			return Value::CreateMap(scope, key, item);
		}
		std::vector<std::pair<Value, Value>> entries;
		for (duckdb::cxx::idx_t i = 0; i + 1 < count; i += 2) {
			entries.emplace_back(RelabelUntyped(scope, value.GetChild(i), key),
			                     RelabelUntyped(scope, value.GetChild(i + 1), item));
		}
		return Value::CreateMap(scope, entries);
	}
	if (Unnamed(type)) {
		return value;
	}
	const auto names = StructNames(value.GetLogicalType());
	std::vector<std::pair<std::string, Value>> fields;
	for (duckdb::cxx::idx_t i = 0; i < count; i++) {
		fields.emplace_back(names[i], RelabelUntyped(scope, value.GetChild(i), type.GetStructChildType(i)));
	}
	return Value::CreateStruct(scope, fields);
}

template std::optional<LogicalType> FilledFromSibling<Connection>(Connection &, const LogicalType &,
                                                                  const LogicalType &, bool);
template std::optional<LogicalType> FilledFromSibling<duckdb::cxx::Context>(duckdb::cxx::Context &, const LogicalType &,
                                                                            const LogicalType &, bool);
template Value RelabelUntyped<Connection>(Connection &, Value, const LogicalType &);
template Value RelabelUntyped<duckdb::cxx::Context>(duckdb::cxx::Context &, Value, const LogicalType &);

bool ContainsUnknownOrAny(const LogicalType &type) {
	return Contains(type, [](LogicalTypeId id) {
		return id == LogicalTypeId::INVALID || id == LogicalTypeId::SQLNULL || id == LogicalTypeId::UNKNOWN ||
		       id == LogicalTypeId::ANY;
	});
}

namespace {

// The type a value binds as with each untyped place taking the type the binder expects there, or nullopt when an
// untyped place gets nothing: the expectation is UNKNOWN, its form differs, or it does not name the field. A place
// the value types itself always keeps its type.
template <class SCOPE>
std::optional<LogicalType> FilledFromExpected(SCOPE &scope, const LogicalType &type, const LogicalType &expected) {
	if (!ContainsUntyped(type)) {
		return Copy(scope, type);
	}
	const auto want = expected.GetTypeId();
	if (want == LogicalTypeId::UNKNOWN) {
		return std::nullopt;
	}
	if (IsUntypedMarker(type)) {
		return Copy(scope, expected);
	}
	const auto id = type.GetTypeId();
	if (id == LogicalTypeId::LIST && (want == LogicalTypeId::LIST || want == LogicalTypeId::ARRAY)) {
		const auto child = FilledFromExpected(scope, type.GetListChildType(),
		                                      want == LogicalTypeId::LIST ? expected.GetListChildType()
		                                                                  : expected.GetArrayChildType());
		if (!child) {
			return std::nullopt;
		}
		return Value::CreateList(scope, *child).GetLogicalType();
	}
	if (id == LogicalTypeId::MAP && want == LogicalTypeId::MAP) {
		const auto key = FilledFromExpected(scope, type.GetMapKeyType(), expected.GetMapKeyType());
		const auto item = FilledFromExpected(scope, type.GetMapValueType(), expected.GetMapValueType());
		if (!key || !item) {
			return std::nullopt;
		}
		return Value::CreateMap(scope, *key, *item).GetLogicalType();
	}
	if (id == LogicalTypeId::STRUCT && want == LogicalTypeId::STRUCT && !Unnamed(type) && !Unnamed(expected)) {
		const auto names = StructNames(type);
		const auto others = StructNames(expected);
		std::vector<std::pair<std::string, Value>> fields;
		for (std::size_t i = 0; i < names.size(); i++) {
			auto field = type.GetStructChildType(i);
			if (ContainsUntyped(field)) {
				const auto found = std::find(others.begin(), others.end(), names[i]);
				if (found == others.end()) {
					return std::nullopt;
				}
				const auto other = expected.GetStructChildType(static_cast<duckdb::cxx::idx_t>(found - others.begin()));
				auto filled = FilledFromExpected(scope, field, other);
				if (!filled) {
					return std::nullopt;
				}
				field = std::move(*filled);
			}
			fields.emplace_back(names[i], Value::CreateNull(scope, field));
		}
		return Value::CreateStruct(scope, fields).GetLogicalType();
	}
	return std::nullopt;
}

} // namespace

template <class SCOPE>
bool FillUntypedFromExpected(SCOPE &scope, Value &value, const LogicalType &expected) {
	auto target = FilledFromExpected(scope, value.GetLogicalType(), expected);
	if (!target) {
		return false;
	}
	value = RelabelUntyped(scope, std::move(value), *target);
	return true;
}

template bool FillUntypedFromExpected<Connection>(Connection &, Value &, const LogicalType &);

} // namespace duckdb_python
