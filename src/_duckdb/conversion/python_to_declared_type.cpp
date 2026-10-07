//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/python_to_declared_type.cpp
//
//
//===----------------------------------------------------------------------===//

#include "python_to_declared_type.hpp"

#include "conversion.hpp"
#include "python_to_value.hpp"
#include "sql_types.hpp"
#include "temporal.hpp"
#include "untyped.hpp"

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

namespace {

// The declared type a value is converted to, and the part of it being converted now, for a refusal's message.
struct RefusalContext {
	const LogicalType &declared;
	const LogicalType &part;
};

[[noreturn]] void RefuseDeclared(const std::string &shown, const RefusalContext &refusal, const std::string &why) {
	auto body = shown + " for ";
	if (!(refusal.part == refusal.declared)) {
		body += refusal.part.ToText() + " in ";
	}
	body += "its declared type " + refusal.declared.ToText() + ", " + why;
	throw DeclaredTypeRefusal("Invalid Input Error: " + body, body);
}

[[noreturn]] void RefuseDeclared(const Value &value, const RefusalContext &refusal, const std::string &why) {
	// A struct whose field names differ only in case has no text form.
	std::string shown = "a value";
	try {
		shown = value.ToText();
	} catch (...) {
	}
	RefuseDeclared(shown + " of type " + value.GetLogicalType().ToText(), refusal, why);
}

// What declared-type conversion carries everywhere: the engine call context and the module's Python
// constructors. Values keep their own unit here; only query parameters cap at microseconds.
struct Conversion {
	duckdb::cxx::Context &scope;
	ConversionContext &ctx;
};

Value ToDeclared(Conversion &conversion, nb::handle object, const TypeTree &node, const LogicalType &whole);

// A date, time or timestamp part: never across a time zone, and a date or timestamp exactly or not at all. Between two
// instants the exact value is returned, since the engine's cast rounds between some units and has none between others.
std::optional<Value> CastTemporalExactly(Conversion &conversion, const Value &value, const RefusalContext &refusal) {
	const auto from = value.GetLogicalType().GetTypeId();
	const auto to = refusal.part.GetTypeId();
	if (AssumesTimeZone(from, to)) {
		RefuseDeclared(value, refusal, "and converting between them would assume a time zone");
	}
	if (!IsInstant(from) || !IsInstant(to)) {
		return std::nullopt;
	}
	auto exact = TryCastTemporalExactly(conversion.scope, value, refusal.part);
	if (!exact) {
		RefuseDeclared(value, refusal, "which cannot hold it exactly");
	}
	return exact;
}

Value TemporalToDeclared(Conversion &conversion, Value value, const RefusalContext &refusal, LogicalTypeId to) {
	if (value.IsNull()) {
		return Value::CreateNull(conversion.scope, refusal.part);
	}
	// A date or time type has no parameters, so an equal id is the declared type itself, held as it is.
	if (value.GetLogicalType().GetTypeId() == to) {
		return value;
	}
	auto exact = CastTemporalExactly(conversion, value, refusal);
	return exact ? std::move(*exact) : value.Cast(conversion.scope, refusal.part);
}

// Each date or time in `from`, followed to where the engine's cast put it in `to`, through the member it chose of each
// union, and held there as `CastTemporalExactly` holds it. Everything else keeps the cast's meaning.
void RefuseLossyCast(Conversion &conversion, const Value &from, const Value &to, const RefusalContext &refusal) {
	if (from.IsNull()) {
		return;
	}
	const auto from_type = from.GetLogicalType();
	const auto to_type = to.GetLogicalType();
	if (to.IsNull()) {
		// The engine's cast of a value leaves NULL where SQL's CAST refuses, as for a list of another length than an
		// array member's.
		RefuseDeclared(from, RefusalContext {refusal.declared, to_type}, "which the engine's cast to it leaves NULL");
	}
	const auto to_id = to_type.GetTypeId();
	if (to_id == LogicalTypeId::UNION) {
		RefuseLossyCast(conversion, from, to.GetChild(1), refusal);
		return;
	}
	if (from_type == to_type) {
		return;
	}
	const auto from_id = from_type.GetTypeId();
	if (KindOf(from_id) != TemporalKind::NONE) {
		CastTemporalExactly(conversion, from, RefusalContext {refusal.declared, to_type});
		return;
	}
	if ((from_id == LogicalTypeId::LIST && (to_id == LogicalTypeId::LIST || to_id == LogicalTypeId::ARRAY)) ||
	    (from_id == LogicalTypeId::MAP && to_id == LogicalTypeId::MAP)) {
		for (duckdb::cxx::idx_t i = 0; i < from.GetChildCount(); i++) {
			RefuseLossyCast(conversion, from.GetChild(i), to.GetChild(i), refusal);
		}
		return;
	}
	if (IsStruct(from_id) && IsStruct(to_id)) {
		const auto pairing = CastPairing(StructNames(from_type), StructNames(to_type));
		for (std::size_t i = 0; i < pairing.size(); i++) {
			if (pairing[i]) {
				RefuseLossyCast(conversion, from.GetChild(i), to.GetChild(*pairing[i]), refusal);
			}
		}
		return;
	}
	// No implicit cast of the engine reaches here; one that later does is refused rather than left unchecked.
	if (ContainsTemporal(from_type)) {
		RefuseDeclared(from, RefusalContext {refusal.declared, to_type},
		               "and a date or time inside it could not be checked there");
	}
}

// The engine's cast puts a value in the UNION member it converts to most cheaply, as CAST does. A date or time that
// member holds only by assuming a time zone, or not exactly, refuses the value rather than sending it elsewhere.
Value UnionToDeclared(Conversion &conversion, nb::handle object, const RefusalContext &refusal) {
	const auto own = PythonToValue(conversion.scope, object, conversion.ctx);
	auto cast = own.Cast(conversion.scope, refusal.part);
	RefuseLossyCast(conversion, own, cast, refusal);
	return cast;
}

// A dict with text keys as the declared STRUCT, each field converted as the field the engine's cast pairs it with. A
// field the cast drops is never read: a NULL stands in, keeping its name and place for the cast's own checks.
Value StructToDeclared(Conversion &conversion, nb::handle object, const TypeTree &node, const LogicalType &whole) {
	auto &scope = conversion.scope;
	std::vector<std::string> names;
	std::vector<nb::object> parts;
	ForEachField(object, [&](const std::string &name, nb::handle part) {
		names.push_back(name);
		parts.push_back(nb::borrow(part));
	});
	const auto pairing = CastPairing(names, node.names);
	std::vector<std::pair<std::string, Value>> fields;
	for (std::size_t i = 0; i < parts.size(); i++) {
		if (pairing[i]) {
			fields.emplace_back(names[i], ToDeclared(conversion, parts[i], node.parts[*pairing[i]], whole));
		} else {
			fields.emplace_back(names[i], Value::CreateNull(scope, scope.CreateType(LogicalTypeId::INTEGER)));
		}
	}
	return Value::CreateStruct(scope, fields).Cast(scope, node.type);
}

Value MapToDeclared(Conversion &conversion, nb::handle object, const TypeTree &node, const LogicalType &whole) {
	std::vector<std::pair<Value, Value>> entries;
	ForEachEntry(object, [&](nb::handle key, nb::handle item) {
		auto fitted_key = ToDeclared(conversion, key, node.parts[0], whole);
		entries.emplace_back(std::move(fitted_key), ToDeclared(conversion, item, node.parts[1], whole));
	});
	return Value::CreateMap(conversion.scope, entries);
}

Value ToDeclared(Conversion &conversion, nb::handle object, const TypeTree &node, const LogicalType &whole) {
	auto &scope = conversion.scope;
	const auto &target = node.type;
	if (object.is_none()) {
		return Value::CreateNull(scope, target);
	}
	const RefusalContext refusal {whole, target};
	// Anything else is read whole and cast as CAST would; a struct is met field by field, so a field it drops is never
	// read, and a union is always checked, since the engine's cast to one leaves NULL where CAST refuses.
	if (!node.converts_per_element) {
		return PythonToValue(conversion.scope, object, conversion.ctx).Cast(scope, target);
	}
	const auto id = node.id;
	const auto form = FormOf(object);
	if ((id == LogicalTypeId::LIST || id == LogicalTypeId::ARRAY) && form == Form::LIST) {
		const auto &element = node.parts[0];
		std::vector<Value> children;
		ForEachElement(object,
		               [&](nb::handle part) { children.push_back(ToDeclared(conversion, part, element, whole)); });
		auto list = children.empty() ? Value::CreateList(scope, element.type) : Value::CreateList(scope, children);
		return id == LogicalTypeId::LIST ? std::move(list) : list.Cast(scope, target);
	}
	if (form == Form::STRUCT && !DictEmpty(object)) {
		if (IsStruct(id)) {
			return StructToDeclared(conversion, object, node, whole);
		}
		// The engine refuses an unnamed struct as a MAP, so that one is left to it.
		if (id == LogicalTypeId::MAP && !OnlyKeyEmpty(object)) {
			return MapToDeclared(conversion, object, node, whole);
		}
	}
	if (id == LogicalTypeId::MAP && form == Form::STRUCT && DictEmpty(object)) {
		return Value::CreateMap(scope, node.parts[0].type, node.parts[1].type);
	}
	if (id == LogicalTypeId::MAP && form == Form::MAP) {
		return MapToDeclared(conversion, object, node, whole);
	}
	if (id == LogicalTypeId::UNION) {
		return UnionToDeclared(conversion, object, refusal);
	}
	if (KindOf(id) != TemporalKind::NONE) {
		return TemporalToDeclared(conversion, PythonToValue(conversion.scope, object, conversion.ctx), refusal, id);
	}
	return PythonToValue(conversion.scope, object, conversion.ctx).Cast(scope, target);
}

} // namespace

TypeTree TypeTreeOf(LogicalType type) {
	const auto id = type.GetTypeId();
	TypeTree node {std::move(type), id, false, {}, {}};
	const auto &described = node.type;
	switch (id) {
	case LogicalTypeId::LIST:
		node.parts.push_back(TypeTreeOf(described.GetListChildType()));
		break;
	case LogicalTypeId::ARRAY:
		node.parts.push_back(TypeTreeOf(described.GetArrayChildType()));
		break;
	case LogicalTypeId::MAP:
		node.parts.push_back(TypeTreeOf(described.GetMapKeyType()));
		node.parts.push_back(TypeTreeOf(described.GetMapValueType()));
		break;
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE:
		node.names = StructNames(described);
		for (duckdb::cxx::idx_t i = 0; i < described.GetStructChildCount(); i++) {
			node.parts.push_back(TypeTreeOf(described.GetStructChildType(i)));
		}
		break;
	default:
		break;
	}
	node.converts_per_element =
	    KindOf(id) != TemporalKind::NONE || IsStruct(id) || id == LogicalTypeId::UNION ||
	    std::any_of(node.parts.begin(), node.parts.end(), [](const auto &part) { return part.converts_per_element; });
	return node;
}

Value PythonToDeclaredType(duckdb::cxx::Context &scope, nb::handle object, const TypeTree &target,
                           ConversionContext &ctx) {
	Conversion conversion {scope, ctx};
	return ToDeclared(conversion, object, target, target.type);
}

} // namespace duckdb_python
