//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/python_to_value.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <utility>
#include <vector>

#include "conversion.hpp"
#include "duckdb_cpp.hpp"

namespace duckdb_python {

/// One Python object as a DuckDB value of the type the object itself implies; `scope` is a Connection, or the
/// Context inside a function callback. When `contains_untyped` is given, it is set to whether the value still
/// holds an untyped place (see untyped.hpp).
template <class SCOPE>
duckdb::cxx::Value PythonToValue(SCOPE &scope, nb::handle object, ConversionContext &ctx,
                                 TimestampPrecision precision = TimestampPrecision::OWN_UNIT,
                                 bool *contains_untyped = nullptr);

extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Connection>(duckdb::cxx::Connection &, nb::handle,
                                                                          ConversionContext &, TimestampPrecision,
                                                                          bool *);
extern template duckdb::cxx::Value PythonToValue<duckdb::cxx::Context>(duckdb::cxx::Context &, nb::handle,
                                                                       ConversionContext &, TimestampPrecision, bool *);

/// How a Python object is read: one value, a list's elements, a dict's named fields, or a dict's entries.
enum class Form : uint8_t { SCALAR, LIST, STRUCT, MAP };

inline Form FormOf(nb::handle object) {
	if (nb::isinstance<nb::list>(object) || nb::isinstance<nb::tuple>(object)) {
		return Form::LIST;
	}
	if (!nb::isinstance<nb::dict>(object)) {
		return Form::SCALAR;
	}
	// A dict maps onto two DuckDB types, so the rule is fixed: string keys make a STRUCT, anything else a MAP.
	for (auto entry : nb::borrow<nb::dict>(object)) {
		if (!nb::isinstance<nb::str>(entry.first)) {
			return Form::MAP;
		}
	}
	return Form::STRUCT;
}

// A list's elements and a dict's entries are all held before any is converted: converting one may run Python code, such
// as a tzinfo's, that changes the list or dict, which would cut the walk short or free a part only it held.
template <class EACH>
void ForEachElement(nb::handle object, const EACH &each) {
	std::vector<nb::object> elements;
	for (nb::handle item : object) {
		elements.push_back(nb::borrow(item));
	}
	for (auto &element : elements) {
		each(element);
	}
}

template <class EACH>
void ForEachField(nb::handle object, const EACH &each) {
	std::vector<std::pair<std::string, nb::object>> fields;
	for (auto entry : nb::borrow<nb::dict>(object)) {
		fields.emplace_back(nb::cast<std::string>(entry.first), nb::borrow(entry.second));
	}
	for (auto &[name, part] : fields) {
		each(name, part);
	}
}

template <class EACH>
void ForEachEntry(nb::handle object, const EACH &each) {
	std::vector<std::pair<nb::object, nb::object>> entries;
	for (auto entry : nb::borrow<nb::dict>(object)) {
		entries.emplace_back(nb::borrow(entry.first), nb::borrow(entry.second));
	}
	for (auto &[key, item] : entries) {
		each(key, item);
	}
}

/// Whether a dict holds no entry, as `ForEachField` reads its entries, since a subclass's `__len__` may say
/// otherwise.
inline bool DictEmpty(nb::handle dict) {
	return PyDict_Size(dict.ptr()) == 0;
}

/// Whether a dict's one key is the empty string, which makes the struct it becomes unnamed, a TUPLE.
inline bool OnlyKeyEmpty(nb::handle dict) {
	if (PyDict_Size(dict.ptr()) != 1) {
		return false;
	}
	for (auto entry : nb::borrow<nb::dict>(dict)) {
		return PyUnicode_GetLength(entry.first.ptr()) == 0;
	}
	return false;
}

} // namespace duckdb_python
