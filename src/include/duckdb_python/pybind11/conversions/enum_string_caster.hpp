#pragma once

#include <nanobind/nanobind.h>
#include <cstdint>
#include <string>

//===----------------------------------------------------------------------===//
// Reusable nanobind type_caster macros for "string / integer or enum" arguments
//===----------------------------------------------------------------------===//
//
// Several DuckDB enums are exposed to Python so that a binding parameter typed as
// the enum accepts a string (and, for most, an integer) naming one of its values.
// These enums are NOT registered as Python types (no nb::enum_), so the caster only
// needs the str/int -> enum direction; there is no registered-instance to delegate to.
//
// The macros collapse the boilerplate into one invocation per enum, so the caster
// rewrite is a single-place change. nanobind requires from_python()/from_cpp() to be
// noexcept, so the DuckDB *FromString/*FromInteger calls (which throw on bad input)
// are wrapped — a bad value reports a generic conversion failure rather than the
// original InvalidInputException message (acceptable; refine post-cutover if needed).
//
// Invoke at GLOBAL scope (outside any namespace); each expands to a full
// `namespace nanobind { namespace detail { ... } }` specialization. Pass fully
// qualified names for the conversion functions and the enum type.

//! str + int + enum form.
#define DUCKDB_PY_ENUM_STRING_INT_CASTER(EnumType, FromStringFn, FromIntegerFn, NameLiteral)                           \
	namespace nanobind {                                                                                               \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		NB_TYPE_CASTER(EnumType, const_name(NameLiteral))                                                              \
		bool from_python(handle src, uint8_t, cleanup_list *) noexcept {                                               \
			try {                                                                                                      \
				if (nanobind::isinstance<nanobind::str>(src)) {                                                        \
					value = FromStringFn(nanobind::cast<std::string>(src));                                            \
					return true;                                                                                       \
				}                                                                                                      \
				if (nanobind::isinstance<nanobind::int_>(src)) {                                                       \
					value = FromIntegerFn(nanobind::cast<int64_t>(src));                                               \
					return true;                                                                                       \
				}                                                                                                      \
				/* Registered nb::enum_ instances aren't int subclasses (unlike pybind11's), so accept a member  */    \
				/* of the registered enum by reading its integer .value.                                         */    \
				nanobind::handle enum_type = nanobind::type<EnumType>();                                               \
				if (enum_type.is_valid() && PyObject_IsInstance(src.ptr(), enum_type.ptr()) == 1) {                    \
					value = FromIntegerFn(nanobind::cast<int64_t>(src.attr("value")));                                 \
					return true;                                                                                       \
				}                                                                                                      \
			} catch (...) {                                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			return false;                                                                                              \
		}                                                                                                              \
		static handle from_cpp(EnumType src, rv_policy, cleanup_list *) noexcept {                                     \
			return nanobind::int_((int64_t)src).release();                                                             \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace nanobind */

//! str + enum form (no integer accepted).
#define DUCKDB_PY_ENUM_STRING_CASTER(EnumType, FromStringFn, NameLiteral)                                              \
	namespace nanobind {                                                                                               \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		NB_TYPE_CASTER(EnumType, const_name(NameLiteral))                                                              \
		bool from_python(handle src, uint8_t, cleanup_list *) noexcept {                                               \
			try {                                                                                                      \
				if (nanobind::isinstance<nanobind::str>(src)) {                                                        \
					value = FromStringFn(nanobind::cast<std::string>(src));                                            \
					return true;                                                                                       \
				}                                                                                                      \
				/* Registered nb::enum_ instances aren't int subclasses; accept a member of the registered enum  */    \
				/* by reading its integer .value (this enum has no FromInteger, so cast the int directly).        */   \
				nanobind::handle enum_type = nanobind::type<EnumType>();                                               \
				if (enum_type.is_valid() && PyObject_IsInstance(src.ptr(), enum_type.ptr()) == 1) {                    \
					value = (EnumType)nanobind::cast<int64_t>(src.attr("value"));                                      \
					return true;                                                                                       \
				}                                                                                                      \
			} catch (...) {                                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			return false;                                                                                              \
		}                                                                                                              \
		static handle from_cpp(EnumType src, rv_policy, cleanup_list *) noexcept {                                     \
			return nanobind::int_((int64_t)src).release();                                                             \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace nanobind */
