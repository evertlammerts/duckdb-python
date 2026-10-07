//===----------------------------------------------------------------------===//
//                         DuckDB
//
// src/_duckdb/conversion/conversion.cpp
//
//
//===----------------------------------------------------------------------===//

#include "conversion.hpp"

#include <nanobind/stl/string.h>

#include <string>

namespace duckdb_python {

ConversionContext::ConversionContext() {
	nb::object datetime = nb::module_::import_("datetime");
	date_cls = datetime.attr("date");
	time_cls = datetime.attr("time");
	datetime_cls = datetime.attr("datetime");
	timedelta_cls = datetime.attr("timedelta");
	timezone_cls = datetime.attr("timezone");
	timezone_utc = timezone_cls.attr("utc");
	nb::object decimal = nb::module_::import_("decimal");
	decimal_cls = decimal.attr("Decimal");
	decimal_context = decimal.attr("Context")(nb::arg("prec") = 45);
	uuid_cls = nb::module_::import_("uuid").attr("UUID");
	int_cls = nb::module_::import_("builtins").attr("int");
	mapping_cls = nb::module_::import_("collections.abc").attr("Mapping");
	two_pow_64 = int_cls("18446744073709551616");
	epoch_date = date_cls(1970, 1, 1);
	epoch_naive = datetime_cls(1970, 1, 1);
	epoch_aware = datetime_cls(1970, 1, 1, 0, 0, 0, 0, timezone_utc);
	one_microsecond = timedelta_cls(0, 0, 1);
}

std::string DescribePythonError(nb::python_error &error) {
	std::string summary;
	try {
		summary = nb::cast<std::string>(nb::handle(error.type()).attr("__name__"));
		const auto text = nb::cast<std::string>(nb::str(nb::handle(error.value())));
		if (!text.empty()) {
			summary += ": " + text;
		}
	} catch (...) {
		summary.clear();
	}
	return summary + "\n" + error.what();
}

UnsupportedTypeException::UnsupportedTypeException(std::string type_name)
    : duckdb::cxx::InvalidInputException("Invalid Input Error: cannot bind a parameter of type " + type_name,
                                         "cannot bind a parameter of type " + type_name),
      type_name(std::move(type_name)) {
}

} // namespace duckdb_python
