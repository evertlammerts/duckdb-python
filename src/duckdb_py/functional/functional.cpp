#include "duckdb_python/functional.hpp"

namespace duckdb {

void DuckDBPyFunctional::Initialize(py::module_ &parent) {
	auto m = parent.def_submodule("functional", "DuckDB Python UDF types.");

	py::enum_<duckdb::PythonUDFType>(m, "PythonUDFType",
	                                 "Enumeration for Python User-Defined Function (UDF) execution types.\n\n"
	                                 "Specifies the data format and execution strategy used when calling\n"
	                                 "Python functions from within DuckDB queries. Different types offer\n"
	                                 "trade-offs between performance, memory usage, and compatibility.")
	    .value("NATIVE", duckdb::PythonUDFType::NATIVE,
	           "Native Python execution using standard Python objects.\n"
	           "Data is converted to/from standard Python types (lists, scalars, etc.)\n"
	           "which provides maximum compatibility but may have higher conversion overhead.")
	    .value("ARROW", duckdb::PythonUDFType::ARROW,
	           "Apache Arrow-based execution using columnar data.\n"
	           "Data is passed as Apache Arrow arrays for vectorized operations,\n"
	           "providing better performance for large datasets and numerical computations.")
	    .export_values();

	py::enum_<duckdb::FunctionNullHandling>(m, "FunctionNullHandling",
	                                        "Enumeration for function NULL value handling strategies.\n\n"
	                                        "Controls how UDFs behave when they encounter NULL input values.")
	    .value("DEFAULT", duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING,
	           "Standard NULL propagation behavior.\n"
	           "Functions automatically return NULL when any input argument is NULL,\n"
	           "following SQL standard semantics without executing the function body.")
	    .value("SPECIAL", duckdb::FunctionNullHandling::SPECIAL_HANDLING,
	           "Custom NULL handling within the function.\n"
	           "Functions receive NULL values as input and implement their own logic\n"
	           "for handling NULLs, allowing for specialized behavior.")
	    .export_values();
}

} // namespace duckdb
