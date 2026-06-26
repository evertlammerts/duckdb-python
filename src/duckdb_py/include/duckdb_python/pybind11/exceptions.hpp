#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace py = nanobind;

namespace duckdb {

void RegisterExceptions(const py::module &m);

} // namespace duckdb
