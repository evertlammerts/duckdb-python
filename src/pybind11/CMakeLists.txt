# this is used for clang-tidy checks
add_library(python_pybind11 OBJECT pyutil.cpp)

target_link_libraries(python_pybind11 PRIVATE _duckdb_dependencies)
