cmake_minimum_required(VERSION 3.22...3.29)
project(duckdb_py LANGUAGES CXX)

# Create compile_commands.json for IntelliSense and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# C++ 11
set(CMAKE_CXX_STANDARD 11)

# ────────────────────────────────────────────
# Policy hygiene
# ────────────────────────────────────────────
if(POLICY CMP0148) # Disallow FindPythonLibs
  cmake_policy(SET CMP0148 NEW)
endif()

if(POLICY CMP0003) # No implicit link directories
  cmake_policy(SET CMP0003 NEW)
endif()

# ────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "CPython executable: ${Python3_EXECUTABLE}")
message(STATUS "CPython include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "CPython libraries: ${Python3_LIBRARIES}")
find_package(pybind11 REQUIRED CONFIG)

# ────────────────────────────────────────────
# Include DuckDB source
# ────────────────────────────────────────────
set(DUCKDB_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(DUCKDB_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(DUCKDB_UNITY_BUILD      OFF CACHE BOOL "" FORCE)
add_subdirectory(external/duckdb EXCLUDE_FROM_ALL)

# ────────────────────────────────────────────
# Temp create INTERFACE lib to deal with leaking
# 3rd party headers in duckdb headers
# See https://github.com/duckdblabs/duckdb-internal/issues/5084
# ────────────────────────────────────────────
add_library(duckdb_thirdparty_headers INTERFACE)

target_include_directories(duckdb_thirdparty_headers INTERFACE
        # include third_party as include dir (we need fastpforlib, concurrentqueue, fsst)
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party>
        # same for third_party/re2
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party/re2>
        # same for third_party/fast_float
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party/fast_float>

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party/utf8proc/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party/libpg_query/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb/third_party/fmt/include>
)

# ────────────────────────────────────────────
# INTERFACE library that **only** carries usage requirements
# ────────────────────────────────────────────
add_library(pytooling INTERFACE)

# 1. All the imported dependency targets: they bring their own include dirs +
#   compile definitions automatically.
target_link_libraries(pytooling INTERFACE
        Python3::Python # defined by FindPython3
        pybind11::pybind11 # defined by pybind11Config.cmake
        duckdb_thirdparty_headers
)
target_include_directories(pytooling INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
        # headers exported by duckdb_static
        $<TARGET_PROPERTY:duckdb_static,INTERFACE_INCLUDE_DIRECTORIES>
)

# ────────────────────────────────────────────
# Descend into the real DuckDB‑Python sources
# ────────────────────────────────────────────
add_subdirectory(src)

pybind11_add_module(_duckdb_py
        duckdb_python.cpp
        $<TARGET_OBJECTS:python_src>
        $<TARGET_OBJECTS:python_arrow>
        $<TARGET_OBJECTS:python_common>
        $<TARGET_OBJECTS:python_functional>
        $<TARGET_OBJECTS:python_jupyter>
        $<TARGET_OBJECTS:python_native>
        $<TARGET_OBJECTS:python_numpy>
        $<TARGET_OBJECTS:python_pandas>
        $<TARGET_OBJECTS:python_pybind11>
        $<TARGET_OBJECTS:python_connection>
        $<TARGET_OBJECTS:python_expression>
        $<TARGET_OBJECTS:python_relation>
        $<TARGET_OBJECTS:python_type>
)
target_link_libraries(_duckdb_py PRIVATE pytooling)
