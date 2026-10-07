cmake_minimum_required(VERSION 3.20...3.29)
project(duckdb_python LANGUAGES CXX)

# The engine floor: the oldest libduckdb this client runs on. The C++ API pin in third_party/duckdb_cpp is
# its lower bound, so raising one means raising the other; see third_party/duckdb_cpp/REF.
set(DUCKDB_FLOOR_MAJOR 2)
set(DUCKDB_FLOOR_MINOR 0)
set(DUCKDB_FLOOR_PATCH 0)

# A -D lands in the cache and would silently outlive the configure that passed it; read it into a normal
# variable and clear the entry, so each configure decides from its own -D, environment, or the defaults.
foreach(_knob DUCKDB_CPP_DIR DUCKDB_ROOT)
  if(DEFINED CACHE{${_knob}})
    set(${_knob} "$CACHE{${_knob}}")
    unset(${_knob} CACHE)
  endif()
endforeach()

if(NOT DUCKDB_CPP_DIR)
  set(DUCKDB_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/duckdb_cpp")
endif()
# The prebuilt engine: its shared library and C headers side by side, as scripts/engine.py fetch lays them out.
if(NOT DUCKDB_ROOT)
  if(DEFINED ENV{DUCKDB_ROOT})
    set(DUCKDB_ROOT "$ENV{DUCKDB_ROOT}")
  else()
    set(DUCKDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/engine")
  endif()
endif()
get_filename_component(DUCKDB_ROOT "${DUCKDB_ROOT}" ABSOLUTE)
if(NOT EXISTS "${DUCKDB_ROOT}/duckdb_v2.h")
  message(FATAL_ERROR "no engine at DUCKDB_ROOT=${DUCKDB_ROOT}: run `uv run scripts/engine.py fetch`, "
                      "or point DUCKDB_ROOT at a directory holding libduckdb and its C headers side by side")
endif()

set(DUCKDB_CPP_DEFAULT_DUCKDB_VERSION "${DUCKDB_FLOOR_MAJOR}.${DUCKDB_FLOOR_MINOR}.${DUCKDB_FLOOR_PATCH}")
include("${DUCKDB_CPP_DIR}/cmake/DuckDBCppApi.cmake")

# DuckDBCppApi.cmake builds a static library without -fPIC, which cannot then link into a shared one on Linux.
set_property(TARGET duckdb_cpp_api PROPERTY POSITION_INDEPENDENT_CODE ON)

# The engine's own C headers, ahead of any copy beside the C++ API: an engine that changed a C signature must
# fail this build instead of corrupting memory at run time.
target_include_directories(duckdb_cpp_api BEFORE PUBLIC "${DUCKDB_ROOT}")
# Pinned to the floor, like Py_LIMITED_API: code cannot reach C API the oldest supported engine lacks.
target_compile_definitions(duckdb_cpp_api PUBLIC
  DUCKDB_V2_API_VERSION_MAJOR=${DUCKDB_FLOOR_MAJOR}
  DUCKDB_V2_API_VERSION_MINOR=${DUCKDB_FLOOR_MINOR}
  DUCKDB_V2_API_VERSION_PATCH=${DUCKDB_FLOOR_PATCH})

# Without Development.SABIModule, STABLE_ABI silently does nothing: the wheel is tagged abi3 but is not.
find_package(Python 3.11 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# nanobind ignores STABLE_ABI below 3.12 and FREE_THREADED off that build, so one call covers every wheel.
nanobind_add_module(_duckdb STABLE_ABI FREE_THREADED NB_DOMAIN duckdb
  src/_duckdb/chunkview.cpp
  src/_duckdb/module.cpp
  src/_duckdb/conversion/conversion.cpp
  src/_duckdb/conversion/sql_types.cpp
  src/_duckdb/conversion/temporal.cpp
  src/_duckdb/conversion/untyped.cpp
  src/_duckdb/conversion/python_to_value.cpp
  src/_duckdb/conversion/python_to_declared_type.cpp
  src/_duckdb/conversion/value_to_python.cpp
  src/_duckdb/registry.cpp
  src/_duckdb/arrow_scan.cpp
  src/_duckdb/numpy_scan.cpp
  src/_duckdb/arrowc.cpp
  src/_duckdb/predicate.cpp
  src/_duckdb/result.cpp
  src/_duckdb/udf.cpp)
target_link_libraries(_duckdb PRIVATE duckdb::cpp_api)

# The DuckDB library ships beside the extension in the wheel and is linked, not loaded at run time.
if(APPLE)
  set_target_properties(_duckdb PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_duckdb PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _duckdb LIBRARY DESTINATION duckdb)

# On Windows DUCKDB_LIBRARY is the link-time import library, so the wheel needs the DLL beside it instead.
get_filename_component(_duckdb_lib_dir "${DUCKDB_LIBRARY}" DIRECTORY)
if(WIN32)
  file(GLOB _duckdb_runtime "${_duckdb_lib_dir}/duckdb.dll")
  if(NOT _duckdb_runtime)
    message(FATAL_ERROR "no duckdb.dll beside ${DUCKDB_LIBRARY}; the engine directory must carry both")
  endif()
else()
  get_filename_component(_duckdb_runtime "${DUCKDB_LIBRARY}" REALPATH)
endif()
get_filename_component(_duckdb_runtime_name "${_duckdb_runtime}" NAME)
install(FILES "${_duckdb_runtime}" DESTINATION duckdb RENAME "${_duckdb_runtime_name}")
