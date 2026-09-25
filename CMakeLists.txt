cmake_minimum_required(VERSION 3.20...3.29)
project(duckdb_python LANGUAGES CXX)

# DuckDBCppApi.cmake ships with DuckDB and finds the library via DUCKDB_ROOT, DUCKDB_VERSION or a download.
if(NOT DUCKDB_CPP_DIR)
  message(FATAL_ERROR "Set -DDUCKDB_CPP_DIR=<duckdb>/tools/cpp or an engine bundle")
endif()
set(DUCKDB_CPP_DEFAULT_DUCKDB_VERSION "1.5.5")
include("${DUCKDB_CPP_DIR}/cmake/DuckDBCppApi.cmake")

# DuckDBCppApi.cmake builds a static library without -fPIC, which cannot then link into a shared one on Linux.
set_property(TARGET duckdb_cpp_api PROPERTY POSITION_INDEPENDENT_CODE ON)

# For building against a plain DuckDB checkout, where duckdb_extension_v2.h sits in src/include instead.
if(DUCKDB_INCLUDE_DIR)
  target_include_directories(duckdb_cpp_api PUBLIC "${DUCKDB_INCLUDE_DIR}")
endif()

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
  src/_duckdb/pyconv.cpp
  src/_duckdb/registry.cpp
  src/_duckdb/arrow_scan.cpp
  src/_duckdb/pandas_scan.cpp
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
    message(FATAL_ERROR "no duckdb.dll beside ${DUCKDB_LIBRARY}; the bundle must carry both")
  endif()
else()
  get_filename_component(_duckdb_runtime "${DUCKDB_LIBRARY}" REALPATH)
endif()
get_filename_component(_duckdb_runtime_name "${_duckdb_runtime}" NAME)
install(FILES "${_duckdb_runtime}" DESTINATION duckdb RENAME "${_duckdb_runtime_name}")
