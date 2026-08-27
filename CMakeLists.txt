cmake_minimum_required(VERSION 3.20...3.29)
project(duckdb_python LANGUAGES CXX)

# DuckDB C++ API. DuckDBCppApi.cmake ships with DuckDB and defines
# duckdb::duckdb (the engine, imported) and duckdb::cpp_api (compiled into us).
# It resolves the engine from exactly one of DUCKDB_ROOT, DUCKDB_VERSION,
# DUCKDB_PROVIDER=system, or a download of the pinned build.
# DUCKDB_CPP_DIR holds duckdb_cpp.cpp and its headers.
if(NOT DUCKDB_CPP_DIR)
  message(FATAL_ERROR "Set -DDUCKDB_CPP_DIR=<duckdb>/tools/cpp or an engine bundle")
endif()
set(DUCKDB_CPP_DEFAULT_DUCKDB_VERSION "1.5.5")
include("${DUCKDB_CPP_DIR}/cmake/DuckDBCppApi.cmake")

# Escape hatch for building against a raw DuckDB checkout, where
# duckdb_extension_v2.h sits in src/include rather than beside duckdb_cpp.cpp.
# An engine bundle carries both, so it does not need this.
if(DUCKDB_INCLUDE_DIR)
  target_include_directories(duckdb_cpp_api PUBLIC "${DUCKDB_INCLUDE_DIR}")
endif()

# Development.SABIModule is what makes STABLE_ABI take effect. Without it the
# build still succeeds and the wheel is still tagged abi3, but the extension
# inside carries a version-specific suffix.
find_package(Python 3.12 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# nanobind applies STABLE_ABI only on non-free-threaded CPython 3.12 and up,
# and FREE_THREADED only on free-threaded interpreters, so one call covers the
# whole wheel matrix.
nanobind_add_module(_duckdb STABLE_ABI FREE_THREADED NB_DOMAIN duckdb
  src/_duckdb/module.cpp)
target_link_libraries(_duckdb PRIVATE duckdb::cpp_api)

# The engine ships beside the extension inside the wheel, linked not dlopened.
if(APPLE)
  set_target_properties(_duckdb PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_duckdb PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS _duckdb LIBRARY DESTINATION duckdb)

# Copy the engine into the package. On Windows DUCKDB_LIBRARY is the import
# library, which is a link-time artifact; the wheel needs the DLL beside the
# extension instead. Everywhere else the two are the same file.
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
