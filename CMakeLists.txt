cmake_minimum_required(VERSION 3.29)

project(duckdb_py LANGUAGES CXX)

# Always use C++11
set(CMAKE_CXX_STANDARD 11)

# Set the library name
set(DUCKDB_PYTHON_LIB_NAME "_duckdb")

# Detect CCache
include(cmake/compiler_launcher.cmake)
setup_compiler_launcher_if_available()

# ────────────────────────────────────────────
# IDE support
# ────────────────────────────────────────────
# Create compile_commands.json for IntelliSense and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# If we're not building through scikit-build-core then we have to set a different dest dir
include(GNUInstallDirs)
set(_DUCKDB_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
if(NOT _DUCKDB_PY_INSTALL_DIR)
  set(_DUCKDB_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

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
# Load duckdb loader module
include(cmake/duckdb_loader.cmake)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  duckdb_configure_for_debug()
else()
  duckdb_configure_for_release()
endif()
duckdb_add_library(duckdb_target)

# ────────────────────────────────────────────
# INTERFACE library that **only** carries dependencies
# ────────────────────────────────────────────
add_library(_duckdb_dependencies INTERFACE)

# All the imported dependency targets: they bring their own include dirs and compile definitions automatically.
target_link_libraries(_duckdb_dependencies INTERFACE
        Python3::Python # defined by FindPython3
        pybind11::pybind11 # defined by pybind11Config.cmake
        duckdb_target
)
target_include_directories(_duckdb_dependencies INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/duckdb_py/include>
)

# ────────────────────────────────────────────
# Descend into the real DuckDB‑Python sources
# ────────────────────────────────────────────
add_subdirectory(src/duckdb_py)

pybind11_add_module(_duckdb
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
# add _duckdb_dependencies
target_link_libraries(_duckdb PRIVATE _duckdb_dependencies)

# ────────────────────────────────────────────
# Put the object file in the correct place
# ────────────────────────────────────────────
install(TARGETS _duckdb LIBRARY DESTINATION "${_DUCKDB_PY_INSTALL_DIR}")
