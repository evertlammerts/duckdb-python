cmake_minimum_required(VERSION 3.29)

project(duckdb_py LANGUAGES CXX)

# CMake defines
set(DUCKDB_UNITY_BUILD_SOURCES_LIST "" CACHE STRING "List of all sources that should be included in an sdist. Will be set automatically by the build backend when building a wheel from an sdist.")
set(DUCKDB_UNITY_BUILD_INCLUDE_LIST "" CACHE STRING "List of include dirs to include in the DuckDB unity build. Will be set automatically by the build backend when building a wheel from an sdist.")

# Always use C++11
set(CMAKE_CXX_STANDARD 11)

# Load build config module - this creates `duckdb_build_config`
include(cmake/duckdb_build_config.cmake)

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
include(cmake/duckdb_loader.cmake)
if(DUCKDB_UNITY_BUILD_SOURCES_LIST STREQUAL "")
  load_duckdb_submodule(DUCKDB_STATIC_TARGET
          SUBMODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb"
  )
else()
  load_duckdb_unity_build(DUCKDB_STATIC_TARGET
          UNITY_BUILD_SOURCES_LIST "${DUCKDB_UNITY_BUILD_SOURCES_LIST}"
          UNITY_BUILD_INCLUDE_LIST "${DUCKDB_UNITY_BUILD_INCLUDE_LIST}"
  )
endif()

# ────────────────────────────────────────────
# INTERFACE library that **only** carries usage requirements
# ────────────────────────────────────────────
add_library(pytooling INTERFACE)

# All the imported dependency targets: they bring their own include dirs and compile definitions automatically.
target_link_libraries(pytooling INTERFACE
        Python3::Python # defined by FindPython3
        pybind11::pybind11 # defined by pybind11Config.cmake
)
target_include_directories(pytooling INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/duckdb_py/include>
        # headers exported by duckdb_static
        $<TARGET_PROPERTY:${DUCKDB_STATIC_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
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
# add pytooling, duckdb build config (for consistent compile options) and duckdb itself
target_link_libraries(_duckdb PRIVATE pytooling duckdb_build_config ${DUCKDB_STATIC_TARGET})

# ────────────────────────────────────────────
# Put the object file in the correct place
# ────────────────────────────────────────────
install(TARGETS _duckdb LIBRARY DESTINATION "${_DUCKDB_PY_INSTALL_DIR}")
