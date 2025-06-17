# cmake/duckdb_loader.cmake
include_guard()      # make the file idempotent
include(CMakeParseArguments)

# ────────────────────────────────────────────
# load_duckdb(<out_target>
#             SUBMODULE_PATH <dir>)
#
# Loads the duckdb source from the given git submodule path.
# * <out_target> is set to the duckdb library name the caller can link against
# * SUBMODULE_PATH <dir> must point to an existing duckdb submodule with its CMakeLists.txt
# ────────────────────────────────────────────
function(load_duckdb_submodule OUT_LIB)
    # parse arguments
    set(options)
    set(oneValueArgs SUBMODULE_PATH)
    cmake_parse_arguments(LDB "${options}" "${oneValueArgs}" "" ${ARGN})
    # load the submodule
    add_subdirectory(${LDB_SUBMODULE_PATH} EXCLUDE_FROM_ALL)
    # create INTERFACE lib to deal with leaking 3rd party headers in duckdb headers
    # See https://github.com/duckdblabs/duckdb-internal/issues/5084
    add_library(duckdb_plus_thirdparty_headers INTERFACE)
    target_include_directories(duckdb_plus_thirdparty_headers INTERFACE
            # include third_party as include dir (we need fastpforlib, concurrentqueue, fsst)
            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party>
            # same for third_party/re2
            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party/re2>
            # same for third_party/fast_float
            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party/fast_float>

            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party/utf8proc/include>
            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party/libpg_query/include>
            $<BUILD_INTERFACE:${LDB_SUBMODULE_PATH}/third_party/fmt/include>
    )
    target_link_libraries(duckdb_plus_thirdparty_headers INTERFACE duckdb_static)
    # return the libraries
    set(${OUT_LIB} duckdb_plus_thirdparty_headers PARENT_SCOPE)
endfunction()

# ────────────────────────────────────────────
# load_duckdb_unity_build(<out_target>
#             UNITY_BUILD_PATH <dir>
#             UNITY_BUILD_INCLUDE_LIST <list>)
#
# Loads the duckdb source from the given unity build path.
# * <out_target> is set to the duckdb library name the caller can link against
# * UNITY_BUILD_PATH <dir> must point to a path containing the extracted duckdb unity build sources
# ────────────────────────────────────────────
function(load_duckdb_unity_build OUT_LIB)
    # parse arguments
    set(options)
    set(multiValueArgs UNITY_BUILD_SOURCES_LIST UNITY_BUILD_INCLUDE_LIST)
    cmake_parse_arguments(LUB "${options}" "" "${multiValueArgs}" ${ARGN})
    # create the duckdb_ub library
    add_library(duckdb_ub STATIC ${LUB_UNITY_BUILD_SOURCES_LIST})
    # add include paths
    target_include_directories(
            duckdb_ub PUBLIC
            ${LUB_UNITY_BUILD_INCLUDE_LIST}
    )
    # return the library
    set(${OUT_LIB} duckdb_ub PARENT_SCOPE)
endfunction()
