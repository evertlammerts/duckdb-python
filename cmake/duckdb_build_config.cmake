# duckdb_build_config.cmake — centralize all DuckDB compile & link flags and macros
include_guard(GLOBAL) # only include once

# CMake defines that can be used for this module
set(DUCKDB_ENABLED_EXTENSIONS "" CACHE STRING "List of all extensions to enable.")
set(DUCKDB_CUSTOM_PLATFORM "" CACHE STRING "Name of the platform to build duckdb for.")
option(DUCKDB_DISABLE_JEMALLOC "Disable jemalloc. By default jemalloc will only be enabled on Linux." OFF)


# ────────────────────────────────────────────
# duckdb_build_config target
#
# Create a single interface target `duckdb_build_config`
# that holds all of duckdb's build config.
# ────────────────────────────────────────────
add_library(duckdb_build_config INTERFACE)

# Always: C++11
target_compile_features(duckdb_build_config INTERFACE cxx_std_11)

# Always build everything with PIC (so static libs can be linked into shared modules without error)
set_target_properties(duckdb_build_config PROPERTIES
        INTERFACE_POSITION_INDEPENDENT_CODE ON
)

# Always‐on flags
target_compile_definitions(duckdb_build_config INTERFACE
        DUCKDB_PYTHON_LIB_NAME=\"duckdb\"
        DUCKDB_EXTENSION_AUTOLOAD_DEFAULT=1
        DUCKDB_EXTENSION_AUTOINSTALL_DEFAULT=1
)

# Enable configured extensions
if(DUCKDB_ENABLED_EXTENSIONS)
    foreach(ext ${DUCKDB_ENABLED_EXTENSIONS})
        string(TOUPPER "${ext}" ext_upper)
        message(STATUS "Enabling DuckDB extension ${ext_upper}")
        # if we need to build httpfs then we require openssl
        if(${ext_upper} STREQUAL "HTTPFS")
            find_package(OpenSSL REQUIRED)
            target_link_libraries(duckdb_build_config INTERFACE
                    OpenSSL::Crypto OpenSSL::SSL
            )
        endif ()
        target_compile_definitions(duckdb_build_config INTERFACE
            DUCKDB_EXTENSION_${ext_upper}_LINKED
        )
    endforeach()
endif()

# Platform‐specific compile + link flags
if(WIN32)
    target_compile_options(duckdb_build_config INTERFACE
        /wd4244 /wd4267 /wd4200 /wd26451 /wd26495
        /D_CRT_SECURE_NO_WARNINGS /utf-8
    )
    target_compile_definitions(duckdb_build_config INTERFACE
            DUCKDB_BUILD_LIBRARY WIN32
    )
    target_link_libraries(duckdb_build_config INTERFACE
            rstrtmgr.lib bcrypt.lib
    )
elseif(APPLE OR UNIX)
    # Add warnings if in Debug (CMake’s build-type "debug" handles -g / -O)
    target_compile_options(duckdb_build_config INTERFACE
            $<$<CONFIG:Debug>:-Wall>
    )
    # macOS needs libc++ and min version
    if(APPLE)
        target_compile_options(duckdb_build_config INTERFACE
                -stdlib=libc++ -mmacosx-version-min=10.7
        )
    endif()
    if(NOT APPLE AND NOT DUCKDB_DISABLE_JEMALLOC)
        # Enable jemalloc if on Linux and not disabled in the build config
        message(STATUS "Enabling jemalloc")
        target_compile_definitions(duckdb_build_config INTERFACE
                DUCKDB_EXTENSION_JEMALLOC_LINKED
        )
    endif()
endif()

# Custom platform override
if(DUCKDB_CUSTOM_PLATFORM)
    message(STATUS "Setting custom platform name: ${DUCKDB_CUSTOM_PLATFORM}")
    target_compile_definitions(duckdb_build_config INTERFACE
            DUCKDB_CUSTOM_PLATFORM=\"${DUCKDB_CUSTOM_PLATFORM}\"
    )
endif()
