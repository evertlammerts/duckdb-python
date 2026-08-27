#!/usr/bin/env bash
# Assemble a self-contained DuckDB engine bundle: the shared library plus the
# C++ API and the two v2 headers it needs. Consumed by pointing both
# DUCKDB_ROOT and DUCKDB_CPP_DIR at the bundle root.
#
# This reconstructs what tools/cpp/example/README.md calls
# scripts/package_cpp_api.py, which did not land in duckdb main. Delete this
# script once it does.
#
#   build_engine_bundle.sh <duckdb-checkout> <output-dir> [build-type]
set -euo pipefail

SRC="${1:?usage: build_engine_bundle.sh <duckdb-checkout> <output-dir> [build-type]}"
OUT="${2:?usage: build_engine_bundle.sh <duckdb-checkout> <output-dir> [build-type]}"
BUILD_TYPE="${3:-release}"

[ -f "$SRC/tools/cpp/duckdb_cpp.cpp" ] || { echo "no tools/cpp in $SRC" >&2; exit 1; }

# Build libduckdb if the checkout has no usable one yet.
LIB=$(find "$SRC/build/$BUILD_TYPE" -name 'libduckdb.dylib' -o -name 'libduckdb.so' -o -name 'duckdb.dll' 2>/dev/null | head -1 || true)
if [ -z "$LIB" ]; then
  echo "building libduckdb ($BUILD_TYPE) in $SRC"
  make -C "$SRC" "$BUILD_TYPE" -j"$(getconf _NPROCESSORS_ONLN)"
  LIB=$(find "$SRC/build/$BUILD_TYPE" -name 'libduckdb.dylib' -o -name 'libduckdb.so' -o -name 'duckdb.dll' | head -1)
fi
[ -n "$LIB" ] || { echo "no libduckdb produced" >&2; exit 1; }

# Refuse a pre-v2 library here rather than at the consumer's link probe.
if command -v nm >/dev/null && [ "$(nm -g "$LIB" 2>/dev/null | grep -c duckdb_v2_ || true)" = "0" ]; then
  echo "ERROR: $LIB exports no duckdb_v2_* symbols (pre-V2 engine)" >&2
  exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/cmake"
cp "$SRC/tools/cpp/duckdb_cpp.hpp" "$SRC/tools/cpp/duckdb_cpp.cpp" "$OUT/"
cp "$SRC/src/include/duckdb_v2.h" "$SRC/src/include/duckdb_extension_v2.h" "$OUT/"
cp "$SRC/tools/cpp/cmake/DuckDBCppApi.cmake" "$OUT/cmake/"
cp "$(readlink -f "$LIB" 2>/dev/null || python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$LIB")" "$OUT/lib/$(basename "$LIB")"

git -C "$SRC" rev-parse HEAD > "$OUT/ENGINE_SHA" 2>/dev/null || echo unknown > "$OUT/ENGINE_SHA"
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1)), engine $(cat "$OUT/ENGINE_SHA" | cut -c1-12)"
