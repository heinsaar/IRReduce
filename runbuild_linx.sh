#!/usr/bin/env bash
set -e

# ── defaults ──────────────────────────────────
BUILD_TYPE="Debug"
IR_FILE="../ir/hlo_1.ir"
EXTRA_ARGS=()

# ── parse args ────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    -debug)   BUILD_TYPE="Debug"   ;;
    -release) BUILD_TYPE="Release" ;;
    *)
      if [[ -z "$IR_SPECIFIED" ]]; then
        IR_FILE="$arg"
        IR_SPECIFIED=1
      else
        EXTRA_ARGS+=("$arg")
      fi
      ;;
  esac
done

echo "Building with configuration: $BUILD_TYPE"
echo "IR file: $IR_FILE"

# ── configure & build ─────────────────────────
mkdir -p build
cd build

if [[ ! -f Makefile ]]; then
  cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
fi
cmake --build . --config "$BUILD_TYPE"

# ── run ───────────────────────────────────────
./IRReduce --input_file "$IR_FILE" "${EXTRA_ARGS[@]}"