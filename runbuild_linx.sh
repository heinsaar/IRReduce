#!/usr/bin/env bash
set -e

# ── defaults ──────────────────────────────────────────────
BUILD_TYPE="Debug"
IR_FILE="../ir/input/hlo_1.ir"        # fallback
EXTRA_ARGS=()

# ── parse all CLI args ────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -debug)   BUILD_TYPE="Debug"   ; shift ;;
    -release) BUILD_TYPE="Release" ; shift ;;
    --input_file=*)                #  --input_file=path
      IR_FILE="${1#*=}"
      shift ;;
    --input_file)                  #  --input_file path
      IR_FILE="$2"
      shift 2 ;;
    *)                               # everything else is passed through
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

echo "Building with configuration: $BUILD_TYPE"
echo "IR file: $IR_FILE"

# ── configure & build ─────────────────────────────────────
mkdir -p build
cd build

if [[ ! -f Makefile ]]; then
  cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
fi
cmake --build . --config "$BUILD_TYPE"

# ── run IRReduce ──────────────────────────────────────────
./IRReduce --input_file "$IR_FILE" "${EXTRA_ARGS[@]}"