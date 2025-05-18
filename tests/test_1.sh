#!/usr/bin/env bash
set -e

# ── defaults ──────────────────────────────────────────────
BUILD_TYPE="Debug"
IR_FILE="./ir/input/hlo_1.ir" # fallback / default
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
    *)                             # everything else is passed through
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

echo "Running test with configuration: $BUILD_TYPE"
echo "IR file: $IR_FILE"

# ── assume build exists in ./build ────────────────────────
EXECUTABLE="./build/IRReduce"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: Executable not found: $EXECUTABLE"
  echo "Make sure the project is built beforehand."
  exit 1
fi

# ── run IRReduce ──────────────────────────────────────────
"$EXECUTABLE" --input_file "$IR_FILE" "${EXTRA_ARGS[@]}"
