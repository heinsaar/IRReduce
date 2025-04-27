#!/bin/bash

# Initialize variables
BUILD_TYPE="Debug"  # Default

# Parse named arguments
for arg in "$@"; do
  case $arg in
    -debug)
      BUILD_TYPE="Debug"
      ;;
    -release)
      BUILD_TYPE="Release"
      ;;
    *)
      # Ignore unknown args
      ;;
  esac
done

# Create build directory if needed
mkdir -p build
cd build

# Configure only if first time (optional optimization)
if [ ! -f Makefile ]; then
  cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
fi

# Build
if cmake --build . --config $BUILD_TYPE; then
    ./IRReduce --input_file ../ir/hlo_1.ir --pass_inlineintermediates
else
  echo "Build failed. Exiting."
  exit 1
fi
