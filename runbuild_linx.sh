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
cmake --build . --config $BUILD_TYPE
if [ $? -eq 0 ]; then
    # Run
    ./IRReduce --input_file ../ir/ir_1.txt --pass_inlineintermediates
else
  echo "Build failed. Exiting."
  exit 1
fi
