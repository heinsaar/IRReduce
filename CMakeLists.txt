﻿# CMakeList.txt : CMake project for IRReduce, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND grep -c Microsoft /proc/version
        OUTPUT_VARIABLE IS_WSL
    )
    if(IS_WSL)
        # This is a solution to a problem originally seen on WSL when trying to use multiple g++ compilers.
        # CMake started to fail its basic compiler checks and complaining that:
        #
        # "The C compiler" [path] "is not able to compile a simple test program."
        #
        # Compiler checks can fail for various reasons, such as incorrect system paths, permission issues,
        # or, in this case, the intricate interplay between WSL, the native filesystem, and the toolchain.
        # When using WSL and native Windows filesystem paths (/mnt/c/...), these checks sometimes produce
        # false negatives due to complexities related to file access permissions, path translation,
        # or system calls that work differently on WSL compared to a native Linux system.
        #
        # Therefore, we disable basic compiler checks on WSL unless a better solution is found.
        # Disabling these checks by setting these variables basically means telling CMake:
        # "trust me, compilers work".
        message(STATUS "DETECTED WSL ENVIRONMENT. SKIPPING COMPILER CHECKS.")
        set(CMAKE_C_COMPILER_WORKS   1 CACHE INTERNAL "")
        set(CMAKE_CXX_COMPILER_WORKS 1 CACHE INTERNAL "")
    endif()
endif()

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("IRReduce")

# Add source to this project's executable.
add_executable (IRReduce "irreduce.cpp" "irreduce.h" "kaizen.h")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET IRReduce PROPERTY CXX_STANDARD 20)
endif()

# Copy the automated build-and-run scripts
if(UNIX)
  configure_file(${CMAKE_SOURCE_DIR}/runbuild_linx.sh ${CMAKE_BINARY_DIR}/runbuild_linx.sh COPYONLY)
  file(READ ${CMAKE_BINARY_DIR}/runbuild_linx.sh CONTENTS)
  string(REPLACE "\r\n" "\n" FIXED_CONTENTS "${CONTENTS}")       # Replace CRLF with LF
  string(REPLACE "\r"   "\n" FIXED_CONTENTS "${FIXED_CONTENTS}") # Replace CR   with LF (in case the file uses old Mac line endings)
  file(WRITE ${CMAKE_BINARY_DIR}/runbuild_linx.sh "${FIXED_CONTENTS}")
endif()

if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/runbuild_win.bat ${CMAKE_BINARY_DIR}/runbuild_win.bat COPYONLY)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# TODO: Add tests and install targets if needed.