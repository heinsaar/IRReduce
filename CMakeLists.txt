# CMakeList.txt : CMake project for IRReduce, include source and define
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
add_executable (IRReduce "IRReduce.cpp" "IRReduce.h" "kaizen.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET IRReduce PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.