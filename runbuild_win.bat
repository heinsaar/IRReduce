@echo off
setlocal enabledelayedexpansion

:: Default build type
set BUILD_TYPE=Debug

:: Parse arguments
for %%a in (%*) do (
    if "%%~a"=="-debug" (
        set BUILD_TYPE=Debug
    )
    if "%%~a"=="-release" (
        set BUILD_TYPE=Release
    )
)

echo Building with configuration: !BUILD_TYPE!

:: Create build directory if missing
if not exist build (
    mkdir build
)
cd build

:: Configure only if first time (optional optimization)
if not exist Makefile (
    cmake -DCMAKE_BUILD_TYPE=!BUILD_TYPE! ..
)

:: Build
cmake --build . --config !BUILD_TYPE!
if ERRORLEVEL 1 (
  echo Build failed. Exiting.
  exit /b 1
)

:: Run
.\IRReduce.exe --input_file ..\ir\ir_1.txt --pass_inlineintermediates
