@echo off
setlocal EnableDelayedExpansion

rem ── defaults ────────────────────────────────
set "BUILD_TYPE=Debug"
set "IR_FILE=..\ir\hlo_1.ir"
set "EXTRA_ARGS="

rem ── parse arguments ─────────────────────────
:parse
if "%~1"=="" goto done
if /I "%~1"=="-debug" (
    set "BUILD_TYPE=Debug"
) else if /I "%~1"=="-release" (
    set "BUILD_TYPE=Release"
) else if not defined IR_SPECIFIED (
    set "IR_FILE=%~1"
    set "IR_SPECIFIED=1"
) else (
    set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
)
shift
goto parse
:done

echo Building with configuration: !BUILD_TYPE!
echo IR file: "!IR_FILE!"

rem ── configure & build ───────────────────────
if not exist build mkdir build
cd build

if not exist Makefile (
    cmake -DCMAKE_BUILD_TYPE=!BUILD_TYPE! ..
)

cmake --build . --config !BUILD_TYPE! || (
    echo Build failed. Exiting.
    endlocal & exit /b 1
)

rem ── choose runtime folder (Debug/Release vs root) ───────────────────────────
if exist ".\!BUILD_TYPE!\IRReduce.exe" (
    set "RUN_EXE=.\!BUILD_TYPE!\IRReduce.exe"
) else (
    rem single-config generators (Ninja/Makefiles) drop exe in current dir
    set "RUN_EXE=.\IRReduce.exe"
)

rem ── run ─────────────────────────────────────
%RUN_EXE% --input_file "!IR_FILE!" !EXTRA_ARGS!

endlocal
