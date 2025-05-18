@echo off
setlocal EnableDelayedExpansion

REM ── defaults ─────────────────────────────────────────────
set "BUILD_TYPE=Debug"
set "IR_FILE=.\ir\input\hlo_1.ir"
set "EXTRA_ARGS="

REM ── parse all CLI args ───────────────────────────────────
:parse
if "%~1"=="" goto done

set "ARG=%~1"

if /I "!ARG!"=="-debug" (
    set "BUILD_TYPE=Debug"
) else if /I "!ARG!"=="-release" (
    set "BUILD_TYPE=Release"
) else if /I "!ARG:~0,13!"=="--input_file=" (
    set "IR_FILE=!ARG:~13!"
) else if /I "!ARG!"=="--input_file" (
    shift
    set "IR_FILE=%~1"
) else (
    set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
)
shift
goto parse

:done

echo Running test with configuration: !BUILD_TYPE!
echo IR file: !IR_FILE!

REM ── assume build exists in .\build ──────────────────────
set "EXECUTABLE=.\build\IRReduce.exe"

if not exist "!EXECUTABLE!" (
    echo ERROR: Executable not found: !EXECUTABLE!
    echo Make sure the project is built beforehand.
    exit /b 1
)

REM ── run IRReduce ─────────────────────────────────────────
"!EXECUTABLE!" --input_file "!IR_FILE!" !EXTRA_ARGS!
