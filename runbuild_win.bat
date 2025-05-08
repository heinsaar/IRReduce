@echo off
setlocal EnableDelayedExpansion

rem ── defaults ────────────────────────────────
set "BUILD_TYPE=Debug"
set "IR_FILE=..\ir\hlo_1.ir"
set "EXTRA_ARGS="

rem ── parse arguments ─────────────────────────
:parse
if "%~1"=="" goto done

set "TOK=%~1"

if /I "!TOK!"=="-debug" (
    set "BUILD_TYPE=Debug"
) else if /I "!TOK!"=="-release" (
    set "BUILD_TYPE=Release"
) else if /I "!TOK!"=="--input_file" (
    shift
    set "IR_FILE=%~1"
) else if /I "!TOK:~0,13!"=="--input_file=" (
    rem remove the prefix --input_file=
    set "IR_FILE=!TOK:~13!"
) else (
    rem pass every other flag/value pair straight through
    set "EXTRA_ARGS=!EXTRA_ARGS! !TOK!"
)

shift
goto parse
:done

echo Building with configuration: !BUILD_TYPE!
echo IR file: "!IR_FILE!"
echo Extra args:^> !EXTRA_ARGS!

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

rem ── choose runtime folder (Debug/Release vs root) ────
if exist ".\!BUILD_TYPE!\IRReduce.exe" (
    set "RUN_EXE=.\!BUILD_TYPE!\IRReduce.exe"
) else (
    set "RUN_EXE=.\IRReduce.exe"
)

rem ── run ─────────────────────────────────────
%RUN_EXE% --input_file "!IR_FILE!" !EXTRA_ARGS!

endlocal
