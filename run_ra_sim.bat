@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RA_SIM_FORCE_EXIT_ON_GUI_CLOSE=1"
set "PYTHON_CMD="
set "PAUSE_ON_ERROR="

echo %CMDCMDLINE% | findstr /I /C:" /c " >nul 2>nul
if %ERRORLEVEL%==0 set "PAUSE_ON_ERROR=1"
if defined RA_SIM_BATCH_PAUSE set "PAUSE_ON_ERROR=1"
if defined RA_SIM_BATCH_NO_PAUSE set "PAUSE_ON_ERROR="

pushd "%SCRIPT_DIR%" >nul 2>nul
if errorlevel 1 (
    echo Unable to open RA-SIM directory: "%SCRIPT_DIR%"
    set "EXITCODE=1"
    goto done
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3"
    )
)

if not defined PYTHON_CMD (
    echo RA-SIM requires Python 3.11 or newer, but no compatible Python was found.
    set "EXITCODE=1"
    goto done
)

%PYTHON_CMD% -m ra_sim %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo RA-SIM exited with code %EXITCODE%.
)

:done
if not "%EXITCODE%"=="0" if defined PAUSE_ON_ERROR pause
popd >nul 2>nul
endlocal & exit /b %EXITCODE%
