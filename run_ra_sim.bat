@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python -m ra_sim %*
) else (
    py -m ra_sim %*
)

set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
