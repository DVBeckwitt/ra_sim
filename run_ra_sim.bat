@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%SCRIPT_DIR%main.py" %*
) else (
    py "%SCRIPT_DIR%main.py" %*
)

set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
