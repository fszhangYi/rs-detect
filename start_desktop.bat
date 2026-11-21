@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_desktop.ps1"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo [rs-detect] exit code %EC%
  pause
)
exit /b %EC%
