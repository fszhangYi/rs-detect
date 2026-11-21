@echo off
setlocal
cd /d "%~dp0"

REM Double-click / cmd entry → PowerShell start_web.ps1
REM Usage:
REM   start_web.bat
REM   start_web.bat 6666
REM   start_web.bat 6666 --no-browser

set "PORT=8787"
set "NO_BROWSER="
if not "%~1"=="" set "PORT=%~1"
if /I "%~2"=="--no-browser" set "NO_BROWSER=-NoBrowser"
if /I "%~1"=="--no-browser" (
  set "PORT=8787"
  set "NO_BROWSER=-NoBrowser"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" -Port %PORT% %NO_BROWSER%
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo [rs-detect] exit code %EC%
  pause
)
exit /b %EC%
