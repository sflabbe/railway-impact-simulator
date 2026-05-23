@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_COLOR=1
set CLICOLOR=0
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows-portable\Test_Portable_Bundle.ps1"
if errorlevel 1 (
  echo.
  echo Portable smoke test failed. Scroll up for details.
  pause
  exit /b 1
)
echo.
echo Portable smoke test passed.
pause
