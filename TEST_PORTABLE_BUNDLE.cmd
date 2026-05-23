@echo off
setlocal
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
