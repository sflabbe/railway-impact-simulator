@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows-portable\Build_Portable_Bundle.ps1"
if errorlevel 1 (
  echo.
  echo Build failed. Scroll up for details.
  pause
  exit /b 1
)
echo.
echo Build finished. See dist_portable\RailwayImpactSimulator_Portable_Windows.zip
pause
