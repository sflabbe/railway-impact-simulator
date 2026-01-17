param(
  [switch]$IncludeUI,
  [string]$PythonVersion = "3.12.10",
  [ValidateSet("amd64","win32","arm64")] [string]$Arch = "amd64",
  [string]$OutputDir = "dist_portable",
  [string]$UiHost = "127.0.0.1",
  [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

function Write-Section($t) {
  Write-Host ""
  Write-Host "==== $t ====" -ForegroundColor Cyan
}

# Script path: <repo>\tools\windows-portable\Build_Portable_Bundle.ps1
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..\..")

if (!(Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
  throw "Cannot find pyproject.toml at repo root: $RepoRoot"
}

$OutRoot   = Join-Path $RepoRoot $OutputDir
$BundleDir = Join-Path $OutRoot "RIS_Portable"
$PyDir     = Join-Path $BundleDir "python"

Write-Section "Clean output"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
if (Test-Path $BundleDir) { Remove-Item -Recurse -Force $BundleDir }
New-Item -ItemType Directory -Force -Path $PyDir | Out-Null

Write-Section "Download Python embeddable ($PythonVersion, $Arch)"
# Official file name pattern:
#   python-<ver>-embed-<arch>.zip
$PyZipName = "python-$PythonVersion-embed-$Arch.zip"
$PyZipPath = Join-Path $OutRoot $PyZipName
$PyUrl = "https://www.python.org/ftp/python/$PythonVersion/$PyZipName"

if (!(Test-Path $PyZipPath)) {
  try {
    Invoke-WebRequest -Uri $PyUrl -OutFile $PyZipPath -UseBasicParsing
  } catch {
    $statusCode = $null
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
      $statusCode = [int]$_.Exception.Response.StatusCode
    }
    $hint = ""
    if ($statusCode -eq 404) {
      $parsedVersion = $null
      try {
        $parsedVersion = [version]$PythonVersion
      } catch {
        $parsedVersion = $null
      }
      if ($parsedVersion -and $parsedVersion.Major -eq 3 -and $parsedVersion.Minor -eq 12 -and $parsedVersion.Build -ge 11) {
        $hint = " Note: Python 3.12.11+ are source-only releases and do not publish Windows embeddable ZIPs. Use 3.12.10 or switch to Python 3.13+ for current Windows binaries."
      }
    }
    throw "Failed to download $PyUrl. Check -PythonVersion/-Arch. Error: $($_.Exception.Message)$hint"
  }
} else {
  Write-Host "Using existing $PyZipPath"
}

Write-Section "Extract Python"
Expand-Archive -Path $PyZipPath -DestinationPath $PyDir -Force

# Ensure required dirs exist
New-Item -ItemType Directory -Force -Path (Join-Path $PyDir "Lib\site-packages") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PyDir "Scripts") | Out-Null

Write-Section "Enable site-packages for embeddable Python"
$pth = Get-ChildItem -Path $PyDir -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) { throw "Could not find python*._pth inside $PyDir" }

$lines = Get-Content $pth.FullName -ErrorAction Stop

# Add Lib\site-packages and current folder to sys.path
if (-not ($lines -contains "Lib\site-packages")) { $lines += "Lib\site-packages" }
if (-not ($lines -contains ".")) { $lines = @(".") + $lines }

# Ensure 'import site' is enabled (uncomment it if present; otherwise append)
$lines = $lines | ForEach-Object { $_ -replace '^\s*#\s*import\s+site\s*$','import site' }
if (-not ($lines -contains "import site")) { $lines += "import site" }

Set-Content -Path $pth.FullName -Value $lines -Encoding Ascii

$PyExe = Join-Path $PyDir "python.exe"
if (!(Test-Path $PyExe)) { throw "python.exe not found at $PyExe" }

Write-Section "Install pip into embedded Python"
$GetPip = Join-Path $OutRoot "get-pip.py"
if (!(Test-Path $GetPip)) {
  Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip -UseBasicParsing
}
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
& $PyExe $GetPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "Failed to install pip" }

Write-Section "Upgrade packaging tools"
# Use --no-cache-dir to avoid long path issues on Windows
# Install pip first
& $PyExe -m pip install --upgrade --no-cache-dir pip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }

# Install setuptools and wheel with flags to avoid Windows path length issues
# --no-deps: Don't install dependencies to avoid setuptools test files with long paths
# PIP_NO_BUILD_ISOLATION: Use system packages instead of isolated build environment
$env:PIP_NO_BUILD_ISOLATION = "1"
& $PyExe -m pip install --upgrade --no-cache-dir --no-deps setuptools wheel --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
  Write-Host "Warning: setuptools/wheel installation had errors. Continuing anyway..." -ForegroundColor Yellow
}

Write-Section "Install project into embedded Python"
Push-Location $RepoRoot
try {
  # Use --no-cache-dir, --prefer-binary, and --no-build-isolation to avoid path length issues
  # --no-build-isolation: Reuse the setuptools we installed above
  # These flags help avoid Windows MAX_PATH limitations
  if ($IncludeUI) {
    & $PyExe -m pip install --no-cache-dir --prefer-binary --no-build-isolation ".[ui]" --no-warn-script-location
  } else {
    & $PyExe -m pip install --no-cache-dir --prefer-binary --no-build-isolation . --no-warn-script-location
  }
  if ($LASTEXITCODE -ne 0) { throw "Failed to install project" }
} finally {
  Pop-Location
}

Write-Section "Copy runtime assets (configs + docs)"
$copyTargets = @(
  "configs",
  "examples",
  "docs",
  "README.md",
  "PROJECT_SUMMARY.md",
  "CITATION_REFERENCE.md",
  "VALIDATION_Pioneer.md",
  "LICENSE"
)
foreach ($t in $copyTargets) {
  $src = Join-Path $RepoRoot $t
  if (Test-Path $src) {
    Copy-Item -Recurse -Force $src (Join-Path $BundleDir $t)
  }
}

Write-Section "Create launchers"

# CLI launcher: opens a ready shell in the bundle root (PATH set)
$runCli = @'
@echo off
setlocal
set "ROOT=%~dp0"
set "PATH=%ROOT%python;%ROOT%python\Scripts;%PATH%"
cd /d "%ROOT%"

echo ============================================================
echo Railway Impact Simulator (Portable)
echo.
echo Try:
echo   railway-sim --help
echo   railway-sim run --config configs\ice1_80kmh.yml --output-dir results\ice1_80
echo ============================================================
echo.
cmd /k
'@
Set-Content -Path (Join-Path $BundleDir "Run_CLI.bat") -Value $runCli -Encoding Ascii

# UI launcher: uses the built-in CLI command that starts Streamlit
$runUi = @"
@echo off
setlocal
set "ROOT=%~dp0"
set "PATH=%ROOT%python;%ROOT%python\Scripts;%PATH%"
cd /d "%ROOT%"

echo ============================================================
echo Railway Impact Simulator UI (Streamlit)
echo URL: http://$UiHost`:$UiPort
echo Close this window to stop the server.
echo ============================================================
echo.

railway-sim ui --host $UiHost --port $UiPort
"@
Set-Content -Path (Join-Path $BundleDir "Run_UI.bat") -Value $runUi -Encoding Ascii

# One-click example runner (CLI)
$ex1 = @'
@echo off
setlocal
set "ROOT=%~dp0"
set "PATH=%ROOT%python;%ROOT%python\Scripts;%PATH%"
cd /d "%ROOT%"

railway-sim run --config "configs\ice1_80kmh.yml" --output-dir "results\ice1_80"
pause
'@
Set-Content -Path (Join-Path $BundleDir "Example_Run_ICE1_80kmh.bat") -Value $ex1 -Encoding Ascii

# Bundle info
$info = @"
Railway Impact Simulator - Portable Windows Bundle

How to run:
- Run_CLI.bat                 -> opens a shell with PATH set
- Example_Run_ICE1_80kmh.bat  -> runs an example simulation
- Run_UI.bat                  -> Streamlit UI (requires IncludeUI build)

Built on: $(Get-Date -Format s)
Python embeddable: $PythonVersion ($Arch)
IncludeUI: $IncludeUI
Repo URL: https://github.com/sflabbe/railway-impact-simulator
"@
Set-Content -Path (Join-Path $BundleDir "BUNDLE_INFO.txt") -Value $info -Encoding UTF8

Write-Section "Create distributable zip"
$ZipOut = Join-Path $OutRoot "RailwayImpactSimulator_Portable_Windows.zip"
if (Test-Path $ZipOut) { Remove-Item -Force $ZipOut }
Compress-Archive -Path (Join-Path $BundleDir "*") -DestinationPath $ZipOut -Force

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "Portable ZIP: $ZipOut"
