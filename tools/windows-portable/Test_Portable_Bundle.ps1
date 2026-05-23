param(
  [string]$BundleDir = "dist_portable\RIS_Portable",
  [switch]$RequireUI
)

$ErrorActionPreference = "Stop"

function Write-Section($t) {
  Write-Host ""
  Write-Host "==== $t ====" -ForegroundColor Cyan
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$BundlePath = Join-Path $RepoRoot $BundleDir
$PyExe = Join-Path $BundlePath "python\python.exe"
$RailwaySimExe = Join-Path $BundlePath "python\Scripts\railway-sim.exe"

Write-Section "Portable bundle smoke test"
if (!(Test-Path $BundlePath)) { throw "Portable bundle not found: $BundlePath" }
if (!(Test-Path $PyExe)) { throw "Embedded python.exe not found: $PyExe" }
if (!(Test-Path $RailwaySimExe)) { throw "railway-sim.exe not found: $RailwaySimExe" }

$env:PATH = "$BundlePath\python;$BundlePath\python\Scripts;$env:PATH"
Push-Location $BundlePath
try {
  Write-Section "Python/package imports"
  & $PyExe -c "import railway_simulator, numpy, pandas, scipy, yaml; print('imports-ok')"
  if ($LASTEXITCODE -ne 0) { throw "Portable import smoke failed" }

  if ($RequireUI) {
    & $PyExe -c "import streamlit, pyarrow, plotly; print('ui-imports-ok')"
    if ($LASTEXITCODE -ne 0) { throw "Portable UI import smoke failed" }
  }

  Write-Section "CLI help"
  & $RailwaySimExe --help | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "railway-sim --help failed" }

  Write-Section "Example simulation"
  $OutDir = Join-Path $BundlePath "results\portable_smoke_ice1_80"
  if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
  & $RailwaySimExe run --config "configs\ice1_80kmh.yml" --output-dir $OutDir
  if ($LASTEXITCODE -ne 0) { throw "portable example simulation failed" }
  if (!(Test-Path $OutDir)) { throw "example output directory was not created: $OutDir" }

  Write-Section "Parametric grid dry-run"
  & $RailwaySimExe study run-grid --spec "configs\studies\impact_parametric_mini.yml" --dry-run --limit 1
  if ($LASTEXITCODE -ne 0) { throw "portable parametric grid dry-run failed" }

  Write-Host ""
  Write-Host "Portable bundle smoke test passed." -ForegroundColor Green
}
finally {
  Pop-Location
}
