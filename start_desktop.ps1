#Requires -Version 5.1
<#
.SYNOPSIS
  Start rs-detect Fluent desktop UI on Windows (PyQt).
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$py = $null
foreach ($cand in @(
    (Join-Path $Root ".venv312\Scripts\python.exe"),
    (Join-Path $Root ".venv\Scripts\python.exe")
  )) {
  if (Test-Path $cand) { $py = $cand; break }
}
if (-not $py) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { $py = $cmd.Source }
}
if (-not $py) { throw "No Python / .venv found" }

$env:PYTHONPATH = "$(Join-Path $Root 'src')"
foreach ($sib in @("sensors", "sensors-arm")) {
  $src = Join-Path (Split-Path $Root -Parent) "$sib\src"
  if (Test-Path $src) { $env:PYTHONPATH = "$src;$($env:PYTHONPATH)" }
}

# Do NOT set QT_QPA_PLATFORM=xcb on Windows (that is Linux/X11).
if ($env:QT_QPA_PLATFORM -eq "xcb") {
  Remove-Item Env:QT_QPA_PLATFORM
}

Write-Host "[rs-detect] root=$Root"
Write-Host "[rs-detect] python=$py"
Write-Host "[rs-detect] starting Fluent desktop..."
& $py -m rs_detect.desktop.app
exit $LASTEXITCODE
