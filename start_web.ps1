#Requires -Version 5.1
# Start rs-detect Web UI on Windows (FastAPI + Vite dist).
# ASCII-only strings so Windows PowerShell 5.1 does not choke on UTF-8 punctuation.
[CmdletBinding()]
param(
  [string]$HostAddr = $(if ($env:RS_DETECT_WEB_HOST) { $env:RS_DETECT_WEB_HOST } else { "127.0.0.1" }),
  [int]$Port = $(if ($env:RS_DETECT_WEB_PORT) { [int]$env:RS_DETECT_WEB_PORT } else { 8787 }),
  [switch]$NoBrowser,
  [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
  foreach ($cand in @(
      (Join-Path $Root ".venv312\Scripts\python.exe"),
      (Join-Path $Root ".venv\Scripts\python.exe"),
      (Join-Path $Root ".venv312\Scripts\python"),
      (Join-Path $Root ".venv\Scripts\python")
    )) {
    if (Test-Path $cand) { return $cand }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "No Python found. Create .venv: python -m venv .venv && .\.venv\Scripts\pip install -e ."
}

function Ensure-WebDist {
  $distIndex = Join-Path $Root "web\dist\index.html"
  $need = $Rebuild -or -not (Test-Path $distIndex) -or ($env:RS_DETECT_WEB_REBUILD -eq "1")
  if (-not $need) {
    Write-Host "[rs-detect] web dist up to date -> web\dist"
    return
  }
  $web = Join-Path $Root "web"
  Push-Location $web
  try {
    if (Get-Command pnpm -ErrorAction SilentlyContinue) {
      if (-not (Test-Path "node_modules\vite")) { pnpm install }
      pnpm run build
    } elseif (Get-Command npm -ErrorAction SilentlyContinue) {
      if (-not (Test-Path "node_modules\vite")) { npm install }
      npm run build
    } else {
      throw "Need Node + pnpm or npm to build web UI (or commit/copy web/dist)."
    }
  } finally {
    Pop-Location
  }
}

$py = Find-Python
$env:RS_DETECT_WEB_HOST = $HostAddr
$env:RS_DETECT_WEB_PORT = "$Port"
$env:PYTHONPATH = "$(Join-Path $Root 'src')" + $(if ($env:PYTHONPATH) { ";$($env:PYTHONPATH)" } else { "" })
foreach ($sib in @("sensors", "sensors-arm")) {
  $src = Join-Path (Split-Path $Root -Parent) "$sib\src"
  if (Test-Path $src) {
    $env:PYTHONPATH = "$src;$($env:PYTHONPATH)"
  }
}

Ensure-WebDist

# Cell C URDF + STL (sibling sensors-arm); required by /assets/models/cs66/...
& $py (Join-Path $Root "scripts\ensure_arm_assets.py")
if ($LASTEXITCODE -ne 0) {
  Write-Host "[rs-detect] WARN: arm URDF assets missing - Arm tab will 404 until sensors-arm is linked"
}

# RealSense CAD overlays (git symlinks break on Windows → DataView STL errors)
& $py (Join-Path $Root "scripts\ensure_realsense_assets.py")
if ($LASTEXITCODE -ne 0) {
  Write-Host "[rs-detect] WARN: D435/D405 STL missing under vendor/realsense_cad - camera CAD overlay will fail"
}

$Url = "http://127.0.0.1:${Port}/"
Write-Host "[rs-detect] root=$Root"
Write-Host "[rs-detect] python=$py"
Write-Host "[rs-detect] web -> $Url  arm -> ${Url}#/arm"

$openBrowser = -not $NoBrowser
if ($env:RS_DETECT_OPEN_BROWSER -eq "0") { $openBrowser = $false }
if ($openBrowser) {
  Start-Job -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 3
    Start-Process $u
  } -ArgumentList $Url | Out-Null
  Write-Host "[rs-detect] will open browser in 3s -> $Url"
}

$flag = Join-Path $Root ".rs_detect_restart"
if (Test-Path $flag) { Remove-Item $flag -Force }

while ($true) {
  & $py -m uvicorn rs_detect.api.server:app --host $HostAddr --port $Port
  $code = $LASTEXITCODE
  if (Test-Path $flag) {
    Remove-Item $flag -Force -ErrorAction SilentlyContinue
    Write-Host "[rs-detect] restart requested - relaunching..."
    Start-Sleep -Milliseconds 500
    continue
  }
  exit $code
}
