param(
  [string]$Name = "ChitraMaya",
  [switch]$SkipFfmpeg = $false,
  [switch]$SwapPolarsLtsCpu = $true,
  [int]$SplitMB = 1900   # SFX volume size (MB). 0 = single file. Default splits
                         # into <2GB parts for GitHub's 2GB release-asset limit.
)

$ErrorActionPreference = "Stop"

Write-Host "== ChitraMaya packaging ==" -ForegroundColor Cyan
Write-Host "Name: $Name" -ForegroundColor Cyan
Write-Host "Repo: $(Get-Location)" -ForegroundColor Cyan

# ── Sanity: run from repo root, in the release venv ──────────────────────
if (-not (Test-Path ".\pyproject.toml")) { throw "Run from the repo root (pyproject.toml not found)." }
if (-not (Test-Path ".\chitramaya\__main__.py")) { throw "Repo layout unexpected: .\chitramaya\__main__.py not found." }
if (-not (Test-Path ".\packaging\windows\chitramaya.spec")) { throw "Missing .\packaging\windows\chitramaya.spec" }
if (-not (Test-Path ".\packaging\windows\chitramaya_entrypoint.py")) { throw "Missing packaging entrypoint." }

if (-not $env:VIRTUAL_ENV) {
  Write-Warning "No active virtualenv detected. Build from the SAME venv you run ChitraMaya in (it must have your CUDA torch / TensorRT / PyNvVideoCodec wheels)."
}

# ── PyInstaller ──────────────────────────────────────────────────────────
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller

# ── polars AVX guard (ultralytics pulls polars; its AVX build crashes on
#    older CPUs). Swap to the LTS-CPU build for portable releases. ─────────
if ($SwapPolarsLtsCpu) {
  $polars = (& python -c "import importlib.util; print('1' if importlib.util.find_spec('polars') else '0')").Trim()
  if ($polars -eq "1") {
    Write-Host "Swapping polars -> polars-lts-cpu (portability)..." -ForegroundColor Yellow
    python -m pip uninstall -y polars | Out-Null
    python -m pip install -U polars-lts-cpu | Out-Null
  }
}

# ── Clean previous ───────────────────────────────────────────────────────
foreach ($d in @(".\build", ".\dist")) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }

# ── Build (pass spec args after the `--`) ────────────────────────────────
$specArgs = @("--name=$Name")
if ($SkipFfmpeg) { $specArgs += "--skip-ffmpeg" }

Write-Host "Running PyInstaller (5-15 min; CUDA/TRT stack is large, output ~5-8 GB)..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean .\packaging\windows\chitramaya.spec -- $specArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$distDir = Join-Path (Resolve-Path ".\dist").Path $Name

# ── .cmd wrapper so cwd = exe dir (config.json / models resolve) ─────────
$cmdPath = Join-Path $distDir "$Name.cmd"
@"
@echo off
setlocal
cd /d %~dp0
"%~dp0$Name.exe" %*
"@ | Set-Content -Encoding ASCII $cmdPath

# ── Copy config template next to the exe (if present) ────────────────────
$cfgSrc = Join-Path (Resolve-Path ".").Path "ChitraMaya-config.json"
if (Test-Path $cfgSrc) {
  Copy-Item -Force $cfgSrc (Join-Path $distDir "ChitraMaya-config.json")
  Write-Host "Copied ChitraMaya-config.json" -ForegroundColor Green
} else {
  Write-Host "No ChitraMaya-config.json at repo root (app will create one on first run)." -ForegroundColor Gray
}

# ── Empty models/engines so the app finds the dir (no weights shipped) ───
$eng = Join-Path $distDir "models\engines"
New-Item -ItemType Directory -Force -Path $eng | Out-Null
"Place compiled .engine files here (use Manage Models / Compile-All-Engines.ps1)." |
  Set-Content -Encoding ASCII (Join-Path $eng "PUT-ENGINES-HERE.txt")

# ── Ship the compile script + a models drop folder ───────────────────────
# CRITICAL for fresh installs: the frozen exe bundles the compile code
# (tools/* via collect_submodules), so this PS1 drives ChitraMaya.exe
# -compile-* to build engines on the TARGET machine's GPU. No Python or
# ./tools folder needed on the clean machine. Ship the PS1 next to the exe;
# it resolves .\ChitraMaya.exe via $PSScriptRoot.
$compileSrc = Join-Path (Resolve-Path ".").Path "Compile-All-Engines.ps1"
if (Test-Path $compileSrc) {
  Copy-Item -Force $compileSrc (Join-Path $distDir "Compile-All-Engines.ps1")
  Write-Host "Copied Compile-All-Engines.ps1" -ForegroundColor Green
} else {
  Write-Warning "Compile-All-Engines.ps1 not found at repo root - fresh installs will have NO way to compile engines. Fix before releasing."
}
("Place source model files here:" + [Environment]::NewLine +
 "  *.pt   - YOLO mosaic detection" + [Environment]::NewLine +
 "  *.pth  - BasicVSR++ mosaic restoration" + [Environment]::NewLine +
 "Then run Compile-All-Engines.ps1 to build engines for THIS machine's GPU.") |
  Set-Content -Encoding ASCII (Join-Path $distDir "models\PUT-MODELS-HERE.txt")

# ── Optional self-extracting installer ──────────────────────────────────
Write-Host "Creating self-extracting installer..." -ForegroundColor Yellow
if (Get-Command 7z -ErrorAction SilentlyContinue) {
  $installerName = "ChitraMaya-install.exe"
  # Remove stale installer + volume parts from a previous run.
  Remove-Item -ErrorAction SilentlyContinue "$installerName", "$installerName.0*"

  if ($SplitMB -gt 0) {
    Write-Host "Splitting into ${SplitMB}MB volumes (GitHub 2GB release-asset limit)..." -ForegroundColor Yellow
    7z a -sfx "-v${SplitMB}m" "$installerName" ".\dist\$Name"
  } else {
    7z a -sfx "$installerName" ".\dist\$Name"
  }

  if ($LASTEXITCODE -eq 0) {
    $parts = Get-ChildItem -ErrorAction SilentlyContinue "$installerName", "$installerName.0*" |
             Sort-Object Name
    if ($parts) {
      Write-Host "Installer parts:" -ForegroundColor Green
      foreach ($p in $parts) {
        $mb = [math]::Round($p.Length / 1MB, 2)
        Write-Host ("  {0}  ({1} MB)" -f $p.Name, $mb) -ForegroundColor Green
      }
      if ($parts.Count -gt 1) {
        Write-Host "Release ALL parts together. Users download all, then run the .001 (or the .exe) to reassemble." -ForegroundColor Cyan
      }
    }
  } else {
    Write-Warning "Failed to create self-extracting installer."
  }
} else {
  Write-Host "7z not found - skipping SFX installer (zip dist\$Name instead)." -ForegroundColor Gray
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Output: $distDir" -ForegroundColor Green
Write-Host "Run:    $cmdPath        (or $Name.exe)" -ForegroundColor Green
