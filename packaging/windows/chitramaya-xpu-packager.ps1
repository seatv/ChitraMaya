# packaging/windows/chitramaya-xpu-packager.ps1
#
# XPU (Intel Arc) edition of chitramaya-packager.ps1. Same ritual, same
# installer output shape -- differences from the NVIDIA packager:
#   * builds packaging\windows\chitramaya-xpu.spec (torch +xpu venv
#     asserted by the spec itself; ffmpeg REQUIRED and capability-probed)
#   * no TensorRT engine steps: no Compile-All-Engines.ps1, no
#     models\engines dir -- engines do not exist on this edition
#   * installer artifacts are named ChitraMaya-xpu-install.* so NVIDIA
#     and XPU release assets can never be confused in a release upload
#
# Run from the repo root, inside the ARC venv (requirements-xpu.txt).

param(
  [string]$Name = "ChitraMaya",
  [switch]$SkipFfmpeg = $false,
  [switch]$SwapPolarsLtsCpu = $true,
  [string]$FfmpegDir = "",   # folder containing the ffmpeg.exe/ffprobe.exe
                             # to bundle. Prepended to PATH for this run so
                             # the spec bundles EXACTLY this build instead of
                             # whatever another tool put first on PATH (field
                             # event on the ROCm edition: a stray
                             # C:\MyPrograms\<other-tool>\ffmpeg.exe was
                             # winning the PATH race).
  [int]$SplitMB = -1   # -1 = AUTO: single volume when the dist fits under
                       # GitHub's 2GB asset limit, else 1900MB parts.
                       # 0 = force single volume; >0 = force that part size.
)

$ErrorActionPreference = "Stop"

Write-Host "== ChitraMaya packaging (XPU / Intel Arc) ==" -ForegroundColor Cyan
Write-Host "Name: $Name" -ForegroundColor Cyan
Write-Host "Repo: $(Get-Location)" -ForegroundColor Cyan

# ── Sanity: run from repo root, in the ARC release venv ──────────────────
if (-not (Test-Path ".\pyproject.toml")) { throw "Run from the repo root (pyproject.toml not found)." }
if (-not (Test-Path ".\chitramaya\__main__.py")) { throw "Repo layout unexpected: .\chitramaya\__main__.py not found." }
if (-not (Test-Path ".\packaging\windows\chitramaya-xpu.spec")) { throw "Missing .\packaging\windows\chitramaya-xpu.spec" }
if (-not (Test-Path ".\packaging\windows\chitramaya_entrypoint.py")) { throw "Missing packaging entrypoint." }

if (-not $env:VIRTUAL_ENV) {
  Write-Warning "No active virtualenv detected. Build from the ARC venv (torch +xpu, requirements-xpu.txt). The spec will refuse a non-xpu torch."
}

# ── ffmpeg preflight (the spec hard-fails without it; fail fast here) ────
if ($FfmpegDir) {
  if (-not (Test-Path (Join-Path $FfmpegDir "ffmpeg.exe"))) {
    throw "-FfmpegDir '$FfmpegDir' does not contain ffmpeg.exe."
  }
  if (-not (Test-Path (Join-Path $FfmpegDir "ffprobe.exe"))) {
    throw "-FfmpegDir '$FfmpegDir' does not contain ffprobe.exe."
  }
  $env:Path = "$FfmpegDir;$env:Path"
  Write-Host "Using -FfmpegDir: $FfmpegDir (prepended to PATH for this run)" -ForegroundColor Cyan
}
if (-not $SkipFfmpeg) {
  $ff = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
  $fp = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
  if (-not ($ff -and $fp)) {
    throw "ffmpeg.exe/ffprobe.exe not on PATH. On the XPU edition ffmpeg IS the decoder and encoder. Pass -FfmpegDir <folder with the gyan.dev 'full' build>, or put it first on PATH (the spec bundles exactly what it finds and needs the QSV encoders)."
  }
  Write-Host ("Bundling ffmpeg from: {0}" -f $ff.Source) -ForegroundColor Cyan
  # Provenance in the build log: exact version line of the chosen binary.
  $ffVer = (& $ff.Source -version 2>$null | Select-Object -First 1)
  if ($ffVer) { Write-Host ("  {0}" -f $ffVer) -ForegroundColor Gray }
}

# ── PyInstaller ──────────────────────────────────────────────────────────
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller

# ── polars AVX guard (same portability concern as the NVIDIA build) ──────
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

Write-Host "Running PyInstaller (torch+xpu stack; smaller than CUDA but still GBs)..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean .\packaging\windows\chitramaya-xpu.spec -- $specArgs
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

# ── Models drop folder (NO engines dir -- TensorRT does not exist here) ──
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "models") | Out-Null
("Place source model files here:" + [Environment]::NewLine +
 "  *.pt   - YOLO mosaic detection" + [Environment]::NewLine +
 "  *.pth  - BasicVSR++ mosaic restoration" + [Environment]::NewLine +
 "The Intel (XPU) edition runs models directly in PyTorch;" + [Environment]::NewLine +
 "there are no TensorRT engines to compile on this edition.") |
  Set-Content -Encoding ASCII (Join-Path $distDir "models\PUT-MODELS-HERE.txt")

# ── Release artifact ─────────────────────────────────────────────────────
# Two shapes, chosen by what actually fits -- both are ONE double-click
# installer exe (r6, ported from the ROCm packager: consistent install
# experience across editions):
#   FITS under GitHub's 2GB asset limit  -> ONE file:
#       ChitraMaya-xpu-install.exe = 7z.sfx extract-dialog stub + the
#       archive. Double-click, pick a folder, extracts. (Falls back to a
#       plain .7z only if 7z.sfx is not installed.)
#   DOES NOT FIT -> the two-stage SFX installer, same as the NVIDIA
#       edition: split volumes + a small ChitraMaya-xpu-install.exe that
#       verifies and extracts them (vendor setup: packaging\windows\
#       vendor\7zSD.sfx + 7zr.exe from the LZMA SDK, 7-zip.org/sdk.html).
Write-Host "Creating release artifact..." -ForegroundColor Yellow
if (Get-Command 7z -ErrorAction SilentlyContinue) {
  $installBase = "ChitraMaya-xpu-install"
  $vendorDir   = ".\packaging\windows\vendor"
  $sfxModule   = Join-Path $vendorDir "7zSD.sfx"
  $sevenZr     = Join-Path $vendorDir "7zr.exe"
  $instSrcDir  = ".\packaging\windows\installer"
  $limitMB     = 1990   # margin under GitHub's 2 GiB asset limit

  # Stale-artifact cleanup, LOUDLY (7z cannot update multivolume archives;
  # a locked leftover volume fails the whole step -- see the NVIDIA
  # packager for the field story).
  $stale = @(Get-ChildItem -File -ErrorAction SilentlyContinue `
             "$installBase.exe", "$installBase.7z", "$installBase.7z.0*")
  foreach ($f in $stale) {
    for ($try = 1; $try -le 5; $try++) {
      try {
        Remove-Item -Force -ErrorAction Stop $f.FullName
        Write-Host ("  removed stale {0}" -f $f.Name) -ForegroundColor Gray
        break
      } catch {
        if ($try -eq 5) {
          Write-Warning ("Stale {0} is locked and could not be deleted." -f $f.Name)
        } else {
          Write-Host ("  stale {0} is locked (attempt {1}/5); retrying in 2s..." -f $f.Name, $try) -ForegroundColor Yellow
          Start-Sleep -Seconds 2
        }
      }
    }
  }
  $survivors = @(Get-ChildItem -File -ErrorAction SilentlyContinue "$installBase.7z*", "$installBase.exe")

  if ($survivors.Count -gt 0) {
    Write-Warning ("Cannot delete stale artifact(s): {0}." -f (($survivors | ForEach-Object Name) -join ', '))
    Write-Warning "Something still holds them open (antivirus scan or an Explorer window)."
    Write-Warning "Close it (or wait a minute) and re-run the packager. Skipping artifact creation."
  } else {
    # ── Pass 1: single plain archive (no volume suffix) ──────────────────
    # AUTO short-circuit: a raw dist bigger than 6000 MB cannot land under
    # the 2GB asset limit (that would need >3:1 on binaries that are
    # already mostly compressed), so skip the wasted single-archive pass
    # and go straight to split volumes.
    $needSplit = $false
    $distMB = [math]::Round((Get-ChildItem -Recurse -File ".\dist\$Name" |
               Measure-Object Length -Sum).Sum / 1MB)
    if ($SplitMB -gt 0) {
      $needSplit = $true   # forced by parameter
    } elseif ($SplitMB -lt 0 -and $distMB -gt 6000) {
      Write-Host ("Dist is {0} MB raw -- cannot fit one 2GB asset; going straight to split volumes." -f $distMB) -ForegroundColor Yellow
      $needSplit = $true
    } else {
      7z a -t7z "$installBase.7z" ".\dist\$Name"
      if ($LASTEXITCODE -ne 0) { Write-Warning "Archive creation failed."; $needSplit = $null }
      else {
        $single = Get-Item "$installBase.7z"
        $mb = [math]::Round($single.Length / 1MB)
        if ($mb -le $limitMB) {
          # r6 (Gman, 2026-08-15: "We produce an install executable - why
          # skip that??"): make the single-file release an INSTALL EXE,
          # consistent with the multi-part editions -- still exactly ONE
          # release asset, but double-clickable. 7z.sfx is 7-Zip's
          # extract-dialog module (ships next to 7z.exe): stub + archive
          # concatenated = an exe that asks for a folder and extracts.
          $sfxGui = $null
          $sevenZCmd = Get-Command 7z -ErrorAction SilentlyContinue
          if ($sevenZCmd) {
            $cand = Join-Path (Split-Path $sevenZCmd.Source) "7z.sfx"
            if (Test-Path $cand) { $sfxGui = $cand }
          }
          if ($sfxGui) {
            $outPath = Join-Path (Get-Location) "$installBase.exe"
            try {
              $outFs = [IO.File]::Create($outPath)
              try {
                foreach ($pf in @($sfxGui, $single.FullName)) {
                  $bytes = [IO.File]::ReadAllBytes((Resolve-Path $pf))
                  $outFs.Write($bytes, 0, $bytes.Length)
                }
              } finally { $outFs.Close() }
              Remove-Item -Force $single.FullName
              $exeMb = [math]::Round((Get-Item $outPath).Length / 1MB)
              Write-Host ""
              Write-Host ("SINGLE-FILE INSTALLER: {0}  ({1} MB)" -f "$installBase.exe", $exeMb) -ForegroundColor Green
              Write-Host "Release this ONE file. Users double-click it, pick a folder," -ForegroundColor Cyan
              Write-Host "and it extracts there -- same experience as the other editions." -ForegroundColor Cyan
            } catch {
              Write-Warning ("SFX assembly failed: {0}. Falling back to the plain archive." -f $_.Exception.Message)
              Write-Host ("SINGLE-FILE RELEASE: {0}  ({1} MB)" -f $single.Name, $mb) -ForegroundColor Green
            }
          } else {
            Write-Warning "7z.sfx not found next to 7z.exe (it ships with the full 7-Zip install). Releasing the plain archive instead."
            Write-Host ("SINGLE-FILE RELEASE: {0}  ({1} MB)" -f $single.Name, $mb) -ForegroundColor Green
            Write-Host "Users extract it anywhere they like (Windows 11 Explorer, 7-Zip, or WinRAR)." -ForegroundColor Cyan
          }
        } else {
          Write-Host ("Archive is {0} MB (> {1} MB limit) -- switching to split volumes + SFX installer." -f $mb, $limitMB) -ForegroundColor Yellow
          Remove-Item -Force "$installBase.7z"
          $needSplit = $true
        }
      }
    }

    # ── Multi-volume + SFX installer (only when it does not fit) ─────────
    if ($needSplit -eq $true) {
      if (-not (Test-Path $sfxModule) -or -not (Test-Path $sevenZr)) {
        Write-Warning ("Vendor files missing ({0} and/or {1})." -f $sfxModule, $sevenZr)
        Write-Warning "Download the LZMA SDK from 7-zip.org/sdk.html (lzma<ver>.7z), then copy"
        Write-Warning "bin\7zSD.sfx and bin\7zr.exe into packaging\windows\vendor\ and re-run."
        Write-Warning "Skipping installer creation."
      } elseif (($instMissing = @(@("install.cmd", "install.ps1", "sfx_config.txt") |
                 Where-Object { -not (Test-Path (Join-Path $instSrcDir $_)) })).Count -gt 0) {
        Write-Warning ("Installer script(s) missing from {0}: {1}." -f $instSrcDir, ($instMissing -join ', '))
        Write-Warning "These ship in the repo (Batch 20d). Restore them and re-run."
        Write-Warning "Skipping installer creation."
      } else {
        $vol = if ($SplitMB -gt 0) { $SplitMB } else { 1900 }
        Write-Host "Splitting into ${vol}MB volumes (GitHub 2GB release-asset limit)..." -ForegroundColor Yellow
        7z a -t7z "-v${vol}m" "$installBase.7z" ".\dist\$Name"
        if ($LASTEXITCODE -ne 0) { Write-Warning "Archive creation failed."; }
        else {
          $parts = @(Get-ChildItem "$installBase.7z.0*" | Sort-Object Name)

          $stage = ".\build\installer_payload"
          Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $stage
          New-Item -ItemType Directory -Force -Path $stage | Out-Null
          Copy-Item (Join-Path $instSrcDir "install.cmd") $stage
          # Stamp BOTH the part count AND the base name: install.ps1
          # defaults to the NVIDIA names; unstamped, the xpu installer
          # hunts for ChitraMaya-install.7z.001 forever (the field bug in
          # the first xpu installer). Regexes match any prior stamp.
          (Get-Content (Join-Path $instSrcDir "install.ps1")) `
            -replace '^\$ExpectedParts = \d+.*$', ('$ExpectedParts = {0}   # stamped by packager' -f $parts.Count) `
            -replace '^\$BaseName\s*=.*$', ('$BaseName      = "{0}"   # stamped by packager' -f $installBase) |
            Set-Content (Join-Path $stage "install.ps1")
          Copy-Item $sevenZr $stage

          $payload7z = ".\build\installer_payload.7z"
          Remove-Item -ErrorAction SilentlyContinue $payload7z
          7z a -t7z $payload7z "$stage\*"
          if ($LASTEXITCODE -eq 0) {
            try {
              $sfxParts = @($sfxModule,
                            (Join-Path $instSrcDir "sfx_config.txt"),
                            $payload7z)
              $outPath = Join-Path (Get-Location) "$installBase.exe"
              $outFs = [IO.File]::Create($outPath)
              try {
                foreach ($pf in $sfxParts) {
                  $bytes = [IO.File]::ReadAllBytes((Resolve-Path $pf))
                  $outFs.Write($bytes, 0, $bytes.Length)
                }
              } finally { $outFs.Close() }
            } catch {
              Write-Warning ("SFX assembly failed: {0}" -f $_.Exception.Message)
            }
          }

          if (Test-Path "$installBase.exe") {
            Write-Host "Installer parts:" -ForegroundColor Green
            foreach ($p in ($parts + (Get-Item "$installBase.exe"))) {
              $mb = [math]::Round($p.Length / 1MB, 2)
              Write-Host ("  {0}  ({1} MB)" -f $p.Name, $mb) -ForegroundColor Green
            }
            Write-Host "Release ALL of the above together. The .exe verifies the volumes and names any missing file before extracting." -ForegroundColor Cyan
          } else {
            Write-Warning "Failed to assemble $installBase.exe."
          }
        }
      }
    }
  }
} else {
  Write-Host "7z not found - skipping release artifact (zip dist\$Name by hand instead)." -ForegroundColor Gray
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Output: $distDir" -ForegroundColor Green
Write-Host "Run:    $cmdPath        (or $Name.exe)" -ForegroundColor Green
