# packaging/windows/chitramaya-packager.ps1
#
# NVIDIA (CUDA / TensorRT) edition -- the original of the three packagers.
# v1.50.00: -FfmpegDir and the AUTO single-file installer exe ported back
# from the ROCm packager (r6, production-proven), so all three editions
# share the same invocation and the same release-artifact shapes.

param(
  [string]$Name = "ChitraMaya",
  [switch]$SkipFfmpeg = $false,
  [switch]$SwapPolarsLtsCpu = $true,
  [string]$FfmpegDir = "",   # folder containing the ffmpeg.exe/ffprobe.exe to
                             # bundle. Prepended to PATH for this run so the
                             # spec bundles EXACTLY this build instead of
                             # whatever another tool put first on PATH (field
                             # event on the ROCm edition: a stray
                             # C:\MyPrograms\<other-tool>\ffmpeg.exe was
                             # winning the PATH race).
  [int]$SplitMB = -1   # -1 = AUTO: single-file installer exe when the dist
                       # fits under GitHub's 2GB asset limit, else 1900MB
                       # parts + shepherd SFX (the CUDA/TRT stack usually
                       # splits). 0 = force single volume; >0 = force that
                       # part size.
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

# ── ffmpeg preflight ─────────────────────────────────────────────────────
# The NVIDIA edition decodes/encodes via PyNvVideoCodec, but ffmpeg does
# the finalize remux and the fallback paths -- and the spec SILENTLY skips
# bundling when nothing is on PATH, shipping a degraded dist. Fail fast
# here instead, and pin the exact build with -FfmpegDir.
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
    throw "ffmpeg.exe/ffprobe.exe not on PATH. ffmpeg performs the finalize remux on this edition; without it the bundle ships degraded. Pass -FfmpegDir <folder with the gyan.dev 'full' build>, put one first on PATH, or -SkipFfmpeg (target machines then need ffmpeg on PATH themselves)."
  }
  Write-Host ("Bundling ffmpeg from: {0}" -f $ff.Source) -ForegroundColor Cyan
  # Provenance in the build log: exact version line of the chosen binary.
  $ffVer = (& $ff.Source -version 2>$null | Select-Object -First 1)
  if ($ffVer) { Write-Host ("  {0}" -f $ffVer) -ForegroundColor Gray }
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

# ── ChitraMaya-config.json is NEVER shipped (Batch 53, field-caught) ─────
# The old "copy config template if present" step silently packaged the
# DEVELOPER'S personal config (output dirs, model paths, dial settings)
# into every build made from a checkout that had one at the repo root.
# The app creates a correct config on first run from the code's own
# defaults (/api/default-config <- models.py dataclasses, the single
# source of truth), so shipping a file here can only cause drift or leak
# the builder's environment. If a repo-root config exists, say so and
# deliberately skip it.
$cfgSrc = Join-Path (Resolve-Path ".").Path "ChitraMaya-config.json"
if (Test-Path $cfgSrc) {
  Write-Host "NOTE: repo-root ChitraMaya-config.json is the builder's personal file -- NOT shipped (app creates defaults on first run)." -ForegroundColor Yellow
} else {
  Write-Host "No ChitraMaya-config.json at repo root (app will create one on first run)." -ForegroundColor Gray
}

# ── VERSION.txt (v1.60, CM-097): the frozen exe hides __version__, and the
#    patch toolchain names + guards from/to versions by this file. ────────
$verMatch = Select-String -Path .\chitramaya\__init__.py -Pattern '__version__\s*=\s*"([^"]+)"'
if ($verMatch) {
  Set-Content -Encoding ASCII (Join-Path $distDir "VERSION.txt") $verMatch.Matches[0].Groups[1].Value
  Write-Host ("Stamped VERSION.txt = {0}" -f $verMatch.Matches[0].Groups[1].Value) -ForegroundColor Green
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

# ── Release artifact ─────────────────────────────────────────────────────
# Two shapes, chosen by what actually fits -- both are ONE double-click
# installer exe (r6, ported from the ROCm packager: consistent install
# experience across editions):
#   FITS under GitHub's 2GB asset limit  -> ONE file:
#       ChitraMaya-install.exe = 7z.sfx extract-dialog stub + the archive.
#       Double-click, pick a folder, extracts. (Falls back to a plain .7z
#       only if 7z.sfx is not installed.)
#   DOES NOT FIT (the usual case for the CUDA/TRT stack) -> split volumes
#       + ChitraMaya-install.exe shepherd SFX that verifies the volumes
#       are all present (naming exactly which file is missing) before
#       extracting -- instead of 7-Zip's bare "Cannot open the file as
#       archive".
#
# One-time vendor setup for the multi-part shape (packaging\windows\vendor\):
#   BOTH files ship in the LZMA SDK's bin\ folder (the modern "extra"
#   package no longer carries SFX modules):
#     https://7-zip.org/sdk.html  -> lzma<ver>.7z  -> bin\7zSD.sfx, bin\7zr.exe
#   7zSD.sfx  - SFX module for installers (prepended to make the .exe)
#   7zr.exe   - standalone console 7z extractor (bundled in the payload so
#               end users do not need 7-Zip installed)
# Both are official Igor Pavlov binaries and redistributable.
Write-Host "Creating release artifact..." -ForegroundColor Yellow
if (Get-Command 7z -ErrorAction SilentlyContinue) {
  $installBase = "ChitraMaya-install"
  $vendorDir   = ".\packaging\windows\vendor"
  $sfxModule   = Join-Path $vendorDir "7zSD.sfx"
  $sevenZr     = Join-Path $vendorDir "7zr.exe"
  $instSrcDir  = ".\packaging\windows\installer"
  $limitMB     = 1990   # margin under GitHub's 2 GiB asset limit

  # Remove stale artifacts from a previous run -- LOUDLY. 7-Zip cannot
  # update a multivolume archive ("Updating for multivolume archives is not
  # implemented"), so a single leftover .7z.001 fails the whole installer
  # step. The old SilentlyContinue delete masked exactly that: a volume
  # still locked (antivirus scan, Explorer preview) survived the delete,
  # 7z hit it, and the NEXT run "worked without changes" once the lock had
  # cleared. Now: retry locked files, and refuse to proceed while any
  # volume remains, naming it.
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
    # and go straight to split volumes. The CUDA/TRT dist usually takes
    # this path.
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
          # consistent with the multi-part shape -- still exactly ONE
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
              # r8: delete the partially-written exe so a corpse never sits
              # next to the good archive looking like a deliverable.
              Remove-Item -Force -ErrorAction SilentlyContinue $outPath
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
        # Preflight BEFORE the multi-minute volume split: the installer
        # scripts (Batch 20d, packaging\windows\installer\) must exist or
        # staging fails after the big archive job has already run.
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

          # r8 (CM-126, field 2026-08-28): the r7 raw-size shortcut assumed a
          # large dist could not compress under the 2GB asset limit; the ROCm
          # stack then landed in ONE under-limit volume, shipping a shepherd
          # exe + one part where a single-file installer would do. A single
          # volume IS the complete archive, so fold it back into the
          # one-file installer instead of guessing better.
          $foldedSingle = $false
          if ($parts.Count -eq 1 -and
              [math]::Round($parts[0].Length / 1MB) -le $limitMB) {
            $soloMB = [math]::Round($parts[0].Length / 1MB)
            Write-Host ("Split produced ONE volume ({0} MB <= {1} MB) -- folding back into the single-file installer (r8/CM-126)." -f $soloMB, $limitMB) -ForegroundColor Yellow
            $sfxGui = $null
            $sevenZCmd = Get-Command 7z -ErrorAction SilentlyContinue
            if ($sevenZCmd) {
              $cand = Join-Path (Split-Path $sevenZCmd.Source) "7z.sfx"
              if (Test-Path $cand) { $sfxGui = $cand }
            }
            if ($sfxGui) {
              $singleArc = "$installBase.7z"
              try {
                Move-Item -Force $parts[0].FullName $singleArc
                $outPath = Join-Path (Get-Location) "$installBase.exe"
                Remove-Item -Force -ErrorAction SilentlyContinue $outPath
                $outFs = [IO.File]::Create($outPath)
                try {
                  foreach ($pf in @($sfxGui, $singleArc)) {
                    $bytes = [IO.File]::ReadAllBytes((Resolve-Path $pf))
                    $outFs.Write($bytes, 0, $bytes.Length)
                  }
                } finally { $outFs.Close() }
                Remove-Item -Force $singleArc
                $exeMb = [math]::Round((Get-Item $outPath).Length / 1MB)
                Write-Host ""
                Write-Host ("SINGLE-FILE INSTALLER: {0}  ({1} MB)" -f "$installBase.exe", $exeMb) -ForegroundColor Green
                Write-Host "Release this ONE file. Users double-click it, pick a folder," -ForegroundColor Cyan
                Write-Host "and it extracts there -- same experience as the other editions." -ForegroundColor Cyan
                $foldedSingle = $true
              } catch {
                Write-Warning ("Fold-back failed: {0}. Shipping shepherd + volume instead." -f $_.Exception.Message)
                # never leave a partially-written exe posing as a deliverable
                Remove-Item -Force -ErrorAction SilentlyContinue "$installBase.exe"
                if (Test-Path $singleArc) {
                  Move-Item -Force $singleArc ($installBase + ".7z.001")
                }
                $parts = @(Get-ChildItem "$installBase.7z.0*" | Sort-Object Name)
              }
            } else {
              Write-Warning "7z.sfx not found next to 7z.exe (full 7-Zip install ships it) -- keeping shepherd + volume."
            }
          }
          if (-not $foldedSingle) {

          # Stage the installer payload; stamp the real part count into the
          # script so it can name any missing volume exactly.
          $stage = ".\build\installer_payload"
          Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $stage
          New-Item -ItemType Directory -Force -Path $stage | Out-Null
          Copy-Item (Join-Path $instSrcDir "install.cmd") $stage
          # Stamp part count AND base name (regexes match any prior stamped
          # value; basename stamping keeps this identical to the xpu/rocm
          # packagers and immune to a committed pre-stamped install.ps1).
          (Get-Content (Join-Path $instSrcDir "install.ps1")) `
            -replace '^\$ExpectedParts = \d+.*$', ('$ExpectedParts = {0}   # stamped by packager' -f $parts.Count) `
            -replace '^\$BaseName\s*=.*$', ('$BaseName      = "{0}"   # stamped by packager' -f $installBase) |
            Set-Content (Join-Path $stage "install.ps1")
          Copy-Item $sevenZr $stage

          # Tiny payload archive + SFX assembly: module + config + payload.
          $payload7z = ".\build\installer_payload.7z"
          Remove-Item -ErrorAction SilentlyContinue $payload7z
          7z a -t7z $payload7z "$stage\*"
          if ($LASTEXITCODE -eq 0) {
            # Assemble sfx-module + config + payload by DIRECT byte
            # concatenation. The previous `cmd /c copy /b ("a + b + c")
            # dest` passed the whole concat list as ONE quoted argument, so
            # cmd searched for a single file literally named "a + b + c" --
            # and its error message went to Out-Null. (First-ever end-to-end
            # run of this step caught it.) PowerShell owns the bytes now,
            # and failures say WHY.
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
              # r8: remove the partial shepherd exe (same corpse rule).
              Remove-Item -Force -ErrorAction SilentlyContinue $outPath
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
  }
} else {
  Write-Host "7z not found - skipping release artifact (zip dist\$Name by hand instead)." -ForegroundColor Gray
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Output: $distDir" -ForegroundColor Green
Write-Host "Run:    $cmdPath        (or $Name.exe)" -ForegroundColor Green
