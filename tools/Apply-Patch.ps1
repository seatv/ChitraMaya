# tools/Apply-Patch.ps1
# ChitraMaya patch apply script (CM-097, v1.60.00; GenSRT lineage).
# This file ships INSIDE every patch zip next to patch-manifest.json and
# payload\. It is kept as a real repo file (not a string literal) so the
# one script that overwrites and deletes files on a user's machine can be
# read, diffed, and reviewed. PURE ASCII ON PURPOSE: Windows PowerShell
# 5.1 decodes BOM-less non-ASCII .ps1 files as cp1252 and mangles them;
# make_patch.py refuses to embed this file if a non-ASCII byte sneaks in.
#
# What it does, in order, stopping at the first problem:
#   1. VERIFY  every file the patch would overwrite matches the release
#              this patch was built from (SHA-256). No partial applies.
#   2. BACKUP  every file it will overwrite or delete to backup-<from>\.
#   3. APPLY   copy payload files in, delete removed files, re-verify.
#   4. CHECK   run ChitraMaya-cli.exe -self-check and report.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\Apply-Patch.ps1 `
#             [-InstallDir "C:\Path\To\ChitraMaya"]
# If -InstallDir is omitted the script looks in the likely places (the
# patch folder itself, its parent, and ChitraMaya siblings of both) and
# asks if it cannot find the install.

param(
  [string]$InstallDir = "",
  [switch]$SkipSelfCheck = $false   # testing aid; the self-check is the
                                    # point of phase 4 -- do not skip it
                                    # on a real install
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $here "patch-manifest.json"
$payloadDir = Join-Path $here "payload"

if (-not (Test-Path $manifestPath)) { throw "patch-manifest.json not found next to this script. Extract the whole patch zip first." }
if (-not (Test-Path $payloadDir)) { throw "payload folder not found next to this script. Extract the whole patch zip first." }

$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
$fromV = if ($manifest.from_version) { $manifest.from_version } else { "unknown" }
$toV = if ($manifest.to_version) { $manifest.to_version } else { "unknown" }

Write-Host "=============================================================="
Write-Host "  ChitraMaya patch: $fromV  ->  $toV"
Write-Host "  files to update/add: $($manifest.files.Count)   deletions: $($manifest.deletions.Count)"
Write-Host "=============================================================="

# -- Locate the install ---------------------------------------------------
function Test-InstallDir([string]$p) {
  if (-not $p) { return $false }
  if (-not (Test-Path $p)) { return $false }
  return (Test-Path (Join-Path $p "ChitraMaya.exe")) -or (Test-Path (Join-Path $p "ChitraMaya-cli.exe"))
}

if (-not $InstallDir) {
  # GenSRT lesson: people unpack the patch as a SIBLING of the install at
  # least as often as inside it. Check both, plus ChitraMaya children.
  $parent = Split-Path -Parent $here
  $candidates = @(
    $here,
    $parent,
    (Join-Path $here "ChitraMaya"),
    (Join-Path $parent "ChitraMaya")
  )
  foreach ($c in $candidates) {
    if (Test-InstallDir $c) { $InstallDir = $c; break }
  }
}
while (-not (Test-InstallDir $InstallDir)) {
  Write-Host ""
  Write-Host "  Could not find the ChitraMaya install folder automatically."
  $InstallDir = Read-Host "  Enter the full path to the folder containing ChitraMaya.exe"
}
$InstallDir = (Resolve-Path $InstallDir).Path
Write-Host "  Install: $InstallDir"
Write-Host ""

# -- Refuse to run while the app is running (Windows locks the exe) -------
$running = Get-Process -Name "ChitraMaya", "ChitraMaya-cli" -ErrorAction SilentlyContinue
if ($running) {
  throw "ChitraMaya is running (PID $(($running | ForEach-Object Id) -join ', ')). Close it and run this script again."
}

function Get-Sha([string]$p) { (Get-FileHash -Algorithm SHA256 -Path $p).Hash.ToLowerInvariant() }

# -- Phase 1: VERIFY (touch nothing until everything checks out) ----------
Write-Host "  [1/4] Verifying the install matches release $fromV ..."
$mismatches = @()
$toApply = @()     # manifest entries that actually need writing
$skipped = 0
foreach ($e in $manifest.files) {
  $rel = $e.path -replace "/", "\"
  $target = Join-Path $InstallDir $rel
  $exists = Test-Path $target
  if ($exists) {
    $cur = Get-Sha $target
    if ($cur -eq $e.sha256_new) { $skipped += 1; continue }   # already patched (re-run safe)
    if ($e.action -eq "update") {
      if ($cur -ne $e.sha256_old) { $mismatches += $rel; continue }
    } else {
      $mismatches += $rel; continue    # 'add' target exists with foreign content
    }
  } else {
    if ($e.action -eq "update") { $mismatches += $rel; continue }
  }
  $toApply += $e
}
if ($mismatches.Count -gt 0) {
  Write-Host ""
  Write-Host "  PATCH REFUSED. These files do not match release ${fromV}:"
  $mismatches | ForEach-Object { Write-Host "    $_" }
  Write-Host ""
  Write-Host "  This install is not the release the patch was built from"
  Write-Host "  (or was modified). Nothing was changed. Download the full"
  Write-Host "  installer for $toV instead."
  exit 1
}
Write-Host "        OK ($($toApply.Count) to write, $skipped already current)."

# -- Phase 2: BACKUP ------------------------------------------------------
$backupDir = Join-Path $InstallDir ("backup-" + $fromV)
Write-Host "  [2/4] Backing up files to be changed -> $backupDir"
$toBackup = @()
foreach ($e in $toApply) {
  $rel = $e.path -replace "/", "\"
  if (Test-Path (Join-Path $InstallDir $rel)) { $toBackup += $rel }
}
foreach ($d in $manifest.deletions) {
  $rel = $d -replace "/", "\"
  if (Test-Path (Join-Path $InstallDir $rel)) { $toBackup += $rel }
}
foreach ($rel in $toBackup) {
  $src = Join-Path $InstallDir $rel
  $dst = Join-Path $backupDir $rel
  $dstDir = Split-Path -Parent $dst
  if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
  Copy-Item -Force $src $dst
}
Write-Host "        OK ($($toBackup.Count) files backed up)."

# -- Phase 3: APPLY -------------------------------------------------------
Write-Host "  [3/4] Applying ..."
foreach ($e in $toApply) {
  $rel = $e.path -replace "/", "\"
  $src = Join-Path $payloadDir $rel
  $dst = Join-Path $InstallDir $rel
  if (-not (Test-Path $src)) { throw "Patch payload is missing $rel -- the zip is incomplete. Nothing further was changed; restore from $backupDir if needed." }
  $dstDir = Split-Path -Parent $dst
  if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
  Copy-Item -Force $src $dst
  $got = Get-Sha $dst
  if ($got -ne $e.sha256_new) { throw "Post-copy hash mismatch on $rel. Restore from $backupDir and re-download the patch." }
}
$deleted = 0
foreach ($d in $manifest.deletions) {
  $rel = $d -replace "/", "\"
  $target = Join-Path $InstallDir $rel
  if (Test-Path $target) { Remove-Item -Force $target; $deleted += 1 }
}
Write-Host "        OK ($($toApply.Count) written, $deleted deleted)."

# -- Phase 4: SELF-CHECK --------------------------------------------------
$cli = Join-Path $InstallDir "ChitraMaya-cli.exe"
if ($SkipSelfCheck) {
  Write-Host "  [4/4] Self-check SKIPPED (-SkipSelfCheck)."
} elseif (Test-Path $cli) {
  Write-Host "  [4/4] Running self-check ..."
  $selfCheckOk = $false
  try {
    & $cli -self-check
    if ($LASTEXITCODE -eq 0) { $selfCheckOk = $true }
  } catch {
    Write-Host "  Self-check could not run: $($_.Exception.Message)"
  }
  if (-not $selfCheckOk) {
    Write-Host ""
    Write-Host "  SELF-CHECK FAILED after patching."
    Write-Host "  Your pre-patch files are preserved in: $backupDir"
    Write-Host "  To roll back: copy them back over the install, or"
    Write-Host "  re-download the full $fromV installer."
    exit 1
  }
  Write-Host "        Self-check passed."
} else {
  Write-Host "  [4/4] ChitraMaya-cli.exe not found -- skipping self-check."
}

Write-Host ""
Write-Host "  DONE. ChitraMaya is now $toV."
Write-Host "  Backup of replaced files: $backupDir"
Write-Host "  (Delete that folder once you are happy with the update.)"
Write-Host "  Note: your models, compiled engines, and settings were not"
Write-Host "  touched -- patches never modify the models folder or config."
