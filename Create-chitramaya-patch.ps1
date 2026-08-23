# Create-chitramaya-patch.ps1
# CM-097 (v1.60.00, GenSRT lineage): orchestrate building a patch zip
# against a PUBLISHED GitHub release. Downloads the released installer
# (cached), extracts it, extracts/locates the new build, and runs
# tools/make_patch.py on the two trees. It then STOPS and prints the
# commands to test the patch. This script publishes nothing.
#
# The old side always comes from the DOWNLOADED released installer -- the
# artifact users actually have -- never from a local dist/ tree.
#
# Usage (repo root, venv active so `python` resolves):
#   .\Create-chitramaya-patch.ps1 -NewDist .\dist\ChitraMaya
#       diff the latest GitHub release against a freshly built dist tree
#   .\Create-chitramaya-patch.ps1 -Tag Release-1.50.00 -NewDist D:\path\to\new\installer\folder
#       explicit base release; -NewDist may be an extracted tree OR a
#       folder holding the new ChitraMaya-install.* assets

param(
  [Parameter(Mandatory = $true)]
  [string]$NewDist,
  [string]$Tag = "",
  [string]$Repo = "seatv/ChitraMaya",
  [string]$CacheDir = ".\.patch-cache",
  [string]$OutDir = ".",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

# r2 (field, 2026-08-18): validate EVERYTHING before the multi-GB
# download -- the first field run spent the whole download only to fail
# on a -NewDist that did not exist yet (dist had not been built).
# Preflight-before-the-expensive-step, same lesson as the installer
# volume split.
if (-not (Get-Command 7z -ErrorAction SilentlyContinue)) {
  throw "7z not found on PATH -- needed to extract installers."
}
if (-not (Test-Path ".\tools\make_patch.py")) {
  throw "Run from the repo root (tools\make_patch.py not found)."
}
if (-not (Test-Path $NewDist)) {
  throw "-NewDist '$NewDist' does not exist. Build the new dist first (run the packager), then re-run. Nothing was downloaded."
}

# -- Resolve the release via the GitHub API ------------------------------
$apiBase = "https://api.github.com/repos/$Repo/releases"
$apiUrl = if ($Tag) { "$apiBase/tags/$Tag" } else { "$apiBase/latest" }
Write-Host "Querying $apiUrl ..."
$rel = Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "ChitraMaya-patcher" }
$relTag = $rel.tag_name
Write-Host "Base release: $relTag  (published $($rel.published_at))"

$assets = @($rel.assets | Where-Object { $_.name -like "ChitraMaya-install*" })
if ($assets.Count -eq 0) {
  throw "Release $relTag has no ChitraMaya-install* assets."
}

# -- Download assets (cached; a 2.9 GB re-fetch per run is the failure
#    mode this cache exists to prevent) -----------------------------------
$relCache = Join-Path $CacheDir $relTag
New-Item -ItemType Directory -Force -Path $relCache | Out-Null
foreach ($a in $assets) {
  $dst = Join-Path $relCache $a.name
  if ((Test-Path $dst) -and ((Get-Item $dst).Length -eq $a.size)) {
    Write-Host "  cached:      $($a.name)  ($([math]::Round($a.size / 1MB)) MB)"
  } else {
    Write-Host "  downloading: $($a.name)  ($([math]::Round($a.size / 1MB)) MB)"
    Invoke-WebRequest -Uri $a.browser_download_url -OutFile $dst -Headers @{ "User-Agent" = "ChitraMaya-patcher" }
  }
}

# -- Extract the released installer (cached) ------------------------------
function Get-ExtractedTree([string]$srcDir, [string]$extractDir) {
  # Prefer the multivolume archive (the installer exe is only the shepherd
  # for split releases); fall back to a single .7z, then a single-file SFX
  # exe (stub + archive concatenated -- 7z opens it directly).
  if (-not (Test-Path (Join-Path $extractDir "done.txt"))) {
    $vol = Get-ChildItem -Path $srcDir -Filter "ChitraMaya-install*.7z.001" | Select-Object -First 1
    $sevenZ = Get-ChildItem -Path $srcDir -Filter "ChitraMaya-install*.7z" | Select-Object -First 1
    $exe = Get-ChildItem -Path $srcDir -Filter "ChitraMaya-install*.exe" |
      Sort-Object Length -Descending | Select-Object -First 1
    $archive = if ($vol) { $vol.FullName }
               elseif ($sevenZ) { $sevenZ.FullName }
               elseif ($exe -and $exe.Length -gt 100MB) { $exe.FullName }
               else { $null }
    if (-not $archive) {
      throw "No extractable installer archive found in $srcDir (need .7z.001, .7z, or a single-file SFX exe)."
    }
    Write-Host "  extracting:  $(Split-Path -Leaf $archive) -> $extractDir"
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    & 7z x -y "-o$extractDir" $archive | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z extraction failed for $archive." }
    Set-Content -Encoding ASCII (Join-Path $extractDir "done.txt") "ok"
  }
  # The archive contains a ChitraMaya\ folder; tolerate flat layouts too.
  $sub = Join-Path $extractDir "ChitraMaya"
  if ((Test-Path (Join-Path $sub "ChitraMaya.exe")) -or (Test-Path (Join-Path $sub "ChitraMaya-cli.exe"))) { return $sub }
  if ((Test-Path (Join-Path $extractDir "ChitraMaya.exe")) -or (Test-Path (Join-Path $extractDir "ChitraMaya-cli.exe"))) { return $extractDir }
  throw "Extracted $extractDir but found no ChitraMaya.exe in it or in ChitraMaya\."
}

$oldTree = Get-ExtractedTree $relCache (Join-Path $relCache "extracted")
Write-Host "Old tree: $oldTree"

# -- Resolve the new tree -------------------------------------------------
$NewDist = (Resolve-Path $NewDist).Path
$newTree = $null
# r3: accept the dist PARENT too (field event 2026-08-22: -NewDist dist
# with the tree at dist\ChitraMaya fell through to the extraction path).
if (-not ((Test-Path (Join-Path $NewDist "ChitraMaya.exe")) -or (Test-Path (Join-Path $NewDist "ChitraMaya-cli.exe")))) {
  $childTree = Join-Path $NewDist "ChitraMaya"
  if ((Test-Path (Join-Path $childTree "ChitraMaya.exe")) -or (Test-Path (Join-Path $childTree "ChitraMaya-cli.exe"))) {
    $NewDist = $childTree
  }
}
if ((Test-Path (Join-Path $NewDist "ChitraMaya.exe")) -or (Test-Path (Join-Path $NewDist "ChitraMaya-cli.exe"))) {
  $newTree = $NewDist
  Write-Host "New tree: $newTree (already extracted)"
  Write-Host "  NOTE: for the FINAL published patch, prefer extracting the"
  Write-Host "  actual new installer artifact rather than dist\ -- the"
  Write-Host "  released archive is what users will have."
} else {
  # r3 (field event 2026-08-22): NEVER reuse the new-side extraction cache.
  # The done.txt short-circuit is correct for the OLD side (published
  # releases are immutable) but a stale-cache landmine here: the rehearsal
  # build's tree survived into the 1.60.01 patch attempt and make_patch's
  # same-version guard fired against LAST build's files. The new artifact
  # changes every build -- extract it fresh every time.
  $newCache = Join-Path $CacheDir "new-extracted"
  if (Test-Path $newCache) {
    Write-Host "  clearing stale new-side extraction cache: $newCache"
    Remove-Item -Recurse -Force $newCache
  }
  $newTree = Get-ExtractedTree $NewDist $newCache
  Write-Host "New tree: $newTree"
}

# -- Diff -----------------------------------------------------------------
$fromV = if (Test-Path (Join-Path $oldTree "VERSION.txt")) { (Get-Content (Join-Path $oldTree "VERSION.txt") -First 1).Trim() } else { $relTag -replace "^Release-", "" }
$toV = if (Test-Path (Join-Path $newTree "VERSION.txt")) { (Get-Content (Join-Path $newTree "VERSION.txt") -First 1).Trim() } else { "" }
$outZip = Join-Path $OutDir ("ChitraMaya-patch-" + $fromV + "-to-" + $(if ($toV) { $toV } else { "new" }) + ".zip")

$mpArgs = @(".\tools\make_patch.py", "--old", $oldTree, "--new", $newTree, "--out", $outZip, "--from-version", $fromV)
if ($toV) { $mpArgs += @("--to-version", $toV) }
Write-Host ""
& $Python @mpArgs
if ($LASTEXITCODE -ne 0) { throw "make_patch.py failed." }

Write-Host ""
Write-Host "=============================================================="
Write-Host "  Patch built: $outZip"
Write-Host "  NOTHING WAS PUBLISHED. Test it first:"
Write-Host "    1. Copy the extracted old tree somewhere disposable:"
Write-Host "       robocopy `"$oldTree`" C:\Temp\CM-patch-test\ChitraMaya /E"
Write-Host "    2. Extract the patch zip next to it and run:"
Write-Host "       powershell -ExecutionPolicy Bypass -File .\Apply-Patch.ps1 -InstallDir C:\Temp\CM-patch-test\ChitraMaya"
Write-Host "    3. Confirm the self-check passes at the end."
Write-Host "  Then upload the patch zip to the $(if ($toV) { $toV } else { 'new' }) release page"
Write-Host "  alongside the full installer, and append the version to"
Write-Host "  packaging\windows\released-versions.txt at publish."
Write-Host "=============================================================="