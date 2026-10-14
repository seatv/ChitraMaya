# Upload-Release.ps1
# Publish a ChitraMaya release to the download host (Cloudflare R2 behind
# https://get.chitramaya.org) with rclone, and (re)write the site's front
# page from what was uploaded.
#
# What it does, in order:
#   1. Finds the release assets in -Folder (installer exe + .7z volumes for
#      each edition, patch zips), hashes them, writes SHA256SUMS.txt (sha256sum
#      format) and checksums-<ver>.md (the Markdown table for the GitLab
#      release description) into -Folder.
#   2. rclone copy  -Folder  ->  <remote>/releases/<ver>/   (assets + SHA256SUMS)
#      then rclone check (size) so a half-uploaded volume is never left silent.
#   3. Copies the shipped guides and the changelog to the site root:
#        docs\UserGuide.html    -> userguide.html
#        docs\ScalingGuide.html -> scalingguide.html
#        CHANGELOG.md           -> changelog.md
#   4. Generates index.html (editions, files, sizes, SHA-256, verify line,
#      mainland line, links) and latest.json, uploads both to the root.
#   5. Prints the public URLs to click-check.
#
# Usage (from the repo root, after the three packagers have run):
#   powershell -ExecutionPolicy Bypass -File .\Upload-Release.ps1 `
#       -Version 1.80.00 -Folder D:\Release\1.80.00
#   ... -DryRun          # hash + generate index.html, upload nothing
#   ... -SkipAssets      # only refresh index.html / guides / changelog
#   ... -IndexOnly       # regenerate + upload index.html alone (fix a typo)
#
# Needs: rclone on PATH with the remote configured (default r2:chitramaya,
# see -Remote); the bucket-scoped token can list and write the bucket but not
# list buckets, which is why this script never calls 'rclone lsd r2:'.
# The URLs are built for the custom domain; nothing here touches DNS.
#
# Hosting policy: releases/<ver>/ keeps current + previous; the Internet
# Archive copy of older releases is a separate, manual step.
#
# PURE ASCII; PowerShell 5.1 compatible.

param(
  [Parameter(Mandatory = $true)][string]$Version,
  [Parameter(Mandatory = $true)][string]$Folder,
  [string]$RepoRoot = $PSScriptRoot,
  [string]$Remote = "r2:chitramaya",
  [string]$PublicBase = "https://get.chitramaya.org",
  [string[]]$Include = @("ChitraMaya-install*", "ChitraMaya-*-install*", "ChitraMaya-patch-*.zip"),
  [switch]$DryRun,
  [switch]$SkipAssets,
  [switch]$IndexOnly
)

$ErrorActionPreference = "Stop"
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version must look like 1.80.00 (got '$Version')" }
if (-not (Test-Path $Folder)) { throw "Folder not found: $Folder" }
$Folder = (Resolve-Path $Folder).Path
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
$Remote = $Remote.TrimEnd('/')
$PublicBase = $PublicBase.TrimEnd('/')
$relPrefix = "releases/$Version"

function Invoke-Rclone {
  param([string[]]$CliArgs, [switch]$AllowFail)
  $line = "rclone " + ($CliArgs -join ' ')
  if ($DryRun) { Write-Host "  [dry-run] $line"; return 0 }
  Write-Host "  > $line"
  & rclone @CliArgs
  $rc = $LASTEXITCODE
  if ($rc -ne 0 -and -not $AllowFail) { throw "rclone failed (exit $rc): $line" }
  return $rc
}

function Format-Size {
  param([long]$Bytes)
  if ($Bytes -ge 1GB) { return ("{0:N2} GB" -f ($Bytes / 1GB)) }
  if ($Bytes -ge 1MB) { return ("{0:N1} MB" -f ($Bytes / 1MB)) }
  return ("{0:N0} KB" -f ($Bytes / 1KB))
}

function Html-Escape { param([string]$s) return $s.Replace('&', '&amp;').Replace('<', '&lt;').Replace('>', '&gt;').Replace('"', '&quot;') }

# -- 0. tools ----------------------------------------------------------------
if (-not $DryRun) {
  if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    throw "rclone is not on PATH (winget install Rclone.Rclone, then configure the remote '$Remote')"
  }
  Write-Host "Checking the remote ($Remote) ..."
  & rclone lsf $Remote --max-depth 1 | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "Cannot list $Remote - check %APPDATA%\rclone\rclone.conf and the token" }
}

# -- 1. assets + checksums ---------------------------------------------------
$files = @()
foreach ($pat in $Include) {
  $files += Get-ChildItem -Path $Folder -Filter $pat -File -ErrorAction SilentlyContinue
}
$files = $files | Sort-Object Name -Unique
if ($files.Count -eq 0 -and -not $IndexOnly) {
  throw "No release assets matching [$($Include -join ', ')] in $Folder"
}

$rows = @()
if ($files.Count -gt 0) {
  Write-Host "Hashing $($files.Count) file(s) in $Folder ..."
  foreach ($f in $files) {
    Write-Host ("  {0}  ({1})" -f $f.Name, (Format-Size $f.Length))
    $h = (Get-FileHash -Algorithm SHA256 -Path $f.FullName).Hash.ToLowerInvariant()
    $rows += [pscustomobject]@{ Name = $f.Name; Size = $f.Length; Sha = $h }
  }
  $sums = @()
  foreach ($r in $rows) { $sums += "$($r.Sha)  $($r.Name)" }
  Set-Content -Encoding ASCII -Path (Join-Path $Folder "SHA256SUMS.txt") -Value ($sums -join "`n")

  $md = @()
  $md += "## SHA-256 checksums"
  $md += ""
  $md += "Downloaded ChitraMaya from a mirror, a forum re-upload, or a friend?"
  $md += "Verify you have the authentic build: in PowerShell run"
  $md += '`Get-FileHash <file> -Algorithm SHA256` and compare against this table.'
  $md += "If a hash does not match, do not run the file -- get it from $PublicBase."
  $md += ""
  $md += "| File | Size (bytes) | SHA-256 |"
  $md += "|---|---|---|"
  foreach ($r in $rows) { $md += "| [$($r.Name)]($PublicBase/$relPrefix/$($r.Name)) | $($r.Size) | ``$($r.Sha)`` |" }
  Set-Content -Encoding ASCII -Path (Join-Path $Folder "checksums-$Version.md") -Value ($md -join "`n")
  Write-Host "  wrote SHA256SUMS.txt and checksums-$Version.md"
}

# -- 2. upload assets --------------------------------------------------------
if (-not $SkipAssets -and -not $IndexOnly -and $rows.Count -gt 0) {
  Write-Host "Uploading assets to $Remote/$relPrefix/ ..."
  $incArgs = @()
  foreach ($pat in $Include) { $incArgs += "--include"; $incArgs += $pat }
  $incArgs += "--include"; $incArgs += "SHA256SUMS.txt"
  Invoke-Rclone (@("copy", $Folder, "$Remote/$relPrefix/", "--progress", "--transfers", "2", "--s3-chunk-size", "64M") + $incArgs) | Out-Null
  Write-Host "Verifying sizes on the remote ..."
  Invoke-Rclone (@("check", $Folder, "$Remote/$relPrefix/", "--size-only", "--one-way") + $incArgs) | Out-Null
}

# -- 3. guides + changelog at the root --------------------------------------
if (-not $IndexOnly) {
  $siteFiles = @(
    @{ Src = (Join-Path (Join-Path $RepoRoot "docs") "UserGuide.html");    Dst = "userguide.html" },
    @{ Src = (Join-Path (Join-Path $RepoRoot "docs") "ScalingGuide.html"); Dst = "scalingguide.html" },
    @{ Src = (Join-Path $RepoRoot "CHANGELOG.md");                         Dst = "changelog.md" }
  )
  foreach ($sf in $siteFiles) {
    if (-not (Test-Path $sf.Src)) { Write-Warning "missing $($sf.Src) - skipped (run tools/build_userguide.py?)"; continue }
    Invoke-Rclone @("copyto", $sf.Src, "$Remote/$($sf.Dst)") | Out-Null
  }
}

# -- 4. index.html + latest.json --------------------------------------------
# Edition grouping by file name prefix.
function Get-Edition {
  param([string]$name)
  if ($name -like "ChitraMaya-rocm-install*") { return "amd" }
  if ($name -like "ChitraMaya-xpu-install*")  { return "intel" }
  if ($name -like "ChitraMaya-patch-*")       { return "patch" }
  if ($name -like "ChitraMaya-install*")      { return "nvidia" }
  return "other"
}

# If the assets were uploaded earlier (-IndexOnly / -SkipAssets without files
# present), rebuild the table from SHA256SUMS.txt + sizes in the folder.
if ($rows.Count -eq 0) {
  $sumsPath = Join-Path $Folder "SHA256SUMS.txt"
  if (-not (Test-Path $sumsPath)) { throw "No assets and no SHA256SUMS.txt in $Folder - nothing to build the page from" }
  foreach ($line in (Get-Content $sumsPath)) {
    if ($line -match '^([0-9a-f]{64})\s{2}(.+)$') {
      $n = $Matches[2]; $p = Join-Path $Folder $n
      $sz = if (Test-Path $p) { (Get-Item $p).Length } else { 0 }
      $rows += [pscustomobject]@{ Name = $n; Size = $sz; Sha = $Matches[1] }
    }
  }
}

$groups = @{ nvidia = @(); amd = @(); intel = @(); patch = @(); other = @() }
foreach ($r in $rows) { $groups[(Get-Edition $r.Name)] += $r }

function Table-Rows {
  param($list)
  $out = @()
  foreach ($r in $list) {
    $url = "$relPrefix/$(Html-Escape $r.Name)"
    $out += ("<tr><td><a href=""{0}"">{1}</a></td><td class=""num"">{2}</td><td><code class=""sha"">{3}</code></td></tr>" -f $url, (Html-Escape $r.Name), (Format-Size $r.Size), $r.Sha)
  }
  return ($out -join "`n")
}

$date = Get-Date -Format "yyyy-MM-dd"
$nvRows = Table-Rows $groups.nvidia
$amdRows = Table-Rows $groups.amd
$intelRows = Table-Rows $groups.intel
$patchRows = Table-Rows $groups.patch
$nvCount = @($groups.nvidia).Count

$nvNote = ""
if ($nvCount -gt 1) {
  $nvNote = @"
<div class="warn"><strong>The NVIDIA installer is $nvCount files and you need all of them.</strong>
The <code>.exe</code> by itself is only the unpacker. Download every file in this table into the <em>same</em> folder, then run <code>ChitraMaya-install.exe</code>.
If the install fails at once, one of the parts is missing or did not finish downloading.</div>
"@
}
$patchSection = ""
if (@($groups.patch).Count -gt 0) {
  $patchSection = @"
<h2 id="patch">Already on the previous version? Patch instead</h2>
<p>A patch zip updates an existing NVIDIA install without the multi-GB download. Pick the one whose <em>from</em> version matches <code>VERSION.txt</code> next to your exe, extract it anywhere, run <code>Apply-Patch.ps1</code> and point it at your install folder. It verifies every file before touching anything.</p>
<table><thead><tr><th>File</th><th>Size</th><th>SHA-256</th></tr></thead><tbody>
$patchRows
</tbody></table>
"@
}

$html = @"
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChitraMaya downloads</title>
<meta name="description" content="Download ChitraMaya $Version - mosaic restoration for NVIDIA, AMD Radeon and Intel Arc GPUs. Every file with its SHA-256.">
<style>
:root { --fg:#1f2328; --bg:#ffffff; --muted:#59636e; --line:#d1d9e0; --code:#f6f8fa; --link:#0969da; --warn:#fff8c5; --warnline:#d4a72c; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --bg:#0d1117; --muted:#9198a1; --line:#30363d; --code:#161b22; --link:#4493f8; --warn:#3b2e00; --warnline:#9e6a03; }
}
html { -webkit-text-size-adjust: 100%; }
body { margin:0; padding:0 16px 48px; background:var(--bg); color:var(--fg);
       font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 2em; margin: 1.2em 0 .2em; }
h2 { font-size: 1.4em; margin: 1.8em 0 .5em; border-bottom: 1px solid var(--line); padding-bottom: .3em; }
h3 { font-size: 1.1em; margin: 1.2em 0 .3em; }
p, li { max-width: 76ch; }
a { color: var(--link); }
code { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: .92em; background: var(--code); padding: .1em .35em; border-radius: 4px; }
table { border-collapse: collapse; margin: .6em 0 1em; display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; white-space: nowrap; }
th { background: var(--code); }
td.num { text-align: right; }
code.sha { font-size: .78em; background: none; padding: 0; color: var(--muted); }
.lead { color: var(--muted); margin-top: 0; }
.warn { background: var(--warn); border-left: 4px solid var(--warnline); padding: .6em 1em; margin: .8em 0; max-width: 76ch; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 4px 18px 10px; margin: 1em 0; }
.foot { color: var(--muted); font-size: .85em; margin-top: 3em; border-top: 1px solid var(--line); padding-top: .6em; }
nav a { margin-right: 1.2em; }
</style>
</head>
<body>
<main>
<h1>ChitraMaya</h1>
<p class="lead">Mosaic restoration for video, on your own GPU. Current release <strong>$Version</strong> ($date). Free software (AGPL-3.0).</p>
<nav><a href="#nvidia">NVIDIA</a> <a href="#amd">AMD Radeon</a> <a href="#intel">Intel Arc</a> <a href="userguide.html">User Guide</a> <a href="scalingguide.html">Scaling Guide</a> <a href="changelog.md">Changelog</a> <a href="https://gitlab.com/seatv/chitramaya">Source &amp; issues</a></nav>

<h2>Which edition?</h2>
<p>One download per GPU vendor; the editions do not mix. <strong>NVIDIA</strong> RTX 30/40/50-series (TensorRT engines, NVDEC/NVENC, RTX Super-Res). <strong>AMD Radeon</strong> RX 7000 / RX 9000 (ROCm; Adrenalin 26.2.2 or newer). <strong>Intel Arc</strong> discrete cards and Core Ultra laptop iGPUs such as the Arc 140V (Resizable BAR effectively required on discrete cards). Install into a <em>new, empty</em> folder; never extract a new version over an old one. First steps, settings and troubleshooting are in the <a href="userguide.html">User Guide</a>, which is also installed next to the app (the <strong>Help</strong> button).</p>

<div class="card" id="nvidia">
<h2>NVIDIA edition</h2>
$nvNote
<table><thead><tr><th>File</th><th>Size</th><th>SHA-256</th></tr></thead><tbody>
$nvRows
</tbody></table>
</div>

<div class="card" id="amd">
<h2>AMD Radeon (ROCm) edition</h2>
<p>Single installer (if more than one file is listed, download them all into one folder). Experimental: tested in-house on an RX 9060 XT; please report what you find.</p>
<table><thead><tr><th>File</th><th>Size</th><th>SHA-256</th></tr></thead><tbody>
$amdRows
</tbody></table>
</div>

<div class="card" id="intel">
<h2>Intel Arc (XPU) edition</h2>
<p>Single installer (if more than one file is listed, download them all into one folder). Experimental: tested in-house on an Arc 140V laptop; please report what you find.</p>
<table><thead><tr><th>File</th><th>Size</th><th>SHA-256</th></tr></thead><tbody>
$intelRows
</tbody></table>
</div>

$patchSection

<h2 id="verify">Verify what you downloaded</h2>
<p>Got the files from a mirror, a forum re-upload or a friend? Check them against the hashes above: in PowerShell, <code>Get-FileHash &lt;file&gt; -Algorithm SHA256</code>. If a hash does not match, do not run the file; download it from this page. All hashes are also in <a href="$relPrefix/SHA256SUMS.txt">SHA256SUMS.txt</a>.</p>

<h2 id="mainland">Mainland China</h2>
<p>This host is reachable from the mainland but can be slow. Inside the app, Manage Models can fetch the models from <code>hf-mirror.com</code> (it falls back to the mirror on its own when huggingface.co is unreachable), and the interface is available in Simplified and Traditional Chinese.</p>

<h2 id="more">More</h2>
<p>Release notes for every version are on the <a href="https://gitlab.com/seatv/chitramaya/-/releases">GitLab Releases page</a>; the short history is in the <a href="changelog.md">changelog</a>. Bugs and questions: <a href="https://gitlab.com/seatv/chitramaya/-/issues">issues</a>. Older releases stay on the GitHub mirrors' release pages while they exist.</p>

<p class="foot">ChitraMaya is free software under the AGPL-3.0; we take no money. If it is useful to you, consider giving to Save the Children.</p>
</main>
</body>
</html>
"@

$indexPath = Join-Path $Folder "index.html"
Set-Content -Encoding UTF8 -Path $indexPath -Value $html
Write-Host "  wrote $indexPath"

$latest = [ordered]@{
  version = $Version
  date = $date
  base = "$PublicBase/$relPrefix/"
  files = @()
}
foreach ($r in $rows) { $latest.files += [ordered]@{ name = $r.Name; edition = (Get-Edition $r.Name); size = $r.Size; sha256 = $r.Sha } }
$latestPath = Join-Path $Folder "latest.json"
Set-Content -Encoding ASCII -Path $latestPath -Value ($latest | ConvertTo-Json -Depth 4)
Write-Host "  wrote $latestPath"

Invoke-Rclone @("copyto", $indexPath, "$Remote/index.html") | Out-Null
Invoke-Rclone @("copyto", $latestPath, "$Remote/latest.json") | Out-Null

# -- 5. what to click --------------------------------------------------------
Write-Host ""
Write-Host "Click-check (the sandbox cannot reach this host; these are yours):"
Write-Host "  $PublicBase/"
Write-Host "  $PublicBase/userguide.html"
Write-Host "  $PublicBase/scalingguide.html"
Write-Host "  $PublicBase/$relPrefix/SHA256SUMS.txt"
foreach ($r in $rows) { Write-Host "  $PublicBase/$relPrefix/$($r.Name)" }
if ($DryRun) { Write-Host ""; Write-Host "DRY RUN: nothing was uploaded. index.html / latest.json / SHA256SUMS.txt were written to $Folder for review." }
