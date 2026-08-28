# Get-ReleaseChecksums.ps1
# Emit SHA-256 checksums for release assets as a Markdown snippet ready to
# paste into a GitHub release description (plus a machine-readable
# checksums.txt block). Why: distribution is GitHub-only, but the audience
# includes regions where GitHub is slow or interfered with, so third-party
# mirrors and forum re-uploads WILL exist (AGPL permits them). Published
# checksums let anyone verify a mirrored copy is the authentic build --
# the cheap answer to the tweaked-builds ecosystem.
#
# Usage (any folder holding the release assets -- e.g. the patch cache):
#   powershell -ExecutionPolicy Bypass -File .\Get-ReleaseChecksums.ps1 `
#       -Folder .\.patch-cache\Release-1.60.00
#   powershell -ExecutionPolicy Bypass -File .\Get-ReleaseChecksums.ps1 `
#       -Folder D:\Downloads\rel -OutFile checksums-1.60.00.md
#
# Then: GitHub release page -> Edit -> paste the snippet at the bottom.
# Editing a release description never touches the assets or their hashes.
#
# PURE ASCII output; PowerShell 5.1 compatible.

param(
  [string]$Folder = ".",
  # Which files to hash. Default: ChitraMaya release artifacts only, so a
  # cache folder's extracted/ subdir or done.txt markers are never included.
  [string[]]$Include = @("ChitraMaya-install*", "ChitraMaya-*-install*", "ChitraMaya-patch-*.zip"),
  [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Folder)) { throw "Folder not found: $Folder" }

$files = @()
foreach ($pat in $Include) {
  $files += Get-ChildItem -Path $Folder -Filter $pat -File -ErrorAction SilentlyContinue
}
$files = $files | Sort-Object Name -Unique
if ($files.Count -eq 0) {
  throw "No release assets matching [$($Include -join ', ')] in $Folder"
}

Write-Host "Hashing $($files.Count) file(s) in $((Resolve-Path $Folder).Path) ..."
$rows = @()
foreach ($f in $files) {
  Write-Host "  $($f.Name) ($([math]::Round($f.Length / 1MB, 1)) MB)"
  $h = (Get-FileHash -Algorithm SHA256 -Path $f.FullName).Hash.ToLowerInvariant()
  $rows += [pscustomobject]@{ Name = $f.Name; Size = $f.Length; Sha = $h }
}

# -- Markdown snippet ----------------------------------------------------
$md = @()
$md += "## SHA-256 checksums"
$md += ""
$md += "Downloaded ChitraMaya from a mirror, a forum re-upload, or a friend?"
$md += "Verify you have the authentic build: in PowerShell run"
$md += '`Get-FileHash <file> -Algorithm SHA256` and compare against this table.'
$md += "If a hash does not match, do not run the file -- get it from this page."
$md += ""
$md += "| File | Size (bytes) | SHA-256 |"
$md += "|---|---|---|"
foreach ($r in $rows) {
  $md += "| $($r.Name) | $($r.Size) | ``$($r.Sha)`` |"
}
$md += ""
$md += "<details><summary>checksums.txt (sha256sum format)</summary>"
$md += ""
$md += '```'
foreach ($r in $rows) {
  # Two-space separator, sha256sum-compatible: <hash>  <name>
  $md += "$($r.Sha)  $($r.Name)"
}
$md += '```'
$md += "</details>"

$text = $md -join "`n"
Write-Host ""
Write-Host "==== COPY BELOW THIS LINE INTO THE RELEASE DESCRIPTION ===="
Write-Host $text
Write-Host "==== COPY ABOVE THIS LINE ===="

if ($OutFile) {
  Set-Content -Encoding ASCII -Path $OutFile -Value $text
  Write-Host ""
  Write-Host "Also written to: $OutFile"
}