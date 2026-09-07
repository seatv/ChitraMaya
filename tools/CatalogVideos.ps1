# tools/CatalogVideos.ps1
# ChitraMaya utility (Phase C prep): catalog every video file across drives
# into a CSV, flagging "UR" (un-rated / uncensored) paths as pristine
# training candidates. Pure ASCII. Windows PowerShell 5.1+ or PowerShell 7+.
#
# Usage:
#   .\CatalogVideos.ps1
#       -> scans every ready fixed + removable drive; writes
#          .\video-inventory-<host>-<timestamp>.csv
#   .\CatalogVideos.ps1 -Drives D,E,F
#   .\CatalogVideos.ps1 -Drives E -OutFile E:\pristine-inv.csv
#   .\CatalogVideos.ps1 -Extensions mp4,ts,mkv,wmv,avi -URSegments UR,Uncensored
#
# The CSV it produces contains REAL file names (may be NSFW-coded). It is a
# LOCAL inventory only -- keep it on your machine. Doctrine holds: weights
# travel, content never does. This script never uploads or moves anything;
# it only reads directory listings and file sizes.
#
# Notes:
#   - Access-denied folders (System Volume Information, $RECYCLE.BIN, etc.)
#     are skipped silently.
#   - Symlinks/junctions are NOT followed (avoids scan loops).
#   - Enable Windows long-path support for paths > 260 chars if you have any.

[CmdletBinding()]
param(
    [string[]]$Drives,
    [string[]]$Extensions = @('mp4', 'ts', 'mkv'),
    [string[]]$URSegments = @('UR'),
    [string]$OutFile
)

$ErrorActionPreference = 'Stop'

# --- Normalize inputs -------------------------------------------------------
$exts = @()
foreach ($e in $Extensions) { $exts += $e.TrimStart('.').ToLower() }

$urSet = New-Object 'System.Collections.Generic.HashSet[string]' (
    [System.StringComparer]::OrdinalIgnoreCase)
foreach ($s in $URSegments) { [void]$urSet.Add($s.Trim()) }

# --- Resolve the drive roots to scan ---------------------------------------
$driveInfo = @{}   # "D:" -> volume label
$roots = @()
if ($Drives) {
    foreach ($d in $Drives) {
        $letter = $d.TrimEnd('\', ':').ToUpper()
        $roots += ("{0}:\" -f $letter)
    }
} else {
    foreach ($di in [System.IO.DriveInfo]::GetDrives()) {
        if ($di.IsReady -and
            ($di.DriveType -eq 'Fixed' -or $di.DriveType -eq 'Removable')) {
            $roots += $di.RootDirectory.FullName
        }
    }
}
# Capture volume labels for every ready drive (identifies external disks).
foreach ($di in [System.IO.DriveInfo]::GetDrives()) {
    if ($di.IsReady) {
        $key = $di.Name.TrimEnd('\')          # e.g. "D:"
        $lbl = ''
        try { $lbl = [string]$di.VolumeLabel } catch { $lbl = '' }
        $driveInfo[$key] = $lbl
    }
}

if (-not $OutFile) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutFile = Join-Path (Get-Location) ("video-inventory-{0}-{1}.csv" -f $env:COMPUTERNAME, $stamp)
}

$scanUtc = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
Write-Host ("[catalog] host={0}  drives={1}  ext={2}  UR={3}" -f `
    $env:COMPUTERNAME, ($roots -join ' '), ($exts -join ','), ($URSegments -join ','))
Write-Host ("[catalog] output: {0}" -f $OutFile)

# --- Scan -------------------------------------------------------------------
$rows = New-Object 'System.Collections.Generic.List[object]'
[long]$totalBytes = 0
$count = 0
$urCount = 0

foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root)) {
        Write-Host ("[catalog] skip (not ready): {0}" -f $root)
        continue
    }
    $driveLetter = $root.Substring(0, 2)      # "D:"
    $label = ''
    if ($driveInfo.ContainsKey($driveLetter)) { $label = $driveInfo[$driveLetter] }
    Write-Host ("[catalog] scanning {0}  [{1}]" -f $root, $label)

    foreach ($ext in $exts) {
        # -Filter is applied by the filesystem (fast on huge trees); the
        # post-check guards the classic 8.3 "*.ts matches *.tsx" overmatch.
        Get-ChildItem -LiteralPath $root -Recurse -File -Force `
            -Filter ("*.{0}" -f $ext) -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Extension.TrimStart('.').ToLower() -ne $ext) { return }

            $full = $_.FullName
            $segments = $full.Split('\')
            $isUr = $false
            $urSeg = ''
            foreach ($seg in $segments) {
                if ($urSet.Contains($seg)) { $isUr = $true; $urSeg = $seg; break }
            }

            # Top folder = first directory under the drive root, if any.
            $topFolder = ''
            if ($segments.Length -ge 2) { $topFolder = $segments[1] }

            $sizeBytes = [int64]$_.Length
            $totalBytes += $sizeBytes
            $count++
            if ($isUr) { $urCount++ }

            $rows.Add([pscustomobject][ordered]@{
                ComputerName = $env:COMPUTERNAME
                ScanUtc      = $scanUtc
                Drive        = $driveLetter
                VolumeLabel  = $label
                TopFolder    = $topFolder
                FileName     = $_.Name
                Extension    = $ext
                SizeBytes    = $sizeBytes
                SizeGB       = [math]::Round($sizeBytes / 1GB, 3)
                LastWrite    = $_.LastWriteTime.ToString('s')
                URFlag       = $isUr
                URSegment    = $urSeg
                FullPath     = $full
            })

            if (($count % 500) -eq 0) {
                Write-Progress -Activity 'Cataloging video files' `
                    -Status ("{0} found ({1} UR)  scanning {2}" -f $count, $urCount, $driveLetter)
            }
        }
    }
}
Write-Progress -Activity 'Cataloging video files' -Completed

# --- Write CSV (UTF8 preserves Japanese / non-ASCII file names) -------------
$rows | Export-Csv -LiteralPath $OutFile -NoTypeInformation -Encoding UTF8

# --- Summary ----------------------------------------------------------------
$totalGB = [math]::Round($totalBytes / 1GB, 1)
$totalTB = [math]::Round($totalBytes / 1TB, 2)
$urBytes = 0L
foreach ($r in $rows) { if ($r.URFlag) { $urBytes += $r.SizeBytes } }
$urTB = [math]::Round($urBytes / 1TB, 2)

Write-Host ''
Write-Host '======================================================================'
Write-Host ("[catalog] DONE: {0} files, {1} GB ({2} TB)" -f $count, $totalGB, $totalTB)
Write-Host ("[catalog] UR (uncensored) candidates: {0} files, {1} TB" -f $urCount, $urTB)
Write-Host '[catalog] by extension:'
$rows | Group-Object Extension | Sort-Object Count -Descending | ForEach-Object {
    Write-Host ("    {0,-6} {1}" -f $_.Name, $_.Count)
}
Write-Host '[catalog] by drive:'
$rows | Group-Object Drive | Sort-Object Name | ForEach-Object {
    Write-Host ("    {0,-4} {1} files" -f $_.Name, $_.Count)
}
Write-Host ("[catalog] CSV: {0}" -f $OutFile)
Write-Host '[catalog] Filter UR candidates later with:'
Write-Host ("    Import-Csv '{0}' | Where-Object URFlag -eq 'True'" -f $OutFile)
Write-Host '======================================================================'
