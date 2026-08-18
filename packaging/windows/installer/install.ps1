# packaging/windows/installer/install.ps1
# ChitraMaya installer brain - runs from inside the ChitraMaya-install.exe SFX.
#
# The .exe users double-click is a tiny self-extractor that unpacks this
# script (plus 7zr.exe) to a temp folder and runs it. This script then:
#   1. finds the folder the .exe was run from (parent-process walk, with
#      sensible fallbacks),
#   2. verifies ALL archive volumes are present - and if not, says exactly
#      which files are missing, in plain language, instead of 7-Zip's
#      "Cannot open the file as [7z] archive",
#   3. extracts with the bundled 7zr.exe and prints what to do next.
#
# ASCII-only output. Always ends with a Read-Host so the window never
# vanishes before the user reads the message.

$ErrorActionPreference = "SilentlyContinue"

$BaseName      = "ChitraMaya-install"   # stamped by the packager (xpu edition uses ChitraMaya-xpu-install)
$ExpectedParts = 0   # 0 = unknown; the packager stamps the real count at build time.
$ReleasesUrl   = "https://github.com/seatv/ChitraMaya/releases"

function Find-InstallerDir {
    # The SFX exe is our grandparent process (exe -> cmd -> powershell).
    try {
        $me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
        $p  = $me
        for ($i = 0; $i -lt 4 -and $p; $i++) {
            $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)"
            if ($p -and $p.ExecutablePath -and
                ([IO.Path]::GetFileName($p.ExecutablePath) -like "$BaseName*.exe")) {
                return (Split-Path $p.ExecutablePath -Parent)
            }
        }
    } catch { }
    return $null
}

# Candidate folders, most reliable first.
$candidates = @()
$fromParent = Find-InstallerDir
if ($fromParent) { $candidates += $fromParent }
$candidates += (Get-Location).Path
$candidates += (Join-Path $env:USERPROFILE "Downloads")
$candidates = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

# Pick the first folder that has ANY part of the archive.
$srcDir = $null
foreach ($d in $candidates) {
    if (Get-ChildItem -Path $d -Filter "$BaseName.7z.0*" -File) { $srcDir = $d; break }
}
if (-not $srcDir) { $srcDir = $candidates[0] }

# Inventory the volumes.
$parts = @(Get-ChildItem -Path $srcDir -Filter "$BaseName.7z.0*" -File | Sort-Object Name)

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  ChitraMaya Installer" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Looking for archive parts in:" -ForegroundColor Gray
Write-Host "    $srcDir"
Write-Host ""

$haveFirst = Test-Path (Join-Path $srcDir "$BaseName.7z.001")

if (-not $haveFirst) {
    Write-Host "  *** THE INSTALL ARCHIVES WERE NOT FOUND ***" -ForegroundColor Red
    Write-Host ""
    # File count is stamped per release: N archive parts + this exe. When
    # the count is unknown (unstamped build), say "at least 2" rather than
    # inventing a number.
    $totalFiles = if ($ExpectedParts -gt 0) { $ExpectedParts + 1 } else { 0 }
    $countText  = if ($totalFiles -gt 0) { "$totalFiles files" } else { "at least 2 files" }
    Write-Host "  This .exe BY ITSELF is not the program - it is only the" -ForegroundColor Yellow
    Write-Host "  unpacker. The release is $countText, and all of them must" -ForegroundColor Yellow
    Write-Host "  be downloaded into the SAME folder:" -ForegroundColor Yellow
    Write-Host ""
    $maxParts = if ($ExpectedParts -gt 0) { $ExpectedParts } else { 1 }
    for ($n = 1; $n -le $maxParts; $n++) {
        $pn = "{0}.7z.{1:d3}" -f $BaseName, $n
        $missingPart = -not (Test-Path (Join-Path $srcDir $pn))
        $tag = if ($missingPart) { "<- MISSING" } else { "<- found" }
        $col = if ($missingPart) { "Red" } else { "Green" }
        Write-Host "      $pn   $tag" -ForegroundColor $col
    }
    if ($ExpectedParts -eq 0) {
        Write-Host "      (plus any further $BaseName.7z.0NN parts in the release)" -ForegroundColor Yellow
    }
    Write-Host "      $BaseName.exe       <- found (this file)" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Download the missing file(s) from:" -ForegroundColor Yellow
    Write-Host "      $ReleasesUrl"
    Write-Host ""
    Write-Host "  Put them next to this .exe, then run it again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

if ($ExpectedParts -gt 0 -and $parts.Count -lt $ExpectedParts) {
    Write-Host "  *** SOME ARCHIVE PARTS ARE MISSING ***" -ForegroundColor Red
    Write-Host ""
    Write-Host ("  This release has {0} archive parts; only {1} found:" -f $ExpectedParts, $parts.Count) -ForegroundColor Yellow
    for ($n = 1; $n -le $ExpectedParts; $n++) {
        $pn = "{0}.7z.{1:d3}" -f $BaseName, $n
        if (Test-Path (Join-Path $srcDir $pn)) {
            Write-Host "      $pn   <- found" -ForegroundColor Green
        } else {
            Write-Host "      $pn   <- MISSING" -ForegroundColor Red
        }
    }
    Write-Host ""
    Write-Host "  Download the missing part(s) from:" -ForegroundColor Yellow
    Write-Host "      $ReleasesUrl"
    Write-Host "  into the same folder, then run this .exe again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

Write-Host "  Found archive parts:" -ForegroundColor Green
foreach ($p in $parts) {
    $mb = [math]::Round($p.Length / 1MB, 1)
    Write-Host ("      {0}  ({1} MB)" -f $p.Name, $mb) -ForegroundColor Green
}
Write-Host ""

# CM-101 (field bug 2026-08-16): where the VOLUMES are and where the app
# should INSTALL are two different questions. $srcDir keys off the exe's
# location (right for finding parts); the destination used to silently
# reuse it, so running the installer from a terminal in C:\MyPrograms
# with the exe on a mapped drive installed onto the mapped drive. Now the
# destination is explicit: default to the terminal's working directory
# when it is a real folder (not the SFX temp dir), else the exe's folder,
# and always let the user type a different path. Double-click users press
# Enter and get the old behavior.
function Test-LocalFixedPath([string]$p) {
    # True only for a path on a LOCAL FIXED disk. Mapped network drives
    # (DriveType 4) and UNC paths are excluded: r2 field lesson - the exe
    # was run from a mapped T: (the dev box's project share) and the
    # installer defaulted the INSTALL there, next to the source tree.
    try {
        if (-not $p) { return $false }
        if ($p.StartsWith("\\")) { return $false }          # UNC
        $root = [IO.Path]::GetPathRoot($p)
        if (-not $root) { return $false }
        $ld = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}'" -f $root.TrimEnd('\')) -ErrorAction Stop
        return ($ld -and [int]$ld.DriveType -eq 3)          # 3 = local fixed
    } catch { return $false }
}

# Destination preference, r2:
#   1. the terminal's working directory, when it is usable and local
#      (a user who ran the exe from a prompt expects "install here"),
#   2. the exe's own folder, but ONLY if that is a local fixed disk,
#   3. %USERPROFILE% - never a mapped drive, never the SFX temp dir.
# In every case the prompt below lets the user type something else, so a
# wrong guess costs one keystroke, not a reinstall.
$tmpRoot = [IO.Path]::GetTempPath().TrimEnd('\')
$cwd = $null
try {
    $c = (Get-Location).Path
    if ($c -and (Test-Path $c) -and
        -not $c.StartsWith($tmpRoot, [StringComparison]::OrdinalIgnoreCase) -and
        -not $c.StartsWith($PSScriptRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-LocalFixedPath $c)) {
        $cwd = $c
    }
} catch { }

if ($cwd) {
    $destDefault = $cwd
} elseif (Test-LocalFixedPath $srcDir) {
    $destDefault = $srcDir
} else {
    $destDefault = $env:USERPROFILE
    Write-Host "  NOTE: this installer is running from a network or mapped" -ForegroundColor Yellow
    Write-Host "  drive ($srcDir)." -ForegroundColor Yellow
    Write-Host "  Installing THERE would put the program on that share, so the" -ForegroundColor Yellow
    Write-Host "  default below is a local folder instead." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "  ChitraMaya will be installed as a 'ChitraMaya' folder inside:" -ForegroundColor Cyan
Write-Host ("      {0}" -f $destDefault)
Write-Host ""
$destInput = Read-Host "  Press Enter to accept, or type a different folder"
$destDir = if ($destInput -and $destInput.Trim()) { $destInput.Trim().Trim('"') } else { $destDefault }
if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
}
if (-not (Test-Path $destDir)) {
    Write-Host ""
    Write-Host "  *** CANNOT CREATE FOLDER: $destDir ***" -ForegroundColor Red
    Write-Host "  Check the path and permissions, then run the installer again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "  Extracting (this can take a few minutes)..." -ForegroundColor Cyan
Write-Host ""

$sevenZr = Join-Path $PSScriptRoot "7zr.exe"
& $sevenZr x -y ("-o{0}" -f $destDir) (Join-Path $srcDir "$BaseName.7z.001")
$rc = $LASTEXITCODE

Write-Host ""
if ($rc -eq 0) {
    Write-Host "  DONE. ChitraMaya was extracted to:" -ForegroundColor Green
    Write-Host ("      {0}\ChitraMaya" -f $destDir)
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "    1. Open the ChitraMaya folder and run ChitraMaya.exe"
    Write-Host "    2. First time only: click Manage Models to download models"
    Write-Host "       and compile engines for your GPU (see the README)."
} else {
    Write-Host "  *** EXTRACTION FAILED (7-Zip exit code $rc) ***" -ForegroundColor Red
    Write-Host ""
    Write-Host "  This usually means a part is INCOMPLETE or CORRUPT - most" -ForegroundColor Yellow
    Write-Host "  often a download that did not finish. Compare the file sizes" -ForegroundColor Yellow
    Write-Host "  above against the release page, re-download the smaller one," -ForegroundColor Yellow
    Write-Host "  and run this .exe again:" -ForegroundColor Yellow
    Write-Host "      $ReleasesUrl"
}
Write-Host ""
Read-Host "  Press Enter to close"
exit $rc
