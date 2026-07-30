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

$BaseName      = "ChitraMaya-install"
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
    Write-Host "  This .exe BY ITSELF is not the program - it is only the" -ForegroundColor Yellow
    Write-Host "  unpacker. The release is THREE files, and all of them must" -ForegroundColor Yellow
    Write-Host "  be downloaded into the SAME folder:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "      $BaseName.7z.001   <- MISSING" -ForegroundColor Red
    if ($ExpectedParts -ge 2 -or $ExpectedParts -eq 0) {
        $missing002 = -not (Test-Path (Join-Path $srcDir "$BaseName.7z.002"))
        $tag = if ($missing002) { "<- MISSING" } else { "<- found" }
        $col = if ($missing002) { "Red" } else { "Green" }
        Write-Host "      $BaseName.7z.002   $tag" -ForegroundColor $col
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
Write-Host "  Extracting (this can take a few minutes)..." -ForegroundColor Cyan
Write-Host ""

$sevenZr = Join-Path $PSScriptRoot "7zr.exe"
& $sevenZr x -y ("-o{0}" -f $srcDir) (Join-Path $srcDir "$BaseName.7z.001")
$rc = $LASTEXITCODE

Write-Host ""
if ($rc -eq 0) {
    Write-Host "  DONE. ChitraMaya was extracted to:" -ForegroundColor Green
    Write-Host ("      {0}\ChitraMaya" -f $srcDir)
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
