# tools/CompareChitraRuns.ps1
# Paired A/B/clean comparison wrapper around tools/ab_eval.py (Batch 75).
#
#   .\tools\CompareChitraRuns.ps1 -clean D:\path\Clean.mp4 `
#       -a D:\path\restored-A.mp4 -b D:\path\restored-B.mp4
#
# Runs ONE ab_eval invocation: metrics for A and B against the pristine
# clip (PSNR-ROI, SSIM-ROI, texture, motion, flicker) plus the A-vs-B
# cross-output mean-abs-difference. Results land in -OutDir as
# ab_eval_results.json (plus optional side-by-side MP4 / contact sheet).
#
# Defaults chosen from field lessons:
#   -Shift 0        : inputs are frame-exact cuts of the same segment
#                     (ab_eval's find_shift needs ~13 GB on long clips).
#   -Scale 1920x1080: 640x360 washed out real scaler differences
#                     (2026-08-31 null result); full 3840x2160 needs
#                     ~26 GB RAM for the cross-diff on a 400-frame clip.
#                     1080p preserves the differences and fits. Pass
#                     -Scale 3840x2160 explicitly on a big-RAM box.

param(
    [Parameter(Mandatory = $true)] [string]$Clean,
    [Parameter(Mandatory = $true)] [string]$A,
    [Parameter(Mandatory = $true)] [string]$B,
    [string]$OutDir = ".\ab_eval_out",
    [string]$Scale = "1920x1080",
    [string]$Shift = "0",
    [int]$ContactSheet = 12,
    [switch]$SideBySide
)

$ErrorActionPreference = "Stop"

foreach ($f in @($Clean, $A, $B)) {
    if (-not (Test-Path -LiteralPath $f)) {
        Write-Error "File not found: $f"
        exit 1
    }
}

# Repo root = parent of the tools\ folder this script lives in.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$AbEval = Join-Path $RepoRoot "tools\ab_eval.py"
if (-not (Test-Path -LiteralPath $AbEval)) {
    Write-Error "ab_eval.py not found at $AbEval"
    exit 1
}

# Prefer the repo venv python; fall back to python on PATH.
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

# Labels from file stems so the results table names the actual runs;
# fall back to A/B if the stems collide.
$LabelA = [System.IO.Path]::GetFileNameWithoutExtension($A)
$LabelB = [System.IO.Path]::GetFileNameWithoutExtension($B)
if ($LabelA -eq $LabelB) { $LabelA = "A"; $LabelB = "B" }

$ArgList = @(
    $AbEval,
    "--original", $Clean,
    "--restored", "$LabelA=$A",
    "--restored", "$LabelB=$B",
    "--shift", $Shift,
    "--scale", $Scale,
    "--out-dir", $OutDir
)
if ($ContactSheet -gt 0) { $ArgList += @("--contact-sheet", "$ContactSheet") }
if ($SideBySide) { $ArgList += "--side-by-side" }

Write-Host "[CompareChitraRuns] python: $Python"
Write-Host "[CompareChitraRuns] A = $LabelA"
Write-Host "[CompareChitraRuns] B = $LabelB"
Write-Host "[CompareChitraRuns] scale=$Scale shift=$Shift out=$OutDir"
Write-Host ""

& $Python @ArgList
$Code = $LASTEXITCODE
if ($Code -ne 0) {
    Write-Error "ab_eval.py exited with code $Code"
    exit $Code
}

Write-Host ""
Write-Host "[CompareChitraRuns] done -- see $OutDir\ab_eval_results.json"
exit 0
