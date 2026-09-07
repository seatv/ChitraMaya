# tools/TrainRestore.ps1
# Fine-tune the BasicVSR++ restorer on pairs from GenDataSet.ps1 (wrapper
# around "ChitraMaya -train-rest", i.e. tools/train_rest_poc.py; Batch T3).
#
#   .\tools\TrainRestore.ps1 -p H:\Train\pairs-v1 -r H:\Train\runs\rest-v0
#   .\tools\TrainRestore.ps1 -p H:\Train\pairs-v2 -r H:\Train\runs\rest-v2 -- --iters 10000
#   .\tools\TrainRestore.ps1 -p H:\Train\pairs-v2 -r H:\Train\runs\rest-v2 -Resume
#
# -Base defaults to lada generic v1.2 (AGPL; attributed derivation). Pass
# -Base to continue from an earlier fine-tune (e.g. models\cm_lada_..._ft_v1.pth).
# -Resume picks up <run>\weights\latest.pth. Everything after "--" (or any
# unrecognised argument) is passed through to train_rest_poc.py unchanged,
# e.g. --iters, --batch, --lr, --no-amp.
#
# Runs are LOCAL artifacts under the user-chosen run folder. Promote a
# finished model by copying <run>\weights\best.pth to
# models\cm_lada_<what>_ft_v<N>.pth (bump N every retrain).

param(
    [Alias("p")]
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PairsPath,

    [Alias("r")]
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$RunPath,

    [string]$Base = "models\lada_mosaic_restoration_model_generic_v1.2.pth",
    [string]$Device = "0",
    [switch]$Resume,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThru
)

$ErrorActionPreference = "Stop"

# Repo root = parent of the tools\ folder this script lives in. The app is
# invoked as a module (python -m chitramaya), so the root must be the cwd.
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "chitramaya\__main__.py"))) {
    Write-Error "chitramaya package not found under $RepoRoot"
    exit 1
}

# Prefer the repo venv python; fall back to python on PATH.
$Python = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

# Resolve the base checkpoint against the repo root when it is relative.
if (-not [System.IO.Path]::IsPathRooted($Base)) {
    $Base = Join-Path $RepoRoot $Base
}
if (-not (Test-Path -LiteralPath $Base)) {
    Write-Error "Base checkpoint not found: $Base"
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $PairsPath "pairs.json"))) {
    Write-Error "pairs.json not found in $PairsPath (run GenDataSet.ps1 first)"
    exit 1
}

$ArgList = @("-m", "chitramaya", "-train-rest",
    "--pairs", $PairsPath, "--base", $Base, "--out", $RunPath, "--device", $Device)
if ($Resume) {
    $Latest = Join-Path $RunPath "weights\latest.pth"
    if (-not (Test-Path -LiteralPath $Latest)) {
        Write-Error "-Resume requested but $Latest does not exist"
        exit 1
    }
    $ArgList += @("--resume", $Latest)
}
if ($PassThru) { $ArgList += ($PassThru | Where-Object { $_ -ne "--" }) }

Write-Host "[TrainRestore] python : $Python"
Write-Host "[TrainRestore] pairs  : $PairsPath"
Write-Host "[TrainRestore] base   : $Base"
Write-Host "[TrainRestore] run    : $RunPath (device $Device)"
if ($Resume) { Write-Host "[TrainRestore] resume : $Latest" }
if ($PassThru) { Write-Host "[TrainRestore] extra  : $($PassThru -join ' ')" }
Write-Host ""

Push-Location $RepoRoot
try {
    & $Python @ArgList
    $Code = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($Code -ne 0) {
    Write-Error "train-rest exited with code $Code"
    exit $Code
}
Write-Host "[TrainRestore] done -> $RunPath\weights\best.pth"
