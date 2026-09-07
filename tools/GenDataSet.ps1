# tools/GenDataSet.ps1
# Build restorer training pairs from pristine video (wrapper around
# "ChitraMaya -make-pairs", i.e. tools/make_rest_pairs.py; Batch T3).
#
#   .\tools\GenDataSet.ps1 -i F:\UR-1080p -d H:\Train\pairs-v1
#   .\tools\GenDataSet.ps1 -i F:\UR-1080p\*.mp4, H:\UR\title.mkv -d H:\Train\pairs-v2
#   .\tools\GenDataSet.ps1 -i H:\UR -d H:\Train\pairs-v2 -Recurse -- --pairs-per-video 120
#
# -VideoPath (-i) accepts folders, files and wildcards (one or many). Folders
# are expanded to the video files inside them (-Recurse walks subfolders).
# Everything after "--" (or any unrecognised argument) is passed through to
# make_rest_pairs.py unchanged, e.g. --pairs-per-video, --clip-len,
# --regions, --det-imgsz (and --val-input once recipe v2 lands).
#
# Datasets are LOCAL artifacts: the output folder is user-chosen and is
# never distributed. Weights travel; content never does.

param(
    [Alias("i")]
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$VideoPath,

    [Alias("d")]
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$DataPath,

    [string]$DetModel = "models\lada_nsfw_detection_model_v1.3.pt",
    [string]$Device = "0",
    [string[]]$Extensions = @(".mp4", ".mkv", ".ts", ".m2ts", ".mov", ".webm"),
    [switch]$Recurse,

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

# Resolve the detector path against the repo root when it is relative.
if (-not [System.IO.Path]::IsPathRooted($DetModel)) {
    $DetModel = Join-Path $RepoRoot $DetModel
}
if (-not (Test-Path -LiteralPath $DetModel)) {
    Write-Error "Detector model not found: $DetModel"
    exit 1
}

# Expand -VideoPath: folders -> video files inside; files/wildcards as given.
$Wanted = $Extensions | ForEach-Object { $_.ToLower() }
$Files = New-Object System.Collections.Generic.List[string]
foreach ($p in $VideoPath) {
    $items = @(Get-ChildItem -Path $p -File -Recurse:$Recurse -ErrorAction SilentlyContinue)
    foreach ($it in $items) {
        if ($Wanted -contains $it.Extension.ToLower()) { $Files.Add($it.FullName) }
    }
}
$Files = @($Files | Sort-Object -Unique)
if ($Files.Count -eq 0) {
    Write-Error ("No video files matched: " + ($VideoPath -join ", ") +
        " (extensions: " + ($Extensions -join " ") + ")")
    exit 1
}

$ArgList = @("-m", "chitramaya", "-make-pairs")
foreach ($f in $Files) { $ArgList += @("--input", $f) }
$ArgList += @("--out", $DataPath, "--det-model", $DetModel, "--device", $Device)
if ($PassThru) { $ArgList += ($PassThru | Where-Object { $_ -ne "--" }) }

Write-Host "[GenDataSet] python : $Python"
Write-Host "[GenDataSet] inputs : $($Files.Count) video(s)"
foreach ($f in $Files) { Write-Host "[GenDataSet]   $f" }
Write-Host "[GenDataSet] out    : $DataPath"
Write-Host "[GenDataSet] det    : $DetModel (device $Device)"
if ($PassThru) { Write-Host "[GenDataSet] extra  : $($PassThru -join ' ')" }
Write-Host ""

Push-Location $RepoRoot
try {
    & $Python @ArgList
    $Code = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($Code -ne 0) {
    Write-Error "make-pairs exited with code $Code"
    exit $Code
}
Write-Host "[GenDataSet] done -> $DataPath"
