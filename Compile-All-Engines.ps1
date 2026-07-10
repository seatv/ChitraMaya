# Compile-All-Engines.ps1
<#
.SYNOPSIS
    Compile all TensorRT engines that chitramaya needs from the model files
    in ./models.

.DESCRIPTION
    Orchestrates the two compile paths in order:
      1. YOLO mosaic detection — one engine per *.pt found in models/
         via `chitramaya -compile-det`
      2. BasicVSR++ mosaic restoration — one set of sub-engines per *.pth
         via `chitramaya -compile-rest`

    Each compile is independent; a failure in one does not block the others.
    A summary at the end shows what built and what didn't.

    Engines land in models/engines/ (YOLO as flat files, BasicVSR++ as a
    per-model sub-directory of 6 sub-engines).

    Engines are compiled FOR THE CURRENT GPU and OS. Re-run this script on
    each target machine.

.PARAMETER ModelsDir
    Directory containing the model files (.onnx, .pt, .pth).
    Defaults to .\models (the repo's models directory).

.PARAMETER DetImgsz
    YOLO opt image size. With dynamic engines, this is the size the engine
    is TUNED for; the engine accepts any size from 32 to Workspace*DetImgsz.
    Defaults to 640 (matches Lada's training and the runtime default).

.PARAMETER DetMaxBatch
    YOLO max batch size for the dynamic engine. Engine accepts batches 1..N.
    Defaults to 8 (matches the pipeline's typical detection batch).

.PARAMETER DetWorkspace
    TRT workspace in GB for YOLO compile. Also caps max imgsz at
    Workspace*DetImgsz. Defaults to 2 GB (memory-safe on 8-12GB GPUs).
    Higher values give TRT more headroom to try larger tactics but require
    more VRAM during compile.

.PARAMETER RestMaxClipLength
    BasicVSR++ max_clip_size. Engines compiled at value N are valid for
    clips of length 1..N at inference. Defaults to 60. Run with a larger
    value to support longer clips; engine files for different mcl values
    coexist (filenames encode the size).

.PARAMETER RestWorkspace
    TRT workspace in GB for the BasicVSR++ compile. Bounds the scratch the
    builder may use, which caps the workspace the engines reserve RESIDENT
    at runtime. Defaults to 2 GB. Smaller leaves more VRAM for the frame
    store (chitramaya keeps everything resident on the GPU; there is no host
    offload). Raise if a build reports it needs more scratch; lower for more
    runtime headroom. Pass 0 for the legacy 95%-of-free (unbounded) behavior.

.PARAMETER SkipDetection
    Skip ALL YOLO detection compiles.

.PARAMETER SkipRestoration
    Skip ALL BasicVSR++ restoration compiles.

.PARAMETER NoFp16
    Build fp32 engines instead of fp16. Larger, slower, more accurate.

.PARAMETER Force
    Recompile engines that already exist.

.EXAMPLE
    .\Compile-All-Engines.ps1

    Compile everything found in .\models at fp16, det opt-imgsz=640,
    max-batch=8, workspace=2GB, mcl=60.

.EXAMPLE
    .\Compile-All-Engines.ps1 -DetWorkspace 4 -DetMaxBatch 16

    Use a larger detection envelope. Requires more VRAM during compile.

.EXAMPLE
    .\Compile-All-Engines.ps1 -RestMaxClipLength 180 -Force

    Recompile the BasicVSR++ sub-engines for mcl=180 ceiling.

.EXAMPLE
    .\Compile-All-Engines.ps1 -SkipRestoration

    Only compile the YOLO detection engines (one per .pt found).
#>

param(
    [string]$ModelsDir = "$PSScriptRoot\models",
    [int]$DetImgsz = 640,
    [int]$DetMaxBatch = 8,
    [int]$DetWorkspace = 2,
    [int]$RestMaxClipLength = 60,
    [int]$RestWorkspace = 2,
    [switch]$SkipDetection,
    [switch]$SkipRestoration,
    [switch]$NoFp16,
    [switch]$Force
)

$ErrorActionPreference = "Continue"  # We want to keep going on failures

# Resolve the chitramaya compiler: prefer the frozen ChitraMaya.exe sitting
# next to this script (clean install), else the 'chitramaya' console script on
# PATH (dev venv). The frozen exe bundles the compile code (tools/*), so a
# machine with only the installer -- no Python, no ./tools -- can still compile.
$ChitraMaya = if (Test-Path (Join-Path $PSScriptRoot 'ChitraMaya.exe')) {
    Join-Path $PSScriptRoot 'ChitraMaya.exe'
} else {
    'chitramaya'
}
Write-Host "[i] Compiler: $ChitraMaya" -ForegroundColor DarkGray

# Resolve and validate the models directory
$ModelsDir = (Resolve-Path -Path $ModelsDir -ErrorAction SilentlyContinue).Path
if (-not $ModelsDir -or -not (Test-Path $ModelsDir -PathType Container)) {
    Write-Host "[!] Models directory not found: $ModelsDir" -ForegroundColor Red
    Write-Host "    Pass -ModelsDir <path> to override the default ('.\models')." -ForegroundColor Yellow
    exit 1
}

$Fp16Flag = if ($NoFp16) { "--no-fp16" } else { "--fp16" }

# Track outcomes for the final summary
$results = @()

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor Cyan
    Write-Host ""
}

function Invoke-CompileStep {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Header $Name
    try {
        & $Command
        $exitCode = $LASTEXITCODE
        $sw.Stop()
        if ($exitCode -eq 0) {
            $status = "OK"
            Write-Host ""
            Write-Host "[$Name] Completed in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green
        } else {
            $status = "FAIL (exit $exitCode)"
            Write-Host ""
            Write-Host "[$Name] Failed with exit code $exitCode" -ForegroundColor Red
        }
    } catch {
        $sw.Stop()
        $status = "ERROR: $($_.Exception.Message)"
        Write-Host ""
        Write-Host "[$Name] $status" -ForegroundColor Red
    }
    $script:results += [PSCustomObject]@{
        Step    = $Name
        Status  = $status
        Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    }
}

$totalSw = [System.Diagnostics.Stopwatch]::StartNew()

# Discover models in the models directory
$DetModels  = if ($SkipDetection)   { @() } else { @(Get-ChildItem -Path $ModelsDir -Filter "*.pt"  -File | Sort-Object Name) }
$RestModels = if ($SkipRestoration) { @() } else { @(Get-ChildItem -Path $ModelsDir -Filter "*.pth" -File | Sort-Object Name) }

Write-Host ""
Write-Host "chitramaya engine compilation" -ForegroundColor White
Write-Host "  Models dir:        $ModelsDir"
Write-Host "  Det opt-imgsz:     $DetImgsz"
Write-Host "  Det max batch:     $DetMaxBatch"
Write-Host "  Det workspace:     $DetWorkspace GB (max imgsz = $($DetWorkspace * $DetImgsz))"
Write-Host "  Rest max clip:     $RestMaxClipLength"
Write-Host "  Rest workspace:    $RestWorkspace GB"
Write-Host "  Precision:         $(if ($NoFp16) {'fp32'} else {'fp16'})"
Write-Host "  Force rebuild:     $Force"
Write-Host ""
Write-Host "  Detection models discovered:   $($DetModels.Count)"
foreach ($m in $DetModels)  { Write-Host "    - $($m.Name)" }
Write-Host "  Restoration models discovered: $($RestModels.Count)"
foreach ($m in $RestModels) { Write-Host "    - $($m.Name)" }

# ---- Step 1: YOLO detection (one engine per .pt) ----
if ($SkipDetection) {
    Write-Host ""
    Write-Host "[Step 1: YOLO detection] SKIPPED" -ForegroundColor Yellow
    $script:results += [PSCustomObject]@{ Step = "Step 1: YOLO detection"; Status = "SKIPPED"; Seconds = 0 }
} elseif ($DetModels.Count -eq 0) {
    Write-Host ""
    Write-Host "[Step 1: YOLO detection] No .pt files found in $ModelsDir" -ForegroundColor Yellow
    $script:results += [PSCustomObject]@{ Step = "Step 1: YOLO detection"; Status = "SKIPPED (no .pt files)"; Seconds = 0 }
} else {
    $i = 0
    foreach ($detModel in $DetModels) {
        $i += 1
        $stepName = "Step 1.$i/$($DetModels.Count): YOLO $($detModel.BaseName)"
        $modelPath = $detModel.FullName  # capture for closure
        Invoke-CompileStep -Name $stepName -Command {
            $stepArgs = @(
                "-compile-det",
                "--det-model",  $modelPath,
                "--det-imgsz",  $DetImgsz,
                "--max-batch",  $DetMaxBatch,
                "--workspace",  $DetWorkspace,
                $Fp16Flag
            )
            if ($Force) { $stepArgs += "--force" }
            & $ChitraMaya @stepArgs
        }
    }
}

# Known non-BasicVSR++ .pth files that may appear in models/.
# These are valid model files for other parts of the pipeline (face parser,
# depth estimation) but are NOT restoration models. compile-rest would happily
# produce a junk engine from them, so we skip explicitly.
$KnownNonRestPth = @(
    "79999_iter.pth",          # BiseNet face-parser
    "rd64-uni-refined.pth"     # MiDaS depth estimation
)

# ---- Step 2: BasicVSR++ restoration (one set of sub-engines per .pth) ----
if ($SkipRestoration) {
    Write-Host ""
    Write-Host "[Step 2: BasicVSR++ restoration] SKIPPED" -ForegroundColor Yellow
    $script:results += [PSCustomObject]@{ Step = "Step 2: BasicVSR++ restoration"; Status = "SKIPPED"; Seconds = 0 }
} elseif ($RestModels.Count -eq 0) {
    Write-Host ""
    Write-Host "[Step 2: BasicVSR++ restoration] No .pth files found in $ModelsDir" -ForegroundColor Yellow
    $script:results += [PSCustomObject]@{ Step = "Step 2: BasicVSR++ restoration"; Status = "SKIPPED (no .pth files)"; Seconds = 0 }
} else {
    $i = 0
    foreach ($restModel in $RestModels) {
        $i += 1
        $stepName = "Step 2.$i/$($RestModels.Count): BasicVSR++ $($restModel.BaseName)"

        # Skip known non-restoration .pth files
        if ($KnownNonRestPth -contains $restModel.Name) {
            Write-Host ""
            Write-Host "$($restModel.Name) is not a valid Restoration model - skipping" -ForegroundColor Yellow
            $script:results += [PSCustomObject]@{
                Step    = $stepName
                Status  = "SKIPPED (not a Restoration model)"
                Seconds = 0
            }
            continue
        }

        $modelPath = $restModel.FullName  # capture for closure
        Invoke-CompileStep -Name $stepName -Command {
            $stepArgs = @(
                "-compile-rest",
                "--rest-model",            $modelPath,
                "--rest-max-clip-length",  $RestMaxClipLength,
                "--workspace",             $RestWorkspace,
                $Fp16Flag
            )
            if ($Force) { $stepArgs += "--force" }
            & $ChitraMaya @stepArgs
        }
    }
}

$totalSw.Stop()

# Summary
Write-Host ""
Write-Host ("=" * 76) -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host ("=" * 76) -ForegroundColor Cyan
Write-Host ""
$results | Format-Table -AutoSize
Write-Host ""
Write-Host "Total elapsed: $([math]::Round($totalSw.Elapsed.TotalMinutes, 1)) minutes"
Write-Host ""

$anyFail = $results | Where-Object { $_.Status -like "FAIL*" -or $_.Status -like "ERROR*" }
if ($anyFail) {
    exit 1
} else {
    exit 0
}
