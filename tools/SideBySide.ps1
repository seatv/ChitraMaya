# tools/SideBySide.ps1
# Stack two or three videos side by side into ONE AV1 mp4 for headset viewing
# (Batch T7a; Quest 3 flip-free comparison of full-length outputs).
#
#   .\tools\SideBySide.ps1 -Inputs A.mp4, B.mp4 [, C.mp4] -Out sbs.mp4
#       [-Labels "lada", "ft_v1"] [-MaxWidth 3840] [-PaneHeight 0]
#       [-Start 00:44:58] [-Duration 00:03:00] [-Quality 28] [-Encoder auto]
#       [-NoLabels] [-NoHwDecode]
#
# What it does:
#   - Picks the AV1 encoder in this order unless -Encoder says otherwise:
#       av1_nvenc (RTX 40/50)  ->  av1_qsv (Intel Arc / iGPU)  ->
#       av1_amf (AMD RDNA3/4)  ->  libsvtav1 (CPU)  ->  libaom-av1 (CPU)
#     and probes ffmpeg's -encoders list so a missing one is skipped, not
#     tried.
#   - Hardware DECODE (-hwaccel cuda / qsv / d3d11va) is used when the
#     matching encoder was picked; -NoHwDecode forces software decode.
#   - Every pane is normalised to the first input's frame rate, scaled to a
#     common height, square pixels, yuv420p; hstack shortest=1 so outputs
#     that differ by a few gap-fill frames still line up.
#   - Total width is capped at -MaxWidth (default 3840: two 1080p panes
#     untouched; three panes scaled to 1280x720 each). Set -MaxWidth 0 for
#     no cap (three 1080p panes = 5760 wide; the Quest 3 decodes that but
#     the player may not like it).
#   - Labels are burnt in top-left of each pane with an explicit Windows
#     font file (drawtext without fontfile crashes gyan ffmpeg builds); if
#     no font is found the labels are dropped, not the run.
#   - Audio: copied from the FIRST input (-map 0:a?).
#   - -Start/-Duration cut the same window from every input (applied after
#     seeking, frame-accurate on the decoded stream).
#
# Output is mp4 with +faststart. Pure ASCII on purpose (PowerShell 5.1 safe).

param(
    [Parameter(Mandatory = $true)] [string[]]$Inputs,
    [Parameter(Mandatory = $true)] [string]$Out,
    [string[]]$Labels = @(),
    [int]$MaxWidth = 3840,
    [int]$PaneHeight = 0,
    [string]$Start = "",
    [string]$Duration = "",
    [int]$Quality = 28,
    [ValidateSet("auto", "av1_nvenc", "av1_qsv", "av1_amf", "libsvtav1", "libaom-av1")]
    [string]$Encoder = "auto",
    [switch]$NoLabels,
    [switch]$NoHwDecode,
    [string]$Ffmpeg = ""
)

$ErrorActionPreference = "Stop"

if ($Inputs.Count -lt 2 -or $Inputs.Count -gt 3) {
    Write-Error "Give two or three inputs (-Inputs A.mp4, B.mp4 [, C.mp4])."
    exit 1
}
foreach ($f in $Inputs) {
    if (-not (Test-Path -LiteralPath $f)) { Write-Error "Input not found: $f"; exit 1 }
}
if ($Labels.Count -gt 0 -and $Labels.Count -ne $Inputs.Count) {
    Write-Error "-Labels must have one entry per input (or be omitted)."
    exit 1
}
if ($Labels.Count -eq 0) {
    $Labels = @($Inputs | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_) })
}

# --- locate ffmpeg / ffprobe -------------------------------------------------
if ($Ffmpeg -eq "") {
    if ($env:CHITRAMAYA_FFMPEG -and (Test-Path -LiteralPath $env:CHITRAMAYA_FFMPEG)) {
        $Ffmpeg = $env:CHITRAMAYA_FFMPEG
    } else {
        $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
        if ($null -eq $cmd) { Write-Error "ffmpeg not found on PATH (or set CHITRAMAYA_FFMPEG / -Ffmpeg)."; exit 1 }
        $Ffmpeg = $cmd.Source
    }
}
$Ffprobe = Join-Path (Split-Path -Parent $Ffmpeg) "ffprobe.exe"
if (-not (Test-Path -LiteralPath $Ffprobe)) {
    $p = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($null -eq $p) { Write-Error "ffprobe not found next to ffmpeg or on PATH."; exit 1 }
    $Ffprobe = $p.Source
}

# --- probe the first input (fps drives every pane) ---------------------------
$probe = & $Ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 $Inputs[0]
$parts = ($probe -split ",")
$srcW = [int]$parts[0]; $srcH = [int]$parts[1]
$fpsExpr = $parts[2]
$fpsNum, $fpsDen = $fpsExpr -split "/"
$fps = [double]$fpsNum / [double]$fpsDen
Write-Host ("[sbs] first input {0}x{1} @ {2:N3} fps; {3} panes" -f $srcW, $srcH, $fps, $Inputs.Count)

# --- pane geometry ------------------------------------------------------------
$n = $Inputs.Count
if ($PaneHeight -le 0) { $PaneHeight = $srcH }
$paneW = [int][math]::Round($srcW * $PaneHeight / $srcH)
if ($MaxWidth -gt 0 -and ($paneW * $n) -gt $MaxWidth) {
    $paneW = [int][math]::Floor($MaxWidth / $n)
    $PaneHeight = [int][math]::Round($paneW * $srcH / $srcW)
}
# even dimensions for yuv420p
$paneW = $paneW - ($paneW % 2); $PaneHeight = $PaneHeight - ($PaneHeight % 2)
Write-Host ("[sbs] pane {0}x{1}; output {2}x{1}" -f $paneW, $PaneHeight, ($paneW * $n))

# --- encoder choice -----------------------------------------------------------
$encList = & $Ffmpeg -hide_banner -encoders 2>$null | Out-String
function Has-Encoder([string]$name) { return ($encList -match ("\s" + [regex]::Escape($name) + "\s")) }

$order = @("av1_nvenc", "av1_qsv", "av1_amf", "libsvtav1", "libaom-av1")
if ($Encoder -ne "auto") { $order = @($Encoder) }
$chosen = $null
foreach ($e in $order) { if (Has-Encoder $e) { $chosen = $e; break } }
if ($null -eq $chosen) { Write-Error "No AV1 encoder available in this ffmpeg build (tried: $($order -join ', '))."; exit 1 }

$encArgs = @()
$hwDecArgs = @()
switch ($chosen) {
    "av1_nvenc" {
        $encArgs = @("-c:v", "av1_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "$Quality", "-b:v", "0", "-pix_fmt", "yuv420p")
        if (-not $NoHwDecode) { $hwDecArgs = @("-hwaccel", "cuda") }
    }
    "av1_qsv" {
        $encArgs = @("-c:v", "av1_qsv", "-preset", "medium", "-global_quality", "$Quality", "-pix_fmt", "nv12")
        if (-not $NoHwDecode) { $hwDecArgs = @("-hwaccel", "qsv") }
    }
    "av1_amf" {
        $encArgs = @("-c:v", "av1_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "$Quality", "-qp_p", "$Quality", "-pix_fmt", "yuv420p")
        if (-not $NoHwDecode) { $hwDecArgs = @("-hwaccel", "d3d11va") }
    }
    "libsvtav1" {
        $encArgs = @("-c:v", "libsvtav1", "-preset", "6", "-crf", "$Quality", "-pix_fmt", "yuv420p")
    }
    "libaom-av1" {
        $encArgs = @("-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "$Quality", "-b:v", "0", "-pix_fmt", "yuv420p")
    }
}
Write-Host ("[sbs] encoder {0} (quality {1}); hw decode: {2}" -f $chosen, $Quality, ($(if ($hwDecArgs.Count -gt 0) { $hwDecArgs[1] } else { "software" })))

# --- font for labels (explicit fontfile, see tools/ab_eval.py _drawtext_font) --
$fontArg = ""
if (-not $NoLabels) {
    $windir = $env:WINDIR; if (-not $windir) { $windir = "C:\Windows" }
    $candidates = @("$windir\Fonts\arialbd.ttf", "$windir\Fonts\arial.ttf", "$windir\Fonts\segoeui.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            $esc = ($c -replace "\\", "/") -replace ":", "\:"
            $fontArg = "fontfile='" + $esc + "':"
            break
        }
    }
    if ($fontArg -eq "") { Write-Host "[sbs] no font file found; labels dropped"; $NoLabels = $true }
}

# --- filter graph -------------------------------------------------------------
$fontSize = [int][math]::Max(24, [math]::Round($PaneHeight / 22))
$filters = @()
$tags = @()
for ($i = 0; $i -lt $n; $i++) {
    $chain = "[{0}:v]fps={1},scale={2}:{3},setsar=1,format=yuv420p" -f $i, $fpsExpr, $paneW, $PaneHeight
    if (-not $NoLabels) {
        $safe = ($Labels[$i] -replace "'", "") -replace ":", "" -replace ",", ""
        $chain += ",drawtext={0}text='{1}':x=24:y=24:fontsize={2}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8" -f $fontArg, $safe, $fontSize
    }
    $chain += "[v$i]"
    $filters += $chain
    $tags += "[v$i]"
}
$filters += (($tags -join "") + "hstack=inputs=${n}:shortest=1[out]")
$graph = ($filters -join ";")

# --- assemble the command -----------------------------------------------------
$ffArgs = @("-hide_banner", "-y", "-loglevel", "warning", "-stats")
for ($i = 0; $i -lt $n; $i++) {
    $ffArgs += $hwDecArgs
    if ($Start -ne "") { $ffArgs += @("-ss", $Start) }
    if ($Duration -ne "") { $ffArgs += @("-t", $Duration) }
    $ffArgs += @("-i", $Inputs[$i])
}
$ffArgs += @("-filter_complex", $graph, "-map", "[out]", "-map", "0:a?", "-c:a", "copy")
$ffArgs += $encArgs
$ffArgs += @("-movflags", "+faststart", $Out)

Write-Host "[sbs] ffmpeg $($ffArgs -join ' ')"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $Ffmpeg @ffArgs
$rc = $LASTEXITCODE
if ($rc -ne 0 -and -not $NoLabels) {
    Write-Host "[sbs] ffmpeg failed with labels (rc=$rc); retrying WITHOUT labels"
    $filters = @(); $tags = @()
    for ($i = 0; $i -lt $n; $i++) {
        $filters += ("[{0}:v]fps={1},scale={2}:{3},setsar=1,format=yuv420p[v{0}]" -f $i, $fpsExpr, $paneW, $PaneHeight)
        $tags += "[v$i]"
    }
    $filters += (($tags -join "") + "hstack=inputs=${n}:shortest=1[out]")
    $graph = ($filters -join ";")
    $idx = [array]::IndexOf($ffArgs, "-filter_complex")
    $ffArgs[$idx + 1] = $graph
    & $Ffmpeg @ffArgs
    $rc = $LASTEXITCODE
}
if ($rc -ne 0 -and $hwDecArgs.Count -gt 0) {
    Write-Host "[sbs] ffmpeg failed with hardware decode (rc=$rc); retrying with software decode"
    $ffArgs = @($ffArgs | Where-Object { $_ -ne "-hwaccel" -and $_ -ne $hwDecArgs[1] })
    & $Ffmpeg @ffArgs
    $rc = $LASTEXITCODE
}
$sw.Stop()
if ($rc -ne 0) { Write-Error "ffmpeg failed (rc=$rc)"; exit $rc }
$size = (Get-Item -LiteralPath $Out).Length / 1MB
Write-Host ("[sbs] done: {0} ({1:N1} MB) in {2:N0}s" -f $Out, $size, $sw.Elapsed.TotalSeconds)
