<#
.SYNOPSIS
  Commit the running ChitraMaya source tree to a local git working copy and
  push it to GitLab -- git without doing git.

.DESCRIPTION
  First time:   .\Publish-ChitraMaya.ps1 -Source D:\MyPrograms\ChitraMaya -Init
  Every drop:   .\Publish-ChitraMaya.ps1 -Source D:\MyPrograms\ChitraMaya
                (after unzipping the drop over the tree)

  What it does: makes sure Git for Windows is present (installs it with winget
  if not), keeps a git working copy under -Work (default
  %USERPROFILE%\ChitraMaya-git), mirrors the source tree into it (models,
  engines, venvs, temp and build output excluded), commits as seatv with the
  version from chitramaya\__init__.py in the message, and pushes to -Repo.
  The first push opens a browser sign-in to GitLab (Git Credential Manager);
  after that it remembers you.

  Nothing in -Source is modified. The working copy is disposable: delete it
  and run with -Init again to rebuild it from GitLab or from the tree.
#>
param(
  [Parameter(Mandatory)][string]$Source,
  [string]$Repo    = "https://gitlab.com/seatv/ChitraMaya.git",
  [string]$Work    = (Join-Path $env:USERPROFILE "ChitraMaya-git"),
  [string]$Message = "",
  [string]$Name    = "seatv",
  [string]$Email   = "97193924+seatv@users.noreply.github.com",
  [switch]$Init,
  [switch]$NoPush
)
$ErrorActionPreference = "Stop"

# ---- git present? ---------------------------------------------------------
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
  Write-Host "Git for Windows not found -- installing with winget (one time)..."
  winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
  Write-Host "Git installed. Close this window, open a NEW PowerShell window and run the script again."
  exit 0
}
$git = $git.Source

# ---- source sanity ---------------------------------------------------------
if (-not (Test-Path (Join-Path $Source "chitramaya\__init__.py"))) {
  throw "$Source does not look like a ChitraMaya tree (no chitramaya\__init__.py)"
}
$verLine = Select-String -Path (Join-Path $Source "chitramaya\__init__.py") -Pattern '__version__\s*=\s*"([^"]+)"'
$version = if ($verLine) { $verLine.Matches[0].Groups[1].Value } else { "unknown" }
Write-Host "Source: $Source  (version $version)"

# ---- working copy ----------------------------------------------------------
if (-not (Test-Path (Join-Path $Work ".git"))) {
  if (-not $Init) { throw "No working copy at $Work yet -- run once with -Init" }
  New-Item -ItemType Directory -Force -Path $Work | Out-Null
  & $git -C $Work init -b main | Out-Null
  & $git -C $Work remote add origin $Repo
  Write-Host "Initialized working copy at $Work -> $Repo"
}
& $git -C $Work config user.name  $Name
& $git -C $Work config user.email $Email
& $git -C $Work config core.autocrlf false

# ---- mirror the tree in (robocopy: /MIR keeps the copy exact, deletions included)
$xd = @(".git", "models", "engines", "venv", ".venv", "env", "__pycache__", "dist", "build",
        "Temp", "temp", "logs", "node_modules", ".pytest_cache", ".mypy_cache", "runs")
# NOTE: *.pth is NOT excluded -- chitramaya\mosaic\restorer\weights\ holds the bundled
# Real-ESRGAN / TemporalFix weights (~11 MB) that the gates and the app need. The big
# models live under models\ and engines\, which are excluded as folders above.
$xf = @("*.engine", "*.pyc", "*.log", "*.zip", "*.bundle", "ChitraMaya-config.json",
        "*.mp4", "*.mkv", "*.ts", "*.av1", "*.hevc", "*.venc", "*.mka", "*.part")
$rc = @("$Source", "$Work", "/MIR", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:1", "/W:1")
$rc += "/XD"; $rc += ($xd | ForEach-Object { Join-Path $Source $_ }); $rc += (Join-Path $Work ".git")
$rc += "/XF"; $rc += $xf
& robocopy @rc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }

# a .gitignore so a stray model never goes up by accident
$gi = Join-Path $Work ".gitignore"
if (-not (Test-Path $gi)) {
@"
models/
engines/
venv/
.venv/
__pycache__/
dist/
build/
Temp/
*.engine
*.pyc
*.log
*.mp4
*.mkv
*.ts
ChitraMaya-config.json
"@ | Set-Content -Path $gi -Encoding UTF8
}

# ---- commit ----------------------------------------------------------------
& $git -C $Work add -A
$pending = & $git -C $Work status --porcelain
if (-not $pending) { Write-Host "Nothing changed since the last commit (tree already published)."; }
else {
  if (-not $Message) {
    $Message = if ($Init) { "ChitraMaya $version -- baseline of the GitLab home (earlier history: GitHub seatv/ChitraMaya and the sandbox bundles)" }
               else       { "ChitraMaya $version" }
  }
  & $git -C $Work commit -q -m $Message
  Write-Host ("Committed: " + (& $git -C $Work log -1 --format="%h %s"))
}

# ---- push ------------------------------------------------------------------
if ($NoPush) { Write-Host "NoPush: stopping before the push."; exit 0 }
Write-Host "Pushing to $Repo (first time: a browser window asks you to sign in to GitLab)..."
& $git -C $Work push -u origin main
if ($LASTEXITCODE -ne 0) { throw "push failed ($LASTEXITCODE) -- see the message above" }
Write-Host "Published. $Repo now holds $version."
