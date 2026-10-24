# Get-DownloadStats.ps1
# Download counts for get.chitramaya.org from Cloudflare's analytics API --
# the replacement for the GitHub release counter.
#
# How it counts: Cloudflare's GraphQL analytics (free plan) reports, per
# request path, the number of requests and the bytes served at the edge,
# cache hits included. For a multi-GB installer the honest number is
#   downloads = bytes served / file size
# (a browser that fetches in ranges, or a resumed download, is still one
# download's worth of bytes; a visitor who opens the page and leaves is
# zero). The file sizes come from latest.json on the site (what
# Upload-Release.ps1 wrote), so the script needs no local release folder.
#
# Output: a table per file for the window (requests, GB served, estimated
# downloads), a per-day table, the top countries, and a CSV ledger that is
# appended on every run (one row per day and file; re-running a day
# replaces that day's rows, so today's partial numbers get corrected
# tomorrow). The ledger is the long-term record: the API only keeps a few
# days of per-path data on the free plan, so run this at least weekly --
# a scheduled task on the dev box is the easy way.
#
# One-time setup:
#   1. Cloudflare dashboard -> the chitramaya.org zone -> Overview: copy the
#      Zone ID (right-hand column).
#   2. My Profile -> API Tokens -> Create Token -> "Read analytics and logs"
#      template, or custom with Zone / Analytics / Read for this zone only.
#      Tokens go into the config file below, never into a chat.
#   3. Create %APPDATA%\ChitraMaya\cloudflare-stats.json:
#        { "zoneId": "<zone id>", "token": "<api token>" }
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\Get-DownloadStats.ps1            # last 7 days
#   ... -Days 3                      # shorter window
#   ... -Since 2026-10-14            # from the release day (the API may refuse old days; the ledger has them)
#   ... -Ledger D:\Stats\downloads.csv
#   ... -Host get.chitramaya.org     # default
#
# PURE ASCII output; PowerShell 5.1 compatible.

param(
  [int]$Days = 7,
  [string]$Since = "",
  [string]$SiteHost = "get.chitramaya.org",
  [string]$Config = (Join-Path $env:APPDATA "ChitraMaya\cloudflare-stats.json"),
  [string]$Ledger = (Join-Path $env:APPDATA "ChitraMaya\download-stats.csv"),
  [string]$LatestJson = "",            # default: https://<SiteHost>/latest.json
  [switch]$NoLedger
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# -- config ------------------------------------------------------------------
if (-not (Test-Path $Config)) {
  throw "Config not found: $Config`nCreate it with: { ""zoneId"": ""<zone id>"", ""token"": ""<api token with Zone/Analytics/Read>"" }"
}
$cfg = Get-Content $Config -Raw | ConvertFrom-Json
if (-not $cfg.zoneId -or -not $cfg.token) { throw "Config needs both zoneId and token: $Config" }
$headers = @{ Authorization = "Bearer $($cfg.token)"; "Content-Type" = "application/json" }
$endpoint = "https://api.cloudflare.com/client/v4/graphql"

# -- file sizes from latest.json --------------------------------------------
if (-not $LatestJson) { $LatestJson = "https://$SiteHost/latest.json" }
try {
  if ($LatestJson -match '^https?://') { $latest = Invoke-RestMethod -Uri $LatestJson -Method Get }
  else { $latest = Get-Content $LatestJson -Raw | ConvertFrom-Json }
} catch { throw "Could not read $LatestJson ($($_.Exception.Message)) -- pass -LatestJson <path to a local latest.json>" }
$relPrefix = ([uri]$latest.base).AbsolutePath          # /releases/1.80.00/
$sizes = @{}
foreach ($f in $latest.files) { $sizes["$relPrefix$($f.name)"] = [long]$f.size }
Write-Host "Release $($latest.version) ($($latest.date)): $($sizes.Count) files known from $LatestJson"

# -- window ------------------------------------------------------------------
$todayUtc = (Get-Date).ToUniversalTime().Date
if ($Since) { $start = ([datetime]::Parse($Since)).Date } else { $start = $todayUtc.AddDays(-($Days - 1)) }
$end = $todayUtc.AddDays(1)
Write-Host ("Window: {0:yyyy-MM-dd} .. {1:yyyy-MM-dd} (UTC days)" -f $start, $todayUtc)

# -- one GraphQL call per day (keeps each query inside the API's range limits)
$query = @'
query ($zone: string, $filter: filter) {
  viewer {
    zones(filter: { zoneTag: $zone }) {
      paths: httpRequestsAdaptiveGroups(limit: 200, filter: $filter, orderBy: [sum_edgeResponseBytes_DESC]) {
        count
        sum { edgeResponseBytes }
        dimensions { clientRequestPath edgeResponseStatus }
      }
      countries: httpRequestsAdaptiveGroups(limit: 50, filter: $filter, orderBy: [sum_edgeResponseBytes_DESC]) {
        count
        sum { edgeResponseBytes }
        dimensions { clientCountryName }
      }
    }
  }
}
'@

$rows = @()        # per day, per path (download files only)
$countryBytes = @{}
$countryReq = @{}
$day = $start
while ($day -lt $end) {
  $vars = @{
    zone = $cfg.zoneId
    filter = @{
      datetime_geq = $day.ToString("yyyy-MM-ddTHH:mm:ssZ")
      datetime_lt  = $day.AddDays(1).ToString("yyyy-MM-ddTHH:mm:ssZ")
      clientRequestHTTPHost = $SiteHost
      requestSource = "eyeball"
    }
  }
  $body = @{ query = $query; variables = $vars } | ConvertTo-Json -Depth 6 -Compress
  try {
    $resp = Invoke-RestMethod -Uri $endpoint -Method Post -Headers $headers -Body $body
  } catch {
    Write-Warning ("{0:yyyy-MM-dd}: API call failed ({1})" -f $day, $_.Exception.Message); $day = $day.AddDays(1); continue
  }
  if ($resp.errors) {
    $msg = ($resp.errors | ForEach-Object { $_.message }) -join "; "
    Write-Warning ("{0:yyyy-MM-dd}: {1}" -f $day, $msg)
    $day = $day.AddDays(1); continue
  }
  $zone = $resp.data.viewer.zones | Select-Object -First 1
  if (-not $zone) { Write-Warning ("{0:yyyy-MM-dd}: no zone in the response (zoneId or token scope?)" -f $day); $day = $day.AddDays(1); continue }

  $perPath = @{}
  foreach ($g in @($zone.paths)) {
    $p = $g.dimensions.clientRequestPath
    $st = [int]$g.dimensions.edgeResponseStatus
    if (-not $sizes.ContainsKey($p)) { continue }                 # only the release files
    if ($st -ne 200 -and $st -ne 206) { continue }                 # full and ranged responses only
    if (-not $perPath.ContainsKey($p)) { $perPath[$p] = @{ req = 0; bytes = [long]0 } }
    $perPath[$p].req += [long]$g.count
    $perPath[$p].bytes += [long]$g.sum.edgeResponseBytes
  }
  foreach ($p in $perPath.Keys) {
    $rows += [pscustomobject]@{
      date = $day.ToString("yyyy-MM-dd")
      file = Split-Path $p -Leaf
      requests = $perPath[$p].req
      bytes = $perPath[$p].bytes
      downloads = [math]::Round($perPath[$p].bytes / [double]$sizes[$p], 2)
    }
  }
  foreach ($g in @($zone.countries)) {
    $c = $g.dimensions.clientCountryName
    if (-not $c) { continue }
    if (-not $countryBytes.ContainsKey($c)) { $countryBytes[$c] = [long]0; $countryReq[$c] = [long]0 }
    $countryBytes[$c] += [long]$g.sum.edgeResponseBytes
    $countryReq[$c] += [long]$g.count
  }
  $day = $day.AddDays(1)
}

if ($rows.Count -eq 0) {
  Write-Host ""
  Write-Host "No release-file traffic in the window (or the API returned nothing -- see warnings above)."
} else {
  # -- per file ---------------------------------------------------------------
  Write-Host ""
  Write-Host "Per file (window total):"
  $byFile = $rows | Group-Object file | ForEach-Object {
    $b = ($_.Group | Measure-Object bytes -Sum).Sum
    $p = "$relPrefix$($_.Name)"
    [pscustomobject]@{
      file = $_.Name
      requests = [long](($_.Group | Measure-Object requests -Sum).Sum)
      GB_served = [math]::Round($b / 1e9, 2)
      downloads = [math]::Round($b / [double]$sizes[$p], 1)
    }
  } | Sort-Object file
  $byFile | Format-Table -AutoSize | Out-String -Width 120 | Write-Host

  # Edition roll-up: NVIDIA = the .exe stub is small, count the big volume (.7z.001)
  $nv = $byFile | Where-Object { $_.file -eq "ChitraMaya-install.7z.001" } | Select-Object -ExpandProperty downloads
  $amd = $byFile | Where-Object { $_.file -like "ChitraMaya-rocm-install*" } | Measure-Object downloads -Sum | Select-Object -ExpandProperty Sum
  $xpu = $byFile | Where-Object { $_.file -like "ChitraMaya-xpu-install*" } | Measure-Object downloads -Sum | Select-Object -ExpandProperty Sum
  $patch = $byFile | Where-Object { $_.file -like "ChitraMaya-patch-*" } | Measure-Object downloads -Sum | Select-Object -ExpandProperty Sum
  Write-Host ("Editions (estimated full downloads): NVIDIA {0}  AMD {1}  Intel {2}  patches {3}" -f
    [math]::Round(($nv + 0), 1), [math]::Round(($amd + 0), 1), [math]::Round(($xpu + 0), 1), [math]::Round(($patch + 0), 1))
  Write-Host "  (NVIDIA counted on the .7z.001 volume; a visitor who took only the .exe stub is not a download)"

  # -- per day ----------------------------------------------------------------
  Write-Host ""
  Write-Host "Per day (all release files):"
  $rows | Group-Object date | ForEach-Object {
    [pscustomobject]@{
      date = $_.Name
      requests = [long](($_.Group | Measure-Object requests -Sum).Sum)
      GB_served = [math]::Round((($_.Group | Measure-Object bytes -Sum).Sum) / 1e9, 2)
      downloads = [math]::Round((($_.Group | Measure-Object downloads -Sum).Sum), 1)
    }
  } | Sort-Object date | Format-Table -AutoSize | Out-String -Width 120 | Write-Host
}

# -- countries (whole host, page views included) -----------------------------
if ($countryBytes.Count -gt 0) {
  Write-Host "Top countries by bytes served (whole host, pages included):"
  $countryBytes.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 12 | ForEach-Object {
    [pscustomobject]@{ country = $_.Key; requests = $countryReq[$_.Key]; GB_served = [math]::Round($_.Value / 1e9, 2) }
  } | Format-Table -AutoSize | Out-String -Width 120 | Write-Host
}

# -- ledger ------------------------------------------------------------------
if (-not $NoLedger -and $rows.Count -gt 0) {
  $dir = Split-Path $Ledger -Parent
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $old = @()
  if (Test-Path $Ledger) { $old = @(Import-Csv $Ledger) }
  $seen = @{}
  foreach ($r in $rows) { $seen["$($r.date)|$($r.file)"] = $true }
  $keep = @($old | Where-Object { -not $seen.ContainsKey("$($_.date)|$($_.file)") })
  $all = @($keep) + @($rows | ForEach-Object { [pscustomobject]@{ date = $_.date; file = $_.file; requests = $_.requests; bytes = $_.bytes; downloads = $_.downloads } })
  $all | Sort-Object date, file | Export-Csv -NoTypeInformation -Encoding ASCII -Path $Ledger
  # Lifetime by edition: NVIDIA on the .7z.001 volume only (the .exe stub and .002
  # would count the same install three times), AMD and Intel on their installers.
  $lnv  = ($all | Where-Object { $_.file -eq "ChitraMaya-install.7z.001" } | Measure-Object downloads -Sum).Sum
  $lamd = ($all | Where-Object { $_.file -like "ChitraMaya-rocm-install*" -and $_.file -notlike "*.7z.00[2-9]" } | Measure-Object downloads -Sum).Sum
  $lxpu = ($all | Where-Object { $_.file -like "ChitraMaya-xpu-install*" -and $_.file -notlike "*.7z.00[2-9]" } | Measure-Object downloads -Sum).Sum
  $lpat = ($all | Where-Object { $_.file -like "ChitraMaya-patch-*" } | Measure-Object downloads -Sum).Sum
  Write-Host ("Ledger: {0} ({1} rows)" -f $Ledger, $all.Count)
  Write-Host ("Lifetime estimated installs: NVIDIA {0}  AMD {1}  Intel {2}  patches {3}  -- total {4}" -f
    [math]::Round(($lnv + 0), 1), [math]::Round(($lamd + 0), 1), [math]::Round(($lxpu + 0), 1), [math]::Round(($lpat + 0), 1),
    [math]::Round(($lnv + $lamd + $lxpu + $lpat), 1))
}
