# Collect-GpuEvents.ps1 -- pull the Windows events that explain a GPU falling off the bus
# into one text file, so nobody has to click through Event Viewer.
#
# Usage (PowerShell, no admin needed for System/Application logs):
#   powershell -ExecutionPolicy Bypass -File .\Collect-GpuEvents.ps1
#   powershell -ExecutionPolicy Bypass -File .\Collect-GpuEvents.ps1 -Days 14
#
# Writes  %USERPROFILE%\Desktop\GpuEvents-<yyyyMMdd-HHmmss>.txt  (path printed at the end)
# and shows a short summary on the console. Pure ASCII, read-only: it changes nothing.

param(
    [int]$Days = 7,
    [string]$Out = ""
)

$ErrorActionPreference = "Continue"
$since = (Get-Date).AddDays(-1 * $Days)
if ($Out -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Out = Join-Path ([Environment]::GetFolderPath("Desktop")) ("GpuEvents-" + $stamp + ".txt")
}

$lines = New-Object System.Collections.Generic.List[string]
function Add($s) { $script:lines.Add([string]$s) }
function Rule($t) { Add(""); Add(("=" * 78)); Add($t); Add(("=" * 78)) }

Add("GPU event collection  " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "  (last " + $Days + " days, since " + $since.ToString("yyyy-MM-dd HH:mm") + ")")
Add("Machine: " + $env:COMPUTERNAME + "   OS: " + (Get-CimInstance Win32_OperatingSystem).Caption)

# ---------------------------------------------------------------- 1. the card now
Rule("1. Display adapters right now (Get-PnpDevice)")
try {
    Get-PnpDevice -Class Display | ForEach-Object {
        $prob = ""
        try {
            $p = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName "DEVPKEY_Device_ProblemCode" -ErrorAction Stop
            $prob = "  problem_code=" + $p.Data
        } catch { }
        Add(("  {0,-10} {1}{2}" -f $_.Status, $_.FriendlyName, $prob))
        Add("             " + $_.InstanceId)
    }
    Add("  (Status Error / Unknown, or problem_code 43 = the driver reported a failure; 45 = device not present)")
} catch { Add("  Get-PnpDevice failed: " + $_.Exception.Message) }

Rule("2. nvidia-smi")
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    $cand = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    if (Test-Path $cand) { $smi = $cand } else { $smi = $null }
} else { $smi = $smi.Source }
if ($smi) {
    try { $o = & $smi 2>&1; $o | ForEach-Object { Add("  " + $_) } } catch { Add("  nvidia-smi failed: " + $_.Exception.Message) }
} else { Add("  nvidia-smi not found") }

# ---------------------------------------------------------------- 3. System log
Rule("3. System log -- GPU / PCIe / power events, oldest first")
$providers = @(
    "nvlddmkm",                                # NVIDIA display driver (Xid text lives here on Windows)
    "Display",                                 # 4101 = TDR: driver stopped responding and recovered
    "Microsoft-Windows-Kernel-Power",          # 41 = hard reset/crash; 137, 172 = power transitions
    "Microsoft-Windows-Kernel-PnP",            # 410/411/420 = device arrival/removal/surprise removal
    "Microsoft-Windows-WHEA-Logger",           # 17 corrected PCIe error, 18/19/20 uncorrected/fatal, 47 memory
    "Microsoft-Windows-Kernel-Boot",
    "EventLog",                                # 6008 = the previous shutdown was unexpected
    "Microsoft-Windows-Power-Troubleshooter",  # 1 = wake from sleep (what woke, when)
    "Microsoft-Windows-Kernel-Processor-Power",
    "PCI", "pci", "ACPI", "Microsoft-Windows-PCI"
)
$sys = @()
try {
    $sys = Get-WinEvent -FilterHashtable @{ LogName = "System"; StartTime = $since } -ErrorAction Stop |
        Where-Object { $providers -contains $_.ProviderName -or $_.Message -match "nvlddmkm|fallen off the bus|Xid|TDR|PCI Express|PCIe|display driver" }
} catch { Add("  Get-WinEvent System failed: " + $_.Exception.Message) }
$sys = @($sys | Sort-Object TimeCreated)
Add("  " + $sys.Count + " matching event(s)")
foreach ($e in $sys) {
    $msg = ""
    if ($e.Message) { $msg = ($e.Message -replace "\s+", " ").Trim() }
    if ($msg.Length -gt 400) { $msg = $msg.Substring(0, 400) + " ..." }
    Add(("  {0}  {1,-6} {2,-42} id={3,-5} {4}" -f $e.TimeCreated.ToString("yyyy-MM-dd HH:mm:ss"), $e.LevelDisplayName, $e.ProviderName, $e.Id, $msg))
}

# ---------------------------------------------------------------- 4. Application log
Rule("4. Application log -- LiveKernelEvent / Windows Error Reporting / NVIDIA services")
$app = @()
try {
    $app = Get-WinEvent -FilterHashtable @{ LogName = "Application"; StartTime = $since } -ErrorAction Stop |
        Where-Object { ($_.ProviderName -eq "Windows Error Reporting" -and $_.Message -match "LiveKernelEvent|nvlddmkm|VIDEO_") -or $_.ProviderName -match "^NVIDIA|nvcontainer|NvContainer" -or $_.Message -match "nvlddmkm|LiveKernelEvent" }
} catch { Add("  Get-WinEvent Application failed: " + $_.Exception.Message) }
$app = @($app | Sort-Object TimeCreated)
Add("  " + $app.Count + " matching event(s)")
foreach ($e in $app) {
    $msg = ""
    if ($e.Message) { $msg = ($e.Message -replace "\s+", " ").Trim() }
    # keep the LiveKernelEvent code and the first parameters, drop the report paths
    $m = [regex]::Match($msg, "LiveKernelEvent[^\s]*\s*(\S+)")
    if ($m.Success) { $msg = "LiveKernelEvent " + $m.Groups[1].Value + "  " + $msg.Substring(0, [Math]::Min(300, $msg.Length)) }
    if ($msg.Length -gt 400) { $msg = $msg.Substring(0, 400) + " ..." }
    Add(("  {0}  {1,-6} {2,-42} id={3,-5} {4}" -f $e.TimeCreated.ToString("yyyy-MM-dd HH:mm:ss"), $e.LevelDisplayName, $e.ProviderName, $e.Id, $msg))
}

# ---------------------------------------------------------------- 5. summary
Rule("5. Summary: count per provider / id")
$all = @($sys) + @($app)
$all | Group-Object ProviderName, Id | Sort-Object Count -Descending | ForEach-Object {
    Add(("  {0,5}  {1}" -f $_.Count, $_.Name))
}

# ---------------------------------------------------------------- 6. power settings that matter
Rule("6. Power plan and PCI Express link-state power management")
try {
    $scheme = (powercfg /GETACTIVESCHEME) 2>&1
    Add("  " + ($scheme -join " "))
    # subgroup PCI Express / setting Link State Power Management
    $q = (powercfg /Q SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5) 2>&1
    $q | ForEach-Object { Add("  " + $_) }
    Add("  (index 0 = Off, 1 = Moderate power savings, 2 = Maximum power savings. Cards that 'fall off the bus'")
    Add("   under load are often cured by Off; the driver's own idle Gen1 link speed is unrelated and harmless.)")
} catch { Add("  powercfg failed: " + $_.Exception.Message) }

# ---------------------------------------------------------------- 7. reading guide
Rule("7. How to read it")
Add("  nvlddmkm 'fallen off the bus' / Xid 79 ......... the card stopped answering on PCIe. Power delivery or the")
Add("                                                    PCIe link itself (PSU, 8-pin cable, slot seating, riser),")
Add("                                                    occasionally a dying card. Not an application fault.")
Add("  WHEA-Logger 17 (corrected PCIe error), recurring  link errors the hardware fixed: signal integrity on the slot.")
Add("                                                    Reseat, try another slot, no riser, PCIe link power Off.")
Add("  WHEA-Logger 18/19/20 ............................ uncorrected / fatal hardware error, same suspects, worse.")
Add("  Display 4101 .................................... TDR: the driver hung and Windows reset it. Software/driver.")
Add("  LiveKernelEvent 141 ............................. video hardware error (VIDEO_ENGINE_TIMEOUT) -- GPU side.")
Add("  LiveKernelEvent 117 ............................. TDR timeout -- driver side.")
Add("  Kernel-Power 41 / EventLog 6008 ................. the machine reset without a clean shutdown (PSU / hard hang).")
Add("  Kernel-PnP 411 / 420 ............................ surprise device removal -- when it names the display adapter,")
Add("                                                    that is the moment the card left the bus.")
Add("  Power-Troubleshooter 1 .......................... a sleep/resume happened; a card that fails right after one")
Add("                                                    points at ASPM / power management, not the PSU.")
Add("")
Add("  The order of events in the minute before the loss is what matters: WHEA errors first = link; a clean")
Add("  'fallen off the bus' with nothing before it = power; 4101 first = driver.")

$lines | Set-Content -Path $Out -Encoding ASCII
Write-Host ""
Write-Host ("wrote " + $Out)
Write-Host ("System events: " + $sys.Count + "   Application events: " + $app.Count)
$all | Group-Object ProviderName, Id | Sort-Object Count -Descending | Select-Object -First 12 | ForEach-Object {
    Write-Host ("  {0,5}  {1}" -f $_.Count, $_.Name)
}
