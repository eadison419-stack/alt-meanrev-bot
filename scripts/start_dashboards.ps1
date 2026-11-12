[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$DashboardScript = Join-Path $ProjectRoot 'scripts\run_dashboard.py'
$LogDirectory = Join-Path $ProjectRoot 'logs'
$Dashboards = @(
    [pscustomobject]@{
        Name = 'Lead'; Port = 8769; Mode = 'lead_usdm'
        HostUrl = 'https://fapi.binance.com'
        State = Join-Path $ProjectRoot 'data\live\lead_portfolio.json'
    }
)

foreach ($required in @($Python, $DashboardScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Dashboard file not found: $required"
    }
}
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Write-DashboardLog {
    param([string]$Message)
    $logPath = Join-Path $LogDirectory ('dashboard-startup-{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
    $line = '{0} | {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff zzz'), $Message
    [IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Write-Host $line
}

function Get-DashboardListener {
    param($Target)
    # Enumerating first avoids treating a free port as a terminating CIM error.
    @(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object LocalPort -eq $Target.Port)
}

function Confirm-Dashboard {
    param($Target, $Listener)
    if ($Listener.Count -ne 1 -or $Listener[0].LocalAddress -ne '127.0.0.1') {
        throw "Port $($Target.Port) is not owned by one localhost listener"
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($Listener[0].OwningProcess)"
    if (-not $process -or $process.Name -notmatch '^python(?:w)?\.exe$' -or
        $process.CommandLine -notmatch 'run_dashboard\.py') {
        throw "Port $($Target.Port) belongs to a different program; refusing duplicate start"
    }
    $match = [regex]::Match($process.CommandLine, '--state(?:=|\s+)(?:"([^"]+)"|(\S+))')
    if (-not $match.Success) { throw "Port $($Target.Port) has no identifiable state argument" }
    $stateArgument = if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
    if (-not [IO.Path]::IsPathRooted($stateArgument)) {
        $stateArgument = Join-Path $ProjectRoot $stateArgument
    }
    if ([IO.Path]::GetFullPath($stateArgument) -ine $Target.State) {
        throw "Port $($Target.Port) uses a different ledger; refusing duplicate start"
    }
    $state = Invoke-RestMethod -Uri "http://127.0.0.1:$($Target.Port)/api/state" -TimeoutSec 10
    if ($state.execution_mode -ne $Target.Mode -or $state.execution_host -ne $Target.HostUrl) {
        throw "Port $($Target.Port) returned a different account mode or host"
    }
    Write-DashboardLog "$($Target.Name) ready: port=$($Target.Port), pid=$($process.ProcessId), mode=$($Target.Mode)"
}

$mutex = [Threading.Mutex]::new($false, 'Local\AltMeanrevDashboardsStartup')
$ownsMutex = $false
$failed = $false
try {
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) {
        Write-DashboardLog 'Another Dashboard launcher is active; exiting without duplicate starts'
        exit 0
    }
    foreach ($target in $Dashboards) {
        try {
            if (-not (Test-Path -LiteralPath $target.State -PathType Leaf)) {
                throw "Ledger not found: $($target.State); no ledger or Runner will be created"
            }
            $ledger = Get-Content -LiteralPath $target.State -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ledger.execution_mode -ne $target.Mode) {
                throw "Ledger mode does not match $($target.Name) port $($target.Port)"
            }
            $listener = @(Get-DashboardListener $target)
            if ($listener.Count -gt 0) {
                Confirm-Dashboard $target $listener
                continue
            }
            if ($CheckOnly) { throw "Dashboard port $($target.Port) is not running" }

            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
            $stdout = Join-Path $LogDirectory "dashboard-$($target.Port)-$stamp.stdout.log"
            $stderr = Join-Path $LogDirectory "dashboard-$($target.Port)-$stamp.stderr.log"
            $arguments = '-u "{0}" --state "{1}" --port {2} --no-open' -f $DashboardScript, $target.State, $target.Port
            $process = Start-Process -FilePath $Python -ArgumentList $arguments `
                -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            Write-DashboardLog "Started $($target.Name): launcher_pid=$($process.Id), port=$($target.Port), stderr=$stderr"
            $deadline = [DateTime]::UtcNow.AddSeconds(30)
            do {
                Start-Sleep -Milliseconds 500
                $listener = @(Get-DashboardListener $target)
                $process.Refresh()
                if ($listener.Count -eq 0 -and $process.HasExited) {
                    throw "Dashboard exited before listening; inspect $stderr"
                }
            } while ($listener.Count -eq 0 -and [DateTime]::UtcNow -lt $deadline)
            if ($listener.Count -eq 0) { throw "Dashboard port $($target.Port) did not become ready within 30 seconds" }
            Confirm-Dashboard $target $listener
        } catch {
            $failed = $true
            Write-DashboardLog "$($target.Name) failed: $($_.Exception.Message)"
        }
    }
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
if ($failed) { exit 1 }
