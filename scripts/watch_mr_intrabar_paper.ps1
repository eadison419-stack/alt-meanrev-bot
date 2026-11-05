[CmdletBinding()]
param(
    [switch]$CheckOnce
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunnerPython = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv\Scripts\python.exe'))
$RunnerScript = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'scripts\run_mr_candidate_paper.py'))
$StatePath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'data\paper\ema50_intrabar_portfolio.json'))
$LogDirectory = Join-Path $ProjectRoot 'logs'
$HeartbeatLimit = [TimeSpan]::FromMinutes(5)
$CheckIntervalSeconds = 20

foreach ($required in @($RunnerPython, $RunnerScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required local Paper runner file not found: $required"
    }
}

function Write-WatchdogLog {
    param([Parameter(Mandatory)][string]$Message)
    $path = Join-Path $LogDirectory ("mr-intrabar-watchdog-{0}.log" -f (Get-Date -Format 'yyyyMMdd'))
    $line = "{0} | {1}{2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff zzz'), $Message, [Environment]::NewLine
    [System.IO.File]::AppendAllText($path, $line, [System.Text.UTF8Encoding]::new($false))
}

function Get-IntrabarRunnerProcesses {
    @(Get-CimInstance -ClassName Win32_Process | Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and $_.CommandLine -and
        $_.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine.IndexOf($StatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine.IndexOf('--intrabar-entries', [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine.IndexOf('--demo', [StringComparison]::OrdinalIgnoreCase) -lt 0
    })
}

function Get-RunnerRoots {
    param([Parameter(Mandatory)][object[]]$Processes)
    $ids = @{}
    foreach ($process in $Processes) { $ids[[string]$process.ProcessId] = $true }
    @($Processes | Where-Object { -not $ids.ContainsKey([string]$_.ParentProcessId) })
}

function Start-IntrabarRunner {
    $oldKey = $env:BINANCE_API_KEY
    $oldSecret = $env:BINANCE_API_SECRET
    $oldMode = $env:BOT_MODE
    $oldTestnet = $env:BINANCE_TESTNET
    try {
        Remove-Item Env:\BINANCE_API_KEY -ErrorAction SilentlyContinue
        Remove-Item Env:\BINANCE_API_SECRET -ErrorAction SilentlyContinue
        $env:BOT_MODE = 'paper'
        $env:BINANCE_TESTNET = 'false'
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $stdout = Join-Path $LogDirectory "mr-intrabar-$stamp.stdout.log"
        $stderr = Join-Path $LogDirectory "mr-intrabar-$stamp.stderr.log"
        $arguments = "-u `"$RunnerScript`" --poll 15 --max-positions 5 --equity 5000 --intrabar-entries --state `"$StatePath`""
        $process = Start-Process -FilePath $RunnerPython -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $script:ManagedRunnerPid = $process.Id
        $script:GraceUntil = (Get-Date).Add($HeartbeatLimit)
        $script:NextAttemptAt = (Get-Date).AddSeconds(60)
        Write-WatchdogLog "Started isolated local Paper runner pid=$($process.Id); strategy=ema50_daily_gate_closed_h1_mean_reversion; poll=15s; live_stop_monitor=on; equity=5000; max_positions=5"
    } finally {
        if ($null -eq $oldKey) { Remove-Item Env:\BINANCE_API_KEY -ErrorAction SilentlyContinue } else { $env:BINANCE_API_KEY = $oldKey }
        if ($null -eq $oldSecret) { Remove-Item Env:\BINANCE_API_SECRET -ErrorAction SilentlyContinue } else { $env:BINANCE_API_SECRET = $oldSecret }
        if ($null -eq $oldMode) { Remove-Item Env:\BOT_MODE -ErrorAction SilentlyContinue } else { $env:BOT_MODE = $oldMode }
        if ($null -eq $oldTestnet) { Remove-Item Env:\BINANCE_TESTNET -ErrorAction SilentlyContinue } else { $env:BINANCE_TESTNET = $oldTestnet }
    }
}

function Stop-ManagedIntrabarRunner {
    param([Parameter(Mandatory)][int]$ProcessId)
    $process = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    $isRunner = $process -and $process.Name -match '^python(?:w)?\.exe$' -and $process.CommandLine -and
        $process.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $process.CommandLine.IndexOf($StatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $process.CommandLine.IndexOf('--intrabar-entries', [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $process.CommandLine.IndexOf('--demo', [StringComparison]::OrdinalIgnoreCase) -lt 0
    if (-not $isRunner) {
        Write-WatchdogLog 'Runner identity could not be confirmed; refusing to terminate any process'
        return $false
    }
    & "$env:SystemRoot\System32\taskkill.exe" /PID $ProcessId /T /F *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-WatchdogLog "taskkill reported exit code $LASTEXITCODE for local Paper runner pid=$ProcessId"
        return $false
    }
    return $true
}

function Invoke-WatchdogCheck {
    $now = Get-Date
    $processes = @(Get-IntrabarRunnerProcesses)
    if ($processes.Count -eq 0) {
        if ($now -lt $script:NextAttemptAt) { return }
        try {
            Start-IntrabarRunner
        } catch {
            Write-WatchdogLog "Local Paper runner start failed: $($_.Exception.GetType().Name)"
            $script:NextAttemptAt = (Get-Date).AddSeconds(60)
        }
        return
    }

    $roots = @(Get-RunnerRoots -Processes $processes)
    if ($roots.Count -ne 1) {
        if (-not $script:MultipleRootsLogged) {
            Write-WatchdogLog "Found $($roots.Count) matching runner roots; refusing to start or terminate a runner"
            $script:MultipleRootsLogged = $true
        }
        return
    }
    $script:MultipleRootsLogged = $false
    $runner = $roots[0]
    $script:ManagedRunnerPid = [int]$runner.ProcessId
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
        if ($now -lt $script:GraceUntil) { return }
        $heartbeatAge = $HeartbeatLimit.Add([TimeSpan]::FromSeconds(1))
    } else {
        $heartbeatAge = $now - (Get-Item -LiteralPath $StatePath).LastWriteTime
    }
    if ($heartbeatAge -le $HeartbeatLimit) {
        $script:GraceUntil = [DateTime]::MinValue
        $script:NextAttemptAt = (Get-Date).AddSeconds(60)
        return
    }

    $startedAt = [DateTime]$runner.CreationDate
    if ($now -lt $script:GraceUntil -or ($now - $startedAt) -lt $HeartbeatLimit) { return }
    if ($now -lt $script:NextAttemptAt) { return }
    Write-WatchdogLog "Local Paper runner heartbeat is stale ($([int]$heartbeatAge.TotalSeconds)s); restarting pid=$script:ManagedRunnerPid"
    if (Stop-ManagedIntrabarRunner -ProcessId $script:ManagedRunnerPid) {
        $script:ManagedRunnerPid = 0
        $script:NextAttemptAt = (Get-Date).AddSeconds(3)
    } else {
        $script:NextAttemptAt = (Get-Date).AddSeconds(60)
    }
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$script:ManagedRunnerPid = 0
$script:NextAttemptAt = [DateTime]::MinValue
$script:GraceUntil = (Get-Date).Add($HeartbeatLimit)
$script:MultipleRootsLogged = $false
$mutex = [System.Threading.Mutex]::new($false, 'Local\AltMeanrevIntrabarPaperWatchdog')
$ownsMutex = $false
try {
    try {
        $ownsMutex = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsMutex = $true
    }
    if (-not $ownsMutex) {
        Write-WatchdogLog 'Another intrabar Paper watchdog already owns the mutex; exiting'
        exit 0
    }
    if ($CheckOnce) {
        Invoke-WatchdogCheck
        exit 0
    }
    Write-WatchdogLog "Intrabar local Paper watchdog started; heartbeat limit=$([int]$HeartbeatLimit.TotalSeconds)s"
    while ($true) {
        try {
            Invoke-WatchdogCheck
        } catch {
            Write-WatchdogLog "Watchdog check failed: $($_.Exception.GetType().Name)"
        }
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
