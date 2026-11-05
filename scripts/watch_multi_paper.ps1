[CmdletBinding()]
param(
    [switch]$CheckOnce
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunnerPython = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv\Scripts\python.exe'))
$RunnerScript = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'scripts\run_multi_paper.py'))
$StatePath = Join-Path $ProjectRoot 'data\paper\portfolio.json'
$LogDirectory = Join-Path $ProjectRoot 'logs'
$HeartbeatLimit = [TimeSpan]::FromMinutes(5)
$CheckIntervalSeconds = 30
$TaskName = 'AltMeanrevPaperRunnerWatchdog'

if (-not (Test-Path -LiteralPath $RunnerPython -PathType Leaf)) {
    throw "Project Python runtime not found: $RunnerPython"
}
if (-not (Test-Path -LiteralPath $RunnerScript -PathType Leaf)) {
    throw "Paper runner script not found: $RunnerScript"
}
if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
    throw "Paper state not found: $StatePath"
}

function Write-WatchdogLog {
    param([Parameter(Mandatory)][string]$Message)

    $path = Join-Path $LogDirectory ("paper-runner-watchdog-{0}.log" -f (Get-Date -Format 'yyyyMMdd'))
    $line = "{0} | {1}{2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff zzz'), $Message, [Environment]::NewLine
    [System.IO.File]::AppendAllText($path, $line, [System.Text.UTF8Encoding]::new($false))
}

function Get-RunnerProcessTree {
    $all = @(Get-CimInstance -ClassName Win32_Process)
    $pythonProcesses = @($all | Where-Object { $_.Name -match '^python(?:w)?\.exe$' })
    $seedProcesses = @($pythonProcesses | Where-Object {
        ($_.ExecutablePath -ieq $RunnerPython -and $_.CommandLine -match 'run_multi_paper\.py') -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0)
    })

    $tracked = @{}
    foreach ($process in $seedProcesses) {
        $tracked[[string]$process.ProcessId] = $process
    }

    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $pythonProcesses) {
            $processId = [string]$process.ProcessId
            $parentId = [string]$process.ParentProcessId
            if ($tracked.ContainsKey($parentId) -and -not $tracked.ContainsKey($processId)) {
                $tracked[$processId] = $process
                $changed = $true
            }
        }
    }

    return @($tracked.Values)
}

function Start-PaperRunner {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $LogDirectory "paper-runner-watchdog-$stamp.stdout.log"
    $stderr = Join-Path $LogDirectory "paper-runner-watchdog-$stamp.stderr.log"
    $arguments = "-u `"$RunnerScript`" --poll 60 --top 20 --universe-refresh 300"

    $env:BOT_MODE = 'paper'
    $env:BINANCE_TESTNET = 'false'
    $process = Start-Process -FilePath $RunnerPython -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr

    $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
    $script:GraceUntil = (Get-Date).Add($HeartbeatLimit)
    $script:RetryDelaySeconds = [Math]::Min(1800, $script:RetryDelaySeconds * 2)
    $script:WasUnhealthy = $true
    Write-WatchdogLog "Started local Paper runner pid=$($process.Id); BOT_MODE=paper; BINANCE_TESTNET=false"
}

function Stop-PaperRunnerTree {
    param([Parameter(Mandatory)][object[]]$Processes)

    $trackedIds = @{}
    foreach ($process in $Processes) {
        $trackedIds[[string]$process.ProcessId] = $true
    }
    $roots = @($Processes | Where-Object { -not $trackedIds.ContainsKey([string]$_.ParentProcessId) })
    foreach ($root in $roots) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $root.ProcessId /T /F *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-WatchdogLog "taskkill reported exit code $LASTEXITCODE for runner pid=$($root.ProcessId)"
        }
    }
}

function Invoke-WatchdogCheck {
    $now = Get-Date
    $processes = @(Get-RunnerProcessTree)
    $stateTime = if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        (Get-Item -LiteralPath $StatePath).LastWriteTime
    } else {
        [DateTime]::MinValue
    }
    $heartbeatAge = $now - $stateTime
    $stale = $heartbeatAge -gt $HeartbeatLimit

    if ($processes.Count -eq 0) {
        if ($now -lt $script:NextAttemptAt) {
            return
        }
        Write-WatchdogLog 'Runner process not found; attempting local Paper restart'
        Start-PaperRunner
        return
    }

    if (-not $stale) {
        if ($script:WasUnhealthy) {
            Write-WatchdogLog "Runner heartbeat recovered; portfolio age=$([int]$heartbeatAge.TotalSeconds)s"
        }
        $script:RetryDelaySeconds = 60
        $script:NextAttemptAt = [DateTime]::MinValue
        $script:GraceUntil = [DateTime]::MinValue
        $script:WasUnhealthy = $false
        return
    }

    $latestProcessStart = $Processes | ForEach-Object { [DateTime]$_.CreationDate } |
        Measure-Object -Maximum | Select-Object -ExpandProperty Maximum
    if ($now -lt $script:GraceUntil -or ($now - $latestProcessStart) -lt $HeartbeatLimit) {
        return
    }
    if ($now -lt $script:NextAttemptAt) {
        return
    }

    Write-WatchdogLog "Runner found but portfolio heartbeat is stale ($([int]$heartbeatAge.TotalSeconds)s); restarting process tree"
    Stop-PaperRunnerTree -Processes $processes
    Start-Sleep -Seconds 3
    if (@(Get-RunnerProcessTree).Count -gt 0) {
        $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
        Write-WatchdogLog 'Runner process tree is still present after stop request; postponing another restart'
        return
    }
    Start-PaperRunner
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$script:RetryDelaySeconds = 60
$script:NextAttemptAt = [DateTime]::MinValue
$script:GraceUntil = [DateTime]::MinValue
$script:WasUnhealthy = $false
$mutex = [System.Threading.Mutex]::new($false, 'Local\AltMeanrevPaperRunnerWatchdog')
$ownsMutex = $false

try {
    try {
        $ownsMutex = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsMutex = $true
    }
    if (-not $ownsMutex) {
        Write-WatchdogLog 'Another watchdog instance already owns the mutex; exiting'
        exit 0
    }

    if ($CheckOnce) {
        Invoke-WatchdogCheck
        exit 0
    }

    Write-WatchdogLog "Watchdog started; heartbeat limit=$([int]$HeartbeatLimit.TotalSeconds)s; task=$TaskName"
    while ($true) {
        try {
            Invoke-WatchdogCheck
        } catch {
            Write-WatchdogLog "Watchdog check failed: $($_.Exception.GetType().Name): $($_.Exception.Message)"
        }
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
} finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
