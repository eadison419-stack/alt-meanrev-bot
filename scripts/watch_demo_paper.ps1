[CmdletBinding()]
param(
    [switch]$CheckOnce
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'windows_credential_store.ps1')

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunnerPython = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv\Scripts\python.exe'))
$RunnerScript = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'scripts\run_mr_candidate_paper.py'))
$LegacyRunnerScript = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot 'scripts\run_multi_paper.py'))
$StatePath = Join-Path $ProjectRoot 'data\paper\demo_portfolio.json'
$LogDirectory = Join-Path $ProjectRoot 'logs'
$HeartbeatLimit = [TimeSpan]::FromMinutes(5)
$CheckIntervalSeconds = 30
$CredentialTarget = 'AltMeanrevBot:BinanceUsdM:Demo'

foreach ($required in @($RunnerPython, $RunnerScript, $StatePath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Demo runner file not found: $required"
    }
}

function Write-WatchdogLog {
    param([Parameter(Mandatory)][string]$Message)
    $path = Join-Path $LogDirectory ("demo-runner-watchdog-{0}.log" -f (Get-Date -Format 'yyyyMMdd'))
    $line = "{0} | {1}{2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff zzz'), $Message, [Environment]::NewLine
    [System.IO.File]::AppendAllText($path, $line, [System.Text.UTF8Encoding]::new($false))
}

function Get-DemoRunnerProcesses {
    $all = @(Get-CimInstance -ClassName Win32_Process)
    $pythonProcesses = @($all | Where-Object { $_.Name -match '^python(?:w)?\.exe$' })
    $seeds = @($pythonProcesses | Where-Object {
        $_.CommandLine -and (
            ($_.ExecutablePath -ieq $RunnerPython -and
             $_.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
            $_.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0
        ) -and $_.CommandLine.IndexOf($StatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
    $tracked = @{}
    foreach ($process in $seeds) { $tracked[[string]$process.ProcessId] = $process }
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
    @($tracked.Values)
}

function Get-LegacyDemoRunnerProcesses {
    @(Get-CimInstance -ClassName Win32_Process | Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and $_.CommandLine -and
        $_.CommandLine.IndexOf($LegacyRunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine.IndexOf($StatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0
    })
}

function Get-DemoRunnerRoots {
    param([Parameter(Mandatory)][object[]]$Processes)
    $ids = @{}
    foreach ($process in $Processes) { $ids[[string]$process.ProcessId] = $true }
    @($Processes | Where-Object { -not $ids.ContainsKey([string]$_.ParentProcessId) })
}

function Get-DemoLedgerHealth {
    $reasons = [System.Collections.Generic.List[string]]::new()
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $snapshot = $state.exchange_snapshot
        if ($state.execution_mode -ne 'demo_usdm') { $reasons.Add('execution_mode') }
        if ($state.strategy_variant -ne 'ema50_deviation_candidate') { $reasons.Add('strategy_variant') }
        if ($state.execution_status -ne 'synchronized') { $reasons.Add('execution_status') }
        if ($snapshot.host -ne 'https://demo-fapi.binance.com') { $reasons.Add('snapshot_host') }
        if (-not $snapshot.at) { $reasons.Add('snapshot_time_missing') }
        if ($null -eq $snapshot.positions) { $reasons.Add('snapshot_positions_missing') }
        if ($reasons.Count -gt 0) {
            return [pscustomobject]@{ Healthy = $false; Reasons = @($reasons.ToArray()) }
        }
        $snapshotAt = [DateTimeOffset]::Parse([string]$snapshot.at)
        $age = [DateTimeOffset]::UtcNow - $snapshotAt.ToUniversalTime()
        if ($age.TotalSeconds -lt -30 -or $age -gt $HeartbeatLimit) {
            $reasons.Add('snapshot_stale')
        }
        $localSymbols = @($state.positions.PSObject.Properties | ForEach-Object { $_.Name } | Sort-Object)
        $exchangeSymbols = @($snapshot.positions.PSObject.Properties | ForEach-Object { $_.Name } | Sort-Object)
        if (($localSymbols -join '|') -ne ($exchangeSymbols -join '|')) {
            $reasons.Add('position_symbols_mismatch')
        }
        return [pscustomobject]@{
            Healthy = ($reasons.Count -eq 0)
            Reasons = @($reasons.ToArray())
        }
    } catch {
        return [pscustomobject]@{ Healthy = $false; Reasons = @('ledger_read_or_parse_error') }
    }
}

function Test-FreshSynchronizedDemoLedger {
    $health = Get-DemoLedgerHealth
    return [bool]$health.Healthy
}

function Start-DemoRunner {
    try {
        $credential = [AltMeanrevCredentialStore]::Read($CredentialTarget)
    } catch [System.ComponentModel.Win32Exception] {
        if ($_.Exception.NativeErrorCode -eq 1168) {
            if (-not $script:MissingCredentialLogged) {
                Write-WatchdogLog 'Demo credential entry is not present in Windows Credential Manager; runner start deferred'
                $script:MissingCredentialLogged = $true
            }
            $script:NextAttemptAt = (Get-Date).AddSeconds(60)
            return
        }
        throw
    }

    $oldKey = $env:BINANCE_API_KEY
    $oldSecret = $env:BINANCE_API_SECRET
    $oldMode = $env:BOT_MODE
    $oldTestnet = $env:BINANCE_TESTNET
    try {
        $env:BINANCE_API_KEY = $credential[0]
        $env:BINANCE_API_SECRET = $credential[1]
        $env:BOT_MODE = 'paper'
        $env:BINANCE_TESTNET = 'true'
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $stdout = Join-Path $LogDirectory "demo-runner-$stamp.stdout.log"
        $stderr = Join-Path $LogDirectory "demo-runner-$stamp.stderr.log"
        $script:LastRunnerStderrPath = $stderr
        $arguments = "-u `"$RunnerScript`" --demo --poll 60 --max-positions 5 --state `"$StatePath`""
        $process = Start-Process -FilePath $RunnerPython -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $script:ManagedRunnerPid = $process.Id
        $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
        $script:GraceUntil = (Get-Date).Add($HeartbeatLimit)
        $script:MissingCredentialLogged = $false
        Write-WatchdogLog "Started USDⓈ-M Demo runner pid=$($process.Id); strategy=ema50_deviation_candidate; BOT_MODE=paper; BINANCE_TESTNET=true"
    } finally {
        if ($null -eq $oldKey) { Remove-Item Env:\BINANCE_API_KEY -ErrorAction SilentlyContinue } else { $env:BINANCE_API_KEY = $oldKey }
        if ($null -eq $oldSecret) { Remove-Item Env:\BINANCE_API_SECRET -ErrorAction SilentlyContinue } else { $env:BINANCE_API_SECRET = $oldSecret }
        if ($null -eq $oldMode) { Remove-Item Env:\BOT_MODE -ErrorAction SilentlyContinue } else { $env:BOT_MODE = $oldMode }
        if ($null -eq $oldTestnet) { Remove-Item Env:\BINANCE_TESTNET -ErrorAction SilentlyContinue } else { $env:BINANCE_TESTNET = $oldTestnet }
        if ($credential) { $credential[0] = $null; $credential[1] = $null }
    }
}

function Get-RunnerBanUntil {
    if (-not $script:LastRunnerStderrPath -or
        -not (Test-Path -LiteralPath $script:LastRunnerStderrPath -PathType Leaf)) {
        return $null
    }
    try {
        $stderr = Get-Content -LiteralPath $script:LastRunnerStderrPath -Raw
        $match = [regex]::Match($stderr, 'banned until\s+(\d+)')
        if (-not $match.Success) { return $null }
        return [DateTimeOffset]::FromUnixTimeMilliseconds([long]$match.Groups[1].Value).LocalDateTime
    } catch {
        return $null
    }
}

function Test-DemoCredentialStored {
    try {
        $credential = [AltMeanrevCredentialStore]::Read($CredentialTarget)
        if ($credential) { $credential[0] = $null; $credential[1] = $null }
        return $true
    } catch [System.ComponentModel.Win32Exception] {
        if ($_.Exception.NativeErrorCode -eq 1168) { return $false }
        throw
    }
}

function Stop-ManagedDemoRunner {
    param([Parameter(Mandatory)][int]$ProcessId)
    $process = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    $isRunner = $false
    if ($process -and $process.Name -match '^python(?:w)?\.exe$' -and $process.CommandLine) {
        $scriptArgument = $process.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0
        $hasDemoState = $process.CommandLine.IndexOf($StatePath, [StringComparison]::OrdinalIgnoreCase) -ge 0
        $isRunner = $hasDemoState -and (
            ($process.ExecutablePath -ieq $RunnerPython -and $scriptArgument) -or
            $process.CommandLine.IndexOf($RunnerScript, [StringComparison]::OrdinalIgnoreCase) -ge 0
        )
    }
    if (-not $isRunner) {
        Write-WatchdogLog 'Managed runner identity could not be confirmed; refusing to terminate any process'
        return $false
    }
    & "$env:SystemRoot\System32\taskkill.exe" /PID $ProcessId /T /F *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-WatchdogLog "taskkill reported exit code $LASTEXITCODE for managed Demo runner pid=$ProcessId"
        return $false
    }
    return $true
}

function Invoke-WatchdogCheck {
    $now = Get-Date
    $legacyProcesses = @(Get-LegacyDemoRunnerProcesses)
    if ($legacyProcesses.Count -gt 0) {
        if (-not $script:UnmanagedRunnerLogged) {
            Write-WatchdogLog "Legacy multi-strategy runner is still attached to the Demo ledger (processes=$($legacyProcesses.Count)); refusing to start the candidate runner"
            $script:UnmanagedRunnerLogged = $true
        }
        return
    }
    $script:UnmanagedRunnerLogged = $false
    $processes = @(Get-DemoRunnerProcesses)
    if ($processes.Count -eq 0) {
        if ($script:ManagedRunnerPid -ne 0) {
            $banUntil = Get-RunnerBanUntil
            $script:ManagedRunnerPid = 0
            if ($banUntil -and $banUntil -gt (Get-Date)) {
                $script:NextAttemptAt = $banUntil.AddSeconds(30)
                $script:RetryDelaySeconds = [Math]::Max(300, $script:RetryDelaySeconds)
                Write-WatchdogLog "Runner exited during a Binance public API ban; retry deferred until $($script:NextAttemptAt.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
                return
            }
            $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
            $script:RetryDelaySeconds = [Math]::Min(1800, $script:RetryDelaySeconds * 2)
            Write-WatchdogLog "Managed runner exited; retry backoff is $([int]($script:NextAttemptAt - (Get-Date)).TotalSeconds)s"
        }
        if ($now -lt $script:NextAttemptAt) { return }
        try {
            Start-DemoRunner
        } catch {
            Write-WatchdogLog "Demo runner start failed safely: $($_.Exception.GetType().Name)"
            $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
            $script:RetryDelaySeconds = [Math]::Min(1800, $script:RetryDelaySeconds * 2)
        }
        return
    }

    $managed = @($processes | Where-Object { [int]$_.ProcessId -eq $script:ManagedRunnerPid })
    if ($managed.Count -eq 0) {
        $roots = @(Get-DemoRunnerRoots -Processes $processes)
        $ledgerEligible = Test-FreshSynchronizedDemoLedger
        if ($roots.Count -eq 1 -and $ledgerEligible) {
            $script:ManagedRunnerPid = [int]$roots[0].ProcessId
            $managed = @($roots[0])
            Write-WatchdogLog "Adopted existing runner pid=$script:ManagedRunnerPid after matching a fresh synchronized Demo ledger snapshot"
        } elseif (-not $script:UnmanagedRunnerLogged) {
            Write-WatchdogLog "A runner without a fresh synchronized Demo ledger is present (roots=$($roots.Count), ledgerEligible=$ledgerEligible); refusing duplicate start or termination"
            $script:UnmanagedRunnerLogged = $true
        }
        if ($managed.Count -eq 0) { return }
    }

    $script:UnmanagedRunnerLogged = $false
    $stateTime = if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        (Get-Item -LiteralPath $StatePath).LastWriteTime
    } else {
        [DateTime]::MinValue
    }
    $heartbeatAge = $now - $stateTime
    $ledgerHealth = Get-DemoLedgerHealth
    $ledgerHealthy = [bool]$ledgerHealth.Healthy
    if ($heartbeatAge -le $HeartbeatLimit) {
        if ($ledgerHealthy) {
            $script:GraceUntil = [DateTime]::MinValue
            $script:RetryDelaySeconds = 60
            $script:LastLedgerWarning = $null
        } else {
            $reasonKey = $ledgerHealth.Reasons -join ','
            if ($script:LastLedgerWarning -ne $reasonKey) {
                Write-WatchdogLog "Runner heartbeat is fresh but Demo ledger health check failed ($reasonKey); leaving runner alive to preserve position management"
                $script:LastLedgerWarning = $reasonKey
            }
        }
        return
    }

    $startedAt = [DateTime]$managed[0].CreationDate
    if ($now -lt $script:GraceUntil -or ($now - $startedAt) -lt $HeartbeatLimit) { return }
    if ($now -lt $script:NextAttemptAt) { return }

    $healthReasons = $ledgerHealth.Reasons -join ','
    Write-WatchdogLog "Demo runner heartbeat is stale (heartbeat=$([int]$heartbeatAge.TotalSeconds)s, ledgerHealthy=$ledgerHealthy, reasons=$healthReasons); restarting managed runner pid=$script:ManagedRunnerPid"
    try {
        if (-not (Test-DemoCredentialStored)) {
            Write-WatchdogLog 'Credential Manager entry is missing; refusing to stop the current runner because restart credentials are unavailable'
            $script:NextAttemptAt = (Get-Date).AddSeconds(60)
            return
        }
    } catch {
        Write-WatchdogLog "Credential Manager check failed safely: $($_.Exception.GetType().Name); refusing runner termination"
        $script:NextAttemptAt = (Get-Date).AddSeconds(60)
        return
    }
    if (Stop-ManagedDemoRunner -ProcessId $script:ManagedRunnerPid) {
        $script:ManagedRunnerPid = 0
        $script:NextAttemptAt = (Get-Date).AddSeconds(3)
    } else {
        $script:NextAttemptAt = (Get-Date).AddSeconds($script:RetryDelaySeconds)
        $script:RetryDelaySeconds = [Math]::Min(1800, $script:RetryDelaySeconds * 2)
    }
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$script:ManagedRunnerPid = 0
$script:NextAttemptAt = [DateTime]::MinValue
$script:GraceUntil = [DateTime]::MinValue
$script:RetryDelaySeconds = 60
$script:LastRunnerStderrPath = $null
$script:MissingCredentialLogged = $false
$script:UnmanagedRunnerLogged = $false
$script:LastLedgerWarning = $null
$mutex = [System.Threading.Mutex]::new($false, 'Local\AltMeanrevDemoPaperRunnerWatchdog')
$ownsMutex = $false

try {
    try {
        $ownsMutex = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsMutex = $true
    }
    if (-not $ownsMutex) {
        Write-WatchdogLog 'Another Demo watchdog instance already owns the mutex; exiting'
        exit 0
    }
    if ($CheckOnce) {
        Invoke-WatchdogCheck
        exit 0
    }
    Write-WatchdogLog "Demo watchdog started; heartbeat limit=$([int]$HeartbeatLimit.TotalSeconds)s"
    while ($true) {
        try {
            Invoke-WatchdogCheck
        } catch {
            Write-WatchdogLog "Demo watchdog check failed safely: $($_.Exception.GetType().Name)"
        }
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
