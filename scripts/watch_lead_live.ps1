[CmdletBinding()]
param([switch]$Armed, [switch]$CheckOnce)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RunnerPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RunnerScript = Join-Path $ProjectRoot 'scripts\run_mr_candidate_paper.py'
$StatePath = Join-Path $ProjectRoot 'data\live\lead_portfolio.json'
$LogDirectory = Join-Path $ProjectRoot 'logs'
$HeartbeatLimit = [TimeSpan]::FromMinutes(5)
$CredentialTarget = 'AltMeanrevBot:BinanceUsdM:Lead'

function Write-LeadWatchdogLog {
    param([string]$Message)
    $path = Join-Path $LogDirectory ('lead-runner-watchdog-{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
    $line = '{0} | {1}{2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff zzz'), $Message, [Environment]::NewLine
    [IO.File]::AppendAllText($path, $line, [Text.UTF8Encoding]::new($false))
}

function Get-LeadRunnerProcesses {
    # Include live candidates using another state path so they block duplicate starts.
    $python = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(?:w)?\.exe$' })
    $tracked = @{}
    foreach ($process in $python) {
        if ($process.CommandLine -and $process.CommandLine -match 'run_mr_candidate_paper\.py' -and
            $process.CommandLine -match '(?<!\S)--live(?!\S)') {
            $tracked[[string]$process.ProcessId] = $process
        }
    }
    do {
        $changed = $false
        foreach ($process in $python) {
            if ($tracked.ContainsKey([string]$process.ParentProcessId) -and
                -not $tracked.ContainsKey([string]$process.ProcessId)) {
                $tracked[[string]$process.ProcessId] = $process
                $changed = $true
            }
        }
    } while ($changed)
    @($tracked.Values)
}

function Get-LeadRunnerRoots {
    param([object[]]$Processes)
    $ids = @{}
    foreach ($process in $Processes) { $ids[[string]$process.ProcessId] = $true }
    @($Processes | Where-Object { -not $ids.ContainsKey([string]$_.ParentProcessId) })
}

function Test-ExpectedLeadRunner {
    param($Process)
    if (-not $Process.CommandLine -or $Process.CommandLine -notmatch '(?<!\S)--live(?!\S)' -or
        $Process.CommandLine -match '(?<!\S)--demo(?!\S)') { return $false }
    $scriptMatch = [regex]::Match($Process.CommandLine, '(?:"([^"]*run_mr_candidate_paper\.py)"|(\S*run_mr_candidate_paper\.py))')
    $stateMatch = [regex]::Match($Process.CommandLine, '--state(?:=|\s+)(?:"([^"]+)"|(\S+))')
    if (-not $scriptMatch.Success -or -not $stateMatch.Success) { return $false }
    $paths = @()
    foreach ($match in @($scriptMatch, $stateMatch)) {
        $value = if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
        if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $ProjectRoot $value }
        $paths += [IO.Path]::GetFullPath($value)
    }
    return ($paths[0] -ieq $RunnerScript -and $paths[1] -ieq $StatePath)
}

function Get-LeadLedgerHealth {
    $reasons = [Collections.Generic.List[string]]::new()
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $snapshot = $state.exchange_snapshot
        if ($state.execution_mode -ne 'lead_usdm') { $reasons.Add('execution_mode') }
        if ($state.strategy_variant -ne 'ema50_daily_gate_closed_h1_mean_reversion') { $reasons.Add('strategy_variant') }
        if ($state.execution_status -ne 'synchronized') { $reasons.Add('execution_status') }
        if ($snapshot.host -ne 'https://fapi.binance.com') { $reasons.Add('snapshot_host') }
        $snapshotAge = [DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse([string]$snapshot.at).ToUniversalTime()
        if ($snapshotAge.TotalSeconds -lt -30 -or $snapshotAge -gt $HeartbeatLimit) { $reasons.Add('snapshot_stale') }
        $heartbeatAge = [DateTime]::UtcNow - (Get-Item -LiteralPath $StatePath).LastWriteTimeUtc
        if ($heartbeatAge.TotalSeconds -lt -30 -or $heartbeatAge -gt $HeartbeatLimit) { $reasons.Add('heartbeat_stale') }
        $local = @($state.positions.PSObject.Properties | ForEach-Object { $_.Name } | Sort-Object)
        $remote = @($snapshot.positions.PSObject.Properties | ForEach-Object { $_.Name } | Sort-Object)
        if (($local -join '|') -ne ($remote -join '|')) { $reasons.Add('position_symbols_mismatch') }
        if ($state.inconsistencies.Count -ne 0) { $reasons.Add('inconsistencies') }
        if ($local.Count -gt 0) {
            if ($state.protective_stop_audit.status -ne 'verified' -or
                $state.protective_stop_audit.mismatch_count -ne 0) { $reasons.Add('protective_stop_audit') }
            $auditAge = [DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse([string]$state.protective_stop_audit.at).ToUniversalTime()
            if ($auditAge.TotalSeconds -lt -30 -or $auditAge -gt $HeartbeatLimit) { $reasons.Add('protective_stop_stale') }
        }
        return [pscustomobject]@{ Healthy = ($reasons.Count -eq 0); Reasons = @($reasons.ToArray()) }
    } catch {
        return [pscustomobject]@{ Healthy = $false; Reasons = @('ledger_read_or_parse_error') }
    }
}

function Set-LeadRetryBackoff {
    param([string]$Reason, [int]$MinimumDelay = 60)
    $delay = [Math]::Min(1800, [Math]::Max($MinimumDelay, $script:RetryDelaySeconds))
    $script:NextAttemptAt = (Get-Date).AddSeconds($delay)
    $script:RetryDelaySeconds = [Math]::Min(1800, $delay * 2)
    Write-LeadWatchdogLog "$Reason; retry in ${delay}s"
}

function Start-LeadRunner {
    if (-not $script:IsArmed) { throw 'Lead watchdog has not been armed' }
    if (@(Get-LeadRunnerProcesses).Count -gt 0) {
        Write-LeadWatchdogLog 'A live Runner appeared before recovery; duplicate start deferred'
        return
    }
    $credential = $null
    $saved = @{}
    foreach ($name in @('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BOT_MODE', 'BINANCE_TESTNET', 'BOT_LEAD_BACKGROUND_CONFIRMATION')) {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    try {
        $credential = [AltMeanrevCredentialStore]::Read($CredentialTarget)
        if (-not $credential -or $credential.Count -lt 2 -or
            [string]::IsNullOrWhiteSpace($credential[0]) -or [string]::IsNullOrWhiteSpace($credential[1])) {
            throw 'Lead credential entry is missing or incomplete'
        }
        $env:BINANCE_API_KEY = $credential[0]
        $env:BINANCE_API_SECRET = $credential[1]
        $env:BOT_MODE = 'live'
        $env:BINANCE_TESTNET = 'false'
        $env:BOT_LEAD_BACKGROUND_CONFIRMATION = 'LEAD'
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $stdout = Join-Path $LogDirectory "lead-runner-$stamp.stdout.log"
        $stderr = Join-Path $LogDirectory "lead-runner-$stamp.stderr.log"
        $arguments = '-u "{0}" --live --background-confirmed --poll 60 --state "{1}"' -f $RunnerScript, $StatePath
        if (@(Get-LeadRunnerProcesses).Count -gt 0) {
            Write-LeadWatchdogLog 'A live Runner appeared during recovery preparation; duplicate start deferred'
            return
        }
        $script:LastStartedAt = [DateTimeOffset]::UtcNow
        $process = Start-Process -FilePath $RunnerPython -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $script:ManagedRunnerPid = $process.Id
        $script:GraceUntil = (Get-Date).Add($HeartbeatLimit)
        Write-LeadWatchdogLog "Started confirmed Lead runner pid=$($process.Id); poll=60s; mode=live; host=fapi.binance.com"
    } finally {
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
        if ($credential) { $credential[0] = $null; if ($credential.Count -gt 1) { $credential[1] = $null } }
    }
}

function Invoke-LeadWatchdogCheck {
    if (-not $script:IsArmed) { throw 'Lead watchdog has not been armed' }
    $now = Get-Date
    $processes = @(Get-LeadRunnerProcesses)
    if ($processes.Count -eq 0) {
        if ($script:ManagedRunnerPid -ne 0) {
            $script:ManagedRunnerPid = 0
            $minimumDelay = 60
            try {
                $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
                $fatal = @($state.events | Where-Object {
                    $_.event -eq 'account_startup_failed' -and
                    [DateTimeOffset]::Parse([string]$_.at) -ge $script:LastStartedAt -and
                    $_.error -in @('AuthenticationError', 'PermissionDenied', 'RateLimitExceeded', 'DDoSProtection')
                })
                if ($fatal.Count -gt 0) { $minimumDelay = 300 }
            } catch { }
            Set-LeadRetryBackoff -Reason 'Managed Lead runner exited' -MinimumDelay $minimumDelay
            return
        }
        if ($now -lt $script:NextAttemptAt) { return }
        try { Start-LeadRunner }
        catch { Set-LeadRetryBackoff -Reason "Lead start failed: $($_.Exception.GetType().Name)" }
        return
    }
    $roots = @(Get-LeadRunnerRoots $processes)
    if ($roots.Count -ne 1 -or -not (Test-ExpectedLeadRunner $roots[0])) {
        if ($script:LastWarning -ne 'runner_identity') {
            Write-LeadWatchdogLog "Multiple or incompatible live runner roots=$($roots.Count); no process will be started or stopped"
            $script:LastWarning = 'runner_identity'
        }
        return
    }
    if ($script:ManagedRunnerPid -ne [int]$roots[0].ProcessId) {
        $script:ManagedRunnerPid = [int]$roots[0].ProcessId
        Write-LeadWatchdogLog "Adopted existing Lead runner pid=$script:ManagedRunnerPid; process was not restarted"
    }
    $health = Get-LeadLedgerHealth
    if ($health.Healthy) {
        if ($script:LastWarning) { Write-LeadWatchdogLog 'Lead account and heartbeat health recovered' }
        $script:RetryDelaySeconds = 60
        $script:LastWarning = $null
    } elseif ($now -ge $script:GraceUntil) {
        $key = $health.Reasons -join ','
        if ($script:LastWarning -ne $key) {
            Write-LeadWatchdogLog "Lead runner is alive but ledger health failed ($key); preserving position management; no forced restart"
            $script:LastWarning = $key
        }
    }
}

if ($CheckOnce) {
    $processes = @(Get-LeadRunnerProcesses)
    $roots = @(Get-LeadRunnerRoots $processes)
    $health = Get-LeadLedgerHealth
    $expected = $roots.Count -eq 1 -and (Test-ExpectedLeadRunner $roots[0])
    [pscustomobject]@{ RunnerRoots = $roots.Count; ExpectedRunner = $expected; LedgerHealthy = $health.Healthy; Reasons = $health.Reasons } | ConvertTo-Json -Compress
    if (-not $expected -or -not $health.Healthy) { exit 1 }
    exit 0
}
if (-not $Armed -or $env:BOT_LEAD_WATCHDOG_CONFIRMATION -cne 'LEAD') {
    [Console]::Error.WriteLine('Lead watchdog requires run-lead-watchdog.ps1 and LEAD arming confirmation')
    exit 2
}
[Environment]::SetEnvironmentVariable('BOT_LEAD_WATCHDOG_CONFIRMATION', $null, 'Process')
foreach ($required in @($RunnerPython, $RunnerScript, $StatePath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required Lead file not found: $required" }
}
. (Join-Path $PSScriptRoot 'windows_credential_store.ps1')
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$script:IsArmed = $true
$script:ManagedRunnerPid = 0
$script:NextAttemptAt = [DateTime]::MinValue
$script:RetryDelaySeconds = 60
$script:LastStartedAt = [DateTimeOffset]::MaxValue
$script:GraceUntil = [DateTime]::MinValue
$script:LastWarning = $null
$mutex = [Threading.Mutex]::new($false, 'Local\AltMeanrevLeadRunnerWatchdog')
$ownsMutex = $false
try {
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { Write-LeadWatchdogLog 'Another Lead watchdog owns the mutex; exiting'; exit 0 }
    [IO.File]::WriteAllText((Join-Path $ProjectRoot 'data\live\lead_watchdog.pid'), [string]$PID)
    Write-LeadWatchdogLog 'Lead watchdog armed for this process lifetime; check=30s; heartbeat=300s; live-process termination disabled'
    while ($true) {
        try { Invoke-LeadWatchdogCheck }
        catch { Write-LeadWatchdogLog "Lead watchdog check failed safely: $($_.Exception.GetType().Name)" }
        Start-Sleep -Seconds 30
    }
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
