param([switch]$AdoptManualPositions)

$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\run_mr_candidate_paper.py'
$credentialStore = Join-Path $projectRoot 'scripts\windows_credential_store.ps1'
$statePath = Join-Path $projectRoot 'data\live\lead_portfolio.json'
$logPath = Join-Path $projectRoot 'logs\mr_lead_live.log'
$stdoutDirectory = Join-Path $projectRoot 'logs'
foreach ($required in @($python, $runner, $credentialStore)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Lead runner file not found: $required"
    }
}

$activeLiveRunners = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:w)?\.exe$' -and $_.CommandLine -and
    $_.CommandLine.IndexOf($runner, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    $_.CommandLine.IndexOf($statePath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    $_.CommandLine -match '(?<!\S)--live(?!\S)'
})
if ($activeLiveRunners.Count -gt 0) {
    $processIds = ($activeLiveRunners.ProcessId | Sort-Object -Unique) -join ', '
    throw "A Lead runner is already active (PID $processIds). Stop it with Ctrl+C in its current PowerShell window, then rerun this script."
}

function Assert-LeadWatchdogInactive {
    $watchdogPath = Join-Path $projectRoot 'scripts\watch_lead_live.ps1'
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(?:powershell|pwsh)\.exe$' -and $_.CommandLine -and
        $_.CommandLine.IndexOf($watchdogPath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine -match '(?<!\S)-Armed(?!\S)'
    })
    if ($active.Count -gt 0) { throw 'Lead watchdog is active; use its -Status or -Stop command before starting another Runner.' }
}
Assert-LeadWatchdogInactive
. $credentialStore
$credential = $null
$savedEnvironment = @{}
foreach ($name in @('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'BOT_MODE', 'BINANCE_TESTNET',
                   'BOT_LEAD_BACKGROUND_CONFIRMATION')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$backgroundProcess = $null
try {
    $credential = [AltMeanrevCredentialStore]::Read('AltMeanrevBot:BinanceUsdM:Lead')
    if (-not $credential -or $credential.Count -lt 2 -or
        [string]::IsNullOrWhiteSpace($credential[0]) -or
        [string]::IsNullOrWhiteSpace($credential[1])) {
        throw 'Binance Lead Portfolio credentials are missing from this Windows user Credential Manager.'
    }

    $confirmation = Read-Host '确认后台启动实盘机器人请输入 LEAD'
    if ($confirmation -cne 'LEAD') {
        Write-Output '未确认，已退出；没有启动后台机器人。'
        exit 2
    }
    Assert-LeadWatchdogInactive

    $env:BINANCE_API_KEY = $credential[0]
    $env:BINANCE_API_SECRET = $credential[1]
    $env:BOT_MODE = 'live'
    $env:BINANCE_TESTNET = 'false'
    $env:BOT_LEAD_BACKGROUND_CONFIRMATION = 'LEAD'

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdoutPath = Join-Path $stdoutDirectory "mr_lead_background-$stamp.stdout.log"
    $stderrPath = Join-Path $stdoutDirectory "mr_lead_background-$stamp.stderr.log"
    $adoptionArgument = if ($AdoptManualPositions) { ' --adopt-manual-positions' } else { '' }
    $argumentLine = "-u `"$runner`" --live --background-confirmed --poll 60 --state `"$statePath`"$adoptionArgument"
    $backgroundProcess = Start-Process -FilePath $python -ArgumentList $argumentLine `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    if ($credential) { $credential[0] = $null; $credential[1] = $null }
    $credential = $null
}

$startDeadline = [DateTime]::UtcNow.AddSeconds(120)
$runnerPidPath = Join-Path $projectRoot 'data\live\ema50_lead_live.pid'
$runnerPid = $null
$ready = $false
while ([DateTime]::UtcNow -lt $startDeadline) {
    if (Test-Path -LiteralPath $runnerPidPath -PathType Leaf) {
        $pidValue = 0
        if ([int]::TryParse((Get-Content -LiteralPath $runnerPidPath -Raw), [ref]$pidValue)) {
            $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
            if ($candidate -and $candidate.Name -match '^python(?:w)?\.exe$' -and
                $candidate.CommandLine -and
                $candidate.CommandLine.IndexOf($runner, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $candidate.CommandLine.IndexOf($statePath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $candidate.CommandLine -match '(?<!\S)--live(?!\S)' -and
                (($candidate.CommandLine -match '(?<!\S)--adopt-manual-positions(?!\S)') -eq [bool]$AdoptManualPositions) -and
                (Test-Path -LiteralPath $statePath -PathType Leaf)) {
                try {
                    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
                    $snapshotAt = [DateTimeOffset]::Parse([string]$state.exchange_snapshot.at)
                    $snapshotAge = [DateTimeOffset]::UtcNow - $snapshotAt.ToUniversalTime()
                    $ready = $state.execution_mode -eq 'lead_usdm' -and
                        $state.execution_status -eq 'synchronized' -and
                        $state.exchange_snapshot.host -eq 'https://fapi.binance.com' -and
                        $state.strategy_variant -eq 'ema50_daily_gate_closed_h1_mean_reversion' -and
                        $state.inconsistencies.Count -eq 0 -and
                        $snapshotAge.TotalSeconds -ge -30 -and $snapshotAge.TotalSeconds -lt 60
                    if ($ready) { $runnerPid = $pidValue; break }
                } catch {
                    $ready = $false
                }
            }
        }
    }
    Start-Sleep -Seconds 2
}

if (-not $ready) {
    $processNow = Get-CimInstance Win32_Process -Filter "ProcessId=$($backgroundProcess.Id)" -ErrorAction SilentlyContinue
    if (-not $processNow) {
        throw "Background runner exited before account synchronization completed. Review $logPath and logs\mr_lead_background-$stamp.stderr.log."
    }
    throw "Background runner has not completed account synchronization. It remains running as launcher PID $($backgroundProcess.Id); review $logPath before closing this window."
}

Write-Output "Lead Portfolio runner is synchronized and running in the background (PID $runnerPid)."
Write-Output 'You can close this PowerShell window. The runner will continue until it is stopped or Windows shuts down.'
Write-Output "Runner log: $logPath"
