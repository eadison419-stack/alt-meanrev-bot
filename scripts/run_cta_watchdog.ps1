# CTA paper watchdog: keep the runner and dashboard alive in the background.
# Checks process liveness and state freshness; restarts anything that died.
# Runs forever. Register it in the Startup folder via install_cta_autostart.ps1.

$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonw    = Join-Path $root '.venv\Scripts\pythonw.exe'
$runner     = Join-Path $root 'scripts\run_cta_paper.py'
$dash       = Join-Path $root 'scripts\run_cta_dashboard_bg.py'
$statePath  = Join-Path $root 'data\paper\cta_portfolio.json'
$pidFile    = Join-Path $root 'data\paper\cta_paper.pid'
$dashPidFile= Join-Path $root 'data\paper\cta_dashboard.pid'
$log        = Join-Path $root 'logs\watchdog.log'

# Record our own pid so stop_cta_local.bat can kill us too.
$wdPid = Join-Path $root 'data\paper\cta_watchdog.pid'
New-Item -ItemType Directory -Force -Path (Split-Path $wdPid) | Out-Null
# Single-instance guard: a second watchdog would double-restart the runner.
if (Test-Path $wdPid) {
    $existing = Get-Content $wdPid
    if ($existing -and (Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue)) {
        Add-Content -Path $log -Value ("{0} watchdog already running (pid {1}); exiting" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $existing)
        exit 0
    }
}
Set-Content -Path $wdPid -Value $PID

function Write-Log([string]$msg) {
    Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
}

function Test-RunnerProcessAlive {
    if (-not (Test-Path $pidFile)) { return $false }
    $raw = Get-Content $pidFile
    if (-not $raw) { return $false }
    $p = Get-Process -Id ([int]$raw) -ErrorAction SilentlyContinue
    return ($null -ne $p)
}

function Test-RunnerHealthy {
    if (-not (Test-RunnerProcessAlive)) { return $false }
    # If the state file exists but hasn't updated in 20 min, the runner is frozen.
    if (Test-Path $statePath) {
        $age = (Get-Date) - (Get-Item $statePath).LastWriteTime
        if ($age.TotalMinutes -gt 20) { return $false }
    }
    return $true
}

function Start-Runner {
    if (Test-Path $pidFile) {
        $old = Get-Content $pidFile
        if ($old) { Stop-Process -Id ([int]$old) -Force -ErrorAction SilentlyContinue }
    }
    Start-Process -FilePath $pythonw -ArgumentList @($runner, '--be-mult', '2.0', '--breadth-min', '0.65') -WindowStyle Hidden
    Write-Log 'runner started'
}

function Test-DashHealthy {
    $conn = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $conn)
}

function Start-Dash {
    if (Test-Path $dashPidFile) {
        $old = Get-Content $dashPidFile
        if ($old) { Stop-Process -Id ([int]$old) -Force -ErrorAction SilentlyContinue }
    }
    # --no-open: don't pop a browser tab on every auto-restart.
    Start-Process -FilePath $pythonw -ArgumentList @($dash, '--state', $statePath, '--no-open') -WindowStyle Hidden
    Write-Log 'dashboard started'
}

Write-Log 'watchdog started'
while ($true) {
    if (-not (Test-RunnerHealthy)) {
        Write-Log 'runner unhealthy, restarting'
        Start-Runner
        Start-Sleep -Seconds 30
    }
    if (-not (Test-DashHealthy)) {
        Write-Log 'dashboard unhealthy, restarting'
        Start-Dash
        Start-Sleep -Seconds 15
    }
    Start-Sleep -Seconds 60
}
