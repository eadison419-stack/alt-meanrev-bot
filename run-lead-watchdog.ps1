[CmdletBinding()]
param([switch]$Status, [switch]$Stop, [switch]$AdoptManualPositions)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$watchdog = Join-Path $projectRoot 'scripts\watch_lead_live.ps1'
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $watchdog -PathType Leaf)) { throw "Lead watchdog script not found: $watchdog" }
if ($Status -and $Stop) { throw 'Use either -Status or -Stop' }
if ($AdoptManualPositions -and ($Status -or $Stop)) {
    throw '-AdoptManualPositions applies only when arming the watchdog'
}
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(?:powershell|pwsh)\.exe$' -and $_.CommandLine -and
    $_.CommandLine.IndexOf($watchdog, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    $_.CommandLine -match '(?<!\S)-Armed(?!\S)'
})
if ($Stop) {
    foreach ($process in $active) { Stop-Process -Id $process.ProcessId -ErrorAction Stop }
    Write-Output "Stopped Lead watchdog count=$($active.Count); Lead Runner was not stopped."
    exit 0
}
if ($Status) {
    Write-Output "Lead watchdog processes=$($active.Count)"
    $statusAdoption = $false
    if ($active.Count -eq 1) {
        $statusAdoption = $active[0].CommandLine -match '(?<!\S)-AdoptManualPositions(?!\S)'
    }
    $statusArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $watchdog, '-CheckOnce')
    if ($statusAdoption) { $statusArguments += '-AdoptManualPositions' }
    & $powershell @statusArguments
    if ($active.Count -ne 1) { exit 1 }
    exit $LASTEXITCODE
}
if ($active.Count -gt 1) { throw 'Multiple Lead watchdog processes are present; inspect them before enabling another' }
if ($active.Count -gt 0) {
    $activeAdoption = $active[0].CommandLine -match '(?<!\S)-AdoptManualPositions(?!\S)'
    if ($activeAdoption -ne [bool]$AdoptManualPositions) {
        throw 'Lead watchdog is already armed with a different manual-adoption mode; stop it before changing modes.'
    }
    Write-Output "Lead watchdog is already armed (PID $(($active.ProcessId | Sort-Object) -join ', ')); no duplicate started."
    exit 0
}
Write-Host 'This arms automatic recovery of the real Lead Runner until this watchdog stops.'
if ($AdoptManualPositions) {
    Write-Host 'This also enables reconstruction and strategy management of untracked Lead positions.'
}
Write-Host 'Windows restart ends this authorization. Stop the watchdog before intentionally stopping the Runner.'
$confirmation = Read-Host 'Type LEAD to arm the live watchdog'
if ($confirmation -cne 'LEAD') { Write-Output 'Not confirmed; Lead watchdog was not started.'; exit 2 }

$oldConfirmation = $env:BOT_LEAD_WATCHDOG_CONFIRMATION
try {
    $env:BOT_LEAD_WATCHDOG_CONFIRMATION = 'LEAD'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $logs = Join-Path $projectRoot 'logs'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Armed' -f $watchdog
    if ($AdoptManualPositions) { $arguments += ' -AdoptManualPositions' }
    $startedAt = [DateTime]::UtcNow
    $process = Start-Process -FilePath $powershell -ArgumentList $arguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logs "lead-watchdog-$stamp.stdout.log") `
        -RedirectStandardError (Join-Path $logs "lead-watchdog-$stamp.stderr.log")
} finally {
    [Environment]::SetEnvironmentVariable('BOT_LEAD_WATCHDOG_CONFIRMATION', $oldConfirmation, 'Process')
}
$deadline = [DateTime]::UtcNow.AddSeconds(20)
$pidFile = Join-Path $projectRoot 'data\live\lead_watchdog.pid'
do {
    if ((Test-Path -LiteralPath $pidFile -PathType Leaf) -and
        (Get-Item -LiteralPath $pidFile).LastWriteTimeUtc -ge $startedAt.AddSeconds(-1)) {
        $watchdogId = 0
        if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw), [ref]$watchdogId) -and $watchdogId -eq $process.Id) {
            $process.Refresh()
            if (-not $process.HasExited) {
                Write-Output "Lead watchdog is armed and running in the background (PID $watchdogId). You can close PowerShell."
                Write-Output 'Use .\run-lead-watchdog.ps1 -Status to check or -Stop to disarm while keeping the Runner running.'
                exit 0
            }
        }
    }
    $process.Refresh()
    if ($process.HasExited) { throw "Lead watchdog exited; inspect logs\lead-watchdog-$stamp.stderr.log" }
    Start-Sleep -Milliseconds 500
} while ([DateTime]::UtcNow -lt $deadline)
throw 'Lead watchdog readiness was not confirmed; inspect logs before starting another instance'
