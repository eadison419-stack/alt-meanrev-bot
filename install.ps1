$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

if (-not (Test-Path -LiteralPath (Join-Path $root 'pyproject.toml'))) {
    throw 'Run this installer from the extracted package root.'
}
$venvPath = Join-Path $root '.venv'
if (Test-Path -LiteralPath $venvPath) {
    throw "A .venv already exists at $venvPath. Use a fresh extraction directory; this installer will not replace or clean it."
}

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
$pythonExe = $null
$pythonArgs = @()
if ($pythonLauncher) {
    foreach ($version in @('3.12', '3.11')) {
        & $pythonLauncher.Source "-$version" --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $pythonLauncher.Source
            $pythonArgs = @("-$version")
            break
        }
    }
}
if (-not $pythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $versionText = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $versionText -in @('3.11', '3.12')) {
            $pythonExe = $pythonCommand.Source
        }
    }
}
if (-not $pythonExe) {
    throw 'Python 3.11 or 3.12 was not found. Install Python for this Windows user, then rerun install.ps1.'
}

Write-Host "Creating project-local virtual environment with $pythonExe $($pythonArgs -join ' ')"
& $pythonExe @pythonArgs -m venv $venvPath
if ($LASTEXITCODE -ne 0) { throw 'Python failed to create .venv.' }

$localPython = Join-Path $venvPath 'Scripts\python.exe'
$uvExe = Join-Path $venvPath 'Scripts\uv.exe'
Write-Host 'Installing uv inside .venv; network access is required.'
& $localPython -m pip install --disable-pip-version-check 'uv>=0.6,<1'
if ($LASTEXITCODE -ne 0) { throw 'Could not install uv into the project-local .venv.' }

Write-Host 'Installing locked runtime dependencies from uv.lock.'
& $uvExe sync --locked --no-dev
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed. Review the error and network access, then rerun in a clean package directory.' }

Write-Host 'Checking configuration import; the bot will not be started.'
& $localPython -c "from bot.config import load_settings; s=load_settings(); print(f'Configuration loaded: {s.exchange} / {s.market_type} / {s.settle}, timezone={s.timezone}')"
if ($LASTEXITCODE -ne 0) { throw 'Configuration import check failed.' }

Write-Host 'Install complete. No runner, account API, ledger, order, or watchdog was started.'
Write-Host 'To run local Paper manually: .\run-local-paper.ps1'
