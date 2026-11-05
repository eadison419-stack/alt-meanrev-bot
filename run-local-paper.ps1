$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$runner = Join-Path $root 'scripts\run_multi_paper.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project environment not found. Run .\install.ps1 first.'
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw 'Paper runner not found at scripts\run_multi_paper.py.'
}

$previousBotMode = $env:BOT_MODE
$previousTestnet = $env:BINANCE_TESTNET
$exitCode = 1
try {
    $env:BOT_MODE = 'paper'
    $env:BINANCE_TESTNET = 'false'
    & $python $runner @args
    $exitCode = $LASTEXITCODE
}
finally {
    $env:BOT_MODE = $previousBotMode
    $env:BINANCE_TESTNET = $previousTestnet
}

exit $exitCode
