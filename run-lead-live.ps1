$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$runner = Join-Path $root 'scripts\run_mr_candidate_paper.py'
$credentialStore = Join-Path $root 'scripts\windows_credential_store.ps1'
foreach ($required in @($python, $runner, $credentialStore)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Lead runner file not found: $required"
    }
}

. $credentialStore
$credential = $null
$oldKey = $env:BINANCE_API_KEY
$oldSecret = $env:BINANCE_API_SECRET
$oldMode = $env:BOT_MODE
$oldTestnet = $env:BINANCE_TESTNET
$exitCode = 1
try {
    $credential = [AltMeanrevCredentialStore]::Read('AltMeanrevBot:BinanceUsdM:Lead')
    if (-not $credential -or $credential.Count -lt 2 -or
        [string]::IsNullOrWhiteSpace($credential[0]) -or
        [string]::IsNullOrWhiteSpace($credential[1])) {
        throw 'Binance Lead Portfolio credentials are missing from this Windows user Credential Manager.'
    }

    $env:BINANCE_API_KEY = $credential[0]
    $env:BINANCE_API_SECRET = $credential[1]
    $env:BOT_MODE = 'live'
    $env:BINANCE_TESTNET = 'false'

    $runnerArgs = @('--live') + @($args)
    if ($runnerArgs -notcontains '--state' -and
        -not ($runnerArgs | Where-Object { $_ -like '--state=*' })) {
        $defaultState = Join-Path $root 'data\live\lead_portfolio.json'
        $runnerArgs += @('--state', $defaultState)
    }

    & $python $runner @runnerArgs
    $exitCode = $LASTEXITCODE
}
finally {
    if ($null -eq $oldKey) { Remove-Item Env:\BINANCE_API_KEY -ErrorAction SilentlyContinue } else { $env:BINANCE_API_KEY = $oldKey }
    if ($null -eq $oldSecret) { Remove-Item Env:\BINANCE_API_SECRET -ErrorAction SilentlyContinue } else { $env:BINANCE_API_SECRET = $oldSecret }
    if ($null -eq $oldMode) { Remove-Item Env:\BOT_MODE -ErrorAction SilentlyContinue } else { $env:BOT_MODE = $oldMode }
    if ($null -eq $oldTestnet) { Remove-Item Env:\BINANCE_TESTNET -ErrorAction SilentlyContinue } else { $env:BINANCE_TESTNET = $oldTestnet }
    if ($credential) { $credential[0] = $null; $credential[1] = $null }
}

exit $exitCode
