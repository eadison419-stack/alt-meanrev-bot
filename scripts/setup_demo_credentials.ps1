[CmdletBinding()]
param(
    [switch]$Replace
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'windows_credential_store.ps1')

$target = 'AltMeanrevBot:BinanceUsdM:Demo'
$existing = $null
$hasExisting = $false
try {
    $existing = [AltMeanrevCredentialStore]::Read($target)
    $hasExisting = $true
} catch [System.ComponentModel.Win32Exception] {
    if ($_.Exception.NativeErrorCode -ne 1168) { throw }
} finally {
    if ($existing) { $existing[0] = $null; $existing[1] = $null }
    $existing = $null
}

if ($hasExisting -and -not $Replace) {
    throw 'Demo credentials already exist in Windows Credential Manager. Use -Replace only when rotating this Demo API key.'
}

function ConvertFrom-SecurePrompt {
    param([Parameter(Mandatory)][Security.SecureString]$Value)
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

$keySecure = Read-Host 'USDⓈ-M Demo API Key' -AsSecureString
$secretSecure = Read-Host 'USDⓈ-M Demo API Secret' -AsSecureString
$key = $null
$secret = $null
try {
    $key = ConvertFrom-SecurePrompt $keySecure
    $secret = ConvertFrom-SecurePrompt $secretSecure
    if ([string]::IsNullOrWhiteSpace($key) -or [string]::IsNullOrWhiteSpace($secret)) {
        throw 'Both Demo credential fields are required.'
    }
    if ($key.Contains("`n") -or $key.Contains("`r") -or $secret.Contains("`n") -or $secret.Contains("`r")) {
        throw 'Credential values must not contain line breaks.'
    }
    [AltMeanrevCredentialStore]::Write($target, $key, $secret)
    Write-Output 'USDⓈ-M Demo credentials were stored in Windows Credential Manager for this Windows user.'
} finally {
    $key = $null
    $secret = $null
    $keySecure.Dispose()
    $secretSecure.Dispose()
}
