import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "watch_lead_live.ps1"
SHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(SHELL is None, reason="Windows PowerShell integration checks")


def _quote(path):
    return str(path).replace("'", "''")


def _run(tmp_path, functions, body):
    names = ",".join(f"'{name}'" for name in functions)
    source = (
        "$ErrorActionPreference='Stop'\nSet-StrictMode -Version Latest\n"
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)\n"
        "$tokens=$null;$errors=$null\n"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{_quote(SOURCE)}',"
        "[ref]$tokens,[ref]$errors)\n"
        "if($errors.Count -gt 0){throw 'PowerShell parse failed'}\n"
        f"foreach($name in @({names})){{\n"
        "$function=$ast.Find({param($node) $node -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true)\n"
        "if(-not $function){throw 'Function not found'}\n"
        "Invoke-Expression $function.Extent.Text\n}\n"
    )
    script = tmp_path / "check.ps1"
    script.write_text(source + body, encoding="utf-8-sig")
    result = subprocess.run(
        [SHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("problem", [None, "snapshot_stale", "heartbeat_stale", "snapshot_host",
                                    "execution_status", "position_symbols_mismatch", "protective_stop_audit",
                                    "protective_stop_stale"])
def test_lead_health_checks_actual_snapshot_age_and_utf8(tmp_path, problem):
    symbol = "牛来/USDT:USDT"
    state = {
        "execution_mode": "lead_usdm", "strategy_variant": "ema50_daily_gate_closed_h1_mean_reversion",
        "execution_status": "synchronized", "positions": {symbol: {}}, "inconsistencies": [],
        "exchange_snapshot": {"host": "https://fapi.binance.com",
                              "at": datetime.now(timezone.utc).isoformat(), "positions": {symbol: {}}},
        "protective_stop_audit": {"status": "verified", "mismatch_count": 0,
                                  "at": datetime.now(timezone.utc).isoformat()},
    }
    if problem == "snapshot_stale":
        state["exchange_snapshot"]["at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    elif problem == "snapshot_host":
        state["exchange_snapshot"]["host"] = "https://demo-fapi.binance.com"
    elif problem == "execution_status":
        state["execution_status"] = "inconsistent"
    elif problem == "position_symbols_mismatch":
        state["exchange_snapshot"]["positions"] = {}
    elif problem == "protective_stop_audit":
        state["protective_stop_audit"]["mismatch_count"] = 1
    elif problem == "protective_stop_stale":
        state["protective_stop_audit"]["at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    raw = json.dumps(state, ensure_ascii=True).replace(
        json.dumps(symbol, ensure_ascii=True), json.dumps(symbol, ensure_ascii=False), 1,
    )
    ledger = tmp_path / "state.json"
    ledger.write_text(raw, encoding="utf-8")
    if problem == "heartbeat_stale":
        old = datetime.now().timestamp() - 600
        os.utime(ledger, (old, old))
    result = _run(tmp_path, ["Get-LeadLedgerHealth"],
                  f"$StatePath='{_quote(ledger)}'\n$HeartbeatLimit=[TimeSpan]::FromMinutes(5)\n"
                  "Get-LeadLedgerHealth | ConvertTo-Json -Compress\n")
    assert result["Healthy"] is (problem is None)
    assert result["Reasons"] == ([] if problem is None else [problem])


def test_live_process_identity_requires_correct_script_state_and_flags(tmp_path):
    result = _run(tmp_path, ["Test-ExpectedLeadRunner"], f"""
$ProjectRoot='{_quote(ROOT)}'
$RunnerScript=Join-Path $ProjectRoot 'scripts\\run_mr_candidate_paper.py'
$StatePath=Join-Path $ProjectRoot 'data\\live\\lead_portfolio.json'
$good='python.exe "'+$RunnerScript+'" --live --state "'+$StatePath+'"'
$relative='python.exe .\\scripts\\run_mr_candidate_paper.py --live --state .\\data\\live\\lead_portfolio.json'
@{{
 Absolute=(Test-ExpectedLeadRunner ([pscustomobject]@{{CommandLine=$good}}))
 Relative=(Test-ExpectedLeadRunner ([pscustomobject]@{{CommandLine=$relative}}))
 WrongState=(Test-ExpectedLeadRunner ([pscustomobject]@{{CommandLine=$good.Replace('lead_portfolio.json','other.json')}}))
 Demo=(Test-ExpectedLeadRunner ([pscustomobject]@{{CommandLine=($good+' --demo')}}))
 NoLiveFlag=(Test-ExpectedLeadRunner ([pscustomobject]@{{CommandLine=$good.Replace('--live','')}}))
}}|ConvertTo-Json -Compress
""")
    assert result == {"Absolute": True, "Relative": True, "WrongState": False,
                      "Demo": False, "NoLiveFlag": False}


def test_discovery_includes_foreign_live_state_and_process_children(tmp_path):
    result = _run(tmp_path, ["Get-LeadRunnerProcesses", "Get-LeadRunnerRoots"], """
function Get-CimInstance {
 @(
  [pscustomobject]@{Name='python.exe';ProcessId=10;ParentProcessId=1;CommandLine='python run_mr_candidate_paper.py --live --state correct.json'}
  [pscustomobject]@{Name='python.exe';ProcessId=11;ParentProcessId=10;CommandLine='python child.py'}
  [pscustomobject]@{Name='python.exe';ProcessId=20;ParentProcessId=1;CommandLine='python run_mr_candidate_paper.py --live --state foreign.json'}
  [pscustomobject]@{Name='python.exe';ProcessId=21;ParentProcessId=20;CommandLine='python child.py'}
  [pscustomobject]@{Name='python.exe';ProcessId=30;ParentProcessId=1;CommandLine='python run_mr_candidate_paper.py --demo --state demo.json'}
 )
}
$processes=@(Get-LeadRunnerProcesses)
@{Processes=@($processes.ProcessId|Sort-Object);Roots=@((Get-LeadRunnerRoots $processes).ProcessId|Sort-Object)}|ConvertTo-Json -Compress
""")
    assert result == {"Processes": [10, 11, 20, 21], "Roots": [10, 20]}


def test_watchdog_recovery_backoff_adoption_and_no_live_process_termination(tmp_path):
    auth_failure = tmp_path / "auth-failure.json"
    auth_failure.write_text(json.dumps({"events": [{"event": "account_startup_failed",
                                                   "at": datetime.now(timezone.utc).isoformat(),
                                                   "error": "AuthenticationError"}]}), encoding="utf-8")
    result = _run(tmp_path, ["Invoke-LeadWatchdogCheck", "Set-LeadRetryBackoff", "Get-LeadRunnerRoots"], f"""
$StatePath='{_quote(tmp_path / 'absent.json')}'
function Get-LeadRunnerProcesses {{ $script:FakeProcesses }}
function Test-ExpectedLeadRunner {{ param($Process); return $script:Expected }}
function Get-LeadLedgerHealth {{ $script:Health }}
function Start-LeadRunner {{ $script:Starts++ ; if($script:FailStart){{throw 'mock start failed'}} }}
function Stop-Process {{ throw 'Live process termination must never be called' }}
function Write-LeadWatchdogLog {{ param($Message);$script:Messages.Add($Message) }}
function Reset-State {{
 $script:IsArmed=$true;$script:ManagedRunnerPid=0;$script:NextAttemptAt=[DateTime]::MinValue
 $script:RetryDelaySeconds=60;$script:GraceUntil=[DateTime]::MinValue;$script:LastWarning=$null
 $script:LastStartedAt=[DateTimeOffset]::MaxValue;$script:Starts=0;$script:FailStart=$false
 $script:FakeProcesses=@();$script:Expected=$true
 $script:Health=[pscustomobject]@{{Healthy=$true;Reasons=@()}}
 $script:Messages=[Collections.Generic.List[string]]::new()
}}
$live=[pscustomobject]@{{ProcessId=10;ParentProcessId=1}}
$results=@{{}}
Reset-State;Invoke-LeadWatchdogCheck;$results.MissingStarts=$script:Starts
Reset-State;$script:NextAttemptAt=(Get-Date).AddMinutes(1);Invoke-LeadWatchdogCheck;$results.BackoffStarts=$script:Starts
Reset-State;$script:FakeProcesses=@($live);Invoke-LeadWatchdogCheck;$results.Adopted=$script:ManagedRunnerPid;$results.AdoptionStarts=$script:Starts
Reset-State;$script:FakeProcesses=@($live);$script:Health=[pscustomobject]@{{Healthy=$false;Reasons=@('snapshot_stale','heartbeat_stale')}};Invoke-LeadWatchdogCheck
$results.StaleStarts=$script:Starts;$results.StaleWarned=($script:Messages[-1] -match 'no forced restart')
Reset-State;$script:FakeProcesses=@($live,[pscustomobject]@{{ProcessId=20;ParentProcessId=1}});Invoke-LeadWatchdogCheck;$results.DuplicateStarts=$script:Starts
Reset-State;$script:FakeProcesses=@($live);$script:Expected=$false;Invoke-LeadWatchdogCheck;$results.ForeignStarts=$script:Starts
Reset-State;$script:ManagedRunnerPid=10;Invoke-LeadWatchdogCheck
$results.ExitWaited=($script:Starts -eq 0 -and $script:NextAttemptAt -gt (Get-Date) -and $script:RetryDelaySeconds -eq 120)
$script:NextAttemptAt=[DateTime]::MinValue;Invoke-LeadWatchdogCheck;$results.RecoveryStarts=$script:Starts
Reset-State;$script:FailStart=$true;Invoke-LeadWatchdogCheck;$results.StartFailureWaited=($script:NextAttemptAt -gt (Get-Date))
Reset-State;$StatePath='{_quote(auth_failure)}';$script:ManagedRunnerPid=10;$script:LastStartedAt=[DateTimeOffset]::UtcNow.AddMinutes(-1);Invoke-LeadWatchdogCheck
$results.AuthenticationWaited=($script:RetryDelaySeconds -eq 600 -and $script:NextAttemptAt -gt (Get-Date).AddSeconds(290))
$script:RetryDelaySeconds=1800;Set-LeadRetryBackoff -Reason 'test cap';$results.BackoffCapped=($script:RetryDelaySeconds -eq 1800)
Reset-State;$script:IsArmed=$false;try{{Invoke-LeadWatchdogCheck;$results.UnarmedBlocked=$false}}catch{{$results.UnarmedBlocked=$true}}
$results|ConvertTo-Json -Compress
""")
    assert result == {"MissingStarts": 1, "BackoffStarts": 0, "Adopted": 10, "AdoptionStarts": 0,
                      "StaleStarts": 0, "StaleWarned": True, "DuplicateStarts": 0, "ForeignStarts": 0,
                      "ExitWaited": True, "RecoveryStarts": 1, "StartFailureWaited": True,
                      "AuthenticationWaited": True, "BackoffCapped": True, "UnarmedBlocked": True}


def test_runner_start_uses_only_lead_credentials_and_restores_environment_on_failure(tmp_path):
    result = _run(tmp_path, ["Start-LeadRunner"], f"""
Add-Type 'public static class AltMeanrevCredentialStore {{ public static int Calls; public static string Target; public static string[] Read(string target) {{ Calls++; Target=target; return new string[] {{"test-key", "test-secret"}}; }} }}'
$ProjectRoot='{_quote(ROOT)}';$RunnerPython='python.exe';$RunnerScript='runner.py'
$StatePath='lead-state.json';$LogDirectory='{_quote(tmp_path)}';$HeartbeatLimit=[TimeSpan]::FromMinutes(5)
$CredentialTarget='AltMeanrevBot:BinanceUsdM:Lead';$script:IsArmed=$true;$script:FailStart=$false
$script:ExistingRunner=$false
function Get-LeadRunnerProcesses {{ if($script:ExistingRunner){{[pscustomobject]@{{ProcessId=99}}}} }}
function Write-LeadWatchdogLog {{param($Message)}}
function Start-Process {{
 param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,[switch]$PassThru,$RedirectStandardOutput,$RedirectStandardError)
 $script:Captured=@{{LeadEnvironment=($env:BOT_MODE -eq 'live' -and $env:BINANCE_TESTNET -eq 'false' -and $env:BOT_LEAD_BACKGROUND_CONFIRMATION -ceq 'LEAD');CredentialMatch=($env:BINANCE_API_KEY -eq 'test-key' -and $env:BINANCE_API_SECRET -eq 'test-secret');Arguments=$ArgumentList;Hidden=($WindowStyle -eq 'Hidden')}}
 if($script:FailStart){{throw 'simulated process creation failure'}}
 [pscustomobject]@{{Id=123}}
}}
$env:BINANCE_API_KEY='original-key';$env:BINANCE_API_SECRET='original-secret';$env:BOT_MODE='paper';$env:BINANCE_TESTNET='true';$env:BOT_LEAD_BACKGROUND_CONFIRMATION='original-confirmation'
Start-LeadRunner
$result=@{{LeadEnvironment=$script:Captured.LeadEnvironment;CredentialMatch=$script:Captured.CredentialMatch;Hidden=$script:Captured.Hidden;CorrectTarget=([AltMeanrevCredentialStore]::Target -eq $CredentialTarget);ManagedPid=$script:ManagedRunnerPid;LiveArguments=($script:Captured.Arguments -match '--live --background-confirmed --poll 60' -and $script:Captured.Arguments -notmatch '--demo');RestoredAfterSuccess=($env:BINANCE_API_KEY -eq 'original-key' -and $env:BOT_MODE -eq 'paper' -and $env:BOT_LEAD_BACKGROUND_CONFIRMATION -eq 'original-confirmation')}}
$script:FailStart=$true;try{{Start-LeadRunner}}catch{{}}
$result.RestoredAfterFailure=($env:BINANCE_API_SECRET -eq 'original-secret' -and $env:BINANCE_TESTNET -eq 'true')
$script:ExistingRunner=$true;Start-LeadRunner
$result.NoCredentialReadForExistingRunner=([AltMeanrevCredentialStore]::Calls -eq 2)
$script:IsArmed=$false;try{{Start-LeadRunner}}catch{{}}
$result.NoCredentialReadWhenUnarmed=([AltMeanrevCredentialStore]::Calls -eq 2)
$result|ConvertTo-Json -Compress
""")
    assert result == {"LeadEnvironment": True, "CredentialMatch": True, "Hidden": True,
                      "CorrectTarget": True, "ManagedPid": 123, "LiveArguments": True,
                      "RestoredAfterSuccess": True, "RestoredAfterFailure": True,
                      "NoCredentialReadForExistingRunner": True,
                      "NoCredentialReadWhenUnarmed": True}


@pytest.mark.parametrize("args", [[], ["-Armed"]])
def test_internal_watchdog_requires_explicit_arming_before_any_runtime_action(args):
    env = dict(os.environ)
    env.pop("BOT_LEAD_WATCHDOG_CONFIRMATION", None)
    result = subprocess.run([SHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SOURCE), *args],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "requires run-lead-watchdog.ps1" in result.stderr


def test_stop_command_stops_only_the_exact_watchdog_and_keeps_runner(tmp_path):
    script = tmp_path / "stop.ps1"
    script.write_text(f"""
$ErrorActionPreference='Stop'
function Get-CimInstance {{
 @(
  [pscustomobject]@{{Name='powershell.exe';ProcessId=71;CommandLine='powershell -File "{_quote(SOURCE)}" -Armed'}}
  [pscustomobject]@{{Name='python.exe';ProcessId=72;CommandLine='python run_mr_candidate_paper.py --live'}}
  [pscustomobject]@{{Name='powershell.exe';ProcessId=73;CommandLine='powershell -File other-watchdog.ps1 -Armed'}}
 )
}}
function Stop-Process {{ param($Id,$ErrorAction);Write-Output "STOPPED=$Id" }}
& '{_quote(ROOT / 'run-lead-watchdog.ps1')}' -Stop
""", encoding="utf-8-sig")
    result = subprocess.run([SHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            capture_output=True, text=True, timeout=30, check=True)
    assert "STOPPED=71" in result.stdout
    assert "STOPPED=72" not in result.stdout and "STOPPED=73" not in result.stdout
    assert "Lead Runner was not stopped" in result.stdout


@pytest.mark.parametrize("entry", ["run-lead-live.ps1", "run-lead-live-background.ps1"])
def test_manual_live_entries_refuse_launch_while_watchdog_is_active(tmp_path, entry):
    script = tmp_path / "manual-launch.ps1"
    script.write_text(f"""
$ErrorActionPreference='Stop'
function Get-CimInstance {{ [pscustomobject]@{{Name='powershell.exe';ProcessId=71;CommandLine='powershell -File "{_quote(SOURCE)}" -Armed'}} }}
& '{_quote(ROOT / entry)}'
""", encoding="utf-8-sig")
    result = subprocess.run([SHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode != 0
    assert "Lead watchdog is active" in result.stderr
