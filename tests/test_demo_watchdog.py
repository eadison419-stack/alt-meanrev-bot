import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import pytest


def test_windows_watchdog_reads_mixed_escaped_and_literal_unicode_as_utf8(tmp_path):
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows PowerShell integration check")
    symbol = "牛来/USDT:USDT"
    state = {
        "execution_mode": "demo_usdm", "strategy_variant": "ema50_deviation_candidate",
        "execution_status": "synchronized", "positions": {symbol: {}},
        "exchange_snapshot": {
            "host": "https://demo-fapi.binance.com",
            "at": datetime.now(timezone.utc).isoformat(), "positions": {symbol: {}},
        },
    }
    # Both spellings represent the same JSON key. An ANSI read corrupts only
    # the literal one and would report a false position mismatch.
    raw = json.dumps(state, ensure_ascii=True).replace(
        json.dumps(symbol, ensure_ascii=True), json.dumps(symbol, ensure_ascii=False), 1,
    )
    ledger = tmp_path / "state.json"
    ledger.write_text(raw, encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "scripts" / "watch_demo_paper.ps1"
    script = tmp_path / "check-health.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$tokens = $null; $parseErrors = $null\n"
        f"$ast = [System.Management.Automation.Language.Parser]::ParseFile('{source}', "
        "[ref]$tokens, [ref]$parseErrors)\n"
        "$function = $ast.Find({param($node) "
        "$node -is [System.Management.Automation.Language.FunctionDefinitionAst] "
        "-and $node.Name -eq 'Get-DemoLedgerHealth'}, $true)\n"
        "Invoke-Expression $function.Extent.Text\n"
        f"$StatePath = '{ledger}'\n"
        "$HeartbeatLimit = [TimeSpan]::FromMinutes(5)\n"
        "Get-DemoLedgerHealth | ConvertTo-Json -Compress\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=30, check=True)
    health = json.loads(result.stdout)
    assert health["Healthy"] is True and health["Reasons"] == []
