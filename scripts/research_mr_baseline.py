"""Versioned public data and offline evaluation of the current MR Runner.

Run from the repository with the project virtual environment. No trading account
is connected. The legacy database is opened read-only and never repaired here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import ccxt
import pandas as pd

from bot.backtest.research_data import (
    HOUR_MS, STREAMS, aggregate_closed_bars, chronological_splits, collect_snapshot,
    compare_frames, digest_file, legacy_frame, legacy_inventory, load_manifest,
    load_stream, profile_bars, utc, write_json_once,
)
from bot.exchange.network import configure_system_proxy


def profile_snapshot(directory, legacy_db):
    manifest = load_manifest(directory)
    report = {"manifest_sha256": digest_file(Path(directory) / "manifest.json"),
              "legacy": legacy_inventory(legacy_db), "symbols": {},
              "excluded": manifest.get("excluded", {}), "valid": True}
    for symbol in manifest["symbols"]:
        one, four, mark = [load_stream(directory, symbol, stream)
                           for stream in ("contract_1h", "contract_4h", "mark_1h")]
        checks = {name: profile_bars(frame, hours * HOUR_MS,
                    start=manifest["start_utc"], end=manifest["end_exclusive_utc"])
                  for name, frame, hours in (("contract_1h", one, 1), ("contract_4h", four, 4),
                                              ("mark_1h", mark, 1))}
        if all(row["valid"] for row in checks.values()):
            daily = aggregate_closed_bars(four, 4, 24, as_of=manifest["end_exclusive_utc"])
            check4 = compare_frames(aggregate_closed_bars(one, 1, 4, as_of=manifest["end_exclusive_utc"]), four)
            checks["daily"] = profile_bars(daily, 24 * HOUR_MS)
            checks["cross_timeframe"] = check4
            if check4["different_rows"]:
                report["valid"] = False
        funding = load_stream(directory, symbol, "funding")
        checks["funding"] = {"rows": len(funding), "first_settlement": funding.index[0].isoformat(),
            "last_settlement": funding.index[-1].isoformat(),
            "maximum_gap_hours": float(funding.index.to_series().diff().max().total_seconds() / 3600),
            "entry_rate_model": "last_observed_settlement; historical live estimate unavailable"}
        checks["legacy_comparison"] = compare_frames(legacy_frame(legacy_db, symbol, "1h"), one)
        report["symbols"][symbol] = checks
        if any(not row["valid"] for row in checks.values() if "valid" in row):
            report["valid"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--legacy-db", type=Path, default=Path("data/market.sqlite"))
    parser.add_argument("--collect-public", action="store_true")
    parser.add_argument("--data-start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-11-10T00:00:00Z")
    parser.add_argument("--evaluation-start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    inventory = legacy_inventory(args.legacy_db)
    if args.collect_public:
        symbols = sorted({row["symbol"] for row in inventory["datasets"]})
        client = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 5000})
        configure_system_proxy(client)
        try:
            if client.apiKey or not client.urls["api"]["fapiPublic"].startswith("https://fapi.binance.com/"):
                raise ValueError("research collector requires a keyless production public client")
            collect_snapshot(client, args.snapshot, symbols, args.data_start, args.end,
                             progress=lambda message: print(message, flush=True))
        finally:
            client.close()
    profile = profile_snapshot(args.snapshot, args.legacy_db)
    output = args.output_dir or args.snapshot / ("audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "data_quality.json", profile)
    write_json_once(output / "evaluation_plan.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": chronological_splits(args.evaluation_start, args.end),
        "baseline_only": True, "parameter_selection": "none",
        "cost_scenarios": [{"name": "baseline", "fee_multiplier": 1, "slippage_multiplier": 1},
                           {"name": "fee_1p5_slip_2", "fee_multiplier": 1.5, "slippage_multiplier": 2},
                           {"name": "fee_2_slip_3", "fee_multiplier": 2, "slippage_multiplier": 3}],
        "promotion_allowed": False,
        "effective_symbols": sorted(profile["symbols"]), "excluded": profile["excluded"],
        "limitations": ["fixed population and current metadata; survivorship bias",
            "historical settled funding is not the live estimated funding rate",
            "hourly candles do not recover the 5-second price path or historical order books",
            "historical signed leverage brackets and Lead whitelist are unavailable",
            "past research touched these periods; chronological test is not certified unseen"],
    })
    print(json.dumps({"output": str(output.resolve()), "data_quality_valid": profile["valid"]}), flush=True)
    return 0 if profile["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
