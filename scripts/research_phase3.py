"""Run predeclared one-factor MR strategy hypotheses on the frozen public snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from bot.backtest.research_data import (
    HOUR_MS, aggregate_closed_bars, digest_file, load_manifest, load_stream, profile_bars,
    utc, write_json_once,
)
from bot.config import load_settings
from scripts.replay_mr_baseline import HYPOTHESES, QUOTE_MODEL, replay


SCENARIOS = {
    "baseline": {"fee_multiplier": 1.0, "slippage_multiplier": 1.0},
    "fee_2_slip_3": {"fee_multiplier": 2.0, "slippage_multiplier": 3.0},
}
STREAMS = ("contract_1h", "contract_4h", "mark_1h", "funding")


def _canonical(symbol: str) -> str:
    return symbol[:-4] + "/USDT:USDT"


def _load_frames(snapshot: Path, manifest: dict) -> dict:
    frames = {}
    for identifier in manifest["symbols"]:
        streams = {stream: load_stream(snapshot, identifier, stream) for stream in STREAMS}
        for stream, hours in (("contract_1h", 1), ("contract_4h", 4), ("mark_1h", 1)):
            quality = profile_bars(streams[stream], hours * HOUR_MS,
                                   start=manifest["start_utc"],
                                   end=manifest["end_exclusive_utc"])
            if not quality["valid"]:
                raise ValueError(f"{identifier} {stream}: {', '.join(quality['errors'])}")
        streams["daily"] = aggregate_closed_bars(
            streams["contract_4h"], 4, 24, as_of=manifest["end_exclusive_utc"],
        )
        frames[_canonical(identifier)] = streams
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hypotheses", nargs="+", choices=sorted(HYPOTHESES),
                        default=["baseline", "daily_history_prefilter", "structure_stop_2p5",
                                 "atr_deviation_0p8", "impulse_counter_deviation_1p2",
                                 "reversal_body_midpoint", "ema_space_1r",
                                 "directional_funding_0p08"])
    parser.add_argument("--windows", nargs="+", choices=["validation", "test"],
                        default=["validation", "test"])
    parser.add_argument("--include-cost-stress", action="store_true",
                        help="also rerun the historical test window at fee x2 / slippage x3")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(lambda _message: None, level="CRITICAL")

    manifest = load_manifest(args.snapshot)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    frames = _load_frames(args.snapshot, manifest)
    metadata = json.loads((args.snapshot / "exchange_info.json").read_text(encoding="utf-8"))["symbols"]
    plan_windows = {row["name"]: row for row in plan["windows"]}
    windows = [plan_windows[name] for name in args.windows]
    for window in windows:
        if not (utc(manifest["start_utc"]) < utc(window["start"]) < utc(window["end_exclusive"])
                <= utc(manifest["end_exclusive_utc"])):
            parser.error(f"{window['name']} window is outside the public snapshot")
    jobs = [(hypothesis, window, "baseline")
            for hypothesis in args.hypotheses for window in windows]
    if args.include_cost_stress:
        if "test" not in args.windows:
            parser.error("--include-cost-stress requires the test window")
        jobs.extend((hypothesis, plan_windows["test"], "fee_2_slip_3")
                    for hypothesis in args.hypotheses)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = load_settings(ROOT / "configs/default.yaml", env_file=None)
    source_files = ["scripts/research_phase3.py", "scripts/replay_mr_baseline.py",
                    "scripts/run_mr_candidate_paper.py", "src/bot/mr_candidate.py",
                    "src/bot/risk.py", "src/bot/lead_risk.py", "src/bot/time_filter.py",
                    "src/bot/indicators.py", "configs/default.yaml"]
    write_json_once(args.output_dir / "run_inputs.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest_file(args.snapshot / "manifest.json"),
        "plan_sha256": digest_file(args.plan),
        "source_sha256": {name: digest_file(ROOT / name) for name in source_files},
        "initial_equity": 1000.0,
        "mode": "lead_usdm",
        "quote_model": QUOTE_MODEL,
        "hypothesis_definitions": {
            name: {
                "prefilter_daily_history": item.prefilter_daily_history,
                "stop_hard_cap_atr": item.stop_hard_cap_atr,
                "deviation_rule": item.deviation_rule,
                "reversal_body_midpoint": item.reversal_body_midpoint,
                "minimum_ema_space_r": item.minimum_ema_space_r,
                "funding_adverse_threshold": item.funding_adverse_threshold,
            } for name, item in HYPOTHESES.items()
        },
        "windows": windows,
        "jobs": [{"hypothesis": h, "window": w["name"], "scenario": s} for h, w, s in jobs],
        "promotion_allowed": False,
        "limitations": [
            "historical test period was previously reviewed and is not certified unseen",
            "historical Lead whitelist and signed leverage brackets were unavailable",
            "hourly synthetic quote/depth model is not a tick or order-book replay",
            "hypotheses are independent exploratory comparisons, not adopted production settings",
        ],
    })

    summaries = []
    fee_base, slip_base = settings.fees.taker_bps, settings.fees.slippage_bps
    for index, (hypothesis, window, scenario) in enumerate(jobs, start=1):
        costs = SCENARIOS[scenario]
        configured = settings.model_copy(update={"fees": settings.fees.model_copy(update={
            "taker_bps": fee_base * costs["fee_multiplier"],
            "slippage_bps": slip_base * costs["slippage_multiplier"],
        })})
        label = f"{hypothesis}_{window['name']}_{scenario}"
        print(f"starting {index}/{len(jobs)} {label}", flush=True)
        result = replay(
            frames, metadata, configured, start=window["start"], end=window["end_exclusive"],
            initial_equity=1000.0, mode="lead_usdm", hypothesis=hypothesis,
            progress=lambda message: print(message, flush=True),
        )
        result["scenario"] = {"name": scenario, **costs}
        result["window"] = window
        target = args.output_dir / f"{label}.json"
        write_json_once(target, result)
        summary = {key: value for key, value in result.items()
                   if key not in {"trades", "equity", "decisions"}}
        summary["output"] = str(target.resolve())
        summary["output_sha256"] = digest_file(target)
        write_json_once(args.output_dir / f"{label}_summary.json", summary)
        summaries.append(summary)
        print(json.dumps({"completed": label, "metrics": result["metrics"],
                          "ranking_metrics": result["ranking_metrics"]}), flush=True)
    write_json_once(args.output_dir / "summary.json", {
        "runs": summaries, "promotion_allowed": False,
    })
    print(f"completed {len(summaries)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
