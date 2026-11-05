"""Print a read-only summary of hypothetical Bollinger forward outcomes."""

import argparse
import json
from pathlib import Path

from bot.config import PROJECT_ROOT
from bot.live.bollinger_outcomes import entry_timing_for_sample, summarize_outcome_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=Path("data/paper/portfolio.json"),
        help="portfolio path used to derive the isolated outcomes sidecar",
    )
    parser.add_argument("--outcomes", type=Path, help="explicit outcomes sidecar path")
    parser.add_argument("--limit", type=int, default=100,
                        help="maximum recent per-sample rows to print; use 0 for all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or greater")

    def project_path(value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    state_path = project_path(args.state)
    outcomes_path = project_path(args.outcomes) if args.outcomes else state_path.with_name(
        f"{state_path.stem}.bollinger-outcomes.json"
    )
    if not outcomes_path.is_file():
        parser.error(f"Bollinger outcomes not found: {outcomes_path}; the Runner must load the tracker first")
    data = json.loads(outcomes_path.read_text(encoding="utf-8"))
    summary = summarize_outcome_state(data)
    summary["state_path"] = str(outcomes_path)
    summary["created_at"] = data.get("created_at")
    summary["last_loaded_at"] = data.get("last_loaded_at")
    summary["last_updated_at"] = data.get("last_updated_at")
    samples = sorted(data.get("samples", {}).values(), key=lambda sample: sample.get("entry_at", ""),
                     reverse=True)
    summary["sample_detail_limit"] = args.limit or "all"
    summary["samples_omitted"] = max(0, len(samples) - args.limit) if args.limit else 0
    sample_details = []
    for sample in (samples if args.limit == 0 else samples[:args.limit]):
        detail = {
            key: sample.get(key)
            for key in ("sample_id", "symbol", "signal_type", "side", "last_closed_bar_at",
                        "signal_close", "entry_at", "entry_price", "rule_parameters",
                        "rule_parameters_source", "cost_parameters", "cost_parameters_source",
                        "entry_timing_source")
        } | {"outcomes": sample.get("outcomes", {})}
        try:
            detail.update(entry_timing_for_sample(sample))
        except (KeyError, TypeError, ValueError, OverflowError):
            detail["entry_timing_status"] = "invalid_timing"
        sample_details.append(detail)
    summary["samples"] = sample_details
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
