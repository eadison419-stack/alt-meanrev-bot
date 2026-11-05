from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from bot.config import load_settings
from bot.live.bollinger_outcomes import BollingerOutcomeTracker, summarize_outcome_state


ENTRY_AT = datetime(2026, 10, 19, 12, 15, tzinfo=timezone.utc)


def hit(signal_type="long_start", *, symbol="AAA/USDT:USDT"):
    return {
        "symbol": symbol,
        "signal_types": [signal_type],
        "current_price": 100.0,
        "signal_close": 99.5,
        "last_closed_bar_at": "2026-10-19T08:00:00+00:00",
    }


def future_hourly_bars(end_at=ENTRY_AT + timedelta(hours=62)):
    start = ENTRY_AT.replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(start=start, end=end_at, freq="1h", inclusive="left", name="timestamp")
    rows = []
    for offset, _ in enumerate(index):
        high = 103.0 + offset
        low = 98.0 - offset / 10
        close = 101.0 + offset / 5
        rows.append((100.0, high, low, close, 1000.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)


@pytest.mark.parametrize("signal_type, expected_gross", [("long_start", 1.8), ("short_trend", -1.8)])
def test_tracker_deduplicates_hits_and_records_completed_horizons(tmp_path, signal_type, expected_gross):
    settings = load_settings(env_file=None)
    path = tmp_path / "bollinger-outcomes.json"
    tracker = BollingerOutcomeTracker(path, settings)
    candidate = hit(signal_type)

    initial = tracker.process_refresh([candidate, candidate], {"AAA/USDT:USDT": future_hourly_bars()}, now=ENTRY_AT)
    assert initial["new_samples"] == 1
    assert initial["newly_completed_horizons"] == 0

    later = ENTRY_AT + timedelta(hours=62)
    result = tracker.process_refresh([candidate], {"AAA/USDT:USDT": future_hourly_bars(later)}, now=later)
    assert result["new_samples"] == 0
    assert result["newly_completed_horizons"] == 4
    assert len(tracker.state["samples"]) == 1

    sample = next(iter(tracker.state["samples"].values()))
    assert sample["entry_timing_status"] == "on_time"
    assert sample["entry_delay_seconds"] == pytest.approx(900)
    four_hour = sample["outcomes"]["4h"]
    assert four_hour["gross_return_pct"] == pytest.approx(expected_gross)
    assert four_hour["actual_horizon_hours"] > 4
    assert four_hour["coverage_complete"] is True
    assert four_hour["mfe_pct"] > 0
    assert four_hour["mae_pct"] < 0
    assert four_hour["estimated_net_return_pct"] < four_hour["gross_return_pct"]
    assert four_hour["funding_included"] is False
    assert path.is_file()

    summary = summarize_outcome_state(tracker.state)
    assert summary["sample_count"] == 1
    assert summary["eligible_sample_count"] == 1
    assert summary["by_signal_type"][signal_type]["horizons"]["4h"]["completed_count"] == 1
    assert summary["by_signal_type"][signal_type]["horizons"]["4h"]["coverage_complete_count"] == 1


def test_tracker_waits_for_closed_bars_and_marks_gaps_in_excursion_coverage(tmp_path):
    settings = load_settings(env_file=None)
    tracker = BollingerOutcomeTracker(tmp_path / "outcomes.json", settings)
    candidate = hit()
    tracker.process_refresh([candidate], {}, now=ENTRY_AT)

    later = ENTRY_AT + timedelta(hours=8)
    bars = future_hourly_bars(later)
    bars = bars.drop(bars.index[1])
    result = tracker.process_refresh([], {"AAA/USDT:USDT": bars}, now=later)

    sample = next(iter(tracker.state["samples"].values()))
    four_hour = sample["outcomes"]["4h"]
    assert result["newly_completed_horizons"] == 1
    assert four_hour["coverage_complete"] is False
    assert four_hour["bars_observed"] < 4


def test_tracker_rejects_naive_signal_timestamps_without_creating_a_sample(tmp_path):
    tracker = BollingerOutcomeTracker(tmp_path / "outcomes.json", load_settings(env_file=None))
    candidate = {**hit(), "last_closed_bar_at": "2026-10-19T08:00:00"}

    result = tracker.process_refresh([candidate], {}, now=ENTRY_AT)

    assert result["invalid_candidates"] == 1
    assert result["summary"]["sample_count"] == 0


def test_tracker_backfills_parameters_for_existing_sidecar_samples(tmp_path):
    settings = load_settings(env_file=None)
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps({
        "version": 1,
        "rules_version": "bollinger_observer_v1",
        "samples": {
            "old-sample": {
                "sample_id": "old-sample",
                "symbol": "AAA/USDT:USDT",
                "signal_type": "long_start",
                "side": "long",
                "last_closed_bar_at": "2026-10-19T08:00:00+00:00",
                "entry_at": ENTRY_AT.isoformat(),
                "entry_price": 100.0,
                "excursion_coverage_starts_at": (ENTRY_AT + timedelta(hours=1)).isoformat(),
                "bars": {},
                "outcomes": {"4h": {"gross_return_pct": 1.0, "estimated_net_return_pct": 0.8}},
            },
        },
    }), encoding="utf-8")

    tracker = BollingerOutcomeTracker(path, settings)
    result = tracker.process_refresh([], {}, now=ENTRY_AT)

    sample = tracker.state["samples"]["old-sample"]
    assert sample["cost_parameters"] == {
        "taker_bps": settings.fees.taker_bps,
        "slippage_bps": settings.fees.slippage_bps,
    }
    assert sample["cost_parameters_source"] == "backfilled_from_config_at_load"
    assert sample["rule_parameters_source"] == "backfilled_from_config_at_load"
    assert sample["entry_timing_status"] == "on_time"
    assert sample["entry_timing_source"] == "backfilled_from_signal_timestamp"
    assert sample["outcomes"]["4h"]["estimated_cost_impact_pct"] == pytest.approx(0.2)
    assert sample["outcomes"]["4h"]["funding_included"] is False
    assert result["enabled"] is True

