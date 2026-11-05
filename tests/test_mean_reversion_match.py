from datetime import datetime, timezone

import pandas as pd

from bot.config import load_settings
from bot.mean_reversion_match import evaluate_mean_reversion_match


NOW = datetime(2026, 10, 16, 8, 30, tzinfo=timezone.utc)


def bars_for(kind: str, count: int = 100) -> pd.DataFrame:
    index = pd.date_range(end="2026-10-16 04:00:00+00:00", periods=count, freq="4h", name="timestamp")
    values = []
    for i in range(count):
        if kind == "up":
            close = 100 + i * 0.5
            open_price, high, low = close - 0.1, close + 0.3, close - 0.3
        else:
            close = 100.0
            open_price, high, low = close, close + 1, close - 1
        values.append((open_price, high, low, close, 100.0))
    return pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=index)


def test_range_like_bars_score_above_confirmed_trend_bars():
    settings = load_settings(env_file=None)
    range_score = evaluate_mean_reversion_match("FLAT", bars_for("flat"), settings, now=NOW)
    trend_score = evaluate_mean_reversion_match("UP", bars_for("up"), settings, now=NOW)

    assert range_score["status"] == "scored"
    assert range_score["route_state"] == "RANGE"
    assert range_score["score"] >= 80
    assert trend_score["route_state"] == "TREND"
    assert trend_score["score"] < range_score["score"]
    assert "trend_tendency_penalty" in trend_score["reasons"]


def test_open_4h_bar_does_not_change_match_score_or_metrics():
    settings = load_settings(env_file=None)
    completed = bars_for("flat")
    open_bar = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.0, 100.0]],
        columns=completed.columns,
        index=[pd.Timestamp("2026-10-16 08:00:00+00:00")],
    )

    baseline = evaluate_mean_reversion_match("FLAT", completed, settings, now=NOW)
    with_open_bar = evaluate_mean_reversion_match("FLAT", pd.concat([completed, open_bar]), settings, now=NOW)

    assert with_open_bar["score"] == baseline["score"]
    assert with_open_bar["metrics"] == baseline["metrics"]
    assert with_open_bar["last_closed_bar_at"] == baseline["last_closed_bar_at"]


def test_unavailable_history_is_not_reported_as_a_low_match_score():
    settings = load_settings(env_file=None)

    result = evaluate_mean_reversion_match("SHORT", bars_for("flat", count=20), settings, now=NOW)

    assert result["score"] is None
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "insufficient_closed_bars"
