from datetime import datetime, timezone

import pandas as pd

from bot.config import load_settings
from bot.trend_match import evaluate_trend_match


NOW = datetime(2026, 10, 16, 8, 30, tzinfo=timezone.utc)


def bars_for(kind: str, count: int = 100) -> pd.DataFrame:
    index = pd.date_range(end="2026-10-16 04:00:00+00:00", periods=count, freq="4h", name="timestamp")
    values = []
    previous = 100.0
    trend_bars = 30 if kind == "early_strength" else 12
    step = 0.13 if kind == "early_strength" else 4.0 if kind.startswith("extreme_") else 2.4
    for i in range(count):
        if i < count - trend_bars:
            close = 100.0
            open_price, high, low = close, 120.0, 80.0
        else:
            direction = -1 if kind == "extreme_low" else 1
            close = 100 + direction * (i - (count - trend_bars) + 1) * step
            open_price = previous
            high = max(open_price, close) + 0.2
            low = min(open_price, close) - 0.2
        values.append((open_price, high, low, close, 100.0))
        previous = close
    return pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=index)


def directional_bars(kind: str, count: int = 100) -> pd.DataFrame:
    index = pd.date_range(end="2026-10-16 04:00:00+00:00", periods=count, freq="4h", name="timestamp")
    values = []
    trend_bars = 30
    previous = 100.0 if kind == "long" else 110.0
    for i in range(count):
        if i < count - trend_bars:
            close = previous
            open_price, high, low = close, 120.0, 80.0
        elif kind == "long":
            close = 100.0 + (i - (count - trend_bars) + 1) * 0.13
            open_price = previous
            high, low = max(open_price, close) + 0.2, min(open_price, close) - 0.2
        else:
            close = 110.0 - (i - (count - trend_bars) + 1) * 0.1
            open_price = previous
            high, low = max(open_price, close) + 0.2, min(open_price, close) - 0.2
        values.append((open_price, high, low, close, 100.0))
        previous = close
    return pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=index)


def test_lower_mid_range_early_strength_scores_above_extended_price():
    settings = load_settings(env_file=None)
    early = evaluate_trend_match("EARLY", bars_for("early_strength"), settings, now=NOW)
    extended = evaluate_trend_match("EXTENDED", bars_for("extended"), settings, now=NOW)

    assert early["status"] == "scored"
    assert 20 <= early["metrics"]["range_position_pct"] <= 60
    assert early["metrics"]["adx_change"] >= settings.trend_match.adx_min_rise
    assert early["score"] > extended["score"]
    assert extended["metrics"]["range_position_pct"] > 75
    assert "range_position_upper_extension_deduction" in extended["deduction_reasons"]
    assert "recent_rally_penalty" in extended["deduction_reasons"]


def test_relative_low_turning_up_favors_long_score():
    settings = load_settings(env_file=None)
    result = evaluate_trend_match("LONG", directional_bars("long"), settings, now=NOW)

    assert 20 <= result["metrics"]["range_position_pct"] <= 60
    assert result["metrics"]["ema50_slope_pct"] > 0
    assert result["trend_long_match_score"] > result["trend_short_match_score"]
    assert result["direction_bias"] == "long"
    assert result["trend_match_score"] == max(
        result["trend_long_match_score"], result["trend_short_match_score"],
    )


def test_relative_high_turning_down_favors_short_score():
    settings = load_settings(env_file=None)
    result = evaluate_trend_match("SHORT", directional_bars("short"), settings, now=NOW)

    assert 40 <= result["metrics"]["range_position_pct"] <= 80
    assert result["metrics"]["ema50_slope_pct"] < 0
    assert result["trend_short_match_score"] > result["trend_long_match_score"]
    assert result["direction_bias"] == "short"


def test_composite_score_aggregation_can_use_mean():
    settings = load_settings(env_file=None)
    settings.trend_match.score_aggregation = "mean"
    result = evaluate_trend_match("BOTH", directional_bars("short"), settings, now=NOW)

    assert result["score_aggregation"] == "mean"
    assert result["trend_match_score"] == round(
        (result["trend_long_match_score"] + result["trend_short_match_score"]) / 2.0, 2,
    )


def test_extreme_one_sided_high_and_low_are_common_deductions():
    settings = load_settings(env_file=None)
    high = evaluate_trend_match("HIGH", bars_for("extreme_high"), settings, now=NOW)
    low = evaluate_trend_match("LOW", bars_for("extreme_low"), settings, now=NOW)

    for result in (high, low):
        assert "extreme_one_sided_move" in result["common_deductions"]
        assert "abnormal_recent_move" in result["common_deductions"]
        assert "extreme_one_sided_move" in result["long"]["deduction_reasons"]
        assert "extreme_one_sided_move" in result["short"]["deduction_reasons"]
        long_before_common = sum(result["long"]["weighted_contributions"].values())
        short_before_common = sum(result["short"]["weighted_contributions"].values())
        expected = max(long_before_common, short_before_common) - sum(result["common_deductions"].values())
        assert result["trend_match_score"] == round(max(0.0, expected), 2)


def test_open_4h_bar_does_not_change_trend_match_score_or_metrics():
    settings = load_settings(env_file=None)
    completed = bars_for("early_strength")
    open_bar = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.0, 100.0]],
        columns=completed.columns,
        index=[pd.Timestamp("2026-10-16 08:00:00+00:00")],
    )

    baseline = evaluate_trend_match("EARLY", completed, settings, now=NOW)
    with_open_bar = evaluate_trend_match("EARLY", pd.concat([completed, open_bar]), settings, now=NOW)

    assert with_open_bar["score"] == baseline["score"]
    assert with_open_bar["trend_long_match_score"] == baseline["trend_long_match_score"]
    assert with_open_bar["trend_short_match_score"] == baseline["trend_short_match_score"]
    assert with_open_bar["trend_match_score"] == baseline["trend_match_score"]
    assert with_open_bar["metrics"] == baseline["metrics"]
    assert with_open_bar["last_closed_bar_at"] == baseline["last_closed_bar_at"]


def test_insufficient_history_is_unavailable_instead_of_zero_score():
    settings = load_settings(env_file=None)

    result = evaluate_trend_match("SHORT", bars_for("range", count=30), settings, now=NOW)

    assert result["score"] is None
    assert result["trend_long_match_score"] is None
    assert result["trend_short_match_score"] is None
    assert result["trend_match_score"] is None
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "insufficient_closed_bars"
