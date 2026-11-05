from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.config import load_settings
from bot import symbol_regime as symbol_regime_module
from bot.symbol_regime import SymbolRegimeState, classify_symbol_regime, classify_symbol_regimes


NOW = datetime(2026, 10, 16, 8, 30, tzinfo=timezone.utc)


def bars_for(kind: str, count: int = 100) -> pd.DataFrame:
    index = pd.date_range(end="2026-10-16 04:00:00+00:00", periods=count, freq="4h", name="timestamp")
    values = []
    for i in range(count):
        if kind == "up":
            close = 100 + i * 0.5
            open_price, high, low = close - 0.1, close + 0.3, close - 0.3
        elif kind == "down":
            close = 150 - i * 0.5
            open_price, high, low = close + 0.1, close + 0.3, close - 0.3
        else:
            close = 100.0
            open_price, high, low = close, close + 1, close - 1
        values.append((open_price, high, low, close, 100.0))
    return pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"], index=index)


def test_trend_up_down_and_range_classify_from_completed_bars():
    settings = load_settings(env_file=None)

    up = classify_symbol_regime("UP/USDT:USDT", bars_for("up"), settings, now=NOW)
    down = classify_symbol_regime("DOWN/USDT:USDT", bars_for("down"), settings, now=NOW)
    sideways = classify_symbol_regime("FLAT/USDT:USDT", bars_for("flat"), settings, now=NOW)

    assert up.state is SymbolRegimeState.TREND
    assert up.reason == "trend_confirmed_up"
    assert down.state is SymbolRegimeState.TREND
    assert down.reason == "trend_confirmed_down"
    assert sideways.state is SymbolRegimeState.RANGE
    assert sideways.reason == "range_confirmed"
    assert up.to_dict()["mode"] == "strategy_routing"
    evidence = up.to_dict()["evidence"]
    assert evidence["confirmation_pass"] is True
    assert evidence["trend_confirmation_current_streak"] == 48
    assert evidence["trend_confirmation_required_bars"] == settings.symbol_regime.confirm_bars
    assert evidence["condition_failures"]["TREND_UP"] == []
    assert evidence["condition_checks"]["RANGE"]["structure_neutral"] is False


def test_open_4h_bar_is_excluded_from_state_and_evidence():
    settings = load_settings(env_file=None)
    completed = bars_for("up")
    current_open = pd.DataFrame(
        [[100.0, 101.0, 20.0, 21.0, 100.0]],
        columns=completed.columns,
        index=[pd.Timestamp("2026-10-16 08:00:00+00:00")],
    )

    baseline = classify_symbol_regime("AAA/USDT:USDT", completed, settings, now=NOW)
    with_open_bar = classify_symbol_regime("AAA/USDT:USDT", pd.concat([completed, current_open]), settings, now=NOW)

    assert with_open_bar.state is baseline.state
    assert with_open_bar.to_dict()["last_closed_bar_at"] == "2026-10-16T08:00:00+00:00"
    assert with_open_bar.to_dict()["evidence"]["close"] == baseline.to_dict()["evidence"]["close"]


def test_insufficient_missing_stale_and_gapped_data_are_uncertain():
    settings = load_settings(env_file=None)
    assert classify_symbol_regime("AAA", None, settings, now=NOW).reason == "bars_missing"
    short = classify_symbol_regime("AAA", bars_for("up", count=20), settings, now=NOW)
    assert short.state is SymbolRegimeState.UNCERTAIN
    assert short.reason == "insufficient_closed_bars"
    assert short.last_closed_bar_at == datetime(2026, 10, 16, 8, tzinfo=timezone.utc)

    stale_time = datetime(2026, 10, 16, 16, 10, tzinfo=timezone.utc)
    stale = classify_symbol_regime("AAA", bars_for("up"), settings, now=stale_time)
    assert stale.state is SymbolRegimeState.UNCERTAIN
    assert stale.reason == "stale_closed_bar"

    gapped_bars = bars_for("up").drop(pd.Timestamp("2026-10-15 12:00:00+00:00"))
    gap = classify_symbol_regime("AAA", gapped_bars, settings, now=NOW)
    assert gap.state is SymbolRegimeState.UNCERTAIN
    assert gap.reason == "closed_bar_gap"


def test_confirmation_requires_consecutive_candidate_bars_and_conflicts_are_uncertain():
    settings = load_settings(env_file=None)
    policy = settings.symbol_regime.model_copy(update={
        "trend_adx_min": 1.0,
        "range_adx_max": 0.5,
        "trend_slope_min_pct": 0.0001,
        "range_slope_max_pct": 0.00005,
    })
    conflict_settings = settings.model_copy(update={"symbol_regime": policy})
    conflict_bars = bars_for("up")
    last = conflict_bars.iloc[-1].copy()
    last["low"] = 100.0
    conflict_bars.iloc[-1] = last

    assessment = classify_symbol_regime("MIXED/USDT:USDT", conflict_bars, conflict_settings, now=NOW)

    assert assessment.state is SymbolRegimeState.UNCERTAIN
    assert assessment.reason in {"trend_structure_conflict", "trend_components_conflict"}
    evidence = assessment.to_dict()["evidence"]
    assert evidence["confirmation_pass"] is False
    assert evidence["condition_failures"]["TREND_UP"]
    assert evidence["condition_failures"]["TREND_DOWN"]


def test_uncertain_single_missing_regime_condition_gets_near_tag(monkeypatch):
    settings = load_settings(env_file=None)

    def one_condition_missing(closed, _policy):
        count = len(closed)
        candidates = ["UNCERTAIN"] * count
        reasons = ["trend_components_conflict"] * count
        metrics = {
            "raw_candidate": "UNCERTAIN",
            "raw_reason": "trend_components_conflict",
            "condition_failures": {
                "TREND_UP": ["structure_not_up"],
                "TREND_DOWN": ["ema_order_not_down", "structure_not_down"],
                "RANGE": ["adx_above_range_max", "structure_not_neutral"],
            },
        }
        return candidates, {"metrics": metrics, "raw_reasons": reasons}

    monkeypatch.setattr(symbol_regime_module, "_raw_candidates", one_condition_missing)
    assessment = classify_symbol_regime("AAA/USDT:USDT", bars_for("flat"), settings, now=NOW)

    assert assessment.state is SymbolRegimeState.UNCERTAIN
    evidence = assessment.to_dict()["evidence"]
    assert evidence["near_trend"] is True
    assert evidence["near_range"] is False
    assert evidence["near_miss_conditions"] == {"TREND_UP": ["structure_not_up"]}


def test_structure_only_near_range_shadow_requires_each_range_numeric_gate(monkeypatch):
    settings = load_settings(env_file=None)

    def structure_only(closed, _policy):
        count = len(closed)
        shadow_bars = [None] * (count - 2) + [
            {"bar_at": "2026-10-16T00:00:00+00:00", "numeric_conditions_pass": True,
             "structure": "DOWN"},
            {"bar_at": "2026-10-16T04:00:00+00:00", "numeric_conditions_pass": True,
             "structure": "DOWN"},
        ]
        return ["UNCERTAIN"] * count, {
            "metrics": {
                "raw_candidate": "UNCERTAIN",
                "raw_reason": "no_clear_regime",
                "condition_failures": {"RANGE": ["structure_not_neutral"]},
            },
            "raw_reasons": ["no_clear_regime"] * count,
            "range_structure_shadow_bars": shadow_bars,
        }

    monkeypatch.setattr(symbol_regime_module, "_raw_candidates", structure_only)
    assessment = classify_symbol_regime("AAA/USDT:USDT", bars_for("flat"), settings, now=NOW)

    evidence = assessment.to_dict()["evidence"]
    assert assessment.state is SymbolRegimeState.UNCERTAIN
    assert evidence["near_range_structure_only"] is True
    assert evidence["range_structure_shadow"]["eligible"] is True
    assert evidence["range_structure_shadow"]["direction_consistent"] is True
    assert evidence["range_structure_shadow"]["required_bars"] == 2


def test_uncertain_one_missing_confirmation_gate_gets_near_trend_tag(monkeypatch):
    settings = load_settings(env_file=None)

    def one_trend_confirmation_missing(closed, _policy):
        candidates = ["RANGE"] * (len(closed) - 4) + [
            "TREND_UP", "TREND_UP", "UNCERTAIN", "TREND_UP",
        ]
        reasons = ["range_components_aligned"] * (len(closed) - 4) + [
            "trend_components_aligned_up", "trend_components_aligned_up",
            "trend_structure_conflict", "trend_components_aligned_up",
        ]
        metrics = {
            "raw_candidate": "TREND_UP",
            "raw_reason": "trend_components_aligned_up",
            "condition_failures": {"TREND_UP": []},
        }
        return candidates, {"metrics": metrics, "raw_reasons": reasons}

    monkeypatch.setattr(symbol_regime_module, "_raw_candidates", one_trend_confirmation_missing)
    assessment = classify_symbol_regime("AAA/USDT:USDT", bars_for("flat"), settings, now=NOW)

    assert assessment.state is SymbolRegimeState.UNCERTAIN
    evidence = assessment.to_dict()["evidence"]
    assert assessment.reason == "confirmation_not_met"
    assert evidence["near_trend"] is True
    assert evidence["near_range"] is False
    assert evidence["near_miss_conditions"] == {
        "TREND_UP": ["trend_confirmation_not_met"],
    }


def test_one_new_candidate_bar_does_not_pass_confirmation(monkeypatch):
    settings = load_settings(env_file=None)

    def one_trend_candidate(closed, _policy):
        candidates = ["RANGE"] * (len(closed) - 1) + ["TREND_UP"]
        reasons = ["range_components_aligned"] * (len(closed) - 1) + ["trend_components_aligned_up"]
        metrics = {"raw_candidate": "TREND_UP", "raw_reason": "trend_components_aligned_up"}
        return candidates, {"metrics": metrics, "raw_reasons": reasons}

    monkeypatch.setattr(symbol_regime_module, "_raw_candidates", one_trend_candidate)
    assessment = classify_symbol_regime("AAA/USDT:USDT", bars_for("flat"), settings, now=NOW)

    assert assessment.state is SymbolRegimeState.UNCERTAIN
    assert assessment.reason == "confirmation_not_met"
    assert assessment.to_dict()["evidence"]["confirmation_candidates"] == [
        "RANGE", "RANGE", "TREND_UP",
    ]
    assert assessment.to_dict()["evidence"]["trend_confirmation_current_streak"] == 1


def test_range_keeps_two_bar_confirmation_when_trend_requires_four(monkeypatch):
    settings = load_settings(env_file=None)

    def two_range_candidates(closed, _policy):
        candidates = ["UNCERTAIN"] * (len(closed) - 2) + ["RANGE", "RANGE"]
        reasons = ["no_clear_regime"] * (len(closed) - 2) + [
            "range_components_aligned", "range_components_aligned",
        ]
        metrics = {"raw_candidate": "RANGE", "raw_reason": "range_components_aligned"}
        return candidates, {"metrics": metrics, "raw_reasons": reasons}

    monkeypatch.setattr(symbol_regime_module, "_raw_candidates", two_range_candidates)
    assessment = classify_symbol_regime("AAA/USDT:USDT", bars_for("flat"), settings, now=NOW)

    assert assessment.state is SymbolRegimeState.RANGE
    assert assessment.reason == "range_confirmed"
    evidence = assessment.to_dict()["evidence"]
    assert evidence["confirmation_candidates"] == ["RANGE", "RANGE"]
    assert evidence["trend_confirmation_pass"] is False
    assert evidence["range_confirmation_pass"] is True


def test_batch_classification_keeps_symbols_independent_and_reports_fetch_errors():
    settings = load_settings(env_file=None)
    assessments = classify_symbol_regimes({
        "UP/USDT:USDT": bars_for("up"),
        "FLAT/USDT:USDT": bars_for("flat"),
        "MISSING/USDT:USDT": None,
    }, settings, now=NOW, fetch_errors={"MISSING/USDT:USDT": "NetworkError"})

    assert assessments["UP/USDT:USDT"].state is SymbolRegimeState.TREND
    assert assessments["FLAT/USDT:USDT"].state is SymbolRegimeState.RANGE
    assert assessments["MISSING/USDT:USDT"].state is SymbolRegimeState.UNCERTAIN
    assert assessments["MISSING/USDT:USDT"].to_dict()["evidence"]["fetch_error"] == "NetworkError"


def test_naive_evaluation_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_symbol_regime("AAA", bars_for("flat"), load_settings(env_file=None),
                               now=datetime(2026, 10, 16, 8, 30))


def test_range_structure_mode_gate():
    ok = symbol_regime_module._range_structure_ok
    for state in ("NEUTRAL", "UP", "DOWN", "CONFLICT"):
        assert ok(state, "any") is True
        assert ok(state, "neutral") is (state == "NEUTRAL")
        assert ok(state, "not_conflict") is (state in ("NEUTRAL", "UP", "DOWN"))
