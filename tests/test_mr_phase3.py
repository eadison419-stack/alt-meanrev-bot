import pandas as pd

from bot.config import load_settings
from bot.regime import Regime
from bot.signals import Side
from scripts.replay_mr_baseline import (
    HYPOTHESES, _daily_history_is_rankable, _ema_space_r, _entry_deviation_thresholds,
    _funding_gate_for_side, _has_raw_signal, _reversal_body_midpoint_pass,
)


def _four_hour_frame(periods=100):
    index = pd.date_range("2026-01-01", periods=periods, freq="4h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 10.0}, index=index)


def test_daily_prefilter_requires_51_closed_bars_and_current_completed_day():
    at = pd.Timestamp("2026-10-10T00:00:00Z")
    daily = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
        index=pd.date_range(end=at - pd.Timedelta(1, unit="D"), periods=51, freq="D"),
    )
    assert _daily_history_is_rankable(daily, at, 50)
    assert not _daily_history_is_rankable(daily.iloc[1:], at, 50)
    assert not _daily_history_is_rankable(daily, at + pd.Timedelta(1, unit="D"), 50)


def test_atr_adaptive_threshold_uses_4h_atr_percent_and_explicit_floor():
    settings = load_settings(env_file=None)
    frame = _four_hour_frame()
    thresholds = _entry_deviation_thresholds(
        frame, settings, {}, "TEST", HYPOTHESES["atr_deviation_0p8"],
    )
    expected = max(0.01, 0.8 * 2 / 100)
    assert thresholds == {Side.LONG: expected, Side.SHORT: expected}


def test_impulse_hypothesis_tightens_only_counter_impulse_side():
    settings = load_settings(env_file=None)
    frame = _four_hour_frame()
    regimes = {"TEST": pd.DataFrame({"regime": [Regime.IMPULSE_UP]})}
    thresholds = _entry_deviation_thresholds(
        frame, settings, regimes, "TEST", HYPOTHESES["impulse_counter_deviation_1p2"],
    )
    assert thresholds[Side.LONG] == 0.01
    assert thresholds[Side.SHORT] == max(0.02, 1.2 * 2 / 100)


def test_side_specific_threshold_blocks_only_the_tightened_signal():
    settings = load_settings(env_file=None)
    candle = {"open": 102.0, "close": 101.5}
    assert _has_raw_signal(candle, 100.0, settings,
                           {Side.LONG: 0.01, Side.SHORT: 0.01})
    assert not _has_raw_signal(candle, 100.0, settings,
                               {Side.LONG: 0.01, Side.SHORT: 0.02})


def test_structure_stop_hypothesis_removes_only_the_two_atr_hard_cap():
    baseline = HYPOTHESES["baseline"]
    candidate = HYPOTHESES["structure_stop_2p5"]
    assert baseline.stop_hard_cap_atr is None
    assert candidate.stop_hard_cap_atr == 2.5
    assert candidate.deviation_rule == baseline.deviation_rule


def test_reversal_body_midpoint_is_directionally_symmetric():
    frame = pd.DataFrame({"open": [100.0, 100.0], "close": [102.0, 101.5]})
    assert _reversal_body_midpoint_pass(frame, Side.LONG)
    assert not _reversal_body_midpoint_pass(frame.assign(close=[102.0, 100.5]), Side.LONG)
    frame = pd.DataFrame({"open": [100.0, 100.0], "close": [98.0, 98.5]})
    assert _reversal_body_midpoint_pass(frame, Side.SHORT)
    assert not _reversal_body_midpoint_pass(frame.assign(close=[98.0, 99.5]), Side.SHORT)


def test_minimum_ema_space_uses_actual_stop_distance_and_side():
    assert _ema_space_r(Side.LONG, 98, 96, 100) == 1
    assert _ema_space_r(Side.SHORT, 102, 104, 100) == 1
    assert _ema_space_r(Side.LONG, 98, 96, 97) < 0
    assert _ema_space_r(Side.LONG, 98, 98, 100) == float("-inf")


def test_directional_funding_gate_blocks_only_adverse_extreme_rates():
    threshold = 0.0008
    assert _funding_gate_for_side(threshold, Side.LONG, threshold) == "funding_rate_blocked"
    assert _funding_gate_for_side(-threshold, Side.SHORT, threshold) == "funding_rate_blocked"
    assert _funding_gate_for_side(-0.004, Side.LONG, threshold) is None
    assert _funding_gate_for_side(0.004, Side.SHORT, threshold) is None
    assert _funding_gate_for_side(float("nan"), Side.LONG, threshold) == "funding_rate_missing"

