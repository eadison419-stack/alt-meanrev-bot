"""Unit tests for the mr_candidate pure strategy functions."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.config import load_settings
from bot.mr_candidate import (MidExitState, breakeven_stop, claim_hourly_entry,
                              deviation_entry_signal, deviation_entry_signal_at_price,
                              mid_band, mid_exit_check, no_progress_update,
                              regime_faded_triggered, structure_invalid_quote_exit,
                              structure_invalid_triggered, structure_reentry_confirmed,
                              trailing_stop)
from bot.regime import Regime
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts.run_mr_candidate_paper import (
    _current_hour_observation,
    _regime_fade_evaluation_event,
    _ticker_price,
)


def _flags(long=True, short=True, flatten=False):
    return TimeFilterFlags(long, short, flatten, False)


def _candle(open_, close):
    return pd.Series({"open": open_, "close": close})


@pytest.fixture(scope="module")
def settings():
    return load_settings(env_file=None)


def _ema50_series(last_close):
    closes = [100.0] * 80 + [last_close]
    return pd.Series(closes, index=pd.date_range("2026-01-01", periods=len(closes), freq="4h"))


def test_deviation_entry_long_below_ema_and_bullish(settings):
    s = deviation_entry_signal(_ema50_series(98.5), _candle(98.2, 98.6), settings,
                               _flags(), deviation_pct=0.01)
    assert s.action is Action.OPEN and s.side is Side.LONG


def test_deviation_entry_short_above_ema_and_bearish(settings):
    s = deviation_entry_signal(_ema50_series(101.5), _candle(101.8, 101.4), settings,
                               _flags(), deviation_pct=0.01)
    assert s.action is Action.OPEN and s.side is Side.SHORT


def test_live_deviation_entry_uses_current_hour_open_and_price(settings):
    long_signal = deviation_entry_signal_at_price(
        _ema50_series(98.5), 98.2, 98.6, settings, _flags(), deviation_pct=0.01,
    )
    short_signal = deviation_entry_signal_at_price(
        _ema50_series(101.5), 101.8, 101.4, settings, _flags(), deviation_pct=0.01,
    )
    assert long_signal.action is Action.OPEN and long_signal.side is Side.LONG
    assert short_signal.action is Action.OPEN and short_signal.side is Side.SHORT


def test_live_deviation_requires_reversal_from_current_hour_open(settings):
    long_without_reversal = deviation_entry_signal_at_price(
        _ema50_series(98.5), 98.7, 98.6, settings, _flags(), deviation_pct=0.01,
    )
    short_without_reversal = deviation_entry_signal_at_price(
        _ema50_series(101.5), 101.3, 101.4, settings, _flags(), deviation_pct=0.01,
    )
    assert long_without_reversal.action is Action.NONE
    assert short_without_reversal.action is Action.NONE
    assert long_without_reversal.reason == "hourly_candle_direction_mismatch"
    assert short_without_reversal.reason == "hourly_candle_direction_mismatch"


def test_hourly_entry_claim_is_once_per_symbol_and_bar():
    attempts = {}
    assert claim_hourly_entry(attempts, "APR/USDT:USDT", "2026-10-26T13:00:00+00:00")
    assert not claim_hourly_entry(attempts, "APR/USDT:USDT", "2026-10-26T13:00:00+00:00")
    assert claim_hourly_entry(attempts, "APR/USDT:USDT", "2026-10-26T14:00:00+00:00")


def test_intrabar_context_uses_only_the_current_hour_candle():
    rows = [
        [0, 10, 11, 9, 10.5, 100],
        [3_600_000, 10.5, 12, 10, 11.5, 120],
    ]
    observation = _current_hour_observation(rows, 4_000_000)
    assert observation["bar_open_ms"] == 3_600_000
    assert observation["open"] == 10.5
    assert observation["hour_key"] == "1970-01-01T01:00:00+00:00"
    assert _current_hour_observation(rows, 7_200_000) is None


def test_intrabar_ticker_rejects_stale_or_invalid_price():
    now_ms = 10_000_000
    assert _ticker_price({"APR/USDT:USDT": {"last": 1.25, "timestamp": now_ms - 1000}},
                         "APR/USDT:USDT", now_ms) == 1.25
    assert _ticker_price({"APR/USDT:USDT": {"last": 1.25, "timestamp": now_ms - 121_000}},
                         "APR/USDT:USDT", now_ms) is None
    assert _ticker_price({"APR/USDT:USDT": {"last": float("inf")}},
                         "APR/USDT:USDT", now_ms) is None


def test_deviation_entry_none_within_threshold(settings):
    s = deviation_entry_signal(_ema50_series(100.2), _candle(100.1, 100.3), settings,
                               _flags(), deviation_pct=0.01)
    assert s.action is Action.NONE
    assert s.reason == "deviation_not_met"


def test_deviation_entry_blocks_when_flatten_only(settings):
    s = deviation_entry_signal(_ema50_series(98.0), _candle(97.8, 98.1), settings,
                               _flags(flatten=True), deviation_pct=0.01)
    assert s.action is Action.NONE


def test_deviation_entry_blocks_when_time_filter_data_is_unavailable(settings):
    unavailable = TimeFilterFlags(
        True, True, False, False, data_ready=False, data_reason="hourly_history_gap",
    )

    signal = deviation_entry_signal(
        _ema50_series(98.0), _candle(97.8, 98.1), settings,
        unavailable, deviation_pct=0.01,
    )

    assert signal.action is Action.NONE
    assert signal.reason == "time_filter_data_unavailable"


def test_mid_band_ema_vs_sma():
    closes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ema = mid_band(closes, 3, True)
    sma = mid_band(closes, 3, False)
    assert float(ema.iloc[-1]) > float(sma.iloc[-1])  # EMA hugs the most recent close


def test_mid_exit_arms_then_fires_long():
    state = MidExitState()
    fire, state = mid_exit_check(Side.LONG, 99.0, 100.0, 100.0, hold_bars=0, state=state)
    assert not fire and state.reached_mid
    fire, state = mid_exit_check(Side.LONG, 99.0, 100.0, 99.5, hold_bars=0, state=state)
    assert fire and state.fired


def test_mid_exit_entry_side_gate_blocks_wrong_side():
    state = MidExitState()
    # long entered above mid should never arm
    fire, state = mid_exit_check(Side.LONG, 101.0, 100.0, 100.0, hold_bars=0, state=state)
    assert not fire and not state.reached_mid


def test_mid_exit_hold_bars_delays_arming():
    state = MidExitState()
    fire, state = mid_exit_check(Side.LONG, 99.0, 100.0, 100.0, hold_bars=1, state=state)
    assert not fire and not state.reached_mid  # needs a 2nd bar
    fire, state = mid_exit_check(Side.LONG, 99.0, 100.0, 100.0, hold_bars=1, state=state)
    assert not fire and state.reached_mid


def test_breakeven_stop_raises_to_entry(settings):
    stop = breakeven_stop(Side.LONG, 100.0, 2.0, 2.0, 95.0,
                          move_to_breakeven_atr=1.0, fee_rate=0.0, slip_rate=0.0)
    assert stop == 100.0


def test_breakeven_stop_not_before_threshold(settings):
    stop = breakeven_stop(Side.LONG, 100.0, 2.0, 1.0, 95.0,
                          move_to_breakeven_atr=1.0, fee_rate=0.0, slip_rate=0.0)
    assert stop == 95.0


def test_trailing_stop_after_trigger(settings):
    stop = trailing_stop(Side.LONG, 110.0, 5.0, 3.0, 100.0,
                         trigger_r=2.5, atr_mult=2.0, stop_atr_mult=2.0)
    assert stop == 105.0


def test_trailing_stop_before_trigger_unchanged(settings):
    stop = trailing_stop(Side.LONG, 110.0, 5.0, 2.0, 100.0,
                         trigger_r=2.5, atr_mult=2.0, stop_atr_mult=2.0)
    assert stop == 100.0


def test_structure_invalid_long(settings):
    assert structure_invalid_triggered(Side.LONG, 99.0, 100.0, None, 2.0, buffer_atr=0.25)
    assert not structure_invalid_triggered(Side.LONG, 99.7, 100.0, None, 2.0, buffer_atr=0.25)


@pytest.mark.parametrize(
    ("side", "close", "expected"),
    [(Side.LONG, 101.1, True), (Side.LONG, 101.0, False),
     (Side.SHORT, 98.9, True), (Side.SHORT, 99.0, False)],
)
def test_structure_reentry_requires_close_beyond_previous_hour_extreme(side, close, expected):
    assert structure_reentry_confirmed(side, 101.0, 99.0, close) is expected


def test_structure_reentry_rejects_invalid_candle_values():
    assert not structure_reentry_confirmed(Side.LONG, float("nan"), 99.0, 101.1)
    assert not structure_reentry_confirmed(Side.SHORT, 98.0, 99.0, 98.9)


def test_fresh_quote_structure_exit_precedes_long_hard_stop():
    assert structure_invalid_quote_exit(
        Side.LONG, 99.4, 100.0, None, 2.0, 98.0, buffer_atr=0.25,
    ) == ("long_structure_invalid", 99.5)
    assert structure_invalid_quote_exit(
        Side.LONG, 99.6, 100.0, None, 2.0, 98.0, buffer_atr=0.25,
    ) is None
    assert structure_invalid_quote_exit(
        Side.LONG, 99.4, 100.0, None, 2.0, 99.6, buffer_atr=0.25,
    ) is None


def test_fresh_quote_structure_exit_precedes_short_hard_stop():
    assert structure_invalid_quote_exit(
        Side.SHORT, 100.6, None, 100.0, 2.0, 102.0, buffer_atr=0.25,
    ) == ("short_structure_invalid", 100.5)
    assert structure_invalid_quote_exit(
        Side.SHORT, 100.4, None, 100.0, 2.0, 102.0, buffer_atr=0.25,
    ) is None
    assert structure_invalid_quote_exit(
        Side.SHORT, 100.6, None, 100.0, 2.0, 100.4, buffer_atr=0.25,
    ) is None


def test_regime_faded_long(settings):
    assert regime_faded_triggered(Side.LONG, Regime.LOW.value, Regime.SIDEWAYS, 0.5, 2.0, min_atr=0.5)
    assert not regime_faded_triggered(Side.LONG, Regime.LOW.value, Regime.LOW, 0.5, 2.0, min_atr=0.5)
    assert not regime_faded_triggered(Side.LONG, Regime.LOW.value, Regime.SIDEWAYS, 1.5, 2.0, min_atr=0.5)


@pytest.mark.parametrize(
    ("progress_atr", "expected_decision"),
    [(0.49, "triggered"), (0.50, "not_triggered"), (0.51, "not_triggered")],
)
@pytest.mark.parametrize(
    ("side", "entry_regime"),
    [(Side.LONG, Regime.LOW), (Side.SHORT, Regime.HIGH)],
)
def test_regime_fade_audit_records_strict_atr_threshold(
    progress_atr, expected_decision, side, entry_regime,
):
    event = _regime_fade_evaluation_event(
        symbol="AAA/USDT:USDT",
        side=side,
        entry_regime=entry_regime,
        current_regime=Regime.SIDEWAYS,
        progress=progress_atr * 2.0,
        entry_atr=2.0,
        min_progress_atr=0.5,
        at="2026-11-02T05:00:00+00:00",
        one_hour_bar_open_at="2026-11-02T04:00:00+00:00",
        regime_bar_open_at="2026-11-02T04:00:00+00:00",
        in_hotlist=False,
        hotlist_rank=None,
    )

    assert event["decision"] == expected_decision
    assert event["triggered"] is (expected_decision == "triggered")
    assert event["progress_atr"] == pytest.approx(progress_atr)
    assert event["entry_regime_matches"] is True
    assert event["current_regime_is_sideways"] is True
    assert event["in_hotlist"] is False


def test_regime_fade_audit_marks_missing_regime_as_skipped():
    event = _regime_fade_evaluation_event(
        symbol="AAA/USDT:USDT",
        side=Side.LONG,
        entry_regime=Regime.LOW,
        current_regime=None,
        progress=0.2,
        entry_atr=1.0,
        min_progress_atr=0.5,
        at="2026-11-02T05:00:00+00:00",
        one_hour_bar_open_at="2026-11-02T04:00:00+00:00",
        regime_bar_open_at=None,
        in_hotlist=False,
        hotlist_rank=None,
    )

    assert event["decision"] == "skipped"
    assert event["skip_reason"] == "regime_data_unavailable"
    assert event["triggered"] is False


def test_no_progress_triggers_after_bars(settings):
    trigger, count = no_progress_update(Side.LONG, 99.0, 100.0, 5, 0.5, 2.0, bars=6, min_atr=0.4)
    assert trigger and count == 6


def test_no_progress_resets_on_favorable(settings):
    trigger, count = no_progress_update(Side.LONG, 101.0, 100.0, 5, 0.5, 2.0, bars=6, min_atr=0.4)
    assert not trigger and count == 0


def test_trade_risk_cap_is_two_percent(settings):
    assert settings.risk.max_single_trade_risk_pct == 0.02
