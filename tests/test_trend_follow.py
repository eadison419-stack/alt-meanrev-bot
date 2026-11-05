"""Unit tests for the Donchian breakout long-only strategy module."""

import numpy as np
import pandas as pd

from bot.trend_follow import (BreakoutConfig, breadth_ok, breakeven_level,
                              indicators, long_entry, long_exit,
                              market_breadth, position_size)


def _frame(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


# bars: (open, high, low, close)
BARS = [
    (100, 110, 95, 105),
    (105, 115, 100, 110),
    (110, 112, 105, 108),
    (108, 120, 107, 118),
    (118, 125, 116, 124),
    (124, 130, 122, 128),
]

CFG = BreakoutConfig(entry_bars=3, exit_bars=2, atr_len=2, atr_mult=2.0)


def test_indicators_upper_and_exit_lower_shift_semantics():
    ind = indicators(_frame(BARS), CFG)
    # upper[i] = max(high[i-3:i]) (excludes current bar)
    assert ind.upper[3] == 115.0   # max(110,115,112)
    assert ind.upper[4] == 120.0   # max(115,112,120)
    assert ind.upper[5] == 125.0   # max(112,120,125)
    # exit_lower[i] = min(low[i-2:i])
    assert ind.exit_lower[3] == 100.0  # min(100,105)
    assert ind.exit_lower[4] == 105.0  # min(105,107)
    assert ind.exit_lower[5] == 107.0  # min(107,116)


def test_atr_is_true_range_mean():
    ind = indicators(_frame(BARS), CFG)
    # TR: 15, 15, 7, 13, 9, 8 ; rolling(2).mean
    assert ind.atr[5] == pytest.approx((9 + 8) / 2)


def test_long_entry_triggers_on_breakout_with_atr_stop():
    triggered, stop = long_entry(_frame(BARS), CFG)
    assert triggered is True
    assert stop == pytest.approx(128.0 - 2.0 * 8.5)  # close - 2x ATR


def test_long_entry_no_signal_when_close_below_upper():
    bars = BARS[:-1]  # drop the breakout bar: close=124 < upper[4]=120? no, 124>120
    # make a frame that does not break out
    bars = BARS[:4] + [(118, 119, 110, 112)]  # close 112 < upper
    triggered, _ = long_entry(_frame(bars), CFG)
    assert triggered is False


def test_long_exit_reverse_breakout():
    bars = BARS + [(128, 129, 116, 101)]  # low stays above stop, close breaks exit_lower
    triggered, reason = long_exit(_frame(bars), CFG, stop=111.0)
    assert triggered is True
    assert reason == "reverse"


def test_long_exit_stop_hit():
    bars = BARS + [(128, 129, 110, 111)]  # low pierces stop=111
    triggered, reason = long_exit(_frame(bars), CFG, stop=111.0)
    assert triggered is True
    assert reason == "stop"


def test_long_exit_no_signal():
    bars = BARS + [(128, 129, 124, 127)]  # neither stop nor reverse
    triggered, reason = long_exit(_frame(bars), CFG, stop=111.0)
    assert triggered is False


def test_position_size_risks_slot_pct():
    size = position_size(equity_per_slot=2000.0, risk_pct=0.01, atr_mult=2.0, atr=10.0)
    # risk = 2000*1% = 20 ; stop distance = 2*10 = 20 ; qty = 1.0
    assert size == pytest.approx(1.0)


def test_position_size_zero_on_bad_atr():
    assert position_size(2000.0, 0.01, 2.0, 0.0) == 0.0
    assert position_size(2000.0, 0.01, 2.0, np.nan) == 0.0


def test_breakout_config_be_mult_defaults_off():
    assert BreakoutConfig().be_mult is None


def test_breakeven_level_disabled_when_none():
    assert breakeven_level(100.0, 5.0, 999.0, None) is None


def test_breakeven_level_not_reached():
    assert breakeven_level(100.0, 5.0, 109.99, 2.0) is None  # 100 + 2*5 = 110


def test_breakeven_level_reached_at_threshold():
    assert breakeven_level(100.0, 5.0, 110.0, 2.0) == 100.0


def test_breakeven_level_raises_only_to_entry():
    assert breakeven_level(100.0, 5.0, 200.0, 2.0) == 100.0


def test_breakeven_level_bad_atr_returns_none():
    assert breakeven_level(100.0, 0.0, 999.0, 2.0) is None
    assert breakeven_level(100.0, float("nan"), 999.0, 2.0) is None


def test_breakout_config_breadth_min_defaults_off():
    assert BreakoutConfig().breadth_min is None


def _trend_frame(n=60, up=True):
    closes = np.linspace(100.0, 160.0, n) if up else np.linspace(160.0, 100.0, n)
    return pd.DataFrame({"open": closes, "high": closes + 1.0,
                         "low": closes - 1.0, "close": closes})


def test_breadth_ok_disabled_when_none():
    assert breadth_ok(0.1, None) is True


def test_breadth_ok_allows_at_or_above_threshold():
    assert breadth_ok(0.6, 0.6) is True
    assert breadth_ok(0.7, 0.6) is True


def test_breadth_ok_blocks_below_threshold():
    assert breadth_ok(0.59, 0.6) is False


def test_breadth_ok_blocks_when_breadth_unknown_and_gate_enabled():
    assert breadth_ok(None, 0.6) is False


def test_market_breadth_returns_fraction_above_ema():
    frames = {"UP": _trend_frame(60, up=True), "DOWN": _trend_frame(60, up=False)}
    assert market_breadth(frames, ema_span=50) == pytest.approx(0.5)


def test_market_breadth_all_above():
    frames = {"A": _trend_frame(60, up=True), "B": _trend_frame(60, up=True)}
    assert market_breadth(frames, ema_span=50) == pytest.approx(1.0)


def test_market_breadth_none_when_no_symbol_has_enough_history():
    frames = {"SHORT": _trend_frame(20, up=True)}
    assert market_breadth(frames, ema_span=50) is None


import pytest  # noqa: E402  (pytest.approx used above)
