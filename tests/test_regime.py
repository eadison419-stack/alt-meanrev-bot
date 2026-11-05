import numpy as np
import pandas as pd
import pytest

from bot.config import load_settings
from bot.indicators import indicator_frame
from bot.regime import Regime, RegimeSnapshot, classify_regime, classify_regimes


def trend_frame(length=120, *, start=100.0, step=0.1):
    close = float(start) + np.arange(length) * float(step)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=pd.date_range("2025-01-01", periods=length, freq="4h", tz="UTC"),
    )


def install_metrics(monkeypatch, *, atr=1.0, adx=25.0, atr_percentile_series=None):
    import bot.regime as module

    def fake_indicators(frame, atr_period, adx_period):
        values = np.full(len(frame), atr, dtype=float)
        if atr_percentile_series is not None:
            values = np.asarray(atr_percentile_series, dtype=float)
        return pd.DataFrame({"atr": values, "adx": adx, "plus_di": 20.0, "minus_di": 10.0}, index=frame.index)

    monkeypatch.setattr(module, "indicator_frame", fake_indicators)


def test_sideways_low_adx_and_low_atr_percentile_blocks_even_impulse(monkeypatch):
    settings = load_settings(env_file=None)
    frame = trend_frame()
    frame.loc[frame.index[-1], "close"] = frame.iloc[-1]["open"] + 10
    frame.loc[frame.index[-1], "high"] = frame.iloc[-1]["close"] + 0.1
    atr_values = np.linspace(2.0, 1.0, len(frame))
    install_metrics(monkeypatch, atr=1.0, adx=19.0, atr_percentile_series=atr_values)
    assert classify_regime(frame, settings).regime is Regime.SIDEWAYS


@pytest.mark.parametrize(("offset", "expected"), [(-6.0, Regime.LOW), (6.0, Regime.HIGH)])
def test_range_zones_use_previous_48_bars_and_current_bar_is_not_in_baseline(monkeypatch, offset, expected):
    settings = load_settings(env_file=None)
    frame = trend_frame(step=0)
    frame.loc[frame.index[-1], ["open", "close"]] = [100 + offset, 100 + offset]
    frame.loc[frame.index[-1], ["high", "low"]] = [101 + offset, 99 + offset]
    frame.loc[frame.index[:-1], "high"] = 110
    frame.loc[frame.index[:-1], "low"] = 90
    install_metrics(monkeypatch, atr=10.0, adx=25.0)
    snapshot = classify_regime(frame, settings)
    assert snapshot.regime is expected
    assert not snapshot.new_high and not snapshot.new_low


@pytest.mark.parametrize(("direction", "expected"), [(1, Regime.IMPULSE_UP), (-1, Regime.IMPULSE_DOWN)])
def test_impulse_has_precedence_over_high_low_zone(monkeypatch, direction, expected):
    settings = load_settings(env_file=None)
    frame = trend_frame(step=0)
    previous_close = float(frame["close"].iloc[-1])
    frame.loc[frame.index[-1], "open"] = previous_close
    frame.loc[frame.index[-1], "close"] = previous_close + direction * 3
    frame.loc[frame.index[-1], "high"] = max(previous_close, previous_close + direction * 3) + 0.1
    frame.loc[frame.index[-1], "low"] = min(previous_close, previous_close + direction * 3) - 0.1
    frame.loc[frame.index[:-1], "high"] = 110
    frame.loc[frame.index[:-1], "low"] = 90
    install_metrics(monkeypatch, atr=1.0, adx=25.0)
    assert classify_regime(frame, settings).regime is expected


@pytest.mark.parametrize(("direction", "expected"), [(1, Regime.SLOW_UP), (-1, Regime.SLOW_DOWN)])
def test_slow_trend_after_zone_checks(monkeypatch, direction, expected):
    settings = load_settings(env_file=None)
    frame = trend_frame(length=120, start=100, step=0)
    # A six-bar net move of 1.2 ATR, while the latest candle body stays below impulse.
    for offset in range(6, -1, -1):
        value = 100 + direction * (6 - offset) * 0.2
        frame.loc[frame.index[-1 - offset], ["open", "close"]] = [value, value]
        frame.loc[frame.index[-1 - offset], ["high", "low"]] = [value + 0.2, value - 0.2]
    frame.loc[frame.index[:-7], "high"] = 110
    frame.loc[frame.index[:-7], "low"] = 90
    # ADX/ATR controlled so this test isolates trend-distance and evaluation order.
    install_metrics(monkeypatch, atr=1.0, adx=25.0)
    assert classify_regime(frame, settings).regime is expected


def test_mid_when_no_other_condition_matches(monkeypatch):
    settings = load_settings(env_file=None)
    frame = trend_frame(length=120, start=100, step=0)
    frame.loc[frame.index[:-1], "high"] = 110
    frame.loc[frame.index[:-1], "low"] = 90
    install_metrics(monkeypatch, atr=1.0, adx=25.0)
    assert classify_regime(frame, settings).regime is Regime.MID


def test_regime_requires_minimum_history_and_timezone_aware_sorted_index():
    settings = load_settings(env_file=None)
    with pytest.raises(ValueError, match="at least 80"):
        classify_regime(trend_frame(length=79), settings)
    frame = trend_frame()
    frame.index = frame.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_regimes(frame, settings)


def test_atr_adx_and_regime_features_do_not_change_past_when_future_bar_added():
    settings = load_settings(env_file=None)
    frame = trend_frame()
    before = indicator_frame(frame, settings.lookback.atr_len, settings.lookback.adx_len)
    expanded = pd.concat([frame, trend_frame(length=1, start=500).set_axis([frame.index[-1] + pd.Timedelta(4, unit="h")])])
    after = indicator_frame(expanded, settings.lookback.atr_len, settings.lookback.adx_len)
    pd.testing.assert_series_equal(before.iloc[-1], after.iloc[-2], check_names=False)
    assert np.isfinite(before["atr"].iloc[-1])
    assert np.isfinite(before["adx"].iloc[-1])


def test_flat_market_has_defined_zero_adx_after_warmup():
    frame = trend_frame(length=100, step=0)
    values = indicator_frame(frame, 14, 14)
    assert values["adx"].iloc[-1] == 0.0
