from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from bot.config import load_settings
from bot.time_filter import evaluate_time_filter, precompute_time_filters

SHANGHAI = ZoneInfo("Asia/Shanghai")


def candles_ending_at(local_close: datetime, *, last_change: float = 0.0, prior_afternoon_dump: bool = False):
    closes = [100.0] * 55
    opens = [100.0] * 55
    highs = [101.0] * 55
    lows = [99.0] * 55
    close_time = local_close
    times = [close_time - timedelta(hours=54 - i) for i in range(55)]
    if prior_afternoon_dump:
        target_date = local_close.date() - timedelta(days=1)
        for i, ts in enumerate(times):
            if ts.date() == target_date and ts.hour == 15:
                opens[i], closes[i], highs[i], lows[i] = 100.0, 96.0, 100.5, 95.5
    opens[-1] = 100.0
    closes[-1] = 100.0 + last_change
    highs[-1] = max(opens[-1], closes[-1]) + 0.5
    lows[-1] = min(opens[-1], closes[-1]) - 0.5
    index = pd.DatetimeIndex([ts.astimezone(ZoneInfo("UTC")) - timedelta(hours=1) for ts in times])
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(55)}, index=index)


def test_morning_pump_blocks_long_and_morning_dump_blocks_short():
    settings = load_settings(env_file=None)
    up = candles_ending_at(datetime(2026, 9, 29, 9, tzinfo=SHANGHAI), last_change=4)
    flags = evaluate_time_filter(datetime(2026, 9, 29, 9, tzinfo=SHANGHAI), up, settings)
    assert flags.morning_move == "UP"
    assert not flags.allow_long_entry and flags.allow_short_entry
    down = candles_ending_at(datetime(2026, 9, 29, 10, tzinfo=SHANGHAI), last_change=-4)
    flags = evaluate_time_filter(datetime(2026, 9, 29, 10, tzinfo=SHANGHAI), down, settings)
    assert flags.morning_move == "DOWN"
    assert flags.allow_long_entry and not flags.allow_short_entry


def test_afternoon_pump_blocks_chase_long_and_spike_tip_short():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 15, tzinfo=SHANGHAI)
    frame = candles_ending_at(when, last_change=4)
    flags = evaluate_time_filter(when, frame, settings)
    assert not flags.allow_long_entry
    assert flags.at_upper_extreme and not flags.allow_short_entry


def test_afternoon_dump_blocks_longs_for_day_and_short_at_low_extreme():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 15, tzinfo=SHANGHAI)
    frame = candles_ending_at(when, last_change=-4)
    flags = evaluate_time_filter(when, frame, settings)
    assert flags.afternoon_dump_today
    assert flags.at_lower_extreme
    assert not flags.allow_long_entry and not flags.allow_short_entry


def test_afternoon_dump_sets_next_morning_watch_flag():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 30, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when, prior_afternoon_dump=True)
    flags = evaluate_time_filter(when, frame, settings)
    assert flags.watch_next_morning_long


def test_night_allows_new_entries_and_does_not_consume_future_bars():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 21, tzinfo=SHANGHAI)
    frame = candles_ending_at(when, last_change=4)
    flags = evaluate_time_filter(when, frame, settings)
    assert not flags.flatten_only
    assert flags.allow_long_entry and flags.allow_short_entry


def test_future_candles_are_ignored_and_stale_snapshot_fails_closed():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when, last_change=4)
    baseline = evaluate_time_filter(when, frame, settings)
    future = frame.copy()
    future_index = future.index[-1] + pd.Timedelta(1, unit="h")
    future.loc[future_index] = [100, 120, 99, 119, 1]
    assert evaluate_time_filter(when, future, settings) == baseline
    stale = evaluate_time_filter(when + timedelta(hours=1), frame, settings)
    assert not stale.allow_long_entry and not stale.allow_short_entry


def test_missing_history_fails_closed_during_entry_session():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when).tail(5)
    flags = evaluate_time_filter(when, frame, settings)
    assert not flags.allow_long_entry and not flags.allow_short_entry


def test_requires_24_prior_hourly_bars_for_range_filter_and_accepts_25():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when)

    insufficient = evaluate_time_filter(when, frame.tail(24), settings)
    sufficient = evaluate_time_filter(when, frame.tail(25), settings)

    assert not insufficient.allow_long_entry and not insufficient.allow_short_entry
    assert insufficient.data_ready is False
    assert insufficient.data_reason == "hourly_history_insufficient"
    assert sufficient.data_ready is True
    assert sufficient.allow_long_entry and sufficient.allow_short_entry


def test_hourly_gap_in_required_history_fails_closed():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when).drop(candles_ending_at(when).index[-10])

    flags = evaluate_time_filter(when, frame, settings)

    assert not flags.allow_long_entry and not flags.allow_short_entry
    assert flags.data_ready is False
    assert flags.data_reason == "hourly_history_gap"


def test_invalid_ohlcv_fails_closed():
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, 9, tzinfo=SHANGHAI)
    frame = candles_ending_at(when)
    frame.loc[frame.index[-1], "high"] = 1.0

    flags = evaluate_time_filter(when, frame, settings)

    assert not flags.allow_long_entry and not flags.allow_short_entry
    assert flags.data_ready is False
    assert flags.data_reason == "invalid_ohlcv"


def test_precomputed_time_filters_match_prefix_evaluation_without_future_data():
    settings = load_settings(env_file=None)
    frame = candles_ending_at(datetime(2026, 9, 29, 15, tzinfo=SHANGHAI), last_change=-4)
    prepared = precompute_time_filters(frame, settings)
    for index in (20, 35, len(frame) - 1):
        close_time = frame.index[index] + pd.Timedelta(1, unit="h")
        expected = evaluate_time_filter(close_time.to_pydatetime(), frame.iloc[: index + 1], settings)
        assert prepared[close_time] == expected


def test_naive_clock_is_rejected():
    settings = load_settings(env_file=None)
    frame = candles_ending_at(datetime(2026, 9, 29, 9, tzinfo=SHANGHAI))
    try:
        evaluate_time_filter(datetime(2026, 9, 29, 9), frame, settings)
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("naive timestamp accepted")


@pytest.mark.parametrize(
    ("hour", "expected_flatten"),
    [(0, False), (7, False), (8, False), (11, False), (12, False), (19, False), (20, False), (23, False)],
)
def test_shanghai_session_boundaries(hour, expected_flatten):
    settings = load_settings(env_file=None)
    when = datetime(2026, 9, 29, hour, tzinfo=SHANGHAI)
    frame = candles_ending_at(when)
    flags = evaluate_time_filter(when, frame, settings)
    assert flags.flatten_only is expected_flatten
