"""Shanghai-session entry filters, evaluated at the 1H candle close."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bot.indicators import atr, validate_ohlcv


@dataclass(frozen=True)
class TimeFilterFlags:
    allow_long_entry: bool
    allow_short_entry: bool
    flatten_only: bool
    watch_next_morning_long: bool
    morning_move: Literal["UP", "DOWN"] | None = None
    afternoon_dump_today: bool = False
    at_upper_extreme: bool = False
    at_lower_extreme: bool = False
    recent_move_atr: float | None = None
    data_ready: bool = True
    data_reason: str | None = None


def _unavailable_flags(flatten_only: bool, reason: str) -> TimeFilterFlags:
    return TimeFilterFlags(
        False, False, flatten_only, False, data_ready=False, data_reason=reason,
    )


def _local_close_times(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    return (index + pd.Timedelta(1, unit="h")).tz_convert(timezone)


def evaluate_time_filter(
    bar_close_time: datetime,
    ohlcv_1h: pd.DataFrame,
    settings,
    *,
    precomputed_atr: pd.Series | None = None,
) -> TimeFilterFlags:
    """Filter new entries using only completed 1H bars available at bar_close_time.

    Session intervals are half-open: morning [08:00, 12:00), afternoon [12:00, 20:00),
    night [20:00, 08:00). Morning/afternoon spikes compare the close-to-close move
    against 1.5 x 1H ATR(14). The afternoon day-long long-entry block begins at the
    first such dump; short chasing is blocked at the lower 10% of the prior 24-bar range.
    """
    if bar_close_time.tzinfo is None:
        raise ValueError("bar_close_time must be timezone-aware")
    local_time = bar_close_time.astimezone(ZoneInfo(settings.timezone))
    wall_time = local_time.timetz().replace(tzinfo=None)
    session = settings.session
    in_morning = session.morning_start <= wall_time < session.morning_end
    in_afternoon = session.afternoon_start <= wall_time < session.afternoon_end
    in_night = not (in_morning or in_afternoon)
    flatten_only = in_night and not session.night_new_entries
    try:
        validate_ohlcv(ohlcv_1h)
    except (TypeError, ValueError):
        return _unavailable_flags(flatten_only, "invalid_ohlcv")
    required_bars = max(
        settings.lookback.atr_len + 1,
        settings.lookback.afternoon_extreme_lookback_1h + 1,
    )
    if len(ohlcv_1h) < required_bars:
        return _unavailable_flags(flatten_only, "hourly_history_insufficient")

    close_times = _local_close_times(ohlcv_1h.index, settings.timezone)
    available = ohlcv_1h.loc[close_times <= pd.Timestamp(bar_close_time).tz_convert(settings.timezone)]
    close_times = _local_close_times(available.index, settings.timezone)
    if len(available) < required_bars:
        return _unavailable_flags(flatten_only, "hourly_history_insufficient")
    if close_times[-1] != pd.Timestamp(bar_close_time).tz_convert(settings.timezone):
        return _unavailable_flags(flatten_only, "latest_bar_close_mismatch")
    expected_steps = pd.Timedelta(1, unit="h")
    gaps = available.index.to_series().diff().ne(expected_steps).to_numpy()
    if gaps[1:].any():
        # Reset the ATR after a missing candle. Require a complete, contiguous
        # suffix long enough for both ATR and the prior-range extreme checks.
        last_gap = int(np.flatnonzero(gaps)[-1])
        available = available.iloc[last_gap:]
        close_times = _local_close_times(available.index, settings.timezone)
        precomputed_atr = None
        if len(available) < required_bars:
            return _unavailable_flags(flatten_only, "hourly_history_gap")

    atr_values = atr(available, settings.lookback.atr_len) if precomputed_atr is None else precomputed_atr.reindex(available.index)
    current_atr = float(atr_values.iloc[-1])
    current_change = float(available["close"].iloc[-1] - available["close"].iloc[-2])
    if not np.isfinite(current_atr) or current_atr <= 0:
        return _unavailable_flags(flatten_only, "atr_unavailable")
    recent_move_atr = abs(current_change) / current_atr
    large_move = recent_move_atr >= settings.signals.impulse_atr_mult
    current_close_time = close_times[-1]
    current_local_time = current_close_time.timetz().replace(tzinfo=None)
    current_date = current_close_time.date()
    morning_move: Literal["UP", "DOWN"] | None = None
    if large_move and settings.session.morning_start <= current_local_time < settings.session.morning_end:
        morning_move = "UP" if current_change > 0 else "DOWN"

    extreme_lookback = settings.lookback.afternoon_extreme_lookback_1h
    at_upper_extreme = False
    at_lower_extreme = False
    if len(available) > extreme_lookback:
        prior = available.iloc[-extreme_lookback - 1 : -1]
        prior_high, prior_low = float(prior["high"].max()), float(prior["low"].min())
        width = prior_high - prior_low
        if width > 0:
            close = float(available["close"].iloc[-1])
            at_upper_extreme = close >= prior_low + (1 - session.afternoon_extreme_pct) * width
            at_lower_extreme = close <= prior_low + session.afternoon_extreme_pct * width

    local_close_dates = [timestamp.date() for timestamp in close_times]
    afternoon_dump_dates: set[date] = set()
    for i, (close_time, change, atr_value) in enumerate(zip(close_times, available["close"].diff(), atr_values)):
        wall = close_time.timetz().replace(tzinfo=None)
        if settings.session.afternoon_start <= wall < settings.session.afternoon_end and np.isfinite(atr_value) and atr_value > 0:
            if change <= -settings.signals.impulse_atr_mult * atr_value:
                afternoon_dump_dates.add(local_close_dates[i])
    afternoon_dump_today = current_date in afternoon_dump_dates
    previous_date = current_date - timedelta(days=1)
    watch_next_morning_long = in_morning and previous_date in afternoon_dump_dates

    allow_long = not flatten_only
    allow_short = not flatten_only
    if morning_move == "UP":
        allow_long = False
    elif morning_move == "DOWN":
        allow_short = False
    if in_afternoon:
        if large_move and current_change > 0:
            allow_long = False
        if afternoon_dump_today:
            allow_long = False
            if at_lower_extreme:
                allow_short = False
        if at_upper_extreme:
            allow_short = False
    return TimeFilterFlags(
        allow_long_entry=allow_long,
        allow_short_entry=allow_short,
        flatten_only=flatten_only,
        watch_next_morning_long=watch_next_morning_long,
        morning_move=morning_move,
        afternoon_dump_today=afternoon_dump_today,
        at_upper_extreme=at_upper_extreme,
        at_lower_extreme=at_lower_extreme,
        recent_move_atr=recent_move_atr,
    )


def precompute_time_filters(ohlcv_1h: pd.DataFrame, settings) -> dict[pd.Timestamp, TimeFilterFlags]:
    """Compute the same causal per-close flags once for reuse across backtest modes."""
    validate_ohlcv(ohlcv_1h)
    full_atr = atr(ohlcv_1h, settings.lookback.atr_len)
    result: dict[pd.Timestamp, TimeFilterFlags] = {}
    for index, open_time in enumerate(ohlcv_1h.index):
        close_time = open_time + pd.Timedelta(1, unit="h")
        history_start = max(0, index - 47)
        history = ohlcv_1h.iloc[history_start : index + 1]
        result[close_time] = evaluate_time_filter(
            close_time.to_pydatetime(),
            history,
            settings,
            precomputed_atr=full_atr.reindex(history.index),
        )
    return result
