"""Causal OHLCV indicators using Wilder smoothing and rolling windows."""

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


def validate_ohlcv(frame: pd.DataFrame) -> None:
    missing = set(OHLC).difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLCV index must contain timezone-aware datetimes")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV timestamps must be strictly increasing and unique")
    values = frame.loc[:, OHLC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLCV prices must be finite")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be greater than or equal to open, close, and low")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be less than or equal to open, close, and high")


def true_range(frame: pd.DataFrame) -> pd.Series:
    validate_ohlcv(frame)
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True).rename("true_range")


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Average true range with Wilder's recursive moving average."""
    return wilder_smooth(true_range(frame), period).rename("atr")


def indicator_frame(frame: pd.DataFrame, atr_period: int, adx_period: int) -> pd.DataFrame:
    """Return ATR, +DI, -DI, and ADX at each bar without using future rows."""
    validate_ohlcv(frame)
    if atr_period <= 0 or adx_period <= 0:
        raise ValueError("indicator periods must be positive")
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = true_range(frame)
    average_tr = wilder_smooth(tr, atr_period)
    smoothed_tr = wilder_smooth(tr, adx_period)
    plus_di = 100 * wilder_smooth(plus_dm, adx_period).div(smoothed_tr.replace(0, np.nan))
    minus_di = 100 * wilder_smooth(minus_dm, adx_period).div(smoothed_tr.replace(0, np.nan))
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs().div(di_sum)
    dx = dx.where(smoothed_tr.isna(), dx.fillna(0.0))
    adx_value = wilder_smooth(dx, adx_period)
    return pd.DataFrame({"atr": average_tr, "plus_di": plus_di, "minus_di": minus_di, "adx": adx_value}, index=frame.index)


def percentile_rank_last(values: np.ndarray) -> float:
    """Mid-rank (ties averaged) of the final value in a complete rolling window."""
    if not np.isfinite(values).all():
        return float("nan")
    last = values[-1]
    less = np.count_nonzero(values < last)
    equal = np.count_nonzero(values == last)
    return 100.0 * (less + equal / 2) / len(values)
