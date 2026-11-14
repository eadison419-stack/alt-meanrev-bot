"""Causal squeeze measurements and directional breakout confirmation."""

import math
from datetime import timedelta

import pandas as pd

from bot.indicators import atr, validate_ohlcv
from bot.signals import Action, Side, Signal


def squeeze_context(frame):
    """Use SMA20 / population sigma for BB and EMA20 +/- 1.5 ATR14 for KC.

    On the first released bar, box statistics refer exclusively to the preceding
    squeeze. A run touching the indicator warmup is unknown, never truncated.
    """
    validate_ohlcv(frame)
    if (len(frame) < 25 or (frame.index.to_series().diff().dropna() != timedelta(hours=1)).any()
            or (frame[["open", "high", "low", "close"]] <= 0).any().any()):
        raise ValueError("incomplete squeeze history")
    close = frame["close"]
    mid = close.rolling(20, min_periods=20).mean()
    sigma = close.rolling(20, min_periods=20).std(ddof=0)
    kc_mid = close.ewm(span=20, adjust=False, min_periods=20).mean()
    width = 1.5 * atr(frame, 14)
    valid = mid.notna() & sigma.notna() & kc_mid.notna() & width.notna()
    squeezed = (mid + 2 * sigma < kc_mid + width) & (mid - 2 * sigma > kc_mid - width)
    end = len(frame) if squeezed.iloc[-1] else len(frame) - 1
    start = end
    while start and valid.iloc[start - 1] and squeezed.iloc[start - 1]:
        start -= 1
    complete = bool(start and valid.iloc[start - 1])
    box = frame.iloc[start:end]
    count = end - start
    result = {"squeeze_on": bool(squeezed.iloc[-1]), "sqz_bars": count,
              "squeeze_complete": complete, "squeeze_released": bool(not squeezed.iloc[-1] and count)}
    if count:
        volumes = pd.to_numeric(box["volume"], errors="coerce")
        volume = float(volumes.mean())
        if not all(math.isfinite(float(v)) and v >= 0 for v in volumes):
            raise ValueError("invalid squeeze volume")
        high, low = float(box["high"].max()), float(box["low"].min())
        result.update(box_high=high, box_low=low, box_mid=(high + low) / 2,
                      squeeze_volume_mean=volume)
    return result


def effective_entry_atr(frame, atr14):
    atr60 = float(atr(frame, 60).iloc[-1])
    if not all(math.isfinite(v) and v > 0 for v in (atr14, atr60)):
        raise ValueError("VSB ATR history unavailable")
    return max(atr14, 0.75 * atr60)


def breakout_signal(frame, side, squeeze, atr14):
    """Confirm only the first released close; the signal buffer uses raw ATR14."""
    def blocked(reason):
        return Signal(None, Action.NONE, reason, 0.0)

    if (not squeeze.get("squeeze_complete") or not squeeze.get("squeeze_released")
            or not 6 <= squeeze["sqz_bars"] <= 36):
        return blocked("vsb_release_unconfirmed")
    side = Side(side)
    bar = frame.iloc[-1]
    close, opening = float(bar["close"]), float(bar["open"])
    high, low = float(bar["high"]), float(bar["low"])
    ema = frame["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    direction = 1 if side is Side.LONG else -1
    if (direction * (close - opening) <= 0 or direction * (close - float(ema.iloc[-1])) <= 0
            or direction * (float(ema.iloc[-1]) - float(ema.iloc[-2])) <= 0):
        return blocked("vsb_ema9_direction_unconfirmed")
    edge = squeeze["box_high"] if side is Side.LONG else squeeze["box_low"]
    if not math.isfinite(atr14) or atr14 <= 0 or direction * (close - edge) <= 0.15 * atr14:
        return blocked("vsb_breakout_buffer_not_met")
    mean_volume, volume = squeeze["squeeze_volume_mean"], float(bar["volume"])
    if (not math.isfinite(volume) or mean_volume <= 0 or volume < 1.8 * mean_volume):
        return blocked("vsb_volume_unconfirmed")
    clv = ((close - low) if side is Side.LONG else (high - close)) / (high - low) if high > low else 0
    if clv < 0.70:
        return blocked("vsb_clv_unconfirmed")
    return Signal(side, Action.OPEN, f"vsb_breakout_{side.value}", 1.0)


def strategy_metadata(source):
    """Keep the frozen strategy and box alongside existing order recovery data."""
    return {"strategy_tag": source.get("strategy_tag", "MR"), **{
        key: source[key] for key in ("box_high", "box_low", "box_mid", "sqz_bars", "squeeze_volume_mean")
        if key in source}}
