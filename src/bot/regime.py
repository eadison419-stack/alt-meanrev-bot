"""Causal 4H regime classification for the Mode A mean-reversion strategy."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from bot.indicators import indicator_frame, percentile_rank_last, validate_ohlcv


class Regime(str, Enum):
    SIDEWAYS = "SIDEWAYS"
    LOW = "LOW"
    HIGH = "HIGH"
    MID = "MID"
    IMPULSE_UP = "IMPULSE_UP"
    IMPULSE_DOWN = "IMPULSE_DOWN"
    SLOW_DOWN = "SLOW_DOWN"
    SLOW_UP = "SLOW_UP"


@dataclass(frozen=True)
class RegimeSnapshot:
    timestamp: pd.Timestamp
    regime: Regime
    atr: float
    adx: float
    atr_percentile_rank: float
    range_position: float
    new_low: bool
    new_high: bool


def classify_regimes(ohlcv: pd.DataFrame, settings) -> pd.DataFrame:
    """Classify each row using only that completed bar and earlier rows.

    Input rows must represent completed bars. Range levels use the prior N bars,
    excluding the row being classified. Early rows remain unclassified (None).
    """
    validate_ohlcv(ohlcv)
    lookback = settings.lookback
    regime_cfg = settings.regime
    features = indicator_frame(ohlcv, lookback.atr_len, lookback.adx_len)
    features["atr_pct"] = 100 * features["atr"] / ohlcv["close"].replace(0, np.nan)
    features["atr_percentile_rank"] = features["atr_pct"].rolling(
        regime_cfg.atrpct_lookback, min_periods=regime_cfg.atrpct_lookback
    ).apply(percentile_rank_last, raw=True)
    features["range_high"] = ohlcv["high"].shift(1).rolling(
        lookback.range_bars_4h, min_periods=lookback.range_bars_4h
    ).max()
    features["range_low"] = ohlcv["low"].shift(1).rolling(
        lookback.range_bars_4h, min_periods=lookback.range_bars_4h
    ).min()
    width = (features["range_high"] - features["range_low"]).replace(0, np.nan)
    features["range_position"] = (ohlcv["close"] - features["range_low"]) / width
    features["new_low"] = ohlcv["low"] < features["range_low"]
    features["new_high"] = ohlcv["high"] > features["range_high"]
    features["net_move"] = ohlcv["close"] - ohlcv["close"].shift(regime_cfg.slow_lookback_4h)
    body = ohlcv["close"] - ohlcv["open"]

    labels: list[Regime | None] = []
    for timestamp, row in features.iterrows():
        ready = all(np.isfinite(row[key]) for key in (
            "atr", "adx", "atr_percentile_rank", "range_high", "range_low", "range_position", "net_move"
        ))
        if not ready:
            labels.append(None)
            continue
        # Preserve the strategy's precedence: SIDEWAYS, IMPULSE, LOW/HIGH, SLOW, MID.
        is_sideways = row["adx"] < regime_cfg.adx_sideways and row["atr_percentile_rank"] <= regime_cfg.atrpct_percentile
        if is_sideways:
            labels.append(Regime.SIDEWAYS)
            continue
        candle_body = body.loc[timestamp]
        if candle_body >= settings.signals.impulse_atr_mult * row["atr"]:
            labels.append(Regime.IMPULSE_UP)
            continue
        if candle_body <= -settings.signals.impulse_atr_mult * row["atr"]:
            labels.append(Regime.IMPULSE_DOWN)
            continue
        if row["range_position"] <= regime_cfg.low_pct or row["new_low"]:
            labels.append(Regime.LOW)
            continue
        if row["range_position"] >= regime_cfg.high_pct or row["new_high"]:
            labels.append(Regime.HIGH)
            continue
        if row["adx"] >= regime_cfg.adx_sideways:
            threshold = regime_cfg.slow_move_atr_mult * row["atr"]
            if row["net_move"] <= -threshold:
                labels.append(Regime.SLOW_DOWN)
                continue
            if row["net_move"] >= threshold:
                labels.append(Regime.SLOW_UP)
                continue
        labels.append(Regime.MID)
    features["regime"] = pd.Series(labels, index=features.index, dtype=object)
    return features


def classify_regime(ohlcv: pd.DataFrame, settings) -> RegimeSnapshot:
    """Classify the latest completed 4H candle, failing closed on short history."""
    minimum = settings.lookback.min_regime_bars_4h
    if len(ohlcv) < minimum:
        raise ValueError(f"regime requires at least {minimum} completed 4H bars")
    features = classify_regimes(ohlcv, settings)
    row = features.iloc[-1]
    if row["regime"] is None:
        raise ValueError("not enough valid history to calculate regime indicators")
    return RegimeSnapshot(
        timestamp=features.index[-1],
        regime=row["regime"],
        atr=float(row["atr"]),
        adx=float(row["adx"]),
        atr_percentile_rank=float(row["atr_percentile_rank"]),
        range_position=float(row["range_position"]),
        new_low=bool(row["new_low"]),
        new_high=bool(row["new_high"]),
    )
