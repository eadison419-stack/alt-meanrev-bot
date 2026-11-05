"""Donchian breakout long-only trend following — a new, independent strategy.

Pure decision module (no side effects, no I/O). It mirrors the validated
prototype `scripts/diagnose_cta_proto.py` exactly: enter long when a closed 4H
bar closes above the prior N-bar high, exit on a reverse M-bar low break or a
2x ATR protective stop. Long-only by design — shorts showed no edge in backtest.

The caller must pass closed bars only (see STRATEGY_BOUNDARIES.md).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakoutConfig:
    entry_bars: int = 20
    exit_bars: int = 10
    atr_len: int = 14
    atr_mult: float = 2.0
    be_mult: float | None = None  # breakeven stop after +be_mult*ATR(entry); None = off
    breadth_min: float | None = None  # min market breadth for new long entries; None = off


@dataclass(frozen=True)
class Indicators:
    atr: np.ndarray
    upper: np.ndarray       # entry long trigger (prior N-bar high, excludes current bar)
    lower: np.ndarray       # unused (long-only)
    exit_lower: np.ndarray  # long exit trigger (prior M-bar low, excludes current bar)


def indicators(frame: pd.DataFrame, cfg: BreakoutConfig) -> Indicators:
    o = frame["open"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    prev_c = np.empty(len(frame))
    prev_c[0] = o[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(cfg.atr_len).mean().to_numpy()
    upper = pd.Series(h).rolling(cfg.entry_bars).max().shift(1).to_numpy()
    lower = pd.Series(l).rolling(cfg.entry_bars).min().shift(1).to_numpy()
    exit_lower = pd.Series(l).rolling(cfg.exit_bars).min().shift(1).to_numpy()
    return Indicators(atr=atr, upper=upper, lower=lower, exit_lower=exit_lower)


def long_entry(frame: pd.DataFrame, cfg: BreakoutConfig) -> tuple[bool, float | None]:
    """Signal on the last closed bar. Returns (triggered, stop_price)."""
    ind = indicators(frame, cfg)
    i = len(frame) - 1
    close = float(frame["close"].iloc[-1])
    atr = ind.atr[i]
    if np.isfinite(ind.upper[i]) and np.isfinite(atr) and atr > 0 and close > ind.upper[i]:
        return True, close - cfg.atr_mult * atr
    return False, None


def long_exit(frame: pd.DataFrame, cfg: BreakoutConfig, stop: float) -> tuple[bool, str | None]:
    """Check the last closed bar for a long exit. Returns (triggered, reason)."""
    ind = indicators(frame, cfg)
    i = len(frame) - 1
    low = float(frame["low"].iloc[-1])
    close = float(frame["close"].iloc[-1])
    if np.isfinite(ind.atr[i]) and low <= stop:
        return True, "stop"
    if np.isfinite(ind.exit_lower[i]) and close < ind.exit_lower[i]:
        return True, "reverse"
    return False, None


def position_size(equity_per_slot: float, risk_pct: float, atr_mult: float, atr: float) -> float:
    """Quantity so that a stop-out risks risk_pct of the slot's capital."""
    if atr <= 0 or not np.isfinite(atr):
        return 0.0
    return (equity_per_slot * risk_pct) / (atr_mult * atr)


def breakeven_level(entry_px: float, entry_atr: float, high_since_entry: float,
                    be_mult: float | None) -> float | None:
    """Breakeven stop level for a long once price has moved be_mult*ATR in its favour.

    Returns ``entry_px`` when the running highest high since entry has reached
    ``entry_px + be_mult * entry_atr``, otherwise ``None`` (not yet reached).
    Always ``None`` when ``be_mult`` is ``None`` (feature disabled) or
    ``entry_atr`` is not a positive finite number. The caller applies the result
    by raising the protective stop; the level is monotonic because the caller
    keeps ``high_since_entry`` non-decreasing.
    """
    if be_mult is None:
        return None
    if entry_atr is None or not math.isfinite(entry_atr) or entry_atr <= 0:
        return None
    if high_since_entry >= entry_px + be_mult * entry_atr:
        return entry_px
    return None


def market_breadth(frames: Mapping[str, pd.DataFrame], ema_span: int = 50) -> float | None:
    """Fraction of symbols whose last closed bar close is above its own EMA(ema_span).

    A cross-sectional confirmation that the universe is broadly trending up
    (used as a risk overlay to gate new long entries). Symbols with too little
    history for a computable EMA are excluded. Returns ``None`` when no symbol
    yields a computable EMA (insufficient data across the whole universe).
    """
    above: list[bool] = []
    for frame in frames.values():
        if frame is None or frame.empty:
            continue
        close = frame["close"].astype(float)
        if len(close) < ema_span:
            continue
        ema = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
        last_close = float(close.iloc[-1])
        last_ema = float(ema.iloc[-1])
        if math.isfinite(last_ema) and last_ema > 0:
            above.append(last_close > last_ema)
    if not above:
        return None
    return sum(above) / len(above)


def breadth_ok(breadth: float | None, min_breadth: float | None) -> bool:
    """Whether a breadth reading clears the entry gate.

    ``min_breadth is None`` disables the gate (always ``True``). When the gate is
    enabled but ``breadth`` is ``None`` (cannot confirm the market is up), entries
    are blocked — the conservative choice for a risk overlay.
    """
    if min_breadth is None:
        return True
    if breadth is None:
        return False
    return breadth >= min_breadth
