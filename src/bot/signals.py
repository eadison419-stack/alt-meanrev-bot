"""Deterministic Mode A entry and exit signals shared by all run modes."""

import math
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

import pandas as pd

from bot.regime import Regime, RegimeSnapshot
from bot.time_filter import TimeFilterFlags


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Action(str, Enum):
    OPEN = "OPEN"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    NONE = "NONE"


@dataclass(frozen=True)
class Signal:
    side: Side | None
    action: Action
    reason: str
    strength: float
    next_side: Side | None = None
    close_first: bool = False
    size_multiplier: float = 1.0
    symbol: str | None = None
    source: str | None = None
    timestamp: datetime | None = None

    @property
    def direction(self) -> str | None:
        return self.side.value if self.side is not None else None


def with_signal_context(signal: Signal, *, symbol: str, source: str,
                        timestamp: datetime) -> Signal:
    """Attach auditable symbol, strategy source, and completed-signal time."""
    if not symbol.strip() or not source.strip():
        raise ValueError("signal symbol and source are required")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("signal timestamp must be timezone-aware")
    return replace(signal, symbol=symbol, source=source, timestamp=timestamp)


def entry_rejection_reason(
    side: Side | str,
    reference_price: float,
    atr_value: float,
    range_low: float | None,
    range_high: float | None,
    structure_low: float | None,
    structure_high: float | None,
    settings,
    *,
    require_structure: bool = False,
) -> str | None:
    """Reject entries that are already beyond their range or structure invalidation."""
    side = Side(side)
    if side is Side.LONG:
        policy = settings.long_entry
        boundary = range_low
        if boundary is not None and not math.isfinite(float(boundary)):
            boundary = None
        if policy.reject_if_below_range and boundary is not None:
            if reference_price < float(boundary) and float(boundary) - reference_price >= policy.max_below_range_atr * atr_value:
                return "long_rejected_below_range"
        structure = structure_low
        if structure is not None and not math.isfinite(float(structure)):
            structure = None
        if (structure is None and require_structure) or (
                structure is not None and float(structure) >= reference_price):
            return "invalid_on_entry"
    else:
        policy = settings.short_entry
        boundary = range_high
        if boundary is not None and not math.isfinite(float(boundary)):
            boundary = None
        if policy.reject_if_above_range and boundary is not None:
            if reference_price > float(boundary) and reference_price - float(boundary) >= policy.max_above_range_atr * atr_value:
                return "short_rejected_above_range"
        structure = structure_high
        if structure is not None and not math.isfinite(float(structure)):
            structure = None
        if (structure is None and require_structure) or (
                structure is not None and float(structure) <= reference_price):
            return "invalid_on_entry"
    return None


def generate_signal(
    regime: RegimeSnapshot,
    candle_1h: pd.Series,
    time_flags: TimeFilterFlags,
    settings,
    *,
    position: Side | str | None = None,
    stop_hit: bool = False,
    bars_1h: pd.DataFrame | None = None,
    bars_4h: pd.DataFrame | None = None,
    entry_reference_price: float | None = None,
    entry_atr: float | None = None,
    range_low: float | None = None,
    range_high: float | None = None,
    structure_low: float | None = None,
    structure_high: float | None = None,
) -> Signal:
    """Create one decision from a completed 4H regime and completed 1H candle."""
    if not {"open", "close"}.issubset(candle_1h.index):
        raise ValueError("1H candle must contain open and close")
    active_position = Side(position) if position is not None else None
    no_signal = Signal(None, Action.NONE, "no_entry_condition", 0.0)

    if active_position is Side.LONG:
        if stop_hit:
            return Signal(active_position, Action.CLOSE, "stop_hit", 1.0)
        if regime.regime is Regime.IMPULSE_UP:
            return Signal(active_position, Action.CLOSE, "4h_impulse_up", 1.0)
        if regime.regime is Regime.HIGH and regime.new_high:
            return Signal(active_position, Action.CLOSE, "high_zone_new_high", 1.0)
        if time_flags.morning_move == "UP":
            return Signal(active_position, Action.CLOSE, "morning_bullish_impulse", 1.0)
    elif active_position is Side.SHORT:
        if stop_hit:
            return Signal(active_position, Action.CLOSE, "stop_hit", 1.0)
        if regime.regime is Regime.IMPULSE_DOWN:
            return Signal(active_position, Action.CLOSE, "4h_impulse_down", 1.0)
        if regime.regime is Regime.LOW and regime.new_low:
            return Signal(active_position, Action.CLOSE, "low_zone_new_low", 1.0)
        if time_flags.morning_move == "DOWN":
            return Signal(active_position, Action.CLOSE, "morning_bearish_impulse", 1.0)

    if active_position is None and not time_flags.data_ready:
        return Signal(None, Action.NONE, "time_filter_data_unavailable", 0.0)

    bearish_1h = float(candle_1h["close"]) < float(candle_1h["open"])
    bullish_1h = float(candle_1h["close"]) > float(candle_1h["open"])
    long_zone_candidate = settings.signals.allow_long and regime.regime is Regime.LOW
    # Mean-reversion entries combine a completed 4H LOW/HIGH state with the
    # direction of the completed 1H candle; later filters stay below this gate.
    entry_long = settings.signals.allow_long and regime.regime is Regime.LOW and bullish_1h
    entry_short = settings.signals.allow_short and regime.regime is Regime.HIGH and bearish_1h
    short_blocked_higher_highs = False
    if entry_short and settings.short_entry.block_if_4h_higher_highs:
        count = settings.short_entry.higher_highs_lookback_4h
        highs = bars_4h["high"].tail(count).astype(float).tolist() if bars_4h is not None and "high" in bars_4h else []
        short_blocked_higher_highs = (
            len(highs) >= count
            and all(current > previous for previous, current in zip(highs, highs[1:]))
        )
    long_rejection = None
    if long_zone_candidate:
        if settings.long_entry.block_if_4h_lower_lows:
            count = settings.long_entry.lower_lows_lookback_4h
            lows = bars_4h["low"].tail(count).astype(float).tolist() if bars_4h is not None and "low" in bars_4h else []
            if len(lows) < count:
                long_rejection = "insufficient_4h_history"
            elif all(current < previous for previous, current in zip(lows, lows[1:])):
                long_rejection = "still_slow_down"
        if long_rejection is None and settings.long_entry.require_reversal_sign:
            count = settings.long_entry.reversal_lookback_1h
            history = bars_1h.tail(count) if bars_1h is not None else pd.DataFrame()
            lows = history["low"].astype(float).tolist() if "low" in history else []
            rising_lows = len(lows) >= count and all(current >= previous for previous, current in zip(lows, lows[1:]))
            candle_has_wick = {"low", "open", "close"}.issubset(candle_1h.index)
            lower_wick = (min(float(candle_1h["open"]), float(candle_1h["close"])) - float(candle_1h["low"])) if candle_has_wick else 0.0
            body = abs(float(candle_1h["close"]) - float(candle_1h["open"]))
            # Bullish close is already required by entry_long; reversal evidence
            # must independently come from rising lows or a sufficiently long wick.
            reversal_confirmed = rising_lows or (lower_wick > 0 and lower_wick >= body)
            if not reversal_confirmed:
                long_rejection = "no_reversal_sign"
    if active_position is not None:
        opposite = Side.SHORT if active_position is Side.LONG else Side.LONG
        opposite_setup = entry_short if opposite is Side.SHORT else entry_long
        opposite_allowed = time_flags.data_ready and (
            time_flags.allow_short_entry if opposite is Side.SHORT else time_flags.allow_long_entry
        )
        if opposite_setup:
            rejection = None
            if entry_reference_price is not None and entry_atr is not None:
                rejection = entry_rejection_reason(
                    opposite, entry_reference_price, entry_atr, range_low, range_high,
                    structure_low, structure_high, settings, require_structure=True,
                )
            if rejection is not None:
                return Signal(active_position, Action.CLOSE, "reverse_signal_close_first", 1.0)
            if settings.signals.reverse_same_bar and opposite_allowed and not time_flags.flatten_only:
                if opposite is Side.SHORT and short_blocked_higher_highs:
                    return Signal(active_position, Action.CLOSE, "reverse_signal_close_first", 1.0, next_side=opposite)
                return Signal(
                    opposite, Action.OPEN, "reverse_after_close", 1.0,
                    close_first=True, size_multiplier=settings.signals.reverse_size_mult,
                )
            return Signal(
                active_position, Action.CLOSE, "reverse_signal_close_first", 1.0,
                next_side=opposite if time_flags.data_ready else None,
            )
        return Signal(None, Action.NONE, "position_held_no_add", 0.0)
    if regime.regime is Regime.SIDEWAYS or time_flags.flatten_only:
        return Signal(None, Action.NONE, "entries_blocked", 0.0)
    if long_zone_candidate and long_rejection:
        return Signal(None, Action.NONE, long_rejection, 0.0)
    if entry_long and time_flags.allow_long_entry:
        if entry_reference_price is not None and entry_atr is not None:
            rejection = entry_rejection_reason(
                Side.LONG, entry_reference_price, entry_atr, range_low, range_high,
                structure_low, structure_high, settings, require_structure=True,
            )
            if rejection is not None:
                return Signal(None, Action.NONE, rejection, 0.0)
        return Signal(Side.LONG, Action.OPEN, "low_zone_1h_bullish_close", 1.0)
    if entry_short and time_flags.allow_short_entry:
        if short_blocked_higher_highs:
            return Signal(None, Action.NONE, "short_blocked_higher_highs", 0.0)
        if entry_reference_price is not None and entry_atr is not None:
            rejection = entry_rejection_reason(
                Side.SHORT, entry_reference_price, entry_atr, range_low, range_high,
                structure_low, structure_high, settings, require_structure=True,
            )
            if rejection is not None:
                return Signal(None, Action.NONE, rejection, 0.0)
        return Signal(Side.SHORT, Action.OPEN, "high_zone_1h_bearish_close", 1.0)

    blocked_long = entry_long and not time_flags.allow_long_entry
    blocked_short = entry_short and not time_flags.allow_short_entry
    if blocked_long or blocked_short:
        desired = Side.LONG if blocked_long else Side.SHORT
        return Signal(None, Action.NONE, f"{desired.value}_blocked_by_time_filter", 0.0)
    return no_signal
