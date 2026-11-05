"""Pure strategy functions for the ``mr_candidate`` mean-reversion paper strategy.

Mirrors ``run_portfolio_backtest(mr_candidate=True)``: hotness top-N universe,
EMA50-deviation entries (long below / short above), and the EMA20 Bollinger
mid-band protection plus the standard structure-stop / staged profit lock /
trailing / invalidation exits.

Everything here is a deterministic, side-effect-free function over closed bars so
it can be unit-tested and reused by a live paper/demo runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from bot.regime import Regime
from bot.signals import Action, Side, Signal


@dataclass
class MidExitState:
    """Bollinger mid-band protection state (mirrors backtest ``_Active`` fields)."""

    reached_mid: bool = False
    hold_count: int = 0
    fired: bool = False


def deviation_entry_signal(
    four_hour_close: pd.Series,
    one_hour_candle,
    settings,
    flags,
    *,
    deviation_pct: float,
) -> Signal:
    """EMA50-deviation mean-reversion signal over a candle-shaped observation.

    ``four_hour_close`` must contain only closed 4H closes. The 1H observation
    may be a completed candle or the current candle's open plus a live price.
    """
    return deviation_entry_signal_at_price(
        four_hour_close,
        float(one_hour_candle["open"]),
        float(one_hour_candle["close"]),
        settings,
        flags,
        deviation_pct=deviation_pct,
    )


def deviation_entry_signal_at_price(
    four_hour_close: pd.Series,
    one_hour_open: float,
    current_price: float,
    settings,
    flags,
    *,
    deviation_pct: float,
) -> Signal:
    """Check a live price against the latest closed 4H EMA50 and 1H open."""
    if not flags.data_ready:
        return Signal(None, Action.NONE, "time_filter_data_unavailable", 0.0)
    if flags.flatten_only:
        return Signal(None, Action.NONE, "entries_blocked", 0.0)
    span = settings.symbol_regime.ema_slow_bars
    ema = four_hour_close.ewm(span=span, adjust=False, min_periods=span).mean()
    if ema.empty or not math.isfinite(float(ema.iloc[-1])) or float(ema.iloc[-1]) <= 0:
        return Signal(None, Action.NONE, "ema_unavailable", 0.0)
    return deviation_entry_signal_at_ema(
        float(ema.iloc[-1]), one_hour_open, current_price, settings, flags,
        deviation_pct=deviation_pct,
    )


def deviation_entry_signal_at_ema(
    closed_ema: float, one_hour_open: float, current_price: float, settings, flags,
    *, deviation_pct: float,
) -> Signal:
    """Recheck a price using the original closed-bar EMA and hourly open."""
    if not flags.data_ready:
        return Signal(None, Action.NONE, "time_filter_data_unavailable", 0.0)
    if flags.flatten_only:
        return Signal(None, Action.NONE, "entries_blocked", 0.0)
    try:
        closed_ema = float(closed_ema)
        close = float(current_price)
        open_price = float(one_hour_open)
    except (TypeError, ValueError, OverflowError):
        return Signal(None, Action.NONE, "invalid_live_price", 0.0)
    if not math.isfinite(closed_ema) or closed_ema <= 0:
        return Signal(None, Action.NONE, "ema_unavailable", 0.0)
    if (not math.isfinite(close) or not math.isfinite(open_price)
            or close <= 0 or open_price <= 0):
        return Signal(None, Action.NONE, "invalid_live_price", 0.0)
    deviation = close / closed_ema - 1.0
    bullish = close > open_price
    bearish = close < open_price
    if settings.signals.allow_long and deviation <= -deviation_pct and bullish and flags.allow_long_entry:
        return Signal(Side.LONG, Action.OPEN, "range_deviation_long", 1.0)
    if settings.signals.allow_short and deviation >= deviation_pct and bearish and flags.allow_short_entry:
        return Signal(Side.SHORT, Action.OPEN, "range_deviation_short", 1.0)
    return Signal(None, Action.NONE, "no_range_deviation", 0.0)


def claim_hourly_entry(attempts: dict[str, str], symbol: str, hour_key: str) -> bool:
    """Persist one qualifying entry attempt per symbol and 1H candle."""
    if attempts.get(symbol) == hour_key:
        return False
    attempts[symbol] = hour_key
    return True


def mid_band(one_hour_close: pd.Series, period: int, ema: bool) -> pd.Series:
    """1H Bollinger mid band (EMA or SMA)."""
    if ema:
        return one_hour_close.ewm(span=period, adjust=False, min_periods=period).mean()
    return one_hour_close.rolling(period, min_periods=period).mean()


def mid_exit_check(
    side: Side,
    entry_price: float,
    mid: float,
    current_close: float,
    *,
    buffer: float = 0.0,
    hold_bars: int = 0,
    state: MidExitState,
) -> tuple[bool, MidExitState]:
    """Advance the mid-band protection state machine; returns (fire_full_close, state)."""
    if not math.isfinite(mid):
        return False, state
    if side is Side.LONG:
        if not state.reached_mid:
            if entry_price < mid and current_close >= mid:
                state.hold_count += 1
                if state.hold_count >= hold_bars + 1:
                    state.reached_mid = True
            else:
                state.hold_count = 0
        elif not state.fired and current_close < mid - buffer:
            state.fired = True
            return True, state
    else:
        if not state.reached_mid:
            if entry_price > mid and current_close <= mid:
                state.hold_count += 1
                if state.hold_count >= hold_bars + 1:
                    state.reached_mid = True
            else:
                state.hold_count = 0
        elif not state.fired and current_close > mid + buffer:
            state.fired = True
            return True, state
    return False, state


def breakeven_stop(
    side: Side,
    entry_price: float,
    entry_atr: float,
    favorable: float,
    current_stop: float,
    *,
    move_to_breakeven_atr: float,
    fee_rate: float,
    slip_rate: float,
) -> float:
    """Raise (long) / lower (short) the stop to breakeven once favourable >= N ATR."""
    if favorable < move_to_breakeven_atr * entry_atr:
        return current_stop
    if side is Side.LONG:
        adjusted = entry_price * (1 + fee_rate) / ((1 - slip_rate) * (1 - fee_rate))
        return max(current_stop, adjusted)
    adjusted = entry_price * (1 - fee_rate) / ((1 + slip_rate) * (1 + fee_rate))
    return min(current_stop, adjusted)


def staged_profit_lock_stop(
    side: Side,
    entry_price: float,
    entry_atr: float,
    initial_risk: float,
    favorable: float,
    current_stop: float,
    *,
    first_trigger_atr: float,
    first_lock_r: float,
    second_trigger_r: float,
    second_lock_r: float,
    fee_rate: float,
    slip_rate: float,
) -> float:
    """Protect +R floors after configured ATR and R-based triggers."""
    if (not all(math.isfinite(value) for value in (
            entry_price, entry_atr, initial_risk, favorable, current_stop,
            first_trigger_atr, first_lock_r, second_trigger_r, second_lock_r,
            fee_rate, slip_rate))
            or entry_price <= 0 or entry_atr <= 0 or initial_risk <= 0
            or favorable < 0 or first_trigger_atr <= 0
            or first_lock_r <= 0 or second_trigger_r <= 0
            or second_lock_r <= first_lock_r or fee_rate < 0
            or slip_rate < 0 or slip_rate >= 1):
        return current_stop

    target_lock_r = 0.0
    if favorable >= first_trigger_atr * entry_atr:
        target_lock_r = first_lock_r
    if favorable >= second_trigger_r * initial_risk:
        target_lock_r = second_lock_r
    if target_lock_r <= 0:
        return current_stop

    side = Side(side)
    locked_profit = target_lock_r * initial_risk
    if side is Side.LONG:
        denominator = (1 - slip_rate) * (1 - fee_rate)
        if denominator <= 0:
            return current_stop
        target_stop = (entry_price * (1 + fee_rate) + locked_profit) / denominator
        return max(current_stop, target_stop) if math.isfinite(target_stop) else current_stop

    denominator = (1 + slip_rate) * (1 + fee_rate)
    target_stop = (entry_price * (1 - fee_rate) - locked_profit) / denominator
    if not math.isfinite(target_stop) or target_stop <= 0:
        return current_stop
    return min(current_stop, target_stop)


def trailing_stop(
    side: Side,
    best_price: float,
    initial_risk: float,
    progress_r: float,
    current_stop: float,
    *,
    trigger_r: float,
    atr_mult: float,
    stop_atr_mult: float,
) -> float:
    """Trail the stop once progress_r passes the trigger (mirrors backtest formula)."""
    if progress_r < trigger_r:
        return current_stop
    distance = atr_mult * initial_risk / stop_atr_mult
    trail = best_price - distance if side is Side.LONG else best_price + distance
    return max(current_stop, trail) if side is Side.LONG else min(current_stop, trail)


def structure_invalidation_price(
    side: Side,
    structure_low: float | None,
    structure_high: float | None,
    entry_atr: float,
    *,
    buffer_atr: float,
) -> float | None:
    """Return the buffered entry-structure invalidation price, if available."""
    try:
        entry_atr = float(entry_atr)
        buffer_atr = float(buffer_atr)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(entry_atr) or entry_atr <= 0:
        return None
    if not math.isfinite(buffer_atr) or buffer_atr < 0:
        return None

    side = Side(side)
    structure = structure_low if side is Side.LONG else structure_high
    if structure is None:
        return None
    try:
        structure = float(structure)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(structure) or structure <= 0:
        return None

    buffer = buffer_atr * entry_atr
    return structure - buffer if side is Side.LONG else structure + buffer


def structure_invalid_quote_exit(
    side: Side,
    current_price: float,
    structure_low: float | None,
    structure_high: float | None,
    entry_atr: float,
    protective_stop: float,
    *,
    buffer_atr: float,
) -> tuple[str, float] | None:
    """Return an intrabar structure exit only when its level precedes the hard stop."""
    invalid_price = structure_invalidation_price(
        side, structure_low, structure_high, entry_atr, buffer_atr=buffer_atr,
    )
    if invalid_price is None:
        return None
    try:
        current_price = float(current_price)
        protective_stop = float(protective_stop)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(current_price) or current_price <= 0
            or not math.isfinite(protective_stop) or protective_stop <= 0):
        return None

    side = Side(side)
    if side is Side.LONG:
        if invalid_price > protective_stop and current_price < invalid_price:
            return "long_structure_invalid", invalid_price
    elif invalid_price < protective_stop and current_price > invalid_price:
        return "short_structure_invalid", invalid_price
    return None


def structure_invalid_triggered(
    side: Side,
    current_close: float,
    structure_low: float | None,
    structure_high: float | None,
    entry_atr: float,
    *,
    buffer_atr: float,
) -> bool:
    """Close when price breaks beyond the entry structure level plus a buffer."""
    invalid_price = structure_invalidation_price(
        side, structure_low, structure_high, entry_atr, buffer_atr=buffer_atr,
    )
    if invalid_price is None:
        return False
    if side is Side.LONG:
        return current_close < invalid_price
    return current_close > invalid_price


def structure_reentry_confirmed(
    side: Side,
    previous_high: float,
    previous_low: float,
    current_close: float,
) -> bool:
    """Confirm a one-hour structure reclaim after repeated invalidation losses."""
    try:
        previous_high = float(previous_high)
        previous_low = float(previous_low)
        current_close = float(current_close)
    except (TypeError, ValueError, OverflowError):
        return False
    if (not all(math.isfinite(value) and value > 0
                for value in (previous_high, previous_low, current_close))
            or previous_low > previous_high):
        return False
    side = Side(side)
    if side is Side.LONG:
        return current_close > previous_high
    return current_close < previous_low


def regime_faded_triggered(
    side: Side,
    position_regime: str,
    current_regime,
    progress: float,
    entry_atr: float,
    *,
    min_atr: float,
) -> bool:
    """Exit when a LOW/HIGH entry lapses to SIDEWAYS with little progress."""
    initial = Regime.LOW if side is Side.LONG else Regime.HIGH
    return (
        position_regime == initial.value
        and current_regime is Regime.SIDEWAYS
        and progress < min_atr * entry_atr
    )


def no_progress_update(
    side: Side,
    current_close: float,
    last_close: float,
    no_progress_count: int,
    favorable: float,
    entry_atr: float,
    *,
    bars: int,
    min_atr: float,
) -> tuple[bool, int]:
    """Advance the no-progress counter; returns (trigger_close, new_count)."""
    moved = current_close > last_close if side is Side.LONG else current_close < last_close
    count = 0 if moved else no_progress_count + 1
    trigger = count >= bars and favorable < min_atr * entry_atr
    return trigger, count
