"""Closed-signal context and timestamped execution-quote validation."""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, ROUND_FLOOR
import math

from bot.signals import Side


@dataclass(frozen=True)
class EntrySignalContext:
    symbol: str
    side: Side
    bar_close_at: datetime
    cycle_started_at: datetime
    evaluated_at: datetime
    signal_open: float
    signal_close: float
    closed_ema: float
    atr: float
    structure_low: float
    structure_high: float


def execution_quote(book, side, observed_at, *, max_age_sec):
    """Return the executable top-of-book side, or a fail-closed reason."""
    try:
        timestamp = float(book["timestamp"])
        bid, bid_quantity = map(float, book["bids"][0][:2])
        ask, ask_quantity = map(float, book["asks"][0][:2])
        values = (timestamp, bid, bid_quantity, ask, ask_quantity)
        if any(not math.isfinite(value) or value <= 0 for value in values) or bid >= ask:
            return None, "entry_quote_invalid"
        quoted_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        age = (observed_at - quoted_at).total_seconds()
    except (IndexError, KeyError, TypeError, ValueError, OverflowError, OSError):
        return None, "entry_quote_invalid"
    if age < 0:
        return None, "entry_quote_from_future"
    if age > max_age_sec:
        return None, "entry_quote_stale"
    is_long = Side(side) is Side.LONG
    return {
        "price": ask if is_long else bid,
        "quantity": ask_quantity if is_long else bid_quantity,
        "bid": bid, "ask": ask,
        "quoted_at": quoted_at.isoformat(),
        "observed_at": observed_at.isoformat(),
        "age_sec": age,
    }, None


def _depth_side(levels, planned, slippage_limit, direction):
    """Measure visible depth and simulate eating this side of the current book."""
    levels = [(Decimal(str(price)), Decimal(str(quantity))) for price, quantity, *_ in levels[:5]]
    if (not levels or any(not math.isfinite(value) or value <= 0 for level in levels for value in level)
            or any(direction * (after[0] - before[0]) <= 0 for before, after in zip(levels, levels[1:]))):
        raise ValueError("invalid depth levels")

    best = levels[0][0]
    limit_price = best * (1 + direction * slippage_limit)
    within_limit = sum(quantity for price, quantity in levels
                       if direction * (price - limit_price) <= 0)
    remaining, cost = planned, Decimal(0)
    for price, quantity in levels:
        take = min(remaining, quantity)
        cost += take * price
        remaining -= take
        if remaining == 0:
            break
    vwap = cost / planned if remaining == 0 else None
    return {
        "levels": len(levels),
        "visible_quantity": float(sum(quantity for _, quantity in levels)),
        "quantity_within_limit": float(within_limit),
        "planned_vwap": float(vwap) if vwap is not None else None,
        "planned_slippage_pct": float(direction * (vwap / best - 1)) if vwap is not None else None,
    }, best, within_limit


def entry_liquidity_observation(book, side, quantity, *, slippage_limit, step_size, minimum_quantity):
    """Estimate liquidity for a planned entry. The result is telemetry only."""
    observation = {"observe_only": True}
    try:
        planned, limit, step, minimum = map(
            lambda value: Decimal(str(value)), (quantity, slippage_limit, step_size, minimum_quantity))
        if not all(math.isfinite(value) for value in (planned, limit, step, minimum)):
            raise ValueError("invalid sizing input")
        if planned <= 0 or step <= 0 or not 0 <= limit < 1 or minimum < 0:
            raise ValueError("invalid sizing input")

        asks, ask, ask_depth = _depth_side(book["asks"], planned, limit, 1)
        bids, bid, bid_depth = _depth_side(book["bids"], planned, limit, -1)
        if bid >= ask:
            raise ValueError("crossed depth")
        entry, reverse = (asks, bids) if Side(side) is Side.LONG else (bids, asks)
        # Compare a conservative size that leaves half the visible depth available.
        depth_fraction = Decimal("0.5")
        supported = min(planned, min(ask_depth, bid_depth) * depth_fraction)
        suggested = (supported / step).to_integral_value(rounding=ROUND_FLOOR) * step
        if suggested < minimum:
            suggested = Decimal(0)
        action = "skip" if suggested == 0 else "reduce" if suggested < planned else "keep"
        spread = (ask - bid) / ((ask + bid) / 2)
        if not all(math.isfinite(value) for depth in (entry, reverse) for value in depth.values() if value is not None):
            raise ValueError("unrepresentable depth result")
    except (DecimalException, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return {**observation, "status": "unavailable", "reason": "invalid_depth_or_sizing"}

    return {
        **observation,
        "status": "estimated",
        "spread_pct": float(spread),
        "slippage_limit_pct": float(limit),
        "visible_depth_fraction": float(depth_fraction),
        "planned_quantity": float(planned),
        "suggested_quantity": float(suggested),
        "suggested_action": action,
        "entry": entry,
        "reverse": reverse,
    }
