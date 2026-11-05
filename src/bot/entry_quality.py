"""Closed-signal context and timestamped execution-quote validation."""
from dataclasses import dataclass
from datetime import datetime, timezone
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
