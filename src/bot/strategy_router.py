"""Explicit per-symbol strategy selection and entry conflict checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bot.signals import Action, Signal


MEAN_REVERSION = "MEAN_REVERSION"
TREND = "TREND"
ROUTER = "ROUTER"


@dataclass(frozen=True)
class StrategyRoute:
    symbol: str
    regime_state: str
    market_risk_state: str
    source: str | None
    position_source: str | None
    route_reason: str
    entry_allowed: bool
    entry_block_reason: str | None
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "regime_state": self.regime_state,
            "market_risk_state": self.market_risk_state,
            "source": self.source,
            "position_source": self.position_source,
            "route_reason": self.route_reason,
            "entry_allowed": self.entry_allowed,
            "entry_block_reason": self.entry_block_reason,
            "timestamp": self.timestamp.isoformat(),
        }


def resolve_strategy_route(symbol: str, regime_state: str | None, market_risk_state: str,
                           *, position_source: str | None, at: datetime,
                           trend_strategy_ready: bool = False) -> StrategyRoute:
    """Resolve an entry route; callers opt in only when a trend evaluator is present."""
    if not symbol.strip() or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("strategy route requires a symbol and timezone-aware timestamp")
    state = (regime_state or "UNCERTAIN").upper()
    market_state = (market_risk_state or "PAUSE_NEW").upper()
    source = MEAN_REVERSION if state == "RANGE" else TREND if state == "TREND" else None

    if state not in {"TREND", "RANGE", "UNCERTAIN"}:
        state, source = "UNCERTAIN", None
    if source is None:
        route_reason = "symbol_regime_uncertain"
    elif source == TREND:
        route_reason = "trend_strategy_selected" if trend_strategy_ready else "trend_strategy_placeholder"
    else:
        route_reason = "mean_reversion_selected"

    if market_state not in {"NORMAL", "RISK_ELEVATED", "PAUSE_NEW"}:
        block_reason = "market_risk_unavailable"
    elif market_state == "PAUSE_NEW":
        block_reason = "market_risk_paused"
    elif source is None:
        block_reason = "symbol_regime_uncertain"
    elif position_source is not None and position_source != source:
        block_reason = "position_owned_by_other_strategy"
    elif source == TREND and not trend_strategy_ready:
        block_reason = "trend_strategy_not_implemented"
    else:
        block_reason = None

    return StrategyRoute(
        symbol=symbol,
        regime_state=state,
        market_risk_state=market_state,
        source=source,
        position_source=position_source,
        route_reason=route_reason,
        entry_allowed=block_reason is None,
        entry_block_reason=block_reason,
        timestamp=at,
    )


def apply_entry_route(signal: Signal, route: StrategyRoute) -> Signal:
    """Preserve non-entry decisions and reject OPEN signals outside the selected route."""
    if signal.action is not Action.OPEN:
        return signal
    reason = route.entry_block_reason
    if reason is None and signal.symbol != route.symbol:
        reason = "signal_symbol_mismatch"
    if reason is None and signal.timestamp is None:
        reason = "signal_timestamp_missing"
    if reason is None and signal.source != route.source:
        reason = "strategy_route_mismatch"
    if reason is None:
        return signal
    return Signal(
        None, Action.NONE, reason, 0.0,
        symbol=route.symbol, source=ROUTER, timestamp=signal.timestamp or route.timestamp,
    )
