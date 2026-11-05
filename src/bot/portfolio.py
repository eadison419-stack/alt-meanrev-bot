"""In-memory single-direction portfolio state; persistence belongs to Phase 6."""

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from zoneinfo import ZoneInfo
from typing import Protocol

from bot.signals import Side
from bot.strategy_router import MEAN_REVERSION, TREND


def classify_exit_category(reason: str | None, net_pnl: float | None = None) -> str | None:
    if not reason:
        return None
    normalized = reason.lower()
    if "liquidat" in normalized or "forced_close" in normalized:
        return "liquidation"
    if "stop" in normalized or "protective" in normalized:
        return "profit_exit" if net_pnl is not None and net_pnl > 0 else "stop_loss"
    if reason == "backtest_end":
        return "strategy_exit"
    return "strategy_exit"


class EntryApproval(Protocol):
    allowed: bool
    reason: str
    intent: object | None


@dataclass(frozen=True, init=False)
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    stop_price: float
    opened_at: datetime
    reason: str
    regime: str
    source: str = MEAN_REVERSION

    def __init__(self, symbol: str, side: Side, quantity: float, entry_price: float,
                 stop_price: float, opened_at: datetime, reason: str, regime: str,
                 source: str | None = None, *, strategy_source: str | None = None):
        if source is not None and strategy_source is not None and source != strategy_source:
            raise ValueError("position source aliases do not match")
        if source is None:
            source = strategy_source if strategy_source is not None else MEAN_REVERSION
        for name, value in (
            ("symbol", symbol), ("side", side), ("quantity", quantity), ("entry_price", entry_price),
            ("stop_price", stop_price), ("opened_at", opened_at), ("reason", reason),
            ("regime", regime), ("source", source),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def strategy_source(self) -> str:
        """Compatibility alias for persisted states and existing callers."""
        return self.source

    def __post_init__(self):
        object.__setattr__(self, "side", Side(self.side))
        if not self.symbol.strip():
            raise ValueError("position symbol is required")
        if not self.reason.strip() or not self.regime.strip():
            raise ValueError("position reason and regime are required")
        if self.source not in {MEAN_REVERSION, TREND}:
            raise ValueError("position source must be MEAN_REVERSION or TREND")
        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("position quantity must be finite and positive")
        if not math.isfinite(self.entry_price) or self.entry_price <= 0:
            raise ValueError("position entry price must be finite and positive")
        if self.stop_price is None or not math.isfinite(self.stop_price) or self.stop_price <= 0:
            raise ValueError("every position requires a finite positive stop price")
        if self.side is Side.LONG and self.stop_price >= self.entry_price:
            raise ValueError("long stop must be below entry price")
        if self.side is Side.SHORT and self.stop_price <= self.entry_price:
            raise ValueError("short stop must be above entry price")
        if self.opened_at.tzinfo is None:
            raise ValueError("position timestamp must be timezone-aware")

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class PortfolioState:
    positions: dict[str, Position] = field(default_factory=dict)
    last_exit_at: dict[str, datetime] = field(default_factory=dict)
    last_exit_category: dict[str, str] = field(default_factory=dict)
    last_exit_source: dict[str, str] = field(default_factory=dict)
    trading_date: date | None = None
    day_start_equity: float | None = None
    daily_halted: bool = False

    @property
    def gross_notional(self) -> float:
        return sum(position.notional for position in self.positions.values())

    def register_open(self, position: Position, approval: EntryApproval) -> None:
        if not approval.allowed or approval.reason != "approved" or approval.intent is None:
            raise ValueError("position can only be registered after an approved risk decision")
        approved_order = approval.intent
        matches_approved_order = (
            approved_order.symbol == position.symbol
            and approved_order.side is position.side
            and approved_order.quantity == position.quantity
            and approved_order.price == position.entry_price
            and approved_order.stop_price == position.stop_price
            and approved_order.reason == position.reason
            and approved_order.regime.value == position.regime
            and (approved_order.source is None or approved_order.source == position.source)
        )
        if not matches_approved_order:
            raise ValueError("position does not match its approved opening intent")
        if position.symbol in self.positions:
            raise ValueError(f"position already open for {position.symbol}")
        self.positions[position.symbol] = position

    def register_close(self, symbol: str, closed_at: datetime, *, reason: str | None = None,
                       net_pnl: float | None = None) -> Position:
        if closed_at.tzinfo is None:
            raise ValueError("close timestamp must be timezone-aware")
        position = self.positions.get(symbol)
        if position is None:
            raise ValueError(f"no open position for {symbol}")
        if closed_at < position.opened_at:
            raise ValueError("close timestamp precedes position open")
        del self.positions[symbol]
        self.last_exit_at[symbol] = closed_at
        self.last_exit_source[symbol] = position.source
        category = classify_exit_category(reason, net_pnl)
        if category is None:
            self.last_exit_category.pop(symbol, None)
        else:
            self.last_exit_category[symbol] = category
        return position

    def observe_equity(self, equity: float, now: datetime, settings) -> bool:
        """Latch a daily loss halt at the configured threshold until day rollover.

        Equity is supplied by the caller and should include realized and unrealized PnL.
        """
        if now.tzinfo is None:
            raise ValueError("equity observation timestamp must be timezone-aware")
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity must be finite and positive")
        local_date = now.astimezone(ZoneInfo(settings.timezone)).date()
        if self.trading_date is None:
            self.trading_date = local_date
            self.day_start_equity = equity
        elif local_date < self.trading_date:
            raise ValueError("equity observation date moved backwards")
        elif local_date > self.trading_date:
            self.trading_date = local_date
            self.day_start_equity = equity
            self.daily_halted = False
        if self.day_start_equity is None:
            raise RuntimeError("day start equity was not initialized")
        drawdown = (self.day_start_equity - equity) / self.day_start_equity
        if drawdown >= settings.risk.daily_drawdown_halt_pct:
            self.daily_halted = True
        return self.daily_halted
