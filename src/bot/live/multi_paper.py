"""Shared multi-symbol paper account with reversible v1 state handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from bot.live.paper import PaperAccount
from bot.portfolio import PortfolioState, Position, classify_exit_category
from bot.signals import Signal

STATE_VERSION = 2
_SAVE_REPLACE_RETRIES = 6
_SAVE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _replace_state_file(source: Path, target: Path) -> None:
    for attempt in range(_SAVE_REPLACE_RETRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == _SAVE_REPLACE_RETRIES:
                raise
            time.sleep(_SAVE_RETRY_DELAYS[attempt])


def _aware(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("paper state timestamps must be timezone-aware")
    return result


def _position_data(position: Position) -> dict[str, Any]:
    return {**asdict(position), "side": position.side.value, "opened_at": position.opened_at.isoformat()}


@dataclass
class MultiPaperState:
    initial_equity: float = 10_000.0
    cash: float = 10_000.0
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    entry_costs: dict[str, dict[str, float]] = field(default_factory=dict)
    isolated_margin_by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    exit_management_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    mr_entry_scan: dict[str, Any] = field(default_factory=dict)
    mr_entry_attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_processed_by_symbol: dict[str, str] = field(default_factory=dict)
    last_observed_by_symbol: dict[str, str] = field(default_factory=dict)
    mark_prices: dict[str, float] = field(default_factory=dict)
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    symbol_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    market_risk_status: dict[str, Any] = field(default_factory=dict)
    entry_after: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    execution_mode: str = "local_paper"
    strategy_variant: str = "immediate_baseline"
    execution_status: str = "local"
    exchange_snapshot: dict[str, Any] = field(default_factory=dict)
    protective_stop_audit: dict[str, Any] = field(default_factory=dict)
    exchange_orders_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    inconsistencies: list[str] = field(default_factory=list)
    foreign_order_ids: list[str] = field(default_factory=list)
    equity_history: list[dict[str, Any]] = field(default_factory=list)
    _pending_save: tuple[Path, Path] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not math.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError("initial paper equity must be finite and positive")
        if not math.isfinite(self.cash):
            raise ValueError("paper cash must be finite")

    @property
    def current_equity(self) -> float:
        if self.execution_mode == "demo_usdm" and self.exchange_snapshot.get("equity") is not None:
            return float(self.exchange_snapshot["equity"])
        unrealized = 0.0
        for symbol, position in self.portfolio.positions.items():
            price = self.mark_prices.get(symbol, position.entry_price)
            direction = 1 if position.side.value == "long" else -1
            unrealized += position.quantity * (price - position.entry_price) * direction
        isolated = sum(float(item.get("initial_margin", 0.0)) for item in self.isolated_margin_by_symbol.values())
        return self.cash + isolated + unrealized

    def record_equity(self, at: datetime | None = None) -> None:
        """Append a mark-to-market equity snapshot (OOS forward-validation aid).

        Called once per cycle by the CTA paper runner so the live equity curve
        (return, max drawdown) can be reconstructed later and compared against
        the backtest baseline. Purely observational: never affects trading.

        Positions are marked to the last *closed* 4H bar, so equity only moves
        at 4H closes; consecutive snapshots with identical equity and position
        count are skipped to keep the state file small (~6 entries/day instead
        of ~288).
        """
        at = at or datetime.now(timezone.utc)
        eq = float(self.current_equity)
        n = len(self.portfolio.positions)
        if self.equity_history and self.equity_history[-1]["equity"] == eq \
                and self.equity_history[-1]["open_positions"] == n:
            return
        self.equity_history.append({
            "at": at.isoformat(),
            "equity": eq,
            "open_positions": n,
        })

    def for_symbol(self, symbol: str) -> "SymbolAccountView":
        return SymbolAccountView(self, symbol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "account_type": "multi_symbol_paper",
            "initial_equity": self.initial_equity,
            "cash": self.cash,
            "positions": {symbol: _position_data(position) for symbol, position in self.portfolio.positions.items()},
            "last_exit_at": {symbol: value.isoformat() for symbol, value in self.portfolio.last_exit_at.items()},
            "last_exit_category": self.portfolio.last_exit_category,
            "last_exit_source": self.portfolio.last_exit_source,
            "trading_date": self.portfolio.trading_date.isoformat() if self.portfolio.trading_date else None,
            "day_start_equity": self.portfolio.day_start_equity,
            "daily_halted": self.portfolio.daily_halted,
            "entry_costs": self.entry_costs,
            "isolated_margin_by_symbol": self.isolated_margin_by_symbol,
            "exit_management_by_symbol": self.exit_management_by_symbol,
            "pending_by_symbol": self.pending_by_symbol,
            "mr_entry_scan": self.mr_entry_scan,
            "mr_entry_attempts": self.mr_entry_attempts,
            "last_processed_by_symbol": self.last_processed_by_symbol,
            "last_observed_by_symbol": self.last_observed_by_symbol,
            "mark_prices": self.mark_prices,
            "watchlist": self.watchlist,
            "symbol_status": self.symbol_status,
            "market_risk_status": self.market_risk_status,
            "entry_after": self.entry_after,
            "events": self.events,
            "execution_mode": self.execution_mode,
            "strategy_variant": self.strategy_variant,
            "execution_status": self.execution_status,
            "exchange_snapshot": self.exchange_snapshot,
            "protective_stop_audit": self.protective_stop_audit,
            "exchange_orders_by_symbol": self.exchange_orders_by_symbol,
            "inconsistencies": self.inconsistencies,
            "foreign_order_ids": self.foreign_order_ids,
            "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiPaperState":
        if data.get("version") != STATE_VERSION or data.get("account_type") != "multi_symbol_paper":
            raise ValueError("unsupported multi-symbol paper state version")
        portfolio = PortfolioState()
        for symbol, raw in data.get("positions", {}).items():
            position_data = raw.copy()
            position_data["opened_at"] = _aware(position_data["opened_at"])
            portfolio.positions[symbol] = Position(**position_data)
        portfolio.last_exit_at = {symbol: _aware(value) for symbol, value in data.get("last_exit_at", {}).items()}
        portfolio.last_exit_category = data.get("last_exit_category", {}).copy()
        portfolio.last_exit_source = data.get("last_exit_source", {}).copy()
        for event in reversed(data.get("events", [])):
            symbol = event.get("symbol")
            if (symbol in portfolio.last_exit_at and symbol not in portfolio.last_exit_source
                    and event.get("event") == "paper_fill" and event.get("action") == "CLOSE"):
                source = event.get("entry_source")
                if source:
                    portfolio.last_exit_source[symbol] = source
            if (symbol in portfolio.last_exit_at and symbol not in portfolio.last_exit_category
                    and event.get("event") == "paper_fill" and event.get("action") == "CLOSE"):
                category = classify_exit_category(event.get("reason"), event.get("net_pnl"))
                if category:
                    portfolio.last_exit_category[symbol] = category
        portfolio.trading_date = datetime.fromisoformat(data["trading_date"]).date() if data.get("trading_date") else None
        portfolio.day_start_equity = data.get("day_start_equity")
        portfolio.daily_halted = bool(data.get("daily_halted", False))
        return cls(
            initial_equity=float(data["initial_equity"]), cash=float(data["cash"]), portfolio=portfolio,
            entry_costs=data.get("entry_costs", {}),
            isolated_margin_by_symbol=data.get("isolated_margin_by_symbol", {}),
            exit_management_by_symbol=data.get("exit_management_by_symbol", {}),
            pending_by_symbol=data.get("pending_by_symbol", {}),
            mr_entry_scan=data.get("mr_entry_scan", {}),
            mr_entry_attempts=data.get("mr_entry_attempts", {}),
            last_processed_by_symbol=data.get("last_processed_by_symbol", {}),
            last_observed_by_symbol=data.get("last_observed_by_symbol", {}),
            mark_prices=data.get("mark_prices", {}), watchlist=data.get("watchlist", []),
            symbol_status=data.get("symbol_status", {}), entry_after=data.get("entry_after", {}),
            market_risk_status=data.get("market_risk_status", {}),
            events=data.get("events", []),
            execution_mode=data.get("execution_mode", "local_paper"),
            strategy_variant=data.get("strategy_variant", "immediate_baseline"),
            execution_status=data.get("execution_status", "local"),
            exchange_snapshot=data.get("exchange_snapshot", {}),
            protective_stop_audit=data.get("protective_stop_audit", {}),
            exchange_orders_by_symbol=data.get("exchange_orders_by_symbol", {}),
            inconsistencies=data.get("inconsistencies", []),
            foreign_order_ids=data.get("foreign_order_ids", []),
            equity_history=data.get("equity_history", []),
        )

    @classmethod
    def load(cls, path: str | Path, *, initial_equity: float = 10_000.0,
             legacy_path: str | Path | None = None) -> "MultiPaperState":
        target = Path(path)
        if target.exists():
            return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        if legacy_path is not None and Path(legacy_path).exists():
            legacy_data = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
            old = PaperAccount.from_dict(legacy_data)
            migrated = cls(initial_equity=old.initial_equity, cash=old.cash, portfolio=old.portfolio,
                           entry_costs={old.symbol: {
                               "entry_fee": old.active_entry_fee,
                               "entry_slippage": old.active_entry_slippage,
                               "funding_paid": old.active_funding_paid,
                           }} if old.position else {},
                           pending_by_symbol={old.symbol: old.pending} if old.pending else {},
                           last_processed_by_symbol={old.symbol: old.last_processed_close} if old.last_processed_close else {},
                           last_observed_by_symbol={old.symbol: old.last_observed_open} if old.last_observed_open else {},
                           mark_prices=old.mark_prices or ({old.symbol: old.position.entry_price} if old.position else {}),
                           events=old.events)
            return migrated
        return cls(initial_equity=initial_equity, cash=initial_equity)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if self._pending_save and self._pending_save[0] == target and self._pending_save[1].exists():
            temporary = self._pending_save[1]
            handle = temporary.open("w", encoding="utf-8", newline="\n")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
            )
            temporary = Path(temporary_name)
            handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._pending_save = target, temporary
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_state_file(temporary, target)
        self._pending_save = None


class SymbolAccountView:
    """Adapt a shared portfolio to PaperTrader's existing symbol-level interface."""

    def __init__(self, state: MultiPaperState, symbol: str):
        self.state = state
        self.symbol = symbol

    @property
    def initial_equity(self):
        return self.state.initial_equity

    @property
    def cash(self):
        return self.state.cash

    @cash.setter
    def cash(self, value):
        self.state.cash = value

    @property
    def portfolio(self):
        return self.state.portfolio

    @property
    def events(self):
        return self.state.events

    @property
    def execution_status(self):
        return self.state.execution_status

    @execution_status.setter
    def execution_status(self, value):
        self.state.execution_status = str(value)

    @property
    def exchange_snapshot(self):
        return self.state.exchange_snapshot

    @exchange_snapshot.setter
    def exchange_snapshot(self, value):
        self.state.exchange_snapshot = value

    @property
    def exchange_orders_by_symbol(self):
        return self.state.exchange_orders_by_symbol

    @property
    def active_exit(self):
        return self.state.exit_management_by_symbol.get(self.symbol)

    @active_exit.setter
    def active_exit(self, value):
        if value is None:
            self.state.exit_management_by_symbol.pop(self.symbol, None)
        else:
            self.state.exit_management_by_symbol[self.symbol] = value

    @property
    def isolated_margin(self):
        return self.state.isolated_margin_by_symbol.get(self.symbol)

    @isolated_margin.setter
    def isolated_margin(self, value):
        if value is None:
            self.state.isolated_margin_by_symbol.pop(self.symbol, None)
        else:
            self.state.isolated_margin_by_symbol[self.symbol] = value

    @property
    def mark_prices(self):
        return self.state.mark_prices

    @property
    def position(self):
        return self.state.portfolio.positions.get(self.symbol)

    @position.setter
    def position(self, value):
        if value is None:
            self.state.portfolio.positions.pop(self.symbol, None)
        else:
            self.state.portfolio.positions[self.symbol] = value

    @property
    def active_entry_fee(self):
        return self.state.entry_costs.get(self.symbol, {}).get("entry_fee", 0.0)

    @active_entry_fee.setter
    def active_entry_fee(self, value):
        self._set_cost("entry_fee", value)

    @property
    def active_entry_slippage(self):
        return self.state.entry_costs.get(self.symbol, {}).get("entry_slippage", 0.0)

    @active_entry_slippage.setter
    def active_entry_slippage(self, value):
        self._set_cost("entry_slippage", value)

    @property
    def active_funding_paid(self):
        return self.state.entry_costs.get(self.symbol, {}).get("funding_paid", 0.0)

    @active_funding_paid.setter
    def active_funding_paid(self, value):
        self._set_cost("funding_paid", value)

    def _set_cost(self, field_name: str, value: float) -> None:
        costs = self.state.entry_costs.setdefault(self.symbol, {})
        costs[field_name] = float(value)
        if not any(costs.values()):
            self.state.entry_costs.pop(self.symbol, None)

    @property
    def pending(self):
        return self.state.pending_by_symbol.get(self.symbol)

    @pending.setter
    def pending(self, value):
        if value is None:
            self.state.pending_by_symbol.pop(self.symbol, None)
        else:
            self.state.pending_by_symbol[self.symbol] = value

    @property
    def last_processed_close(self):
        return self.state.last_processed_by_symbol.get(self.symbol)

    @last_processed_close.setter
    def last_processed_close(self, value):
        if value is None:
            self.state.last_processed_by_symbol.pop(self.symbol, None)
        else:
            self.state.last_processed_by_symbol[self.symbol] = value

    @property
    def last_observed_open(self):
        return self.state.last_observed_by_symbol.get(self.symbol)

    @last_observed_open.setter
    def last_observed_open(self, value):
        if value is None:
            self.state.last_observed_by_symbol.pop(self.symbol, None)
        else:
            self.state.last_observed_by_symbol[self.symbol] = value

    def current_equity(self) -> float:
        return self.state.current_equity

    def set_symbol_status(self, regime, signal: Signal, at: datetime, *, route: dict | None = None) -> None:
        self.state.symbol_status[self.symbol] = {
            "regime": regime.regime.value,
            "action": signal.action.value,
            "side": signal.side.value if signal.side else None,
            "reason": signal.reason,
            "at": at.isoformat(),
            "symbol": signal.symbol or self.symbol,
            "source": signal.source,
            "signal_timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
            "strategy_route": route,
        }
