"""Local paper execution using public market data and the shared strategy/risk rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import replace
import copy
import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path
import traceback
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from bot.backtest.engine import _apply_slippage, _snapshot
from bot.exchange.binance_usdm import DEMO_REST_ROOT
from bot.indicators import atr, validate_ohlcv
from bot.portfolio import PortfolioState, Position, classify_exit_category
from bot.regime import Regime, RegimeSnapshot, classify_regimes
from bot.risk import (OrderIntent, assess_entry, calculate_adaptive_stop,
                      calculate_order_quantity, guard_signal)
from bot.signals import Action, Side, Signal, entry_rejection_reason, generate_signal, with_signal_context
from bot.strategy_router import (MEAN_REVERSION, ROUTER, TREND, apply_entry_route,
                                 resolve_strategy_route)
from bot.trend_strategy import (calculate_trend_quantity, evaluate_trend_entry,
                                evaluate_trend_exit, initialize_trend_exit_state)
from bot.time_filter import evaluate_time_filter

STATE_VERSION = 1
HOUR = pd.Timedelta(3_600_000_000_000, unit="ns")
PAPER_RUNNER_BUILD_ID = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
TIME_FILTER_BUILD_ID = hashlib.sha256(
    Path(evaluate_time_filter.__code__.co_filename).read_bytes(),
).hexdigest()[:12]


def _dt(value: str | datetime | pd.Timestamp) -> datetime:
    result = pd.Timestamp(value).to_pydatetime()
    if result.tzinfo is None:
        raise ValueError("paper timestamps must be timezone-aware")
    return result


def _intent_dict(intent: OrderIntent) -> dict[str, Any]:
    data = asdict(intent)
    data["side"] = intent.side.value
    data["action"] = intent.action.value
    data["regime"] = intent.regime.value
    return data


def _intent_from_dict(data: dict[str, Any]) -> OrderIntent:
    return OrderIntent(**data)


def _shadow_regime_without_sideways_precedence(
    regime: RegimeSnapshot, bars_4h: pd.DataFrame, settings,
) -> tuple[RegimeSnapshot, str]:
    """Return the next legacy regime if SIDEWAYS precedence were skipped in shadow only."""
    if regime.regime is not Regime.SIDEWAYS:
        return regime, "legacy_regime_unchanged"
    if (bars_4h.empty or not {"open", "close"}.issubset(bars_4h.columns)
            or not math.isfinite(float(regime.atr)) or float(regime.atr) <= 0
            or not math.isfinite(float(regime.range_position))):
        return regime, "missing_or_invalid_priority_inputs"

    last_bar = bars_4h.iloc[-1]
    body = float(last_bar["close"]) - float(last_bar["open"])
    impulse_threshold = settings.signals.impulse_atr_mult * float(regime.atr)
    if body >= impulse_threshold:
        counterfactual = Regime.IMPULSE_UP
        reason = "impulse_precedes_range_extreme"
    elif body <= -impulse_threshold:
        counterfactual = Regime.IMPULSE_DOWN
        reason = "impulse_precedes_range_extreme"
    elif regime.new_low or regime.range_position <= settings.regime.low_pct:
        counterfactual = Regime.LOW
        reason = "range_low_precedes_sideways"
    elif regime.new_high or regime.range_position >= settings.regime.high_pct:
        counterfactual = Regime.HIGH
        reason = "range_high_precedes_sideways"
    else:
        counterfactual = Regime.MID
        reason = "no_extreme_after_sideways"
    return replace(regime, regime=counterfactual), reason


def proposal_structure_signal_reason(
    side: Side | str,
    candle_close: float,
    active_state: dict[str, Any],
) -> str | None:
    """Return the existing entry-candle structure exit reason for a closed bar."""
    side = Side(side)
    if side is Side.LONG:
        level = active_state.get("entry_signal_low")
        if level is not None and candle_close < float(level):
            return "structure_signal_low"
    else:
        level = active_state.get("entry_signal_high")
        if level is not None and candle_close > float(level):
            return "structure_signal_high"
    return None


def runner_tail_is_exempt(side: Side | str, entry_price: float, position_stop: float | None,
                          exit_state: dict[str, Any], exits) -> bool:
    """Whether a TP1 runner has a confirmed break-even-or-better protective stop."""
    if not exits.exempt_max_hold_after_tp1 or not exit_state.get("tp1_taken"):
        return False
    if not exits.require_breakeven_for_exempt:
        return True
    if not exit_state.get("breakeven_moved"):
        return False
    stop = exit_state.get("protective_stop")
    if stop is None:
        stop = position_stop
    if stop is None or not math.isfinite(float(stop)):
        return False
    side = Side(side)
    return float(stop) >= entry_price if side is Side.LONG else float(stop) <= entry_price


def effective_max_hold_hours(side: Side | str, entry_price: float, position_stop: float | None,
                             exit_state: dict[str, Any], exits) -> float | None:
    """Return the applicable hour cap, or None when the position has no time cap."""
    regular_limit = float(exits.max_hold_bars)
    if regular_limit <= 0:
        return None
    if runner_tail_is_exempt(side, entry_price, position_stop, exit_state, exits):
        runner_limit = float(exits.runner_max_hold_bars)
        return runner_limit if runner_limit > 0 else None
    return regular_limit


def _safe_exception_frames(exc: Exception) -> list[dict[str, Any]]:
    """Return bounded traceback locations without exception text or local values."""
    frames = traceback.extract_tb(exc.__traceback__)[-6:]
    return [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in frames
    ]


@dataclass
class PaperAccount:
    """JSON-persisted paper cash, position, pending intent and idempotency cursor."""

    symbol: str
    initial_equity: float = 10_000.0
    cash: float = 10_000.0
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    active_entry_fee: float = 0.0
    active_entry_slippage: float = 0.0
    active_funding_paid: float = 0.0
    active_exit: dict[str, Any] | None = None
    isolated_margin: dict[str, float] | None = None
    pending: dict[str, Any] | None = None
    last_processed_close: str | None = None
    last_observed_open: str | None = None
    mark_prices: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.symbol.strip() or not math.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError("paper symbol and positive initial equity are required")
        if not math.isfinite(self.cash):
            raise ValueError("paper cash must be finite")
        if len(self.portfolio.positions) > 1 or any(key != self.symbol for key in self.portfolio.positions):
            raise ValueError("paper account supports one configured symbol")

    @property
    def position(self) -> Position | None:
        return self.portfolio.positions.get(self.symbol)

    def to_dict(self) -> dict[str, Any]:
        position = self.position
        return {
            "version": STATE_VERSION,
            "symbol": self.symbol,
            "initial_equity": self.initial_equity,
            "cash": self.cash,
            "position": None if position is None else {
                **asdict(position), "side": position.side.value,
                "opened_at": position.opened_at.isoformat(),
            },
            "last_exit_at": {key: value.isoformat() for key, value in self.portfolio.last_exit_at.items()},
            "last_exit_category": self.portfolio.last_exit_category,
            "last_exit_source": self.portfolio.last_exit_source,
            "trading_date": self.portfolio.trading_date.isoformat() if self.portfolio.trading_date else None,
            "day_start_equity": self.portfolio.day_start_equity,
            "daily_halted": self.portfolio.daily_halted,
            "active_entry_fee": self.active_entry_fee,
            "active_entry_slippage": self.active_entry_slippage,
            "active_funding_paid": self.active_funding_paid,
            "active_exit": self.active_exit,
            "isolated_margin": self.isolated_margin,
            "pending": self.pending,
            "last_processed_close": self.last_processed_close,
            "last_observed_open": self.last_observed_open,
            "mark_prices": self.mark_prices,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperAccount":
        if data.get("version") != STATE_VERSION:
            raise ValueError("unsupported paper state version")
        portfolio = PortfolioState()
        if data.get("position"):
            raw = data["position"].copy()
            raw["opened_at"] = _dt(raw["opened_at"])
            portfolio.positions[data["symbol"]] = Position(**raw)
        portfolio.last_exit_at = {key: _dt(value) for key, value in data.get("last_exit_at", {}).items()}
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
            symbol=data["symbol"], initial_equity=float(data["initial_equity"]), cash=float(data["cash"]),
            portfolio=portfolio, active_entry_fee=float(data.get("active_entry_fee", 0)),
            active_entry_slippage=float(data.get("active_entry_slippage", 0)),
            active_funding_paid=float(data.get("active_funding_paid", 0)),
            active_exit=data.get("active_exit"),
            isolated_margin=data.get("isolated_margin"),
            pending=data.get("pending"), last_processed_close=data.get("last_processed_close"),
            last_observed_open=data.get("last_observed_open"), mark_prices=data.get("mark_prices", {}),
            events=data.get("events", []),
        )

    @classmethod
    def load(cls, path: str | Path, *, symbol: str, initial_equity: float = 10_000.0) -> "PaperAccount":
        target = Path(path)
        if not target.exists():
            return cls(symbol=symbol, initial_equity=initial_equity, cash=initial_equity)
        account = cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        if account.symbol != symbol:
            raise ValueError(f"paper state belongs to {account.symbol}, not {symbol}")
        return account

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)


class PaperTrader:
    """One-symbol Paper trader with optional, explicitly locked Demo execution."""

    def __init__(self, account: PaperAccount, settings, *, execution=None):
        self.account = account
        self.settings = settings
        self.execution = execution
        self.fee_rate = settings.fees.taker_bps / 10_000
        self.slippage_rate = settings.fees.slippage_bps / 10_000
        self.margin_spec: dict[str, float] | None = None
        self.gross_notional_cap_pct = settings.sizing.max_gross_notional_pct
        self.market_risk_state = "NORMAL"
        self.market_risk_multiplier = 1.0

    @property
    def _trend_execution_enabled(self) -> bool:
        if self.execution is None:
            return True
        state = getattr(self.account, "state", None)
        return (
            getattr(state, "execution_mode", None) == "demo_usdm"
            and getattr(self.execution, "host", None) == DEMO_REST_ROOT
        )

    @property
    def _execution_mode(self) -> str:
        state = getattr(self.account, "state", None)
        return getattr(state, "execution_mode", "local_paper" if self.execution is None else "unverified")

    def _reserved_directional_notional(self, side: Side) -> float:
        state = getattr(self.account, "state", None)
        pending_by_symbol = getattr(state, "pending_by_symbol", {})
        reserved = 0.0
        for symbol, pending in pending_by_symbol.items():
            if symbol == self.account.symbol or pending.get("execution_type") != "pullback":
                continue
            if pending.get("side") != side.value:
                continue
            quantity = float(pending.get("planned_quantity") or 0.0)
            reference_price = float(pending.get("reference_price") or 0.0)
            if quantity > 0 and reference_price > 0:
                reserved += quantity * reference_price
            else:
                reserved += (self.settings.sizing.margin_per_symbol_usdt
                             * self.settings.sizing.leverage)
        return reserved

    def _reserved_entry_count(self) -> int:
        state = getattr(self.account, "state", None)
        pending_by_symbol = getattr(state, "pending_by_symbol", {})
        return sum(
            1 for symbol, pending in pending_by_symbol.items()
            if symbol != self.account.symbol and pending.get("execution_type") == "pullback"
        )

    def _evaluate_near_range_structure_shadow(
        self, *, regime, candle_1h: pd.Series, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame,
        flags, reference_price: float, atr_value: float, range_low: float | None,
        range_high: float | None, structure_low: float | None, structure_high: float | None,
        now: datetime, funding_rate: float | None, market_risk_state: str,
        step_size: float | None, minimum_quantity: float | None,
        allowed_entry_sources: set[str] | None, allow_entries: bool,
        entry_execution_allowed: bool, symbol_regime_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare legacy entry checks with a SIDEWAYS-priority shadow, without state changes."""
        legacy_signal = generate_signal(
            regime, candle_1h, flags, self.settings, position=None,
            bars_1h=bars_1h, bars_4h=bars_4h,
            entry_reference_price=float(reference_price), entry_atr=float(atr_value),
            range_low=range_low, range_high=range_high,
            structure_low=structure_low, structure_high=structure_high,
        )
        legacy_signal = with_signal_context(
            legacy_signal, symbol=self.account.symbol, source=MEAN_REVERSION, timestamp=now,
        )
        counterfactual_regime, counterfactual_regime_reason = (
            _shadow_regime_without_sideways_precedence(regime, bars_4h, self.settings)
        )
        counterfactual_signal = generate_signal(
            counterfactual_regime, candle_1h, flags, self.settings, position=None,
            bars_1h=bars_1h, bars_4h=bars_4h,
            entry_reference_price=float(reference_price), entry_atr=float(atr_value),
            range_low=range_low, range_high=range_high,
            structure_low=structure_low, structure_high=structure_high,
        )
        counterfactual_signal = with_signal_context(
            counterfactual_signal, symbol=self.account.symbol, source=MEAN_REVERSION, timestamp=now,
        )
        candle_open = float(candle_1h["open"])
        candle_close = float(candle_1h["close"])
        one_hour_direction = ("BULLISH" if candle_close > candle_open else
                              "BEARISH" if candle_close < candle_open else "DOJI")
        legacy_direction_gate = (
            regime.regime is Regime.LOW and one_hour_direction == "BULLISH"
        ) or (
            regime.regime is Regime.HIGH and one_hour_direction == "BEARISH"
        )
        counterfactual_direction_gate = (
            counterfactual_regime.regime is Regime.LOW and one_hour_direction == "BULLISH"
        ) or (
            counterfactual_regime.regime is Regime.HIGH and one_hour_direction == "BEARISH"
        )
        result: dict[str, Any] = {
            "event": "near_range_structure_shadow",
            "rules_version": "range_structure_sideways_priority_v1",
            "shadow_only": True,
            "symbol": self.account.symbol,
            "signal_timestamp": now.isoformat(),
            "legacy_regime": regime.regime.value,
            "counterfactual_regime": counterfactual_regime.regime.value,
            "counterfactual_regime_reason": counterfactual_regime_reason,
            "sideways_priority_skipped": regime.regime is Regime.SIDEWAYS,
            "sideways_priority_override": counterfactual_regime.regime is not regime.regime,
            "one_hour_direction": one_hour_direction,
            "legacy_direction_gate_pass": bool(legacy_direction_gate),
            "counterfactual_direction_gate_pass": bool(counterfactual_direction_gate),
            "signal_action": legacy_signal.action.value,
            "signal_side": legacy_signal.side.value if legacy_signal.side else None,
            "signal_reason": legacy_signal.reason,
            "counterfactual_signal_action": counterfactual_signal.action.value,
            "counterfactual_signal_side": (
                counterfactual_signal.side.value if counterfactual_signal.side else None
            ),
            "counterfactual_signal_reason": counterfactual_signal.reason,
            "structure_confirmation": symbol_regime_evidence.get("range_structure_shadow", {}),
            "execution_gates": {
                "allow_entries": bool(allow_entries),
                "entry_execution_allowed": bool(entry_execution_allowed),
                "mean_reversion_pool_allowed": (
                    MEAN_REVERSION in allowed_entry_sources
                    if allowed_entry_sources is not None else None
                ),
            },
            "risk_assessment_scenario": "counterfactual_signal",
            "risk_assessment": {
                "status": "not_assessed_no_counterfactual_open_signal",
                "allowed": None,
                "reason": None,
            },
            "hypothetical_entry_allowed": False,
            "baseline_hypothetical_entry_allowed": False,
        }
        if counterfactual_signal.action is not Action.OPEN or counterfactual_signal.side is None:
            return result

        reference = float(reference_price)
        atr_value = float(atr_value)
        structure_price = (structure_low if counterfactual_signal.side is Side.LONG
                           else structure_high)
        if self.settings.stops.use_structure_stop and structure_price is not None:
            raw_stop_distance = (
                reference - float(structure_price) if counterfactual_signal.side is Side.LONG
                else float(structure_price) - reference
            ) + self.settings.stops.structure_buffer_atr * atr_value
        else:
            raw_stop_distance = self.settings.stops.hard_cap_atr * atr_value
        raw_stop_width_atr = raw_stop_distance / atr_value if atr_value > 0 else None
        risk: dict[str, Any] = {
            "status": "not_assessed_market_metadata_missing",
            "allowed": False,
            "reason": "market_metadata_missing",
            "raw_stop_width_atr": raw_stop_width_atr,
            "stop_price": None,
            "quantity": None,
            "details": {},
        }
        if market_risk_state == "PAUSE_NEW" or self.market_risk_multiplier <= 0:
            risk.update({"status": "rejected", "allowed": False, "reason": "market_risk_paused"})
            result["risk_assessment"] = risk
            return result
        if step_size is None or minimum_quantity is None:
            result["risk_assessment"] = risk
            return result

        try:
            stop_price, stop_size_multiplier = calculate_adaptive_stop(
                counterfactual_signal.side, reference, atr_value, structure_price, self.settings,
            )
            risk["stop_price"] = float(stop_price)
            risk["stop_width_atr"] = abs(reference - float(stop_price)) / atr_value
            equity = self._current_equity()
            quantity = calculate_order_quantity(
                equity, reference, self.account.portfolio, self.settings,
                step_size=float(step_size), minimum_quantity=float(minimum_quantity),
                size_multiplier=(counterfactual_signal.size_multiplier * stop_size_multiplier
                                 * self.market_risk_multiplier),
                gross_notional_cap_pct=self.gross_notional_cap_pct,
            )
            risk["quantity"] = float(quantity)
            if quantity <= 0:
                target_notional = (
                    self.settings.sizing.margin_per_symbol_usdt
                    * self.settings.sizing.leverage * counterfactual_signal.size_multiplier
                    * stop_size_multiplier * self.market_risk_multiplier
                )
                tightened = self.gross_notional_cap_pct < self.settings.sizing.max_gross_notional_pct
                over_budget = (
                    tightened
                    and self.account.portfolio.gross_notional + target_notional
                    > equity * self.gross_notional_cap_pct
                )
                risk.update({
                    "status": "rejected",
                    "reason": "market_risk_budget_cap" if over_budget else "no_remaining_portfolio_capacity",
                })
                result["risk_assessment"] = risk
                return result

            intent = OrderIntent(
                self.account.symbol, counterfactual_signal.side, Action.OPEN, quantity, reference,
                counterfactual_signal.reason, counterfactual_regime.regime,
                float(stop_price), MEAN_REVERSION,
            )
            decision = assess_entry(
                intent, copy.deepcopy(self.account.portfolio), equity, funding_rate, now,
                self.settings, gross_notional_cap_pct=self.gross_notional_cap_pct,
                market_risk_state=market_risk_state,
                reserved_long_notional=self._reserved_directional_notional(Side.LONG),
                reserved_short_notional=self._reserved_directional_notional(Side.SHORT),
                reserved_symbols=self._reserved_entry_count(),
            )
            risk.update({
                "status": "approved" if decision.allowed else "rejected",
                "allowed": bool(decision.allowed),
                "reason": decision.reason,
                "details": decision.details or {},
            })
            if decision.allowed:
                state = getattr(self.account, "state", None)
                pending_by_symbol = getattr(state, "pending_by_symbol", {})
                reserved_entries = [
                    item for symbol, item in pending_by_symbol.items()
                    if symbol != self.account.symbol and item.get("execution_type") == "pullback"
                ]
                reserved_notional = sum(
                    float(item.get("planned_quantity", 0.0))
                    * float(item.get("reference_price", 0.0))
                    for item in reserved_entries
                )
                total_notional = (self.account.portfolio.gross_notional
                                  + intent.notional + reserved_notional)
                if total_notional > equity * self.gross_notional_cap_pct:
                    risk.update({
                        "status": "rejected", "allowed": False,
                        "reason": ("market_risk_budget_reserved_by_pending_entries"
                                   if self.gross_notional_cap_pct < self.settings.sizing.max_gross_notional_pct
                                   else "gross_notional_reserved_by_pending_entries"),
                    })
                elif self.account.cash < (
                        intent.notional / self.settings.sizing.leverage
                        + intent.notional * self.fee_rate):
                    risk.update({
                        "status": "rejected", "allowed": False,
                        "reason": "insufficient_available_isolated_margin",
                    })
        except (ArithmeticError, TypeError, ValueError) as exc:
            risk.update({"status": "rejected", "allowed": False, "reason": str(exc)})
        result["risk_assessment"] = risk
        gates_allow_entry = bool(
            allow_entries
            and entry_execution_allowed
            and (allowed_entry_sources is None or MEAN_REVERSION in allowed_entry_sources)
        )
        result["hypothetical_entry_allowed"] = bool(
            risk.get("allowed")
            and gates_allow_entry
        )
        result["baseline_hypothetical_entry_allowed"] = bool(
            legacy_signal.action is Action.OPEN
            and legacy_signal.side is counterfactual_signal.side
            and risk.get("allowed")
            and gates_allow_entry
        )
        return result

    @staticmethod
    def _trend_confirmation_fields(pending: dict[str, Any]) -> dict[str, Any]:
        if pending.get("source") != TREND:
            return {}
        return {
            "confirm_bars": pending.get("confirm_bars"),
            "confirmation_pass": pending.get("confirmation_pass"),
            "confirmation_bar_time": pending.get("confirmation_bar_time"),
        }

    @staticmethod
    def _hot_universe_exemption_fields(pending: dict[str, Any], *, status: str,
                                       end_reason: str | None = None) -> dict[str, Any]:
        if not pending.get("hot_universe_exemption"):
            return {}
        fields = {
            "exemption_granted": True,
            "exemption_reason": pending.get("exemption_reason"),
            "exemption_expires_at": pending.get("exemption_expires_at"),
            "exemption_status": status,
        }
        if end_reason is not None:
            fields["exemption_end_reason"] = end_reason
        return fields

    def _trend_quantity(self, equity: float, entry_price: float, stop_price: float, *,
                        step_size: float, minimum_quantity: float) -> float:
        quantity, _, _ = calculate_trend_quantity(
            equity, entry_price, stop_price, self.settings,
            step_size=step_size, minimum_quantity=minimum_quantity,
            existing_gross_notional=self.account.portfolio.gross_notional,
            gross_notional_cap_pct=self.gross_notional_cap_pct,
        )
        if self.market_risk_multiplier < 1.0:
            step = Decimal(str(step_size))
            quantity = float((Decimal(str(quantity)) * Decimal(str(self.market_risk_multiplier)) / step)
                             .to_integral_value(rounding=ROUND_FLOOR) * step)
        return quantity if quantity >= minimum_quantity else 0.0

    def _event(self, event: str, **details) -> None:
        details.setdefault("symbol", self.account.symbol)
        details.setdefault("execution_mode", self._execution_mode)
        self.account.events.append({"event": event, "at": datetime.now(timezone.utc).isoformat(), **details})
        if event in {"entry_confirmed", "pending_entry_started", "pending_entry_cancelled",
                     "pending_entry_timeout_fallback", "pending_entry_filled",
                     "hot_universe_exemption_granted", "signal_rejected",
                     "intent_rejected", "structure_invalidation_comparison"}:
            logger.info("{} symbol={} details={}", event.upper(), self.account.symbol, details)

    def _proposal_v1(self) -> bool:
        state = self.account.active_exit or {}
        return state.get("policy") == "proposal_v1"

    def _structure_invalidation_levels(self, position, state: dict[str, Any]) -> dict[str, Any] | None:
        exits = self.settings.exits
        if not exits.structure_invalid_enabled:
            return None
        is_long = position.side is Side.LONG
        key = "entry_structure_low" if is_long else "entry_structure_high"
        structure_point = state.get(key)
        atr_value = float(state.get("entry_atr", 0.0) or 0.0)
        if structure_point is None or not math.isfinite(atr_value) or atr_value <= 0:
            return None
        structure_point = float(structure_point)
        buffer = exits.structure_invalid_buffer_atr * atr_value
        invalid_price = structure_point - buffer if is_long else structure_point + buffer
        protective_stop = float(state.get("protective_stop", position.stop_price))
        structure_is_tighter = invalid_price > protective_stop if is_long else invalid_price < protective_stop
        trigger_price = max(invalid_price, protective_stop) if is_long else min(invalid_price, protective_stop)
        return {
            "side": "long" if is_long else "short",
            "structure_point": structure_point,
            "buffer": buffer,
            "buffer_atr": exits.structure_invalid_buffer_atr,
            "entry_atr": atr_value,
            "invalid_price": invalid_price,
            "protective_stop": protective_stop,
            "trigger_price": trigger_price,
            "trigger_source": "structure_invalid" if structure_is_tighter else "protective_stop",
            "structure_invalid_is_looser": not structure_is_tighter,
        }

    def _price_invalidation_reason(self, position, state: dict[str, Any], price: float) -> str | None:
        exits = self.settings.exits
        levels = self._structure_invalidation_levels(position, state)
        if position.side is Side.LONG:
            if (levels is not None and levels["trigger_source"] == "structure_invalid"
                    and price < levels["invalid_price"]):
                return "long_structure_invalid"
            if (exits.regime_fade_enabled and position.regime == Regime.LOW.value
                    and state.get("latest_regime") == Regime.SIDEWAYS.value
                    and price - position.entry_price
                    < exits.regime_fade_min_atr * float(state.get("entry_atr", 0.0))):
                return "long_regime_faded"
        else:
            if (levels is not None and levels["trigger_source"] == "structure_invalid"
                    and price > levels["invalid_price"]):
                return "short_structure_invalid"
            if (exits.regime_fade_enabled and position.regime == Regime.HIGH.value
                    and state.get("latest_regime") == Regime.SIDEWAYS.value
                    and position.entry_price - price
                    < exits.regime_fade_min_atr * float(state.get("entry_atr", 0.0))):
                return "short_regime_faded"
        return None

    def _proposal_partial_close(self, quantity: float, raw_price: float, at: datetime, reason: str) -> bool:
        position = self.account.position
        state = self.account.active_exit or {}
        if position is None or quantity <= 0:
            return False
        step, minimum = (self.execution._market_rules(position.symbol)[:2]
                         if self.execution is not None else (1e-8, 0.0))
        if step > 0:
            quantity = math.floor((quantity + step * 1e-10) / step) * step
        quantity = min(float(quantity), position.quantity)
        remaining = position.quantity - quantity
        if quantity < minimum or remaining <= 0 or remaining < minimum:
            return False
        old_stop = (self.account.exchange_orders_by_symbol.get(position.symbol, {}).get("stop")
                    if self.execution is not None else None)
        if self.execution is not None:
            result = self.execution.reduce(position, quantity, stop=old_stop, reason=reason)
            fill = result["fill"]
            close_qty, close_price = float(fill["filled"]), float(fill["average"])
            if not math.isclose(close_qty, quantity, rel_tol=1e-8, abs_tol=max(step, 1e-10)):
                self.account.execution_status = "inconsistent"
                raise RuntimeError("Demo partial fill quantity differs from requested reduction")
            remaining_position = result["position"]
            actual_remaining = float(remaining_position["quantity"])
            close_fee = fill.get("fee_cost")
            if close_fee is None:
                close_fee = close_qty * close_price * self.fee_rate
            stop = result.get("stop")
            orders = self.account.exchange_orders_by_symbol[position.symbol]
            orders["stop"] = stop
            stale_stops = result.get("stale_stops") or []
            if stale_stops:
                orders["stale_stops"] = stale_stops
                self.account.execution_status = "inconsistent"
                self._event("protective_stop_cleanup_pending", order_ids=[
                    str(item.get("algoId") or item.get("id")) for item in stale_stops
                ], reason=result.get("stop_cleanup_error") or "superseded_bot_stops_remain")
            else:
                orders.pop("stale_stops", None)
            self.account.mark_prices[position.symbol] = close_price
        else:
            close_price, _ = _apply_slippage(raw_price, position.side, is_entry=False,
                                             slippage_rate=self.slippage_rate)
            close_qty = quantity
            close_fee = close_qty * close_price * self.fee_rate
            actual_remaining = remaining
            stop = None

        sign = 1 if position.side is Side.LONG else -1
        gross = close_qty * (close_price - position.entry_price) * sign
        initial_quantity = float(state.get("initial_quantity", position.quantity))
        entry_share = float(state.get("initial_entry_fee", self.account.active_entry_fee)) * close_qty / initial_quantity
        funding_share = self.account.active_funding_paid * close_qty / position.quantity
        state["realized_gross"] = float(state.get("realized_gross", 0.0)) + gross
        state["realized_exit_fees"] = float(state.get("realized_exit_fees", 0.0)) + float(close_fee)
        state["realized_entry_fee"] = float(state.get("realized_entry_fee", 0.0)) + entry_share
        state["realized_funding"] = float(state.get("realized_funding", 0.0)) + funding_share
        state["realized_exit_quantity"] = float(state.get("realized_exit_quantity", 0.0)) + close_qty
        state["realized_exit_notional"] = float(state.get("realized_exit_notional", 0.0)) + close_qty * close_price
        self.account.active_funding_paid -= funding_share
        self.account.active_entry_fee -= entry_share
        self.account.portfolio.positions[position.symbol] = replace(
            position, quantity=actual_remaining)
        if self.execution is None:
            margin = self.account.isolated_margin
            if margin:
                fraction = close_qty / position.quantity
                released = float(margin["initial_margin"]) * fraction
                margin["initial_margin"] -= released
                self.account.cash += released + gross - float(close_fee)
        state["protective_stop"] = (float(stop["stop_price"]) if stop
                                    else float(state.get("protective_stop", position.stop_price)))
        net = gross - float(close_fee) - entry_share - funding_share
        self._event("paper_partial_fill", action="REDUCE", side=position.side.value,
                    quantity=close_qty, remaining_quantity=actual_remaining, price=close_price,
                    reason=reason, gross_pnl=gross, fees=float(close_fee) + entry_share,
                    funding_paid=funding_share, net_pnl=net,
                    orderId=fill.get("id") if self.execution is not None else None,
                    host=getattr(self.execution, "host", None),
                    execution_source="binance_usdm_demo" if self.execution is not None else "local_paper")
        if self.execution is not None:
            try:
                self._refresh_exchange_snapshot()
                self.account.exchange_orders_by_symbol[position.symbol].pop(
                    "snapshot_reconciliation_required", None,
                )
            except Exception as exc:
                self.account.execution_status = "inconsistent"
                self.account.exchange_orders_by_symbol[position.symbol][
                    "snapshot_reconciliation_required"
                ] = True
                self._event("execution_snapshot_refresh_failed", reason=type(exc).__name__,
                            orderId=fill.get("id"), action="REDUCE")
        return True

    def _update_protective_stop(self, stop_price: float, reason: str, *, current_price: float) -> bool:
        position = self.account.position
        state = self.account.active_exit or {}
        if position is None:
            return False
        old = float(state.get("protective_stop", position.stop_price))
        if not math.isfinite(current_price) or current_price <= 0:
            return False
        if self.execution is not None:
            orders = self.account.exchange_orders_by_symbol.get(position.symbol, {})
            try:
                # Compare the exchange-precision price so sub-tick improvements
                # do not replace the same protective stop on every poll.
                stop_price = float(self.execution.exchange.price_to_precision(position.symbol, stop_price))
            except Exception as exc:
                self.account.execution_status = "inconsistent"
                self._event("protective_stop_update_failed", reason=type(exc).__name__,
                            frames=_safe_exception_frames(exc))
                return False
        improved = stop_price > old if position.side is Side.LONG else stop_price < old
        if not improved:
            return False
        already_triggered_side = (stop_price >= current_price if position.side is Side.LONG
                                  else stop_price <= current_price)
        if already_triggered_side:
            self._event("protective_stop_update_skipped", side=position.side.value,
                        stop_price=stop_price, current_price=float(current_price),
                        previous_stop=old, reason=reason, skip_reason="would_trigger_immediately")
            return False
        if self.execution is not None:
            try:
                orders["stop"] = self.execution.replace_stop(position, stop=orders.get("stop"), stop_price=stop_price)
            except Exception as exc:
                self.account.execution_status = "inconsistent"
                self._event("protective_stop_update_failed", reason=type(exc).__name__,
                            frames=_safe_exception_frames(exc))
                return False
            stop_price = float(orders["stop"]["stop_price"])
        state["protective_stop"] = stop_price
        # Keep Position.stop_price as the entry-time risk stop. A managed
        # breakeven/trailing stop may cross entry; its authoritative value is
        # active_exit.protective_stop and the exchange algo order.
        self._event("protective_stop_updated", side=position.side.value, stop_price=stop_price,
                    previous_stop=old, reason=reason)
        return True

    def _manage_proposal_tick(self, current_price: float, now: datetime) -> bool:
        position = self.account.position
        state = self.account.active_exit or {}
        if position is None or not self._proposal_v1():
            return False
        policy = self.settings.exit_proposal_v1
        side_sign = 1 if position.side is Side.LONG else -1
        atr_value = float(state["entry_atr"])
        best = float(state["best_price"])
        mfe = max(0.0, side_sign * (best - position.entry_price))
        initial_quantity = float(state["initial_quantity"])
        changed = False

        breakeven_distance = float(self.settings.stops.move_to_breakeven_atr) * atr_value
        if mfe >= breakeven_distance and not state.get("breakeven_moved"):
            remaining_entry_fee = float(state.get("initial_entry_fee", 0.0)) * position.quantity / max(initial_quantity, 1e-12)
            funding_per_unit = self.account.active_funding_paid / max(position.quantity, 1e-12)
            cost_per_unit = remaining_entry_fee / max(position.quantity, 1e-12) + funding_per_unit
            if position.side is Side.LONG:
                exit_fill = (position.entry_price + cost_per_unit) / (1 - self.fee_rate)
                breakeven = exit_fill / (1 - self.slippage_rate)
                moved = self._update_protective_stop(max(float(state["protective_stop"]), breakeven),
                                                     "breakeven_atr", current_price=current_price)
            else:
                exit_fill = (position.entry_price - cost_per_unit) / (1 + self.fee_rate)
                breakeven = exit_fill / (1 + self.slippage_rate)
                moved = self._update_protective_stop(min(float(state["protective_stop"]), breakeven),
                                                     "breakeven_atr", current_price=current_price)
            if moved:
                state["breakeven_moved"] = True
                changed = True

        tp1_distance = policy.tp1_atr * atr_value
        tp1_price_reached = (current_price >= position.entry_price + tp1_distance
                             if position.side is Side.LONG
                             else current_price <= position.entry_price - tp1_distance)
        if not state.get("tp1_taken") and mfe >= tp1_distance and tp1_price_reached:
            quantity = initial_quantity * policy.tp1_frac
            step, minimum = (self.execution._market_rules(position.symbol)[:2]
                             if self.execution is not None else (1e-8, 0.0))
            rounded_quantity = math.floor((quantity + step * 1e-10) / step) * step
            if position.quantity - rounded_quantity < minimum:
                return self._close(current_price, now, "take_profit_1_close_remainder") or changed
            if self._proposal_partial_close(quantity, current_price, now, "take_profit_1"):
                state["tp1_taken"] = True
                state["breakeven_moved"] = True
                changed = True
                position = self.account.position
                state = self.account.active_exit
                remaining_entry_fee = float(state["initial_entry_fee"]) * position.quantity / initial_quantity
                funding_per_unit = self.account.active_funding_paid / position.quantity
                cost_per_unit = remaining_entry_fee / position.quantity + funding_per_unit
                if position.side is Side.LONG:
                    exit_fill = (position.entry_price + cost_per_unit) / (1 - self.fee_rate)
                    breakeven = exit_fill / (1 - self.slippage_rate)
                    self._update_protective_stop(max(float(state["protective_stop"]), breakeven),
                                                 "breakeven_after_tp1", current_price=current_price)
                else:
                    exit_fill = (position.entry_price - cost_per_unit) / (1 + self.fee_rate)
                    breakeven = exit_fill / (1 + self.slippage_rate)
                    self._update_protective_stop(min(float(state["protective_stop"]), breakeven),
                                                 "breakeven_after_tp1", current_price=current_price)

        position = self.account.position
        if position is None:
            return changed
        state = self.account.active_exit
        best = float(state["best_price"])
        mfe = max(0.0, side_sign * (best - position.entry_price))
        tp2_distance = policy.tp2_atr * atr_value
        tp2_price_reached = (current_price >= position.entry_price + tp2_distance
                             if position.side is Side.LONG
                             else current_price <= position.entry_price - tp2_distance)
        if (state.get("tp1_taken") and not state.get("tp2_taken")
                and mfe >= tp2_distance and tp2_price_reached):
            quantity = initial_quantity * policy.tp2_frac
            step, minimum = (self.execution._market_rules(position.symbol)[:2]
                             if self.execution is not None else (1e-8, 0.0))
            remainder = position.quantity - math.floor((quantity + step * 1e-10) / step) * step
            if position.quantity <= quantity + max(step, 1e-10) or remainder < minimum:
                closed = self._close(current_price, now, "take_profit_2_close_remainder")
                return closed or changed
            if self._proposal_partial_close(quantity, current_price, now, "take_profit_2"):
                state["tp2_taken"] = True
                changed = True
        position = self.account.position
        if position is None:
            return changed
        state = self.account.active_exit
        if state.get("tp1_taken"):
            best = float(state["best_price"])
            regime = state.get("latest_regime")
            tight = ((position.side is Side.LONG and regime == Regime.HIGH.value)
                     or (position.side is Side.SHORT and regime == Regime.LOW.value))
            distance = (policy.trail_tight_atr if tight else policy.trail_atr) * atr_value
            trailing = best - distance if position.side is Side.LONG else best + distance
            old = float(state.get("protective_stop", position.stop_price))
            target = max(old, trailing) if position.side is Side.LONG else min(old, trailing)
            if self._update_protective_stop(target, "atr_trailing", current_price=current_price):
                changed = True
        return changed

    def _current_equity(self) -> float:
        shared = getattr(self.account, "current_equity", None)
        if callable(shared):
            return float(shared())
        position = self.account.position
        if position is None:
            return float(self.account.cash)
        price = self.account.mark_prices.get(self.account.symbol, position.entry_price)
        direction = 1 if position.side is Side.LONG else -1
        margin = float((getattr(self.account, "isolated_margin", None) or {}).get("initial_margin", 0.0))
        return float(self.account.cash + margin + position.quantity * (price - position.entry_price) * direction)

    def _close(self, raw_price: float, at: datetime, reason: str, *, liquidation: bool = False) -> bool:
        position = self.account.position
        if position is None:
            return False
        was_proposal_v1 = self._proposal_v1()
        close_intent = OrderIntent(
            self.account.symbol, position.side, Action.CLOSE, position.quantity, raw_price,
            reason, Regime(position.regime), source=position.source,
        )
        if self.execution is not None:
            self._event("order_intent", intent=_intent_dict(close_intent), allowed=True,
                        decision="demo_reduce_only_close")
            orders = self.account.exchange_orders_by_symbol.get(self.account.symbol, {})
            result = self.execution.close(position, stop=orders.get("stop"), reason=reason)
            if result.get("already_flat") or not result.get("fill"):
                self.account.execution_status = "inconsistent"
                return False
            fill = result["fill"]
            close_qty, close_price = float(fill["filled"]), float(fill["average"])
            sign = 1 if position.side is Side.LONG else -1
            gross = close_qty * (close_price - position.entry_price) * sign
            close_fee = fill.get("fee_cost")
            if close_fee is None:
                close_fee = close_qty * close_price * self.fee_rate
            exit_state = self.account.active_exit or {}
            gross_total = float(exit_state.get("realized_gross", 0.0)) + gross
            total_fees = (self.account.active_entry_fee + float(close_fee)
                          + float(exit_state.get("realized_exit_fees", 0.0))
                          + float(exit_state.get("realized_entry_fee", 0.0)))
            total_funding = float(exit_state.get("realized_funding", 0.0)) + self.account.active_funding_paid
            net = gross_total - total_fees - total_funding
            self.account.portfolio.register_close(self.account.symbol, at, reason=reason, net_pnl=net)
            exit_state = self.account.active_exit or {}
            risk = float(exit_state.get("initial_risk") or abs(position.entry_price - position.stop_price))
            side_sign = 1 if position.side is Side.LONG else -1
            managed = bool(exit_state.get("enabled")) and risk > 0 and exit_state.get("best_price") is not None
            mfe_r = side_sign * (float(exit_state["best_price"]) - position.entry_price) / risk if managed else None
            self._event("paper_fill", action="CLOSE", side=position.side.value,
                        quantity=close_qty, price=close_price, average=close_price,
                        stop_price=exit_state.get("protective_stop", position.stop_price), reason=reason,
                        entry_reason=position.reason, entry_regime=position.regime,
                         source=position.source, entry_source=position.source,
                        orderId=fill["id"], host=result.get("host"), execution_source="binance_usdm_demo",
                        held_hours=max(0.0, (at - position.opened_at).total_seconds() / 3600),
                        mfe_r=mfe_r, mfe_available=managed, gross_pnl=gross_total, fees=total_fees,
                        funding_paid=total_funding, net_pnl=net)
            self.account.exchange_orders_by_symbol.pop(self.account.symbol, None)
            self.account.active_entry_fee = self.account.active_entry_slippage = self.account.active_funding_paid = 0.0
            self.account.active_exit = None
            self.account.isolated_margin = None
            self.account.mark_prices[self.account.symbol] = close_price
            self._refresh_exchange_snapshot()
            if was_proposal_v1:
                self.account.last_processed_close = pd.Timestamp(at).floor("h").isoformat()
            return True
        self._event("order_intent", intent=_intent_dict(close_intent), allowed=True, decision="paper_close")
        fill, slip_per_unit = _apply_slippage(raw_price, position.side, is_entry=False, slippage_rate=self.slippage_rate)
        direction = 1 if position.side is Side.LONG else -1
        gross = position.quantity * (fill - position.entry_price) * direction
        fee = position.quantity * fill * self.fee_rate
        exit_state = self.account.active_exit or {}
        gross_total = float(exit_state.get("realized_gross", 0.0)) + gross
        total_fees = (self.account.active_entry_fee + fee
                      + float(exit_state.get("realized_exit_fees", 0.0))
                      + float(exit_state.get("realized_entry_fee", 0.0)))
        total_funding = float(exit_state.get("realized_funding", 0.0)) + self.account.active_funding_paid
        margin = getattr(self.account, "isolated_margin", None)
        liquidation_fee = 0.0
        if margin:
            liquidation_fee = position.quantity * fill * float(margin.get("liquidation_fee_rate", 0.0)) if liquidation else 0.0
            allocated = float(margin["initial_margin"])
            uncapped_refund = allocated + gross - fee - liquidation_fee - self.account.active_funding_paid
            refund = max(0.0, uncapped_refund)
            net = gross_total - total_fees - total_funding - liquidation_fee
            self.account.cash += refund
            if uncapped_refund < 0:
                self._event("isolated_deficit_capped", deficit=round(-uncapped_refund, 8))
        else:
            net = gross_total - total_fees - total_funding
            self.account.cash += gross - fee
        self.account.portfolio.register_close(self.account.symbol, at, reason=reason, net_pnl=net)
        risk = float(exit_state.get("initial_risk") or abs(position.entry_price - position.stop_price))
        sign = 1 if position.side is Side.LONG else -1
        managed = bool(exit_state.get("enabled")) and risk > 0 and exit_state.get("best_price") is not None
        mfe_r = sign * (float(exit_state["best_price"]) - position.entry_price) / risk if managed else None
        mae_complete = managed and bool(exit_state.get("mae_history_complete"))
        worst = exit_state.get("worst_price")
        mae_r = max(0.0, -sign * (float(worst) - position.entry_price) / risk) if mae_complete and worst is not None else None
        held_hours = max(0.0, (at - position.opened_at).total_seconds() / 3600)
        self._event("paper_fill", action="CLOSE", symbol=self.account.symbol, side=position.side.value,
                    quantity=position.quantity, price=fill,
                    stop_price=(self.account.active_exit or {}).get("protective_stop", position.stop_price), reason=reason,
                    entry_reason=position.reason, entry_regime=position.regime,
                     source=position.source, entry_source=position.source, held_hours=held_hours,
                    mfe_r=mfe_r, mfe_available=managed, mae_r=mae_r, mae_available=mae_complete,
                    gross_pnl=gross_total, fees=total_fees + liquidation_fee, slippage_paid=self.account.active_entry_slippage + position.quantity * slip_per_unit,
                    funding_paid=total_funding, liquidation_fee=liquidation_fee,
                    execution_price_source="last_trade", trigger_price_source="mark_price" if liquidation else None,
                    net_pnl=net)
        self.account.active_entry_fee = self.account.active_entry_slippage = self.account.active_funding_paid = 0.0
        self.account.active_exit = None
        self.account.isolated_margin = None
        if was_proposal_v1:
            self.account.last_processed_close = pd.Timestamp(at).floor("h").isoformat()
        return True

    def _refresh_exchange_snapshot(self) -> None:
        snapshot = self.execution.sync_snapshot()
        self.account.exchange_snapshot = snapshot
        if snapshot.get("available_usdt") is not None:
            self.account.cash = float(snapshot["available_usdt"])

    def manual_close(self, raw_price: float, at: datetime, *, request_id: str) -> bool:
        """Close one paper position at the supplied latest public trade price."""
        if self.account.position is None:
            return False
        if not math.isfinite(raw_price) or raw_price <= 0:
            raise ValueError("manual close price must be finite and positive")
        pending = self.account.pending
        if pending is not None:
            self._event("signal_cancelled", reason="manual_close", action=pending.get("action"),
                        pending_reason=pending.get("reason"))
            self.account.pending = None
        self.account.mark_prices[self.account.symbol] = float(raw_price)
        if not self._close(raw_price, at, "manual_close"):
            return False
        self._event("manual_close_request_processed", request_id=request_id, price=raw_price)
        return True

    @staticmethod
    def _new_exit_state(pending: dict[str, Any], side: Side, quantity: float,
                        entry_price: float, entry_fee: float, stop: float,
                        proposal_enabled: bool) -> dict[str, Any]:
        midpoint = pending.get("entry_midpoint")
        signal_close = pending.get("entry_signal_close")
        return {
            "policy": "proposal_v1" if proposal_enabled else None,
            "invalidation_enabled": True, "entry_atr": float(pending["atr"]),
            "initial_quantity": float(quantity), "initial_entry_fee": float(entry_fee),
            "tp1_taken": False, "tp2_taken": False, "no_progress_reduced": False,
            "entry_signal_low": pending.get("entry_signal_low"),
            "entry_signal_high": pending.get("entry_signal_high"),
            "entry_structure_low": pending.get("entry_structure_low"),
            "entry_structure_high": pending.get("entry_structure_high"),
            "entry_range_low": pending.get("entry_range_low"),
            "entry_range_high": pending.get("entry_range_high"),
            "no_progress_count": 0,
            "no_progress_last_close": pending.get("entry_signal_close"),
            "entry_midpoint": midpoint,
            "last_signal_close": signal_close,
            "midline_armed": (signal_close is not None and midpoint is not None
                              and (float(signal_close) >= float(midpoint) if side is Side.LONG
                                   else float(signal_close) <= float(midpoint))),
            "realized_gross": 0.0, "realized_exit_fees": 0.0,
            "realized_entry_fee": 0.0, "realized_funding": 0.0,
            "realized_exit_quantity": 0.0, "realized_exit_notional": 0.0,
            "protective_stop": float(stop), "breakeven_moved": False,
        }

    def _record_pullback_risk_reject(self, pending: dict[str, Any], at: datetime, detail: str) -> None:
        if pending.get("execution_type") == "pullback":
            event_reason = detail if detail in {
                "long_rejected_below_range", "short_rejected_above_range", "invalid_on_entry"
            } else "risk_reject"
            self._event("pending_entry_cancelled", reason=event_reason, detail=detail,
                        signal_time=pending.get("signal_time"), cancelled_at=at.isoformat(),
                        source=pending.get("source") or MEAN_REVERSION,
                        **self._hot_universe_exemption_fields(
                            pending, status="rejected", end_reason=detail,
                        ),
                        **self._trend_confirmation_fields(pending))

    def _execute_pending(self, execution_price: float, at: datetime, funding_rate: float | None, *,
                         step_size: float = 1e-8, minimum_quantity: float = 0.0) -> None:
        pending = self.account.pending
        if pending is None:
            return
        expected = _dt(pending["execute_at"])
        if at < expected:
            return
        self.account.pending = None
        if at != expected:
            self._event("intent_expired", reason="next_hour_open_missed", expected_at=expected.isoformat(),
                        observed_at=at.isoformat(),
                        **self._hot_universe_exemption_fields(
                            pending, status="expired", end_reason="next_hour_open_missed",
                        ))
            return
        signal = Signal(
            Side(pending["side"]) if pending["side"] else None,
            Action(pending["action"]), pending["reason"], float(pending["strength"]),
            close_first=bool(pending.get("close_first", False)), size_multiplier=float(pending.get("size_multiplier", 1.0)),
            symbol=self.account.symbol, source=pending.get("source") or MEAN_REVERSION,
            timestamp=_dt(pending.get("signal_time") or pending["execute_at"]),
        )
        if signal.action is Action.CLOSE:
            self._close(execution_price, at, signal.reason)
            return
        if signal.action is not Action.OPEN or signal.side is None:
            return
        confirmation_fields = self._trend_confirmation_fields(pending)
        if self.gross_notional_cap_pct <= 0:
            self._event("intent_rejected", reason="market_risk_paused", signal_source=signal.source,
                        **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, "market_risk_paused")
            return
        if signal.source not in {MEAN_REVERSION, TREND} or (
                signal.source == TREND and not self._trend_execution_enabled):
            reason = ("trend_execution_mode_not_allowed" if signal.source == TREND
                      else "invalid_strategy_source")
            self._event("intent_rejected", reason=reason, signal_source=signal.source,
                        signal_time=signal.timestamp.isoformat() if signal.timestamp else None,
                        **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, reason)
            return
        is_trend = signal.source == TREND
        held_position = self.account.position
        if (held_position is not None
                and held_position.source != signal.source):
            self._event("intent_rejected", reason="position_owned_by_other_strategy",
                        signal_source=signal.source, position_source=held_position.source,
                        signal_time=signal.timestamp.isoformat() if signal.timestamp else None,
                        **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, "position_owned_by_other_strategy")
            return
        if not is_trend:
            rejection = entry_rejection_reason(
                signal.side, float(execution_price), float(pending["atr"]),
                pending.get("entry_range_low"), pending.get("entry_range_high"),
                pending.get("entry_structure_low"), pending.get("entry_structure_high"), self.settings,
                require_structure=("entry_structure_low" in pending if signal.side is Side.LONG
                                    else "entry_structure_high" in pending),
            )
            if rejection is not None:
                self._event("intent_rejected", reason=rejection, reference_price=float(execution_price),
                            entry_range_low=pending.get("entry_range_low"),
                            entry_range_high=pending.get("entry_range_high"),
                            entry_structure_low=pending.get("entry_structure_low"),
                            entry_structure_high=pending.get("entry_structure_high"),
                            signal_source=signal.source, **confirmation_fields)
                self._record_pullback_risk_reject(pending, at, rejection)
                return
        if self.margin_spec is not None and not {
            "maintenance_margin_rate", "liquidation_fee_rate"
        }.issubset(self.margin_spec):
            self._event("intent_rejected", reason="public_isolated_margin_metadata_unavailable",
                        signal_source=signal.source, **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, "public_isolated_margin_metadata_unavailable")
            return
        if self.account.position is not None and not signal.close_first:
            self._event("intent_rejected", reason="position_still_open",
                        signal_source=signal.source, **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, "position_still_open")
            return
        if self.account.position is not None:
            if not self._close(execution_price, at, "reverse_close_first"):
                self._event("intent_rejected", reason="reverse_close_not_confirmed",
                            signal_source=signal.source, **confirmation_fields)
                return
        fill, slip_per_unit = _apply_slippage(execution_price, signal.side, is_entry=True, slippage_rate=self.slippage_rate)
        if not is_trend:
            rejection = entry_rejection_reason(
                signal.side, float(fill), float(pending["atr"]),
                pending.get("entry_range_low"), pending.get("entry_range_high"),
                pending.get("entry_structure_low"), pending.get("entry_structure_high"), self.settings,
                require_structure=("entry_structure_low" in pending if signal.side is Side.LONG
                                    else "entry_structure_high" in pending),
            )
            if rejection is not None:
                self._event("intent_rejected", reason=rejection, reference_price=float(fill),
                            signal_source=signal.source, **confirmation_fields)
                self._record_pullback_risk_reject(pending, at, rejection)
                return
        equity = self._current_equity()
        try:
            atr_value = float(pending["atr"])
            if is_trend:
                stop = float(pending["stop_price"])
                if ((signal.side is Side.LONG and fill <= stop)
                        or (signal.side is Side.SHORT and fill >= stop)):
                    raise ValueError("trend_stop_not_beyond_actual_fill")
                quantity = self._trend_quantity(
                    equity, fill, stop, step_size=step_size, minimum_quantity=minimum_quantity,
                )
                if quantity <= 0:
                    raise ValueError("trend_risk_size_below_exchange_minimum")
                proposal_enabled = False
            else:
                proposal_enabled = bool(self.settings.exit_proposal_v1.enabled)
                structure_price = pending.get("entry_structure_low" if signal.side is Side.LONG
                                              else "entry_structure_high")
                stop, stop_size_multiplier = calculate_adaptive_stop(
                    signal.side, fill, atr_value, structure_price, self.settings,
                )
                size_multiplier = (signal.size_multiplier * stop_size_multiplier
                                   * self.market_risk_multiplier)
                quantity = calculate_order_quantity(
                    equity, fill, self.account.portfolio, self.settings,
                    step_size=step_size, minimum_quantity=minimum_quantity,
                    size_multiplier=size_multiplier,
                    gross_notional_cap_pct=self.gross_notional_cap_pct,
                    stop_price=stop,
                )
                if quantity <= 0 or quantity < minimum_quantity:
                    target_notional = (self.settings.sizing.margin_per_symbol_usdt
                                       * self.settings.sizing.leverage * size_multiplier)
                    market_cap_tightened = self.gross_notional_cap_pct < self.settings.sizing.max_gross_notional_pct
                    over_market_budget = (market_cap_tightened
                                          and self.account.portfolio.gross_notional + target_notional
                                          > equity * self.gross_notional_cap_pct)
                    reason = ("stop_too_wide" if stop_size_multiplier < 1 else
                              "market_risk_budget_cap" if over_market_budget else
                              "stop_distance_size_reduced_below_exchange_minimum")
                    self._event("intent_rejected", reason=reason, signal_source=signal.source,
                                **confirmation_fields)
                    self._record_pullback_risk_reject(pending, at, reason)
                    return
            order = OrderIntent(self.account.symbol, signal.side, Action.OPEN, quantity, fill, signal.reason,
                                Regime(pending["regime"]), stop, signal.source)
            approval = guard_signal(signal, order, self.account.portfolio, equity, funding_rate, at, self.settings,
                                    bypass_cooldown=signal.close_first,
                                    gross_notional_cap_pct=self.gross_notional_cap_pct,
                                    market_risk_state=self.market_risk_state,
                                    reserved_long_notional=self._reserved_directional_notional(Side.LONG),
                                    reserved_short_notional=self._reserved_directional_notional(Side.SHORT),
                                    reserved_symbols=self._reserved_entry_count())
        except ValueError as exc:
            self._event("intent_rejected", reason=str(exc), signal_source=signal.source,
                        **confirmation_fields,
                        intent=_intent_dict(order) if "order" in locals() else None)
            self._record_pullback_risk_reject(pending, at, str(exc))
            return
        self._event("order_intent", intent=_intent_dict(order), decision=approval.reason,
                    allowed=approval.allowed, source=signal.source, direction=signal.direction,
                    signal_timestamp=signal.timestamp.isoformat() if signal.timestamp else None,
                    funding_rate=funding_rate, risk_details=approval.details or {},
                    market_risk_state=self.market_risk_state, **confirmation_fields)
        if not approval.allowed:
            self._record_pullback_risk_reject(pending, at, approval.reason)
            return
        if self.execution is not None:
            required_margin = order.notional / self.settings.sizing.leverage
            required_with_fee = required_margin + order.notional * self.fee_rate
            if self.account.cash < required_with_fee:
                self._event("intent_rejected", reason="insufficient_available_isolated_margin",
                            required_margin=required_margin, required_fee=required_with_fee - required_margin,
                            available_cash=self.account.cash, intent=_intent_dict(order),
                            **self._hot_universe_exemption_fields(
                                pending, status="rejected", end_reason="insufficient_available_isolated_margin",
                            ))
                self._record_pullback_risk_reject(pending, at, "insufficient_available_isolated_margin")
                return
            try:
                result = self.execution.open(order, stop_price=stop)
            except Exception as exc:
                reason = f"demo_execution_failed:{type(exc).__name__}"
                self._event("intent_rejected", reason=reason, intent=_intent_dict(order),
                            host=getattr(self.execution, "host", None),
                            **self._hot_universe_exemption_fields(
                                pending, status="rejected", end_reason=reason,
                            ))
                self._record_pullback_risk_reject(pending, at, reason)
                return
            live_position = result["position"]
            average = float(live_position["entry_price"])
            actual_quantity = float(live_position["quantity"])
            confirmed_stop = float(result["stop"]["stop_price"])
            position = Position(self.account.symbol, signal.side, actual_quantity, average, confirmed_stop, at,
                                signal.reason, order.regime.value, signal.source or MEAN_REVERSION)
            self.account.portfolio.positions[self.account.symbol] = position
            self.account.mark_prices[self.account.symbol] = average
            fee = result["fill"].get("fee_cost")
            fee = actual_quantity * average * self.fee_rate if fee is None else float(fee)
            self.account.active_entry_fee = fee
            self.account.active_entry_slippage = 0.0
            self.account.active_funding_paid = 0.0
            self.account.active_exit = {
                "enabled": bool(self.settings.exit_management.enabled),
                "initial_risk": abs(average - confirmed_stop), "best_price": average, "worst_price": average,
                "mae_history_complete": True, "observation_checkpoints": [],
                "protective_stop": confirmed_stop,
                **self._new_exit_state(pending, signal.side, actual_quantity, average, fee, confirmed_stop,
                                      proposal_enabled),
            }
            margin = float(live_position.get("initial_margin") or actual_quantity * average / self.settings.sizing.leverage)
            self.account.isolated_margin = {
                "initial_margin": margin, "maintenance_margin_rate": 0.0,
                "maintenance_amount": 0.0, "liquidation_fee_rate": self.fee_rate,
                "leverage": float(self.settings.sizing.leverage),
            }
            self.account.exchange_orders_by_symbol[self.account.symbol] = {
                "entry_order_id": result["fill"]["id"], "stop": result["stop"],
                "reason": signal.reason, "regime": order.regime.value,
                "intent": _intent_dict(order), "entry_at": at.isoformat(),
            }
            self._event("paper_fill", action="OPEN", side=signal.side.value, quantity=actual_quantity,
                        amount=actual_quantity, price=average, average=average, stop_price=confirmed_stop,
                        reason=signal.reason, regime=position.regime, source=signal.source,
                        signal_time=signal.timestamp.isoformat() if signal.timestamp else None,
                        orderId=result["fill"]["id"],
                        host=result["host"], execution_source="binance_usdm_demo", fee=fee,
                        **self._hot_universe_exemption_fields(
                            pending, status="filled", end_reason="pullback_filled",
                        ))
            if pending.get("execution_type") == "pullback":
                self._event("pending_entry_filled", signal_time=pending.get("signal_time"),
                            source=pending.get("source"),
                            confirmed_at=pending.get("confirmed_at"), filled_at=at.isoformat(),
                            **self._hot_universe_exemption_fields(
                                pending, status="filled", end_reason="pullback_filled",
                            ),
                            entry_delay_sec=max(0.0, (at - _dt(pending["confirmed_at"])).total_seconds()),
                            signal_to_fill_delay_sec=max(0.0, (at - _dt(pending["signal_time"])).total_seconds()),
                            reference_price=pending.get("reference_price"), pullback_price=pending.get("pullback_price"),
                            quote_at=pending.get("quote_at"), quote_received_at=pending.get("quote_received_at"),
                            quote_age_sec=pending.get("quote_age_sec"), quote_price=pending.get("quote_price"),
                            quote_source=pending.get("quote_source"),
                            execution_price=average, side=signal.side.value)
            self._refresh_exchange_snapshot()
            return
        position = Position(self.account.symbol, signal.side, quantity, fill, stop, at,
                            signal.reason, order.regime.value, signal.source or MEAN_REVERSION)
        fee = quantity * fill * self.fee_rate
        margin_spec = self.margin_spec or {"maintenance_margin_rate": 0.025, "maintenance_amount": 0.0,
                                          "liquidation_fee_rate": self.fee_rate}
        leverage = float(self.settings.sizing.leverage)
        initial_margin = quantity * fill / leverage
        if self.account.cash < initial_margin + fee:
            self._event("intent_rejected", reason="insufficient_available_isolated_margin",
                        required_margin=initial_margin, available_cash=self.account.cash,
                        signal_source=signal.source, **confirmation_fields)
            self._record_pullback_risk_reject(pending, at, "insufficient_available_isolated_margin")
            return
        self.account.portfolio.register_open(position, approval)
        self.account.cash -= initial_margin + fee
        self.account.isolated_margin = {
            "initial_margin": initial_margin,
            "maintenance_margin_rate": float(margin_spec["maintenance_margin_rate"]),
            "maintenance_amount": float(margin_spec.get("maintenance_amount", 0.0)),
            "liquidation_fee_rate": float(margin_spec["liquidation_fee_rate"]),
            "leverage": leverage,
        }
        self.account.active_entry_fee = fee
        self.account.active_entry_slippage = quantity * slip_per_unit
        self.account.active_funding_paid = 0.0
        self.account.active_exit = {
            "enabled": bool(self.settings.exit_management.enabled) and not is_trend,
            "initial_risk": abs(fill - stop),
            "best_price": fill,
            "worst_price": fill,
            "mae_history_complete": not is_trend,
            "observation_checkpoints": [],
            "invalidation_enabled": not is_trend,
            "policy": "proposal_v1" if proposal_enabled else None,
            "protective_stop": stop,
        }
        if is_trend:
            self.account.active_exit["trend_exit_state"] = initialize_trend_exit_state(
                fill, stop, pending.get("structure_break_price"),
                stop_mode=self.settings.trend_strategy.stop_mode,
            )
            self.account.active_exit.update(confirmation_fields)
        else:
            self.account.active_exit.update(
                self._new_exit_state(pending, signal.side, quantity, fill, fee, stop, proposal_enabled)
            )
        self._event("paper_fill", action="OPEN", symbol=self.account.symbol, side=signal.side.value,
                    quantity=quantity, price=fill, stop_price=stop, reason=signal.reason,
                    regime=position.regime, source=signal.source,
                    signal_time=signal.timestamp.isoformat() if signal.timestamp else None, fee=fee,
                    **self._hot_universe_exemption_fields(
                        pending, status="filled", end_reason="pullback_filled",
                    ),
                    **confirmation_fields)
        if pending.get("execution_type") == "pullback":
            self._event("pending_entry_filled", signal_time=pending.get("signal_time"),
                        source=pending.get("source"),
                        confirmed_at=pending.get("confirmed_at"), filled_at=at.isoformat(),
                        **self._hot_universe_exemption_fields(
                            pending, status="filled", end_reason="pullback_filled",
                        ),
                        entry_delay_sec=max(0.0, (at - _dt(pending["confirmed_at"])).total_seconds()),
                        signal_to_fill_delay_sec=max(0.0, (at - _dt(pending["signal_time"])).total_seconds()),
                        reference_price=pending.get("reference_price"), pullback_price=pending.get("pullback_price"),
                        quote_at=pending.get("quote_at"), quote_received_at=pending.get("quote_received_at"),
                        quote_age_sec=pending.get("quote_age_sec"), quote_price=pending.get("quote_price"),
                        quote_source=pending.get("quote_source"),
                        execution_price=fill, side=signal.side.value,
                        **confirmation_fields)

    def _pullback_permission_status(self, pending: dict[str, Any], bars_1h: pd.DataFrame,
                                    bars_4h: pd.DataFrame, *, step_size: float = 1e-8,
                                    minimum_quantity: float = 0.0) -> tuple[bool, str | None]:
        """Recheck the original completed-bar permit while its pullback is pending."""
        signal_time = _dt(pending["signal_time"])
        latest_close = bars_1h.index[-1] + HOUR
        if latest_close != pd.Timestamp(signal_time):
            return False, "entry_signal_bar_expired"

        if pending.get("source") == TREND:
            if not self._trend_execution_enabled:
                return False, "trend_execution_mode_not_allowed"
            position = self.account.position
            try:
                decision = evaluate_trend_entry(
                    self.account.symbol, bars_1h, bars_4h, self.settings, now=signal_time,
                    market_risk_state=self.market_risk_state,
                    position_source=position.source if position is not None else None,
                    has_open_position=position is not None,
                    equity=self._current_equity(), step_size=step_size,
                    minimum_quantity=minimum_quantity,
                    existing_gross_notional=self.account.portfolio.gross_notional,
                    gross_notional_cap_pct=self.gross_notional_cap_pct,
                )
            except (ArithmeticError, IndexError, TypeError, ValueError) as exc:
                return False, f"trend_entry_recheck_failed:{type(exc).__name__}"
            side = Side(pending["side"])
            if (decision.signal.action is not Action.OPEN or decision.signal.side is not side
                    or decision.signal.reason != pending.get("reason")):
                return False, f"entry_permission_changed:{decision.signal.reason}"
            return True, None
        if pending.get("source") not in (None, MEAN_REVERSION):
            return False, "invalid_strategy_source"

        regimes = classify_regimes(bars_4h, self.settings)
        regime_closes = regimes.index + pd.Timedelta(14_400_000_000_000, unit="ns")
        usable = np.flatnonzero(regime_closes <= latest_close)
        if not len(usable):
            return False, "entry_regime_unavailable"
        rindex = int(usable[-1])
        regime = _snapshot(regimes.iloc[rindex], regimes.index[rindex])
        if regime is None:
            return False, "entry_regime_unavailable"

        flags = evaluate_time_filter(latest_close.to_pydatetime(), bars_1h.tail(48), self.settings)
        regime_row = regimes.iloc[rindex]
        swing = bars_1h.tail(self.settings.stops.swing_lookback_1h)
        signal = generate_signal(regime, bars_1h.iloc[-1], flags, self.settings,
                                 position=None, bars_1h=bars_1h, bars_4h=bars_4h,
                                 entry_reference_price=float(bars_1h.iloc[-1]["close"]),
                                 entry_atr=float(atr(bars_1h, self.settings.lookback.atr_len).iloc[-1]),
                                 range_low=regime_row.get("range_low"), range_high=regime_row.get("range_high"),
                                 structure_low=float(swing["low"].min()) if len(swing) else None,
                                 structure_high=float(swing["high"].max()) if len(swing) else None)
        signal = with_signal_context(signal, symbol=self.account.symbol, source=MEAN_REVERSION,
                                     timestamp=_dt(pending["signal_time"]))
        side = Side(pending["side"])
        if (signal.action is not Action.OPEN or signal.side is not side
                or signal.reason != pending.get("reason")):
            return False, f"entry_permission_changed:{signal.reason}"
        return True, None

    def _advance_pullback_pending(self, current_price: float, now: datetime, funding_rate: float | None,
                                  *, step_size: float, minimum_quantity: float,
                                  permission_valid: bool = True, permission_detail: str | None = None,
                                  quote_fresh: bool = True, quote_metadata: dict[str, Any] | None = None) -> bool:
        pending = self.account.pending
        if pending is None or pending.get("execution_type") != "pullback":
            return False
        if self.gross_notional_cap_pct <= 0:
            self.account.pending = None
            self._event("pending_entry_cancelled", reason="market_risk_paused",
                        signal_time=pending.get("signal_time"), cancelled_at=now.isoformat(),
                        source=pending.get("source") or MEAN_REVERSION,
                        **self._hot_universe_exemption_fields(
                            pending, status="cancelled", end_reason="market_risk_paused",
                        ),
                        **self._trend_confirmation_fields(pending))
            return True
        pending_source = pending.get("source") or MEAN_REVERSION
        if pending_source not in {MEAN_REVERSION, TREND} or (
                pending_source == TREND and not self._trend_execution_enabled):
            reason = "trend_execution_mode_not_allowed" if pending_source == TREND else "invalid_strategy_source"
            self.account.pending = None
            self._event("pending_entry_cancelled", reason=reason,
                        signal_time=pending.get("signal_time"), source=pending_source,
                        cancelled_at=now.isoformat(),
                        **self._hot_universe_exemption_fields(
                            pending, status="cancelled", end_reason=reason,
                        ),
                        **self._trend_confirmation_fields(pending))
            return True
        held_position = self.account.position
        if (held_position is not None
                and held_position.source != (pending.get("source") or MEAN_REVERSION)):
            self.account.pending = None
            self._event("pending_entry_cancelled", reason="position_owned_by_other_strategy",
                        signal_time=pending.get("signal_time"),
                        signal_source=pending.get("source") or MEAN_REVERSION,
                        position_source=held_position.source, cancelled_at=now.isoformat(),
                        **self._hot_universe_exemption_fields(
                            pending, status="cancelled", end_reason="position_owned_by_other_strategy",
                        ),
                        **self._trend_confirmation_fields(pending))
            return True
        side = Side(pending["side"])
        signal_time = _dt(pending["signal_time"])
        expires_at = _dt(pending["expires_at"])
        exemption_expired = bool(pending.get("hot_universe_exemption")) and now >= expires_at
        if now >= expires_at and (self.settings.execution.timeout_action == "cancel" or exemption_expired):
            self.account.pending = None
            self._event("pending_entry_cancelled", reason="timeout", signal_time=signal_time.isoformat(),
                        cancelled_at=now.isoformat(), expires_at=expires_at.isoformat(),
                        **({"detail": "hot_universe_exemption_expired"} if exemption_expired else {}),
                        source=pending_source,
                        **self._hot_universe_exemption_fields(
                            pending,
                            status="expired" if exemption_expired else "cancelled",
                            end_reason="plan_expired" if exemption_expired else "timeout",
                        ),
                        **self._trend_confirmation_fields(pending))
            return True

        if not permission_valid:
            self.account.pending = None
            self._event("pending_entry_cancelled", reason="invalidated", detail=permission_detail,
                        signal_time=signal_time.isoformat(), cancelled_at=now.isoformat(),
                        source=pending_source,
                        **self._hot_universe_exemption_fields(
                            pending, status="cancelled", end_reason=permission_detail or "invalidated",
                        ),
                        **self._trend_confirmation_fields(pending))
            return True

        if pending_source == TREND:
            structure = pending.get("structure_break_price")
            rejection = None
        else:
            structure_key = "entry_structure_low" if side is Side.LONG else "entry_structure_high"
            structure = pending.get(structure_key)
            rejection = entry_rejection_reason(
                side, float(current_price), float(pending["atr"]),
                pending.get("entry_range_low"), pending.get("entry_range_high"),
                pending.get("entry_structure_low"), pending.get("entry_structure_high"), self.settings,
            )
        structure_broken = (structure is not None and
                            (current_price < float(structure) if side is Side.LONG
                             else current_price > float(structure)))
        if structure_broken or rejection is not None:
            self.account.pending = None
            detail = "entry_structure_broken" if structure_broken else rejection
            event_reason = (detail if detail in {
                "long_rejected_below_range", "short_rejected_above_range", "invalid_on_entry"
            } else "invalidated")
            self._event("pending_entry_cancelled", reason=event_reason, detail=detail,
                        signal_time=signal_time.isoformat(), cancelled_at=now.isoformat(), price=current_price,
                        source=pending_source,
                        **self._hot_universe_exemption_fields(
                            pending, status="cancelled", end_reason=detail,
                        ),
                        **self._trend_confirmation_fields(pending),
                        structure_price=float(structure) if structure_broken else None,
                        range_boundary=(pending.get("entry_range_low") if side is Side.LONG
                                        else pending.get("entry_range_high")),
                        entry_structure_low=pending.get("entry_structure_low"),
                        entry_structure_high=pending.get("entry_structure_high"))
            return True

        if now >= expires_at:
            if not quote_fresh:
                self.account.pending = None
                self._event("pending_entry_cancelled", reason="risk_reject", detail="fresh_quote_unavailable_for_fallback",
                            signal_time=signal_time.isoformat(), cancelled_at=now.isoformat(),
                            expires_at=expires_at.isoformat(), source=pending_source,
                            **self._hot_universe_exemption_fields(
                                pending, status="cancelled", end_reason="fresh_quote_unavailable_for_fallback",
                            ),
                            **self._trend_confirmation_fields(pending))
                return True
            self._event("pending_entry_timeout_fallback", signal_time=signal_time.isoformat(),
                        at=now.isoformat(), price=current_price, source=pending_source,
                        **self._trend_confirmation_fields(pending))
            pending.update(quote_metadata or {})
            pending["execute_at"] = now.isoformat()
            self._execute_pending(current_price, now, funding_rate,
                                  step_size=step_size, minimum_quantity=minimum_quantity)
            return True

        if not quote_fresh:
            return False
        pullback_price = float(pending["pullback_price"])
        reached = (current_price <= pullback_price if side is Side.LONG else current_price >= pullback_price)
        if not reached or now <= _dt(pending["confirmed_at"]):
            return False
        pending.update(quote_metadata or {})
        pending["execute_at"] = now.isoformat()
        self._execute_pending(current_price, now, funding_rate,
                              step_size=step_size, minimum_quantity=minimum_quantity)
        return True

    def monitor_tick(self, *, current_price: float, now: datetime, mark_price: float | None = None,
                     observation_context: dict | None = None) -> bool:
        """Update portfolio marks, stop checks, and portfolio-wide daily risk state."""
        now = _dt(now)
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError("current price must be finite and positive")
        previous_mark = self.account.mark_prices.get(self.account.symbol)
        self.account.mark_prices[self.account.symbol] = float(current_price)
        changed = previous_mark != float(current_price)
        position = self.account.position
        if position is not None:
            if self.account.active_exit is None:
                # Legacy positions cannot recover favorable/adverse excursions from before this run.
                self.account.active_exit = {
                    "enabled": False, "initial_risk": abs(position.entry_price - position.stop_price),
                    "best_price": position.entry_price, "worst_price": current_price,
                    "mae_history_complete": False, "observation_checkpoints": [],
                    "protective_stop": position.stop_price,
                }
            active_exit = self.account.active_exit
            active_exit.setdefault("observation_checkpoints", [])
            if active_exit.get("enabled"):
                active_exit["best_price"] = (
                    max(float(active_exit["best_price"]), current_price) if position.side is Side.LONG
                    else min(float(active_exit["best_price"]), current_price)
                )
                worst = active_exit.get("worst_price", position.entry_price)
                active_exit["worst_price"] = (
                    min(float(worst), current_price) if position.side is Side.LONG
                    else max(float(worst), current_price)
                )
                active_exit.setdefault("mae_history_complete", False)
            held_seconds = max(0.0, (now - position.opened_at).total_seconds())
            sign = 1 if position.side is Side.LONG else -1
            risk = float(active_exit.get("initial_risk") or abs(position.entry_price - position.stop_price))
            managed = bool(active_exit.get("enabled")) and risk > 0 and active_exit.get("best_price") is not None
            mfe_r = sign * (float(active_exit["best_price"]) - position.entry_price) / risk if managed else None
            mae_complete = managed and bool(active_exit.get("mae_history_complete"))
            worst = active_exit.get("worst_price")
            mae_r = max(0.0, -sign * (float(worst) - position.entry_price) / risk) if mae_complete and worst is not None else None
            exit_fill, _ = _apply_slippage(current_price, position.side, is_entry=False, slippage_rate=self.slippage_rate)
            estimated_net = (
                position.quantity * (exit_fill - position.entry_price) * sign
                - self.account.active_entry_fee - position.quantity * exit_fill * self.fee_rate
                - self.account.active_funding_paid
            )
            protective_stop = float(active_exit.get("protective_stop", position.stop_price))
            stop_fill, _ = _apply_slippage(protective_stop, position.side, is_entry=False, slippage_rate=self.slippage_rate)
            estimated_stop_net = (
                position.quantity * (stop_fill - position.entry_price) * sign
                - self.account.active_entry_fee - position.quantity * stop_fill * self.fee_rate
                - self.account.active_funding_paid
            )
            policy = self.settings.exit_management
            checkpoints = (
                (float(policy.time_check_hours), "12h", float(policy.time_min_progress_r), True),
                (float(policy.time_exit_hours), "24h", float(policy.breakeven_trigger_r), False),
            )
            for hours, label, progress_limit, requires_nonpositive_net in checkpoints:
                if hours in active_exit["observation_checkpoints"] or held_seconds < hours * 3600:
                    continue
                scheduled_at = position.opened_at + timedelta(hours=hours)
                condition_met = None if mfe_r is None else mfe_r < progress_limit
                if condition_met is not None and requires_nonpositive_net:
                    condition_met = condition_met and estimated_net <= 0
                self.account.events.append({
                    "event": "management_observation_checkpoint", "at": now.isoformat(),
                    "symbol": position.symbol, "side": position.side.value,
                    "opened_at": position.opened_at.isoformat(), "checkpoint": label,
                    "checkpoint_hours": hours, "scheduled_at": scheduled_at.isoformat(),
                    "checkpoint_delay_seconds": max(0.0, held_seconds - hours * 3600),
                    "held_hours": held_seconds / 3600, "entry_reason": position.reason,
                    "entry_regime": position.regime, "price": float(current_price),
                    "current_r": sign * (current_price - position.entry_price) / risk if risk > 0 else None,
                    "initial_risk": risk if risk > 0 else None,
                    "mfe_r": mfe_r, "mfe_available": managed, "mae_r": mae_r,
                    "mae_available": mae_complete, "estimated_net_if_closed": estimated_net,
                    "protective_stop": protective_stop,
                    "stop_estimated_net_if_triggered": estimated_stop_net,
                    "condition_met": condition_met,
                    "time_exit_enabled": bool(policy.time_exit_enabled),
                    **(observation_context or {}),
                })
                active_exit["observation_checkpoints"].append(hours)
                changed = True
        if position is not None:
            margin = getattr(self.account, "isolated_margin", None)
            if margin and mark_price is not None and math.isfinite(mark_price) and mark_price > 0:
                direction = 1 if position.side is Side.LONG else -1
                isolated_balance = float(margin["initial_margin"]) + position.quantity * (mark_price - position.entry_price) * direction
                maintenance = max(0.0, position.quantity * mark_price * float(margin["maintenance_margin_rate"])
                                  - float(margin.get("maintenance_amount", 0.0)))
                if isolated_balance <= maintenance and self.execution is None:
                    if self._close(current_price, now, "liquidation_mark_trigger", liquidation=True):
                        changed = True
                        position = None
        if position is not None:
            active_exit = self.account.active_exit
            stop = float(active_exit.get("protective_stop", position.stop_price))
            structure_levels = self._structure_invalidation_levels(position, active_exit)
            hit = (position.side is Side.LONG and current_price <= stop) or (
                position.side is Side.SHORT and current_price >= stop)
            if hit:
                if structure_levels is not None:
                    self._event("structure_invalidation_comparison", reason="stop_hit_public_price",
                                observed_price=current_price, **structure_levels)
                if self._close(current_price, now, "stop_hit_public_price"):
                    changed = True
                    position = None
            elif (active_exit.get("invalidation_enabled")
                  and (invalidation_reason := self._price_invalidation_reason(position, active_exit, current_price))):
                if structure_levels is not None and invalidation_reason in {
                    "long_structure_invalid", "short_structure_invalid"
                }:
                    self._event("structure_invalidation_comparison", reason=invalidation_reason,
                                observed_price=current_price, **structure_levels)
                if self._close(current_price, now, invalidation_reason):
                    changed = True
                    position = None
            elif self._proposal_v1():
                try:
                    changed = self._manage_proposal_tick(current_price, now) or changed
                except Exception as exc:
                    if self.execution is not None:
                        self.account.execution_status = "inconsistent"
                    self._event("proposal_exit_management_failed", reason=type(exc).__name__,
                                frames=_safe_exception_frames(exc))
            elif active_exit.get("enabled"):
                side = position.side
                favorable = (
                    float(active_exit["best_price"]) - position.entry_price if side is Side.LONG
                    else position.entry_price - float(active_exit["best_price"])
                )
                risk = float(active_exit["initial_risk"])
                policy = self.settings.exit_management
                if policy.time_exit_enabled:
                    progress_r = favorable / risk
                    held_hours = (now - position.opened_at).total_seconds() / 3600
                    estimated_exit, _ = _apply_slippage(
                        current_price, side, is_entry=False, slippage_rate=self.slippage_rate
                    )
                    direction = 1 if side is Side.LONG else -1
                    net_if_closed = (
                        position.quantity * (estimated_exit - position.entry_price) * direction
                        - self.account.active_entry_fee - position.quantity * estimated_exit * self.fee_rate
                        - self.account.active_funding_paid
                    )
                    reason = None
                    if (held_hours >= policy.time_check_hours
                            and progress_r < policy.time_min_progress_r and net_if_closed <= 0):
                        reason = "time_stop_12h_no_progress"
                    elif (held_hours >= policy.time_exit_hours
                            and progress_r < policy.breakeven_trigger_r):
                        reason = "time_exit_24h_unprotected"
                    if reason:
                        if self._close(current_price, now, reason):
                            changed = True
                            position = None
                if position is None:
                    pass
                else:
                    updated_stop = stop
                    if favorable >= policy.breakeven_trigger_r * risk:
                        if side is Side.LONG:
                            breakeven = position.entry_price * (1 + self.fee_rate) / ((1 - self.slippage_rate) * (1 - self.fee_rate))
                            updated_stop = max(updated_stop, breakeven)
                        else:
                            breakeven = position.entry_price * (1 - self.fee_rate) / ((1 + self.slippage_rate) * (1 + self.fee_rate))
                            updated_stop = min(updated_stop, breakeven)
                    if favorable >= policy.trailing_trigger_r * risk:
                        trail_distance = policy.trailing_atr_mult * risk / self.settings.signals.stop_atr_mult
                        trailing = (
                            float(active_exit["best_price"]) - trail_distance if side is Side.LONG
                            else float(active_exit["best_price"]) + trail_distance
                        )
                        updated_stop = max(updated_stop, trailing) if side is Side.LONG else min(updated_stop, trailing)
                    if updated_stop != stop:
                        if self.execution is None:
                            active_exit["protective_stop"] = updated_stop
                            changed = True
                        else:
                            orders = self.account.exchange_orders_by_symbol.get(self.account.symbol, {})
                            try:
                                orders["stop"] = self.execution.replace_stop(
                                    position, stop=orders.get("stop"), stop_price=updated_stop,
                                )
                                active_exit["protective_stop"] = float(orders["stop"]["stop_price"])
                                changed = True
                            except Exception as exc:
                                self.account.execution_status = "inconsistent"
                                self._event("protective_stop_update_failed", reason=type(exc).__name__,
                                            frames=_safe_exception_frames(exc))
                        if active_exit.get("protective_stop") != stop:
                            self._event("protective_stop_updated", side=side.value,
                                        stop_price=active_exit["protective_stop"],
                                        best_price=active_exit["best_price"], favorable_r=favorable / risk)
        risk_state_before = (
            self.account.portfolio.trading_date,
            self.account.portfolio.day_start_equity,
            self.account.portfolio.daily_halted,
        )
        self.account.portfolio.observe_equity(max(1e-12, self._current_equity()), now, self.settings)
        risk_state_after = (
            self.account.portfolio.trading_date,
            self.account.portfolio.day_start_equity,
            self.account.portfolio.daily_halted,
        )
        return changed or risk_state_before != risk_state_after

    def check_max_hold(self, current_price: float, now: datetime) -> bool:
        """Apply the maximum-hold insurance on the runner poll cadence."""
        position = self.account.position
        if position is None or position.source == TREND:
            return False
        exit_state = self.account.active_exit or {}
        max_hold_hours = effective_max_hold_hours(
            position.side, position.entry_price, position.stop_price, exit_state, self.settings.exits,
        )
        if max_hold_hours is None or max_hold_hours <= 0:
            return False
        pending = self.account.pending
        if pending and (pending.get("action") == Action.CLOSE.value or pending.get("close_first")):
            return False
        now = _dt(now)
        opened_at = _dt(position.opened_at)
        held_hours = max(0.0, (now - opened_at).total_seconds() / 3600)
        if held_hours < max_hold_hours:
            return False
        if self._close(current_price, now, "max_hold_24h"):
            self.account.last_processed_close = pd.Timestamp(now).floor("h").isoformat()
            logger.warning("MAX_HOLD_EXIT symbol={} held_hours={} limit_hours={}",
                           self.account.symbol, round(held_hours, 3), max_hold_hours)
            return True
        return False

    def close_on_structure_signal(
        self,
        *,
        candle_close: float,
        current_price: float,
        now: datetime,
        close_key: str,
    ) -> tuple[bool, bool]:
        """Return (checked, closed) after applying the existing closed-bar exit."""
        position = self.account.position
        active_state = self.account.active_exit or {}
        if (position is None or not self._proposal_v1()
                or not self.settings.exit_proposal_v1.structure_enabled):
            return False, False

        reason = proposal_structure_signal_reason(position.side, candle_close, active_state)
        if reason is None:
            return True, False

        regime = active_state.get("latest_regime", position.regime)
        if not self._close(current_price, now, reason):
            return False, False

        self.account.last_processed_close = close_key
        self._event("proposal_structure_exit", reason=reason,
                    signal_close=candle_close, regime=regime)
        return True, True

    def update(self, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame, *, current_open_at: datetime,
               current_open: float, current_price: float, funding_rate: float | None,
               now: datetime, allow_entries: bool = True, step_size: float = 1e-8,
               minimum_quantity: float = 0.0, entry_not_before: datetime | None = None,
               symbol_in_hot_universe: bool | None = None,
               allowed_entry_sources: set[str] | None = None,
               entry_execution_allowed: bool = True,
               mark_price: float | None = None, wait_pullback: bool = False,
               pullback_quote_fresh: bool = True,
               pullback_quote_metadata: dict[str, Any] | None = None,
               skip_structure_signal_close_key: str | None = None,
               gross_notional_cap_pct: float | None = None,
               mean_reversion_only: bool = False,
               symbol_regime_state: str | None = None,
                symbol_regime_evidence: dict[str, Any] | None = None,
               market_risk_state: str = "NORMAL") -> bool:
        """Consume completed bars once, fill queued signals at next hour open, and watch stops."""
        validate_ohlcv(bars_1h)
        validate_ohlcv(bars_4h)
        current_open_at, now = _dt(current_open_at), _dt(now)
        market_risk_state = str(market_risk_state or "PAUSE_NEW").upper()
        symbol_regime_evidence = dict(symbol_regime_evidence or {})
        if market_risk_state not in {"NORMAL", "RISK_ELEVATED", "PAUSE_NEW"}:
            raise ValueError("market risk state is invalid")
        self.market_risk_state = market_risk_state
        self.market_risk_multiplier = (0.0 if market_risk_state == "PAUSE_NEW" else
                                       self.settings.market_risk.elevated_size_multiplier
                                       if market_risk_state == "RISK_ELEVATED" else 1.0)
        state_budget = self.settings.sizing.max_gross_notional_pct * self.market_risk_multiplier
        self.gross_notional_cap_pct = (0.0 if market_risk_state == "PAUSE_NEW" else
                                       min(state_budget, gross_notional_cap_pct)
                                       if gross_notional_cap_pct is not None else state_budget)
        if not math.isfinite(self.gross_notional_cap_pct) or self.gross_notional_cap_pct < 0:
            raise ValueError("gross notional budget must be finite and non-negative")
        if not all(math.isfinite(v) and v > 0 for v in (current_open, current_price)):
            raise ValueError("current prices must be finite and positive")
        changed = self.monitor_tick(current_price=current_price, mark_price=mark_price, now=now)
        current_open_key = pd.Timestamp(current_open_at).isoformat()
        permission_valid, permission_detail = True, None
        pending = self.account.pending
        timeout_cancel_due = (pending is not None and pending.get("execution_type") == "pullback"
                              and now >= _dt(pending["expires_at"])
                              and self.settings.execution.timeout_action == "cancel")
        if (pending is not None and pending.get("execution_type") == "pullback"
                and not timeout_cancel_due):
            permission_valid, permission_detail = self._pullback_permission_status(
                pending, bars_1h, bars_4h, step_size=step_size,
                minimum_quantity=minimum_quantity,
            )
        if self._advance_pullback_pending(current_price, now, funding_rate,
                                          step_size=step_size, minimum_quantity=minimum_quantity,
                                          permission_valid=permission_valid, permission_detail=permission_detail,
                                          quote_fresh=pullback_quote_fresh,
                                          quote_metadata=pullback_quote_metadata):
            changed = True
        if self.account.last_observed_open != current_open_key:
            if not (self.account.pending and self.account.pending.get("execution_type") == "pullback"):
                self._execute_pending(current_price, current_open_at, funding_rate,
                                      step_size=step_size, minimum_quantity=minimum_quantity)
            self.account.last_observed_open = current_open_key
            changed = True

        close_at = bars_1h.index[-1] + HOUR
        close_key = close_at.isoformat()
        if close_at.to_pydatetime() <= now and self.account.last_processed_close != close_key:
            position = self.account.position
            if position is not None and position.source == TREND:
                active_state = self.account.active_exit or {}
                trend_state = active_state.get("trend_exit_state")
                if trend_state is None:
                    trend_state = initialize_trend_exit_state(
                        position.entry_price, position.stop_price,
                        active_state.get("structure_break_price"),
                        stop_mode=self.settings.trend_strategy.stop_mode,
                    )
                try:
                    decision = evaluate_trend_exit(
                        position, bars_1h, self.settings, now=now,
                        exit_state=trend_state, current_price=float(current_price),
                    )
                except (ArithmeticError, IndexError, TypeError, ValueError) as exc:
                    self.account.last_processed_close = close_key
                    self._event("trend_exit_evaluation_failed", source=TREND,
                                reason=f"{type(exc).__name__}:{str(exc)[:200]}",
                                signal_time=close_key)
                    return True
                trend_exit_state = dict(decision.exit_state)
                previous_stop = float(active_state.get("protective_stop", position.stop_price))
                previous_stop_reason = str(
                    trend_state.get("active_stop_reason", "structure_stop") if trend_state else "structure_stop"
                )
                requested_stop = float(decision.active_stop_price)
                stop_improved = (requested_stop > previous_stop if position.side is Side.LONG
                                 else requested_stop < previous_stop)
                active_stop = previous_stop
                if stop_improved:
                    stop_reason = str(trend_exit_state.get("active_stop_reason", "trend_stop"))
                    if self._update_protective_stop(
                            requested_stop, f"trend_{stop_reason}", current_price=float(current_price)):
                        active_stop = float(active_state["protective_stop"])
                        trend_exit_state["active_stop_price"] = active_stop
                    elif decision.signal.action is not Action.CLOSE:
                        # Keep the evaluator's excursion cursor but never claim a tighter stop
                        # than the local Paper stop or the Demo exchange has confirmed.
                        trend_exit_state["active_stop_price"] = previous_stop
                        trend_exit_state["active_stop_reason"] = previous_stop_reason
                else:
                    trend_exit_state["active_stop_price"] = previous_stop
                    trend_exit_state["active_stop_reason"] = previous_stop_reason
                active_state["trend_exit_state"] = trend_exit_state
                active_state["protective_stop"] = active_stop
                active_state["initial_risk"] = float(trend_exit_state["initial_risk"])
                active_state["best_price"] = float(trend_exit_state.get(
                    "best_favorable_price", position.entry_price))
                self.account.active_exit = active_state
                self.account.last_processed_close = close_key
                confirmation_fields = {
                    "confirm_bars": active_state.get("confirm_bars"),
                    "confirmation_pass": active_state.get("confirmation_pass"),
                    "confirmation_bar_time": active_state.get("confirmation_bar_time"),
                }
                self._event("trend_exit_evaluated", source=TREND,
                            action=decision.signal.action.value, reason=decision.signal.reason,
                            signal_time=(decision.signal.timestamp.isoformat()
                                         if decision.signal.timestamp else close_key),
                            active_stop_price=active_stop,
                            requested_stop_price=requested_stop,
                            evidence=decision.evidence, **confirmation_fields)
                if decision.signal.action is Action.CLOSE:
                    if self._close(current_price, now, decision.signal.reason):
                        return True
                return True
            regimes = classify_regimes(bars_4h, self.settings)
            regime_closes = regimes.index + pd.Timedelta(14_400_000_000_000, unit="ns")
            usable = np.flatnonzero(regime_closes <= close_at)
            if len(usable):
                rindex = int(usable[-1])
                regime = _snapshot(regimes.iloc[rindex], regimes.index[rindex])
                one_hour_atr = atr(bars_1h, self.settings.lookback.atr_len).iloc[-1]
                if regime is not None and np.isfinite(one_hour_atr) and one_hour_atr > 0:
                    position = self.account.position
                    if position is not None:
                        active_state = self.account.active_exit or {}
                        active_state.setdefault("invalidation_enabled", True)
                        active_state.setdefault("entry_atr", float(one_hour_atr))
                        active_state.setdefault("entry_range_low", None)
                        active_state.setdefault("entry_range_high", None)
                        active_state.setdefault("no_progress_count", 0)
                        self.account.active_exit = active_state
                        active_state["latest_regime"] = regime.regime.value
                        structural_reason = None
                        candle_close = float(bars_1h.iloc[-1]["close"])
                        exits = self.settings.exits
                        structure_levels = self._structure_invalidation_levels(position, active_state)
                        if position.side is Side.LONG:
                            if (structure_levels is not None
                                    and structure_levels["trigger_source"] == "structure_invalid"
                                    and candle_close < structure_levels["invalid_price"]):
                                structural_reason = "long_structure_invalid"
                            elif (exits.regime_fade_enabled and position.regime == Regime.LOW.value
                                  and regime.regime is Regime.SIDEWAYS
                                  and candle_close - position.entry_price
                                  < exits.regime_fade_min_atr * float(active_state["entry_atr"])):
                                structural_reason = "long_regime_faded"
                        else:
                            if (structure_levels is not None
                                    and structure_levels["trigger_source"] == "structure_invalid"
                                    and candle_close > structure_levels["invalid_price"]):
                                structural_reason = "short_structure_invalid"
                            elif (exits.regime_fade_enabled and position.regime == Regime.HIGH.value
                                  and regime.regime is Regime.SIDEWAYS
                                  and position.entry_price - candle_close
                                  < exits.regime_fade_min_atr * float(active_state["entry_atr"])):
                                structural_reason = "short_regime_faded"
                        if structural_reason:
                            if structure_levels is not None and structural_reason in {
                                "long_structure_invalid", "short_structure_invalid"
                            }:
                                self._event("structure_invalidation_comparison", reason=structural_reason,
                                            observed_price=candle_close, **structure_levels)
                            if self._close(current_price, now, structural_reason):
                                self.account.last_processed_close = close_key
                                self._event("signal_invalidation_exit", reason=structural_reason,
                                            signal_close=candle_close, regime=regime.regime.value)
                                return True

                        previous_close = active_state.get("no_progress_last_close")
                        if previous_close is not None:
                            moved_favorably = (candle_close > float(previous_close) if position.side is Side.LONG
                                               else candle_close < float(previous_close))
                            active_state["no_progress_count"] = (
                                0 if moved_favorably else int(active_state.get("no_progress_count", 0)) + 1
                            )
                        active_state["no_progress_last_close"] = candle_close
                        best_price = float(active_state.get("best_price", position.entry_price))
                        progress_atr = ((best_price - position.entry_price) if position.side is Side.LONG
                                        else (position.entry_price - best_price)) / max(float(active_state["entry_atr"]), 1e-12)
                        if (int(active_state.get("no_progress_count", 0)) >= exits.no_progress_bars
                                and progress_atr < exits.no_progress_min_atr
                                and not active_state.get("no_progress_reduced")):
                            if exits.no_progress_action == "close":
                                if self._close(current_price, now, "no_progress"):
                                    self.account.last_processed_close = close_key
                                    return True
                            elif self._proposal_partial_close(position.quantity * 0.5, current_price, now, "no_progress"):
                                self.account.active_exit["no_progress_reduced"] = True
                                active_state = self.account.active_exit
                        if self._proposal_v1() and self.settings.exit_proposal_v1.structure_enabled:
                            signal_reason = None
                            if close_key != skip_structure_signal_close_key:
                                signal_reason = proposal_structure_signal_reason(
                                    position.side, candle_close, active_state,
                                )
                            if signal_reason is not None:
                                structural_reason = signal_reason
                            elif position.side is Side.LONG:
                                if (active_state.get("midline_armed")
                                      and active_state.get("last_signal_close") is not None
                                      and active_state.get("entry_midpoint") is not None
                                      and float(active_state["last_signal_close"]) >= float(active_state["entry_midpoint"])
                                      and candle_close < float(active_state["entry_midpoint"])):
                                    structural_reason = "structure_midline_reversal"
                                elif (position.regime == Regime.LOW.value
                                      and regime.regime in (Regime.SIDEWAYS, Regime.MID)
                                      and candle_close - position.entry_price < 0.5 * float(active_state["entry_atr"])):
                                    structural_reason = "structure_regime_stall"
                                if (active_state.get("entry_midpoint") is not None
                                        and active_state.get("last_signal_close") is not None
                                        and float(active_state["last_signal_close"]) < float(active_state["entry_midpoint"])
                                        and candle_close >= float(active_state["entry_midpoint"])):
                                    active_state["midline_armed"] = True
                            else:
                                if (active_state.get("midline_armed")
                                      and active_state.get("last_signal_close") is not None
                                      and active_state.get("entry_midpoint") is not None
                                      and float(active_state["last_signal_close"]) <= float(active_state["entry_midpoint"])
                                      and candle_close > float(active_state["entry_midpoint"])):
                                    structural_reason = "structure_midline_reversal"
                                elif (position.regime == Regime.HIGH.value
                                      and regime.regime in (Regime.SIDEWAYS, Regime.MID)
                                      and position.entry_price - candle_close < 0.5 * float(active_state["entry_atr"])):
                                    structural_reason = "structure_regime_stall"
                                if (active_state.get("entry_midpoint") is not None
                                        and active_state.get("last_signal_close") is not None
                                        and float(active_state["last_signal_close"]) > float(active_state["entry_midpoint"])
                                        and candle_close <= float(active_state["entry_midpoint"])):
                                    active_state["midline_armed"] = True
                            active_state["last_signal_close"] = candle_close
                        if structural_reason:
                            closed = self._close(current_price, now, structural_reason)
                            if closed:
                                self.account.last_processed_close = close_key
                                self._event("proposal_structure_exit", reason=structural_reason,
                                            signal_close=candle_close, regime=regime.regime.value)
                                return True
                    flags = evaluate_time_filter(close_at.to_pydatetime(), bars_1h.tail(48), self.settings)
                    position = self.account.position
                    regime_row = regimes.iloc[rindex]
                    raw_low, raw_high = regime_row.get("range_low"), regime_row.get("range_high")
                    range_low = float(raw_low) if raw_low is not None and np.isfinite(raw_low) else None
                    range_high = float(raw_high) if raw_high is not None and np.isfinite(raw_high) else None
                    swing = bars_1h.tail(self.settings.stops.swing_lookback_1h)
                    structure_low = float(swing["low"].min()) if len(swing) else None
                    structure_high = float(swing["high"].max()) if len(swing) else None
                    route_state = "RANGE" if mean_reversion_only else symbol_regime_state or "RANGE"
                    position_source = position.source if position is not None else None
                    route = resolve_strategy_route(
                        self.account.symbol, route_state, market_risk_state,
                        position_source=position_source, at=close_at.to_pydatetime(),
                        trend_strategy_ready=self._trend_execution_enabled,
                    )
                    if symbol_regime_evidence:
                        logger.info("Symbol regime routing evidence {}", json.dumps({
                            "symbol": self.account.symbol,
                            "state": route_state,
                            "assessment_reason": symbol_regime_evidence.get("assessment_reason"),
                            "last_closed_bar_at": symbol_regime_evidence.get("last_closed_bar_at"),
                            "condition_checks": symbol_regime_evidence.get("condition_checks", {}),
                            "condition_failures": symbol_regime_evidence.get("condition_failures", {}),
                            "near_range": symbol_regime_evidence.get("near_range", False),
                            "near_trend": symbol_regime_evidence.get("near_trend", False),
                            "near_miss_conditions": symbol_regime_evidence.get("near_miss_conditions", {}),
                            "range_confirmation_candidates": symbol_regime_evidence.get(
                                "range_confirmation_candidates", []),
                            "range_confirmation_current_streak": symbol_regime_evidence.get(
                                "range_confirmation_current_streak", 0),
                            "near_range_structure_only": symbol_regime_evidence.get(
                                "near_range_structure_only", False),
                            "range_structure_shadow": symbol_regime_evidence.get(
                                "range_structure_shadow", {}),
                        }, ensure_ascii=False, sort_keys=True))
                    if (route_state == "UNCERTAIN"
                            and symbol_regime_evidence.get("near_range_structure_only")
                            and position is None and self.account.pending is None):
                        try:
                            shadow = self._evaluate_near_range_structure_shadow(
                                regime=regime, candle_1h=bars_1h.iloc[-1],
                                bars_1h=bars_1h, bars_4h=bars_4h, flags=flags,
                                reference_price=float(current_price), atr_value=float(one_hour_atr),
                                range_low=range_low, range_high=range_high,
                                structure_low=structure_low, structure_high=structure_high,
                                now=close_at.to_pydatetime(), funding_rate=funding_rate,
                                market_risk_state=market_risk_state, step_size=step_size,
                                minimum_quantity=minimum_quantity,
                                allowed_entry_sources=allowed_entry_sources,
                                allow_entries=allow_entries,
                                entry_execution_allowed=entry_execution_allowed,
                                symbol_regime_evidence=symbol_regime_evidence,
                            )
                        except Exception as exc:
                            shadow = {
                                "event": "near_range_structure_shadow",
                                "shadow_only": True,
                                "symbol": self.account.symbol,
                                "signal_timestamp": close_at.isoformat(),
                                "diagnostic_error": type(exc).__name__,
                            }
                        logger.info("Near RANGE structure shadow {}", json.dumps(
                            shadow, ensure_ascii=False, sort_keys=True))
                    trend_decision = None
                    classification_evidence = {}
                    if ((route.source == MEAN_REVERSION and position is None)
                            or (position is not None and position_source == MEAN_REVERSION)):
                        signal = generate_signal(regime, bars_1h.iloc[-1], flags, self.settings,
                                                 position=position.side if position else None,
                                                 bars_1h=bars_1h, bars_4h=bars_4h,
                                                 entry_reference_price=float(current_price),
                                                 entry_atr=float(one_hour_atr), range_low=range_low,
                                                 range_high=range_high, structure_low=structure_low,
                                                 structure_high=structure_high)
                        if position is None:
                            diagnostic_flags = replace(
                                flags, allow_long_entry=True, allow_short_entry=True,
                                flatten_only=False, data_ready=True, data_reason=None,
                            )
                            raw_candidate = generate_signal(
                                regime, bars_1h.iloc[-1], diagnostic_flags, self.settings,
                                position=None, bars_1h=bars_1h, bars_4h=bars_4h,
                                entry_reference_price=float(current_price),
                                entry_atr=float(one_hour_atr), range_low=range_low,
                                range_high=range_high, structure_low=structure_low,
                                structure_high=structure_high,
                            )
                            if raw_candidate.action is Action.OPEN and raw_candidate.side is not None:
                                direction_allowed = (
                                    flags.allow_long_entry if raw_candidate.side is Side.LONG
                                    else flags.allow_short_entry
                                )
                                filter_allowed = bool(
                                    flags.data_ready and not flags.flatten_only and direction_allowed
                                )
                                local_close = close_at.to_pydatetime().astimezone(
                                    ZoneInfo(self.settings.timezone),
                                )
                                reason = "time_filter_allowed"
                                if not flags.data_ready:
                                    reason = flags.data_reason or "time_filter_data_unavailable"
                                elif flags.flatten_only:
                                    reason = "night_new_entries_disabled"
                                elif not direction_allowed:
                                    reason = signal.reason
                                    if not reason.endswith("_blocked_by_time_filter"):
                                        reason = "time_filter_direction_blocked"
                                audit = {
                                    "at": now.isoformat(),
                                    "symbol": self.account.symbol,
                                    "source": MEAN_REVERSION,
                                    "side": raw_candidate.side.value,
                                    "decision": "ALLOW" if filter_allowed else "BLOCK",
                                    "reason": reason,
                                    "candidate_reason": raw_candidate.reason,
                                    "actual_signal_reason": signal.reason,
                                    "bar_open_at": bars_1h.index[-1].isoformat(),
                                    "bar_close_at": close_at.isoformat(),
                                    "bar_close_local": local_close.isoformat(),
                                    "recent_move_atr": flags.recent_move_atr,
                                    "morning_move": flags.morning_move,
                                    "afternoon_dump_today": flags.afternoon_dump_today,
                                    "at_upper_extreme": flags.at_upper_extreme,
                                    "at_lower_extreme": flags.at_lower_extreme,
                                    "watch_next_morning_long": flags.watch_next_morning_long,
                                    "data_ready": flags.data_ready,
                                    "data_reason": flags.data_reason,
                                    "allow_long_entry": flags.allow_long_entry,
                                    "allow_short_entry": flags.allow_short_entry,
                                    "flatten_only": flags.flatten_only,
                                    "runner_build_id": PAPER_RUNNER_BUILD_ID,
                                    "time_filter_build_id": TIME_FILTER_BUILD_ID,
                                }
                                self._event("entry_time_filter_decision", **audit)
                                log = logger.info if filter_allowed else logger.warning
                                log("MR time-filter decision {}", json.dumps(
                                    audit, ensure_ascii=False, sort_keys=True,
                                ))
                        signal_source = MEAN_REVERSION
                    elif (route.source == TREND and route.entry_allowed
                          and position is None and self._trend_execution_enabled):
                        trend_decision = evaluate_trend_entry(
                            self.account.symbol, bars_1h, bars_4h, self.settings,
                            now=close_at.to_pydatetime(), market_risk_state=market_risk_state,
                            position_source=position_source, has_open_position=position is not None,
                            pending_source=(self.account.pending.get("source")
                                            if self.account.pending is not None else None),
                            equity=self._current_equity(), step_size=step_size,
                            minimum_quantity=minimum_quantity,
                            existing_gross_notional=self.account.portfolio.gross_notional,
                            gross_notional_cap_pct=self.gross_notional_cap_pct,
                        )
                        signal = trend_decision.signal
                        signal_source = signal.source or ROUTER
                    else:
                        signal = Signal(None, Action.NONE,
                                        route.entry_block_reason or route.route_reason, 0.0)
                        signal_source = ROUTER
                    signal = with_signal_context(
                        signal, symbol=self.account.symbol, source=signal_source,
                        timestamp=close_at.to_pydatetime(),
                    )
                    held_position = self.account.position
                    active_exit = self.account.active_exit or {}
                    max_hold_hours = (effective_max_hold_hours(
                        held_position.side, held_position.entry_price, held_position.stop_price,
                        active_exit, self.settings.exits,
                    ) if held_position is not None else None)
                    if (held_position is not None and held_position.source != TREND
                            and max_hold_hours is not None
                            and (now - held_position.opened_at).total_seconds() / 3600
                            >= max_hold_hours and signal.action is not Action.CLOSE):
                        signal = with_signal_context(
                            Signal(held_position.side, Action.CLOSE, "max_hold_24h", 1.0),
                            symbol=self.account.symbol, source=held_position.source,
                            timestamp=close_at.to_pydatetime(),
                        )
                    candidate_action = signal.action.value
                    candidate_side = signal.side.value if signal.side else None
                    candidate_reason = signal.reason
                    candidate_source = signal.source
                    signal = apply_entry_route(signal, route)
                    route_details = route.to_dict()
                    route_details["strategy_mode"] = (
                        "MEAN_REVERSION_ONLY" if mean_reversion_only else "REGIME_ROUTED"
                    )
                    route_details["observed_symbol_regime_state"] = symbol_regime_state
                    if trend_decision is not None:
                        classification_evidence = (trend_decision.assessment.evidence
                                                    if trend_decision.assessment is not None else {})
                        route_details["trend_entry_decision"] = trend_decision.to_dict()
                        route_details["trend_confirmation"] = {
                            "confirm_bars": self.settings.symbol_regime.confirm_bars,
                            "confirmation_pass": bool(classification_evidence.get(
                                "trend_confirmation_pass", False)),
                            "confirmation_bar_time": (trend_decision.confirmation_bar_time.isoformat()
                                                       if trend_decision.confirmation_bar_time else None),
                            "candidates": classification_evidence.get("trend_confirmation_candidates", []),
                        }
                    trend_plan_expiry = (trend_decision.expires_at if trend_decision is not None else None)
                    exemption_candidate = bool(
                        signal.action is Action.OPEN
                        and signal.source == TREND
                        and not allow_entries
                        and symbol_in_hot_universe is False
                        and entry_execution_allowed
                        and self._trend_execution_enabled
                        and self.settings.symbol_regime.confirm_bars == 3
                        and bool(classification_evidence.get("trend_confirmation_pass", False))
                        and trend_decision is not None
                        and trend_decision.pullback_price is not None
                        and trend_decision.stop_price is not None
                        and trend_plan_expiry is not None
                        and _dt(trend_plan_expiry) > now
                    )
                    route_details.update({
                        "execution_mode": self._execution_mode,
                        "candidate_action": candidate_action,
                        "candidate_side": candidate_side,
                        "candidate_reason": candidate_reason,
                        "candidate_source": candidate_source,
                        "action": signal.action.value,
                        "side": signal.direction,
                        "direction": signal.direction,
                        "reason": signal.reason,
                        "result_action": signal.action.value,
                        "result_side": signal.side.value if signal.side else None,
                        "signal_source": signal.source,
                        "signal_timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
                        "hot_universe_eligible": symbol_in_hot_universe,
                        "hot_universe_exemption_candidate": exemption_candidate,
                        "adaptive_pool_allowed_sources": (sorted(allowed_entry_sources)
                                                           if allowed_entry_sources is not None else None),
                        "adaptive_pool_source_allowed": (
                            signal.source in allowed_entry_sources
                            if allowed_entry_sources is not None and signal.action is Action.OPEN else None
                        ),
                    })
                    self._event("strategy_route_evaluated", **route_details)
                    logger.info("Strategy route {}", json.dumps(route_details, ensure_ascii=False, sort_keys=True))
                    self.account.last_processed_close = close_key
                    set_status = getattr(self.account, "set_symbol_status", None)
                    if callable(set_status):
                        set_status(regime, signal, close_at.to_pydatetime(), route=route_details)
                    if (signal.action is Action.OPEN and allowed_entry_sources is not None
                            and signal.source not in allowed_entry_sources):
                        self._event(
                            "signal_rejected", reason="source_not_in_adaptive_candidate_pool",
                            action=signal.action.value, source=signal.source,
                            allowed_sources=sorted(allowed_entry_sources),
                            signal_time=close_at.isoformat(),
                        )
                        return True
                    if signal.action is Action.NONE and signal.reason in {
                        "long_rejected_below_range", "short_rejected_above_range", "invalid_on_entry"
                    }:
                        self._event("signal_rejected", reason=signal.reason,
                                    reference_price=float(current_price), atr=float(one_hour_atr),
                                    range_low=range_low, range_high=range_high,
                                    entry_structure_low=structure_low, entry_structure_high=structure_high,
                                    signal_time=close_at.isoformat())
                    if signal.action in (Action.OPEN, Action.CLOSE):
                        cutoff = _dt(entry_not_before) if entry_not_before is not None else None
                        hot_universe_exempted = False
                        if signal.action is Action.OPEN and self.gross_notional_cap_pct <= 0:
                            self._event("signal_rejected", reason="market_risk_paused", action="OPEN",
                                        signal_time=close_at.isoformat())
                            changed = True
                            return changed
                        if signal.action is Action.OPEN and not allow_entries:
                            if exemption_candidate:
                                hot_universe_exempted = True
                            else:
                                expiry = trend_plan_expiry.isoformat() if trend_plan_expiry else None
                                expired_candidate = bool(
                                    signal.source == TREND and symbol_in_hot_universe is False
                                    and entry_execution_allowed and self.settings.symbol_regime.confirm_bars == 3
                                    and classification_evidence.get("trend_confirmation_pass", False)
                                    and trend_decision is not None and trend_decision.pullback_price is not None
                                    and trend_decision.stop_price is not None and trend_plan_expiry is not None
                                    and _dt(trend_plan_expiry) <= now
                                )
                                self._event(
                                    "signal_ignored", reason="symbol_outside_hot_universe",
                                    action=signal.action.value,
                                    **({
                                        "exemption_granted": False,
                                        "exemption_reason": "trend_confirmed_pullback",
                                        "exemption_expires_at": expiry,
                                        "exemption_status": "expired_before_grant",
                                    } if expired_candidate else {}),
                                    source=signal.source,
                                )
                                changed = True
                                return changed
                        if signal.action is Action.OPEN and cutoff and close_at.to_pydatetime() < cutoff:
                            self._event("signal_ignored", reason="symbol_outside_hot_universe",
                                        filter="entry_cooldown", action=signal.action.value,
                                        cutoff_at=cutoff.isoformat(), signal_time=close_at.isoformat(),
                                        source=signal.source)
                            changed = True
                            return changed
                        if (signal.action is Action.OPEN and self.settings.execution.confirm_on_1h_close
                                and (now - close_at.to_pydatetime()).total_seconds()
                                > self.settings.execution.confirm_delay_sec_max):
                            self._event("signal_rejected", reason="confirm_delay_exceeded", action="OPEN",
                                        signal_time=close_at.isoformat(), confirmed_at=now.isoformat(),
                                        delay_sec=(now - close_at.to_pydatetime()).total_seconds())
                            changed = True
                            return changed
                        trend_pending = None
                        if signal.action is Action.OPEN and signal.source == TREND:
                            evidence = (trend_decision.assessment.evidence
                                        if trend_decision is not None and trend_decision.assessment is not None else {})
                            confirmation_fields = {
                                "confirm_bars": self.settings.symbol_regime.confirm_bars,
                                "confirmation_pass": bool(evidence.get("trend_confirmation_pass", False)),
                                "confirmation_bar_time": (trend_decision.confirmation_bar_time.isoformat()
                                                           if trend_decision is not None
                                                           and trend_decision.confirmation_bar_time else None),
                            }
                            if (not self._trend_execution_enabled or trend_decision is None
                                    or trend_decision.pullback_price is None or trend_decision.stop_price is None
                                    or not confirmation_fields["confirmation_pass"]):
                                self._event("signal_rejected", reason="trend_entry_plan_unavailable",
                                            action="OPEN", source=TREND, signal_time=close_at.isoformat(),
                                            **confirmation_fields)
                                changed = True
                                return changed
                            planned_entry_price = float(trend_decision.pullback_price)
                            planned_stop = float(trend_decision.stop_price)
                            preflight_quantity = self._trend_quantity(
                                self._current_equity(), planned_entry_price, planned_stop,
                                step_size=step_size, minimum_quantity=minimum_quantity,
                            )
                            if preflight_quantity <= 0:
                                rejection_reason = "trend_risk_size_below_exchange_minimum"
                            else:
                                preflight_intent = OrderIntent(
                                    self.account.symbol, signal.side, Action.OPEN, preflight_quantity,
                                    planned_entry_price, signal.reason, regime.regime, planned_stop, TREND,
                                )
                                preflight = guard_signal(
                                    signal, preflight_intent, self.account.portfolio,
                                    self._current_equity(), funding_rate, now, self.settings,
                                    gross_notional_cap_pct=self.gross_notional_cap_pct,
                                    market_risk_state=self.market_risk_state,
                                    reserved_long_notional=self._reserved_directional_notional(Side.LONG),
                                    reserved_short_notional=self._reserved_directional_notional(Side.SHORT),
                                    reserved_symbols=self._reserved_entry_count(),
                                )
                                self._event("order_intent", intent=_intent_dict(preflight_intent),
                                            decision=preflight.reason, allowed=preflight.allowed,
                                            source=TREND, direction=signal.direction,
                                            signal_timestamp=signal.timestamp.isoformat()
                                            if signal.timestamp else None,
                                            funding_rate=funding_rate,
                                            risk_details=preflight.details or {},
                                            market_risk_state=self.market_risk_state,
                                            phase="pending_preflight", **confirmation_fields)
                                rejection_reason = None if preflight.allowed else preflight.reason
                                if rejection_reason is None:
                                    state = getattr(self.account, "state", None)
                                    pending_by_symbol = getattr(state, "pending_by_symbol", {})
                                    reserved_entries = [
                                        item for item in pending_by_symbol.values()
                                        if item.get("execution_type") == "pullback"
                                    ]
                                    reserved_notional = sum(
                                        float(item.get("planned_quantity", 0.0))
                                        * float(item.get("reference_price", 0.0))
                                        for item in reserved_entries
                                    )
                                    total_notional = (self.account.portfolio.gross_notional
                                                      + preflight_intent.notional + reserved_notional)
                                    if total_notional > self._current_equity() * self.gross_notional_cap_pct:
                                        rejection_reason = (
                                            "market_risk_budget_reserved_by_pending_entries"
                                            if self.gross_notional_cap_pct
                                            < self.settings.sizing.max_gross_notional_pct
                                            else "gross_notional_reserved_by_pending_entries"
                                        )
                                    elif self.account.cash < (
                                            preflight_intent.notional / self.settings.sizing.leverage
                                            + preflight_intent.notional * self.fee_rate):
                                        rejection_reason = "insufficient_available_isolated_margin"
                            if rejection_reason is not None:
                                self._event("signal_rejected", reason=rejection_reason, action="OPEN",
                                            source=TREND, signal_time=close_at.isoformat(),
                                            **({
                                                "exemption_granted": False,
                                                "exemption_reason": "trend_confirmed_pullback",
                                                "exemption_expires_at": trend_decision.expires_at.isoformat(),
                                                "exemption_status": "not_granted_risk_rejected",
                                            } if hot_universe_exempted else {}),
                                            **confirmation_fields)
                                changed = True
                                return changed
                            trend_pending = {
                                "execution_type": "pullback",
                                "signal_time": close_at.isoformat(),
                                "confirmed_at": now.isoformat(),
                                "reference_price": planned_entry_price,
                                "signal_reference_price": trend_decision.reference_price,
                                "planned_entry_price": planned_entry_price,
                                "pullback_price": planned_entry_price,
                                "expires_at": trend_decision.expires_at.isoformat(),
                                "planned_quantity": preflight_quantity,
                                "stop_price": planned_stop,
                                "structure_break_price": trend_decision.structure_break_price,
                                **confirmation_fields,
                            }
                            if hot_universe_exempted:
                                trend_pending.update({
                                    "hot_universe_exemption": True,
                                    "exemption_reason": "trend_confirmed_pullback",
                                    "exemption_expires_at": trend_decision.expires_at.isoformat(),
                                })
                        self.account.pending = {
                            "execute_at": close_at.isoformat(), "action": signal.action.value,
                            "side": signal.side.value if signal.side else None, "reason": signal.reason,
                            "strength": signal.strength, "close_first": signal.close_first,
                            "size_multiplier": signal.size_multiplier, "atr": float(one_hour_atr),
                            "regime": regime.regime.value,
                            "symbol": signal.symbol, "source": signal.source,
                            "signal_time": signal.timestamp.isoformat() if signal.timestamp else close_at.isoformat(),
                        }
                        if signal.action is Action.OPEN:
                            self.account.pending.update({
                                "entry_signal_low": float(bars_1h.iloc[-1]["low"]),
                                "entry_signal_high": float(bars_1h.iloc[-1]["high"]),
                                "entry_signal_close": float(bars_1h.iloc[-1]["close"]),
                                "low_zone_lower": range_low,
                                "entry_structure_low": structure_low,
                                "entry_structure_high": structure_high,
                                "entry_range_low": range_low,
                                "entry_range_high": range_high,
                                "entry_midpoint": ((range_low + range_high) / 2
                                                   if range_low is not None and range_high is not None else None),
                                "entry_1h_midpoint": (float(bars_1h.iloc[-1]["high"])
                                                       + float(bars_1h.iloc[-1]["low"])) / 2,
                            })
                            if signal.source == TREND and trend_pending is not None:
                                self.account.pending.update(trend_pending)
                                if self.account.pending.get("hot_universe_exemption"):
                                    self._event(
                                        "hot_universe_exemption_granted", exemption_granted=True,
                                        exemption_reason="trend_confirmed_pullback",
                                        exemption_expires_at=trend_pending["exemption_expires_at"],
                                        symbol=self.account.symbol, source=TREND,
                                        signal_time=close_at.isoformat(), **confirmation_fields,
                                    )
                                self._event("entry_confirmed", symbol=self.account.symbol,
                                            side=signal.side.value, source=TREND,
                                            signal_time=close_at.isoformat(),
                                            confirmed_at=now.isoformat(),
                                            reference_price=trend_pending["signal_reference_price"],
                                            pullback_price=trend_pending["pullback_price"],
                                            stop_price=trend_pending["stop_price"],
                                            reason=signal.reason, **confirmation_fields)
                                self._event("pending_entry_started", symbol=self.account.symbol,
                                            side=signal.side.value, source=TREND,
                                            signal_time=close_at.isoformat(),
                                            confirmed_at=now.isoformat(),
                                            reference_price=trend_pending["signal_reference_price"],
                                            pullback_price=trend_pending["pullback_price"],
                                            expires_at=trend_pending["expires_at"],
                                            stop_price=trend_pending["stop_price"],
                                            **self._hot_universe_exemption_fields(
                                                self.account.pending, status="active",
                                            ),
                                            reason=signal.reason, **confirmation_fields)
                            elif wait_pullback and self.settings.execution.wait_pullback:
                                reference = float(bars_1h.iloc[-1]["close"])
                                distance = self.settings.execution.pullback_atr * float(one_hour_atr)
                                midpoint = float(self.account.pending["entry_1h_midpoint"])
                                pullback_price = (max(reference - distance, midpoint) if signal.side is Side.LONG
                                                  else min(reference + distance, midpoint))
                                pullback_price = (min(pullback_price, reference) if signal.side is Side.LONG
                                                  else max(pullback_price, reference))
                                if signal.side is Side.LONG and pullback_price >= reference:
                                    pullback_price = reference - distance
                                elif signal.side is Side.SHORT and pullback_price <= reference:
                                    pullback_price = reference + distance
                                structure_price = self.account.pending.get(
                                    "entry_structure_low" if signal.side is Side.LONG else "entry_structure_high")
                                try:
                                    reserved_entries = [item for item in self.account.state.pending_by_symbol.values()
                                                        if item.get("execution_type") == "pullback"]
                                    if len(self.account.portfolio.positions) + len(reserved_entries) >= self.settings.sizing.max_symbols:
                                        raise ValueError("max_symbols_reserved_by_pending_entries")
                                    preflight_stop, preflight_size_multiplier = calculate_adaptive_stop(
                                        signal.side, reference, float(one_hour_atr), structure_price, self.settings,
                                    )
                                    preflight_quantity = calculate_order_quantity(
                                        self._current_equity(), reference, self.account.portfolio, self.settings,
                                        step_size=step_size, minimum_quantity=minimum_quantity,
                                        size_multiplier=(signal.size_multiplier * preflight_size_multiplier
                                                         * self.market_risk_multiplier),
                                        gross_notional_cap_pct=self.gross_notional_cap_pct,
                                    )
                                    preflight_intent = (OrderIntent(
                                        self.account.symbol, signal.side, Action.OPEN, preflight_quantity, reference,
                                        signal.reason, regime.regime, preflight_stop, signal.source,
                                    ) if preflight_quantity > 0 else None)
                                    if preflight_intent is None:
                                        target_notional = (self.settings.sizing.margin_per_symbol_usdt
                                                           * self.settings.sizing.leverage
                                                           * signal.size_multiplier * preflight_size_multiplier
                                                           * self.market_risk_multiplier)
                                        tightened = (self.gross_notional_cap_pct
                                                     < self.settings.sizing.max_gross_notional_pct)
                                        over_budget = (tightened and self.account.portfolio.gross_notional
                                                       + target_notional > self._current_equity()
                                                       * self.gross_notional_cap_pct)
                                        reject_detail = ("market_risk_budget_cap" if over_budget
                                                         else "no_remaining_portfolio_capacity")
                                    else:
                                        preflight = guard_signal(
                                            signal, preflight_intent, self.account.portfolio,
                                            self._current_equity(), funding_rate, now, self.settings,
                                            gross_notional_cap_pct=self.gross_notional_cap_pct,
                                            market_risk_state=self.market_risk_state,
                                            reserved_long_notional=self._reserved_directional_notional(Side.LONG),
                                            reserved_short_notional=self._reserved_directional_notional(Side.SHORT),
                                            reserved_symbols=self._reserved_entry_count(),
                                        )
                                        if preflight.allowed:
                                            reserved_notional = sum(
                                                float(item.get("planned_quantity", 0.0))
                                                * float(item.get("reference_price", 0.0))
                                                for item in reserved_entries
                                            )
                                            total_notional = (self.account.portfolio.gross_notional
                                                              + preflight_intent.notional + reserved_notional)
                                            if total_notional > self._current_equity() * self.gross_notional_cap_pct:
                                                reject_detail = ("market_risk_budget_reserved_by_pending_entries"
                                                                 if self.gross_notional_cap_pct
                                                                 < self.settings.sizing.max_gross_notional_pct
                                                                 else "gross_notional_reserved_by_pending_entries")
                                            elif self.account.cash < (
                                                    preflight_intent.notional / self.settings.sizing.leverage
                                                    + preflight_intent.notional * self.fee_rate):
                                                reject_detail = "insufficient_available_isolated_margin"
                                            else:
                                                reject_detail = None
                                        else:
                                            reject_detail = preflight.reason
                                except ValueError as exc:
                                    reject_detail = str(exc)
                                if reject_detail is not None:
                                    self.account.pending = None
                                    self._event("pending_entry_cancelled", reason="risk_reject",
                                                detail=reject_detail, signal_time=close_at.isoformat(),
                                                confirmed_at=now.isoformat())
                                    changed = True
                                    return changed
                                expires_at = now + timedelta(
                                    minutes=15 * self.settings.execution.wait_bars_15m)
                                self.account.pending.update({
                                    "execution_type": "pullback", "signal_time": close_at.isoformat(),
                                    "confirmed_at": now.isoformat(), "reference_price": reference,
                                    "pullback_price": pullback_price, "expires_at": expires_at.isoformat(),
                                    "planned_quantity": preflight_quantity,
                                })
                                self._event("entry_confirmed", symbol=self.account.symbol,
                                            side=signal.side.value, source=signal.source,
                                            signal_time=close_at.isoformat(),
                                            confirmed_at=now.isoformat(), reference_price=reference,
                                            atr=float(one_hour_atr), reason=signal.reason)
                                self._event("pending_entry_started", symbol=self.account.symbol,
                                            side=signal.side.value, source=signal.source,
                                            signal_time=close_at.isoformat(),
                                            confirmed_at=now.isoformat(), reference_price=reference,
                                            atr=float(one_hour_atr), pullback_price=pullback_price,
                                            expires_at=expires_at.isoformat(), reason=signal.reason)
                        if self.account.pending is not None:
                            self._event("signal_queued", **self.account.pending)
                    changed = True
        # A just-closed bar's signal is executable at the currently observed
        # next-hour open, even when the first poll arrives after that boundary.
        if (self.account.last_observed_open == current_open_key and self.account.pending is not None
                and self.account.pending.get("execution_type") != "pullback"):
            pending_at = _dt(self.account.pending["execute_at"])
            if pending_at == current_open_at:
                self._execute_pending(current_price, current_open_at, funding_rate,
                                      step_size=step_size, minimum_quantity=minimum_quantity)
                changed = True
        return changed
