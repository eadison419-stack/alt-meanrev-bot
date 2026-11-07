"""Fail-closed entry risk checks and stop-aware order intent construction."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from datetime import datetime
import json
import math
from typing import Any

from loguru import logger

from bot.portfolio import PortfolioState, Position
from bot.regime import Regime
from bot.signals import Action, Side, Signal


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    action: Action
    quantity: float
    price: float
    reason: str
    regime: Regime
    stop_price: float | None = None
    source: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "action", Action(self.action))
        object.__setattr__(self, "regime", Regime(self.regime))
        if not self.symbol.strip():
            raise ValueError("order symbol is required")
        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("order quantity must be finite and positive")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("order price must be finite and positive")
        if not self.reason.strip():
            raise ValueError("order reason is required")
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("order strategy source cannot be empty")
        if self.action is Action.OPEN:
            if self.stop_price is None or not math.isfinite(self.stop_price) or self.stop_price <= 0:
                raise ValueError("every opening order requires a finite positive stop price")
            if self.side is Side.LONG and self.stop_price >= self.price:
                raise ValueError("long stop must be below order price")
            if self.side is Side.SHORT and self.stop_price <= self.price:
                raise ValueError("short stop must be above order price")

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    intent: OrderIntent | None = None
    details: dict[str, Any] | None = None


def _market_risk_multiplier(market_risk_state: str, settings) -> float:
    state = str(market_risk_state or "PAUSE_NEW").upper()
    if state == "NORMAL":
        return 1.0
    if state == "RISK_ELEVATED":
        return float(settings.market_risk.elevated_size_multiplier)
    if state == "PAUSE_NEW":
        return 0.0
    raise ValueError("market_risk_state_invalid")


def calculate_stop_price(side: Side | str, entry_price: float, atr_value: float, settings) -> float:
    side = Side(side)
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry price must be finite and positive")
    if not math.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("ATR must be finite and positive")
    distance = settings.signals.stop_atr_mult * atr_value
    stop = entry_price - distance if side is Side.LONG else entry_price + distance
    if stop <= 0:
        raise ValueError("calculated stop price must be positive")
    return stop


def calculate_adaptive_stop(side: Side | str, entry_price: float, atr_value: float,
                            structure_price: float | None, settings) -> tuple[float, float]:
    """Return a structure-aware stop and sizing multiplier for its un-capped width."""
    side = Side(side)
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("entry price and ATR must be finite and positive")
    policy = settings.stops
    if not policy.use_structure_stop or structure_price is None:
        raw_distance = policy.hard_cap_atr * atr_value
    else:
        if not math.isfinite(structure_price) or structure_price <= 0:
            raise ValueError("structure price must be finite and positive")
        raw_distance = ((entry_price - structure_price) if side is Side.LONG
                        else (structure_price - entry_price)) + policy.structure_buffer_atr * atr_value
    raw_atr = raw_distance / atr_value
    if raw_atr <= 0:
        raise ValueError("structure level is on the wrong side of entry")
    size_multiplier = 1.0
    if raw_atr < policy.min_stop_atr:
        if policy.min_stop_action == "reject":
            raise ValueError("stop_too_tight")
        raw_distance = policy.min_stop_atr * atr_value
        raw_atr = policy.min_stop_atr
    if raw_atr > policy.max_stop_atr:
        if policy.wide_stop_action == "reject":
            raise ValueError("stop_too_wide")
        size_multiplier = policy.max_stop_atr / raw_atr
    actual_distance = min(raw_distance, policy.hard_cap_atr * atr_value)
    actual_distance = max(actual_distance, policy.min_stop_atr * atr_value)
    stop = entry_price - actual_distance if side is Side.LONG else entry_price + actual_distance
    if stop <= 0:
        raise ValueError("calculated stop price must be positive")
    return stop, size_multiplier


def calculate_order_quantity(
    equity: float,
    mark_price: float,
    portfolio: PortfolioState,
    settings,
    *,
    step_size: float,
    minimum_quantity: float = 0.0,
    size_multiplier: float = 1.0,
    gross_notional_cap_pct: float | None = None,
    stop_price: float | None = None,
    target_margin_usdt: float | None = None,
) -> float:
    """Size to target margin, then cap quantity by stop risk when a stop is supplied."""
    values = (equity, mark_price, step_size, minimum_quantity, size_multiplier)
    if (any(not math.isfinite(value) for value in values)
            or (stop_price is not None and not math.isfinite(stop_price))
            or equity <= 0 or mark_price <= 0 or step_size <= 0
            or minimum_quantity < 0 or size_multiplier <= 0):
        raise ValueError("equity, mark price, step size, and minimum quantity are invalid")
    target_margin = (settings.sizing.margin_per_symbol_usdt
                     if target_margin_usdt is None else target_margin_usdt)
    if not math.isfinite(target_margin) or target_margin <= 0:
        raise ValueError("target margin must be finite and positive")
    target_notional = (Decimal(str(target_margin))
                       * Decimal(str(settings.sizing.leverage)) * Decimal(str(size_multiplier)))
    cap_pct = settings.sizing.max_gross_notional_pct
    if gross_notional_cap_pct is not None:
        if not math.isfinite(gross_notional_cap_pct) or gross_notional_cap_pct < 0:
            raise ValueError("gross notional cap must be finite and non-negative")
        cap_pct = min(cap_pct, gross_notional_cap_pct)
    gross_cap = Decimal(str(equity)) * Decimal(str(cap_pct))
    current_gross = Decimal(str(portfolio.gross_notional))
    remaining_gross = max(Decimal(0), gross_cap - current_gross)
    if stop_price is None and remaining_gross < target_notional:
        return 0.0
    raw_quantity = target_notional / Decimal(str(mark_price))
    if stop_price is not None:
        if stop_price <= 0 or stop_price == mark_price:
            raise ValueError("stop price must be positive and differ from mark price")
        cost_rate = (Decimal(str(settings.fees.taker_bps)) / Decimal(10_000)
                     + Decimal(str(settings.risk.max_slippage_pct)))
        risk_per_unit = (abs(Decimal(str(mark_price)) - Decimal(str(stop_price)))
                         + (Decimal(str(mark_price)) + Decimal(str(stop_price))) * cost_rate)
        risk_budget = Decimal(str(equity)) * Decimal(str(settings.risk.max_single_trade_risk_pct))
        if risk_per_unit <= 0 or risk_budget <= 0:
            return 0.0
        raw_quantity = min(raw_quantity, risk_budget / risk_per_unit)
        raw_quantity = min(raw_quantity, remaining_gross / Decimal(str(mark_price)))
        if raw_quantity <= 0:
            return 0.0
    step = Decimal(str(step_size))
    quantity = (raw_quantity / step).to_integral_value(rounding=ROUND_FLOOR) * step
    if quantity < Decimal(str(minimum_quantity)):
        return 0.0
    return float(quantity)


def assess_entry(
    intent: OrderIntent,
    portfolio: PortfolioState,
    equity: float,
    funding_rate: float | None,
    now: datetime,
    settings,
    *,
    bypass_cooldown: bool = False,
    bars_since_exit: int | None = None,
    gross_notional_cap_pct: float | None = None,
    market_risk_state: str = "NORMAL",
    reserved_long_notional: float = 0.0,
    reserved_short_notional: float = 0.0,
    reserved_symbols: int = 0,
    per_symbol_margin_usdt: float | None = None,
) -> RiskDecision:
    """Approve or reject an opening intent; all missing risk inputs fail closed."""
    if intent.action is not Action.OPEN:
        return RiskDecision(True, "not_an_opening_order")
    halted = portfolio.observe_equity(equity, now, settings)
    if halted:
        return RiskDecision(False, "daily_drawdown_halt")
    try:
        risk_multiplier = _market_risk_multiplier(market_risk_state, settings)
    except ValueError:
        return RiskDecision(False, "market_risk_state_invalid")
    if risk_multiplier == 0:
        return RiskDecision(False, "market_risk_paused")
    funding_rejection = funding_rate_rejection_reason(funding_rate, settings)
    if funding_rejection is not None:
        return RiskDecision(False, funding_rejection)
    previous_exit = portfolio.last_exit_at.get(intent.symbol)
    if previous_exit is not None:
        if now.tzinfo is None or previous_exit.tzinfo is None:
            return RiskDecision(False, "cooldown_timestamp_invalid")
        elapsed_hours = (now - previous_exit).total_seconds() / 3600
        category = portfolio.last_exit_category.get(intent.symbol)
        category_hours = {
            "stop_loss": settings.cooldowns.stop_loss_hours,
            "profit_exit": settings.cooldowns.profit_exit_hours,
            "strategy_exit": settings.cooldowns.strategy_exit_hours,
            "liquidation": settings.cooldowns.liquidation_hours,
        }
        category_cooldown = category_hours.get(category, settings.signals.cooldown_hours)
        previous_source = portfolio.last_exit_source.get(intent.symbol)
        is_strategy_switch = previous_source is not None and previous_source != intent.source
        if is_strategy_switch and bars_since_exit is not None and (
                isinstance(bars_since_exit, bool) or not isinstance(bars_since_exit, int)
                or bars_since_exit < 0):
            return RiskDecision(False, "strategy_switch_cooldown_bar_count_invalid")
        switch_cooldown_hours = (settings.cooldowns.strategy_switch_hours
                                 if is_strategy_switch and bars_since_exit is None else 0.0)
        switch_required_bars = (settings.cooldowns.strategy_switch_bars
                                if is_strategy_switch and bars_since_exit is not None else 0)
        switch_cooldown_active = (
            bars_since_exit < switch_required_bars if switch_required_bars else
            elapsed_hours < switch_cooldown_hours
        )
        if bypass_cooldown and not is_strategy_switch:
            category_cooldown = 0.0
        category_cooldown_active = elapsed_hours < category_cooldown
        if category_cooldown_active or switch_cooldown_active:
            if bars_since_exit is not None and switch_required_bars:
                switch_remaining = max(0, switch_required_bars - bars_since_exit)
                category_remaining = max(0.0, category_cooldown - elapsed_hours)
                switch_has_priority = switch_cooldown_active and switch_remaining > category_remaining
            else:
                switch_has_priority = switch_cooldown_active and switch_cooldown_hours > category_cooldown
            reason = "strategy_switch_cooldown_active" if switch_has_priority else "cooldown_active"
            return RiskDecision(False, reason, details={
                "elapsed_hours": elapsed_hours,
                "required_hours": category_cooldown if category_cooldown_active else switch_cooldown_hours,
                "bars_since_exit": bars_since_exit,
                "required_bars": switch_required_bars,
                "last_exit_source": previous_source,
                "signal_source": intent.source,
            })
    if intent.symbol in portfolio.positions:
        held = portfolio.positions[intent.symbol]
        if intent.source != held.source:
            return RiskDecision(False, "position_owned_by_other_strategy", details={
                "position_source": held.source,
                "signal_source": intent.source,
            })
        return RiskDecision(False, "symbol_already_open")
    if not isinstance(reserved_symbols, int) or reserved_symbols < 0:
        return RiskDecision(False, "reserved_exposure_invalid")
    if len(portfolio.positions) + reserved_symbols >= settings.sizing.max_symbols:
        return RiskDecision(False, "max_symbols_reached")
    symbol_margin = (settings.sizing.margin_per_symbol_usdt
                     if per_symbol_margin_usdt is None else per_symbol_margin_usdt)
    if not math.isfinite(symbol_margin) or symbol_margin <= 0:
        return RiskDecision(False, "per_symbol_margin_invalid")
    symbol_cap = (symbol_margin * settings.sizing.leverage
                  * risk_multiplier)
    if intent.notional > symbol_cap:
        return RiskDecision(False, "per_symbol_notional_cap", details={
            "order_notional": intent.notional,
            "max_notional": symbol_cap,
        })
    stop_loss = intent.quantity * abs(intent.price - intent.stop_price)
    cost_rate = settings.fees.taker_bps / 10_000 + settings.risk.max_slippage_pct
    round_trip_cost = intent.quantity * (intent.price + intent.stop_price) * cost_rate
    estimated_risk = stop_loss + round_trip_cost
    risk_limit = equity * settings.risk.max_single_trade_risk_pct
    if estimated_risk > risk_limit:
        return RiskDecision(False, "single_trade_risk_cap", details={
            "estimated_stop_loss": stop_loss,
            "estimated_round_trip_fees_and_slippage": round_trip_cost,
            "estimated_risk": estimated_risk,
            "max_risk": risk_limit,
            "equity": equity,
        })
    gross_cap_pct = settings.sizing.max_gross_notional_pct * risk_multiplier
    if gross_notional_cap_pct is not None:
        if not math.isfinite(gross_notional_cap_pct) or gross_notional_cap_pct < 0:
            return RiskDecision(False, "gross_notional_cap_invalid")
        gross_cap_pct = min(gross_cap_pct, gross_notional_cap_pct)
        if gross_cap_pct == 0:
            return RiskDecision(False, "market_risk_paused")
    for reserved in (reserved_long_notional, reserved_short_notional):
        if not math.isfinite(reserved) or reserved < 0:
            return RiskDecision(False, "reserved_exposure_invalid")
    gross_cap = equity * gross_cap_pct
    reserved_gross = reserved_long_notional + reserved_short_notional
    if portfolio.gross_notional + reserved_gross + intent.notional > gross_cap:
        return RiskDecision(False, "market_risk_budget_cap" if gross_cap_pct
                            < settings.sizing.max_gross_notional_pct
                            else "gross_notional_cap")
    current_directional = sum(
        position.notional for position in portfolio.positions.values()
        if position.side is intent.side
    )
    reserved_directional = reserved_long_notional if intent.side is Side.LONG else reserved_short_notional
    directional_cap = equity * settings.sizing.max_directional_notional_pct * risk_multiplier
    directional_total = current_directional + reserved_directional + intent.notional
    if directional_total > directional_cap:
        return RiskDecision(False, "directional_exposure_cap", details={
            "direction": intent.side.value,
            "current_notional": current_directional,
            "reserved_notional": reserved_directional,
            "order_notional": intent.notional,
            "total_notional": directional_total,
            "max_notional": directional_cap,
        })
    return RiskDecision(True, "approved", intent, details={
        "estimated_stop_loss": stop_loss,
        "estimated_round_trip_fees_and_slippage": round_trip_cost,
        "estimated_risk": estimated_risk,
        "max_risk": risk_limit,
        "risk_multiplier": risk_multiplier,
        "gross_notional_cap": gross_cap,
        "directional_exposure": directional_total,
        "directional_notional_cap": directional_cap,
    })


def funding_rate_rejection_reason(funding_rate: float | None, settings) -> str | None:
    """Return the shared fail-closed funding gate reason for an opening order."""
    if funding_rate is None or not math.isfinite(funding_rate):
        return "funding_rate_missing"
    if abs(funding_rate) >= settings.risk.funding_abs_block:
        return "funding_rate_blocked"
    return None


def guard_signal(
    signal: Signal,
    intent: OrderIntent | None,
    portfolio: PortfolioState,
    equity: float,
    funding_rate: float | None,
    now: datetime,
    settings,
    *,
    bypass_cooldown: bool = False,
    bars_since_exit: int | None = None,
    gross_notional_cap_pct: float | None = None,
    market_risk_state: str = "NORMAL",
    reserved_long_notional: float = 0.0,
    reserved_short_notional: float = 0.0,
    reserved_symbols: int = 0,
) -> RiskDecision:
    """Route every OPEN signal through the same risk checks before execution."""
    if signal.action is not Action.OPEN:
        return RiskDecision(True, "signal_does_not_open")

    def reject(reason: str, *, details: dict[str, Any] | None = None) -> RiskDecision:
        decision = RiskDecision(False, reason, details=details)
        held = portfolio.positions.get(intent.symbol) if intent is not None else None
        audit = {
            "symbol": signal.symbol or (intent.symbol if intent else None),
            "direction": signal.direction,
            "source": signal.source,
            "signal_source": signal.source,
            "position_source": held.source if held else None,
            "reason": reason,
            "market_risk_state": market_risk_state,
            "order_notional": intent.notional if intent else None,
            "stop_price": intent.stop_price if intent else None,
            "funding_rate": (funding_rate if funding_rate is not None and math.isfinite(funding_rate)
                             else None),
            "details": details or {},
        }
        logger.warning("Risk entry rejected {}", json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return decision

    if intent is None:
        return reject("opening_intent_missing")
    if intent.action is not Action.OPEN:
        return reject("opening_intent_action_mismatch")
    if signal.side is None or signal.side is not intent.side:
        return reject("signal_order_side_mismatch")
    if signal.symbol is not None and signal.symbol != intent.symbol:
        return reject("signal_order_symbol_mismatch")
    if not isinstance(signal.source, str) or signal.source not in {"MEAN_REVERSION", "TREND"}:
        return reject("strategy_source_missing" if not signal.source else "strategy_source_invalid")
    if intent.source is not None and intent.source != signal.source:
        return reject("signal_order_source_mismatch", details={
            "signal_source": signal.source,
            "order_source": intent.source,
        })
    try:
        _market_risk_multiplier(market_risk_state, settings)
    except ValueError:
        return reject("market_risk_state_invalid")
    held = portfolio.positions.get(intent.symbol)
    if held is not None and held.source != signal.source:
        return reject("position_owned_by_other_strategy", details={
            "position_source": held.source,
            "signal_source": signal.source,
        })
    intent = OrderIntent(
        intent.symbol, intent.side, intent.action, intent.quantity, intent.price,
        intent.reason, intent.regime, intent.stop_price, signal.source,
    )
    decision = assess_entry(
        intent, portfolio, equity, funding_rate, now, settings,
        bypass_cooldown=bypass_cooldown,
        bars_since_exit=bars_since_exit,
        gross_notional_cap_pct=gross_notional_cap_pct,
        market_risk_state=market_risk_state,
        reserved_long_notional=reserved_long_notional,
        reserved_short_notional=reserved_short_notional,
        reserved_symbols=reserved_symbols,
    )
    if not decision.allowed:
        return reject(decision.reason, details=decision.details)
    return decision
