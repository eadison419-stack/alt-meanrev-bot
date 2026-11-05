"""Independent, closed-bar trend decisions and risk plans; no execution side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR
import json
import math
from typing import Any

import pandas as pd
from loguru import logger

from bot.data.klines import normalize_symbol
from bot.indicators import atr, validate_ohlcv
from bot.portfolio import Position
from bot.signals import Action, Side, Signal, with_signal_context
from bot.strategy_router import ROUTER, TREND, apply_entry_route, resolve_strategy_route
from bot.symbol_regime import SymbolRegimeAssessment, SymbolRegimeState, classify_symbol_regime


HOUR = pd.Timedelta(1, unit="h")


@dataclass(frozen=True)
class TrendEntryDecision:
    signal: Signal
    assessment: SymbolRegimeAssessment | None = None
    signal_bar_time: datetime | None = None
    confirmation_bar_time: datetime | None = None
    reference_price: float | None = None
    pullback_price: float | None = None
    expires_at: datetime | None = None
    stop_price: float | None = None
    structure_break_price: float | None = None
    quantity: float | None = None
    risk_budget: float | None = None
    risk_per_unit: float | None = None
    route: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.signal.symbol,
            "source": self.signal.source,
            "action": self.signal.action.value,
            "direction": self.signal.direction,
            "reason": self.signal.reason,
            "signal_time": self.signal.timestamp.isoformat() if self.signal.timestamp else None,
            "signal_bar_time": self.signal_bar_time.isoformat() if self.signal_bar_time else None,
            "confirmation_bar_time": (self.confirmation_bar_time.isoformat()
                                      if self.confirmation_bar_time else None),
            "reference_price": self.reference_price,
            "pullback_price": self.pullback_price,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "stop_price": self.stop_price,
            "structure_break_price": self.structure_break_price,
            "quantity": self.quantity,
            "risk_budget": self.risk_budget,
            "risk_per_unit": self.risk_per_unit,
            "route": self.route,
            "classification": self.assessment.to_dict() if self.assessment else None,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TrendExitDecision:
    signal: Signal
    active_stop_price: float
    exit_state: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.signal.symbol,
            "source": self.signal.source,
            "action": self.signal.action.value,
            "direction": self.signal.direction,
            "reason": self.signal.reason,
            "signal_time": self.signal.timestamp.isoformat() if self.signal.timestamp else None,
            "active_stop_price": self.active_stop_price,
            "exit_state": self.exit_state,
            "evidence": self.evidence,
        }


def _closed_bars(frame: pd.DataFrame | None, now: datetime, interval: pd.Timedelta) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    validate_ohlcv(frame)
    return frame.loc[frame.index + interval <= pd.Timestamp(now)]


def _context_signal(symbol: str, side: Side | None, action: Action, reason: str,
                    source: str, at: datetime) -> Signal:
    return with_signal_context(
        Signal(side, action, reason, 1.0 if action is not Action.NONE else 0.0),
        symbol=symbol, source=source, timestamp=at,
    )


def _blocked_entry(symbol: str, at: datetime, reason: str, *, source: str = TREND,
                   assessment: SymbolRegimeAssessment | None = None,
                   signal_bar_time: datetime | None = None,
                   route: dict[str, Any] | None = None,
                   evidence: dict[str, Any] | None = None) -> TrendEntryDecision:
    decision = TrendEntryDecision(
        signal=_context_signal(symbol, None, Action.NONE, reason, source, at),
        assessment=assessment,
        signal_bar_time=signal_bar_time,
        confirmation_bar_time=(assessment.last_closed_bar_at if assessment else None),
        route=route or {},
        evidence=evidence or {},
    )
    logger.opt(lazy=True).info("Trend strategy entry decision {}", lambda: json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True))
    return decision


def calculate_trend_stop(side: Side | str, entry_price: float, atr_value: float,
                         closed_bars_1h: pd.DataFrame, settings) -> tuple[float, float]:
    """Return (protective stop, raw structure invalidation) without capping structure risk."""
    side = Side(side)
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("trend entry price must be finite and positive")
    if not math.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("trend ATR must be finite and positive")
    if closed_bars_1h.empty:
        raise ValueError("trend stop requires closed 1H bars")

    policy = settings.trend_strategy
    swing = closed_bars_1h.tail(policy.stop_swing_lookback_1h)
    if len(swing) < policy.stop_swing_lookback_1h:
        raise ValueError("insufficient_closed_bars_for_trend_stop")
    structure = float(swing["low"].min() if side is Side.LONG else swing["high"].max())
    if not math.isfinite(structure) or structure <= 0:
        raise ValueError("trend structure level must be finite and positive")
    if side is Side.LONG and structure >= entry_price:
        raise ValueError("trend_structure_not_beyond_entry")
    if side is Side.SHORT and structure <= entry_price:
        raise ValueError("trend_structure_not_beyond_entry")

    if policy.stop_mode == "structure":
        distance = ((entry_price - structure) if side is Side.LONG else (structure - entry_price))
        distance += policy.structure_buffer_atr * atr_value
    else:
        distance = policy.atr_stop_multiple * atr_value
    stop = entry_price - distance if side is Side.LONG else entry_price + distance
    if stop <= 0 or distance <= 0:
        raise ValueError("trend stop must be positive and on the protective side")
    distance_atr = distance / atr_value
    if distance_atr < policy.min_stop_atr or distance_atr > policy.max_stop_atr:
        raise ValueError("trend_stop_distance_out_of_range")
    return float(stop), structure


def calculate_trend_quantity(equity: float, entry_price: float, stop_price: float,
                             settings, *, step_size: float, minimum_quantity: float = 0.0,
                             existing_gross_notional: float = 0.0,
                             gross_notional_cap_pct: float | None = None) -> tuple[float, float, float]:
    """Size from account risk and stop loss, then cap gross exposure and leverage."""
    values = (equity, entry_price, stop_price, step_size, minimum_quantity, existing_gross_notional)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("trend sizing inputs must be finite")
    if equity <= 0 or entry_price <= 0 or stop_price <= 0 or step_size <= 0:
        raise ValueError("trend equity, prices, and quantity step must be positive")
    if minimum_quantity < 0 or existing_gross_notional < 0:
        raise ValueError("trend minimum quantity and existing exposure cannot be negative")
    if entry_price == stop_price:
        raise ValueError("trend stop distance must be positive")

    policy = settings.trend_strategy
    fee_rate = (float(settings.fees.taker_bps) + float(settings.fees.slippage_bps)) / 10_000
    risk_per_unit = abs(entry_price - stop_price) + (entry_price + stop_price) * fee_rate
    if not math.isfinite(risk_per_unit) or risk_per_unit <= 0:
        raise ValueError("trend risk per unit must be finite and positive")
    risk_budget = equity * policy.risk_per_trade_pct
    risk_quantity = risk_budget / risk_per_unit

    cap_pct = float(settings.sizing.max_gross_notional_pct)
    if gross_notional_cap_pct is not None:
        if not math.isfinite(gross_notional_cap_pct) or gross_notional_cap_pct < 0:
            raise ValueError("trend gross notional cap must be finite and non-negative")
        cap_pct = min(cap_pct, gross_notional_cap_pct)
    remaining_gross = max(0.0, equity * cap_pct - existing_gross_notional)
    leverage_cap = min(3.0, float(settings.sizing.leverage))
    maximum_quantity = min(equity * leverage_cap, remaining_gross) / entry_price
    raw_quantity = min(risk_quantity, maximum_quantity)
    step = Decimal(str(step_size))
    quantity = (Decimal(str(raw_quantity)) / step).to_integral_value(rounding=ROUND_FLOOR) * step
    if quantity < Decimal(str(minimum_quantity)):
        return 0.0, float(risk_budget), float(risk_per_unit)
    return float(quantity), float(risk_budget), float(risk_per_unit)


def evaluate_trend_entry(symbol: str, bars_1h: pd.DataFrame | None, bars_4h: pd.DataFrame | None,
                         settings, *, now: datetime, market_risk_state: str = "NORMAL",
                         position_source: str | None = None, has_open_position: bool = False,
                         pending_source: str | None = None, equity: float,
                         step_size: float, minimum_quantity: float = 0.0,
                         existing_gross_notional: float = 0.0,
                         gross_notional_cap_pct: float | None = None,
                         symbol_regime_assessment: SymbolRegimeAssessment | None = None) -> TrendEntryDecision:
    """Generate one trend pullback plan from completed bars; never executes orders.

    ``symbol_regime_assessment`` lets a caller pass an already-computed 4H
    classification (e.g. from a runner cycle that just classified the same closed
    bars), avoiding the duplicate classify_symbol_regime call. When omitted the
    classification is computed here, preserving the original behavior exactly.
    """
    if not symbol.strip() or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("trend entry requires a symbol and timezone-aware timestamp")
    excluded = {"BTCUSDT", "ETHUSDT"}
    excluded.update(normalize_symbol(value) for value in settings.universe.exclude_symbols)
    if normalize_symbol(symbol) in excluded:
        return _blocked_entry(symbol, now, "trend_symbol_excluded")

    try:
        closed_1h = _closed_bars(bars_1h, now, HOUR)
        if closed_1h.empty:
            return _blocked_entry(symbol, now, "no_closed_1h_bars")
        signal_bar_time = (closed_1h.index[-1] + HOUR).to_pydatetime()
        if (pd.Timestamp(now) - pd.Timestamp(signal_bar_time)).total_seconds() < 0:
            return _blocked_entry(symbol, now, "latest_1h_bar_unclosed")
        expires_at = min(
            pd.Timestamp(now).to_pydatetime()
            + timedelta(minutes=settings.trend_strategy.pullback_ttl_minutes),
            signal_bar_time + timedelta(hours=1),
        )
        if expires_at <= pd.Timestamp(now).to_pydatetime():
            return _blocked_entry(symbol, signal_bar_time, "trend_pullback_expired",
                                  signal_bar_time=signal_bar_time,
                                  evidence={"expires_at": expires_at.isoformat()})
        required_1h = max(settings.trend_strategy.entry_ema_bars,
                          settings.trend_strategy.exit_ema_bars,
                          settings.lookback.atr_len + 1,
                          settings.trend_strategy.stop_swing_lookback_1h)
        if len(closed_1h) < required_1h:
            return _blocked_entry(symbol, signal_bar_time, "insufficient_closed_1h_bars",
                                  signal_bar_time=signal_bar_time,
                                  evidence={"closed_bar_count": int(len(closed_1h)),
                                            "required_bar_count": int(required_1h)})
        recent_1h = closed_1h.tail(required_1h).index
        if len(recent_1h) > 1 and not (recent_1h[1:] - recent_1h[:-1] == HOUR).all():
            return _blocked_entry(symbol, signal_bar_time, "closed_1h_bar_gap",
                                  signal_bar_time=signal_bar_time)

        assessment = symbol_regime_assessment
        if assessment is None:
            assessment = classify_symbol_regime(
                symbol, bars_4h, settings, now=signal_bar_time,
            )
        if assessment.state is not SymbolRegimeState.TREND:
            return _blocked_entry(symbol, signal_bar_time, assessment.reason,
                                  assessment=assessment, signal_bar_time=signal_bar_time)
        if assessment.reason == "trend_confirmed_up":
            side = Side.LONG
        elif assessment.reason == "trend_confirmed_down":
            side = Side.SHORT
        else:
            return _blocked_entry(symbol, signal_bar_time, "trend_direction_unavailable",
                                  assessment=assessment, signal_bar_time=signal_bar_time)

        route = resolve_strategy_route(
            symbol, assessment.state.value, market_risk_state,
            position_source=position_source, at=signal_bar_time,
            trend_strategy_ready=True,
        )
        route_details = route.to_dict()
        if not route.entry_allowed:
            return _blocked_entry(symbol, signal_bar_time, route.entry_block_reason or route.route_reason,
                                  source=ROUTER, assessment=assessment,
                                  signal_bar_time=signal_bar_time, route=route_details)
        if has_open_position:
            return _blocked_entry(symbol, signal_bar_time, "position_already_open",
                                  source=ROUTER, assessment=assessment,
                                  signal_bar_time=signal_bar_time, route=route_details)
        if pending_source is not None:
            return _blocked_entry(symbol, signal_bar_time, "pending_entry_exists",
                                  source=ROUTER, assessment=assessment,
                                  signal_bar_time=signal_bar_time,
                                  route={**route_details, "pending_source": pending_source})

        close = float(closed_1h["close"].iloc[-1])
        open_price = float(closed_1h["open"].iloc[-1])
        ema_entry = closed_1h["close"].ewm(
            span=settings.trend_strategy.entry_ema_bars, adjust=False,
            min_periods=settings.trend_strategy.entry_ema_bars,
        ).mean().iloc[-1]
        if not math.isfinite(float(ema_entry)):
            return _blocked_entry(symbol, signal_bar_time, "trend_entry_ema_unavailable",
                                  assessment=assessment, signal_bar_time=signal_bar_time)
        directional_bar = (close > open_price if side is Side.LONG else close < open_price)
        correct_ema_side = (close > float(ema_entry) if side is Side.LONG else close < float(ema_entry))
        if not directional_bar or not correct_ema_side:
            reason = "trend_1h_not_aligned_up" if side is Side.LONG else "trend_1h_not_aligned_down"
            return _blocked_entry(symbol, signal_bar_time, reason, assessment=assessment,
                                  signal_bar_time=signal_bar_time,
                                  route=route_details,
                                  evidence={"close": close, "open": open_price,
                                            "entry_ema": float(ema_entry),
                                            "directional_bar": directional_bar,
                                            "correct_ema_side": correct_ema_side})

        atr_value = float(atr(closed_1h, settings.lookback.atr_len).iloc[-1])
        if not math.isfinite(atr_value) or atr_value <= 0:
            return _blocked_entry(symbol, signal_bar_time, "trend_atr_unavailable",
                                  assessment=assessment, signal_bar_time=signal_bar_time,
                                  route=route_details)
        signal_bar = closed_1h.iloc[-1]
        midpoint = (float(signal_bar["high"]) + float(signal_bar["low"])) / 2
        distance = settings.trend_strategy.pullback_atr * atr_value
        if side is Side.LONG:
            pullback_price = max(close - distance, midpoint)
            if pullback_price >= close:
                pullback_price = close - distance
        else:
            pullback_price = min(close + distance, midpoint)
            if pullback_price <= close:
                pullback_price = close + distance
        try:
            stop_price, structure_price = calculate_trend_stop(
                side, pullback_price, atr_value, closed_1h, settings,
            )
        except ValueError as exc:
            reason = str(exc) if str(exc).startswith("trend_") else "trend_stop_invalid"
            return _blocked_entry(symbol, signal_bar_time, reason,
                                  assessment=assessment, signal_bar_time=signal_bar_time,
                                  route=route_details,
                                  evidence={"entry_price": pullback_price, "atr_1h": atr_value,
                                            "error": str(exc)[:300]})
        quantity, risk_budget, risk_per_unit = calculate_trend_quantity(
            equity, pullback_price, stop_price, settings,
            step_size=step_size, minimum_quantity=minimum_quantity,
            existing_gross_notional=existing_gross_notional,
            gross_notional_cap_pct=gross_notional_cap_pct,
        )
        if quantity <= 0:
            return _blocked_entry(symbol, signal_bar_time, "trend_risk_size_below_exchange_minimum",
                                  assessment=assessment, signal_bar_time=signal_bar_time,
                                  route=route_details,
                                  evidence={"risk_budget": risk_budget,
                                            "risk_per_unit": risk_per_unit,
                                            "pullback_price": pullback_price,
                                            "stop_price": stop_price})

        signal = _context_signal(symbol, side, Action.OPEN, "trend_pullback_candidate", TREND,
                                 signal_bar_time)
        decision = TrendEntryDecision(
            signal=apply_entry_route(signal, route),
            assessment=assessment,
            signal_bar_time=signal_bar_time,
            confirmation_bar_time=assessment.last_closed_bar_at,
            reference_price=close,
            pullback_price=float(pullback_price),
            expires_at=expires_at,
            stop_price=stop_price,
            structure_break_price=structure_price,
            quantity=quantity,
            risk_budget=risk_budget,
            risk_per_unit=risk_per_unit,
            route=route_details,
            evidence={
                "entry_ema": float(ema_entry),
                "atr_1h": atr_value,
                "signal_bar_open": open_price,
                "signal_bar_high": float(signal_bar["high"]),
                "signal_bar_low": float(signal_bar["low"]),
                "pullback_ttl_minutes": settings.trend_strategy.pullback_ttl_minutes,
                "pullback_trigger": "fresh_quote_touch",
                "stop_mode": settings.trend_strategy.stop_mode,
                "stop_distance_atr": abs(pullback_price - stop_price) / atr_value,
                "risk_per_trade_pct": settings.trend_strategy.risk_per_trade_pct,
                "leverage_cap": min(3.0, float(settings.sizing.leverage)),
            },
        )
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return _blocked_entry(symbol, now, "trend_input_invalid",
                              evidence={"error_type": type(exc).__name__, "error": str(exc)[:300]})

    logger.opt(lazy=True).info("Trend strategy entry decision {}", lambda: json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True))
    return decision


def initialize_trend_exit_state(entry_price: float, stop_price: float,
                                structure_break_price: float | None, *,
                                stop_mode: str = "structure") -> dict[str, Any]:
    """Create the minimal mutable state required by the trend exit evaluator."""
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(stop_price) or stop_price <= 0:
        raise ValueError("trend exit state requires finite positive entry and stop prices")
    if entry_price == stop_price:
        raise ValueError("trend initial risk must be positive")
    if stop_mode not in {"structure", "atr"}:
        raise ValueError("trend stop mode must be structure or atr")
    return {
        "initial_stop_price": float(stop_price),
        "active_stop_price": float(stop_price),
        "active_stop_reason": "structure_stop" if stop_mode == "structure" else "atr_stop",
        "structure_break_price": (float(structure_break_price)
                                   if structure_break_price is not None else None),
        "initial_risk": abs(float(entry_price) - float(stop_price)),
        "best_favorable_price": float(entry_price),
        "last_processed_close": None,
    }


def _stop_was_hit(side: Side, low: float, high: float, stop: float) -> bool:
    return low <= stop if side is Side.LONG else high >= stop


def _breakeven_stop(side: Side, entry_price: float, settings) -> float:
    fee_rate = float(settings.fees.taker_bps) / 10_000
    exit_cost = (float(settings.fees.taker_bps) + float(settings.fees.slippage_bps)) / 10_000
    if side is Side.LONG:
        return entry_price * (1 + fee_rate) / (1 - exit_cost)
    return entry_price * (1 - fee_rate) / (1 + exit_cost)


def _log_trend_exit_decision(decision: TrendExitDecision, enabled: bool) -> None:
    if enabled:
        logger.info("Trend strategy exit decision {}",
                    json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True))


def evaluate_trend_exit(position: Position, bars_1h: pd.DataFrame | None, settings, *, now: datetime,
                        exit_state: dict[str, Any], current_price: float | None = None,
                        log_decision: bool = True) -> TrendExitDecision:
    """Evaluate trend-owned exits from completed 1H bars and an optional live stop quote."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("trend exit evaluation time must be timezone-aware")
    state = dict(exit_state)
    side = position.side
    initial_stop = float(state.get("initial_stop_price", position.stop_price))
    active_stop = float(state.get("active_stop_price", position.stop_price))
    active_stop_reason = str(state.get("active_stop_reason", "structure_stop"))
    initial_risk = float(state.get("initial_risk", abs(position.entry_price - initial_stop)))
    if position.source != TREND:
        signal = _context_signal(position.symbol, None, Action.NONE, "position_not_owned_by_trend", ROUTER, now)
        decision = TrendExitDecision(signal, active_stop, state, {"position_source": position.source})
        _log_trend_exit_decision(decision, log_decision)
        return decision
    expected_initial_risk = abs(position.entry_price - initial_stop)
    if (not math.isfinite(initial_stop) or initial_stop <= 0
            or not math.isfinite(active_stop) or active_stop <= 0
            or not math.isfinite(initial_risk) or initial_risk <= 0
            or not math.isclose(initial_risk, expected_initial_risk, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("trend exit state has invalid stop or initial risk")
    if current_price is not None and (not math.isfinite(current_price) or current_price <= 0):
        raise ValueError("trend stop quote must be finite and positive")

    closed = _closed_bars(bars_1h, now, HOUR)
    entry_at = pd.Timestamp(position.opened_at)
    first_full_bar = entry_at.floor("h") + HOUR
    if closed.empty:
        after_entry = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ema = pd.Series(dtype=float)
    else:
        after_entry = closed.loc[closed.index >= first_full_bar]
        last_processed = state.get("last_processed_close")
        if last_processed:
            after_entry = after_entry.loc[(after_entry.index + HOUR) > pd.Timestamp(last_processed)]
        ema = closed["close"].ewm(
            span=settings.trend_strategy.exit_ema_bars, adjust=False,
            min_periods=settings.trend_strategy.exit_ema_bars,
        ).mean()
    structure_break = state.get("structure_break_price")
    if structure_break is not None:
        structure_break = float(structure_break)
        if not math.isfinite(structure_break) or structure_break <= 0:
            raise ValueError("trend exit state has invalid structure level")
        if side is Side.LONG and structure_break >= position.entry_price:
            raise ValueError("long trend structure must be below entry")
        if side is Side.SHORT and structure_break <= position.entry_price:
            raise ValueError("short trend structure must be above entry")
    if ((side is Side.LONG and initial_stop >= position.entry_price)
            or (side is Side.SHORT and initial_stop <= position.entry_price)):
        raise ValueError("trend initial stop must be on the protective side of entry")
    best = float(state.get("best_favorable_price", position.entry_price))

    for bar_at, candle in after_entry.iterrows():
        bar_close_at = (bar_at + HOUR).to_pydatetime()
        candle_close = float(candle["close"])
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])
        # Historical OHLC checks the stop that was active before this bar; a stop
        # tightened from this bar's high/low only becomes active after its close.
        if _stop_was_hit(side, candle_low, candle_high, active_stop):
            signal = _context_signal(position.symbol, side, Action.CLOSE,
                                     f"trend_{active_stop_reason}_hit", TREND, bar_close_at)
            state["last_processed_close"] = bar_close_at.isoformat()
            decision = TrendExitDecision(signal, active_stop, state,
                                         {"bar_low": candle_low, "bar_high": candle_high,
                                          "trigger_stop": active_stop})
            _log_trend_exit_decision(decision, log_decision)
            return decision

        if structure_break is not None:
            broken = (candle_close < float(structure_break) if side is Side.LONG
                      else candle_close > float(structure_break))
            if broken:
                signal = _context_signal(position.symbol, side, Action.CLOSE,
                                         "trend_structure_break", TREND, bar_close_at)
                state["last_processed_close"] = bar_close_at.isoformat()
                decision = TrendExitDecision(signal, active_stop, state,
                                             {"signal_close": candle_close,
                                              "structure_break_price": float(structure_break)})
                _log_trend_exit_decision(decision, log_decision)
                return decision

        ema_value = float(ema.loc[bar_at])
        if math.isfinite(ema_value):
            ema_broken = candle_close < ema_value if side is Side.LONG else candle_close > ema_value
            if ema_broken:
                signal = _context_signal(position.symbol, side, Action.CLOSE,
                                         "trend_key_ema_break", TREND, bar_close_at)
                state["last_processed_close"] = bar_close_at.isoformat()
                decision = TrendExitDecision(signal, active_stop, state,
                                             {"signal_close": candle_close,
                                              "exit_ema": ema_value,
                                              "exit_ema_bars": settings.trend_strategy.exit_ema_bars})
                _log_trend_exit_decision(decision, log_decision)
                return decision

        best = max(best, candle_high) if side is Side.LONG else min(best, candle_low)
        favorable = (best - position.entry_price if side is Side.LONG
                     else position.entry_price - best)
        if favorable >= settings.trend_strategy.breakeven_trigger_r * initial_risk:
            breakeven = _breakeven_stop(side, position.entry_price, settings)
            improved = max(active_stop, breakeven) if side is Side.LONG else min(active_stop, breakeven)
            if improved != active_stop:
                active_stop, active_stop_reason = improved, "breakeven_stop"
        if favorable >= settings.trend_strategy.trailing_trigger_r * initial_risk:
            atr_value = float(atr(closed.loc[:bar_at], settings.lookback.atr_len).iloc[-1])
            if math.isfinite(atr_value) and atr_value > 0:
                distance = settings.trend_strategy.trailing_atr_multiple * atr_value
                trailing = best - distance if side is Side.LONG else best + distance
                improved = max(active_stop, trailing) if side is Side.LONG else min(active_stop, trailing)
                if improved != active_stop:
                    active_stop, active_stop_reason = improved, "trailing_stop"
        state["best_favorable_price"] = best
        state["active_stop_price"] = active_stop
        state["active_stop_reason"] = active_stop_reason
        state["last_processed_close"] = bar_close_at.isoformat()

    if current_price is not None and _stop_was_hit(side, current_price, current_price, active_stop):
        signal = _context_signal(position.symbol, side, Action.CLOSE,
                                 f"trend_{active_stop_reason}_hit", TREND, now)
        decision = TrendExitDecision(signal, active_stop, state,
                                     {"current_price": current_price, "trigger_stop": active_stop})
    else:
        signal = _context_signal(position.symbol, None, Action.NONE, "trend_position_held", TREND,
                                 (after_entry.index[-1] + HOUR).to_pydatetime()
                                 if not after_entry.empty else now)
        decision = TrendExitDecision(signal, active_stop, state,
                                     {"bars_processed": int(len(after_entry)),
                                      "best_favorable_price": best})
    _log_trend_exit_decision(decision, log_decision)
    return decision
