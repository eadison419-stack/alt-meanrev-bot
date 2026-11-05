"""Causal, report-only audit of the market and per-symbol strategy router."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from bot.backtest.engine import _snapshot
from bot.backtest.portfolio import _historical_rank, _quote_turnover
from bot.data.klines import TIMEFRAME_MS, normalize_symbol
from bot.indicators import atr, validate_ohlcv
from bot.market_risk import classify_market_risk
from bot.regime import classify_regimes
from bot.signals import Action, Signal, generate_signal, with_signal_context
from bot.strategy_router import MEAN_REVERSION, TREND, apply_entry_route, resolve_strategy_route
from bot.symbol_regime import SymbolRegimeState, classify_symbol_regime
from bot.time_filter import evaluate_time_filter
from bot.trend_strategy import evaluate_trend_entry

HOUR = pd.Timedelta(TIMEFRAME_MS["1h"], unit="ms")
FOUR_HOURS = pd.Timedelta(TIMEFRAME_MS["4h"], unit="ms")
EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}
STALE_GRACE = pd.Timedelta(5, unit="min")


@dataclass(frozen=True)
class RoutingAuditResult:
    """Hourly routing decisions; this sidecar never opens or manages positions."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _last_funding(funding: pd.DataFrame | None, close_time: pd.Timestamp) -> tuple[float | None, str | None]:
    if funding is None or funding.empty:
        return None, None
    if not isinstance(funding.index, pd.DatetimeIndex) or funding.index.tz is None:
        raise ValueError("funding timestamps must be timezone-aware")
    index = int(funding.index.searchsorted(close_time, side="left") - 1)
    if index < 0:
        return None, None
    rate = float(funding["funding_rate"].iloc[index])
    if not np.isfinite(rate):
        return None, funding.index[index].isoformat()
    return rate, funding.index[index].isoformat()


def _cache_expired(assessment, now: pd.Timestamp) -> bool:
    last_closed = getattr(assessment, "last_closed_bar_at", None)
    return (last_closed is not None
            and now - pd.Timestamp(last_closed) > FOUR_HOURS + STALE_GRACE
            and getattr(assessment, "reason", "") != "stale_closed_bar")


def _mean_reversion_candidate(
    symbol: str,
    close_time: pd.Timestamp,
    open_time: pd.Timestamp,
    candle: pd.Series,
    one_hour: pd.DataFrame,
    four_hour: pd.DataFrame,
    regimes: pd.DataFrame,
    atr_values: pd.Series,
    settings,
) -> tuple[Signal, dict[str, Any]]:
    """Apply the existing mean-reversion generator to the just-closed bars."""
    one_close = one_hour.index + HOUR
    one_prefix = one_hour.loc[one_close <= close_time]
    four_close = four_hour.index + FOUR_HOURS
    four_prefix = four_hour.loc[four_close <= close_time]
    idx = int(one_prefix.index.searchsorted(open_time, side="right") - 1)
    if idx < 0 or one_prefix.index[idx] != open_time:
        return Signal(None, Action.NONE, "closed_1h_bar_missing", 0.0), {}
    close_ns = regimes.index.asi8 + FOUR_HOURS.value
    ridx = int(np.searchsorted(close_ns, close_time.value, side="right") - 1)
    if ridx < 0 or four_prefix.empty:
        return Signal(None, Action.NONE, "entry_regime_unavailable", 0.0), {}
    regime = _snapshot(regimes.iloc[ridx], regimes.index[ridx])
    if regime is None:
        return Signal(None, Action.NONE, "entry_regime_unavailable", 0.0), {}
    atr_value = atr_values.loc[open_time]
    if not np.isfinite(atr_value) or float(atr_value) <= 0:
        return Signal(None, Action.NONE, "entry_atr_unavailable", 0.0), {}

    history = one_prefix.tail(48)
    flags = evaluate_time_filter(
        close_time.to_pydatetime(), history, settings,
        precomputed_atr=atr_values.reindex(history.index),
    )
    regime_row = regimes.iloc[ridx]
    swing = one_prefix.tail(settings.stops.swing_lookback_1h)
    range_low, range_high = regime_row.get("range_low"), regime_row.get("range_high")
    signal = generate_signal(
        regime, candle, flags, settings,
        bars_1h=one_prefix, bars_4h=four_prefix,
        entry_reference_price=float(candle["close"]), entry_atr=float(atr_value),
        range_low=(float(range_low) if range_low is not None and np.isfinite(range_low) else None),
        range_high=(float(range_high) if range_high is not None and np.isfinite(range_high) else None),
        structure_low=float(swing["low"].min()) if not swing.empty else None,
        structure_high=float(swing["high"].max()) if not swing.empty else None,
    )
    context = {
        "mean_reversion_regime": regime.regime.value,
        "mean_reversion_atr_1h": float(atr_value),
        "time_filter": flags.to_dict() if hasattr(flags, "to_dict") else vars(flags),
        "mean_reversion_range_low": (float(range_low) if range_low is not None and np.isfinite(range_low) else None),
        "mean_reversion_range_high": (float(range_high) if range_high is not None and np.isfinite(range_high) else None),
    }
    return signal, context


def run_routing_audit(
    bars_1h: dict[str, pd.DataFrame],
    bars_4h: dict[str, pd.DataFrame],
    market_bars_4h: pd.DataFrame | None,
    funding: dict[str, pd.DataFrame],
    onboard_ms: dict[str, int],
    settings,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int = 20,
    initial_equity: float = 10_000.0,
) -> RoutingAuditResult:
    """Replay one routing decision per closed 1H candidate without execution.

    The portfolio backtester remains the sole PnL simulator. This sidecar records
    each causal classification, strategy selection, signal and rejection reason.
    In particular, a TREND pullback candidate is never counted as a filled trade.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if (start.tzinfo is None or end.tzinfo is None or end <= start
            or not 1 <= top_n <= 20 or initial_equity <= 0):
        raise ValueError("routing audit requires ordered aware bounds, top_n 1..20, and positive equity")
    if market_bars_4h is not None and not market_bars_4h.empty:
        validate_ohlcv(market_bars_4h)

    excluded = EXCLUDED_SYMBOLS | {normalize_symbol(value) for value in settings.universe.exclude_symbols}
    symbols = sorted(set(bars_1h).intersection(bars_4h))
    symbols = [symbol for symbol in symbols if normalize_symbol(symbol) not in excluded]
    timelines: dict[str, pd.DataFrame] = {}
    turnovers: dict[str, np.ndarray] = {}
    regimes: dict[str, pd.DataFrame] = {}
    atr_values: dict[str, pd.Series] = {}
    for symbol in symbols:
        one, four = bars_1h[symbol], bars_4h[symbol]
        validate_ohlcv(one)
        validate_ohlcv(four)
        if one.index.tz is None or four.index.tz is None:
            raise ValueError(f"{symbol} candle timestamps must be timezone-aware")
        timelines[symbol] = one.loc[(one.index >= start - pd.Timedelta(30, unit="D"))
                                    & (one.index < end)].copy()
        if timelines[symbol].empty:
            continue
        turnovers[symbol] = _quote_turnover(timelines[symbol])
        regimes[symbol] = classify_regimes(four, settings)
        atr_values[symbol] = atr(timelines[symbol], settings.lookback.atr_len)
    symbols = [symbol for symbol in symbols if symbol in timelines]
    timestamps = sorted({timestamp + HOUR for symbol in symbols
                         for timestamp in timelines[symbol].index
                         if start <= timestamp + HOUR <= end})
    if not timestamps:
        raise ValueError("no closed 1H bars fall within the routing audit window")

    rows: list[dict[str, Any]] = []
    market_cache: dict[pd.Timestamp, Any] = {}
    symbol_cache: dict[tuple[str, pd.Timestamp], Any] = {}
    index_maps = {symbol: {at: i for i, at in enumerate(timelines[symbol].index)} for symbol in symbols}
    for close_time in timestamps:
        market_key = close_time.floor("4h")
        if market_key not in market_cache or _cache_expired(market_cache[market_key], close_time):
            if market_bars_4h is None or market_bars_4h.empty:
                market_history = market_bars_4h
            else:
                market_history = market_bars_4h.loc[market_bars_4h.index + FOUR_HOURS <= close_time]
            market_cache[market_key] = classify_market_risk(
                market_history, settings, now=close_time.to_pydatetime(),
            )
        market = market_cache[market_key]
        market_evidence = dict(market.evidence)
        market_last_closed = market_evidence.get("last_closed_bar_at")
        if market_last_closed:
            market_evidence["closed_bar_age_seconds"] = (
                close_time - pd.Timestamp(market_last_closed)
            ).total_seconds()
        market_evidence["as_of_timestamp"] = close_time.isoformat()
        market = replace(market, checked_at=close_time.to_pydatetime(), evidence=market_evidence)
        ranked = _historical_rank(close_time, timelines, turnovers, onboard_ms, settings, limit=top_n)
        ranked = [symbol for symbol in ranked if normalize_symbol(symbol) not in excluded]
        open_time = close_time - HOUR
        for symbol in ranked:
            index = index_maps[symbol].get(open_time)
            if index is None:
                continue
            candle = timelines[symbol].iloc[index]
            regime_key = (symbol, close_time.floor("4h"))
            if regime_key not in symbol_cache or _cache_expired(symbol_cache[regime_key], close_time):
                symbol_four = bars_4h[symbol].loc[bars_4h[symbol].index + FOUR_HOURS <= close_time]
                symbol_cache[regime_key] = classify_symbol_regime(
                    symbol, symbol_four, settings, now=close_time.to_pydatetime(),
                )
            cached_assessment = symbol_cache[regime_key]
            symbol_evidence = dict(cached_assessment.evidence)
            if cached_assessment.last_closed_bar_at is not None:
                symbol_evidence["closed_bar_age_seconds"] = (
                    close_time - pd.Timestamp(cached_assessment.last_closed_bar_at)
                ).total_seconds()
            assessment = replace(
                cached_assessment, checked_at=close_time.to_pydatetime(), evidence=symbol_evidence,
            )
            route = resolve_strategy_route(
                symbol, assessment.state.value, market.state.value,
                position_source=None, at=close_time.to_pydatetime(),
                trend_strategy_ready=True,
            )
            market_size_multiplier = (float(settings.market_risk.elevated_size_multiplier)
                                      if market.state.value == "RISK_ELEVATED" else 1.0)

            signal: Signal
            signal_context: dict[str, Any] = {}
            if assessment.state is SymbolRegimeState.RANGE:
                raw_signal, signal_context = _mean_reversion_candidate(
                    symbol, close_time, open_time, candle, timelines[symbol], bars_4h[symbol],
                    regimes[symbol], atr_values[symbol], settings,
                )
                signal = with_signal_context(
                    raw_signal, symbol=symbol, source=MEAN_REVERSION,
                    timestamp=close_time.to_pydatetime(),
                )
                routed_signal = apply_entry_route(signal, route)
            elif assessment.state is SymbolRegimeState.TREND:
                one_hour_closed = timelines[symbol].loc[timelines[symbol].index + HOUR <= close_time]
                four_hour_closed = bars_4h[symbol].loc[bars_4h[symbol].index + FOUR_HOURS <= close_time]
                decision = evaluate_trend_entry(
                    symbol, one_hour_closed, four_hour_closed, settings,
                    now=close_time.to_pydatetime(), market_risk_state=market.state.value,
                    equity=initial_equity * market_size_multiplier,
                    step_size=1e-8, minimum_quantity=0.0,
                )
                signal = decision.signal
                signal_context = {"trend_decision": decision.to_dict(),
                                  "market_size_multiplier": market_size_multiplier}
                routed_signal = apply_entry_route(signal, route)
            else:
                signal = with_signal_context(
                    Signal(None, Action.NONE, assessment.reason, 0.0),
                    symbol=symbol, source="ROUTER", timestamp=close_time.to_pydatetime(),
                )
                routed_signal = signal

            raw_signal_time = signal.timestamp
            if raw_signal_time is not None and pd.Timestamp(raw_signal_time) > close_time:
                raise AssertionError(f"future signal timestamp for {symbol}: {raw_signal_time}")
            funding_rate, funding_at = _last_funding(funding.get(symbol), close_time)
            closed_1h_at = open_time + HOUR
            if closed_1h_at > close_time:
                raise AssertionError(f"unclosed 1H candle included for {symbol}")
            accepted_candidate = routed_signal.action is Action.OPEN
            rejection = None
            if signal.action is Action.OPEN and not accepted_candidate:
                rejection = routed_signal.reason
            if accepted_candidate and funding_rate is None:
                accepted_candidate = False
                rejection = "funding_rate_missing"
            elif (accepted_candidate and abs(funding_rate) >= settings.risk.funding_abs_block):
                accepted_candidate = False
                rejection = "funding_rate_blocked"
            if rejection is None and route.entry_block_reason:
                rejection = route.entry_block_reason
            execution_state = "no_entry_signal"
            if signal.action is Action.OPEN and not accepted_candidate:
                execution_state = "rejected"
            if rejection in {"funding_rate_missing", "funding_rate_blocked"}:
                execution_state = "risk_rejected"
            elif accepted_candidate and route.source == TREND:
                execution_state = "not_filled_by_decision_audit"
            elif accepted_candidate:
                execution_state = "baseline_fill_engine_required"
            elif route.entry_block_reason:
                execution_state = "route_blocked"
            elif signal.action is Action.NONE and signal.reason not in {"none", ""}:
                execution_state = "signal_filtered"
            rows.append({
                "decision_time": close_time.isoformat(),
                "closed_1h_bar_time": closed_1h_at.isoformat(),
                "closed_4h_bar_time": (assessment.last_closed_bar_at.isoformat()
                                        if assessment.last_closed_bar_at else None),
                "symbol": symbol,
                "hot_rank": ranked.index(symbol) + 1,
                "market_risk": market.to_dict(),
                "symbol_regime": assessment.to_dict(),
                "route": route.to_dict(),
                "route_reject_reason": route.entry_block_reason,
                "source": signal.source,
                "signal_source": route.source,
                "raw_signal_source": signal.source,
                "decision_source": routed_signal.source,
                "signal_direction": routed_signal.direction,
                "signal_action": routed_signal.action.value,
                "signal_reason": routed_signal.reason,
                "raw_signal_action": signal.action.value,
                "raw_signal_reason": signal.reason,
                "signal_time": raw_signal_time.isoformat() if raw_signal_time else None,
                "market_size_multiplier": market_size_multiplier,
                "accepted_candidate": accepted_candidate,
                "reject_reason": rejection,
                "execution_state": execution_state,
                "funding_rate_asof": funding_rate,
                "funding_observed_at": funding_at,
                "signal_context": signal_context,
            })

    keys = [(row["decision_time"], row["symbol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("routing audit contains duplicate symbol decisions in one round")
    transitions: list[dict[str, str]] = []
    previous: dict[str, str] = {}
    for row in rows:
        symbol, state = row["symbol"], row["symbol_regime"]["state"]
        if symbol in previous and previous[symbol] != state:
            transitions.append({
                "symbol": symbol, "decision_time": row["decision_time"],
                "from": previous[symbol], "to": state,
            })
        previous[symbol] = state

    rejected: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    source_candidates: dict[str, int] = {}
    for row in rows:
        state = row["symbol_regime"]["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
        source = row["signal_source"]
        if row["accepted_candidate"] and source:
            source_candidates[source] = source_candidates.get(source, 0) + 1
        reason = row["reject_reason"]
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
    pause_open = sum(row["accepted_candidate"] and row["market_risk"]["state"] == "PAUSE_NEW"
                     for row in rows)
    uncertain_open = sum(row["accepted_candidate"] and row["symbol_regime"]["state"] == "UNCERTAIN"
                         for row in rows)
    if pause_open or uncertain_open:
        raise AssertionError("routing audit allowed an entry under PAUSE_NEW or UNCERTAIN")
    summary = {
        "decision_count": len(rows),
        "state_decisions": state_counts,
        "accepted_signal_candidates_by_source": source_candidates,
        "rejected_decisions_by_reason": dict(sorted(rejected.items())),
        "symbol_regime_transitions": transitions,
        "invariants": {
            "signal_time_not_after_latest_closed_1h_bar": True,
            "no_open_candidate_during_pause_new": pause_open == 0,
            "no_open_candidate_for_uncertain_symbol": uncertain_open == 0,
            "one_decision_per_symbol_per_hour": True,
            "btc_eth_excluded_from_candidate_pool": True,
            "position_ownership_and_switch_cooldown": (
                "validated by existing risk/router contract tests; this decision-only audit does not simulate positions"
            ),
        },
        "execution_scope": "decision_audit_only; use portfolio replay for fills and PnL",
        "open_signals_without_funding_asof": sum(
            row["raw_signal_action"] == Action.OPEN.value and row["funding_rate_asof"] is None
            for row in rows
        ),
    }
    return RoutingAuditResult(tuple(rows), summary)
