"""Synchronized multi-symbol backtest with historical hot-universe ranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from bot.backtest.engine import Trade, _apply_slippage, _snapshot
from bot.backtest.metrics import summarize_backtest
from bot.data.klines import TIMEFRAME_MS, rank_hot_candidates
from bot.indicators import atr, validate_ohlcv
from bot.market_risk import classify_market_risk
from bot.mr_candidate import staged_profit_lock_stop
from bot.portfolio import PortfolioState, Position
from bot.regime import Regime, classify_regimes
from bot.risk import OrderIntent, calculate_adaptive_stop, calculate_order_quantity, guard_signal
from bot.signals import (Action, Side, Signal, entry_rejection_reason, generate_signal,
                         with_signal_context)
from bot.strategy_router import MEAN_REVERSION, TREND, apply_entry_route, resolve_strategy_route
from bot.symbol_regime import SymbolRegimeState, classify_symbol_regime
from bot.time_filter import evaluate_time_filter
from bot.trend_strategy import (calculate_trend_quantity, calculate_trend_stop,
                                evaluate_trend_entry, evaluate_trend_exit,
                                initialize_trend_exit_state)

HOUR = pd.Timedelta(1, unit="h")
FOUR_HOURS = pd.Timedelta(4, unit="h")
FOUR_HOURS_NS = 4 * TIMEFRAME_MS["1h"]
PortfolioExitPolicy = Literal["breakeven_trail", "time_managed"]


def _range_deviation_signal(candle: pd.Series, four_hour_history: pd.DataFrame, settings,
                            flags, *, deviation_pct: float) -> Signal:
    """Experimental mean-reversion trigger for confirmed-RANGE symbols.

    Replaces the legacy LOW/HIGH gate with a deviation-from-EMA50 trigger:
    enter long when price closes a threshold below EMA50 on a bullish 1H bar,
    short when it closes a threshold above on a bearish 1H bar.
    """
    if flags.flatten_only:
        return Signal(None, Action.NONE, "entries_blocked", 0.0)
    ema = four_hour_history["close"].ewm(
        span=settings.symbol_regime.ema_slow_bars, adjust=False,
        min_periods=settings.symbol_regime.ema_slow_bars,
    ).mean()
    if ema.empty or not np.isfinite(float(ema.iloc[-1])) or float(ema.iloc[-1]) <= 0:
        return Signal(None, Action.NONE, "ema_unavailable", 0.0)
    close = float(candle["close"])
    deviation = close / float(ema.iloc[-1]) - 1.0
    bullish = close > float(candle["open"])
    bearish = close < float(candle["open"])
    if settings.signals.allow_long and deviation <= -deviation_pct and bullish and flags.allow_long_entry:
        return Signal(Side.LONG, Action.OPEN, "range_deviation_long", 1.0)
    if settings.signals.allow_short and deviation >= deviation_pct and bearish and flags.allow_short_entry:
        return Signal(Side.SHORT, Action.OPEN, "range_deviation_short", 1.0)
    return Signal(None, Action.NONE, "no_range_deviation", 0.0)


@dataclass
class _Active:
    position: Position
    entry_fee: float
    entry_slippage: float
    funding_paid: float
    initial_margin: float
    initial_risk: float
    protective_stop: float
    best_price: float
    entry_atr: float
    structure_low: float | None
    structure_high: float | None
    range_low: float | None
    range_high: float | None
    last_close: float
    no_progress_count: int = 0
    worst_price: float = float("inf")
    trend_exit_state: dict | None = None
    initial_quantity: float = 0.0
    tp1_taken: bool = False
    tp2_taken: bool = False
    reached_mid: bool = False
    mid_hold_count: int = 0
    mid_fired: bool = False


@dataclass(frozen=True)
class PortfolioBacktestResult:
    trades: tuple[Trade, ...]
    equity_curve: pd.Series
    metrics: dict[str, float | int | None]
    skipped_entries: dict[str, int]
    symbols_seen: int
    ranking_hours: int
    shadow_long_signals: tuple[dict, ...] = ()
    long_trade_excursions: tuple[dict, ...] = ()
    rejected_signals_by_source: dict[str, dict[str, int]] | None = None
    market_risk_execution_summary: dict[str, dict[str, float | int]] | None = None
    integrity_checks: dict[str, bool | int] | None = None
    result_fingerprint: str | None = None
    trend_funnel: tuple[dict, ...] = ()
    trend_funnel_fill_rejections: tuple[dict, ...] = ()
    trend_stops: tuple[dict, ...] = ()
    hot_universe_exemption_records: tuple[dict, ...] = ()


def _trade_overlap_counts(trades: tuple[Trade, ...] | list[Trade]) -> tuple[int, int, int]:
    by_symbol: dict[str, list[Trade]] = {}
    for trade in trades:
        by_symbol.setdefault(trade.symbol, []).append(trade)
    overlap_pairs = 0
    cross_source_pairs = 0
    invalid_intervals = 0
    for symbol_trades in by_symbol.values():
        active: list[Trade] = []
        for trade in sorted(symbol_trades, key=lambda row: (row.opened_at, row.closed_at, row.source)):
            if trade.closed_at <= trade.opened_at:
                invalid_intervals += 1
            active = [previous for previous in active if previous.closed_at > trade.opened_at]
            overlap_pairs += len(active)
            cross_source_pairs += sum(previous.source != trade.source for previous in active)
            active.append(trade)
    return overlap_pairs, cross_source_pairs, invalid_intervals


def _fingerprint_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _fingerprint_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _fingerprint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _result_fingerprint(trades: tuple[Trade, ...], equity_curve: pd.Series,
                        metrics: dict[str, float | int | None], skipped_entries: dict[str, int],
                        rejected_signals_by_source: dict[str, dict[str, int]],
                        market_risk_execution: dict[str, dict[str, float | int]],
                        integrity_checks: dict[str, bool | int], *, symbols_seen: int,
                        ranking_hours: int, shadow_long_signals: list[dict],
                        long_trade_excursions: list[dict]) -> str:
    trade_rows = []
    for trade in trades:
        trade_rows.append({
            "symbol": trade.symbol,
            "side": trade.side.value,
            "opened_at": trade.opened_at.isoformat(),
            "closed_at": trade.closed_at.isoformat(),
            "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price),
            "quantity": float(trade.quantity),
            "gross_pnl": float(trade.gross_pnl),
            "net_pnl": float(trade.net_pnl),
            "fees": float(trade.fees),
            "slippage_paid": float(trade.slippage_paid),
            "funding_paid": float(trade.funding_paid),
            "entry_reason": trade.entry_reason,
            "exit_reason": trade.exit_reason,
            "source": trade.source,
        })
    payload = {
        "trades": trade_rows,
        "equity_curve": [[timestamp.isoformat(), float(value)]
                         for timestamp, value in equity_curve.items()],
        "metrics": metrics,
        "skipped_entries": skipped_entries,
        "rejected_signals_by_source": rejected_signals_by_source,
        "market_risk_execution": market_risk_execution,
        "integrity_checks": integrity_checks,
        "symbols_seen": symbols_seen,
        "ranking_hours": ranking_hours,
        "shadow_long_signals": shadow_long_signals,
        "long_trade_excursions": long_trade_excursions,
    }
    encoded = json.dumps(_fingerprint_safe(payload), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _quote_turnover(frame: pd.DataFrame) -> np.ndarray:
    typical = frame[["open", "high", "low", "close"]].mean(axis=1).to_numpy(dtype=float)
    return frame["volume"].to_numpy(dtype=float) * typical


def _hot_candidates(
    timestamp: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    turnovers: dict[str, np.ndarray],
    onboard_ms: dict[str, int],
    settings,
) -> tuple[list[tuple[str, float]], dict[str, dict[str, float | None]], dict[str, dict]]:
    """Return the volume/listing-filtered candidate universe and its metrics.

    Mirrors live `filter_tradable_crypto_universe`: a symbol must be listed long
    enough, have a closed bar at this hour, and clear `min_quote_volume_24h`.
    Returns (candidates, activity, tickers) so callers can rank by hotness
    (`_historical_rank`) or gate on match scores (adaptive universe).
    """
    close_ms = int(timestamp.value // 1_000_000)
    min_age_ms = settings.universe.min_list_days * 86_400_000
    candidates: list[tuple[str, float]] = []
    activity: dict[str, dict[str, float | None]] = {}
    tickers: dict[str, dict] = {}
    for symbol, frame in frames.items():
        if close_ms - onboard_ms.get(symbol, close_ms) < min_age_ms:
            continue
        bar_time = timestamp - HOUR
        index = frame.index.searchsorted(bar_time, side="right") - 1
        if index < 24 or frame.index[index] != bar_time:
            continue
        quote = turnovers[symbol]
        turnover_24h = float(np.sum(quote[index - 23:index + 1]))
        if not np.isfinite(turnover_24h) or turnover_24h < settings.universe.min_quote_volume_24h:
            continue
        prior_median = float(np.median(quote[index - 24:index]))
        closes = frame["close"].to_numpy(dtype=float)
        activity[symbol] = {
            "volume_burst_ratio": float(quote[index] / prior_median) if prior_median > 0 else None,
            "return_4h_pct": float((closes[index] / closes[index - 4] - 1.0) * 100),
            "return_24h_pct": float((closes[index] / closes[index - 24] - 1.0) * 100),
        }
        candidates.append((symbol, turnover_24h))
        tickers[symbol] = {"quoteVolume": turnover_24h, "count": None}
    return candidates, activity, tickers


def _historical_rank(
    timestamp: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    turnovers: dict[str, np.ndarray],
    onboard_ms: dict[str, int],
    settings,
    *,
    limit: int,
) -> list[str]:
    candidates, activity, tickers = _hot_candidates(timestamp, frames, turnovers, onboard_ms, settings)
    if not candidates:
        return []
    return [row["symbol"] for row in rank_hot_candidates(candidates, tickers, activity, limit=limit)]


def run_portfolio_backtest(
    bars_1h: dict[str, pd.DataFrame],
    bars_4h: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
    onboard_ms: dict[str, int],
    settings,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int = 20,
    initial_equity: float = 10_000.0,
    exit_policy: PortfolioExitPolicy = "breakeven_trail",
    record_long_shadows: bool = False,
    enable_trend_routing: bool = False,
    market_bars_4h: pd.DataFrame | None = None,
    mr_entry_mode: str = "legacy",
    mr_deviation_pct: float = 0.02,
    mr_candidate: bool = False,
    record_trend_funnel: bool = False,
    hot_universe_exemption: bool = False,
    exit_proposal_v1: bool = False,
    bollinger_mid_exit: bool = False,
    bollinger_mid_period: int = 20,
    bollinger_mid_ema: bool = True,
    bollinger_mid_buffer_sigma: float = 0.0,
    bollinger_mid_hold_bars: int = 0,
    bollinger_mid_partial: float = 1.0,
    breakeven_r_mult: float | None = None,
    btc_bull_long_only: bool = False,
) -> PortfolioBacktestResult:
    """Replay hourly portfolio decisions; entries fill at the next bar open.

    The historical universe contains currently active contracts only. Quote turnover
    is estimated from OHLCV and historical trade counts are unavailable in CCXT OHLCV.
    Trend routing is opt-in; when enabled, trend signals use the same next-1H-open
    fill clock as mean-reversion signals, without simulating the 45-minute pullback.

    ``bollinger_mid_exit`` is an opt-in protection: a long entered near the lower 1H
    Bollinger band (short near the upper) that reaches the mid band but fails to hold
    and rolls back is closed. A 5-window sweep found the best configuration is EMA20
    mid (bollinger_mid_ema=True), immediate arming (hold_bars=0), no sigma buffer
    (buffer_sigma=0), and full close (partial=1.0); it is a modest aggregate
    improvement, not a stable edge.
    """
    if mr_candidate:
        # Unified preset for the in-sample-validated "new MR" candidate: hotness
        # top-N universe (no routing / no RANGE gate), EMA50-deviation entries,
        # and the EMA20 Bollinger mid-band protection as the only added exit.
        # No TP1/TP2; use 1.0-ATR/+0.2R then 1.5R/+0.5R protection.
        # Pass individual flags only to
        # override a specific sub-setting of the preset.
        enable_trend_routing = False
        mr_entry_mode = "ema_deviation"
        mr_deviation_pct = 0.01
        exit_proposal_v1 = False
        breakeven_r_mult = None
        bollinger_mid_exit = True
        bollinger_mid_ema = True
        bollinger_mid_period = 20
        bollinger_mid_hold_bars = 0
        bollinger_mid_buffer_sigma = 0.0
        bollinger_mid_partial = 1.0
    if exit_policy not in ("breakeven_trail", "time_managed"):
        raise ValueError("unsupported portfolio exit policy")
    if not 1 <= top_n <= 20 or initial_equity <= 0 or not bars_1h:
        raise ValueError("portfolio data, top_n between 1 and 20, and positive initial equity are required")
    if market_bars_4h is not None and not market_bars_4h.empty:
        validate_ohlcv(market_bars_4h)
        if market_bars_4h.index.tz is None:
            raise ValueError("market 4H candle timestamps must be timezone-aware")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("portfolio bounds must be timezone-aware and ordered")

    btc_ema50: pd.Series | None = None
    btc_ema50_slope: pd.Series | None = None
    if btc_bull_long_only and market_bars_4h is not None and not market_bars_4h.empty:
        btc_ema50 = market_bars_4h["close"].ewm(
            span=settings.symbol_regime.ema_slow_bars, adjust=False,
            min_periods=settings.symbol_regime.ema_slow_bars,
        ).mean()
        btc_ema50_slope = btc_ema50 / btc_ema50.shift(6) - 1.0

    symbols = sorted(set(bars_1h).intersection(bars_4h))
    if not symbols:
        raise ValueError("no symbols have both 1H and 4H history")
    timelines: dict[str, pd.DataFrame] = {}
    turn: dict[str, np.ndarray] = {}
    regimes: dict[str, pd.DataFrame] = {}
    atrs: dict[str, pd.Series] = {}
    mid_bands: dict[str, pd.Series] = {}
    sig_bands: dict[str, pd.Series] = {}
    for symbol in symbols:
        one, four = bars_1h[symbol], bars_4h[symbol]
        validate_ohlcv(one)
        validate_ohlcv(four)
        if one.index.tz is None or four.index.tz is None:
            raise ValueError(f"{symbol} candle timestamps must be timezone-aware")
        timelines[symbol] = one.loc[(one.index >= start - pd.Timedelta(30, unit="D")) & (one.index < end)].copy()
        if timelines[symbol].empty:
            continue
        turn[symbol] = _quote_turnover(timelines[symbol])
        regimes[symbol] = classify_regimes(four, settings)
        atrs[symbol] = atr(timelines[symbol], settings.lookback.atr_len)
        if bollinger_mid_exit:
            close_s = timelines[symbol]["close"]
            if bollinger_mid_ema:
                mid_bands[symbol] = close_s.ewm(span=bollinger_mid_period, adjust=False,
                                                min_periods=bollinger_mid_period).mean()
            else:
                mid_bands[symbol] = close_s.rolling(bollinger_mid_period,
                                                    min_periods=bollinger_mid_period).mean()
            if bollinger_mid_buffer_sigma > 0:
                sig_bands[symbol] = close_s.rolling(bollinger_mid_period,
                                                    min_periods=bollinger_mid_period).std()
    symbols = [symbol for symbol in symbols if symbol in timelines]
    timestamps = sorted({timestamp for symbol in symbols for timestamp in timelines[symbol].index
                         if start <= timestamp + HOUR <= end})
    if not timestamps:
        raise ValueError("no 1H bars fall within the requested portfolio window")

    fee_rate = settings.fees.taker_bps / 10_000
    slip_rate = settings.fees.slippage_bps / 10_000
    cash = float(initial_equity)
    portfolio = PortfolioState()
    active: dict[str, _Active] = {}
    pending: dict[str, dict] = {}
    marks: dict[str, float] = {}
    trades: list[Trade] = []
    equity_values: list[float] = []
    equity_times: list[pd.Timestamp] = []
    skipped: dict[str, int] = {}
    rejected_by_source: dict[str, dict[str, int]] = {MEAN_REVERSION: {}, TREND: {}}
    market_risk_execution = {
        state: {
            "opened_count": 0,
            "opened_quantity": 0.0,
            "opened_notional": 0.0,
            "configured_size_multiplier": (
                1.0 if state == "NORMAL" else
                float(settings.market_risk.elevated_size_multiplier) if state == "RISK_ELEVATED" else 0.0
            ),
        }
        for state in ("NORMAL", "RISK_ELEVATED", "PAUSE_NEW")
    }
    indices = {symbol: {ts: i for i, ts in enumerate(timelines[symbol].index)} for symbol in symbols}
    funding_rows = {symbol: funding.get(symbol, pd.DataFrame()).to_dict("records") for symbol in symbols}
    funding_indices = {symbol: funding.get(symbol, pd.DataFrame()).index.asi8 for symbol in symbols
                       if symbol in funding and not funding[symbol].empty}
    funding_cursor = {symbol: 0 for symbol in symbols}
    last_prices: dict[str, float] = {}
    last_exit_bar_index: dict[str, int] = {}
    shadow_long_signals: list[dict] = []
    long_trade_excursions: list[dict] = []
    trend_funnel_events: list[dict] = []
    trend_funnel_fill_rejections: list[dict] = []
    trend_stop_records: list[dict] = []
    hot_universe_exemption_records: list[dict] = []
    exempted_open: set[str] = set()
    market_risk_cache: dict[pd.Timestamp, object] = {}
    symbol_regime_cache: dict[tuple[str, pd.Timestamp], object] = {}
    closed_bar_input_checks = 0
    closed_bar_input_violations = 0
    actionable_signals_checked = 0
    future_signal_timestamp_violations = 0

    def check_closed_frame(frame: pd.DataFrame | None, interval: pd.Timedelta,
                           as_of: pd.Timestamp) -> None:
        nonlocal closed_bar_input_checks, closed_bar_input_violations
        if frame is None or frame.empty:
            return
        closed_bar_input_checks += 1
        if frame.index[-1] + interval > as_of:
            closed_bar_input_violations += 1

    def equity(mark_overrides: dict[str, float] | None = None) -> float:
        value = cash
        for symbol, item in active.items():
            price = (mark_overrides or {}).get(symbol, marks.get(symbol, item.position.entry_price))
            side_sign = 1 if item.position.side is Side.LONG else -1
            value += item.initial_margin + item.position.quantity * (price - item.position.entry_price) * side_sign
        return value

    def close_position(symbol: str, raw_price: float, at: pd.Timestamp, reason: str) -> None:
        nonlocal cash
        item = active.pop(symbol)
        position = item.position
        fill, exit_slip = _apply_slippage(raw_price, position.side, is_entry=False, slippage_rate=slip_rate)
        side_sign = 1 if position.side is Side.LONG else -1
        gross = position.quantity * (fill - position.entry_price) * side_sign
        exit_fee = position.quantity * fill * fee_rate
        # Funding is deducted from free cash at each settlement while open.
        cash += item.initial_margin + gross - exit_fee
        total_fees = item.entry_fee + exit_fee
        trade = Trade(
            symbol=symbol, side=position.side, opened_at=pd.Timestamp(position.opened_at), closed_at=at,
            entry_price=position.entry_price, exit_price=fill, quantity=position.quantity,
            gross_pnl=gross, net_pnl=gross - total_fees - item.funding_paid, fees=total_fees,
            slippage_paid=item.entry_slippage + position.quantity * exit_slip,
            funding_paid=item.funding_paid, entry_reason=position.reason, exit_reason=reason,
            strategy_source=position.source,
        )
        trades.append(trade)
        if symbol in exempted_open:
            exempted_open.discard(symbol)
            hot_universe_exemption_records.append({
                "symbol": symbol, "side": position.side.value,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": at.isoformat(), "exit_reason": reason,
                "entry_price": float(position.entry_price),
                "net_pnl": float(trade.net_pnl),
            })
        if record_trend_funnel and position.source == TREND:
            trend_stop_records.append({
                "symbol": symbol, "side": position.side.value,
                "opened_at": position.opened_at.isoformat(),
                "closed_at": at.isoformat(), "exit_reason": reason,
                "entry_price": float(position.entry_price),
                "initial_stop_price": float(position.stop_price),
                "initial_risk": float(item.initial_risk),
                "stop_distance_pct": float(item.initial_risk / position.entry_price * 100),
                "net_pnl": float(trade.net_pnl),
            })
        if record_long_shadows and position.side is Side.LONG:
            long_trade_excursions.append({
                "symbol": symbol, "opened_at": trade.opened_at, "closed_at": trade.closed_at,
                "entry_price": position.entry_price, "quantity": position.quantity,
                "initial_risk_usdt": item.initial_risk * position.quantity,
                "mfe_r": max(0.0, item.best_price - position.entry_price) / item.initial_risk,
                "mae_r": max(0.0, position.entry_price - item.worst_price) / item.initial_risk,
                "net_pnl": trade.net_pnl, "r_multiple": trade.net_pnl / (item.initial_risk * position.quantity),
                "exit_reason": reason,
                "held_hours": (trade.closed_at - trade.opened_at).total_seconds() / 3600,
            })
        portfolio.register_close(symbol, at.to_pydatetime(), reason=reason, net_pnl=trade.net_pnl)
        symbol_indices = indices[symbol]
        close_index = symbol_indices.get(at)
        if close_index is None:
            previous_bar_index = symbol_indices.get(at - HOUR)
            if previous_bar_index is not None:
                close_index = previous_bar_index + 1
        if close_index is not None:
            last_exit_bar_index[symbol] = close_index

    def partial_close(symbol: str, raw_price: float, at: pd.Timestamp, reason: str,
                      frac: float) -> None:
        """Close ``frac`` of the initial position quantity at a limit fill price."""
        nonlocal cash
        item = active[symbol]
        position = item.position
        closed_qty = item.initial_quantity * frac
        if closed_qty <= 0:
            return
        if closed_qty >= position.quantity - 1e-12:
            close_position(symbol, raw_price, at, reason + "_close_remainder")
            return
        fill, exit_slip = _apply_slippage(raw_price, position.side, is_entry=False,
                                          slippage_rate=slip_rate)
        side_sign = 1 if position.side is Side.LONG else -1
        gross = closed_qty * (fill - position.entry_price) * side_sign
        exit_fee = closed_qty * fill * fee_rate
        frac_of_current = closed_qty / position.quantity
        closed_entry_fee = item.entry_fee * frac_of_current
        closed_funding = item.funding_paid * frac_of_current
        closed_margin = item.initial_margin * frac_of_current
        closed_entry_slip = item.entry_slippage * frac_of_current
        cash += closed_margin + gross - exit_fee
        total_fees = closed_entry_fee + exit_fee
        trade = Trade(
            symbol=symbol, side=position.side, opened_at=pd.Timestamp(position.opened_at),
            closed_at=at, entry_price=position.entry_price, exit_price=fill,
            quantity=closed_qty, gross_pnl=gross,
            net_pnl=gross - total_fees - closed_funding, fees=total_fees,
            slippage_paid=closed_entry_slip + closed_qty * exit_slip,
            funding_paid=closed_funding, entry_reason=position.reason,
            exit_reason=reason, strategy_source=position.source,
        )
        trades.append(trade)
        item.entry_fee -= closed_entry_fee
        item.entry_slippage -= closed_entry_slip
        item.funding_paid -= closed_funding
        item.initial_margin -= closed_margin
        item.position = Position(
            position.symbol, position.side, position.quantity - closed_qty,
            position.entry_price, position.stop_price, position.opened_at,
            position.reason, position.regime, position.source,
        )

    def record_rejection(source: str | None, reason: str, *, symbol: str | None = None,
                         at: pd.Timestamp | None = None, side: Side | None = None,
                         signal_reason: str | None = None) -> None:
        normalized_source = source or MEAN_REVERSION
        source_counts = rejected_by_source.setdefault(normalized_source, {})
        source_counts[reason] = source_counts.get(reason, 0) + 1
        skipped[reason] = skipped.get(reason, 0) + 1
        if symbol is not None and at is not None:
            logger.opt(lazy=True).warning("Portfolio backtest routed entry rejected {}", lambda: json.dumps({
                "time": at.isoformat(),
                "bar_index": indices[symbol].get(at - HOUR),
                "symbol": symbol,
                "side": side.value if side is not None else None,
                "source": normalized_source,
                "signal_reason": signal_reason,
                "reason": reason,
            }, ensure_ascii=False, sort_keys=True))

    def reject_open(symbol: str, intent: dict, at: pd.Timestamp, reason: str,
                    *, position_source: str | None = None, details: dict | None = None) -> None:
        source = intent.get("source", MEAN_REVERSION) or MEAN_REVERSION
        record_rejection(source, reason)
        if record_trend_funnel and source == TREND:
            trend_funnel_fill_rejections.append({
                "ts": at.isoformat(), "symbol": symbol,
                "side": str(intent.get("side")), "reason": reason,
                "details": details or {},
            })
        logger.opt(lazy=True).warning("Portfolio backtest entry rejected {}", lambda: json.dumps({
            "time": at.isoformat(),
            "bar_index": indices[symbol].get(at),
            "symbol": symbol,
            "side": intent.get("side"),
            "source": source,
            "signal_reason": intent.get("reason"),
            "position_source": position_source,
            "reason": reason,
            "details": details or {},
        }, ensure_ascii=False, sort_keys=True))

    for open_time in timestamps:
        close_time = open_time + HOUR
        btc_bull = True
        if btc_bull_long_only and btc_ema50 is not None:
            btc_closes = market_bars_4h.index.asi8 + FOUR_HOURS_NS * 1_000_000
            btc_idx = int(np.searchsorted(btc_closes, close_time.value, side="right") - 1)
            if btc_idx >= 0:
                slope = btc_ema50_slope.iloc[btc_idx]
                btc_bull = bool(np.isfinite(slope) and slope > 0)
        current_rows = {symbol: timelines[symbol].iloc[indices[symbol][open_time]]
                        for symbol in symbols if open_time in indices[symbol]}
        for symbol, candle in current_rows.items():
            last_prices[symbol] = float(candle["close"])
        marks.update({symbol: last_prices[symbol] for symbol in active if symbol in last_prices})

        # Settle historical funding for positions held before the settlement time.
        for symbol in list(active):
            settlements = funding.get(symbol)
            if settlements is None or settlements.empty:
                continue
            cursor = funding_cursor[symbol]
            times = funding_indices[symbol]
            while cursor < len(times) and times[cursor] <= open_time.value + 60_000_000_000:
                if times[cursor] > open_time.value - HOUR.value:
                    settlement = settlements.iloc[cursor]
                    mark = settlement.get("mark_price")
                    if pd.isna(mark) or float(mark) <= 0:
                        mark = last_prices.get(symbol, active[symbol].position.entry_price)
                    sign = 1 if active[symbol].position.side is Side.LONG else -1
                    amount = active[symbol].position.quantity * float(mark) * float(settlement["funding_rate"]) * sign
                    active[symbol].funding_paid += amount
                    cash -= amount
                cursor += 1
            funding_cursor[symbol] = cursor

        # Fill orders scheduled at the previous completed 1H candle.
        for symbol, intent in list(pending.items()):
            candle = current_rows.get(symbol)
            pending.pop(symbol, None)
            if candle is None:
                continue
            raw_open = float(candle["open"])
            if intent["action"] == Action.CLOSE.value:
                if symbol in active:
                    close_position(symbol, raw_open, open_time, intent["reason"])
                continue
            if intent["action"] != Action.OPEN.value or not intent.get("eligible", False):
                continue
            source = intent.get("source", MEAN_REVERSION) or MEAN_REVERSION
            held = portfolio.positions.get(symbol)
            if held is not None and source != held.source:
                reject_open(symbol, intent, open_time, "POSITION_OWNERSHIP_CONFLICT",
                            position_source=held.source,
                            details={"signal_source": source, "position_source": held.source})
                continue
            if symbol in active and intent.get("close_first"):
                close_position(symbol, raw_open, open_time, "reverse_close_first")
            if symbol in active:
                reject_open(symbol, intent, open_time, "position_still_open",
                            position_source=active[symbol].position.source)
                continue
            side = Side(intent["side"])
            fill, entry_slip = _apply_slippage(raw_open, side, is_entry=True, slippage_rate=slip_rate)
            open_marks = ({key: float(row["open"]) for key, row in current_rows.items() if key in active}
                          if enable_trend_routing else None)
            equity_at_open = equity(open_marks)
            regime = Regime(intent["regime"])
            is_trend_execution = enable_trend_routing and source == TREND
            trend_stop_state = None
            trend_sizing = None
            if is_trend_execution:
                try:
                    stop, structure = calculate_trend_stop(
                        side, fill, float(intent["atr"]), intent["trend_stop_bars"], settings,
                    )
                    sizing_equity = (equity_at_open * float(intent.get("market_size_multiplier", 1.0))
                                     * float(intent.get("size_multiplier", 1.0)))
                    quantity, risk_budget, risk_per_unit = calculate_trend_quantity(
                        sizing_equity, fill, stop, settings, step_size=1e-8,
                        minimum_quantity=0.0, existing_gross_notional=portfolio.gross_notional,
                    )
                    trend_stop_state = initialize_trend_exit_state(
                        fill, stop, structure, stop_mode=settings.trend_strategy.stop_mode,
                    )
                    trend_sizing = {"risk_budget": risk_budget, "risk_per_unit": risk_per_unit}
                except (KeyError, TypeError, ValueError) as error:
                    reject_open(symbol, intent, open_time, str(error) or "trend_entry_invalid",
                                details={"error_type": type(error).__name__, "fill_price": fill})
                    continue
                size_multiplier = float(intent.get("size_multiplier", 1.0))
                structure_low = structure if side is Side.LONG else None
                structure_high = structure if side is Side.SHORT else None
                range_low = range_high = None
            else:
                size_multiplier = float(intent.get("size_multiplier", 1.0))
                structure = intent.get("structure_low" if side is Side.LONG else "structure_high")
                entry_rejection = entry_rejection_reason(
                    side, fill, float(intent["atr"]), intent.get("range_low"), intent.get("range_high"),
                    intent.get("structure_low"), intent.get("structure_high"), settings,
                    require_structure=("structure_low" in intent if side is Side.LONG else "structure_high" in intent),
                )
                if entry_rejection is not None:
                    reject_open(symbol, intent, open_time, entry_rejection,
                                details={"fill_price": fill})
                    continue
                try:
                    stop, stop_multiplier = calculate_adaptive_stop(
                        side, fill, float(intent["atr"]), structure, settings,
                    )
                except ValueError as error:
                    if str(error) != "stop_too_wide":
                        raise
                    reject_open(symbol, intent, open_time, "stop_too_wide",
                                details={"error": str(error), "fill_price": fill})
                    continue
                size_multiplier *= stop_multiplier
                market_size_multiplier = float(intent.get("market_size_multiplier", 1.0))
                quantity = calculate_order_quantity(
                    equity_at_open, fill, portfolio, settings,
                    step_size=1e-8, minimum_quantity=0.0,
                    size_multiplier=size_multiplier * market_size_multiplier,
                    gross_notional_cap_pct=(settings.sizing.max_gross_notional_pct
                                            * market_size_multiplier),
                )
                structure_low, structure_high = intent.get("structure_low"), intent.get("structure_high")
                range_low, range_high = intent.get("range_low"), intent.get("range_high")
            if quantity <= 0:
                key = ("trend_quantity_zero" if is_trend_execution else
                       "stop_too_wide" if stop_multiplier < 1 else "no_remaining_portfolio_capacity")
                reject_open(symbol, intent, open_time, key,
                            details={"quantity": quantity, "size_multiplier": size_multiplier,
                                     **(trend_sizing or {})})
                continue
            order = OrderIntent(symbol, side, Action.OPEN, quantity, fill, intent["reason"],
                                regime, stop, source)
            signal = with_signal_context(
                Signal(side, Action.OPEN, intent["reason"], float(intent["strength"])),
                symbol=symbol, source=source, timestamp=open_time.to_pydatetime(),
            )
            approval = guard_signal(signal, order, portfolio, equity_at_open, intent.get("funding_rate"),
                                    open_time.to_pydatetime(), settings,
                                    bypass_cooldown=bool(intent.get("close_first")),
                                    bars_since_exit=(indices[symbol][open_time] - last_exit_bar_index[symbol]
                                                     if symbol in last_exit_bar_index else None),
                                    market_risk_state=intent.get("market_risk_state", "NORMAL"))
            if not approval.allowed:
                held = portfolio.positions.get(symbol)
                reject_open(symbol, intent, open_time, approval.reason,
                            position_source=held.source if held is not None else None,
                            details=approval.details)
                continue
            entry_fee = quantity * fill * fee_rate
            leverage = float(settings.sizing.leverage)
            margin = quantity * fill / leverage
            if cash < margin + entry_fee:
                reject_open(symbol, intent, open_time, "insufficient_shared_margin",
                            details={"available_cash": cash, "required_margin_and_fee": margin + entry_fee})
                continue
            position = Position(symbol, side, quantity, fill, stop, open_time.to_pydatetime(),
                                intent["reason"], regime.value, source=signal.source)
            portfolio.register_open(position, approval)
            cash -= margin + entry_fee
            market_state = intent.get("market_risk_state", "NORMAL")
            applied_market_multiplier = float(intent.get("market_size_multiplier", 1.0))
            risk_state_summary = market_risk_execution.setdefault(market_state, {
                "opened_count": 0,
                "opened_quantity": 0.0,
                "opened_notional": 0.0,
                "configured_size_multiplier": applied_market_multiplier,
            })
            risk_state_summary["opened_count"] += 1
            risk_state_summary["opened_quantity"] += quantity
            risk_state_summary["opened_notional"] += quantity * fill
            logger.opt(lazy=True).info("Portfolio backtest entry approved {}", lambda: json.dumps({
                "time": open_time.isoformat(),
                "symbol": symbol,
                "side": side.value,
                "source": source,
                "market_risk_state": market_state,
                "market_size_multiplier": applied_market_multiplier,
                "quantity": quantity,
                "notional": quantity * fill,
                "risk": approval.details or {},
            }, ensure_ascii=False, sort_keys=True))
            active[symbol] = _Active(position, entry_fee, quantity * entry_slip, 0.0, margin,
                                     abs(fill - stop), stop, fill, float(intent["atr"]),
                                      structure_low, structure_high, range_low, range_high,
                                     float(intent.get("signal_close", fill)))
            active[symbol].worst_price = fill
            active[symbol].trend_exit_state = trend_stop_state
            active[symbol].initial_quantity = quantity
            marks[symbol] = float(candle["open"])
            if intent.get("hot_universe_exempted"):
                exempted_open.add(symbol)

        # Existing stops are evaluated before any newly calculated protection or time exit.
        time_exit_reasons: dict[str, str] = {}
        invalidation_reasons: dict[str, str] = {}
        for symbol, item in list(active.items()):
            candle = current_rows.get(symbol)
            if candle is None:
                continue
            side = item.position.side
            stop_hit = ((side is Side.LONG and float(candle["low"]) <= item.protective_stop)
                        or (side is Side.SHORT and float(candle["high"]) >= item.protective_stop))
            if stop_hit:
                open_price = float(candle["open"])
                gap = ((side is Side.LONG and open_price <= item.protective_stop)
                       or (side is Side.SHORT and open_price >= item.protective_stop))
                if enable_trend_routing and item.position.source == TREND:
                    stop_reason = ((item.trend_exit_state or {}).get("active_stop_reason")
                                   or "structure_stop")
                    reason = f"trend_{stop_reason}_{'gap' if gap else 'hit'}"
                else:
                    reason = "stop_gap" if gap else "stop_hit"
                close_position(symbol, open_price if gap else item.protective_stop, close_time,
                               reason)
                continue
            item.best_price = (max(item.best_price, float(candle["high"])) if side is Side.LONG
                               else min(item.best_price, float(candle["low"])))
            if side is Side.LONG:
                item.worst_price = min(item.worst_price, float(candle["low"]))
            current_close = float(candle["close"])
            if enable_trend_routing and item.position.source == TREND:
                index = indices[symbol][open_time]
                decision = evaluate_trend_exit(
                    item.position, timelines[symbol].iloc[:index + 1], settings,
                    now=close_time.to_pydatetime(), exit_state=item.trend_exit_state,
                )
                item.trend_exit_state = decision.exit_state
                item.protective_stop = decision.active_stop_price
                if decision.signal.action is Action.CLOSE:
                    invalidation_reasons[symbol] = decision.signal.reason
                continue
            if bollinger_mid_exit:
                mid = float(mid_bands[symbol].iloc[indices[symbol][open_time]])
                if math.isfinite(mid):
                    sigma = 0.0
                    if bollinger_mid_buffer_sigma > 0 and symbol in sig_bands:
                        sigma = float(sig_bands[symbol].iloc[indices[symbol][open_time]])
                        sigma = sigma if math.isfinite(sigma) else 0.0
                    buffer = bollinger_mid_buffer_sigma * sigma
                    if side is Side.LONG:
                        if not item.reached_mid:
                            if item.position.entry_price < mid and current_close >= mid:
                                item.mid_hold_count += 1
                                if item.mid_hold_count >= bollinger_mid_hold_bars + 1:
                                    item.reached_mid = True
                            else:
                                item.mid_hold_count = 0
                        elif not item.mid_fired and current_close < mid - buffer:
                            if bollinger_mid_partial >= 1.0:
                                close_position(symbol, float(candle["close"]), close_time,
                                               "bollinger_mid_rejection")
                                continue
                            else:
                                partial_close(symbol, float(candle["close"]), close_time,
                                              "bollinger_mid_rejection", bollinger_mid_partial)
                                item.mid_fired = True
                    else:
                        if not item.reached_mid:
                            if item.position.entry_price > mid and current_close <= mid:
                                item.mid_hold_count += 1
                                if item.mid_hold_count >= bollinger_mid_hold_bars + 1:
                                    item.reached_mid = True
                            else:
                                item.mid_hold_count = 0
                        elif not item.mid_fired and current_close > mid + buffer:
                            if bollinger_mid_partial >= 1.0:
                                close_position(symbol, float(candle["close"]), close_time,
                                               "bollinger_mid_rejection")
                                continue
                            else:
                                partial_close(symbol, float(candle["close"]), close_time,
                                              "bollinger_mid_rejection", bollinger_mid_partial)
                                item.mid_fired = True
            exit_cfg = settings.exits
            invalidation = None
            if side is Side.LONG:
                if (exit_cfg.structure_invalid_enabled and item.structure_low is not None
                        and current_close < item.structure_low - exit_cfg.structure_invalid_buffer_atr * item.entry_atr):
                    invalidation = "long_structure_invalid"
            else:
                if (exit_cfg.structure_invalid_enabled and item.structure_high is not None
                        and current_close > item.structure_high + exit_cfg.structure_invalid_buffer_atr * item.entry_atr):
                    invalidation = "short_structure_invalid"
            regime_frame = regimes[symbol]
            regime_closes = regime_frame.index.asi8 + FOUR_HOURS_NS
            current_ridx = int(np.searchsorted(regime_closes, close_time.value, side="right") - 1)
            if invalidation is None and exit_cfg.regime_fade_enabled and current_ridx >= 0:
                current_regime = regime_frame.iloc[current_ridx]["regime"]
                progress = ((current_close - item.position.entry_price) if side is Side.LONG
                            else (item.position.entry_price - current_close))
                initial_regime = Regime.LOW.value if side is Side.LONG else Regime.HIGH.value
                if (item.position.regime == initial_regime and current_regime is Regime.SIDEWAYS
                        and progress < exit_cfg.regime_fade_min_atr * item.entry_atr):
                    invalidation = "long_regime_faded" if side is Side.LONG else "short_regime_faded"
            moved_favorably = current_close > item.last_close if side is Side.LONG else current_close < item.last_close
            item.no_progress_count = 0 if moved_favorably else item.no_progress_count + 1
            item.last_close = current_close
            favorable = ((item.best_price - item.position.entry_price) if side is Side.LONG
                         else (item.position.entry_price - item.best_price))
            if (invalidation is None and item.no_progress_count >= exit_cfg.no_progress_bars
                    and favorable < exit_cfg.no_progress_min_atr * item.entry_atr):
                invalidation = "no_progress"
            if invalidation is not None:
                invalidation_reasons[symbol] = invalidation
            if exit_proposal_v1:
                policy = settings.exit_proposal_v1
                atr_value = item.entry_atr
                side_sign = 1 if side is Side.LONG else -1
                entry = item.position.entry_price
                extreme = float(candle["high"]) if side is Side.LONG else float(candle["low"])
                if not item.tp1_taken:
                    tp1_distance = policy.tp1_atr * atr_value
                    if side_sign * (extreme - entry) >= tp1_distance:
                        partial_close(symbol, entry + side_sign * tp1_distance, close_time,
                                      "take_profit_1", policy.tp1_frac)
                        item.tp1_taken = True
                elif not item.tp2_taken:
                    tp2_distance = policy.tp2_atr * atr_value
                    if side_sign * (extreme - entry) >= tp2_distance:
                        partial_close(symbol, entry + side_sign * tp2_distance, close_time,
                                      "take_profit_2", policy.tp2_frac)
                        item.tp2_taken = True
                if symbol not in active:
                    continue
            progress_r = favorable / item.initial_risk
            if exit_policy == "time_managed":
                held_hours = (close_time - pd.Timestamp(item.position.opened_at)).total_seconds() / 3600
                close_fill, _ = _apply_slippage(float(candle["close"]), side, is_entry=False, slippage_rate=slip_rate)
                sign = 1 if side is Side.LONG else -1
                net_if_closed = (item.position.quantity * (close_fill - item.position.entry_price) * sign
                                 - item.entry_fee - item.position.quantity * close_fill * fee_rate - item.funding_paid)
                if (held_hours >= settings.exit_management.time_check_hours
                        and progress_r < settings.exit_management.time_min_progress_r and net_if_closed <= 0):
                    time_exit_reasons[symbol] = "time_stop_12h_no_progress"
                elif (held_hours >= settings.exit_management.time_exit_hours
                      and progress_r < settings.exit_management.breakeven_trigger_r):
                    time_exit_reasons[symbol] = "time_exit_24h_unprotected"
            if mr_candidate and not exit_proposal_v1:
                item.protective_stop = staged_profit_lock_stop(
                    side, item.position.entry_price, item.entry_atr,
                    item.initial_risk, favorable, item.protective_stop,
                    first_trigger_atr=settings.stops.move_to_breakeven_atr,
                    first_lock_r=settings.mr_candidate_exit.first_lock_r,
                    second_trigger_r=settings.mr_candidate_exit.second_lock_trigger_r,
                    second_lock_r=settings.mr_candidate_exit.second_lock_r,
                    fee_rate=fee_rate, slip_rate=slip_rate,
                )
            else:
                if breakeven_r_mult is not None:
                    breakeven_hit = favorable >= breakeven_r_mult * item.initial_risk
                else:
                    breakeven_hit = favorable >= settings.stops.move_to_breakeven_atr * item.entry_atr
                if breakeven_hit:
                    entry = item.position.entry_price
                    if side is Side.LONG:
                        adjusted = entry * (1 + fee_rate) / ((1 - slip_rate) * (1 - fee_rate))
                        item.protective_stop = max(item.protective_stop, adjusted)
                    else:
                        adjusted = entry * (1 - fee_rate) / ((1 + slip_rate) * (1 + fee_rate))
                        item.protective_stop = min(item.protective_stop, adjusted)
            if exit_proposal_v1:
                if item.tp1_taken:
                    current_regime = (regime_frame.iloc[current_ridx]["regime"]
                                      if current_ridx >= 0 else None)
                    tight = ((side is Side.LONG and current_regime is Regime.HIGH)
                             or (side is Side.SHORT and current_regime is Regime.LOW))
                    distance = ((settings.exit_proposal_v1.trail_tight_atr if tight
                                 else settings.exit_proposal_v1.trail_atr) * item.entry_atr)
                    trailing = (item.best_price - distance if side is Side.LONG
                                else item.best_price + distance)
                    item.protective_stop = (max(item.protective_stop, trailing) if side is Side.LONG
                                             else min(item.protective_stop, trailing))
            elif progress_r >= (settings.mr_candidate_exit.trailing_trigger_r if mr_candidate
                                else settings.exit_management.trailing_trigger_r):
                distance = (settings.exit_management.trailing_atr_mult * item.initial_risk
                            / settings.signals.stop_atr_mult)
                trailing = item.best_price - distance if side is Side.LONG else item.best_price + distance
                item.protective_stop = (max(item.protective_stop, trailing) if side is Side.LONG
                                         else min(item.protective_stop, trailing))

        # Recreate the eligible universe at this historical hour close.
        ranked = _historical_rank(close_time, timelines, turn, onboard_ms, settings, limit=top_n)
        eligible = set(ranked)
        if hot_universe_exemption and enable_trend_routing:
            # Simulate the live "hot-universe exemption": out-of-rank symbols that
            # independently classify as a confirmed TREND may still open TREND
            # positions. Run every symbol through the decision pass; non-confirmed
            # out-of-rank symbols are skipped after regime classification below.
            decision_symbols = list(dict.fromkeys([*ranked, *symbols]))
        else:
            decision_symbols = list(dict.fromkeys([*ranked, *[symbol for symbol in active if symbol not in eligible]]))
        for symbol in decision_symbols:
            candle = current_rows.get(symbol)
            if candle is None:
                continue
            if symbol not in eligible and symbol not in active and not (hot_universe_exemption and enable_trend_routing):
                continue
            if symbol in invalidation_reasons:
                pending[symbol] = {"action": Action.CLOSE.value,
                                   "reason": invalidation_reasons[symbol], "eligible": True}
                continue
            if symbol in time_exit_reasons:
                side = active[symbol].position.side
                pending[symbol] = {"action": Action.CLOSE.value,
                                   "reason": time_exit_reasons[symbol], "eligible": True}
                continue
            regimes_frame = regimes[symbol]
            regime_closes = regimes_frame.index.asi8 + FOUR_HOURS_NS * 1_000_000
            ridx = int(np.searchsorted(regime_closes, close_time.value, side="right") - 1)
            if ridx < 0:
                continue
            regime = _snapshot(regimes_frame.iloc[ridx], regimes_frame.index[ridx])
            if regime is None:
                continue
            idx = indices[symbol][open_time]
            atr_value = atrs[symbol].iloc[idx]
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            frame = timelines[symbol]
            history = frame.iloc[max(0, idx - 47):idx + 1]
            one_hour_history = frame.iloc[:idx + 1]
            four_hour_history = bars_4h[symbol].iloc[:ridx + 1]
            check_closed_frame(one_hour_history, HOUR, close_time)
            check_closed_frame(four_hour_history, FOUR_HOURS, close_time)
            # The time filter only feeds mean-reversion entry signals, so it is
            # computed lazily in the MR branches below; the TREND and UNCERTAIN
            # paths (and the hot-universe-exemption skip) never read it.
            position_side = active[symbol].position.side if symbol in active else None
            regime_row = regimes_frame.iloc[ridx]
            signal_swing = frame.iloc[max(0, idx - settings.stops.swing_lookback_1h + 1):idx + 1]
            signal_range_low, signal_range_high = regime_row.get("range_low"), regime_row.get("range_high")
            trend_intent_context: dict = {}
            market_state = "NORMAL"
            exempted = False
            if enable_trend_routing:
                market_key = close_time.floor("4h")
                if market_key not in market_risk_cache:
                    market_history = (market_bars_4h.loc[market_bars_4h.index + FOUR_HOURS <= close_time]
                                      if market_bars_4h is not None else None)
                    check_closed_frame(market_history, FOUR_HOURS, close_time)
                    market_risk_cache[market_key] = classify_market_risk(
                        market_history, settings, now=close_time.to_pydatetime(),
                    )
                market_state = market_risk_cache[market_key].state.value
                regime_key = (symbol, market_key)
                if regime_key not in symbol_regime_cache:
                    four_hour_closed = bars_4h[symbol].loc[
                        bars_4h[symbol].index + FOUR_HOURS <= close_time
                    ]
                    check_closed_frame(four_hour_closed, FOUR_HOURS, close_time)
                    symbol_regime_cache[regime_key] = classify_symbol_regime(
                        symbol, four_hour_closed, settings, now=close_time.to_pydatetime(),
                    )
                assessment = symbol_regime_cache[regime_key]
                exempted = (hot_universe_exemption and symbol not in eligible and symbol not in active)
                if (exempted and not (assessment.state is SymbolRegimeState.TREND
                                      and bool(assessment.evidence.get("trend_confirmation_pass", False)))):
                    continue
                route = resolve_strategy_route(
                    symbol, assessment.state.value, market_state,
                    position_source=(active[symbol].position.source if symbol in active else None),
                    at=close_time.to_pydatetime(), trend_strategy_ready=True,
                )
                market_size_multiplier = (
                    float(settings.market_risk.elevated_size_multiplier)
                    if market_state == "RISK_ELEVATED" else 1.0
                )
                trend_intent_context = {
                    "market_risk_state": market_state,
                    "market_size_multiplier": market_size_multiplier,
                }
                if symbol in active and active[symbol].position.source == TREND:
                    signal = Signal(None, Action.NONE, "trend_position_managed", 0.0,
                                    symbol=symbol, source=TREND, timestamp=close_time.to_pydatetime())
                elif symbol not in active and assessment.state is SymbolRegimeState.TREND:
                    one_hour_closed = frame.loc[frame.index + HOUR <= close_time]
                    four_hour_closed = bars_4h[symbol].loc[
                        bars_4h[symbol].index + FOUR_HOURS <= close_time
                    ]
                    check_closed_frame(one_hour_closed, HOUR, close_time)
                    check_closed_frame(four_hour_closed, FOUR_HOURS, close_time)
                    # Generate a would-be candidate under NORMAL while the actual
                    # route remains authoritative below. This lets the audit count
                    # only technically valid TREND opens blocked by PAUSE_NEW.
                    candidate_market_state = "NORMAL" if market_state == "PAUSE_NEW" else market_state
                    decision = evaluate_trend_entry(
                        symbol, one_hour_closed, four_hour_closed, settings,
                        now=close_time.to_pydatetime(), market_risk_state=candidate_market_state,
                        equity=equity() * market_size_multiplier, step_size=1e-8,
                        existing_gross_notional=portfolio.gross_notional,
                        symbol_regime_assessment=assessment,
                    )
                    signal = apply_entry_route(decision.signal, route)
                    if decision.signal.action is Action.OPEN and signal.action is not Action.OPEN:
                        record_rejection(
                            decision.signal.source, signal.reason, symbol=symbol, at=close_time,
                            side=decision.signal.side, signal_reason=decision.signal.reason,
                        )
                    if record_trend_funnel:
                        direction = "up" if assessment.reason == "trend_confirmed_up" else "down"
                        trend_funnel_events.append({
                            "ts": close_time.isoformat(),
                            "symbol": symbol,
                            "direction": direction,
                            "route_block": (decision.route or {}).get("entry_block_reason"),
                            "entry_action": decision.signal.action.value,
                            "entry_side": (decision.signal.side.value if decision.signal.side else None),
                            "entry_reason": decision.signal.reason,
                            "final_action": signal.action.value,
                            "final_side": (signal.side.value if signal.side else None),
                            "final_reason": signal.reason,
                            "queued": signal.action is Action.OPEN,
                        })
                    if signal.action is Action.OPEN:
                        required_stop_bars = max(settings.trend_strategy.stop_swing_lookback_1h,
                                                 settings.lookback.atr_len + 1)
                        trend_intent_context = {
                            "market_risk_state": market_state,
                            "market_size_multiplier": market_size_multiplier,
                            "trend_stop_bars": one_hour_closed.tail(required_stop_bars).copy(),
                        }
                elif symbol not in active and assessment.state is SymbolRegimeState.UNCERTAIN:
                    signal = Signal(None, Action.NONE, assessment.reason, 0.0,
                                    symbol=symbol, source="ROUTER", timestamp=close_time.to_pydatetime())
                else:
                    flags = evaluate_time_filter(
                        close_time.to_pydatetime(), history, settings,
                        precomputed_atr=atrs[symbol].reindex(history.index),
                    )
                    if mr_entry_mode == "ema_deviation" and symbol not in active:
                        raw_signal = _range_deviation_signal(
                            candle, four_hour_history, settings, flags,
                            deviation_pct=mr_deviation_pct,
                        )
                    else:
                        raw_signal = generate_signal(
                            regime, candle, flags, settings, position=position_side,
                            bars_1h=one_hour_history, bars_4h=four_hour_history,
                            entry_reference_price=float(candle["close"]), entry_atr=float(atr_value),
                            range_low=float(signal_range_low) if signal_range_low is not None and np.isfinite(signal_range_low) else None,
                            range_high=float(signal_range_high) if signal_range_high is not None and np.isfinite(signal_range_high) else None,
                            structure_low=float(signal_swing["low"].min()) if not signal_swing.empty else None,
                            structure_high=float(signal_swing["high"].max()) if not signal_swing.empty else None,
                        )
                    routed = with_signal_context(
                        raw_signal, symbol=symbol, source=MEAN_REVERSION,
                        timestamp=close_time.to_pydatetime(),
                    )
                    signal = apply_entry_route(routed, route)
                    if routed.action is Action.OPEN and signal.action is not Action.OPEN:
                        record_rejection(
                            routed.source, signal.reason, symbol=symbol, at=close_time,
                            side=routed.side, signal_reason=routed.reason,
                        )
            else:
                flags = evaluate_time_filter(
                    close_time.to_pydatetime(), history, settings,
                    precomputed_atr=atrs[symbol].reindex(history.index),
                )
                signal = generate_signal(
                    regime, candle, flags, settings, position=position_side,
                    bars_1h=one_hour_history, bars_4h=four_hour_history,
                    entry_reference_price=float(candle["close"]), entry_atr=float(atr_value),
                    range_low=float(signal_range_low) if signal_range_low is not None and np.isfinite(signal_range_low) else None,
                    range_high=float(signal_range_high) if signal_range_high is not None and np.isfinite(signal_range_high) else None,
                    structure_low=float(signal_swing["low"].min()) if not signal_swing.empty else None,
                    structure_high=float(signal_swing["high"].max()) if not signal_swing.empty else None,
                )
            # Sidecar only: record a valid, eligible A long signal. Shadow state
            # never enters the live replay's pending orders, portfolio, or cash.
            if (record_long_shadows and signal.action is Action.OPEN and signal.side is Side.LONG
                    and signal.source in {None, MEAN_REVERSION} and symbol in eligible):
                fframe = funding.get(symbol)
                rate = None
                if fframe is not None and not fframe.empty:
                    fidx = int(fframe.index.searchsorted(close_time, side="left") - 1)
                    if fidx >= 0:
                        rate = float(fframe["funding_rate"].iloc[fidx])
                shadow_long_signals.append({
                    "symbol": symbol, "signal_time": close_time,
                    "signal_open": float(candle["open"]), "signal_low": float(candle["low"]),
                    "signal_close": float(candle["close"]), "reason": signal.reason,
                    "strength": float(signal.strength), "size_multiplier": float(signal.size_multiplier),
                    "atr": float(atr_value), "regime": regime.regime.value,
                    "funding_rate": rate,
                    "structure_low": float(signal_swing["low"].min()) if not signal_swing.empty else None,
                    "structure_high": float(signal_swing["high"].max()) if not signal_swing.empty else None,
                    "range_low": float(signal_range_low) if signal_range_low is not None and np.isfinite(signal_range_low) else None,
                    "range_high": float(signal_range_high) if signal_range_high is not None and np.isfinite(signal_range_high) else None,
                })
            if (symbol in active and not (enable_trend_routing
                                          and active[symbol].position.source == TREND)
                    and close_time - pd.Timestamp(active[symbol].position.opened_at)
                    >= pd.Timedelta(settings.exits.max_hold_bars, unit="h")
                    and signal.action is not Action.CLOSE):
                signal = Signal(active[symbol].position.side, Action.CLOSE, "max_hold_24h", 1.0)
            if signal.action is Action.OPEN:
                source = signal.source if signal.source is not None else MEAN_REVERSION
            elif symbol in active:
                source = active[symbol].position.source
            else:
                source = signal.source if signal.source in {MEAN_REVERSION, TREND} else MEAN_REVERSION
            signal = with_signal_context(signal, symbol=symbol, source=source,
                                         timestamp=close_time.to_pydatetime())
            if (btc_bull_long_only and not btc_bull and signal.action is Action.OPEN
                    and signal.side is Side.LONG):
                signal = Signal(None, Action.NONE, "btc_bear_long_gate", 0.0,
                                symbol=symbol, source=source, timestamp=close_time.to_pydatetime())
            if signal.action in {Action.OPEN, Action.CLOSE}:
                actionable_signals_checked += 1
                signal_time = pd.Timestamp(signal.timestamp) if signal.timestamp is not None else None
                if (signal_time is None or signal_time.tzinfo is None
                        or signal_time > close_time):
                    future_signal_timestamp_violations += 1
            if signal.action is Action.CLOSE:
                pending[symbol] = {"action": Action.CLOSE.value, "reason": signal.reason,
                                   "source": signal.source, "eligible": True}
            elif (signal.action is Action.OPEN and signal.side is not None
                  and (symbol in eligible or exempted or (signal.close_first and symbol in active))):
                fframe = funding.get(symbol)
                rate = None
                if fframe is not None and not fframe.empty:
                    fidx = int(fframe.index.searchsorted(close_time, side="left") - 1)
                    if fidx >= 0:
                        rate = float(fframe["funding_rate"].iloc[fidx])
                pending[symbol] = {
                    "action": Action.OPEN.value, "side": signal.side.value, "reason": signal.reason,
                    "source": signal.source,
                    "strength": signal.strength, "close_first": signal.close_first,
                    "size_multiplier": signal.size_multiplier,
                    "atr": float(atr_value), "regime": regime.regime.value,
                    "funding_rate": rate, "eligible": True,
                    "hot_universe_exempted": exempted,
                    **trend_intent_context,
                }
                if signal.action is Action.OPEN:
                    lookback = settings.stops.swing_lookback_1h
                    swing = frame.iloc[max(0, idx - lookback + 1):idx + 1]
                    row = regimes_frame.iloc[ridx]
                    pending[symbol].update({
                        "structure_low": float(swing["low"].min()) if not swing.empty else None,
                        "structure_high": float(swing["high"].max()) if not swing.empty else None,
                        "range_low": float(row["range_low"]) if np.isfinite(row.get("range_low", np.nan)) else None,
                        "range_high": float(row["range_high"]) if np.isfinite(row.get("range_high", np.nan)) else None,
                        "signal_close": float(candle["close"]),
                    })

        mark_prices = {symbol: float(current_rows[symbol]["close"]) for symbol in active if symbol in current_rows}
        marks.update(mark_prices)
        equity_value = equity()
        portfolio.observe_equity(max(equity_value, 1e-12), close_time.to_pydatetime(), settings)
        equity_values.append(equity_value)
        equity_times.append(close_time)

    # Force any remaining positions closed at the final observed close.
    final_time = pd.Timestamp(timestamps[-1] + HOUR)
    for symbol in list(active):
        if symbol in last_prices:
            close_position(symbol, last_prices[symbol], final_time, "backtest_end")
    if equity_values:
        equity_values[-1] = equity()
    curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_times), name="equity")
    metrics = summarize_backtest(trades, curve, initial_equity)
    overlap_pairs, cross_source_pairs, invalid_trade_intervals = _trade_overlap_counts(trades)
    integrity_checks: dict[str, bool | int] = {
        "closed_bar_input_checks": closed_bar_input_checks,
        "closed_bar_input_violations": closed_bar_input_violations,
        "all_signal_inputs_use_closed_bars": closed_bar_input_violations == 0,
        "actionable_signals_checked": actionable_signals_checked,
        "future_signal_timestamp_violations": future_signal_timestamp_violations,
        "all_signal_timestamps_at_or_before_close": future_signal_timestamp_violations == 0,
        "overlapping_position_pairs_same_symbol": overlap_pairs,
        "cross_strategy_position_overlap_pairs": cross_source_pairs,
        "invalid_trade_intervals": invalid_trade_intervals,
        "no_dual_strategy_position_overlap": cross_source_pairs == 0,
        "no_overlapping_positions_per_symbol": overlap_pairs == 0 and invalid_trade_intervals == 0,
    }
    integrity_checks["all_passed"] = all(
        bool(integrity_checks[key])
        for key in (
            "all_signal_inputs_use_closed_bars",
            "all_signal_timestamps_at_or_before_close",
            "no_dual_strategy_position_overlap",
            "no_overlapping_positions_per_symbol",
        )
    )
    fingerprint = _result_fingerprint(
        tuple(trades), curve, metrics, skipped, rejected_by_source,
        market_risk_execution, integrity_checks, symbols_seen=len(symbols),
        ranking_hours=len(timestamps), shadow_long_signals=shadow_long_signals,
        long_trade_excursions=long_trade_excursions,
    )
    return PortfolioBacktestResult(tuple(trades), curve, metrics, skipped, len(symbols), len(timestamps),
                                   tuple(shadow_long_signals), tuple(long_trade_excursions),
                                   rejected_by_source, market_risk_execution,
                                   integrity_checks, fingerprint,
                                   trend_funnel=tuple(trend_funnel_events),
                                   trend_funnel_fill_rejections=tuple(trend_funnel_fill_rejections),
                                   trend_stops=tuple(trend_stop_records),
                                   hot_universe_exemption_records=tuple(hot_universe_exemption_records))
