"""Offline hourly scenario using the production Runner's decision functions.

This is conditional research, not tick/order-book reconstruction. No exchange
client is created. Quotes/depth are explicitly modeled, not observed evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import ccxt
from loguru import logger
import pandas as pd

from bot.backtest.engine import Trade
from bot.backtest.metrics import summarize_backtest
from bot.backtest.research_data import (
    HOUR_MS, aggregate_closed_bars, digest_file, load_manifest, load_stream, profile_bars, utc,
    write_json_once,
)
from bot import risk as risk_module
from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import hot_activity_metrics, rank_hot_candidates
from bot.exchange.clock import activate_clock, active_clock
from bot.execution import EntryPreflightRejected
from bot.indicators import atr
from bot.lead_risk import settings_for_execution
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskState, classify_market_risk
from bot.mr_candidate import deviation_entry_signal_at_ema
from bot.regime import Regime, classify_regimes
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner

HOUR = pd.Timedelta(1, unit="h")
FOUR_HOURS = pd.Timedelta(4, unit="h")
RAW_SIGNAL_FLAGS = TimeFilterFlags(True, True, False, False)
QUOTE_MODEL = "hour-boundary contract open; one tick per side; synthetic depth; configured fill slippage"


@dataclass(frozen=True)
class ResearchHypothesis:
    name: str = "baseline"
    prefilter_daily_history: bool = False
    stop_hard_cap_atr: float | None = None
    deviation_rule: str = "fixed_1pct"
    reversal_body_midpoint: bool = False
    minimum_ema_space_r: float = 0.0
    funding_adverse_threshold: float | None = None


HYPOTHESES = {
    "baseline": ResearchHypothesis(),
    "daily_history_prefilter": ResearchHypothesis(
        name="daily_history_prefilter", prefilter_daily_history=True),
    "structure_stop_2p5": ResearchHypothesis(
        name="structure_stop_2p5", stop_hard_cap_atr=2.5),
    "atr_deviation_0p8": ResearchHypothesis(
        name="atr_deviation_0p8", deviation_rule="atr_0p8_floor_1pct"),
    "impulse_counter_deviation_1p2": ResearchHypothesis(
        name="impulse_counter_deviation_1p2", deviation_rule="impulse_counter_1p2_floor_2pct"),
    "reversal_body_midpoint": ResearchHypothesis(
        name="reversal_body_midpoint", reversal_body_midpoint=True),
    "ema_space_1r": ResearchHypothesis(
        name="ema_space_1r", minimum_ema_space_r=1.0),
    "directional_funding_0p08": ResearchHypothesis(
        name="directional_funding_0p08", funding_adverse_threshold=0.0008),
}


class ReplayClock:
    def __init__(self, at):
        self.at = utc(at)
        self.window_end = None

    def now(self):
        return self.at.to_pydatetime()

    def lower_now(self):
        return self.now()

    def require_fresh(self):
        return None  # Deterministic scenario time; no wall-clock or exchange reads.


def _canonical(symbol):
    return symbol[:-4] + "/USDT:USDT"


def _rows(frame):
    values = frame[["open", "high", "low", "close", "volume"]]
    return [[int(at.timestamp() * 1000), *row]
            for at, row in zip(values.index, values.itertuples(index=False, name=None))]


class ReplayPublic:
    """Only explicitly supplied public rows and a labeled quote scenario."""

    def __init__(self, frames, metadata, clock):
        self.frames, self.clock = frames, clock
        self.markets = {}
        self.hourly_activity = {}
        for row in metadata:
            symbol = _canonical(row["symbol"])
            filters = {item["filterType"]: item for item in row["filters"]}
            lot = filters["LOT_SIZE"]
            price = filters["PRICE_FILTER"]
            notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
            self.markets[symbol] = {"id": row["symbol"], "info": row,
                "precision": {"amount": float(lot["stepSize"]), "price": float(price["tickSize"])},
                "limits": {"amount": {"min": float(lot["minQty"])},
                           "cost": {"min": float(notional.get("notional", notional.get("minNotional", 0)))}}}
            hourly = frames[symbol]["contract_1h"]
            if {"quote_volume", "trade_count"} <= set(hourly.columns):
                activity = hourly[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
                self.hourly_activity[symbol] = (
                    activity, hourly["quote_volume"].to_numpy(), hourly["trade_count"].to_numpy(),
                )

    def contract_price(self, symbol):
        frame = self.frames[symbol]["contract_1h"]
        at = self.clock.at.floor("h")
        if at != self.clock.window_end and at in frame.index:
            return float(frame.loc[at, "open"])
        closed = frame.loc[frame.index + HOUR <= self.clock.at]
        if closed.empty or closed.index[-1] + HOUR != at:
            raise ValueError("research contract quote is missing")
        return float(closed.iloc[-1]["close"])

    def mark_price(self, symbol):
        frame = self.frames[symbol]["mark_1h"]
        at = self.clock.at.floor("h")
        if at != self.clock.window_end and at in frame.index:
            return float(frame.loc[at, "open"])
        closed = frame.loc[frame.index + HOUR <= self.clock.at]
        if closed.empty or closed.index[-1] + HOUR != at:
            raise ValueError("research mark price is missing")
        return float(closed.iloc[-1]["close"])

    def fetch_ticker(self, symbol):
        return {"last": self.contract_price(symbol), "timestamp": int(self.clock.at.timestamp() * 1000)}

    def fetch_order_book(self, symbol, *, limit):
        price = self.contract_price(symbol)
        tick = self.markets[symbol]["precision"]["price"]
        return {"timestamp": int(self.clock.at.timestamp() * 1000),
                "bids": [[max(tick, price - tick), 1e15]], "asks": [[price + tick, 1e15]],
                "research_model": QUOTE_MODEL}

    def fetch_ohlcv(self, symbol, timeframe, *, limit):
        if timeframe != "1d":
            raise ValueError("unexpected research OHLCV request")
        frame = self.frames[symbol]["daily"]
        closed = _closed_prefix(frame, self.clock.at, pd.Timedelta(1, unit="D"), limit - 1)
        # Production fetches a fixed count including today's unfinished candle.
        # Its fields are synthetic and are discarded by the unchanged Runner.
        price = self.contract_price(symbol)
        return _rows(closed) + [[int(self.clock.at.floor("D").timestamp() * 1000),
                                 price, price, price, price, 0]]

    def fetch_funding_rate(self, symbol):
        funding = self.frames[symbol]["funding"]
        index = funding.index.searchsorted(self.clock.at, side="right") - 1
        if index < 0 or self.clock.at - funding.index[index] > pd.Timedelta(12, unit="h"):
            return {"fundingRate": None}
        return {"fundingRate": float(funding.iloc[index]["funding_rate"])}

    def price_to_precision(self, symbol, price):
        return ccxt.decimal_to_precision(str(price), ccxt.ROUND,
            self.markets[symbol]["precision"]["price"], ccxt.TICK_SIZE)

    def amount_to_precision(self, symbol, amount):
        return ccxt.decimal_to_precision(str(amount), ccxt.TRUNCATE,
            self.markets[symbol]["precision"]["amount"], ccxt.TICK_SIZE)


class ReplayExecution:
    """In-memory wallet behind the existing Runner protocol; no signed methods."""

    label = "Research hourly scenario"
    adopt_manual_positions = False

    def __init__(self, public, state, settings, initial_equity):
        self.exchange, self.state, self.settings = public, state, settings
        self.wallet = float(initial_equity)
        self.positions, self.stops, self.history = {}, {}, {}
        self.lead_market_ids = {row["id"] for row in public.markets.values()}
        self.counter = 0
        self.exit_override = {}
        self.funding_cursor = public.clock.at

    def sync_snapshot(self):
        positions, unrealized, margin = {}, 0.0, 0.0
        for symbol, position in self.positions.items():
            mark = self.exchange.mark_price(symbol)
            quantity, entry = position["quantity"], position["entry_price"]
            pnl = quantity * (mark - entry) * (1 if position["side"] == "long" else -1)
            unrealized += pnl
            margin += quantity * entry / 3
            positions[symbol] = {**position, "mark_price": mark, "unrealized_pnl": pnl}
        snapshot = {"at": self.exchange.clock.now().isoformat(), "positions": positions,
                    "equity": self.wallet + unrealized, "usdt_balance": self.wallet,
                    "available_usdt": self.wallet - margin}
        runner._observe_account_equity(self.state, snapshot, self.settings, self.exchange.clock.now())
        return snapshot

    def audit_active_protective_stops(self, positions, orders):
        errors = []
        if set(positions) != set(self.positions):
            errors.append("research position mismatch")
        for symbol in positions:
            if orders.get(symbol, {}).get("stop") != self.stops.get(symbol):
                errors.append("research stop mismatch")
        return errors

    def cancel_orphaned_bot_stops(self, _live):
        return []

    def open(self, intent, *, stop_price, before_submit, submission_deadline, **kwargs):
        intent, stop_price = before_submit(intent, stop_price)
        if self.exchange.clock.now() > submission_deadline:
            raise ValueError("research submission is past deadline")
        quantity = float(self.exchange.amount_to_precision(intent.symbol, intent.quantity))
        if quantity <= 0 or intent.symbol in self.positions:
            raise ValueError("invalid research entry")
        sign = 1 if intent.side is Side.LONG else -1
        price = intent.price * (1 + sign * self.settings.fees.slippage_bps / 10000)
        fee = quantity * price * self.settings.fees.taker_bps / 10000
        if self.sync_snapshot()["available_usdt"] < quantity * price / 3 + fee:
            raise EntryPreflightRejected("research_3x_margin_insufficient")
        self.counter += 1
        self.wallet -= fee
        live = {"side": intent.side.value, "quantity": quantity, "entry_price": price}
        self.positions[intent.symbol] = live
        opened_at = self.exchange.clock.now().isoformat()
        self.history[(intent.symbol, opened_at)] = {**live, "entry_fee": fee,
                                                   "entry_reference": intent.price}
        stop = {"id": f"research-stop-{self.counter}", "stop_price": stop_price, "amount": quantity}
        self.stops[intent.symbol] = stop
        return {"position": live, "stop": stop,
                "fill": {"id": f"research-open-{self.counter}", "filled": quantity,
                         "average": price, "fee_cost": fee, "closed_at": opened_at}}

    def close(self, position, *, stop=None, reason=None):
        if position.symbol not in self.positions:
            return {"already_flat": True}
        sign = 1 if position.side is Side.LONG else -1
        reference = self.exit_override.pop(position.symbol, self.exchange.contract_price(position.symbol))
        price = reference * (1 - sign * self.settings.fees.slippage_bps / 10000)
        gross = position.quantity * (price - position.entry_price) * sign
        fee = position.quantity * price * self.settings.fees.taker_bps / 10000
        self.wallet += gross - fee
        self.positions.pop(position.symbol)
        self.stops.pop(position.symbol)
        self.counter += 1
        saved = self.history[(position.symbol, position.opened_at.isoformat())]
        saved.update(exit_reference=reference, exit_price=price)
        return {"fill": {"id": f"research-close-{self.counter}", "filled": position.quantity,
                         "average": price, "fee_cost": fee, "realized_pnl": gross,
                         "closed_at": self.exchange.clock.now().isoformat()}}

    def funding_fee(self, symbol, start, end):
        saved = self.history[(symbol, start.isoformat())]
        rows = self.exchange.frames[symbol]["funding"]
        rows = rows.loc[(rows.index > pd.Timestamp(start)) & (rows.index <= pd.Timestamp(end))]
        sign = 1 if saved["side"] == "long" else -1
        return float(-(rows["funding_rate"] * rows["mark_price"]).sum() * saved["quantity"] * sign)

    def settle_funding(self):
        for symbol, position in self.state.portfolio.positions.items():
            rows = self.exchange.frames[symbol]["funding"]
            rows = rows.loc[(rows.index > max(self.funding_cursor, pd.Timestamp(position.opened_at)))
                            & (rows.index <= self.exchange.clock.at)]
            sign = 1 if position.side is Side.LONG else -1
            self.wallet -= float((rows["funding_rate"] * rows["mark_price"]).sum()) * position.quantity * sign
        self.funding_cursor = self.exchange.clock.at

    def replace_stop(self, position, *, stop, stop_price):
        old = self.stops[position.symbol]
        if stop != old or (stop_price - old["stop_price"]) * (1 if position.side is Side.LONG else -1) < 0:
            raise ValueError("research stop must only tighten")
        self.counter += 1
        updated = {**old, "id": f"research-stop-{self.counter}", "stop_price": stop_price}
        self.stops[position.symbol] = updated
        return updated


def _daily_history_is_rankable(daily, at, span):
    expected_close = at.floor("D")
    closed_count = daily.index.searchsorted(expected_close - pd.Timedelta(1, unit="D"), side="right")
    latest_closed = (daily.index[closed_count - 1] + pd.Timedelta(1, unit="D")
                     if closed_count else None)
    return closed_count >= span + 1 and latest_closed == expected_close


def _rank(public, one, settings, at, *, prefilter_daily_history=False):
    candidates, tickers, activity = [], {}, {}
    daily_history_rejected = 0
    excluded = {"BTCUSDT", "ETHUSDT", *settings.universe.exclude_symbols}
    for symbol, frame in one.items():
        market = public.markets[symbol]
        if market["id"] in excluded or len(frame) < 25:
            continue
        onboard = int(market["info"]["onboardDate"])
        if at.timestamp() * 1000 - onboard < settings.universe.min_list_days * 86400000:
            continue
        end = public.frames[symbol]["contract_1h"].index.searchsorted(at - HOUR, side="right")
        prices, quote_volume, trade_count = public.hourly_activity[symbol]
        quote = float(quote_volume[end - 24:end].sum())
        if quote < settings.universe.min_quote_volume_24h:
            continue
        if prefilter_daily_history:
            daily = public.frames[symbol]["daily"]
            if not _daily_history_is_rankable(
                    daily, at, settings.symbol_regime.ema_slow_bars):
                daily_history_rejected += 1
                continue
        candidates.append((symbol, quote))
        tickers[symbol] = {"quoteVolume": quote, "count": int(trade_count[end - 24:end].sum())}
        # Reuse the exact production activity score over the same 25 closed bars.
        now_ms = int(at.timestamp() * 1000)
        rows = [[now_ms - (25 - i) * HOUR_MS, *values]
                for i, values in enumerate(prices[end - 25:end])]
        activity[symbol] = hot_activity_metrics(rows, now_ms=now_ms)
    ranked = rank_hot_candidates(candidates, tickers, activity, limit=20)
    return ranked, {"rankable_before_prefilter": len(candidates) + daily_history_rejected,
                    "rankable_after_prefilter": len(candidates),
                    "daily_history_rejected": daily_history_rejected,
                    "top20_count": len(ranked)}


class _ClosedRegimes(dict):
    def __init__(self, four, settings):
        super().__init__()
        self.four, self.settings = four, settings

    def get(self, symbol, default=None):
        if symbol not in self and symbol in self.four:
            self[symbol] = classify_regimes(self.four[symbol], self.settings)
        return super().get(symbol, default)


def _entry_deviation_thresholds(frame4, settings, regimes, symbol, hypothesis):
    rule = hypothesis.deviation_rule
    if rule == "fixed_1pct":
        return {Side.LONG: runner.DEV_PCT, Side.SHORT: runner.DEV_PCT}
    close = float(frame4["close"].iloc[-1])
    atr_value = float(atr(frame4, settings.lookback.atr_len).iloc[-1])
    if close <= 0 or not math.isfinite(atr_value) or atr_value <= 0:
        return {Side.LONG: math.inf, Side.SHORT: math.inf}
    atr_pct = atr_value / close
    if rule == "atr_0p8_floor_1pct":
        threshold = max(0.01, 0.8 * atr_pct)
        return {Side.LONG: threshold, Side.SHORT: threshold}
    if rule == "impulse_counter_1p2_floor_2pct":
        threshold = max(0.02, 1.2 * atr_pct)
        regimes_for_symbol = regimes.get(symbol)
        current = (regimes_for_symbol.iloc[-1]["regime"]
                   if regimes_for_symbol is not None and len(regimes_for_symbol) else None)
        current = current.value if isinstance(current, Regime) else str(current)
        return {
            Side.LONG: threshold if current == Regime.IMPULSE_DOWN.value else runner.DEV_PCT,
            Side.SHORT: threshold if current == Regime.IMPULSE_UP.value else runner.DEV_PCT,
        }
    raise ValueError(f"unknown entry deviation rule: {rule}")


def _raw_signal_side(candle, ema, settings, thresholds=None):
    thresholds = thresholds or {Side.LONG: runner.DEV_PCT, Side.SHORT: runner.DEV_PCT}
    for side, allowed in ((Side.LONG, (True, False)), (Side.SHORT, (False, True))):
        flags = TimeFilterFlags(allowed[0], allowed[1], False, False)
        signal = deviation_entry_signal_at_ema(
            ema, candle["open"], candle["close"], settings, flags,
            deviation_pct=thresholds[side],
        )
        if signal.action is Action.OPEN:
            return side
    return None


def _has_raw_signal(candle, ema, settings, thresholds=None):
    return _raw_signal_side(candle, ema, settings, thresholds) is not None


def _reversal_body_midpoint_pass(frame1, side):
    if frame1 is None or len(frame1) < 2:
        return False
    previous = frame1.iloc[-2]
    midpoint = (float(previous["open"]) + float(previous["close"])) / 2
    current_close = float(frame1.iloc[-1]["close"])
    return current_close > midpoint if side is Side.LONG else current_close < midpoint


def _ema_space_r(side, close, stop, ema):
    risk = abs(float(close) - float(stop))
    if risk <= 0 or not all(math.isfinite(float(value)) for value in (close, stop, ema)):
        return float("-inf")
    direction = 1 if side is Side.LONG else -1
    return direction * (float(ema) - float(close)) / risk


def _funding_gate_for_side(rate, side, threshold):
    if rate is None:
        return "funding_rate_missing"
    try:
        rate = float(rate)
    except (TypeError, ValueError, OverflowError):
        return "funding_rate_missing"
    if not math.isfinite(rate):
        return "funding_rate_missing"
    adverse = rate >= threshold if side is Side.LONG else rate <= -threshold
    return "funding_rate_blocked" if adverse else None


def _with_entry_regime(candidate, regimes):
    # In Runner candidate generation the legacy regime is metadata. The daily
    # and structure gates are independent of it. Resolve it before risk/entry,
    # so regime-fade exits receive the exact original entry regime.
    regime = regimes.get(candidate[0]).iloc[-1]["regime"]
    return (*candidate[:8], regime.value if isinstance(regime, Regime) else Regime.MID.value,
            *candidate[9:])


def _check_protective_stops(state, execution, public, one, clock, fee, slip, *, open_only=False):
    for symbol, position in list(state.portfolio.positions.items()):
        stop = execution.stops[symbol]["stop_price"]
        if open_only:
            open_price = extreme = public.contract_price(symbol)
        else:
            candle = one[symbol].iloc[-1]
            open_price = float(candle["open"])
            extreme = candle["low"] if position.side is Side.LONG else candle["high"]
        hit = extreme <= stop if position.side is Side.LONG else extreme >= stop
        if hit:
            reference = min(open_price, stop) if position.side is Side.LONG else max(open_price, stop)
            execution.exit_override[symbol] = reference
            runner._close_position(True, execution, state, symbol, reference,
                                   "protective_stop_exchange_algo", clock.now(), fee, slip)


def replay(frames, metadata, settings, *, start, end, initial_equity=1000, mode="lead_usdm",
           hypothesis="baseline", progress=print):
    """The timing and fills are modeled; all strategy decisions call the Runner."""
    start, end = utc(start), utc(end)
    if end <= start or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("ordered hourly replay bounds are required")
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown research hypothesis: {hypothesis}")
    variant = HYPOTHESES[hypothesis]
    clock = ReplayClock(start)
    clock.window_end = end
    previous_clock = active_clock()
    activate_clock(clock)
    try:
        return _replay(frames, metadata, settings, start, end, initial_equity, mode, clock,
                       progress, variant)
    finally:
        activate_clock(previous_clock)


def _replay(frames, metadata, settings, start, end, initial_equity, mode, clock, progress, hypothesis):
    state = MultiPaperState(initial_equity=initial_equity, cash=initial_equity,
                            execution_mode=mode, execution_status="synchronized")
    public = ReplayPublic(frames, metadata, clock)
    execution = ReplayExecution(public, state, settings, initial_equity)
    settings = settings_for_execution(settings, mode)
    if hypothesis.stop_hard_cap_atr is not None:
        settings = settings.model_copy(update={"stops": settings.stops.model_copy(update={
            "hard_cap_atr": hypothesis.stop_hard_cap_atr,
        })})
    if hypothesis.funding_adverse_threshold is not None:
        settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={
            "funding_abs_block": hypothesis.funding_adverse_threshold,
        })})
    execution.settings = settings
    fee, slip = settings.fees.taker_bps / 10000, settings.fees.slippage_bps / 10000
    cache = runner.ClosedMarketCache()
    regimes, four_key = {}, None
    reentry_events, event_cursor = [], 0
    equity, risk_states, ranking_stats, research_filter_rejections = {}, Counter(), Counter(), Counter()
    symbols = sorted(frames)
    for number, at in enumerate(pd.date_range(start, end, freq="h")):
        clock.at = at + pd.Timedelta(1, unit="s") if at < end else at
        one = {symbol: _closed_prefix(frames[symbol]["contract_1h"], at, HOUR, 59) for symbol in symbols}
        if four_key != at.floor("4h"):
            four = {symbol: _closed_prefix(frames[symbol]["contract_4h"], at, FOUR_HOURS, 99)
                    for symbol in symbols}
            regimes = _ClosedRegimes(four, settings)
            span = settings.symbol_regime.ema_slow_bars
            ema = {symbol: (float(frame["close"].ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1])
                            if len(frame) >= span else float("nan")) for symbol, frame in four.items()}
            four_key = at.floor("4h")
            market_risk = classify_market_risk(four["BTC/USDT:USDT"], settings, now=clock.now())
        execution.settle_funding()
        # Contract stops active at the start of the completed hour. Favorable
        # extremes never retroactively move a stop inside that same hour.
        _check_protective_stops(state, execution, public, one, clock, fee, slip)
        if at == end:
            break
        _check_protective_stops(state, execution, public, one, clock, fee, slip, open_only=True)
        if not runner._demo_sync_stopped(execution, state, clock.now()):
            raise ValueError("research account reconciliation failed")
        if not runner._audit_demo_protective_stops(execution, state, clock.now()):
            raise ValueError("research protection audit failed")
        runner._manage_demo_positions(public, execution, state, settings, clock.now(), fee, slip)
        runner._evaluate_exits(state, execution, True, one, regimes, settings, fee, slip, clock.now())
        runner._manage_demo_positions(public, execution, state, settings, clock.now(), fee, slip)
        hotlist, rank_stats = _rank(public, one, settings, at,
                                    prefilter_daily_history=hypothesis.prefilter_daily_history)
        ranking_stats.update(rank_stats)
        state.watchlist = [{**row, "rank": i + 1} for i, row in enumerate(hotlist)]
        risk_states[market_risk.state.value] += 1
        # Only this small subset is required to reconstruct the entry loss lock.
        reentry_events.extend(e for e in state.events[event_cursor:]
                              if (e.get("event") == "structure_reentry_confirmed"
                                  or e.get("action") == "CLOSE"))
        event_cursor = len(state.events)
        reentry = runner._structure_reentry_states(reentry_events, symbols)
        if market_risk.state is not MarketRiskState.PAUSE_NEW:
            candidates = []
            for row in hotlist:
                symbol = row["symbol"]
                if symbol in state.portfolio.positions:
                    continue
                thresholds = _entry_deviation_thresholds(four[symbol], settings, regimes,
                                                          symbol, hypothesis)
                if not _has_raw_signal(one[symbol].iloc[-1], ema[symbol], settings, thresholds):
                    # Runner checks breakout confirmations even when there is no new signal.
                    if any(reentry[(symbol, side)][2] for side in (Side.LONG, Side.SHORT)):
                        runner._record_structure_reentry_confirmations(state, symbol, one[symbol], clock.now(), reentry)
                    continue
                raw_side = _raw_signal_side(one[symbol].iloc[-1], ema[symbol], settings, thresholds)
                if raw_side is None:
                    continue
                if (hypothesis.reversal_body_midpoint
                        and not _reversal_body_midpoint_pass(one[symbol], raw_side)):
                    research_filter_rejections["reversal_body_midpoint"] += 1
                    continue
                candidate = runner._entry_candidate(public, state, True, symbol, one, four,
                    {}, settings, clock.now(), reentry, closed_market_cache=cache,
                    deviation_pct=thresholds[raw_side])
                if candidate is not None:
                    if _ema_space_r(raw_side, candidate[3], candidate[4], ema[symbol]) < hypothesis.minimum_ema_space_r:
                        research_filter_rejections["ema_space_below_minimum"] += 1
                        continue
                    candidates.append(_with_entry_regime(candidate, regimes))
            for candidate in sorted(candidates, key=lambda row: -row[1]):
                previous_runner_funding_gate = runner.funding_rate_rejection_reason
                previous_risk_funding_gate = risk_module.funding_rate_rejection_reason
                if hypothesis.funding_adverse_threshold is not None:
                    side = candidate[2]
                    threshold = hypothesis.funding_adverse_threshold
                    gate = lambda rate, _settings: _funding_gate_for_side(rate, side, threshold)
                    runner.funding_rate_rejection_reason = gate
                    risk_module.funding_rate_rejection_reason = gate
                try:
                    runner._submit_entry_candidate(public, state, execution, True, public.markets,
                        (*candidate, None), settings, settings.sizing.max_symbols, fee, slip,
                        clock.now(), market_risk)
                finally:
                    runner.funding_rate_rejection_reason = previous_runner_funding_gate
                    risk_module.funding_rate_rejection_reason = previous_risk_funding_gate
        state.exchange_snapshot = execution.sync_snapshot()
        equity[at] = state.current_equity
        if number % 240 == 0:
            progress(f"replay {number}h {at.isoformat()} equity={state.current_equity:.4f} positions={len(state.portfolio.positions)}")
    clock.at = end
    execution.settle_funding()
    for symbol in list(state.portfolio.positions):
        runner._close_position(True, execution, state, symbol, public.contract_price(symbol),
                               "research_window_close", clock.now(), fee, slip)
    state.exchange_snapshot = execution.sync_snapshot()
    equity[end] = state.current_equity
    closes = [e for e in state.events if e.get("action") == "CLOSE"]
    trades = []
    for event in closes:
        opened = pd.Timestamp(event["opened_at"])
        saved = execution.history[(event["symbol"], opened.isoformat())]
        quantity = event["quantity"]
        slippage = quantity * (abs(saved["entry_price"] - saved["entry_reference"])
                               + abs(saved["exit_price"] - saved["exit_reference"]))
        trades.append(Trade(event["symbol"], Side(event["side"]), opened, pd.Timestamp(event["at"]),
            saved["entry_price"], event["price"], quantity, event["gross_pnl"], event["net_pnl"],
            event["fees"], slippage, -event["funding_paid"], "runner_mr_baseline", event["reason"]))
    curve = pd.Series(equity, name="equity")
    metrics = summarize_backtest(trades, curve, initial_equity)
    residual = state.current_equity - initial_equity - sum(trade.net_pnl for trade in trades)
    if not math.isclose(residual, 0, abs_tol=1e-7):
        raise ValueError("research wallet and trade PnL do not reconcile")
    return {"hypothesis": hypothesis.name, "hypothesis_parameters": {
                "prefilter_daily_history": hypothesis.prefilter_daily_history,
                "stop_hard_cap_atr": hypothesis.stop_hard_cap_atr,
                "deviation_rule": hypothesis.deviation_rule,
                "reversal_body_midpoint": hypothesis.reversal_body_midpoint,
                "minimum_ema_space_r": hypothesis.minimum_ema_space_r,
                "funding_adverse_threshold": hypothesis.funding_adverse_threshold,
            },
            "ranking_metrics": dict(ranking_stats),
            "research_filter_rejections": dict(research_filter_rejections),
            "metrics": metrics, "reconciliation_residual": residual,
            "quote_model": QUOTE_MODEL, "fixed_leverage_model": 3,
            "market_risk_hours": dict(risk_states),
            "rejections": dict(Counter(e.get("reason") for e in state.events
                if e.get("event") in {"entry_risk_rejected", "entry_preflight_decision", "entry_signal_blocked"}
                and e.get("reason"))),
            "exit_reasons": dict(Counter(e["reason"] for e in closes)),
            "decisions": [e for e in state.events if e.get("event") == "entry_preflight_decision"][:100],
            "trades": [{**e, "execution_source": "research_ohlc_model"} for e in closes],
            "equity": [{"at": at.isoformat(), "equity": value} for at, value in curve.items()]}


def _closed_prefix(frame, as_of, duration, limit):
    stop = frame.index.searchsorted(as_of - duration, side="right")
    return frame.iloc[max(0, stop - limit):stop]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["lead_usdm", "demo_usdm"], default="lead_usdm")
    parser.add_argument("--initial-equity", type=float, default=1000)
    parser.add_argument("--window", choices=["train", "validation", "test"],
                        help="run one of the already frozen windows")
    parser.add_argument("--scenario", choices=["baseline", "fee_1p5_slip_2", "fee_2_slip_3"],
                        help="run one planned cost scenario in the selected window")
    parser.add_argument("--hypothesis", choices=sorted(HYPOTHESES), default="baseline",
                        help="run one explicitly defined offline research variant")
    args = parser.parse_args()
    manifest = load_manifest(args.snapshot)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not math.isfinite(args.initial_equity) or args.initial_equity <= 0:
        parser.error("initial equity must be positive")
    if args.scenario and not args.window:
        parser.error("--scenario requires --window")
    jobs = []
    for window in plan["windows"]:
        if args.window and window["name"] != args.window:
            continue
        if not (utc(manifest["start_utc"]) < utc(window["start"]) < utc(window["end_exclusive"])
                <= utc(manifest["end_exclusive_utc"])):
            parser.error("evaluation window is outside the public snapshot")
        scenarios = plan["cost_scenarios"] if window["name"] == "test" else plan["cost_scenarios"][:1]
        jobs.extend((window, scenario) for scenario in scenarios
                    if not args.scenario or args.scenario == scenario["name"])
    if not jobs:
        parser.error("the requested window and cost scenario are absent from the plan")
    frames = {}
    for identifier in manifest["symbols"]:
        streams = {stream: load_stream(args.snapshot, identifier, stream)
                   for stream in ("contract_1h", "contract_4h", "mark_1h", "funding")}
        for stream, hours in (("contract_1h", 1), ("contract_4h", 4), ("mark_1h", 1)):
            quality = profile_bars(streams[stream], hours * HOUR_MS, start=manifest["start_utc"],
                                   end=manifest["end_exclusive_utc"])
            if not quality["valid"]:
                parser.error(f"{identifier} {stream}: {', '.join(quality['errors'])}")
        streams["daily"] = aggregate_closed_bars(streams["contract_4h"], 4, 24,
                                                as_of=manifest["end_exclusive_utc"])
        frames[_canonical(identifier)] = streams
    metadata = json.loads((args.snapshot / "exchange_info.json").read_text(encoding="utf-8"))["symbols"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = load_settings(PROJECT_ROOT / "configs/default.yaml", env_file=None)
    sources = ["scripts/replay_mr_baseline.py", "src/bot/backtest/research_data.py",
               "scripts/run_mr_candidate_paper.py", "src/bot/mr_candidate.py", "src/bot/risk.py",
               "src/bot/lead_risk.py", "src/bot/time_filter.py", "configs/default.yaml"]
    write_json_once(args.output_dir / "run_inputs.json", {
        "plan_sha256": digest_file(args.plan), "manifest_sha256": digest_file(args.snapshot / "manifest.json"),
        "source_sha256": {name: digest_file(PROJECT_ROOT / name) for name in sources},
        "initial_equity": args.initial_equity, "mode": args.mode, "jobs": jobs,
        "hypothesis": args.hypothesis, "promotion_allowed": False, "quote_model": QUOTE_MODEL,
    })
    logger.remove()
    logger.add(lambda message: None, level="CRITICAL")
    summaries = []
    for window, scenario in jobs:
        configured = settings.model_copy(update={"fees": settings.fees.model_copy(update={
            "taker_bps": settings.fees.taker_bps * scenario["fee_multiplier"],
            "slippage_bps": settings.fees.slippage_bps * scenario["slippage_multiplier"],
        })})
        label = window["name"] + "_" + scenario["name"]
        print("starting " + label, flush=True)
        result = replay(frames, metadata, configured, start=window["start"], end=window["end_exclusive"],
            initial_equity=args.initial_equity, mode=args.mode,
            hypothesis=args.hypothesis,
            progress=lambda message: print(message, flush=True))
        write_json_once(args.output_dir / (label + ".json"), result)
        summaries.append({"window": window, "scenario": scenario,
                          **{key: value for key, value in result.items() if key not in {"trades", "equity", "decisions"}}})
        write_json_once(args.output_dir / (label + "_summary.json"), summaries[-1])
        print(json.dumps({"completed": label, "metrics": result["metrics"]}), flush=True)
    write_json_once(args.output_dir / "summary.json", {"runs": summaries, "promotion_allowed": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
