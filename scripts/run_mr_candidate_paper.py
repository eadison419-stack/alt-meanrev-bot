"""Run the ``mr_candidate`` mean-reversion strategy on paper, Demo, or Lead Portfolio.

Independent entry point mirroring ``scripts/run_cta_paper.py``, but for the
standalone mean-reversion candidate (``run_portfolio_backtest(mr_candidate=True)``):
hotness top-N universe, EMA50-deviation entries (long below / short above), and the
EMA20 Bollinger mid-band protection plus the standard structure-stop / staged
profit-lock / trailing / invalidation exits.

Execution modes (same strategy loop):
  BINANCE_TESTNET=false -> local_paper  (local virtual fills, no orders, no key)
  BINANCE_TESTNET=true  -> demo_usdm   (Binance USD-M testnet, needs API key)
  --live + BOT_MODE=live -> lead_usdm  (production Lead Portfolio, interactive confirmation)

Run (Windows, from project root):
  python scripts/run_mr_candidate_paper.py --once   # local paper, one cycle
  python scripts/run_mr_candidate_paper.py          # local paper, continuous
  python scripts/run_mr_candidate_paper.py --demo   # Binance testnet demo
  python scripts/run_mr_candidate_paper.py --live   # production Lead Portfolio; type LEAD to confirm
  python scripts/run_mr_candidate_paper.py --intrabar-entries  # closed-1H entries, live-stop local Paper

The isolated local Paper mode uses the daily EMA50 direction gate and closed-1H
reversal entries. It keeps live-price stop monitoring and must not be combined
with --demo or --live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import ccxt
import pandas as pd
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import (filter_tradable_crypto_universe, hot_activity_metrics,
                             ohlcv_frame, score_hot_candidates)
from bot.exchange.binance_usdm import (create_demo_usdm, create_lead_usdm,
                                       resolve_execution_mode)
from bot.exchange.network import configure_system_proxy
from bot.entry_quality import EntrySignalContext, execution_quote
from bot.execution import (DemoExecutionService, EntryPreflightRejected,
                           LeadPortfolioExecutionService)
from bot.indicators import atr, validate_ohlcv
from bot.live.multi_paper import MultiPaperState
from bot.live import mr_scan
from bot.market_risk import (MarketRiskAssessment, MarketRiskState,
                             fetch_market_risk_status)
from bot.mr_candidate import (MidExitState, staged_profit_lock_stop,
                              deviation_entry_signal, deviation_entry_signal_at_ema,
                              mid_band, mid_exit_check, no_progress_update,
                              regime_faded_triggered, structure_invalid_quote_exit,
                              structure_invalid_triggered, structure_reentry_confirmed,
                              trailing_stop)
from bot.portfolio import Position, classify_exit_category
from bot.regime import Regime, classify_regimes
from bot.risk import (OrderIntent, assess_entry, calculate_adaptive_stop,
                      calculate_order_quantity, funding_rate_rejection_reason)
from bot.signals import Action, Side
from bot.strategy_router import MEAN_REVERSION
from bot.time_filter import TimeFilterFlags, evaluate_time_filter
from bot.utils.logging import redact_log_record, redacted_excepthook

HOUR_MS = 3600 * 1000
FOUR_H_MS = 4 * HOUR_MS
DAY_MS = 24 * HOUR_MS
TOP_N = 20
DEV_PCT = 0.01
MID_PERIOD = 20
BAR_KEY = "__mr_bar__"  # reserved key in exit_management_by_symbol for the last processed 1H bar
HOTLIST_CACHE_KEY = "__mr_hotlist_cache__"
HOTLIST_REFRESH_SECONDS = 300
LIVE_TICKER_MAX_AGE_MS = 120_000
DEMO_POSITION_QUOTE_MAX_AGE_MS = 15_000
DEMO_POSITION_SYNC_MAX_AGE_SECONDS = 60
DEMO_STOP_UPDATE_MIN_INTERVAL_SECONDS = 15
DEMO_POSITION_MONITOR_POLL_SECONDS = 5
PROTECTIVE_STOP_AUDIT_INTERVAL_SECONDS = 300
ENTRY_BAR_MAX_DELAY = timedelta(minutes=10)
STRUCTURE_REENTRY_FAILURES = 2
BAN_UNTIL_RE = re.compile(r"banned until\s+(\d+)", re.IGNORECASE)


@dataclass
class IntrabarMarketCache:
    """Reuse closed candles between 15-second live-stop price polls."""

    markets: dict | None = None
    one_hour: dict = field(default_factory=dict)
    current_hour: dict = field(default_factory=dict)
    four_hour: dict = field(default_factory=dict)
    regimes: dict = field(default_factory=dict)
    selected_symbols: set[str] = field(default_factory=set)
    bars_hour_key: int | None = None


@dataclass
class ClosedMarketCache:
    """Cache only completed higher-timeframe data within its UTC period."""

    four_hour: dict = field(default_factory=dict)
    daily_direction: dict = field(default_factory=dict)


def _utc_now():
    return datetime.now(timezone.utc)


def _next_poll_delay(now, poll, *, align_hourly=False, cycle_started_at=None):
    if not align_hourly:
        return float(poll)
    now = now.astimezone(timezone.utc)
    if (cycle_started_at is not None
            and now.replace(minute=0, second=0, microsecond=0)
            > cycle_started_at.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)):
        return 0.0
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return min(float(poll), (next_hour - now).total_seconds())


def _confirm_live(args, parser):
    if not args.live:
        if args.background_confirmed:
            parser.error("--background-confirmed requires --live")
        return False

    if args.background_confirmed:
        confirmation = os.environ.pop("BOT_LEAD_BACKGROUND_CONFIRMATION", None)
        if confirmation != "LEAD":
            parser.error("background live mode requires a confirmed Lead launcher")
        return True

    if not sys.stdin.isatty():
        parser.error("--live requires an interactive console confirmation")
    print("实盘将通过 Binance Lead Portfolio API 发送真实委托。输入 LEAD 才继续：", flush=True)
    try:
        return input().strip() == "LEAD"
    except EOFError:
        return False


class _RunnerCheckpoint:
    """Keep monitoring in memory during storage outages; entries require a save."""

    def __init__(self, state, path):
        self.state, self.path = state, path
        self.failed = False

    def __call__(self):
        recovery = None
        if self.failed:
            recovery = {"event": "state_persistence_recovered", "at": _utc_now().isoformat()}
            self.state.events.append(recovery)
        try:
            self.state.save(self.path)
        except OSError as exc:
            if recovery is not None:
                self.state.events.pop()
            if not self.failed:
                self.state.events.append({
                    "event": "state_persistence_unavailable", "at": _utc_now().isoformat(),
                    "error": type(exc).__name__, "new_entries_paused": True,
                })
                logger.error("MR state persistence unavailable; entries paused: {}", type(exc).__name__)
            self.failed = True
            raise
        if self.failed:
            logger.info("MR state persistence recovered")
        self.failed = False

    def try_save(self):
        try:
            self()
        except OSError:
            return False
        return True


def _wait_with_monitor(seconds, monitor=None):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(DEMO_POSITION_MONITOR_POLL_SECONDS, remaining) if monitor else remaining)
        if monitor is not None:
            monitor()


def _initialize_demo(execution):
    for attempt in range(3):
        try:
            return execution.initialize()
        except ccxt.BaseError as exc:
            if attempt == 2 or not mr_scan.retryable_read(exc):
                raise
            logger.warning("{} startup read failed; retry {}/3: {}",
                           _execution_label(execution), attempt + 1, type(exc).__name__)
            time.sleep(attempt + 1)


def _read_entry_data(read, *, phase, retry=False, deadline=None, monitor=None, symbol=None):
    for attempt in range(2 if retry else 1):
        try:
            return read()
        except ccxt.BaseError as exc:
            exc.mr_phase, exc.mr_endpoint, exc.mr_symbol = phase, "public", symbol
            if attempt or not retry or not mr_scan.retryable_read(exc):
                raise
            if monitor is not None:
                monitor()
            if deadline is not None and _utc_now() >= deadline:
                raise
            logger.warning("MR public read retry phase={} symbol={} error={}",
                           phase, symbol, type(exc).__name__)


def _source_fingerprint(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unavailable"


RUNNER_BUILD_ID = _source_fingerprint(__file__)
TIME_FILTER_BUILD_ID = _source_fingerprint(evaluate_time_filter.__code__.co_filename)


def _execution_source(state):
    return ("binance_usdm_lead" if state.execution_mode == "lead_usdm"
            else "binance_usdm_demo")


def _execution_label(execution, state=None):
    label = getattr(execution, "label", None)
    if label:
        return str(label)
    mode = getattr(state, "execution_mode", None)
    if mode == "lead_usdm":
        return "Lead Portfolio"
    return "local Paper" if mode == "local_paper" else "Demo"


def _target_margin_per_symbol(state, settings):
    if state.execution_mode in {"demo_usdm", "lead_usdm"}:
        return state.current_equity * settings.sizing.live_margin_equity_pct
    return settings.sizing.margin_per_symbol_usdt


def _default_poll_for_mode(mode_name):
    return 60 if mode_name in {"demo_usdm", "lead_usdm"} else 300


def _strategy_variant_for_mode(mode_name, *, intrabar_entries=False):
    if mode_name in {"demo_usdm", "lead_usdm"} or intrabar_entries:
        return "ema50_daily_gate_closed_h1_mean_reversion"
    return "ema50_deviation_candidate"


def _filter_lead_candidates(hotlist, markets, execution):
    allowed_market_ids = getattr(execution, "lead_market_ids", set())
    if not allowed_market_ids:
        raise RuntimeError("Lead Portfolio contract whitelist is unavailable; refusing entries")
    return [
        row for row in hotlist
        if str((markets.get(row["symbol"]) or {}).get("id", "")).upper()
        in allowed_market_ids
    ]


# --- open / close helpers (demo + local, both sides) ---------------------------------

def _open_position(is_demo, execution, state, symbol, side, price, stop, qty, leverage,
                   fee_rate, slip_rate, reason, regime_value, at, *,
                   entry_preflight=None, submission_deadline=None, entry_audit=None,
                   scan_row=None, checkpoint=None, recovery_meta=None):
    if is_demo:
        intent = OrderIntent(symbol, side, Action.OPEN, qty, price, reason,
                             Regime(regime_value), stop, MEAN_REVERSION)
        attempt = None
        if scan_row is not None:
            if checkpoint is None:
                raise mr_scan.ScanCheckpointError(
                    f"{_execution_label(execution, state)} scan requires durable state storage"
                )
            client_id = mr_scan.client_order_id(symbol, side.value, scan_row["bar"])
            attempt = state.mr_entry_attempts.setdefault(client_id, {
                "symbol": symbol, "side": side.value, "bar": scan_row["bar"],
                "clientOrderId": client_id, "status": "prepared", "reason": reason,
                "regime": regime_value, "meta": recovery_meta,
            })
            if attempt["status"] != "prepared":
                raise RuntimeError("entry attempt already submitted; reconciliation required")
            if entry_audit is not None:
                entry_audit["clientOrderId"] = client_id
            mr_scan.checkpoint(checkpoint)

            def on_submit(amount, actual_stop, submitted_at):
                attempt.update(status="submitting", quantity=amount, stop_price=actual_stop,
                               submitted_at=submitted_at.isoformat(),
                               audit=dict(entry_audit or {}))
                scan_row.update(phase="submit_or_protect", order_may_have_been_sent=True)
                mr_scan.attempt_outcome(state, attempt, "submitting", submitted_at,
                                        phase=scan_row["phase"], order_may_have_been_sent=True)
                mr_scan.checkpoint(checkpoint)
                quote_deadline = (entry_audit or {}).get("quote_valid_until")
                if quote_deadline and _utc_now() > datetime.fromisoformat(quote_deadline):
                    raise EntryPreflightRejected("entry_quote_stale")

        try:
            kwargs = ({"before_submit": entry_preflight,
                       "submission_deadline": submission_deadline}
                      if entry_preflight is not None else {})
            if attempt is not None:
                kwargs.update(client_order_id=client_id, on_submit=on_submit)
            result = execution.open(intent, stop_price=stop, **kwargs)
        except EntryPreflightRejected as exc:
            if attempt is not None:
                attempt["status"] = "not_submitted"
                scan_row["order_may_have_been_sent"] = False
            if entry_audit is not None and entry_audit.get("decision") != "BLOCK":
                _record_preflight_decision(state, entry_audit, str(exc), _utc_now())
            if attempt is not None:
                mr_scan.attempt_outcome(state, attempt, "not_submitted", _utc_now(),
                                        phase=scan_row.get("phase"), reason=str(exc),
                                        order_may_have_been_sent=False)
                mr_scan.checkpoint(checkpoint)
            return None
        except Exception as exc:
            if attempt is not None:
                # A failed pre-submit checkpoint prevents execution.open from sending.
                uncertain = (attempt["status"] == "submitting"
                             and not isinstance(exc, mr_scan.ScanCheckpointError))
                if uncertain:
                    attempt["status"] = "uncertain"
                    state.execution_status = "entry_reconciliation_required"
                else:
                    attempt["status"] = "prepared"
                    scan_row["order_may_have_been_sent"] = False
                mr_scan.attempt_outcome(
                    state, attempt, "uncertain" if uncertain else "not_submitted", _utc_now(),
                    phase=scan_row.get("phase"), error=type(exc).__name__,
                    order_may_have_been_sent=uncertain,
                )
                mr_scan.checkpoint(checkpoint)
            raise
        if attempt is not None:
            attempt.update(status="protected", result=result)
            mr_scan.checkpoint(checkpoint)
        return _record_demo_open(state, symbol, side, reason, regime_value, result, entry_audit)

    return _open_local_position(state, symbol, side, price, stop, qty, leverage,
                                fee_rate, slip_rate, reason, regime_value, at)


def _record_demo_open(state, symbol, side, reason, regime_value, result, entry_audit=None):
    state.exchange_orders_by_symbol[symbol] = {"stop": result["stop"]}
    fill = result["fill"]
    pos = result["position"]
    opened_at = _fill_timestamp(fill, _utc_now())
    position = Position(symbol, side, float(pos["quantity"]), float(pos["entry_price"]),
                        float(result["stop"]["stop_price"]), opened_at, reason, regime_value, MEAN_REVERSION)
    state.portfolio.positions[symbol] = position
    if fill.get("fee_cost") is not None:
        state.entry_costs[symbol] = {"entry_fee": float(fill["fee_cost"])}
    _invalidate_demo_stop_audit(state)
    state.events.append({"event": "paper_fill", "action": "OPEN", "symbol": symbol,
                         "side": side.value, "at": opened_at.isoformat(), "quantity": position.quantity,
                         "price": position.entry_price, "reason": reason, "source": MEAN_REVERSION,
                         "fee": fill.get("fee_cost"), "orderId": fill.get("id"),
                         "clientOrderId": (fill.get("clientOrderId")
                                           or (entry_audit or {}).get("clientOrderId")),
                         "execution_source": _execution_source(state)})
    if entry_audit is not None:
        _record_entry_execution_fill(state, position, result, entry_audit)
    return position


def _open_local_position(state, symbol, side, price, stop, qty, leverage,
                         fee_rate, slip_rate, reason, regime_value, at):
    entry_fill = price * (1 + slip_rate) if side is Side.LONG else price * (1 - slip_rate)
    margin = qty * entry_fill / leverage
    entry_fee = qty * entry_fill * fee_rate
    if state.cash < margin + entry_fee:
        return None
    state.cash -= margin + entry_fee
    state.isolated_margin_by_symbol[symbol] = {"initial_margin": margin, "leverage": leverage}
    state.entry_costs[symbol] = {"entry_fee": entry_fee}
    position = Position(symbol, side, qty, entry_fill, stop, at, reason, regime_value, MEAN_REVERSION)
    state.portfolio.positions[symbol] = position
    state.mark_prices[symbol] = entry_fill
    state.events.append({"event": "paper_fill", "action": "OPEN", "symbol": symbol,
                         "side": side.value, "at": at.isoformat(), "quantity": qty,
                         "price": entry_fill, "reason": reason, "source": MEAN_REVERSION})
    return position


def _fill_timestamp(fill, fallback):
    return _verified_fill_timestamp(fill) or fallback


def _entry_order_limits(market, price):
    step = float((market.get("precision") or {}).get("amount") or 1e-8) or 1e-8
    limits = market.get("limits") or {}
    minimum = float((limits.get("amount") or {}).get("min") or 0.0)
    minimum_cost = float((limits.get("cost") or {}).get("min") or 0.0)
    return step, max(minimum, minimum_cost / price)


def _record_preflight_decision(state, audit, reason, checked_at):
    audit.update({"event": "entry_preflight_decision", "at": checked_at.isoformat(),
                  "checked_at": checked_at.isoformat(), "source": MEAN_REVERSION,
                  "decision": "BLOCK" if reason else "ALLOW", "reason": reason})
    state.events.append(dict(audit))
    if reason:
        _set_symbol_status(state, audit["symbol"], action="BLOCK", side=Side(audit["side"]),
                           reason=reason, at=checked_at)
    logger.info("MR entry preflight {}", json.dumps(audit, ensure_ascii=False, sort_keys=True))


def _entry_preflight(exchange, execution, state, settings, market, context, flags,
                     funding_rate, market_risk, audit, *, propagate_read_errors=False,
                     scan_row=None):
    """Build a guard run by the executor after account/contract preparation."""
    deadline = context.bar_close_at + timedelta(seconds=settings.execution.confirm_delay_sec_max)
    audit.update({
        "symbol": context.symbol, "side": context.side.value,
        "bar_close_at": context.bar_close_at.isoformat(),
        "cycle_started_at": context.cycle_started_at.isoformat(),
        "signal_evaluated_at": context.evaluated_at.isoformat(),
        "submission_deadline": deadline.isoformat(),
        "signal_open": context.signal_open, "signal_close": context.signal_close,
        "closed_4h_ema": context.closed_ema, "signal_atr": context.atr,
        "public_quote": None, "demo_quote": None,
    })
    multiplier = (settings.market_risk.elevated_size_multiplier
                  if market_risk.state is MarketRiskState.RISK_ELEVATED else 1.0)
    gross_cap = settings.sizing.max_gross_notional_pct * multiplier

    def reject(reason):
        _record_preflight_decision(state, audit, reason, _utc_now())
        raise EntryPreflightRejected(reason)

    def quote(client, *, required=False):
        try:
            book = client.fetch_order_book(context.symbol, limit=5)
        except ccxt.BaseError as exc:
            if required and propagate_read_errors:
                raise
            return None, f"entry_quote_unavailable:{type(exc).__name__}"
        return execution_quote(book, context.side, _utc_now(),
                               max_age_sec=settings.execution.entry_quote_max_age_sec)

    def before_submit(intent, planned_stop):
        if _utc_now() > deadline:
            reject("entry_signal_expired")
        # Public quote is optional attribution evidence. Fetch Demo last so the
        # executable reference does not wait behind an additional network call.
        audit["public_quote"], public_error = quote(exchange)
        audit["public_quote_error"] = public_error
        if _utc_now() > deadline:
            reject("entry_signal_expired")
        if scan_row is not None:
            scan_row.update(phase="entry_quote", endpoint="demo")
        demo_quote, quote_error = quote(execution.exchange, required=True)
        audit["demo_quote"] = demo_quote
        if quote_error:
            reject(quote_error)
        checked_at = _utc_now()
        if checked_at > deadline:
            reject("entry_signal_expired")
        # Recheck age after all quote acquisition, including a delayed response.
        quote_age = (checked_at - datetime.fromisoformat(demo_quote["quoted_at"])).total_seconds()
        if quote_age < 0 or quote_age > settings.execution.entry_quote_max_age_sec:
            reject("entry_quote_stale")
        audit["quote_valid_until"] = (
            datetime.fromisoformat(demo_quote["quoted_at"])
            + timedelta(seconds=settings.execution.entry_quote_max_age_sec)
        ).isoformat()
        price = demo_quote["price"]
        direction = 1 if context.side is Side.LONG else -1
        adverse_atr = direction * (price - context.signal_close) / context.atr
        audit.update({
            "adverse_quote_atr": adverse_atr,
            "observe_only": True,
            "atr_threshold_observations": [
                {"threshold_atr": threshold, "would_block": adverse_atr > threshold}
                for threshold in settings.execution.entry_adverse_atr_observe
            ],
        })
        if structure_invalid_triggered(
                context.side, price, context.structure_low, context.structure_high,
                context.atr, buffer_atr=settings.exits.structure_invalid_buffer_atr):
            reject("entry_structure_invalid")
        if ((context.side is Side.LONG and price <= planned_stop)
                or (context.side is Side.SHORT and price >= planned_stop)):
            reject("entry_price_beyond_planned_stop")
        signal = deviation_entry_signal_at_ema(
            context.closed_ema, context.signal_open, price, settings, flags,
            deviation_pct=DEV_PCT,
        )
        if signal.action is not Action.OPEN or signal.side is not context.side:
            reject("entry_signal_no_longer_valid")
        structure = context.structure_low if context.side is Side.LONG else context.structure_high
        try:
            stop, _ = calculate_adaptive_stop(context.side, price, context.atr, structure, settings)
            stop = float(execution.exchange.price_to_precision(context.symbol, stop))
        except (TypeError, ValueError, OverflowError):
            reject("entry_stop_invalid")
        if (not math.isfinite(stop) or stop <= 0
                or (context.side is Side.LONG and stop >= price)
                or (context.side is Side.SHORT and stop <= price)):
            reject("entry_stop_invalid")
        step, minimum = _entry_order_limits(market, price)
        target_margin = _target_margin_per_symbol(state, settings)
        qty = calculate_order_quantity(
            state.current_equity, price, state.portfolio, settings,
            step_size=step, minimum_quantity=minimum, size_multiplier=multiplier,
            stop_price=stop, gross_notional_cap_pct=gross_cap,
            target_margin_usdt=target_margin,
        )
        if qty <= 0:
            reject("entry_quantity_unavailable")
        if state.execution_mode == "lead_usdm":
            available = (state.exchange_snapshot or {}).get("available_usdt")
            required_margin = (qty * price / settings.sizing.leverage
                               * (1 + settings.fees.taker_bps / 10_000
                                  + settings.risk.max_slippage_pct))
            if (available is None or not math.isfinite(float(available))
                    or float(available) < required_margin):
                reject("insufficient_or_unavailable_margin")
        refreshed = replace(intent, price=price, quantity=qty, stop_price=stop)
        approval = assess_entry(
            refreshed, state.portfolio, state.current_equity, funding_rate, checked_at, settings,
            gross_notional_cap_pct=gross_cap, market_risk_state=market_risk.state.value,
            per_symbol_margin_usdt=target_margin,
        )
        if not approval.allowed:
            reject(approval.reason)
        audit.update({"planned_price": price, "planned_quantity": qty,
                      "planned_stop": stop,
                      "top_quote_covers_quantity": demo_quote["quantity"] >= qty})
        _record_preflight_decision(state, audit, None, _utc_now())
        return refreshed, stop

    return before_submit, deadline


def _record_entry_execution_fill(state, position, result, audit):
    direction = 1 if position.side is Side.LONG else -1
    atr_value = audit["signal_atr"]
    demo_price = audit["demo_quote"]["price"]
    public = audit.get("public_quote")
    event = {
        **audit, "event": "entry_execution_quality", "at": position.opened_at.isoformat(),
        "fill_at": position.opened_at.isoformat(), "fill_price": position.entry_price,
        "fill_time_verified": _verified_fill_timestamp(result["fill"]) is not None,
        "orderId": result["fill"].get("id"),
        "clientOrderId": result["fill"].get("clientOrderId") or audit.get("clientOrderId"),
        "quantity": position.quantity,
        "order_submitted_at": result.get("order_submitted_at"),
        "signal_to_fill_atr": direction * (position.entry_price - audit["signal_close"]) / atr_value,
        "demo_quote_to_fill_atr": direction * (position.entry_price - demo_price) / atr_value,
        "signal_to_public_quote_atr": (direction * (public["price"] - audit["signal_close"]) / atr_value
                                       if public else None),
        "public_to_demo_quote_atr": (direction * (demo_price - public["price"]) / atr_value
                                     if public else None),
        "quote_time_gap_sec": (abs((datetime.fromisoformat(audit["demo_quote"]["quoted_at"])
                                     - datetime.fromisoformat(public["quoted_at"])).total_seconds())
                               if public else None),
    }
    state.events.append(event)


def _finite_pnl_value(value):
    """Keep unavailable PnL inputs and arithmetic results explicitly unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _is_structure_invalidation_loss(reason, net_pnl):
    if reason not in {"long_structure_invalid", "short_structure_invalid"}:
        return False
    value = _finite_pnl_value(net_pnl)
    return value is None or value <= 0


def _mr_exit_category(reason, net_pnl):
    if _is_structure_invalidation_loss(reason, net_pnl):
        return "stop_loss"
    return classify_exit_category(reason, net_pnl)


def _structure_reentry_state(events, symbol, side):
    """Return consecutive same-side invalidation losses and whether rearming is pending."""
    return _structure_reentry_states(events, {symbol}).get(
        (symbol, Side(side)), (0, None, False),
    )


def _structure_reentry_states(events, symbols):
    """Derive each symbol/side re-entry gate in one pass over persisted events."""
    relevant = set(symbols)
    failures = {(symbol, side): 0
                for symbol in relevant for side in (Side.LONG, Side.SHORT)}
    latest_loss_at = {(symbol, side): None
                      for symbol in relevant for side in (Side.LONG, Side.SHORT)}
    confirmations = set()
    for event in events:
        symbol = event.get("symbol")
        if symbol not in relevant or event.get("source", MEAN_REVERSION) != MEAN_REVERSION:
            continue
        if event.get("event") == "structure_reentry_confirmed":
            confirmations.add((
                symbol, event.get("side"), event.get("after_structure_loss_at"),
            ))
            continue
        if event.get("event") != "paper_fill" or event.get("action") != "CLOSE":
            continue
        for side in (Side.LONG, Side.SHORT):
            key = (symbol, side)
            reason = event.get("reason")
            if (event.get("side") == side.value
                    and reason == f"{side.value}_structure_invalid"
                    and _is_structure_invalidation_loss(reason, event.get("net_pnl"))):
                failures[key] += 1
                latest_loss_at[key] = event.get("at")
            else:
                failures[key] = 0
                latest_loss_at[key] = None
    return {
        key: (failures[key], latest_loss_at[key],
              failures[key] >= STRUCTURE_REENTRY_FAILURES
              and (key[0], key[1].value, latest_loss_at[key]) not in confirmations)
        for key in failures
    }


def _record_structure_reentry_confirmations(state, symbol, frame1, now, reentry_states):
    if len(frame1) < 2:
        return
    for side in (Side.LONG, Side.SHORT):
        count, latest_loss_at, waiting = reentry_states.get(
            (symbol, side), (0, None, False),
        )
        if not waiting:
            continue
        try:
            loss_at = datetime.fromisoformat(latest_loss_at)
        except (TypeError, ValueError):
            continue
        confirmation = None
        for index in range(1, len(frame1)):
            if frame1.index[index] - frame1.index[index - 1] != timedelta(hours=1):
                continue
            candle_close_at = (frame1.index[index] + timedelta(hours=1)).to_pydatetime()
            if candle_close_at <= loss_at or candle_close_at > now:
                continue
            previous = frame1.iloc[index - 1]
            current = frame1.iloc[index]
            if structure_reentry_confirmed(
                    side, previous["high"], previous["low"], current["close"]):
                confirmation = (candle_close_at, previous, current)
                break
        if confirmation is None:
            continue
        confirmed_at, previous, current = confirmation
        event = {
            "event": "structure_reentry_confirmed",
            "at": now.isoformat(),
            "source": MEAN_REVERSION,
            "symbol": symbol,
            "side": side.value,
            "bar_close_at": confirmed_at.isoformat(),
            "confirmation_source": "closed_1h_ohlcv",
            "after_structure_loss_at": latest_loss_at,
            "consecutive_structure_losses": count,
            "previous_high": float(previous["high"]),
            "previous_low": float(previous["low"]),
            "close": float(current["close"]),
        }
        state.events.append(event)
        reentry_states[(symbol, side)] = (count, latest_loss_at, False)
        _set_symbol_status(state, symbol, action="REENTRY_READY", side=side,
                           reason="structure_reentry_confirmed", at=now)
        logger.info("MR structure re-entry confirmed {} side={} bar_close_at={} consecutive_losses={}",
                    symbol, side.value, confirmed_at.isoformat(), count)


def _record_structure_reentry_block(state, symbol, side, bar_close_time, now,
                                    failure_count, latest_loss_at):
    if any(
        event.get("event") == "entry_signal_blocked"
        and event.get("reason") == "structure_reentry_confirmation_required"
        and event.get("symbol") == symbol
        and event.get("side") == side.value
        and event.get("bar_close_at") == bar_close_time.isoformat()
        and event.get("after_structure_loss_at") == latest_loss_at
        for event in state.events
    ):
        return
    _set_symbol_status(state, symbol, action="BLOCKED", side=side,
                       reason="structure_reentry_confirmation_required", at=now)
    state.events.append({
        "event": "entry_signal_blocked",
        "at": now.isoformat(),
        "source": MEAN_REVERSION,
        "symbol": symbol,
        "side": side.value,
        "reason": "structure_reentry_confirmation_required",
        "bar_close_at": bar_close_time.isoformat(),
        "after_structure_loss_at": latest_loss_at,
        "consecutive_structure_losses": failure_count,
    })
    logger.info("MR entry candidate held for structure confirmation {} side={} bar_close_at={} consecutive_losses={}",
                symbol, side.value, bar_close_time.isoformat(), failure_count)


def _reconcile_structure_invalidation_cooldowns(state):
    """Apply the six-hour cooldown to legacy exit classifications in the 8766 ledger."""
    latest_by_symbol = {}
    symbols = set(state.portfolio.last_exit_at)
    for event in reversed(state.events):
        symbol = event.get("symbol")
        if (symbol not in symbols or symbol in latest_by_symbol
                or event.get("event") != "paper_fill" or event.get("action") != "CLOSE"):
            continue
        latest_by_symbol[symbol] = event
    for symbol, event in latest_by_symbol.items():
        if event.get("source", MEAN_REVERSION) != MEAN_REVERSION:
            continue
        if not _is_structure_invalidation_loss(event.get("reason"), event.get("net_pnl")):
            continue
        try:
            event_at = datetime.fromisoformat(event["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if event_at != state.portfolio.last_exit_at.get(symbol):
            continue
        if state.portfolio.last_exit_category.get(symbol) == "stop_loss":
            continue
        previous_category = state.portfolio.last_exit_category.get(symbol)
        state.portfolio.last_exit_category[symbol] = "stop_loss"
        state.events.append({
            "event": "entry_cooldown_policy_updated",
            "at": datetime.now(timezone.utc).isoformat(),
            "source": MEAN_REVERSION,
            "symbol": symbol,
            "reason": event.get("reason"),
            "net_pnl": _finite_pnl_value(event.get("net_pnl")),
            "previous_category": previous_category,
            "category": "stop_loss",
            "cooldown_hours": 6,
        })


def _verified_fill_timestamp(fill):
    raw = fill.get("closed_at") if isinstance(fill, dict) else None
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo is not None else None


def _aware_datetime(value):
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _demo_trade_observations(state, position, *, closed_at, reconciled_at=None):
    meta = state.exit_management_by_symbol.get(position.symbol) or {}
    entry = float(position.entry_price)
    initial_stop = float(meta.get("telemetry_initial_stop_price", position.stop_price))
    initial_risk = abs(entry - initial_stop)
    best = meta.get("telemetry_best_price")
    worst = meta.get("telemetry_worst_price")
    through = _aware_datetime(meta.get("telemetry_excursion_through"))
    close_time = _aware_datetime(closed_at)
    opened_at = _aware_datetime(position.opened_at)
    history_complete = bool(meta.get("telemetry_history_complete"))
    bar_count = int(meta.get("telemetry_bar_count", 0) or 0)
    observed_mfe_r = observed_mae_r = None
    if initial_risk > 0 and best is not None and worst is not None:
        best = float(best)
        worst = float(worst)
        if position.side is Side.LONG:
            observed_mfe_r = max(0.0, best - entry) / initial_risk
            observed_mae_r = max(0.0, entry - worst) / initial_risk
        else:
            observed_mfe_r = max(0.0, entry - best) / initial_risk
            observed_mae_r = max(0.0, worst - entry) / initial_risk
    excursion_available = bool(
        close_time is not None and through is not None and through <= close_time
        and history_complete
        and (bar_count + int(meta.get("telemetry_sample_count", 0) or 0)) > 0
        and observed_mfe_r is not None
    )
    held_seconds = ((close_time - opened_at).total_seconds()
                    if close_time is not None and opened_at is not None else None)
    held_hours = held_seconds / 3600 if held_seconds is not None and held_seconds >= 0 else None
    reconciled_time = _aware_datetime(reconciled_at)
    observed_before_close = bool(close_time is not None and through is not None and through <= close_time)
    observed_to_reconciliation = bool(close_time is None and reconciled_time is not None)
    observations = {
        "opened_at": position.opened_at.isoformat() if position.opened_at else None,
        "closed_at": close_time.isoformat() if close_time else None,
        "exit_time_verified": close_time is not None,
        "held_hours": held_hours,
        "initial_stop_price": initial_stop,
        "initial_risk": initial_risk if initial_risk > 0 else None,
        "mfe_price": float(best) if observed_before_close and best is not None else None,
        "mae_price": float(worst) if observed_before_close and worst is not None else None,
        "mfe_r": observed_mfe_r if excursion_available else None,
        "mae_r": observed_mae_r if excursion_available else None,
        "mfe_available": excursion_available,
        "mae_available": excursion_available,
        "observed_mfe_r": observed_mfe_r if observed_before_close else None,
        "observed_mae_r": observed_mae_r if observed_before_close else None,
        "mfe_to_reconciliation_r": observed_mfe_r if observed_to_reconciliation else None,
        "mae_to_reconciliation_r": observed_mae_r if observed_to_reconciliation else None,
        "mfe_to_reconciliation_price": float(best) if observed_to_reconciliation and best is not None else None,
        "mae_to_reconciliation_price": float(worst) if observed_to_reconciliation and worst is not None else None,
        "excursion_source": "fresh_ticker_samples_and_closed_1h_ohlc",
        "excursion_through": through.isoformat() if through else None,
        "excursion_history_complete": history_complete,
        "excursion_bar_count": bar_count,
        "excursion_sample_count": int(meta.get("telemetry_sample_count", 0) or 0),
        "reconciled_at": reconciled_time.isoformat() if reconciled_time else None,
        "elapsed_to_reconciliation_hours": (
            max(0.0, (reconciled_time - opened_at).total_seconds() / 3600)
            if reconciled_time is not None and opened_at is not None else None
        ),
    }
    return observations


def _update_demo_excursions(meta, position, *, frame=None, price=None, observed_at=None):
    entry = float(position.entry_price)
    opened_at = _aware_datetime(position.opened_at)
    if not meta.get("telemetry_enabled"):
        meta.update({
            "telemetry_enabled": True,
            "telemetry_initial_stop_price": float(position.stop_price),
            "telemetry_best_price": entry,
            "telemetry_worst_price": entry,
            "telemetry_history_complete": False,
            "telemetry_bar_count": 0,
            "telemetry_sample_count": 0,
        })
    else:
        meta.setdefault("telemetry_initial_stop_price", float(position.stop_price))
        meta.setdefault("telemetry_best_price", entry)
        meta.setdefault("telemetry_worst_price", entry)
        meta.setdefault("telemetry_history_complete", True)
        meta.setdefault("telemetry_bar_count", 0)
        meta.setdefault("telemetry_sample_count", 0)

    if price is not None and observed_at is not None:
        price = float(price)
        if math.isfinite(price) and price > 0:
            if position.side is Side.LONG:
                meta["telemetry_best_price"] = max(float(meta["telemetry_best_price"]), price)
                meta["telemetry_worst_price"] = min(float(meta["telemetry_worst_price"]), price)
            else:
                meta["telemetry_best_price"] = min(float(meta["telemetry_best_price"]), price)
                meta["telemetry_worst_price"] = max(float(meta["telemetry_worst_price"]), price)
            sampled_at = _aware_datetime(observed_at)
            if sampled_at is not None:
                previous_sample = _aware_datetime(meta.get("telemetry_last_sample_at"))
                if previous_sample and sampled_at - previous_sample > timedelta(minutes=5):
                    meta["telemetry_history_complete"] = False
                meta["telemetry_last_sample_at"] = sampled_at.isoformat()
                meta["telemetry_sample_count"] = int(meta.get("telemetry_sample_count", 0) or 0) + 1
                previous_through = _aware_datetime(meta.get("telemetry_excursion_through"))
                if previous_through is None or sampled_at > previous_through:
                    meta["telemetry_excursion_through"] = sampled_at.isoformat()

    if frame is None:
        return
    bar_open = _aware_datetime(frame.index[-1].to_pydatetime())
    if opened_at is None or bar_open is None or bar_open < opened_at:
        return
    previous_bar = _aware_datetime(meta.get("telemetry_last_bar_open"))
    if previous_bar is not None and bar_open <= previous_bar:
        return
    if previous_bar is not None and bar_open - previous_bar > timedelta(hours=1):
        meta["telemetry_history_complete"] = False
    elif previous_bar is None and bar_open - opened_at > timedelta(hours=1):
        meta["telemetry_history_complete"] = False

    high = float(frame["high"].iloc[-1])
    low = float(frame["low"].iloc[-1])
    if position.side is Side.LONG:
        meta["telemetry_best_price"] = max(float(meta["telemetry_best_price"]), high)
        meta["telemetry_worst_price"] = min(float(meta["telemetry_worst_price"]), low)
    else:
        meta["telemetry_best_price"] = min(float(meta["telemetry_best_price"]), low)
        meta["telemetry_worst_price"] = max(float(meta["telemetry_worst_price"]), high)
    meta["telemetry_last_bar_open"] = bar_open.isoformat()
    bar_through = bar_open + timedelta(hours=1)
    previous_through = _aware_datetime(meta.get("telemetry_excursion_through"))
    if previous_through is None or bar_through > previous_through:
        meta["telemetry_excursion_through"] = bar_through.isoformat()
    meta["telemetry_bar_count"] = int(meta["telemetry_bar_count"]) + 1


def _demo_close_values(execution, state, position, fill, closed_at):
    entry_fee = (state.entry_costs.get(position.symbol, {}).get("entry_fee")
                 if position.symbol in state.entry_costs else None)
    entry_fee_getter = getattr(execution, "resolve_entry_fee", None)
    if entry_fee is None and callable(entry_fee_getter):
        entry_fee = entry_fee_getter(position.symbol, position.side, position.quantity,
                                     position.entry_price, position.opened_at)
    entry_fee = _finite_pnl_value(entry_fee)
    exit_fee = _finite_pnl_value(fill.get("exit_fee", fill.get("fee_cost")))
    gross = fill.get("gross_pnl", fill.get("realized_pnl"))
    if gross is None:
        quantity = _finite_pnl_value(fill.get("quantity", fill.get("filled", position.quantity)))
        average = _finite_pnl_value(fill.get("average"))
        if average is not None and quantity is not None:
            direction = 1 if position.side is Side.LONG else -1
            gross = quantity * (average - position.entry_price) * direction
    gross = _finite_pnl_value(gross)
    funding_getter = getattr(execution, "funding_fee", None)
    funding = (funding_getter(position.symbol, position.opened_at, closed_at)
               if callable(funding_getter) else None)
    funding = _finite_pnl_value(funding)
    fees = (_finite_pnl_value(entry_fee + exit_fee)
            if entry_fee is not None and exit_fee is not None else None)
    net = (_finite_pnl_value(gross - fees + funding)
           if gross is not None and fees is not None and funding is not None else None)
    return gross, fees, funding, net


def _reconcile_demo_funding(execution, state, now, *, position_monitor=None):
    """Recheck late-posted funding without changing fills or account balances."""
    fetch = getattr(execution, "funding_history", None)
    if not callable(fetch):
        return False
    previous = next((e for e in reversed(state.events)
                     if e.get("event") == "funding_reconciliation_checked"), {})
    checked_at = _aware_datetime(previous.get("at"))
    if checked_at is not None and (now - checked_at).total_seconds() < 300:
        return False
    active, trades = {}, []
    for event in state.events:
        if (event.get("event") != "paper_fill"
                or event.get("execution_source") != getattr(execution, "source", "binance_usdm_demo")):
            continue
        symbol = event["symbol"]
        if event.get("action") == "OPEN":
            active[symbol] = event
            continue
        if event.get("action") != "CLOSE":
            continue
        opened = active.pop(symbol, {})
        start = _aware_datetime(event.get("opened_at") or opened.get("at"))
        end = _aware_datetime(event.get("closed_at") or event.get("at"))
        gross, fees = _finite_pnl_value(event.get("gross_pnl")), _finite_pnl_value(event.get("fees"))
        if start is None or end is None or end < start or gross is None or fees is None:
            continue
        age = (now - end).total_seconds()
        if age < 300 or (age > 86400 and event.get("funding_rechecked_at")):
            continue
        try:
            market_id = execution.exchange.market(symbol)["id"]
        except (ccxt.BaseError, KeyError):
            continue
        trades.append({"event": event, "symbol": market_id, "start": start, "end": end,
                       "gross": gross, "fees": fees, "income": []})
    if not trades:
        return False
    cursor, through = min(t["start"] for t in trades), max(t["end"] for t in trades)
    rows = []
    while cursor <= through:
        if position_monitor is not None:
            position_monitor()
        until = min(through, cursor + timedelta(days=5) - timedelta(milliseconds=1))
        batch = fetch(cursor, until)
        if batch is None:
            state.events.append({"event": "funding_reconciliation_checked",
                                 "at": now.isoformat(), "status": "unavailable"})
            return True
        rows.extend(batch)
        cursor = until + timedelta(milliseconds=1)
    ambiguous = set()
    for row in rows:
        owners = [i for i, trade in enumerate(trades) if row["symbol"] == trade["symbol"]
                  and int(trade["start"].timestamp() * 1000) <= row["time"]
                  <= int(trade["end"].timestamp() * 1000)]
        if len(owners) > 1:
            ambiguous.update(owners)
        elif owners:
            trades[owners[0]]["income"].append(row["income"])
    adjusted = 0
    for i, trade in enumerate(trades):
        if i in ambiguous:
            continue
        event = trade["event"]
        income = math.fsum(trade["income"])
        net = _finite_pnl_value(trade["gross"] - trade["fees"] + income)
        if net is None:
            continue
        old_funding, old_net = event.get("funding_paid"), event.get("net_pnl")
        event["funding_rechecked_at"] = now.isoformat()
        funding_value, net_value = _finite_pnl_value(old_funding), _finite_pnl_value(old_net)
        if (funding_value is not None and net_value is not None
                and abs(funding_value - income) <= 1e-8 and abs(net_value - net) <= 1e-8):
            continue
        event.update(funding_paid=income, net_pnl=net, pnl_recorded=True,
                     funding_source="binance_income_recheck")
        state.events.append({
            "event": "funding_reconciliation_adjusted", "at": now.isoformat(),
            "symbol": event["symbol"], "opened_at": trade["start"].isoformat(),
            "closed_at": trade["end"].isoformat(), "previous_funding": funding_value,
            "funding_income": income, "previous_net_pnl": net_value, "net_pnl": net,
        })
        adjusted += 1
    state.events.append({"event": "funding_reconciliation_checked", "at": now.isoformat(),
                         "status": "ambiguous" if ambiguous else "completed",
                         "checked_trades": len(trades) - len(ambiguous), "adjusted_trades": adjusted})
    return True


def _close_position(is_demo, execution, state, symbol, price, reason, at, fee_rate, slip_rate):
    position = state.portfolio.positions.get(symbol)
    if position is None:
        return None
    if is_demo:
        stop = state.exchange_orders_by_symbol.get(symbol, {}).get("stop")
        result = execution.close(position, stop=stop, reason=reason)
        if result.get("already_flat"):
            _demo_sync_stopped(execution, state, at)
            return position
        fill = result["fill"]
        verified_closed_at = _verified_fill_timestamp(fill)
        closed_at = verified_closed_at or at
        gross, fees, funding, net = _demo_close_values(execution, state, position, fill, closed_at)
        observations = _demo_trade_observations(
            state, position, closed_at=verified_closed_at,
        )
        state.portfolio.positions.pop(symbol, None)
        state.exchange_orders_by_symbol.pop(symbol, None)
        state.exit_management_by_symbol.pop(symbol, None)
        state.entry_costs.pop(symbol, None)
        _invalidate_demo_stop_audit(state)
        close_event = {"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                       "side": position.side.value, "at": closed_at.isoformat(), "reason": reason,
                       "quantity": fill.get("filled", position.quantity), "price": fill.get("average"),
                       "gross_pnl": gross, "funding_paid": funding, "net_pnl": net,
                       "pnl_recorded": net is not None, "source": MEAN_REVERSION,
                       "execution_source": _execution_source(state), "orderId": fill.get("id"),
                       **observations}
        if fees is not None:
            close_event["fees"] = fees
        state.events.append(close_event)
        state.portfolio.last_exit_at[symbol] = closed_at
        state.portfolio.last_exit_source[symbol] = MEAN_REVERSION
        state.portfolio.last_exit_category[symbol] = _mr_exit_category(reason, net)
        return position

    sign = 1 if position.side is Side.LONG else -1
    exit_fill = price * (1 - slip_rate) if position.side is Side.LONG else price * (1 + slip_rate)
    gross = position.quantity * (exit_fill - position.entry_price) * sign
    exit_fee = position.quantity * exit_fill * fee_rate
    entry_fee = state.entry_costs.get(symbol, {}).get("entry_fee", 0.0)
    net = gross - exit_fee - entry_fee
    margin = float(state.isolated_margin_by_symbol.pop(symbol, {}).get("initial_margin", 0.0))
    state.cash += margin + net
    state.portfolio.positions.pop(symbol, None)
    state.entry_costs.pop(symbol, None)
    state.exit_management_by_symbol.pop(symbol, None)
    state.mark_prices.pop(symbol, None)
    state.events.append({"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                         "side": position.side.value, "at": at.isoformat(),
                         "quantity": position.quantity, "price": exit_fill, "reason": reason,
                         "net_pnl": net, "source": MEAN_REVERSION})
    state.portfolio.last_exit_at[symbol] = at
    state.portfolio.last_exit_source[symbol] = MEAN_REVERSION
    state.portfolio.last_exit_category[symbol] = classify_exit_category(reason, net)
    return net


def _demo_sync_stopped(execution, state, at):
    snapshot = execution.sync_snapshot()
    state.exchange_snapshot = snapshot
    live = set(snapshot["positions"])
    for symbol in list(state.portfolio.positions):
        if symbol not in live:
            position = state.portfolio.positions[symbol]
            order_data = state.exchange_orders_by_symbol.get(symbol, {})
            stop = order_data.get("stop") if isinstance(order_data, dict) else None
            stop_fill = None
            resolver = getattr(execution, "resolve_protective_stop_algo_fill", None)
            if callable(resolver):
                try:
                    stop_fill = resolver(symbol, stop, position.quantity)
                except Exception as exc:
                    logger.warning("Cannot verify {} stop fill for {}: {}",
                                   _execution_label(execution, state), symbol, type(exc).__name__)
            try:
                cleanup = getattr(execution, "cancel_saved_stop_if_open", None)
                if callable(cleanup):
                    cleanup(symbol, stop)
            except Exception as exc:
                mismatch = f"cannot verify or cancel stale stop for {symbol}"
                state.execution_status = "inconsistent"
                state.inconsistencies = [mismatch]
                state.protective_stop_audit = {
                    "at": at.isoformat(), "status": "error",
                    "position_count": len(state.portfolio.positions),
                    "mismatch_count": 1, "mismatches": [mismatch],
                }
                logger.error("{} position {} is flat but stop cleanup failed: {}",
                             _execution_label(execution, state), symbol, type(exc).__name__)
                return False

            verified_closed_at = _verified_fill_timestamp(stop_fill) if stop_fill else None
            closed_at = verified_closed_at or at
            gross = fees = funding = net = None
            if stop_fill:
                gross, fees, funding, net = _demo_close_values(
                    execution, state, position, stop_fill, closed_at,
                )
            observations = _demo_trade_observations(
                state, position, closed_at=verified_closed_at,
                reconciled_at=at if verified_closed_at is None else None,
            )
            state.portfolio.positions.pop(symbol, None)
            state.exchange_orders_by_symbol.pop(symbol, None)
            state.exit_management_by_symbol.pop(symbol, None)
            state.entry_costs.pop(symbol, None)
            reason = "protective_stop_exchange_algo" if stop_fill else "exchange_flat_unverified"
            state.portfolio.last_exit_at[symbol] = closed_at
            state.portfolio.last_exit_category[symbol] = _mr_exit_category(reason, net)
            state.portfolio.last_exit_source[symbol] = MEAN_REVERSION
            _invalidate_demo_stop_audit(state)
            close_event = {"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                           "side": position.side.value, "at": closed_at.isoformat(), "reason": reason,
                           "quantity": stop_fill.get("quantity") if stop_fill else position.quantity,
                           "price": stop_fill.get("average") if stop_fill else None,
                           "gross_pnl": gross, "funding_paid": funding, "net_pnl": net,
                           "pnl_recorded": net is not None, "source": MEAN_REVERSION,
                           "execution_source": _execution_source(state),
                           "orderId": stop_fill.get("order_id") if stop_fill else None,
                           "algoId": stop_fill.get("algo_id") if stop_fill else None,
                           **observations}
            if fees is not None:
                close_event["fees"] = fees
            state.events.append(close_event)
    local = set(state.portfolio.positions)
    pending_symbols = {row["symbol"] for row in mr_scan.unresolved_attempts(state)}
    unknown = (live - local) - pending_symbols
    if unknown and getattr(execution, "adopt_manual_positions", False):
        if state.execution_mode != "lead_usdm" or pending_symbols:
            reason = "manual position adoption requires a synchronized Lead account without pending entries"
            state.execution_status = "inconsistent"
            state.inconsistencies = [reason]
            state.protective_stop_audit = {
                "at": at.isoformat(), "status": "error",
                "position_count": len(state.portfolio.positions),
                "mismatch_count": 1, "mismatches": [reason],
            }
            return False
        for symbol in sorted(unknown):
            try:
                _adopt_lead_manual_position(execution, state, symbol,
                                           snapshot["positions"][symbol], at)
            except Exception as exc:
                reason = f"manual position adoption blocked for {symbol}: {type(exc).__name__}"
                state.execution_status = "inconsistent"
                state.inconsistencies = [reason]
                state.protective_stop_audit = {
                    "at": at.isoformat(), "status": "error",
                    "position_count": len(state.portfolio.positions),
                    "mismatch_count": 1, "mismatches": [reason],
                }
                state.events.append({
                    "event": "manual_position_adoption_blocked", "at": at.isoformat(),
                    "symbol": symbol, "reason": type(exc).__name__,
                    "execution_source": _execution_source(state),
                })
                logger.error("Lead manual position adoption blocked for {}: {}",
                             symbol, type(exc).__name__)
                return False
        local = set(state.portfolio.positions)
    if (live - local) - pending_symbols or local - live:
        state.execution_status = "inconsistent"
        mismatch = "exchange and local position symbols differ"
        state.protective_stop_audit = {
            "at": at.isoformat(), "status": "error",
            "position_count": len(state.portfolio.positions),
            "mismatch_count": 1, "mismatches": [mismatch],
        }
        state.inconsistencies = [mismatch]
        state.events.append({"event": "demo_position_sync_mismatch", "at": at.isoformat(),
                             "local_symbols": sorted(local), "exchange_symbols": sorted(live)})
        return False
    orphan_cleanup = getattr(execution, "cancel_orphaned_bot_stops", None)
    if callable(orphan_cleanup) and not pending_symbols:
        try:
            cancelled = orphan_cleanup(live)
        except Exception as exc:
            mismatch = (
                f"orphaned {_execution_label(execution, state)} protective stop cleanup: "
                f"{type(exc).__name__}"
            )
            state.execution_status = "inconsistent"
            state.inconsistencies = [mismatch]
            state.protective_stop_audit = {
                "at": at.isoformat(), "status": "error",
                "position_count": len(state.portfolio.positions),
                "mismatch_count": 1, "mismatches": [mismatch],
            }
            logger.error("{} orphan-stop cleanup failed: {}",
                         _execution_label(execution, state), type(exc).__name__)
            return False
        for row in cancelled:
            state.events.append({"event": "demo_orphan_stop_cancelled", "at": at.isoformat(),
                                 "symbol": row["symbol"], "algoId": row["algo_id"],
                                 "execution_source": _execution_source(state)})
            logger.warning("Canceled orphaned bot protective stop for {} algoId={}",
                           row["symbol"], row["algo_id"])
    for symbol in local:
        meta = state.exit_management_by_symbol.setdefault(symbol, {})
        stop = state.exchange_orders_by_symbol.get(symbol, {}).get("stop") or {}
        if stop.get("stop_price") is not None:
            observed_stop = float(stop["stop_price"])
            previous_stop = meta.get("protective_stop")
            position = state.portfolio.positions[symbol]
            if (previous_stop is not None
                    and _stop_is_less_protective(position.side, observed_stop, float(previous_stop))):
                incident = {"previous_stop": float(previous_stop), "observed_stop": observed_stop}
                if meta.get("stop_regression") != incident:
                    state.events.append({
                        "event": "protective_stop_regression_detected", "at": at.isoformat(),
                        "symbol": symbol, "side": position.side.value,
                        **incident, "algoId": stop.get("id"),
                        "execution_source": _execution_source(state),
                    })
                    logger.error(
                        "{} protective stop regressed for {}: recorded={} exchange={} algoId={}",
                        _execution_label(execution, state), symbol, previous_stop, observed_stop, stop.get("id"),
                    )
                meta["stop_regression"] = incident
                meta["protective_stop"] = float(previous_stop)
            else:
                regression = meta.pop("stop_regression", None)
                if regression:
                    state.events.append({
                        "event": "protective_stop_regression_cleared", "at": at.isoformat(),
                        "symbol": symbol, "side": position.side.value,
                        "previous_stop": regression["previous_stop"], "stop_price": observed_stop,
                        "algoId": stop.get("id"), "execution_source": _execution_source(state),
                    })
                meta["protective_stop"] = observed_stop
            meta["enabled"] = True
        else:
            meta["enabled"] = False
    stop_regressions = _demo_stop_regression_mismatches(state)
    if stop_regressions:
        state.protective_stop_audit = {
            "at": at.isoformat(), "status": "mismatch",
            "position_count": len(state.portfolio.positions),
            "mismatch_count": len(stop_regressions), "mismatches": stop_regressions,
        }
    audit = state.protective_stop_audit
    if audit.get("status") in {"mismatch", "error"}:
        state.execution_status = "inconsistent"
        state.inconsistencies = list(audit.get("mismatches", []))
    else:
        state.execution_status = "synchronized"
        state.inconsistencies = []
    return True


def _adoption_closed_frame(exchange, symbol, opened_at, timeframe, timeframe_ms,
                           required_bars, history_bars):
    opened_ms = int(opened_at.timestamp() * 1000)
    limit = min(1499, max(required_bars, history_bars))
    since = max(0, opened_ms - (limit + 1) * timeframe_ms)
    rows = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit + 1)
    completed = _closed(rows, timeframe_ms, opened_ms)
    frame = ohlcv_frame(completed)
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.tail(history_bars)
    expected_open_ms = (opened_ms // timeframe_ms) * timeframe_ms - timeframe_ms
    if len(frame) < required_bars or not len(frame):
        raise RuntimeError(f"insufficient completed {timeframe} history")
    if int(frame.index[-1].timestamp() * 1000) != expected_open_ms:
        raise RuntimeError(f"completed {timeframe} history does not reach the entry time")
    if len(frame) > 1:
        differences = frame.index.to_series().diff().dropna().dt.total_seconds() * 1000
        if not (differences == timeframe_ms).all():
            raise RuntimeError(f"completed {timeframe} history has gaps")
    validate_ohlcv(frame)
    return frame


def _adopt_lead_manual_position(execution, state, symbol, exchange_position, at):
    if not getattr(execution, "adopt_manual_positions", False):
        raise RuntimeError("manual adoption is disabled")
    if state.execution_mode != "lead_usdm" or symbol in state.portfolio.positions:
        raise RuntimeError("manual adoption is restricted to untracked Lead positions")
    settings = execution.settings
    if not execution.exchange.markets:
        execution.exchange.load_markets()
    market_id = str(execution.exchange.market(symbol).get("id") or "").upper()
    if not market_id or market_id not in getattr(execution, "lead_market_ids", set()):
        raise RuntimeError("manual position is outside the verified Lead contract whitelist")
    recovered = execution.reconstruct_open_position(symbol, exchange_position, at=at)
    side = Side(recovered["side"])
    opened_at = recovered["opened_at"]
    atr_required = settings.lookback.atr_len + 1
    frame1 = _adoption_closed_frame(
        execution.exchange, symbol, opened_at, "1h", HOUR_MS,
        max(atr_required, settings.stops.swing_lookback_1h),
        60,
    )
    regime_required = max(
        settings.lookback.min_regime_bars_4h,
        settings.regime.atrpct_lookback,
        settings.lookback.range_bars_4h + 1,
        settings.lookback.slow_lookback_4h + 1,
        settings.lookback.adx_len * 3,
    )
    frame4 = _adoption_closed_frame(
        execution.exchange, symbol, opened_at, "4h", FOUR_H_MS, regime_required,
        max(regime_required, 100),
    )
    atr_series = atr(frame1, settings.lookback.atr_len)
    entry_atr = float(atr_series.iloc[-1])
    if not math.isfinite(entry_atr) or entry_atr <= 0:
        raise RuntimeError("entry-time ATR is unavailable")
    swing = frame1.tail(settings.stops.swing_lookback_1h)
    swing_low, swing_high = float(swing["low"].min()), float(swing["high"].max())
    structure_price = swing_low if side is Side.LONG else swing_high
    strategy_stop, _ = calculate_adaptive_stop(
        side, float(recovered["entry_price"]), entry_atr, structure_price, settings,
    )
    regimes = classify_regimes(frame4, settings)
    entry_regime = regimes.iloc[-1]["regime"]
    if entry_regime is None:
        raise RuntimeError("entry-time 4H regime is unavailable")
    regime_value = entry_regime.value if isinstance(entry_regime, Regime) else str(entry_regime)

    ticker = execution.exchange.fetch_ticker(symbol)
    checked_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    market_price = _ticker_price(
        {symbol: ticker}, symbol, checked_ms, max_age_ms=DEMO_POSITION_QUOTE_MAX_AGE_MS,
    )
    if market_price is None:
        raise RuntimeError("fresh contract price is unavailable")
    observed_at = _ticker_observed_at({symbol: ticker})
    if observed_at is None or observed_at > datetime.now(timezone.utc) + timedelta(seconds=30):
        raise RuntimeError("contract price timestamp is invalid")

    candidate_position = Position(
        symbol, side, float(recovered["quantity"]), float(recovered["entry_price"]),
        float(strategy_stop), opened_at, "manual_position_adopted", regime_value, MEAN_REVERSION,
    )
    stop = execution.adopt_manual_position_stop(
        candidate_position, strategy_stop=strategy_stop, market_price=market_price,
    )
    position = Position(
        symbol, side, float(recovered["quantity"]), float(recovered["entry_price"]),
        float(stop["stop_price"]), opened_at, "manual_position_adopted", regime_value, MEAN_REVERSION,
    )
    state.portfolio.positions[symbol] = position
    state.exchange_orders_by_symbol[symbol] = {"stop": stop}
    state.entry_costs[symbol] = {"entry_fee": float(recovered["entry_fee"])}
    state.mark_prices[symbol] = market_price
    _initialize_entry_metadata(
        state, symbol, position, True, market_price, float(stop["stop_price"]),
        entry_atr, swing_low, swing_high, observed_price=market_price,
        observed_at=observed_at, telemetry_history_complete=False,
    )
    _invalidate_demo_stop_audit(state)
    state.events.append({
        "event": "manual_position_adopted", "at": at.isoformat(), "symbol": symbol,
        "side": side.value, "quantity": position.quantity,
        "entry_price": position.entry_price, "opened_at": opened_at.isoformat(),
        "reason": position.reason, "regime": position.regime,
        "fee": float(recovered["entry_fee"]),
        "stop_price": position.stop_price, "entry_history_complete": True,
        "excursion_history_complete": False, "execution_source": _execution_source(state),
        "strategy_sample_eligible": False,
        "sample_exclusion_reason": "manual_position_adopted",
    })
    logger.warning("Lead adopted manual position {} {} quantity={} entry={} stop={}",
                   symbol, side.value, position.quantity, position.entry_price,
                   position.stop_price)


def _invalidate_demo_stop_audit(state):
    if state.execution_mode in {"demo_usdm", "lead_usdm"}:
        state.protective_stop_audit = {}


def _stop_is_less_protective(side, observed: float, previous: float) -> bool:
    return observed < previous if side is Side.LONG else observed > previous


def _demo_stop_regression_mismatches(state) -> list[str]:
    mismatches = []
    for symbol, position in state.portfolio.positions.items():
        meta = state.exit_management_by_symbol.get(symbol) or {}
        stop = state.exchange_orders_by_symbol.get(symbol, {}).get("stop") or {}
        recorded = meta.get("protective_stop")
        observed = stop.get("stop_price")
        if recorded is None or observed is None:
            continue
        if _stop_is_less_protective(position.side, float(observed), float(recorded)):
            mismatches.append(
                f"{symbol}: active {'Lead Portfolio' if state.execution_mode == 'lead_usdm' else 'Demo'} stop "
                f"{float(observed)} is less protective than recorded stop {float(recorded)}"
            )
    return mismatches


def _demo_stop_audit_due(state, now):
    audit = state.protective_stop_audit
    if audit.get("status") != "verified":
        return True
    try:
        audited_at = datetime.fromisoformat(audit["at"])
        if audited_at.tzinfo is None:
            audited_at = audited_at.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return True
    return (now - audited_at).total_seconds() >= PROTECTIVE_STOP_AUDIT_INTERVAL_SECONDS


def _audit_demo_protective_stops(execution, state, now):
    try:
        mismatches = list(execution.audit_active_protective_stops(
            state.portfolio.positions, state.exchange_orders_by_symbol,
        ) or [])
        mismatches.extend(_demo_stop_regression_mismatches(state))
        status = "mismatch" if mismatches else "verified"
    except Exception as exc:
        mismatches = [f"protective stop audit: {type(exc).__name__}"]
        status = "error"
    state.protective_stop_audit = {
        "at": now.isoformat(), "status": status,
        "position_count": len(state.portfolio.positions),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
    }
    if mismatches:
        state.execution_status = "inconsistent"
        state.inconsistencies = mismatches
        logger.error("{} protective stop audit {}: {}",
                     _execution_label(execution, state), status, "; ".join(mismatches))
        return False
    state.execution_status = "synchronized"
    state.inconsistencies = []
    return True


def _demo_snapshot_is_fresh(state, now):
    snapshot_at = _aware_datetime((state.exchange_snapshot or {}).get("at"))
    if snapshot_at is None:
        return False
    age = (now - snapshot_at).total_seconds()
    return 0 <= age <= DEMO_POSITION_SYNC_MAX_AGE_SECONDS


def _manage_demo_positions(exchange, execution, state, settings, now, fee_rate, slip_rate):
    """Manage Demo time exits and protective stop moves from fresh contract quotes."""
    if not state.portfolio.positions:
        return False

    audit = state.protective_stop_audit or {}
    audit_at = _aware_datetime(audit.get("at"))
    recent_audit_error = (
        audit.get("status") in {"mismatch", "error"}
        and audit_at is not None
        and 0 <= (now - audit_at).total_seconds() < DEMO_POSITION_SYNC_MAX_AGE_SECONDS
    )
    snapshot_fresh = _demo_snapshot_is_fresh(state, now)
    if (not snapshot_fresh or (state.execution_status != "synchronized" and not recent_audit_error)):
        if not _demo_sync_stopped(execution, state, now):
            logger.error("{} position monitor paused: exchange state is not synchronized",
                         _execution_label(execution, state))
            return True

    if state.execution_status != "synchronized":
        if recent_audit_error or not _audit_demo_protective_stops(execution, state, now):
            return True
    if _demo_stop_audit_due(state, now) and not _audit_demo_protective_stops(execution, state, now):
        return True
    if state.execution_status != "synchronized":
        return True

    dirty = False
    decision_at = datetime.now(timezone.utc)
    max_hold_hours = float(settings.exits.max_hold_bars)
    for symbol, position in list(state.portfolio.positions.items()):
        opened_at = _aware_datetime(position.opened_at)
        if opened_at is None:
            continue
        held_hours = (decision_at - opened_at).total_seconds() / 3600
        if held_hours >= max_hold_hours:
            _close_position(True, execution, state, symbol, None, "max_hold_24h",
                            decision_at, fee_rate, slip_rate)
            if symbol not in state.portfolio.positions:
                dirty = True
                logger.info("{} max-hold exit {} held_hours={:.3f}",
                            _execution_label(execution, state), symbol, held_hours)

    if not state.portfolio.positions:
        return dirty
    if not exchange.markets:
        exchange.load_markets()

    stops = settings.stops
    for symbol, position in list(state.portfolio.positions.items()):
        try:
            ticker = exchange.fetch_ticker(symbol)
            quote_error = None
        except Exception as exc:
            ticker = {}
            quote_error = type(exc).__name__

        checked_at = datetime.now(timezone.utc)
        checked_ms = int(checked_at.timestamp() * 1000)
        quote_timestamp = (ticker or {}).get("timestamp")
        try:
            quote_age_ms = checked_ms - int(quote_timestamp)
        except (TypeError, ValueError, OverflowError):
            quote_age_ms = None
        price = (_ticker_price({symbol: ticker}, symbol, checked_ms,
                               max_age_ms=DEMO_POSITION_QUOTE_MAX_AGE_MS)
                 if quote_error is None else None)
        if quote_error is not None:
            quote_status = f"unavailable:{quote_error}"
        elif price is not None:
            quote_status = "fresh"
        elif quote_age_ms is not None and quote_age_ms > DEMO_POSITION_QUOTE_MAX_AGE_MS:
            quote_status = "stale"
        else:
            quote_status = "missing_or_invalid"

        meta = state.exit_management_by_symbol.setdefault(symbol, {})
        monitor = meta.setdefault("position_monitor", {})
        previous_status = monitor.get("quote_status")
        observed_at = _ticker_observed_at({symbol: ticker}, symbol)
        monitor.update({
            "checked_at": checked_at.isoformat(),
            "quote_status": quote_status,
            "quote_at": observed_at.isoformat() if observed_at is not None else None,
            "quote_age_ms": quote_age_ms,
        })
        if previous_status != quote_status:
            state.events.append({
                "event": "demo_position_quote_status", "at": checked_at.isoformat(),
                "symbol": symbol, "status": quote_status,
                "previous_status": previous_status, "quote_age_ms": quote_age_ms,
                "execution_mode": state.execution_mode,
            })
            if quote_status == "fresh":
                logger.info("{} position quote recovered for {}", _execution_label(execution, state), symbol)
            else:
                logger.warning("{} position quote unavailable for {} status={}",
                               _execution_label(execution, state), symbol, quote_status)
            dirty = True
        dirty = True  # Persist each valid excursion sample and the latest freshness result.
        if price is None:
            state.exit_management_by_symbol[symbol] = meta
            continue

        _update_demo_excursions(meta, position, price=price, observed_at=observed_at)
        state.mark_prices[symbol] = price
        entry = float(position.entry_price)
        best_price = float(meta.get("best_price", price))
        worst_price = float(meta.get("worst_price", price))
        if position.side is Side.LONG:
            best_price = max(best_price, price)
            worst_price = min(worst_price, price)
        else:
            best_price = min(best_price, price)
            worst_price = max(worst_price, price)
        meta["best_price"] = best_price
        meta["worst_price"] = worst_price

        entry_atr = float(meta.get("entry_atr", 0.0) or 0.0)
        initial_risk = float(meta.get("initial_risk", 0.0) or 0.0)
        tracked_order = (state.exchange_orders_by_symbol.get(symbol) or {}).get("stop")
        if entry_atr <= 0 or initial_risk <= 0 or not tracked_order:
            state.exit_management_by_symbol[symbol] = meta
            continue
        tracked_stop = float(tracked_order.get("stop_price", position.stop_price))
        recorded_stop = float(meta.get("protective_stop", position.stop_price))
        current_stop = (max(tracked_stop, recorded_stop) if position.side is Side.LONG
                        else min(tracked_stop, recorded_stop))
        meta["protective_stop"] = current_stop

        quote_exit = structure_invalid_quote_exit(
            position.side, price, meta.get("structure_low"), meta.get("structure_high"),
            entry_atr, current_stop,
            buffer_atr=settings.exits.structure_invalid_buffer_atr,
        )
        if quote_exit is not None:
            reason, invalid_price = quote_exit
            state.events.append({
                "event": "structure_invalidation_quote_triggered",
                "at": checked_at.isoformat(), "symbol": symbol,
                "side": position.side.value, "reason": reason,
                "quote_price": price,
                "quote_at": observed_at.isoformat() if observed_at is not None else None,
                "quote_age_ms": quote_age_ms,
                "invalidation_price": invalid_price,
                "protective_stop": current_stop,
                "entry_atr": entry_atr,
                "buffer_atr": settings.exits.structure_invalid_buffer_atr,
            })
            dirty = True
            _close_position(True, execution, state, symbol, price, reason,
                            checked_at, fee_rate, slip_rate)
            if symbol not in state.portfolio.positions:
                logger.info("{} structure invalidation exit {} {} quote={} level={} hard_stop={}",
                            _execution_label(execution, state), symbol, position.side.value, price,
                            invalid_price, current_stop)
            continue

        favorable = (best_price - entry) if position.side is Side.LONG else (entry - best_price)
        locked_stop = staged_profit_lock_stop(
            position.side, entry, entry_atr, initial_risk, favorable, current_stop,
            first_trigger_atr=stops.move_to_breakeven_atr,
            first_lock_r=settings.mr_candidate_exit.first_lock_r,
            second_trigger_r=settings.mr_candidate_exit.second_lock_trigger_r,
            second_lock_r=settings.mr_candidate_exit.second_lock_r,
            fee_rate=fee_rate, slip_rate=slip_rate,
        )
        progress_r = favorable / initial_risk
        proposed_stop = trailing_stop(
            position.side, best_price, initial_risk, progress_r, locked_stop,
            trigger_r=settings.mr_candidate_exit.trailing_trigger_r,
            atr_mult=settings.exit_management.trailing_atr_mult,
            stop_atr_mult=settings.signals.stop_atr_mult,
        )
        try:
            proposed_stop = float(execution.exchange.price_to_precision(symbol, proposed_stop))
        except (ccxt.BaseError, TypeError, ValueError, OverflowError):
            state.exit_management_by_symbol[symbol] = meta
            continue
        improved = (proposed_stop > current_stop if position.side is Side.LONG
                    else proposed_stop < current_stop)
        if not improved:
            state.exit_management_by_symbol[symbol] = meta
            continue
        if ((position.side is Side.LONG and proposed_stop >= price)
                or (position.side is Side.SHORT and proposed_stop <= price)):
            state.exit_management_by_symbol[symbol] = meta
            continue

        first_breakeven_move = _stop_is_less_protective(position.side, current_stop, locked_stop)
        if not first_breakeven_move:
            last_update = _aware_datetime(meta.get("last_fast_stop_update_at"))
            if (last_update is not None
                    and (checked_at - last_update).total_seconds()
                    < DEMO_STOP_UPDATE_MIN_INTERVAL_SECONDS):
                state.exit_management_by_symbol[symbol] = meta
                continue

        if state.protective_stop_audit.get("status") != "verified":
            state.exit_management_by_symbol[symbol] = meta
            continue
        try:
            updated = execution.replace_stop(position, stop=tracked_order,
                                             stop_price=proposed_stop)
        except Exception as exc:
            _invalidate_demo_stop_audit(state)
            logger.warning("{} stop move failed for {}: {}",
                           _execution_label(execution, state), symbol, type(exc).__name__)
            state.exit_management_by_symbol[symbol] = meta
            continue

        state.exchange_orders_by_symbol.setdefault(symbol, {})["stop"] = updated
        actual_stop = float(updated.get("stop_price", proposed_stop))
        meta["protective_stop"] = actual_stop
        meta["last_fast_stop_update_at"] = checked_at.isoformat()
        stop_reason = "breakeven_or_trailing"
        state.events.append({
            "event": "protective_stop_updated", "action": "REPLACE",
            "symbol": symbol, "side": position.side.value, "at": checked_at.isoformat(),
            "previous_stop": tracked_stop, "stop_price": actual_stop,
            "requested_stop": proposed_stop, "previous_algo_id": tracked_order.get("id"),
            "algoId": updated.get("id"), "reason": stop_reason,
            "execution_source": _execution_source(state),
        })
        logger.info("{} protective stop updated for {}: {} -> {} algoId={}",
                    _execution_label(execution, state), symbol, tracked_stop, actual_stop, updated.get("id"))
        _invalidate_demo_stop_audit(state)
        state.exit_management_by_symbol[symbol] = meta

    return dirty


# --- universe + bars ------------------------------------------------------------------

def _closed(frame_rows, timeframe_ms, now_ms):
    return [r for r in frame_rows if len(r) >= 6 and int(r[0]) + timeframe_ms <= now_ms]


def _daily_direction_gate(exchange, symbol, signal_at, span, *, fail_on_market_error=False):
    """Return a fail-closed direction from completed UTC daily candles only."""
    signal_at = signal_at.replace(tzinfo=timezone.utc) if signal_at.tzinfo is None else signal_at
    signal_at = signal_at.astimezone(timezone.utc)
    signal_ms = int(signal_at.timestamp() * 1000)
    try:
        rows = exchange.fetch_ohlcv(symbol, "1d", limit=max(span + 1, 100))
    except ccxt.BaseError as exc:
        _raise_market_error_if_needed(exc, fail_on_market_error)
        return {
            "ready": False, "direction": None, "reason": "daily_data_unavailable",
            "daily_bars": 0, "signal_at": signal_at.isoformat(),
            "error_type": type(exc).__name__,
        }

    try:
        completed = _closed(rows, DAY_MS, signal_ms)
        frame = ohlcv_frame(completed)
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        if len(frame) < span + 1:
            return {
                "ready": False, "direction": None,
                "reason": "daily_history_insufficient", "daily_bars": len(frame),
                "signal_at": signal_at.isoformat(),
            }
        expected_close = signal_at.replace(hour=0, minute=0, second=0, microsecond=0)
        if frame.index[-1].to_pydatetime() + timedelta(days=1) != expected_close:
            return {
                "ready": False, "direction": None, "reason": "daily_data_stale",
                "daily_bars": len(frame), "signal_at": signal_at.isoformat(),
            }

        closes = pd.to_numeric(frame["close"], errors="coerce")
        if closes.isna().any() or any(not math.isfinite(float(value)) for value in closes):
            raise ValueError("invalid daily close")
        ema = closes.ewm(span=span, adjust=False, min_periods=span).mean()
        daily_close = float(closes.iloc[-1])
        daily_ema = float(ema.iloc[-1])
        if not math.isfinite(daily_ema) or daily_ema <= 0 or daily_close <= 0:
            raise ValueError("invalid daily EMA")
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "ready": False, "direction": None, "reason": "daily_data_invalid",
            "daily_bars": 0, "signal_at": signal_at.isoformat(),
            "error_type": type(exc).__name__,
        }

    if daily_close > daily_ema:
        direction = "long"
    elif daily_close < daily_ema:
        direction = "short"
    else:
        return {
            "ready": False, "direction": None,
            "reason": "daily_close_equals_ema50", "daily_bars": len(frame),
            "daily_close": daily_close, "daily_ema50": daily_ema,
            "daily_bar_close_time": (frame.index[-1].to_pydatetime()
                                     + timedelta(days=1)).isoformat(),
            "signal_at": signal_at.isoformat(),
        }

    return {
        "ready": True, "direction": direction, "reason": None,
        "daily_bars": len(frame), "daily_close": daily_close,
        "daily_ema50": daily_ema,
        "daily_bar_close_time": (frame.index[-1].to_pydatetime()
                                 + timedelta(days=1)).isoformat(),
        "signal_at": signal_at.isoformat(),
    }


def _cached_daily_direction_gate(exchange, symbol, signal_at, span, cache, **kwargs):
    if cache is None:
        return _daily_direction_gate(exchange, symbol, signal_at, span, **kwargs)
    key = (int(signal_at.timestamp() * 1000) // DAY_MS, span)
    saved = cache.daily_direction.get(symbol)
    if saved is not None and saved[0] == key:
        return {**saved[1], "signal_at": signal_at.isoformat()}
    result = _daily_direction_gate(exchange, symbol, signal_at, span, **kwargs)
    if result.get("ready"):
        cache.daily_direction[symbol] = (key, result)
    return result


def _cached_four_hour(exchange, symbol, now_ms, cache):
    key = now_ms // FOUR_H_MS
    saved = cache.four_hour.get(symbol) if cache is not None else None
    if saved is not None and saved[0] == key:
        return saved[1]
    rows = _closed(exchange.fetch_ohlcv(symbol, "4h", limit=100), FOUR_H_MS, now_ms)
    if not rows or int(rows[-1][0]) + FOUR_H_MS != key * FOUR_H_MS:
        return None
    try:
        frame = ohlcv_frame(rows)
        validate_ohlcv(frame)
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if cache is not None:
        cache.four_hour[symbol] = (key, frame)
    return frame


def _daily_direction_rejection_reason(gate, side):
    if not gate["ready"]:
        return gate["reason"]
    if side.value != gate["direction"]:
        return f"daily_direction_{gate['direction']}_only"
    return None


def _daily_gate_status_fields(gate):
    fields = {key: gate[key] for key in (
        "daily_bars", "daily_close", "daily_ema50",
        "daily_bar_close_time", "signal_at", "error_type",
    ) if key in gate}
    if "direction" in gate:
        fields["daily_direction"] = gate["direction"]
    return fields


def _current_hour_observation(rows, now_ms):
    for row in reversed(rows):
        if len(row) < 6:
            continue
        opened_ms = int(row[0])
        if opened_ms <= now_ms < opened_ms + HOUR_MS:
            return {
                "bar_open_ms": opened_ms,
                "hour_key": datetime.fromtimestamp(opened_ms / 1000, timezone.utc).isoformat(),
                "open": float(row[1]),
            }
    return None


def _is_rate_limit_error(exc):
    return (isinstance(exc, (ccxt.DDoSProtection, ccxt.RateLimitExceeded))
            or getattr(exc, "http_status", None) in {418, 429})


def _raise_market_error_if_needed(exc, enabled):
    if enabled:
        raise exc


def _market_retry_delay_seconds(exc, *, poll, attempt, now=None):
    """Honor Binance's explicit ban expiry; otherwise use bounded backoff."""
    now = now or datetime.now(timezone.utc)
    match = BAN_UNTIL_RE.search(str(exc))
    if match:
        retry_at = int(match.group(1)) / 1000
        remaining = retry_at - now.timestamp()
        if remaining > 0:
            return max(float(poll), remaining + 3.0)
    base = max(float(poll), 30.0 if _is_rate_limit_error(exc) else float(poll))
    return min(900.0, base * (2 ** min(max(0, int(attempt) - 1), 5)))


def _persisted_market_retry_after(state):
    evidence = (state.market_risk_status or {}).get("evidence") or {}
    raw = evidence.get("retry_after")
    if not raw:
        return None
    try:
        retry_after = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if retry_after.tzinfo is None:
        return None
    return retry_after.astimezone(timezone.utc)


def _record_live_stop_monitor_status(state, now, *, tickers=None, now_ms=None,
                                     default_reason=None):
    """Persist whether each open local Paper stop has a fresh public quote."""
    now_ms = now_ms if now_ms is not None else int(now.timestamp() * 1000)
    tickers = tickers or {}
    for symbol in list(state.portfolio.positions):
        ticker = tickers.get(symbol) or {}
        price = _ticker_price(tickers, symbol, now_ms) if default_reason is None else None
        timestamp = ticker.get("timestamp")
        quote_at = None
        quote_age = None
        if price is not None and timestamp is not None:
            quote_at = datetime.fromtimestamp(int(timestamp) / 1000, timezone.utc).isoformat()
            quote_age = max(0.0, (now_ms - int(timestamp)) / 1000)
        status = "active" if price is not None else "unavailable"
        reason = None if price is not None else (default_reason or "ticker_missing_or_stale")
        meta = state.exit_management_by_symbol.setdefault(symbol, {})
        previous = meta.get("live_stop_monitor") or {}
        observation = {
            "status": status,
            "checked_at": now.isoformat(),
            "quote_at": quote_at,
            "quote_age_sec": round(quote_age, 3) if quote_age is not None else None,
            "price": price,
            "reason": reason,
        }
        meta["live_stop_monitor"] = observation
        if (previous.get("status"), previous.get("reason")) != (status, reason):
            state.events.append({
                "event": "live_stop_monitor_state",
                "at": now.isoformat(),
                "symbol": symbol,
                "state": status,
                "reason": reason or "fresh_public_quote",
                "warning": ("本地保护止损没有新鲜行情，当前无法执行实时价格监控"
                            if status == "unavailable" else "本地保护止损已恢复实时价格监控"),
                "quote_age_sec": observation["quote_age_sec"],
            })


def _mark_market_data_unavailable(state, now, error, *, retry_after=None,
                                  reason="market_data_unavailable", tickers=None):
    error_type = type(error).__name__ if isinstance(error, BaseException) else str(error)
    evidence = {
        "monitoring_enabled": True,
        "fetch_error": error_type,
    }
    if retry_after is not None:
        evidence["retry_after"] = retry_after.isoformat()
    status = {
        "mode": "entry_gate",
        "state": MarketRiskState.PAUSE_NEW.value,
        "reason": reason,
        "checked_at": now.isoformat(),
        "symbol": "BTC/USDT:USDT",
        "timeframe": "4h",
        "evidence": evidence,
    }
    previous = state.market_risk_status
    if (previous.get("state"), previous.get("reason")) != (status["state"], status["reason"]):
        state.events.append({"event": "market_risk_observed", "at": status["checked_at"], **status})
    state.market_risk_status = status
    _record_live_stop_monitor_status(
        state, now, tickers=tickers,
        default_reason=error_type if tickers is None else None,
    )


def _ticker_price(tickers, symbol, now_ms, *, max_age_ms=LIVE_TICKER_MAX_AGE_MS):
    ticker = tickers.get(symbol) or {}
    value = ticker.get("last")
    if value is None:
        value = ticker.get("close")
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    timestamp = ticker.get("timestamp")
    if timestamp is None:
        return None
    try:
        age_ms = now_ms - int(timestamp)
    except (TypeError, ValueError, OverflowError):
        return None
    if age_ms < -30_000 or age_ms > max_age_ms:
        return None
    return price


def _ticker_observed_at(tickers, symbol):
    timestamp = (tickers.get(symbol) or {}).get("timestamp")
    try:
        return datetime.fromtimestamp(int(timestamp) / 1000, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _build_universe(exchange, settings, now, now_ms, *, fail_on_market_error=False,
                    on_progress=None, retry_reads=False):
    deadline = now.replace(minute=0, second=0, microsecond=0) + timedelta(
        seconds=settings.execution.confirm_delay_sec_max)
    read_kwargs = {"retry": retry_reads, "deadline": deadline, "monitor": on_progress}
    markets = _read_entry_data(exchange.load_markets, phase="universe_markets", **read_kwargs)
    tickers = _read_entry_data(exchange.fetch_tickers, phase="universe_tickers", **read_kwargs)
    candidates = filter_tradable_crypto_universe(markets, tickers, settings, now=now)
    one_hour: dict[str, pd.DataFrame] = {}
    current_hour: dict[str, dict] = {}
    activity: dict[str, dict] = {}
    for symbol, _q in candidates:
        if on_progress is not None:
            on_progress()
        try:
            rows = _read_entry_data(
                lambda: exchange.fetch_ohlcv(symbol, "1h", limit=60),
                phase="universe_hourly", symbol=symbol, **read_kwargs,
            )
        except ccxt.BaseError as exc:
            _raise_market_error_if_needed(exc, fail_on_market_error)
            continue
        observation = _current_hour_observation(rows, now_ms)
        if observation is not None:
            current_hour[symbol] = observation
        try:
            activity[symbol] = hot_activity_metrics(rows, now_ms=now_ms)
        except Exception:
            activity[symbol] = {"volume_burst_ratio": None, "return_4h_pct": None, "return_24h_pct": None}
        closed = _closed(rows, HOUR_MS, now_ms)
        if closed:
            one_hour[symbol] = ohlcv_frame(closed)
    scored = score_hot_candidates(candidates, tickers, activity)
    hotlist = [{**row, "rank": rank}
               for rank, row in enumerate(scored[:TOP_N], start=1)]
    return markets, hotlist, one_hour, current_hour, tickers


def _fetch_selected_bars(exchange, symbols, settings, now_ms, *, fail_on_market_error=False,
                         on_progress=None):
    one_hour: dict[str, pd.DataFrame] = {}
    current_hour: dict[str, dict] = {}
    four_hour: dict[str, pd.DataFrame] = {}
    regimes: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if on_progress is not None:
            on_progress()
        try:
            rows1 = exchange.fetch_ohlcv(symbol, "1h", limit=60)
        except ccxt.BaseError as exc:
            _raise_market_error_if_needed(exc, fail_on_market_error)
            continue
        observation = _current_hour_observation(rows1, now_ms)
        if observation is not None:
            current_hour[symbol] = observation
        closed1 = _closed(rows1, HOUR_MS, now_ms)
        if closed1:
            one_hour[symbol] = ohlcv_frame(closed1)
        try:
            rows4 = exchange.fetch_ohlcv(symbol, "4h", limit=100)
        except ccxt.BaseError as exc:
            _raise_market_error_if_needed(exc, fail_on_market_error)
            continue
        closed4 = _closed(rows4, FOUR_H_MS, now_ms)
        if not closed4:
            continue
        frame4 = ohlcv_frame(closed4)
        four_hour[symbol] = frame4
        try:
            regimes[symbol] = classify_regimes(frame4, settings)
        except Exception:
            regimes[symbol] = None
        if on_progress is not None:
            on_progress()
    return one_hour, current_hour, four_hour, regimes


def _hotlist_cache_is_fresh(state, now):
    if not state.watchlist:
        return False
    cached = state.exit_management_by_symbol.get(HOTLIST_CACHE_KEY) or {}
    try:
        refreshed_at = datetime.fromisoformat(cached["refreshed_at"])
        if refreshed_at.tzinfo is None:
            return False
        age = (now - refreshed_at.astimezone(timezone.utc)).total_seconds()
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= age < HOTLIST_REFRESH_SECONDS


def _set_symbol_status(state, symbol, *, action="NONE", side=None, reason,
                       at, regime="UNKNOWN", source=MEAN_REVERSION):
    state.symbol_status[symbol] = {
        "regime": regime,
        "action": action,
        "side": side.value if isinstance(side, Side) else side,
        "reason": reason,
        "at": at.isoformat(),
        "symbol": symbol,
        "source": source,
    }


def _fetch_entry_funding_rate(exchange, symbol, *, fail_on_market_error=False):
    try:
        funding = exchange.fetch_funding_rate(symbol)
        return (float(funding["fundingRate"])
                if funding and funding.get("fundingRate") is not None else None)
    except (ccxt.BaseError, KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ccxt.BaseError):
            _raise_market_error_if_needed(exc, fail_on_market_error)
        logger.warning("funding rate unavailable for {}; entry will fail closed: {}",
                       symbol, type(exc).__name__)
        return None


def _record_funding_entry_block(state, symbol, side, regime, funding_rate, reason,
                                settings, now):
    limit = float(settings.risk.funding_abs_block)
    _set_symbol_status(state, symbol, side=side, reason=reason, at=now, regime=regime)
    state.symbol_status[symbol]["funding_rate"] = funding_rate
    state.symbol_status[symbol]["funding_abs_limit"] = limit
    state.events.append({
        "event": "entry_signal_blocked", "at": now.isoformat(), "symbol": symbol,
        "side": side.value, "source": MEAN_REVERSION, "reason": reason,
        "funding_rate": funding_rate, "funding_abs_limit": limit,
    })
    logger.warning("MR entry candidate excluded {} side={} funding_rate={} abs_limit={} reason={}",
                   symbol, side.value, funding_rate, limit, reason)


def _record_daily_entry_block(state, symbol, side, regime, gate, reason, now):
    _set_symbol_status(state, symbol, side=side, reason=reason, at=now, regime=regime)
    state.symbol_status[symbol].update(_daily_gate_status_fields(gate))
    state.events.append({
        "event": "entry_signal_blocked", "at": now.isoformat(), "symbol": symbol,
        "side": side.value, "source": MEAN_REVERSION, "reason": reason,
        **_daily_gate_status_fields(gate),
    })
    logger.info("MR entry candidate excluded {} side={} reason={} daily_close={} daily_ema50={} daily_bars={}",
                symbol, side.value, reason, gate.get("daily_close"),
                gate.get("daily_ema50"), gate.get("daily_bars"))


def _entry_bar_close_time(frame1, now, *, max_delay=ENTRY_BAR_MAX_DELAY):
    """Return the latest 1H close only when it is closed and still actionable."""
    if (now.tzinfo is None or frame1 is None or frame1.empty
            or not isinstance(frame1.index, pd.DatetimeIndex) or frame1.index.tz is None):
        return None, "hourly_bar_or_clock_unavailable"
    opened_at = frame1.index[-1].to_pydatetime()
    if opened_at.tzinfo is None:
        return None, "hourly_bar_timestamp_naive"
    opened_utc = opened_at.astimezone(timezone.utc)
    if any((opened_utc.minute, opened_utc.second, opened_utc.microsecond)):
        return None, "hourly_bar_timestamp_unaligned"
    close_at = opened_at + timedelta(hours=1)
    delay = now.astimezone(timezone.utc) - close_at.astimezone(timezone.utc)
    if delay.total_seconds() < 0:
        return None, "hourly_bar_not_closed"
    if delay > max_delay:
        return None, "hourly_bar_stale"
    return close_at, None


def _time_filter_block_reasons(flags, side, bar_close_time, settings):
    if not flags.data_ready:
        return [flags.data_reason or "time_filter_data_unavailable"]
    local_close = bar_close_time.astimezone(ZoneInfo(settings.timezone))
    wall_time = local_close.timetz().replace(tzinfo=None)
    session = settings.session
    reasons = []
    if flags.flatten_only:
        reasons.append("night_new_entries_disabled")
    elif side is Side.LONG and not flags.allow_long_entry:
        if flags.morning_move == "UP":
            reasons.append("morning_up_blocks_long")
        if session.afternoon_start <= wall_time < session.afternoon_end:
            if flags.recent_move_atr is not None and flags.recent_move_atr >= settings.signals.impulse_atr_mult:
                reasons.append("afternoon_up_impulse_blocks_long")
            if flags.afternoon_dump_today:
                reasons.append("afternoon_dump_blocks_long")
    elif side is Side.SHORT and not flags.allow_short_entry:
        if flags.morning_move == "DOWN":
            reasons.append("morning_down_blocks_short")
        if session.afternoon_start <= wall_time < session.afternoon_end:
            if flags.afternoon_dump_today and flags.at_lower_extreme:
                reasons.append("afternoon_dump_lower10_blocks_short")
            if flags.at_upper_extreme:
                reasons.append("afternoon_upper10_blocks_short")
    return reasons or (["time_filter_denied_unclassified"]
                       if (side is Side.LONG and not flags.allow_long_entry)
                       or (side is Side.SHORT and not flags.allow_short_entry) else [])


def _record_time_filter_decision(state, symbol, side, frame1, bar_close_time,
                                 flags, settings, now):
    reasons = _time_filter_block_reasons(flags, side, bar_close_time, settings)
    allowed = (flags.data_ready and not flags.flatten_only
               and (flags.allow_long_entry if side is Side.LONG else flags.allow_short_entry))
    local_close = bar_close_time.astimezone(ZoneInfo(settings.timezone))
    close_change = None
    close_change_pct = None
    if len(frame1) >= 2:
        previous_close = float(frame1["close"].iloc[-2])
        current_close = float(frame1["close"].iloc[-1])
        close_change = current_close - previous_close
        if previous_close != 0:
            close_change_pct = close_change / previous_close
    event = {
        "event": "entry_time_filter_decision",
        "at": now.isoformat(),
        "symbol": symbol,
        "source": MEAN_REVERSION,
        "side": side.value,
        "decision": "ALLOW" if allowed else "BLOCK",
        "reason": "time_filter_allowed" if allowed else reasons[0],
        "blocking_reasons": reasons,
        "bar_open_at": frame1.index[-1].to_pydatetime().isoformat(),
        "bar_close_at": bar_close_time.isoformat(),
        "bar_close_local": local_close.isoformat(),
        "session": ("morning" if settings.session.morning_start <= local_close.timetz().replace(tzinfo=None) < settings.session.morning_end
                    else "afternoon" if settings.session.afternoon_start <= local_close.timetz().replace(tzinfo=None) < settings.session.afternoon_end
                    else "night"),
        "close_to_close_change": close_change,
        "close_to_close_change_pct": close_change_pct,
        "recent_move_atr": flags.recent_move_atr,
        "impulse_atr_threshold": float(settings.signals.impulse_atr_mult),
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
        "runner_build_id": RUNNER_BUILD_ID,
        "time_filter_build_id": TIME_FILTER_BUILD_ID,
    }
    state.events.append(event)
    if not allowed:
        _set_symbol_status(state, symbol, action="BLOCKED", side=side,
                           reason=event["reason"], at=now)
        state.symbol_status[symbol].update({
            "entry_bar_close_at": event["bar_close_at"],
            "time_filter_build_id": TIME_FILTER_BUILD_ID,
        })
    level = logger.info if allowed else logger.warning
    level("MR time-filter decision {}", json.dumps(event, ensure_ascii=False, sort_keys=True))
    return allowed


def _record_entry_bar_rejection(state, symbol, frame1, reason, now):
    event = {
        "event": "entry_bar_rejected", "at": now.isoformat(), "symbol": symbol,
        "source": MEAN_REVERSION, "reason": reason,
        "bar_open_at": frame1.index[-1].to_pydatetime().isoformat() if frame1 is not None and len(frame1) else None,
        "runner_build_id": RUNNER_BUILD_ID,
    }
    state.events.append(event)
    _set_symbol_status(state, symbol, action="BLOCKED", reason=reason, at=now)
    logger.warning("MR entry bar rejected {} reason={} bar_open_at={}",
                   symbol, reason, event["bar_open_at"])


# --- strategy loop --------------------------------------------------------------------

def _update_market_risk(exchange, state, settings, now) -> MarketRiskAssessment:
    """Refresh the BTC 4H new-entry gate and the status shown by the dashboard."""
    assessment = fetch_market_risk_status(exchange, settings, now=now)
    status = assessment.to_dict()
    previous = state.market_risk_status
    if (previous.get("state"), previous.get("reason")) != (status["state"], status["reason"]):
        state.events.append({"event": "market_risk_observed", "at": status["checked_at"],
                             **status})
    level = "warning" if assessment.state is MarketRiskState.PAUSE_NEW else "info"
    getattr(logger, level)("Market risk observation {}",
                          json.dumps(status, ensure_ascii=False, sort_keys=True))
    state.market_risk_status = status
    return assessment


def _reconcile_pending_entries(execution, state, checkpoint):
    for attempt in mr_scan.unresolved_attempts(state):
        try:
            report = execution.inspect_entry_attempt(attempt)
        except Exception as exc:
            report = {"status": "query_failed", "error": type(exc).__name__}
        symbol = attempt["symbol"]
        status = report["status"]
        if status == "protected":
            result = report["result"]
            local = state.portfolio.positions.get(symbol)
            live = result["position"]
            if local is not None and (local.side.value != attempt["side"]
                    or not math.isclose(local.quantity, live["quantity"], rel_tol=1e-8)):
                status = "local_position_mismatch"
            else:
                if local is None:
                    local = _record_demo_open(state, symbol, Side(attempt["side"]),
                                              attempt["reason"], attempt["regime"], result,
                                              attempt.get("audit"))
                if symbol not in state.exit_management_by_symbol:
                    meta = attempt["meta"]
                    _initialize_entry_metadata(
                        state, symbol, local, True, meta["last_close"], attempt["stop_price"],
                        meta["entry_atr"], meta["structure_low"], meta["structure_high"],
                    )
                    state.exit_management_by_symbol[symbol]["telemetry_history_complete"] = False
                state.exchange_orders_by_symbol[symbol] = {"stop": result["stop"]}
                attempt["status"] = "completed"
        elif status == "not_filled" and symbol not in state.portfolio.positions:
            attempt["status"] = "not_filled"
        reconciliation = {"status": status, "error": report.get("error")}
        for field in ("reason", "evidence"):
            if field in report:
                reconciliation[field] = report[field]
        if attempt.get("reconciliation") != reconciliation:
            attempt["reconciliation"] = reconciliation
            state.events.append({"event": "entry_order_reconciliation", "at": _utc_now().isoformat(),
                                 "symbol": symbol, "clientOrderId": attempt["clientOrderId"],
                                 **attempt["reconciliation"]})
            mr_scan.attempt_outcome(state, attempt, attempt["status"], _utc_now(),
                                    phase="reconciliation", reconciled=True, error=report.get("error"),
                                    order_may_have_been_sent=True)
            logger.warning("MR entry reconciliation symbol={} status={} error={}",
                           symbol, status, report.get("error"))
        scan = state.mr_entry_scan
        if scan.get("bar") == attempt["bar"] and attempt["status"] in {"completed", "not_filled"}:
            mr_scan.outcome(state, scan, symbol, "reconciled", _utc_now(),
                            side=attempt["side"], order_may_have_been_sent=True)
        mr_scan.checkpoint(checkpoint)
    pending = bool(mr_scan.unresolved_attempts(state))
    if pending:
        state.execution_status = "entry_reconciliation_required"
    return not pending


def _cycle(exchange, execution, state, settings, now, is_demo, max_pos,
           intrabar_entries=False, market_cache: IntrabarMarketCache | None = None,
           position_monitor=None, closed_market_cache: ClosedMarketCache | None = None,
           checkpoint=None):
    now_ms = int(now.timestamp() * 1000)
    fee_rate = settings.fees.taker_bps / 10_000
    slip_rate = settings.fees.slippage_bps / 10_000
    leverage = int(settings.sizing.leverage)
    if is_demo:
        _reconcile_pending_entries(execution, state, checkpoint)
    resume_symbols = (state.mr_entry_scan.get("symbols", [])
                      if is_demo and state.mr_entry_scan
                      and not mr_scan.expired(state.mr_entry_scan, _utc_now()) else [])

    if position_monitor is not None:
        position_monitor()
    market_risk = _update_market_risk(exchange, state, settings, now)
    if position_monitor is not None:
        position_monitor()
    market_error = market_risk.evidence.get("fetch_error") if market_risk.evidence else None
    if intrabar_entries and market_error:
        error_name = str(market_error).split(":", 1)[0]
        error_class = getattr(ccxt, error_name, ccxt.ExchangeError)
        if not isinstance(error_class, type) or not issubclass(error_class, ccxt.BaseError):
            error_class = ccxt.ExchangeError
        raise error_class(str(market_error))

    if intrabar_entries and market_cache is not None:
        refresh_universe = (market_cache.markets is None or not _hotlist_cache_is_fresh(state, now))
        if refresh_universe:
            markets, hotlist, one_hour, current_hour, _universe_tickers = _build_universe(
                exchange, settings, now, now_ms, fail_on_market_error=True,
            )
            market_cache.markets = markets
            market_cache.one_hour.update(one_hour)
            market_cache.current_hour.update(current_hour)
        else:
            markets = market_cache.markets
            hotlist = state.watchlist

        top_symbols = [row["symbol"] for row in hotlist]
        track = sorted(set(top_symbols) | set(state.portfolio.positions) | set(resume_symbols))
        hour_key = now_ms // HOUR_MS
        hour_offset_ms = now_ms % HOUR_MS
        hour_changed = market_cache.bars_hour_key != hour_key
        new_symbols = set(track) - market_cache.selected_symbols
        refresh_closed_bars = (
            market_cache.bars_hour_key is None
            or (hour_changed and hour_offset_ms >= 15_000)
            or (not hour_changed and bool(new_symbols))
        )
        if refresh_closed_bars:
            symbols_to_refresh = track if hour_changed else sorted(new_symbols)
            if hour_changed:
                market_cache.one_hour.clear()
                market_cache.current_hour.clear()
                market_cache.four_hour.clear()
                market_cache.regimes.clear()
            one_hour, current_hour, four_hour, regimes = _fetch_selected_bars(
                exchange, symbols_to_refresh, settings, now_ms, fail_on_market_error=True,
            )
            market_cache.one_hour.update(one_hour)
            market_cache.current_hour.update(current_hour)
            market_cache.four_hour.update(four_hour)
            market_cache.regimes.update(regimes)
            market_cache.selected_symbols.update(symbols_to_refresh)
            if not hour_changed or hour_offset_ms >= 15_000 or market_cache.bars_hour_key is None:
                market_cache.bars_hour_key = hour_key
        one_hour = market_cache.one_hour
        current_hour = market_cache.current_hour
        four_hour = market_cache.four_hour
        regimes = market_cache.regimes
        tickers = {}
        for symbol in sorted(state.portfolio.positions):
            tickers[symbol] = exchange.fetch_ticker(symbol)
    elif intrabar_entries and _hotlist_cache_is_fresh(state, now):
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        hotlist = state.watchlist
        top_symbols = [row["symbol"] for row in hotlist]
        track = sorted(set(top_symbols) | set(state.portfolio.positions) | set(resume_symbols))
        one_hour, current_hour, four_hour, regimes = _fetch_selected_bars(
            exchange, track, settings, now_ms, fail_on_market_error=True,
        )
    else:
        markets, hotlist, one_hour, current_hour, tickers = _build_universe(
            exchange, settings, now, now_ms, fail_on_market_error=intrabar_entries,
            on_progress=position_monitor, retry_reads=is_demo,
        )
        top_symbols = [row["symbol"] for row in hotlist]
        track = sorted(set(top_symbols) | set(state.portfolio.positions) | set(resume_symbols))
        four_hour: dict[str, pd.DataFrame] = {}
        regimes: dict[str, pd.DataFrame] = {}
        for symbol in track:
            if position_monitor is not None:
                position_monitor()
            if symbol not in one_hour or (intrabar_entries and symbol not in current_hour):
                try:
                    rows1 = exchange.fetch_ohlcv(symbol, "1h", limit=60)
                    observation = _current_hour_observation(rows1, now_ms)
                    if observation is not None:
                        current_hour[symbol] = observation
                    closed1 = _closed(rows1, HOUR_MS, now_ms)
                    if closed1:
                        one_hour[symbol] = ohlcv_frame(closed1)
                except ccxt.BaseError as exc:
                    _raise_market_error_if_needed(exc, intrabar_entries)
                    pass
            try:
                if is_demo:
                    frame4 = _cached_four_hour(exchange, symbol, now_ms, closed_market_cache)
                else:
                    rows4 = exchange.fetch_ohlcv(symbol, "4h", limit=100)
                    closed4 = _closed(rows4, FOUR_H_MS, now_ms)
                    frame4 = ohlcv_frame(closed4) if closed4 else None
            except ccxt.BaseError as exc:
                _raise_market_error_if_needed(exc, intrabar_entries)
                continue
            if frame4 is None:
                continue
            four_hour[symbol] = frame4
            try:
                regimes[symbol] = classify_regimes(frame4, settings)
            except Exception:
                regimes[symbol] = None
            if position_monitor is not None:
                position_monitor()

    if state.execution_mode == "lead_usdm":
        hotlist = _filter_lead_candidates(hotlist, markets, execution)
    state.watchlist = hotlist
    if intrabar_entries and not _hotlist_cache_is_fresh(state, now):
        state.exit_management_by_symbol[HOTLIST_CACHE_KEY] = {"refreshed_at": now.isoformat()}

    top_symbols = [row["symbol"] for row in hotlist]
    decision_ms = (int(datetime.now(timezone.utc).timestamp() * 1000) if is_demo else now_ms)
    max_quote_age_ms = (DEMO_POSITION_QUOTE_MAX_AGE_MS if is_demo
                        else LIVE_TICKER_MAX_AGE_MS)
    current_prices = {
        symbol: _ticker_price(tickers, symbol, decision_ms, max_age_ms=max_quote_age_ms)
        for symbol in track
    }
    if is_demo:
        for symbol, position in state.portfolio.positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            meta = state.exit_management_by_symbol.get(symbol) or {}
            _update_demo_excursions(
                meta, position, price=price,
                observed_at=_ticker_observed_at(tickers, symbol),
            )
            state.exit_management_by_symbol[symbol] = meta
    if intrabar_entries:
        missing_price_symbols = [symbol for symbol in state.portfolio.positions
                                 if current_prices.get(symbol) is None]
        _record_live_stop_monitor_status(state, now, tickers=tickers, now_ms=decision_ms)
        if missing_price_symbols:
            _mark_market_data_unavailable(
                state, now, "ticker_missing_or_stale", reason="live_price_unavailable",
                tickers=tickers,
            )
            market_risk = MarketRiskAssessment(
                MarketRiskState.PAUSE_NEW, "live_price_unavailable", now,
                "BTC/USDT:USDT", "4h", {"missing_position_quotes": missing_price_symbols},
            )
        entry_wait_reason = "热度 Top20 候选；日线 EMA50 定方向，4H EMA50 偏离与已收盘 1H 反转确认；止损实时监控"
    else:
        entry_wait_reason = "热度 Top20 候选；开仓还需满足 EMA50 偏离与 1H 均值回归确认"

    for row in hotlist:
        symbol = row["symbol"]
        regime_value = "UNKNOWN"
        reg_frame = regimes.get(symbol)
        if reg_frame is not None and len(reg_frame):
            regime = reg_frame.iloc[-1]["regime"]
            if isinstance(regime, Regime):
                regime_value = regime.value
        _set_symbol_status(state, symbol, reason=entry_wait_reason, at=now,
                           regime=regime_value)

    if is_demo:
        if not _demo_sync_stopped(execution, state, now):
            logger.error("{} positions do not match the local ledger; entries paused",
                         _execution_label(execution, state))
            return
        if _demo_stop_audit_due(state, now):
            _audit_demo_protective_stops(execution, state, now)
        if position_monitor is not None:
            position_monitor()

    if intrabar_entries:
        for symbol, price in current_prices.items():
            if price is not None and symbol in state.portfolio.positions:
                state.mark_prices[symbol] = price
        _evaluate_live_stop_exits(state, execution, is_demo, current_prices,
                                  fee_rate, slip_rate, now)

    # find the latest closed 1H bar time across tracked symbols
    latest_bar = None
    for frame in one_hour.values():
        if frame is not None and len(frame):
            t = frame.index[-1]
            if latest_bar is None or t > latest_bar:
                latest_bar = t
    last_processed = state.exit_management_by_symbol.get(BAR_KEY)
    new_exit_bar = latest_bar is not None and str(latest_bar) != last_processed
    if new_exit_bar:
        # Exit counters and the existing exit ordering run only once per hour.
        state.exit_management_by_symbol[BAR_KEY] = str(latest_bar)
        _evaluate_exits(state, execution, is_demo, one_hour, regimes, settings,
                        fee_rate, slip_rate, now)
        mr_scan.checkpoint(checkpoint)
    scan = state.mr_entry_scan
    entry_due = (latest_bar is not None and
                 ((is_demo and (scan.get("bar") != latest_bar.isoformat()
                                or scan.get("status") not in {"completed", "expired"}))
                  or (not is_demo and new_exit_bar)))
    if entry_due:
        market_ready_at = _utc_now() if is_demo else now
        try:
            _evaluate_entries(exchange, state, execution, is_demo, markets, top_symbols, one_hour,
                              four_hour, regimes, settings, max_pos, fee_rate, slip_rate, now,
                              market_risk, daily_direction_gate=intrabar_entries,
                              position_monitor=position_monitor, closed_market_cache=closed_market_cache,
                              checkpoint=checkpoint)
        finally:
            if is_demo:
                finished_at = _utc_now()
                state.events.append({
                    "event": "entry_cycle_timing", "at": finished_at.isoformat(),
                    "source": MEAN_REVERSION, "cycle_started_at": now.isoformat(),
                    "market_ready_at": market_ready_at.isoformat(),
                    "entry_finished_at": finished_at.isoformat(),
                    "bar_close_at": (latest_bar + timedelta(hours=1)).isoformat(),
                    "status": state.mr_entry_scan.get("status", "partial"),
                    "runner_build_id": RUNNER_BUILD_ID,
                })
                mr_scan.checkpoint(checkpoint)
    elif is_demo and closed_market_cache is not None:
        # Warm tomorrow/next-hour candidates outside the entry scan. The daily
        # key expires at UTC midnight, so no unfinished daily bar is reused.
        for symbol in top_symbols:
            if position_monitor is not None:
                position_monitor()
            _cached_daily_direction_gate(
                exchange, symbol, now, settings.symbol_regime.ema_slow_bars,
                closed_market_cache,
            )

    # refresh marks
    for symbol in list(state.portfolio.positions):
        price = current_prices.get(symbol) if intrabar_entries else None
        frame = one_hour.get(symbol)
        if price is not None:
            state.mark_prices[symbol] = price
        elif frame is not None and len(frame):
            state.mark_prices[symbol] = float(frame["close"].iloc[-1])


def _evaluate_live_stop_exits(state, execution, is_demo, current_prices,
                              fee_rate, slip_rate, now):
    if is_demo:
        return
    for symbol, position in list(state.portfolio.positions.items()):
        price = current_prices.get(symbol)
        if price is None:
            continue
        meta = state.exit_management_by_symbol.get(symbol) or {}
        stop = float(meta.get("protective_stop", position.stop_price))
        hit = (position.side is Side.LONG and price <= stop) or (
            position.side is Side.SHORT and price >= stop
        )
        if hit:
            _close_position(False, None, state, symbol, price, "protective_stop",
                            now, fee_rate, slip_rate)
            logger.info("MR live protective stop {} {} price={} stop={}",
                        symbol, position.side.value, price, stop)


def _frame_latest_bar_at(frame):
    if frame is None or not len(frame):
        return None
    value = frame.index[-1]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _regime_fade_check_skipped_event(
    *, symbol, position, at, reason, one_hour_bar_open_at=None,
    in_hotlist=False, hotlist_rank=None,
):
    return {
        "event": "regime_fade_check_skipped",
        "at": at,
        "symbol": symbol,
        "side": position.side.value,
        "entry_regime": getattr(position.regime, "value", position.regime),
        "one_hour_bar_open_at": one_hour_bar_open_at,
        "in_hotlist": bool(in_hotlist),
        "hotlist_rank": hotlist_rank,
        "decision": "skipped",
        "skip_reason": reason,
    }


def _regime_fade_evaluation_event(
    *, symbol, side, entry_regime, current_regime, progress, entry_atr,
    min_progress_atr, at, one_hour_bar_open_at, regime_bar_open_at,
    in_hotlist, hotlist_rank,
):
    entry_regime = getattr(entry_regime, "value", entry_regime)
    current_regime_value = getattr(current_regime, "value", None)
    valid_metrics = (
        math.isfinite(float(entry_atr)) and float(entry_atr) > 0
        and math.isfinite(float(progress))
    )
    progress_atr = float(progress) / float(entry_atr) if valid_metrics else None
    entry_regime_matches = entry_regime == (
        Regime.LOW.value if side is Side.LONG else Regime.HIGH.value
    )
    progress_below_threshold = (
        float(progress) < float(min_progress_atr) * float(entry_atr)
        if valid_metrics else None
    )

    if not valid_metrics:
        triggered = False
        decision = "skipped"
        skip_reason = "invalid_progress_or_entry_atr"
    elif current_regime is None:
        triggered = False
        decision = "skipped"
        skip_reason = "regime_data_unavailable"
    else:
        triggered = regime_faded_triggered(
            side, entry_regime, current_regime, float(progress), float(entry_atr),
            min_atr=float(min_progress_atr),
        )
        decision = "triggered" if triggered else "not_triggered"
        skip_reason = None

    return {
        "event": "regime_fade_evaluated",
        "at": at,
        "symbol": symbol,
        "side": side.value,
        "one_hour_bar_open_at": one_hour_bar_open_at,
        "regime_bar_open_at": regime_bar_open_at,
        "entry_regime": entry_regime,
        "current_regime": current_regime_value,
        "entry_regime_matches": entry_regime_matches,
        "current_regime_is_sideways": current_regime is Regime.SIDEWAYS,
        "progress": float(progress),
        "entry_atr": float(entry_atr),
        "progress_atr": progress_atr,
        "min_progress_atr": float(min_progress_atr),
        "progress_below_threshold": progress_below_threshold,
        "triggered": bool(triggered),
        "decision": decision,
        "skip_reason": skip_reason,
        "in_hotlist": bool(in_hotlist),
        "hotlist_rank": hotlist_rank,
    }


def _evaluate_exits(state, execution, is_demo, one_hour, regimes, settings,
                    fee_rate, slip_rate, now):
    stops = settings.stops
    exits = settings.exits
    hotlist_ranks = {
        row.get("symbol"): row.get("rank")
        for row in state.watchlist
        if isinstance(row, dict) and row.get("symbol")
    }
    for symbol in list(state.portfolio.positions):
        position = state.portfolio.positions[symbol]
        frame = one_hour.get(symbol)
        if frame is None or len(frame) < 2:
            event = _regime_fade_check_skipped_event(
                symbol=symbol, position=position, at=now.isoformat(),
                reason="one_hour_data_unavailable",
                one_hour_bar_open_at=_frame_latest_bar_at(frame),
                in_hotlist=symbol in hotlist_ranks,
                hotlist_rank=hotlist_ranks.get(symbol),
            )
            state.events.append(event)
            logger.info("Regime fade check skipped {}", event)
            continue
        meta = state.exit_management_by_symbol.get(symbol) or {}
        meta.setdefault("enabled", True)
        bar = frame.iloc[-1]
        current_close = float(bar["close"])
        current_open = float(bar["open"])
        current_high = float(bar["high"])
        current_low = float(bar["low"])
        side = position.side
        entry = float(position.entry_price)
        entry_atr = float(meta.get("entry_atr", 0.0) or 0.0)
        initial_risk = float(meta.get("initial_risk", 0.0) or 0.0)
        if entry_atr <= 0 or initial_risk <= 0:
            event = _regime_fade_check_skipped_event(
                symbol=symbol, position=position, at=now.isoformat(),
                reason="invalid_entry_risk",
                one_hour_bar_open_at=_frame_latest_bar_at(frame),
                in_hotlist=symbol in hotlist_ranks,
                hotlist_rank=hotlist_ranks.get(symbol),
            )
            state.events.append(event)
            logger.info("Regime fade check skipped {}", event)
            continue
        if is_demo:
            _update_demo_excursions(meta, position, frame=frame)
        tracked_order = state.exchange_orders_by_symbol.get(symbol, {}).get("stop") if is_demo else None
        if is_demo and tracked_order and tracked_order.get("stop_price") is not None:
            tracked_stop = float(tracked_order["stop_price"])
            recorded_stop = float(meta.get("protective_stop", position.stop_price))
            current_stop = (max(tracked_stop, recorded_stop) if side is Side.LONG
                            else min(tracked_stop, recorded_stop))
        else:
            current_stop = float(meta.get("protective_stop", position.stop_price))
        meta["protective_stop"] = current_stop

        # 1. protective stop (local only; demo stop is on-exchange)
        if not is_demo:
            stop = current_stop
            if side is Side.LONG and current_low <= stop:
                gap = current_open <= stop
                event = _regime_fade_check_skipped_event(
                    symbol=symbol, position=position, at=now.isoformat(),
                    reason=("higher_priority_exit_stop_gap" if gap
                            else "higher_priority_exit_stop_hit"),
                    one_hour_bar_open_at=_frame_latest_bar_at(frame),
                    in_hotlist=symbol in hotlist_ranks,
                    hotlist_rank=hotlist_ranks.get(symbol),
                )
                state.events.append(event)
                logger.info("Regime fade check skipped {}", event)
                _close_position(is_demo, execution, state, symbol,
                                current_open if gap else stop,
                                "stop_gap" if gap else "stop_hit", now, fee_rate, slip_rate)
                continue
            if side is Side.SHORT and current_high >= stop:
                gap = current_open >= stop
                event = _regime_fade_check_skipped_event(
                    symbol=symbol, position=position, at=now.isoformat(),
                    reason=("higher_priority_exit_stop_gap" if gap
                            else "higher_priority_exit_stop_hit"),
                    one_hour_bar_open_at=_frame_latest_bar_at(frame),
                    in_hotlist=symbol in hotlist_ranks,
                    hotlist_rank=hotlist_ranks.get(symbol),
                )
                state.events.append(event)
                logger.info("Regime fade check skipped {}", event)
                _close_position(is_demo, execution, state, symbol,
                                current_open if gap else stop,
                                "stop_gap" if gap else "stop_hit", now, fee_rate, slip_rate)
                continue

        # 2. update best/worst
        best_price = float(meta.get("best_price", current_high if side is Side.LONG else current_low))
        worst_price = float(meta.get("worst_price", current_low if side is Side.LONG else current_high))
        best_price = max(best_price, current_high) if side is Side.LONG else min(best_price, current_low)
        worst_price = min(worst_price, current_low) if side is Side.LONG else max(worst_price, current_high)
        meta["best_price"] = best_price
        meta["worst_price"] = worst_price

        # 3. Bollinger mid exit
        mid = float(mid_band(frame["close"], MID_PERIOD, True).iloc[-1])
        ms = MidExitState(reached_mid=bool(meta.get("mid_reached")),
                          hold_count=int(meta.get("mid_hold", 0)),
                          fired=bool(meta.get("mid_fired")))
        fire, ms = mid_exit_check(side, entry, mid, current_close, buffer=0.0, hold_bars=0, state=ms)
        meta["mid_reached"] = ms.reached_mid
        meta["mid_hold"] = ms.hold_count
        meta["mid_fired"] = ms.fired
        if fire:
            event = _regime_fade_check_skipped_event(
                symbol=symbol, position=position, at=now.isoformat(),
                reason="higher_priority_exit_bollinger_mid_rejection",
                one_hour_bar_open_at=_frame_latest_bar_at(frame),
                in_hotlist=symbol in hotlist_ranks,
                hotlist_rank=hotlist_ranks.get(symbol),
            )
            state.events.append(event)
            _close_position(is_demo, execution, state, symbol, current_close,
                            "bollinger_mid_rejection", now, fee_rate, slip_rate)
            continue

        # 4. structure invalid
        if structure_invalid_triggered(side, current_close, meta.get("structure_low"),
                                       meta.get("structure_high"), entry_atr,
                                       buffer_atr=exits.structure_invalid_buffer_atr):
            reason = "long_structure_invalid" if side is Side.LONG else "short_structure_invalid"
            event = _regime_fade_check_skipped_event(
                symbol=symbol, position=position, at=now.isoformat(),
                reason=f"higher_priority_exit_{reason}",
                one_hour_bar_open_at=_frame_latest_bar_at(frame),
                in_hotlist=symbol in hotlist_ranks,
                hotlist_rank=hotlist_ranks.get(symbol),
            )
            state.events.append(event)
            _close_position(is_demo, execution, state, symbol, current_close, reason,
                            now, fee_rate, slip_rate)
            continue

        # 5. regime fade
        reg_frame = regimes.get(symbol)
        current_regime = None
        if reg_frame is not None and len(reg_frame):
            row = reg_frame.iloc[-1]["regime"]
            current_regime = row if isinstance(row, Regime) else None
        progress = (current_close - entry) if side is Side.LONG else (entry - current_close)
        fade_event = _regime_fade_evaluation_event(
            symbol=symbol,
            side=side,
            entry_regime=position.regime,
            current_regime=current_regime,
            progress=progress,
            entry_atr=entry_atr,
            min_progress_atr=exits.regime_fade_min_atr,
            at=now.isoformat(),
            one_hour_bar_open_at=_frame_latest_bar_at(frame),
            regime_bar_open_at=_frame_latest_bar_at(reg_frame),
            in_hotlist=symbol in hotlist_ranks,
            hotlist_rank=hotlist_ranks.get(symbol),
        )
        state.events.append(fade_event)
        logger.info("Regime fade evaluation {}", fade_event)
        if fade_event["triggered"]:
            reason = "long_regime_faded" if side is Side.LONG else "short_regime_faded"
            attempt_event = {
                "event": "regime_fade_exit_attempted",
                "at": now.isoformat(),
                "symbol": symbol,
                "side": side.value,
                "reason": reason,
                "one_hour_bar_open_at": fade_event["one_hour_bar_open_at"],
                "regime_bar_open_at": fade_event["regime_bar_open_at"],
                "progress_atr": fade_event["progress_atr"],
                "min_progress_atr": fade_event["min_progress_atr"],
                "in_hotlist": fade_event["in_hotlist"],
                "hotlist_rank": fade_event["hotlist_rank"],
            }
            state.events.append(attempt_event)
            logger.warning("Regime fade exit attempted {}", attempt_event)
            _close_position(is_demo, execution, state, symbol, current_close, reason,
                            now, fee_rate, slip_rate)
            if symbol not in state.portfolio.positions:
                logger.info("Regime fade exit fill recorded for {} reason={}", symbol, reason)
            continue

        # 6. no progress
        favorable = (best_price - entry) if side is Side.LONG else (entry - best_price)
        trigger, count = no_progress_update(side, current_close, float(meta.get("last_close", current_close)),
                                            int(meta.get("no_progress_count", 0)), favorable, entry_atr,
                                            bars=exits.no_progress_bars, min_atr=exits.no_progress_min_atr)
        meta["no_progress_count"] = count
        meta["last_close"] = current_close
        if trigger:
            _close_position(is_demo, execution, state, symbol, current_close, "no_progress",
                            now, fee_rate, slip_rate)
            continue

        # 7. max hold for local Paper; Demo uses the independent fast monitor.
        if not is_demo and position.opened_at is not None:
            held_hours = (now - position.opened_at).total_seconds() / 3600
            if held_hours >= exits.max_hold_bars:
                _close_position(is_demo, execution, state, symbol, current_close, "max_hold_24h",
                                now, fee_rate, slip_rate)
                continue

        if is_demo:
            state.exit_management_by_symbol[symbol] = meta
            continue

        # Staged profit locks + trailing (local Paper only; Demo uses fresh quotes).
        new_stop = staged_profit_lock_stop(
            side, entry, entry_atr, initial_risk, favorable, current_stop,
            first_trigger_atr=stops.move_to_breakeven_atr,
            first_lock_r=settings.mr_candidate_exit.first_lock_r,
            second_trigger_r=settings.mr_candidate_exit.second_lock_trigger_r,
            second_lock_r=settings.mr_candidate_exit.second_lock_r,
            fee_rate=fee_rate, slip_rate=slip_rate,
        )
        progress_r = favorable / initial_risk if initial_risk > 0 else 0.0
        new_stop = trailing_stop(side, best_price, initial_risk, progress_r, new_stop,
                                 trigger_r=settings.mr_candidate_exit.trailing_trigger_r,
                                 atr_mult=settings.exit_management.trailing_atr_mult,
                                 stop_atr_mult=settings.signals.stop_atr_mult)
        if abs(new_stop - current_stop) > 1e-12:
            meta["protective_stop"] = new_stop

        state.exit_management_by_symbol[symbol] = meta


def _entry_candidate(exchange, state, is_demo, symbol, one_hour, four_hour, regimes,
                     settings, now, structure_reentry_states, *, daily_direction_gate=False,
                     position_monitor=None, closed_market_cache=None, scan_row=None):
    span = settings.symbol_regime.ema_slow_bars
    if position_monitor is not None:
        position_monitor()
    if symbol in state.portfolio.positions:
        return None
    frame1 = one_hour.get(symbol)
    frame4 = four_hour.get(symbol)
    if frame1 is None or frame4 is None or len(frame1) < settings.lookback.atr_len + 1 or len(frame4) < span:
        return None
    evaluated_at = _utc_now() if is_demo else now
    max_delay = (timedelta(seconds=settings.execution.confirm_delay_sec_max)
                 if is_demo else ENTRY_BAR_MAX_DELAY)
    bar_close_time, bar_rejection = _entry_bar_close_time(
        frame1, evaluated_at, max_delay=max_delay,
    )
    if bar_rejection is not None:
        _record_entry_bar_rejection(state, symbol, frame1, bar_rejection, evaluated_at)
        return None
    try:
        validate_ohlcv(frame1)
    except (TypeError, ValueError):
        _record_entry_bar_rejection(state, symbol, frame1, "invalid_ohlcv", now)
        return None
    atr_series = atr(frame1, settings.lookback.atr_len)
    try:
        flags = evaluate_time_filter(bar_close_time, frame1, settings, precomputed_atr=atr_series)
    except Exception as exc:
        logger.exception("MR time-filter failed closed for {}: {}", symbol, type(exc).__name__)
        flags = TimeFilterFlags(
            False, False, False, False,
            data_ready=False, data_reason=f"time_filter_error:{type(exc).__name__}",
        )
    candle = frame1.iloc[-1]
    if is_demo:
        _record_structure_reentry_confirmations(
            state, symbol, frame1, evaluated_at, structure_reentry_states,
        )
    candidate_flags = replace(flags, allow_long_entry=True, allow_short_entry=True,
                               flatten_only=False, data_ready=True, data_reason=None)
    candidate_signal = deviation_entry_signal(
        frame4["close"], candle, settings, candidate_flags, deviation_pct=DEV_PCT,
    )
    if candidate_signal.action is not Action.OPEN:
        return None
    if scan_row is not None:
        scan_row["side"] = candidate_signal.side.value
    if not _record_time_filter_decision(
            state, symbol, candidate_signal.side, frame1, bar_close_time, flags,
            settings, evaluated_at):
        return None
    sig = deviation_entry_signal(frame4["close"], candle, settings, flags,
                                 deviation_pct=DEV_PCT)
    if sig.action is not Action.OPEN:
        return None
    entry_atr = float(atr_series.iloc[-1])
    close = float(candle["close"])
    swing_low = float(frame1["low"].tail(settings.stops.swing_lookback_1h).min())
    swing_high = float(frame1["high"].tail(settings.stops.swing_lookback_1h).max())
    structure = swing_low if sig.side is Side.LONG else swing_high
    try:
        stop, _mult = calculate_adaptive_stop(sig.side, close, entry_atr, structure, settings)
    except ValueError:
        return None
    closed_ema = float(frame4["close"].ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1])
    dev = close / closed_ema - 1.0
    strength = -dev if sig.side is Side.LONG else dev
    reg_frame = regimes.get(symbol)
    regime_value = Regime.MID.value
    if reg_frame is not None and len(reg_frame):
        row = reg_frame.iloc[-1]["regime"]
        if isinstance(row, Regime):
            regime_value = row.value
    daily_gate = None
    if is_demo or daily_direction_gate:
        if scan_row is not None:
            scan_row.update(phase="daily_gate", endpoint="public", side=sig.side.value)
        daily_gate = _cached_daily_direction_gate(
            exchange, symbol, bar_close_time, span, closed_market_cache if is_demo else None,
            fail_on_market_error=(is_demo or daily_direction_gate),
        )
        daily_reason = _daily_direction_rejection_reason(daily_gate, sig.side)
        if daily_reason is not None:
            _record_daily_entry_block(state, symbol, sig.side, regime_value,
                                      daily_gate, daily_reason, now)
            return None
    if is_demo:
        failures, latest_loss_at, waiting = structure_reentry_states.get(
            (symbol, sig.side), (0, None, False),
        )
        if waiting:
            _record_structure_reentry_block(
                state, symbol, sig.side, bar_close_time, now,
                failures, latest_loss_at,
            )
            return None
    context = EntrySignalContext(
        symbol, sig.side, bar_close_time, now, evaluated_at,
        float(candle["open"]), close, closed_ema, entry_atr, swing_low, swing_high,
    ) if is_demo else None
    return (symbol, strength, sig.side, close, stop, entry_atr,
                       swing_low, swing_high, regime_value, sig.reason, daily_gate,
                       context, flags)


def _initialize_entry_metadata(state, symbol, opened, is_demo, close, stop,
                               entry_atr, swing_low, swing_high, *, observed_price=None,
                               observed_at=None, telemetry_history_complete=True):
    initial_risk = (abs(opened.entry_price - opened.stop_price)
                    if is_demo else abs(close - stop))
    state.exit_management_by_symbol[symbol] = {
        "entry_atr": entry_atr,
        "enabled": True,
        "protective_stop": float(opened.stop_price),
        "structure_low": swing_low,
        "structure_high": swing_high,
        "initial_risk": initial_risk,
        "best_price": (opened.entry_price if is_demo and observed_price is None
                       else close if observed_price is None else observed_price),
        "worst_price": (opened.entry_price if is_demo and observed_price is None
                        else close if observed_price is None else observed_price),
        "last_close": close,
        "no_progress_count": 0,
        "mid_reached": False,
        "mid_hold": 0,
        "mid_fired": False,
    }
    if is_demo:
        state.exit_management_by_symbol[symbol].update({
            "telemetry_enabled": True,
            "telemetry_initial_stop_price": float(opened.stop_price),
            "telemetry_best_price": float(opened.entry_price if observed_price is None else observed_price),
            "telemetry_worst_price": float(opened.entry_price if observed_price is None else observed_price),
            "telemetry_history_complete": bool(telemetry_history_complete),
            "telemetry_bar_count": 0,
            "telemetry_sample_count": 0,
            "telemetry_last_sample_at": (observed_at or opened.opened_at).isoformat(),
        })


def _submit_entry_candidate(exchange, state, execution, is_demo, markets, candidate,
                            settings, max_pos, fee_rate, slip_rate, now, market_risk, *,
                            position_monitor=None, scan_row=None, checkpoint=None):
    (symbol, _strength, side, close, stop, entry_atr, swing_low, swing_high,
     regime_value, reason, daily_gate, context, flags, funding_rate) = candidate
    max_delay = timedelta(seconds=settings.execution.confirm_delay_sec_max)
    market_risk_state = market_risk.state.value
    market_risk_multiplier = (settings.market_risk.elevated_size_multiplier
                              if market_risk.state is MarketRiskState.RISK_ELEVATED else 1.0)
    if (state.execution_mode == "lead_usdm"
            and str((markets.get(symbol) or {}).get("id", "")).upper()
            not in getattr(execution, "lead_market_ids", set())):
        return None
    if is_demo and len(state.portfolio.positions) >= max_pos:
        return None
    if position_monitor is not None:
        position_monitor()
    if is_demo:
        if state.execution_status != "synchronized" or mr_scan.unresolved_attempts(state):
            return None
        if _utc_now() > context.bar_close_at + max_delay:
            state.events.append({"event": "entry_bar_rejected", "symbol": symbol,
                                 "at": _utc_now().isoformat(), "reason": "hourly_bar_stale"})
            return None
        funding_rate = _fetch_entry_funding_rate(exchange, symbol, fail_on_market_error=True)
        funding_reason = funding_rate_rejection_reason(funding_rate, settings)
        if funding_reason is not None:
            _record_funding_entry_block(state, symbol, side, regime_value,
                                        funding_rate, funding_reason, settings, _utc_now())
            return None
    step_size = 1e-8
    minimum_quantity = 0.0
    if is_demo:
        step_size, minimum_quantity = _entry_order_limits(markets.get(symbol) or {}, close)
    target_margin = _target_margin_per_symbol(state, settings)
    qty = calculate_order_quantity(state.current_equity, close, state.portfolio, settings,
                                   step_size=step_size, minimum_quantity=minimum_quantity,
                                   size_multiplier=market_risk_multiplier,
                                   stop_price=stop,
                                   gross_notional_cap_pct=(settings.sizing.max_gross_notional_pct
                                                           * market_risk_multiplier),
                                   target_margin_usdt=target_margin)
    if qty <= 0:
        return None
    intent = OrderIntent(symbol, side, Action.OPEN, qty, close, reason, Regime(regime_value),
                         stop, MEAN_REVERSION)
    approval = assess_entry(intent, state.portfolio, state.current_equity,
                             funding_rate, now, settings,
                             gross_notional_cap_pct=(settings.sizing.max_gross_notional_pct
                                                     * market_risk_multiplier),
                             market_risk_state=market_risk_state,
                             per_symbol_margin_usdt=target_margin)
    if not approval.allowed:
        return None
    execution_kwargs = {}
    if is_demo:
        if scan_row is not None:
            scan_row.update(phase="demo_prepare", endpoint="demo")
        audit = {}
        guard, deadline = _entry_preflight(
            exchange, execution, state, settings, markets.get(symbol) or {},
            context, flags, funding_rate, market_risk, audit,
            propagate_read_errors=scan_row is not None,
            scan_row=scan_row,
        )
        execution_kwargs = {"entry_preflight": guard, "submission_deadline": deadline,
                            "entry_audit": audit}
        if scan_row is not None:
            execution_kwargs.update(scan_row=scan_row, checkpoint=checkpoint,
                                    recovery_meta={"entry_atr": entry_atr, "structure_low": swing_low,
                                                   "structure_high": swing_high, "last_close": close})
    opened = _open_position(is_demo, execution, state, symbol, side, close, stop, qty,
                            int(settings.sizing.leverage), fee_rate, slip_rate, reason,
                            regime_value, now, **execution_kwargs)
    if opened is not None:
        _initialize_entry_metadata(state, symbol, opened, is_demo, close, stop,
                                   entry_atr, swing_low, swing_high)
        if daily_gate is not None:
            _set_symbol_status(state, symbol, action="OPEN", side=side,
                               reason=reason, at=now, regime=regime_value)
            state.symbol_status[symbol].update(_daily_gate_status_fields(daily_gate))
        logger.info("MR open {} {} qty={} entry={} stop={} reason={}",
                    symbol, side.value, opened.quantity, round(opened.entry_price, 6),
                    round(float(opened.stop_price), 6), reason)

    return opened


def _evaluate_entries(exchange, state, execution, is_demo, markets, top_symbols, one_hour, four_hour,
                      regimes, settings, max_pos, fee_rate, slip_rate, now,
                      market_risk, *, daily_direction_gate=False, position_monitor=None,
                      closed_market_cache=None, checkpoint=None):
    if not is_demo:
        return _evaluate_local_entries(
            exchange, state, execution, is_demo, markets, top_symbols, one_hour, four_hour,
            regimes, settings, max_pos, fee_rate, slip_rate, now, market_risk,
            daily_direction_gate=daily_direction_gate, position_monitor=position_monitor,
            closed_market_cache=closed_market_cache,
        )
    bars = [frame.index[-1] for frame in one_hour.values() if frame is not None and len(frame)]
    if not bars:
        return
    scan = mr_scan.scan_for(state, max(bars), top_symbols,
                            settings.execution.confirm_delay_sec_max, checkpoint)
    if scan["status"] in {"completed", "expired"}:
        return
    _reconcile_structure_invalidation_cooldowns(state)
    reentry = _structure_reentry_states(state.events, scan["symbols"])

    def finish(symbol, status, **details):
        return mr_scan.outcome(state, scan, symbol, status, _utc_now(), checkpoint, **details)

    def failure(symbol, exc):
        row = scan["outcomes"][symbol]
        uncertain = bool(mr_scan.unresolved_attempts(state))
        retry = mr_scan.retryable_read(exc) and not uncertain and row["attempts"] < 2
        status = "uncertain" if uncertain else "retry" if retry else "error"
        finish(symbol, status, error=type(exc).__name__,
               order_may_have_been_sent=uncertain or row.get("order_may_have_been_sent", False))
        logger.warning("MR scan symbol={} phase={} endpoint={} error={} may_sent={} status={}",
                       symbol, row["phase"], row["endpoint"], type(exc).__name__,
                       row["order_may_have_been_sent"], status)
        if uncertain:
            state.execution_status = "entry_reconciliation_required"
            return False
        # Authentication, rate limits, account conflicts and programming errors
        # stop this scan. Only known symbol failures and bounded reads continue.
        return mr_scan.symbol_error(exc) or mr_scan.retryable_read(exc)

    try:
        if (state.execution_status != "synchronized" or mr_scan.unresolved_attempts(state)):
            scan["status"] = "paused"
            return
        if market_risk.state is MarketRiskState.PAUSE_NEW:
            for symbol in scan["symbols"]:
                if symbol not in scan["outcomes"]:
                    finish(symbol, "blocked", reason="market_risk_pause")
            scan["status"] = "completed"
            return
        # First attempt every candidate, then retry only unfinished reads once.
        for _ in range(2):
            candidates = []
            for symbol in scan["symbols"]:
                row = scan["outcomes"].get(symbol)
                if row is not None and row.get("status") not in {"retry", "evaluating", "candidate"}:
                    continue
                if row is not None and row.get("attempts", 0) >= 2:
                    finish(symbol, "error", reason="read_retry_exhausted")
                    continue
                if mr_scan.expired(scan, _utc_now()):
                    finish(symbol, "expired", reason="original_signal_deadline")
                    continue
                if len(state.portfolio.positions) >= max_pos:
                    finish(symbol, "blocked", reason="position_limit")
                    continue
                attempts = (row or {}).get("attempts", 0) + 1
                row = finish(symbol, "evaluating", attempts=attempts,
                             phase="candidate", endpoint="public", order_may_have_been_sent=False)
                try:
                    candidate = _entry_candidate(
                        exchange, state, True, symbol, one_hour, four_hour, regimes, settings,
                        now, reentry, position_monitor=position_monitor,
                        closed_market_cache=closed_market_cache, scan_row=row,
                    )
                    if candidate is None:
                        finish(symbol, "evaluated", reason="no_eligible_signal")
                    else:
                        finish(symbol, "candidate", side=candidate[2].value)
                        candidates.append(candidate)
                except mr_scan.ScanCheckpointError:
                    raise
                except Exception as exc:
                    if not failure(symbol, exc):
                        scan["status"] = "paused"
                        return
            candidates.sort(key=lambda candidate: -candidate[1])
            for candidate in candidates:
                symbol = candidate[0]
                row = scan["outcomes"][symbol]
                if (state.execution_status != "synchronized" or mr_scan.unresolved_attempts(state)):
                    scan["status"] = "paused"
                    return
                if mr_scan.expired(scan, _utc_now()):
                    finish(symbol, "expired", reason="original_signal_deadline")
                    continue
                row.update(phase="funding_and_risk", endpoint="public")
                try:
                    opened = _submit_entry_candidate(
                        exchange, state, execution, True, markets, (*candidate, None), settings,
                        max_pos, fee_rate, slip_rate, now, market_risk,
                        position_monitor=position_monitor, scan_row=row, checkpoint=checkpoint,
                    )
                    if opened is not None:
                        client_id = mr_scan.client_order_id(symbol, candidate[2].value, scan["bar"])
                        state.mr_entry_attempts[client_id]["status"] = "completed"
                        mr_scan.attempt_outcome(
                            state, state.mr_entry_attempts[client_id], "completed", _utc_now(),
                            phase="fill_confirmed", order_may_have_been_sent=True,
                        )
                    finish(symbol, "opened" if opened is not None else "rejected")
                except mr_scan.ScanCheckpointError:
                    raise
                except Exception as exc:
                    if not failure(symbol, exc):
                        scan["status"] = "paused"
                        return
        scan["status"] = ("expired" if any(row["status"] == "expired"
                                          for row in scan["outcomes"].values()) else "completed")
    finally:
        state.events.append({"event": "entry_scan_finished", "at": _utc_now().isoformat(),
                             "bar": scan["bar"], "deadline": scan["deadline"],
                             "status": scan["status"], "symbols": len(scan["symbols"]),
                             "outcomes": {s: r["status"] for s, r in scan["outcomes"].items()}})
        mr_scan.checkpoint(checkpoint)


def _evaluate_local_entries(exchange, state, execution, is_demo, markets, top_symbols, one_hour, four_hour,
                      regimes, settings, max_pos, fee_rate, slip_rate, now,
                      market_risk: MarketRiskAssessment, *, daily_direction_gate=False,
                      position_monitor=None, closed_market_cache=None):
    structure_reentry_states = {}
    if market_risk.state is MarketRiskState.PAUSE_NEW:
        logger.warning("New entries paused by BTC market risk gate: {}", market_risk.reason)
        return

    open_count = len(state.portfolio.positions)
    slots = max(0, max_pos - open_count)
    if slots <= 0:
        return
    candidates = []
    for symbol in top_symbols:
        candidate = _entry_candidate(
            exchange, state, is_demo, symbol, one_hour, four_hour, regimes, settings, now,
            structure_reentry_states, daily_direction_gate=daily_direction_gate,
            position_monitor=position_monitor, closed_market_cache=closed_market_cache,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda x: -x[1])
    eligible_candidates = []
    for candidate in candidates:
        if position_monitor is not None:
            position_monitor()
        if len(eligible_candidates) >= slots:
            break
        (symbol, _strength, side, _close, _stop, _entry_atr, _low, _high,
         regime_value, _reason, _daily_gate, _context, _flags) = candidate
        funding_rate = _fetch_entry_funding_rate(
            exchange, symbol, fail_on_market_error=daily_direction_gate,
        )
        funding_reason = funding_rate_rejection_reason(funding_rate, settings)
        if funding_reason is not None:
            _record_funding_entry_block(state, symbol, side, regime_value,
                                        funding_rate, funding_reason, settings, now)
            continue
        eligible_candidates.append((*candidate, funding_rate))

    for candidate in eligible_candidates:
        _submit_entry_candidate(exchange, state, execution, is_demo, markets, candidate,
                                settings, max_pos, fee_rate, slip_rate, now, market_risk,
                                position_monitor=position_monitor)


def main() -> int:
    sys.excepthook = redacted_excepthook
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=None,
                        help="poll interval in seconds (default: 60 for exchange modes, 300 for local paper)")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="run on Binance USDT-M testnet demo (default: local paper, no key)")
    parser.add_argument("--live", action="store_true",
                        help="run the 8766 strategy on a Binance Lead Portfolio account")
    parser.add_argument("--adopt-manual-positions", action="store_true",
                        help="Lead only: reconstruct and manage untracked positions with the existing strategy")
    parser.add_argument("--background-confirmed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--intrabar-entries", action="store_true",
                        help="local paper only: closed-1H entries with live-price stop monitoring")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args()
    if args.poll is not None and args.poll < 1:
        parser.error("--poll must be at least one second")
    if args.demo and args.live:
        parser.error("--demo and --live cannot be combined")
    if args.adopt_manual_positions and not args.live:
        parser.error("--adopt-manual-positions requires --live")
    if args.intrabar_entries and (args.demo or args.live):
        parser.error("--intrabar-entries is local-paper only")

    live_confirmed = _confirm_live(args, parser)
    if args.live and not live_confirmed:
        print("未确认，已退出；未连接账户。")
        return 2

    settings = load_settings()
    mode = resolve_execution_mode(bot_mode=settings.runtime.bot_mode,
                                  testnet=(settings.runtime.binance_testnet if args.live else args.demo),
                                  live_flag=args.live,
                                  live_confirmed=live_confirmed)
    if args.poll is None:
        args.poll = _default_poll_for_mode(mode.name)

    def path(value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    default_state = (Path("data/live/lead_portfolio.json") if args.live
                     else Path("data/paper/mr_candidate_portfolio.json"))
    state_path = path(args.state or default_state).resolve()
    if args.live and state_path.is_relative_to(PROJECT_ROOT / "data" / "paper"):
        parser.error("--live requires a separate state file outside data/paper")
    state_path_existed = state_path.exists()

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    log_name = ("mr_lead_live.log" if args.live else
                "mr_intrabar_paper.log" if args.intrabar_entries else "mr_candidate_paper.log")
    logger.add(log_dir / log_name, rotation="10 MB", retention=7,
               enqueue=True, level="INFO", backtrace=False, diagnose=False,
               filter=redact_log_record)
    pid_name = ("ema50_lead_live.pid" if args.live else
                "ema50_intrabar_paper.pid" if args.intrabar_entries else "mr_candidate_paper.pid")
    pid_dir = "live" if args.live else "paper"
    pid_path = PROJECT_ROOT / "data" / pid_dir / pid_name
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    is_exchange_execution = mode.name in {"demo_usdm", "lead_usdm"}
    is_demo = is_exchange_execution

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    account_exchange = None
    execution = None
    try:
        state = MultiPaperState.load(state_path, initial_equity=args.equity)
        if state_path_existed and state.execution_mode != mode.name:
            raise RuntimeError(
                f"state file execution mode is {state.execution_mode}; refusing to reuse it for {mode.name}"
            )
        if not state_path_existed:
            state.execution_mode = mode.name
        checkpoint = _RunnerCheckpoint(state, state_path)
        if is_exchange_execution:
            if mode.name == "lead_usdm":
                account_exchange = create_lead_usdm(settings.runtime)
                execution = LeadPortfolioExecutionService(account_exchange, settings)
                execution.adopt_manual_positions = bool(args.adopt_manual_positions)
            else:
                account_exchange = create_demo_usdm(settings.runtime)
                execution = DemoExecutionService(account_exchange, settings)
            try:
                state.exchange_snapshot = _initialize_demo(execution)
            except ccxt.BaseError as exc:
                state.events.append({"event": "account_startup_failed", "at": _utc_now().isoformat(),
                                     "phase": "account_initialization", "endpoint": mode.name,
                                     "error": type(exc).__name__, "order_may_have_been_sent": False})
                if state_path_existed:
                    checkpoint.try_save()
                logger.error("{} startup failed before entry processing: {}", mode.name,
                             type(exc).__name__)
                return 1
            try:
                entries_reconciled = _reconcile_pending_entries(execution, state, checkpoint)
            except mr_scan.ScanCheckpointError:
                entries_reconciled = False
            if entries_reconciled and not _demo_sync_stopped(execution, state, datetime.now(timezone.utc)):
                checkpoint.try_save()
                raise RuntimeError(f"{mode.name} positions do not match the local ledger; refusing to start")
            if not state_path_existed:
                state.initial_equity = float(state.exchange_snapshot["equity"])
                state.cash = state.initial_equity

        candidate_variant = _strategy_variant_for_mode(
            mode.name, intrabar_entries=args.intrabar_entries,
        )
        if state.strategy_variant != candidate_variant:
            state.events.append({"event": "strategy_variant_changed",
                                 "at": datetime.now(timezone.utc).isoformat(),
                                 "previous": state.strategy_variant,
                                 "current": candidate_variant})
            state.strategy_variant = candidate_variant
        checkpoint.try_save()

        logger.info("Starting MR candidate: mode={} strategy={} max_positions={} risk_per_trade_pct={} poll={} intrabar_entries={} adopt_manual_positions={} runner_build_id={} time_filter_build_id={} state={}",
                    mode.name, candidate_variant, args.max_positions,
                    settings.risk.max_single_trade_risk_pct * 100, args.poll,
                    args.intrabar_entries, bool(args.adopt_manual_positions),
                    RUNNER_BUILD_ID, TIME_FILTER_BUILD_ID, state_path)
        market_cache = IntrabarMarketCache() if args.intrabar_entries else None
        closed_market_cache = ClosedMarketCache() if is_exchange_execution else None
        retry_after = _persisted_market_retry_after(state) if args.intrabar_entries else None
        market_failure_attempt = 0
        last_position_monitor_at = 0.0

        def monitor_demo_positions(*, force=False):
            nonlocal last_position_monitor_at
            if not is_demo or not state.portfolio.positions:
                return False
            monotonic_now = time.monotonic()
            if (not force and monotonic_now - last_position_monitor_at
                    < DEMO_POSITION_MONITOR_POLL_SECONDS):
                return False
            last_position_monitor_at = monotonic_now
            monitor_at = datetime.now(timezone.utc)
            try:
                dirty = _manage_demo_positions(
                    exchange, execution, state, settings, monitor_at,
                    settings.fees.taker_bps / 10_000,
                    settings.fees.slippage_bps / 10_000,
                )
            except Exception as exc:
                reason = f"position monitor failed: {type(exc).__name__}"
                state.execution_status = "inconsistent"
                state.inconsistencies = [reason]
                state.protective_stop_audit = {
                    "at": monitor_at.isoformat(), "status": "error",
                    "position_count": len(state.portfolio.positions),
                    "mismatch_count": 1, "mismatches": [reason],
                }
                for symbol in state.portfolio.positions:
                    meta = state.exit_management_by_symbol.setdefault(symbol, {})
                    monitor = meta.setdefault("position_monitor", {})
                    previous_status = monitor.get("quote_status")
                    monitor.update({
                        "checked_at": monitor_at.isoformat(),
                        "quote_status": f"monitor_error:{type(exc).__name__}",
                    })
                    if previous_status != monitor["quote_status"]:
                        state.events.append({
                            "event": "demo_position_quote_status", "at": monitor_at.isoformat(),
                            "symbol": symbol, "status": monitor["quote_status"],
                            "previous_status": previous_status,
                            "execution_mode": state.execution_mode,
                        })
                logger.exception("{} position monitor failed: {}",
                                 _execution_label(execution, state), type(exc).__name__)
                dirty = True
            if dirty:
                state.record_equity(datetime.now(timezone.utc))
                checkpoint.try_save()
            return dirty

        while True:
            now = datetime.now(timezone.utc)
            if checkpoint.failed:
                monitor_demo_positions(force=True)
                if not checkpoint.try_save():
                    logger.info("MR state save pending; new entries remain paused")
                    if args.once:
                        return 1
                    _wait_with_monitor(min(float(args.poll), 30.0),
                                       monitor_demo_positions if is_demo else None)
                    continue
            if args.intrabar_entries and retry_after is not None and now < retry_after:
                state.record_equity(now)
                checkpoint.try_save()
                remaining = max(1.0, (retry_after - now).total_seconds())
                time.sleep(min(float(args.poll), remaining))
                continue
            cycle_completed = False
            try:
                monitor_demo_positions(force=True)
                if checkpoint.failed:
                    raise mr_scan.ScanCheckpointError("MR state storage unavailable")
                _cycle(exchange, execution, state, settings, now, is_demo, args.max_positions,
                       intrabar_entries=args.intrabar_entries, market_cache=market_cache,
                       position_monitor=monitor_demo_positions if is_demo else None,
                       closed_market_cache=closed_market_cache, checkpoint=checkpoint)
                monitor_demo_positions()
                market_failure_attempt = 0
                retry_after = None
                cycle_completed = True
            except ccxt.BaseError as exc:
                if is_demo:
                    state.events.append({"event": "entry_cycle_failed", "at": _utc_now().isoformat(),
                                         "cycle_started_at": now.isoformat(), "status": "partial",
                                         "phase": getattr(exc, "mr_phase", "outside_candidate_scan"),
                                         "endpoint": getattr(exc, "mr_endpoint", "unidentified"),
                                         "symbol": getattr(exc, "mr_symbol", None),
                                         "error": type(exc).__name__,
                                         "order_may_have_been_sent": bool(mr_scan.unresolved_attempts(state))})
                if args.intrabar_entries:
                    market_failure_attempt += 1
                    failed_at = datetime.now(timezone.utc)
                    retry_delay = _market_retry_delay_seconds(
                        exc, poll=args.poll, attempt=market_failure_attempt, now=failed_at,
                    )
                    retry_after = failed_at + timedelta(seconds=retry_delay)
                    _mark_market_data_unavailable(
                        state, failed_at, exc, retry_after=retry_after,
                    )
                    logger.warning(
                        "public market data failed; requests paused until {} ({}s): {}",
                        retry_after.isoformat(), int(retry_delay), type(exc).__name__,
                    )
                else:
                    logger.warning("MR cycle failed phase={} endpoint={} error={}",
                                   getattr(exc, "mr_phase", "outside_candidate_scan"),
                                   getattr(exc, "mr_endpoint", "unidentified"), type(exc).__name__)
            except mr_scan.ScanCheckpointError:
                logger.error("MR entry checkpoint failed; entries paused until storage recovers")
            if is_demo and not checkpoint.failed:
                _reconcile_demo_funding(execution, state, _utc_now(),
                                       position_monitor=monitor_demo_positions)
            state.record_equity(now)
            saved = checkpoint.try_save()
            logger.info("MR candidate equity={} open_positions={}",
                        round(state.current_equity, 4), len(state.portfolio.positions))
            if args.once:
                return 0 if saved and cycle_completed else 1
            if retry_after is None:
                poll_delay = _next_poll_delay(_utc_now(), args.poll, align_hourly=is_demo,
                                              cycle_started_at=now)
                _wait_with_monitor(poll_delay, monitor_demo_positions if is_demo else None)
            else:
                remaining = max(1.0, (retry_after - datetime.now(timezone.utc)).total_seconds())
                time.sleep(min(float(args.poll), remaining))
    finally:
        exchange.close()
        if account_exchange is not None:
            account_exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
