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
from types import SimpleNamespace
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import ccxt
import pandas as pd
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import (filter_tradable_crypto_universe, hot_activity_metrics,
                             ohlcv_frame, rank_mean_reversion_candidates,
                             score_hot_candidates)
from bot.exchange.binance_usdm import (create_demo_usdm, create_lead_usdm,
                                       resolve_execution_mode)
from bot.exchange.network import configure_system_proxy, rate_limit_cooldown_remaining
from bot.exchange.clock import ExchangeClock, ExchangeClockError, active_clock, activate_clock
from bot.exchange.public_candles import CandleBatchReader, prefetched_read
from bot.entry_quality import EntrySignalContext, entry_liquidity_observation, execution_quote
from bot.execution import (DemoExecutionService, EntryPreflightRejected,
                           LeadPortfolioExecutionService)
from bot.indicators import atr, validate_ohlcv
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import effective_max_hold_hours, runner_tail_is_exempt
from bot.live import mr_scan
from bot.market_risk import (MarketRiskAssessment, MarketRiskState,
                             fetch_market_risk_status)
from bot.mr_candidate import (MidExitState, staged_profit_lock_stop,
                              deviation_entry_signal, deviation_entry_signal_at_ema,
                              deviation_structure_side,
                              mid_band, mid_exit_check, no_progress_update,
                              regime_faded_triggered, structure_invalid_quote_exit,
                              structure_invalid_triggered, structure_reentry_confirmed,
                              trailing_stop)
from bot.portfolio import Position, classify_exit_category
from bot.lead_risk import lead_entry_exposure, settings_for_execution
from bot.regime import Regime, classify_regimes
from bot.risk import (OrderIntent, assess_entry, calculate_adaptive_stop,
                      calculate_order_quantity, funding_rate_rejection_reason)
from bot.signals import Action, Side, Signal
from bot.vsb import breakout_signal, effective_entry_atr, squeeze_context, strategy_metadata
from bot.strategy_router import MEAN_REVERSION
from bot.time_filter import TimeFilterFlags, evaluate_time_filter
from bot.utils.logging import redact_log_record, redacted_excepthook

HOUR_MS = 3600 * 1000
FOUR_H_MS = 4 * HOUR_MS
DAY_MS = 24 * HOUR_MS
TOP_N = 20
MIN_DAILY_UNIVERSE_BARS = 55
DEV_PCT = 0.01
DEV_ATR_MULTIPLIER = 0.8
DEV_RULE = "atr_0p8_floor_1pct"
EMA9_RECLAIM_PERIOD = 9
DYNAMIC_LIMIT_RANGE_ATR = 1.8
DYNAMIC_LIMIT_BODY_RETRACE = 0.441
DYNAMIC_LIMIT_TTL_HOURS = 3
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
EXTERNAL_CLOSE_FILL_RECHECK_SECONDS = 300
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
    universe_four_hour: dict = field(default_factory=dict)
    lead_hourly: dict = field(default_factory=dict)
    lead_squeeze_hourly: dict = field(default_factory=dict)
    lead_universe: tuple | None = None
    lead_universe_at: datetime | None = None


def _utc_now():
    if active_clock() is not None:
        return active_clock().now()
    return datetime.now(timezone.utc)


def _require_entry_clock():
    if active_clock() is not None:
        active_clock().require_fresh()


def _refresh_exchange_clock(clock, exchange, state):
    if clock.age() < clock.REFRESH_SECONDS:
        return
    try:
        status = clock.synchronize(exchange)
    except Exception as exc:
        state.events.append({"event": "exchange_clock_sync_failed", "at": _utc_now().isoformat(),
                             "error": type(exc).__name__, "new_entries_paused": clock.age() > clock.MAX_AGE_SECONDS})
        logger.warning("Exchange clock refresh failed: {}", type(exc).__name__)
    else:
        state.events.append({"event": "exchange_clock_calibrated", "at": _utc_now().isoformat(), **status})


def _initialize_exchange_clock(exchange, state):
    clock = ExchangeClock()
    status = clock.synchronize(exchange)
    activate_clock(clock)
    state.events.append({"event": "exchange_clock_calibrated", "at": _utc_now().isoformat(), **status})
    return clock


def _background_reads_allowed(now):
    boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (boundary - now).total_seconds() > 60


def _wait_for_closed_hour():
    clock = active_clock()
    if clock is None:
        return
    boundary = _utc_now().replace(minute=0, second=0, microsecond=0)
    # At a UTC boundary the data cutoff must also confirm that hour has closed.
    delay = (boundary - clock.lower_now()).total_seconds()
    if delay > 0:
        time.sleep(delay + .001)


def _public_read_deadline(now, settings):
    hour = now.replace(minute=0, second=0, microsecond=0)
    entry_deadline = hour + timedelta(seconds=settings.execution.confirm_delay_sec_max)
    return entry_deadline if now <= entry_deadline else hour + timedelta(hours=1, seconds=-5)


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
        if time.monotonic() >= deadline:
            return  # Start the next cycle before another potentially slow monitor read.
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


def _record_max_hold_waiver(state, symbol, position, meta, held_hours, now, regular_limit):
    if (held_hours < regular_limit or not meta.get("trailing_started")
            or meta.get("max_hold_waiver_logged")):
        return False
    state.events.append({
        "event": "max_hold_24h_waived_trailing", "at": now.isoformat(),
        "symbol": symbol, "side": position.side.value, "held_hours": held_hours,
        "trailing_trigger_r": meta.get("trailing_trigger_r", 2.0),
        "protective_stop": meta.get("protective_stop", position.stop_price),
    })
    meta["max_hold_waiver_logged"] = True
    return True


def _apply_max_hold_exit(is_demo, execution, state, symbol, position, price, now,
                         settings, fee_rate, slip_rate):
    opened_at = _aware_datetime(position.opened_at)
    if opened_at is None:
        return False
    held_hours = max(0.0, (now - opened_at).total_seconds() / 3600)
    meta = state.exit_management_by_symbol.get(symbol) or {}
    regular_limit = float(settings.exits.max_hold_bars)
    exempt = runner_tail_is_exempt(
        position.side, position.entry_price, position.stop_price, meta, settings.exits,
    )
    _record_max_hold_waiver(state, symbol, position, meta, held_hours, now, regular_limit)
    max_hold_hours = effective_max_hold_hours(
        position.side, position.entry_price, position.stop_price, meta, settings.exits,
    )
    if max_hold_hours is None or max_hold_hours <= 0 or held_hours < max_hold_hours:
        return False
    reason = "max_hold_runner" if exempt else "max_hold_24h"
    _close_position(is_demo, execution, state, symbol, price, reason, now, fee_rate, slip_rate)
    return True


def _mark_trailing_started(state, symbol, position, meta, progress_r, active_stop,
                           proposed_stop, trigger_r, now):
    if progress_r < trigger_r:
        return False
    protected = (
        active_stop >= proposed_stop and active_stop > position.entry_price
        if position.side is Side.LONG
        else active_stop <= proposed_stop and active_stop < position.entry_price
    )
    if not protected:
        return False
    if not meta.get("trailing_started"):
        meta["trailing_started"] = True
        meta["trailing_started_at"] = now.isoformat()
        meta["trailing_trigger_r"] = float(trigger_r)
        state.events.append({
            "event": "trailing_stop_started", "at": now.isoformat(),
            "symbol": symbol, "side": position.side.value,
            "progress_r": progress_r, "protective_stop": active_stop,
            "trailing_stop": proposed_stop, "trigger_r": float(trigger_r),
        })
    return True


def _target_margin_per_symbol(state, settings):
    if state.execution_mode == "lead_usdm":
        return None  # Lead quantity is determined by notional and stop risk, before leverage.
    if state.execution_mode == "demo_usdm":
        return state.current_equity * settings.sizing.live_margin_equity_pct
    return settings.sizing.margin_per_symbol_usdt


def _entry_capital_limits(state, settings, side, now):
    if state.execution_mode != "lead_usdm":
        return {}, {}
    exposure = lead_entry_exposure(state, settings, now)
    notional = state.current_equity * settings.lead_risk.max_symbol_notional_pct
    common = {"exposure": exposure,
              "portfolio_risk_cap_pct": settings.lead_risk.max_portfolio_stop_risk_pct}
    return ({**common, "side": side, "target_notional_usdt": notional},
            {**common, "per_symbol_notional_usdt": notional})


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


def _filter_lead_market_candidates(candidates, markets, lead_market_ids):
    if not lead_market_ids:
        raise RuntimeError("Lead Portfolio contract whitelist is unavailable; refusing entries")
    allowed = {str(value).upper() for value in lead_market_ids}
    return [
        (symbol, quote_volume) for symbol, quote_volume in candidates
        if str((markets.get(symbol) or {}).get("id", "")).upper() in allowed
    ]


def _record_lead_whitelist_observation(state, execution, ranked_hotlist,
                                      filtered_hotlist, markets, bar, now):
    """Persist the ranking and Lead whitelist result once per closed 1H bar."""
    bar_key = bar.isoformat()
    metadata_key = "__lead_whitelist_shadow__"
    metadata = state.exit_management_by_symbol.setdefault(metadata_key, {})
    if metadata.get("bar") == bar_key:
        return False
    filtered_ranks = {row["symbol"]: row.get("rank") for row in filtered_hotlist}
    ranked = []
    for index, row in enumerate(ranked_hotlist, start=1):
        symbol = row["symbol"]
        market_id = str((markets.get(symbol) or {}).get("id", "")).upper()
        ranked.append({
            "rank": int(row.get("rank", index)),
            "symbol": symbol,
            "market_id": market_id,
            "whitelisted": symbol in filtered_ranks,
            "filtered_rank": filtered_ranks.get(symbol),
        })
    state.events.append({
        "event": "lead_whitelist_ranking_shadow",
        "at": now.isoformat(),
        "bar": bar_key,
        "source": MEAN_REVERSION,
        "execution_mode": state.execution_mode,
        "top_n": len(ranked),
        "whitelist_size": len(getattr(execution, "lead_market_ids", ())),
        "ranked_candidates": ranked,
    })
    metadata["bar"] = bar_key
    return True


def _atr_deviation_threshold(frame4, settings):
    """Return the entry threshold from closed 4H bars, failing closed if invalid."""
    if frame4 is None or len(frame4) < settings.lookback.atr_len:
        return None, None, None
    try:
        validate_ohlcv(frame4)
        atr_value = float(atr(frame4, settings.lookback.atr_len).iloc[-1])
        close = float(frame4["close"].iloc[-1])
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None, None, None
    if not math.isfinite(atr_value) or atr_value <= 0 or not math.isfinite(close) or close <= 0:
        return None, None, None
    atr_pct = atr_value / close
    return atr_value, atr_pct, max(DEV_PCT, DEV_ATR_MULTIPLIER * atr_pct)


def _record_atr_deviation_shadow(state, symbol, frame1, frame4, bar_close_time,
                                 evaluated_at, closed_ema, close, flags, settings,
                                 baseline_signal, stop, entry_atr, regime_value,
                                 daily_gate, scan_row, active_threshold):
    """Retain the fixed-threshold comparison before applying the active ATR gate."""
    shadow_key = f"{symbol}|{bar_close_time.isoformat()}"
    if scan_row is not None and scan_row.get("atr_deviation_shadow_key") == shadow_key:
        return
    atr4, atr_pct, threshold = _atr_deviation_threshold(frame4, settings)
    if threshold is None:
        shadow_signal = None
        shadow_reason = "closed_4h_atr_unavailable"
        shadow_available = False
    else:
        try:
            shadow_signal = deviation_entry_signal_at_ema(
                closed_ema, float(frame1.iloc[-1]["open"]), close, settings, flags,
                deviation_pct=threshold,
            )
            shadow_reason = shadow_signal.reason
            shadow_available = True
        except Exception as exc:  # Observability must not affect the live candidate.
            shadow_signal = None
            shadow_reason = f"shadow_evaluation_error:{type(exc).__name__}"
            shadow_available = False
    shadow_allowed = shadow_signal is not None and shadow_signal.action is Action.OPEN
    close4 = float(frame4["close"].iloc[-1])
    event = {
        "event": "atr_deviation_entry_shadow",
        "at": evaluated_at.isoformat(),
        "shadow_key": shadow_key,
        "symbol": symbol,
        "source": MEAN_REVERSION,
        "execution_mode": state.execution_mode,
        "runner_build_id": RUNNER_BUILD_ID,
        "bar_open_at": frame1.index[-1].to_pydatetime().isoformat(),
        "bar_close_at": bar_close_time.isoformat(),
        "closed_4h_bar_open_at": frame4.index[-1].to_pydatetime().isoformat(),
        "closed_4h_bar_close_at": (frame4.index[-1] + timedelta(hours=4)).isoformat(),
        "entry_close": close,
        "closed_4h_close": close4,
        "closed_4h_atr": atr4,
        "closed_4h_atr_pct": atr_pct,
        "baseline_threshold_pct": DEV_PCT,
        "shadow_threshold_rule": "max(0.01, 0.8 * closed_4h_atr_pct)",
        "shadow_threshold_pct": threshold,
        "deviation_pct": close / closed_ema - 1.0,
        "baseline_decision": "ALLOW",
        "baseline_side": baseline_signal.side.value,
        "shadow_decision": ("UNAVAILABLE" if not shadow_available
                            else "ALLOW" if shadow_allowed else "FILTER"),
        "shadow_side": shadow_signal.side.value if shadow_allowed else None,
        "shadow_reason": shadow_reason,
        "active_deviation_rule": DEV_RULE,
        "active_threshold_pct": active_threshold,
        "active_decision": ("UNAVAILABLE" if active_threshold is None else
                            "ALLOW" if abs(close / closed_ema - 1.0) >= active_threshold else "FILTER"),
        "time_filter_decision": "ALLOW",
        "daily_gate": ({
            "ready": daily_gate.get("ready"),
            "direction": daily_gate.get("direction"),
            "reason": daily_gate.get("reason"),
        } if daily_gate is not None else None),
        "stop_price": stop,
        "entry_atr_1h": entry_atr,
        "regime": regime_value,
    }
    state.events.append(event)
    if scan_row is not None:
        scan_row["atr_deviation_shadow_key"] = shadow_key


# --- open / close helpers (demo + local, both sides) ---------------------------------

def _open_position(is_demo, execution, state, symbol, side, price, stop, qty, leverage,
                   fee_rate, slip_rate, reason, regime_value, at, *,
                   entry_preflight=None, submission_deadline=None, entry_audit=None,
                   scan_row=None, checkpoint=None, recovery_meta=None,
                   order_type="market", limit_expires_at=None):
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
                "order_type": order_type, "expires_at": limit_expires_at,
                "stop_client_algo_id": (f"mrb_s_{client_id[-24:]}"
                                        if order_type == "limit" else None),
            })
            if scan_row.get("atr_deviation_shadow_key"):
                attempt["atr_deviation_shadow_key"] = scan_row["atr_deviation_shadow_key"]
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

            def on_protection(stop_order):
                attempt["preprotected_stop"] = dict(stop_order)
                mr_scan.checkpoint(checkpoint)

        try:
            kwargs = ({"before_submit": entry_preflight,
                       "submission_deadline": submission_deadline}
                      if entry_preflight is not None else {})
            if attempt is not None:
                kwargs.update(client_order_id=client_id, on_submit=on_submit)
            if order_type == "limit":
                if attempt is None:
                    raise mr_scan.ScanCheckpointError("dynamic live limit requires a durable scan checkpoint")
                kwargs.update(limit_expires_at=limit_expires_at, on_protection=on_protection)
            if state.execution_mode == "lead_usdm" and entry_audit is not None:
                kwargs["on_safety_plan"] = lambda plan: entry_audit.update(capital_safety=dict(plan))
            submit = execution.open_limit if order_type == "limit" else execution.open
            result = submit(intent, stop_price=stop, **kwargs)
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
        if result.get("pending_limit"):
            attempt.update(status="limit_pending", order_id=result.get("order_id"),
                           preprotected_stop=result.get("stop"),
                           limit_price=result.get("price"))
            scan_row.update(phase="limit_order_pending", order_may_have_been_sent=True)
            mr_scan.attempt_outcome(state, attempt, "limit_pending", _utc_now(),
                                    phase=scan_row["phase"], order_may_have_been_sent=True)
            mr_scan.checkpoint(checkpoint)
            return {"pending_limit": True, "clientOrderId": client_id,
                    "order_id": result.get("order_id"), "price": result.get("price")}
        if result.get("not_filled"):
            attempt.update(status="not_filled")
            mr_scan.attempt_outcome(state, attempt, "not_filled", _utc_now(),
                                    phase="limit_order_terminal", order_may_have_been_sent=True)
            mr_scan.checkpoint(checkpoint)
            return None
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
    if result.get("capital_safety") is not None:
        state.events.append({"event": "lead_entry_capital_verified", "symbol": symbol,
                             "at": opened_at.isoformat(), **result["capital_safety"]})
    if entry_audit is not None:
        _record_entry_execution_fill(state, position, result, entry_audit)
    return position


def _open_local_position(state, symbol, side, price, stop, qty, leverage,
                         fee_rate, slip_rate, reason, regime_value, at):
    entry_fill = price * (1 + slip_rate) if side is Side.LONG else price * (1 - slip_rate)
    margin = qty * entry_fill / leverage
    entry_fee = qty * entry_fill * fee_rate
    if state.cash < margin + entry_fee:
        _record_entry_rejection(state, symbol, side, "available_cash_insufficient", "local_execution", at,
                                details={"available_cash": state.cash, "required_margin": margin,
                                         "entry_fee": entry_fee, "planned_quantity": qty})
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


def _process_local_dynamic_limits(state, current_prices, settings, now, fee_rate, slip_rate):
    """Simulate resting pullback limits from fresh public ticks in local Paper."""
    for key, attempt in list(state.mr_entry_attempts.items()):
        if attempt.get("status") != "paper_limit_pending":
            continue
        symbol = attempt["symbol"]
        try:
            expires_at = datetime.fromisoformat(
                str(attempt["expires_at"]).replace("Z", "+00:00"),
            )
            side = Side(attempt["side"])
            limit_price = float(attempt["limit_price"])
            stop_price = float(attempt["stop_price"])
            quantity = float(attempt["quantity"])
            entry_atr = float(attempt["meta"]["entry_atr"])
            structure_low = float(attempt["meta"]["structure_low"])
            structure_high = float(attempt["meta"]["structure_high"])
        except (KeyError, TypeError, ValueError, OverflowError):
            # Keep the attempt pending, which blocks further local entries, until
            # its persisted order intent can be read safely.
            continue
        if expires_at.tzinfo is None:
            continue
        if now >= expires_at:
            attempt["status"] = "paper_limit_expired"
            state.events.append({"event": "dynamic_limit_entry_expired", "at": now.isoformat(),
                                 "symbol": symbol, "side": side.value,
                                 "limit_price": limit_price, "expires_at": expires_at.isoformat(),
                                 "execution_mode": state.execution_mode})
            continue
        price = current_prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            continue
        price = float(price)
        touched = price <= limit_price if side is Side.LONG else price >= limit_price
        if not touched:
            continue
        if symbol in state.portfolio.positions:
            attempt["status"] = "paper_limit_position_conflict"
            state.events.append({"event": "dynamic_limit_entry_conflict", "at": now.isoformat(),
                                 "symbol": symbol, "reason": "position_already_exists"})
            continue
        opened = _open_local_position(
            state, symbol, side, limit_price, stop_price, quantity,
            int(settings.sizing.leverage), fee_rate, 0.0,
            attempt["reason"], attempt["regime"], now,
        )
        if opened is None:
            attempt["status"] = "paper_limit_rejected"
            continue
        _initialize_entry_metadata(
            state, symbol, opened, False, limit_price, stop_price,
            entry_atr, structure_low, structure_high,
        )
        state.exit_management_by_symbol[symbol].update(strategy_metadata(attempt["meta"]))
        attempt["status"] = "paper_limit_filled"
        attempt["filled_at"] = now.isoformat()
        attempt["fill_price"] = opened.entry_price
        state.events.append({"event": "dynamic_limit_entry_filled", "at": now.isoformat(),
                             "symbol": symbol, "side": side.value,
                             "limit_price": limit_price, "observed_price": price,
                             "fill_price": opened.entry_price,
                             "clientOrderId": key,
                             "fill_model": "fresh_tick_touch_at_limit",
                             "ema9_reclaim_side": attempt.get("ema9_reclaim_side")})
        stop_touched = (price <= stop_price if side is Side.LONG
                        else price >= stop_price)
        if stop_touched:
            _close_position(False, None, state, symbol, price, "protective_stop",
                            now, fee_rate, slip_rate)


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


def _record_entry_rejection(state, symbol, side, reason, phase, at, *,
                            bar_close_at=None, details=None, scan_row=None):
    event = {"event": "entry_risk_rejected", "at": at.isoformat(),
             "symbol": symbol, "side": side.value, "reason": reason, "phase": phase,
             "bar_close_at": bar_close_at.isoformat() if bar_close_at is not None else None,
             "equity": state.current_equity, "source": MEAN_REVERSION,
             "details": details or {}}
    state.events.append(event)
    if scan_row is not None:
        scan_row["reason"] = reason
    _set_symbol_status(state, symbol, action="BLOCK", side=side, reason=reason, at=at)


def _entry_preflight(exchange, execution, state, settings, market, context, flags,
                     funding_rate, market_risk, audit, *, propagate_read_errors=False,
                     scan_row=None, limit_entry_price=None):
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
        "deviation_threshold_pct": context.deviation_pct,
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
            return None, f"entry_quote_unavailable:{type(exc).__name__}", None
        quote_data, error = execution_quote(book, context.side, _utc_now(),
                                            max_age_sec=settings.execution.entry_quote_max_age_sec)
        return quote_data, error, book

    def before_submit(intent, planned_stop):
        try:
            _require_entry_clock()
        except ExchangeClockError:
            reject("exchange_clock_stale_or_unavailable")
        if _utc_now() > deadline:
            reject("entry_signal_expired")
        if not math.isfinite(context.deviation_pct) or context.deviation_pct <= 0:
            reject("entry_deviation_unavailable")
        if state.execution_mode == "lead_usdm":
            if (mr_scan.unresolved_attempts(state)
                    or not _demo_sync_stopped(execution, state, _utc_now())
                    or not _audit_demo_protective_stops(execution, state, _utc_now())):
                reject("lead_account_not_synchronized")
        # Public quote is optional attribution evidence. Fetch Demo last so the
        # executable reference does not wait behind an additional network call.
        audit["public_quote"], public_error, _ = quote(exchange)
        audit["public_quote_error"] = public_error
        if _utc_now() > deadline:
            reject("entry_signal_expired")
        if scan_row is not None:
            scan_row.update(phase="entry_quote", endpoint="demo")
        demo_quote, quote_error, execution_book = quote(execution.exchange, required=True)
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
        quote_price = demo_quote["price"]
        price = quote_price if limit_entry_price is None else float(limit_entry_price)
        if limit_entry_price is not None:
            already_touched = (quote_price <= price if context.side is Side.LONG
                               else quote_price >= price)
            if already_touched:
                reject("dynamic_limit_price_already_touched")
        direction = 1 if context.side is Side.LONG else -1
        adverse_atr = direction * (quote_price - context.signal_close) / context.atr
        audit.update({
            "adverse_quote_atr": adverse_atr,
            "observe_only": True,
            "atr_threshold_observations": [
                {"threshold_atr": threshold, "would_block": adverse_atr > threshold}
                for threshold in settings.execution.entry_adverse_atr_observe
            ],
        })
        if structure_invalid_triggered(
                context.side, quote_price, context.structure_low, context.structure_high,
                context.atr, buffer_atr=settings.exits.structure_invalid_buffer_atr):
            reject("entry_structure_invalid")
        if ((context.side is Side.LONG and quote_price <= planned_stop)
                or (context.side is Side.SHORT and quote_price >= planned_stop)):
            reject("entry_price_beyond_planned_stop")
        if context.strategy_tag == "VSB":
            # A resting retracement limit can lie inside the breakout buffer.
            # Confirm the fresh market still holds the frozen breakout instead.
            direction = 1 if context.side is Side.LONG else -1
            adverse_dev = -direction * (quote_price - context.closed_ema) / context.closed_ema
            valid_breakout = (context.breakout_level is not None
                              and direction * (quote_price - context.breakout_level) > 0
                              and -1.2 * context.deviation_pct <= adverse_dev < context.deviation_pct)
            signal = Signal(context.side if valid_breakout else None,
                            Action.OPEN if valid_breakout else Action.NONE, "vsb_quote_recheck", 1.0)
        else:
            signal = deviation_entry_signal_at_ema(
                context.closed_ema, context.signal_open, quote_price, settings, flags,
                deviation_pct=context.deviation_pct,
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
        try:
            quantity_limits, approval_limits = _entry_capital_limits(state, settings, context.side, checked_at)
        except (KeyError, TypeError, ValueError, OverflowError):
            reject("lead_entry_risk_inputs_unavailable")
        quantity_audit = {}
        qty = calculate_order_quantity(
            state.current_equity, price, state.portfolio, settings,
            step_size=step, minimum_quantity=minimum, size_multiplier=multiplier,
            stop_price=stop, gross_notional_cap_pct=gross_cap,
            target_margin_usdt=target_margin,
            audit=quantity_audit, **quantity_limits,
        )
        audit.update(planned_price=price, planned_quantity=qty, planned_stop=stop,
                     quantity_risk=quantity_audit)
        if qty <= 0:
            reject(quantity_audit.get("reason") or "entry_quantity_unavailable")
        refreshed = replace(intent, price=price, quantity=qty, stop_price=stop)
        approval = assess_entry(
            refreshed, state.portfolio, state.current_equity, funding_rate, checked_at, settings,
            gross_notional_cap_pct=gross_cap, market_risk_state=market_risk.state.value,
            per_symbol_margin_usdt=target_margin,
            **approval_limits,
        )
        audit["risk_approval"] = {"reason": approval.reason, "details": approval.details}
        if not approval.allowed:
            reject(approval.reason)
        audit.update({"planned_price": price, "planned_quantity": qty,
                      "planned_stop": stop,
                      "top_quote_covers_quantity": demo_quote["quantity"] >= qty})
        if quantity_limits:
            exposure = quantity_limits["exposure"]
            audit.update({"existing_portfolio_stop_risk": exposure.stop_risk,
                          "existing_portfolio_capital_risk": exposure.capital_risk,
                          "existing_portfolio_giveback_risk": exposure.giveback_risk,
                          "existing_portfolio_weighted_giveback_risk": exposure.weighted_giveback_risk,
                          "portfolio_giveback_risk_weight": settings.lead_risk.giveback_risk_weight,
                          "new_planned_risk": approval.details["estimated_risk"],
                          "max_portfolio_stop_risk": state.current_equity * settings.lead_risk.max_portfolio_stop_risk_pct})
        if state.execution_mode == "lead_usdm":
            try:
                audit["entry_liquidity"] = entry_liquidity_observation(
                    execution_book, context.side, qty, slippage_limit=settings.risk.max_slippage_pct,
                    step_size=step, minimum_quantity=minimum,
                )
            except Exception as exc:
                # Keep diagnostic failures out of the order approval path.
                audit["entry_liquidity"] = {"observe_only": True, "status": "unavailable",
                                            "reason": type(exc).__name__}
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
    if "entry_liquidity" in audit:
        event["entry_liquidity"] = {**audit["entry_liquidity"],
                                    "actual_slippage_pct": direction * (position.entry_price / demo_price - 1)}
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
            "at": _utc_now().isoformat(),
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


def _position_sample_is_eligible(position, observed_at):
    opened_at = _aware_datetime(position.opened_at)
    sampled_at = _aware_datetime(observed_at)
    return opened_at is not None and sampled_at is not None and sampled_at >= opened_at


def _position_bar_is_eligible(position, frame, as_of):
    bar_open = _aware_datetime(frame.index[-1].to_pydatetime())
    return (bar_open is not None and _position_sample_is_eligible(position, bar_open)
            and bar_open + timedelta(hours=1) <= as_of)


def _update_demo_excursions(meta, position, *, frame=None, price=None, observed_at=None,
                            as_of=None):
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

    if price is not None and _position_sample_is_eligible(position, observed_at):
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
    if not _position_bar_is_eligible(position, frame, as_of or _utc_now()):
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
    previous = (getattr(state, "exit_management_by_symbol", {}).get(position.symbol) or {}).get("manual_realized", {})
    if previous:
        gross = gross + previous["gross_pnl"] if gross is not None else None
        fees = fees + previous["fees"] if fees is not None else None
    net = (_finite_pnl_value(gross - fees + funding)
           if gross is not None and fees is not None and funding is not None else None)
    return gross, fees, funding, net


def _resolve_exchange_flat_close_fill(execution, state, position, observed_flat_at):
    resolver = getattr(execution, "resolve_exchange_flat_close_fill", None)
    if not callable(resolver):
        return None
    try:
        return resolver(position.symbol, position.side, position.opened_at,
                        observed_flat_at, position.quantity)
    except Exception as exc:
        logger.warning("Cannot verify {} external close for {}: {}",
                       _execution_label(execution, state), position.symbol,
                       type(exc).__name__)
        return None


def _external_close_position(state, event):
    symbol = event.get("symbol")
    opened_at = _aware_datetime(event.get("opened_at"))
    if not symbol or opened_at is None:
        return None, None
    open_event = next((candidate for candidate in reversed(state.events)
                       if candidate.get("event") == "paper_fill"
                       and candidate.get("action") == "OPEN"
                       and candidate.get("symbol") == symbol
                       and _aware_datetime(candidate.get("at")) == opened_at), None)
    if open_event is None:
        return None, None
    try:
        side = Side(open_event["side"])
        quantity = _finite_pnl_value(event.get("quantity"))
        opened_quantity = _finite_pnl_value(open_event.get("quantity"))
        entry_price = _finite_pnl_value(open_event.get("price"))
    except (KeyError, ValueError):
        return None, None
    if (quantity is None or opened_quantity is None or entry_price is None
            or not math.isclose(quantity, opened_quantity, rel_tol=1e-8, abs_tol=1e-12)):
        return None, None
    return SimpleNamespace(symbol=symbol, side=side, quantity=quantity,
                           entry_price=entry_price, opened_at=opened_at), open_event


def _reconcile_demo_unverified_closes(execution, state, now):
    """Backfill manual/external flat fills when bounded exchange evidence becomes available."""
    resolver = getattr(execution, "resolve_exchange_flat_close_fill", None)
    if not callable(resolver):
        return False
    changed = False
    for event in list(state.events):
        if (event.get("event") != "paper_fill" or event.get("action") != "CLOSE"
                or event.get("reason") != "exchange_flat_unverified"
                or event.get("execution_source") != _execution_source(state)):
            continue
        last_check = _aware_datetime(event.get("external_close_lookup_at"))
        if (last_check is not None
                and (now - last_check).total_seconds() < EXTERNAL_CLOSE_FILL_RECHECK_SECONDS):
            continue
        position, open_event = _external_close_position(state, event)
        observed_flat_at = _aware_datetime(event.get("at"))
        if position is None or observed_flat_at is None:
            continue
        event["external_close_lookup_at"] = now.isoformat()
        fill = _resolve_exchange_flat_close_fill(execution, state, position, observed_flat_at)
        closed_at = _verified_fill_timestamp(fill)
        if fill is None or closed_at is None:
            continue
        entry_fee = _finite_pnl_value(open_event.get("fee"))
        fee_state = SimpleNamespace(entry_costs={})
        if entry_fee is not None:
            fee_state.entry_costs[position.symbol] = {"entry_fee": entry_fee}
        gross, fees, funding, net = _demo_close_values(
            execution, fee_state, position, fill, closed_at,
        )
        order_ids = list(fill.get("order_ids") or [])
        held_seconds = (closed_at - position.opened_at).total_seconds()
        event.update({
            "reason": "external_close_exchange_trades_verified",
            "closed_at": closed_at.isoformat(), "exit_time_verified": True,
            "held_hours": held_seconds / 3600 if held_seconds >= 0 else None,
            "quantity": fill.get("quantity"), "price": fill.get("average"),
            "gross_pnl": gross, "fees": fees, "funding_paid": funding,
            "net_pnl": net, "pnl_recorded": net is not None,
            "orderId": order_ids[0] if len(order_ids) == 1 else None,
            "external_order_ids": order_ids,
            "external_close_reconciled_at": now.isoformat(),
        })
        state.events.append({
            "event": "external_close_fill_reconciled", "at": now.isoformat(),
            "symbol": position.symbol, "opened_at": position.opened_at.isoformat(),
            "closed_at": closed_at.isoformat(), "quantity": fill.get("quantity"),
            "average": fill.get("average"), "gross_pnl": gross, "fees": fees,
            "funding_paid": funding, "net_pnl": net, "pnl_recorded": net is not None,
            "external_order_ids": order_ids,
            "execution_source": _execution_source(state),
        })
        changed = True
    return changed


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
        if end is not None and event.get("funding_end_exclusive"):
            end -= timedelta(milliseconds=1)
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


def _manual_close_event_totals(event, position, meta):
    if not meta.get("manual_position_changed") and position.reason != "manual_position_adopted":
        return
    previous = meta.get("manual_realized", {})
    quantity = float(event["quantity"]) + previous.get("quantity", 0.0)
    price, gross = _finite_pnl_value(event.get("price")), _finite_pnl_value(event.get("gross_pnl"))
    notional = float(event["quantity"]) * price + previous.get("notional", 0.0) if price is not None else None
    event.update(strategy_sample_eligible=False, sample_exclusion_reason="manual_position_changed",
                 final_fill_quantity=event["quantity"], quantity=quantity,
                 price=notional / quantity if notional is not None else None,
                 entry_price=((notional - (1 if position.side is Side.LONG else -1) * gross) / quantity)
                 if notional is not None and gross is not None else None)


def _close_position(is_demo, execution, state, symbol, price, reason, at, fee_rate, slip_rate):
    position = state.portfolio.positions.get(symbol)
    if position is None:
        return None
    if is_demo:
        manual_meta = dict(state.exit_management_by_symbol.get(symbol) or {})
        stop = state.exchange_orders_by_symbol.get(symbol, {}).get("stop")
        try:
            result = execution.close(position, stop=stop, reason=reason)
        except Exception as exc:
            state.execution_status = "close_reconciliation_required"
            state.inconsistencies = [f"close reconciliation required for {symbol}"]
            state.events.append({"event": "position_close_pending", "symbol": symbol,
                                 "at": at.isoformat(), "reason": reason,
                                 "error": type(exc).__name__})
            raise
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
        _manual_close_event_totals(close_event, position, manual_meta)
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


def _observe_account_equity(state, snapshot, settings, at):
    observed_at = _aware_datetime(snapshot.get("at"))
    equity = float(snapshot.get("equity", 0))
    if (observed_at is None or not -30 <= (at - observed_at).total_seconds() <= 60
            or not math.isfinite(equity) or equity <= 0):
        raise ValueError("fresh positive account equity is required")
    portfolio = state.portfolio
    previous = (portfolio.trading_date, portfolio.daily_halted)
    portfolio.observe_equity(equity, observed_at, settings)
    if previous != (portfolio.trading_date, portfolio.daily_halted):
        state.events.append({
            "event": "daily_risk_halt_triggered" if portfolio.daily_halted else "daily_risk_day_started",
            "at": observed_at.isoformat(), "trading_date": portfolio.trading_date.isoformat(),
            "equity": equity, "day_start_equity": portfolio.day_start_equity,
            "drawdown": (portfolio.day_start_equity - equity) / portfolio.day_start_equity,
            "threshold": settings.risk.daily_drawdown_halt_pct,
        })


def _lead_manual_position_isolation_active(state):
    audit = state.protective_stop_audit or {}
    return (
        state.execution_mode == "lead_usdm"
        and state.execution_status == "inconsistent"
        and audit.get("scope") == "manual_position_change"
        and bool(audit.get("blocked_symbols"))
    )


def _demo_sync_stopped(execution, state, at):
    snapshot = execution.sync_snapshot()
    if isinstance(execution, DemoExecutionService):
        try:
            _observe_account_equity(state, snapshot, execution.settings, _utc_now())
        except (TypeError, ValueError, OverflowError):
            state.execution_status = "inconsistent"
            state.inconsistencies = ["account equity snapshot is stale or invalid"]
            return False
    state.exchange_snapshot = snapshot
    if (state.execution_mode == "lead_usdm" and getattr(execution, "adopt_manual_positions", False)
            and callable(getattr(execution, "reconcile_manual_position", None))):
        if not _sync_lead_position_changes(execution, state, snapshot,
                                           _aware_datetime(snapshot.get("at")) or at):
            return False
    live = set(snapshot["positions"])
    for symbol in list(state.portfolio.positions):
        if symbol not in live:
            position = state.portfolio.positions[symbol]
            order_data = state.exchange_orders_by_symbol.get(symbol, {})
            stop = order_data.get("stop") if isinstance(order_data, dict) else None
            strategy_fill = None
            if order_data.get("close_attempt"):
                try:
                    result = execution.close(position, stop=stop,
                                             reason=order_data["close_attempt"].get("reason"))
                    strategy_fill = result.get("fill")
                except Exception as exc:
                    state.execution_status = "close_reconciliation_required"
                    state.inconsistencies = [f"close reconciliation required for {symbol}"]
                    logger.warning("Flat close reconciliation pending for {}: {}", symbol, type(exc).__name__)
                    return False
            stop_fill = None
            resolver = getattr(execution, "resolve_protective_stop_algo_fill", None)
            if callable(resolver):
                try:
                    stop_fill = resolver(symbol, stop, position.quantity)
                except Exception as exc:
                    logger.warning("Cannot verify {} stop fill for {}: {}",
                                   _execution_label(execution, state), symbol, type(exc).__name__)
            external_fill = None
            if stop_fill is None and strategy_fill is None:
                external_fill = _resolve_exchange_flat_close_fill(execution, state, position, at)
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

            verified_fill = stop_fill or strategy_fill or external_fill
            verified_closed_at = _verified_fill_timestamp(verified_fill) if verified_fill else None
            closed_at = verified_closed_at or at
            gross = fees = funding = net = None
            if verified_fill:
                gross, fees, funding, net = _demo_close_values(
                    execution, state, position, verified_fill, closed_at,
                )
            observations = _demo_trade_observations(
                state, position, closed_at=verified_closed_at,
                reconciled_at=at if verified_closed_at is None else None,
            )
            manual_meta = dict(state.exit_management_by_symbol.get(symbol) or {})
            state.portfolio.positions.pop(symbol, None)
            state.exchange_orders_by_symbol.pop(symbol, None)
            state.exit_management_by_symbol.pop(symbol, None)
            state.entry_costs.pop(symbol, None)
            close_attempt = order_data.get("close_attempt") or {}
            reason = ("protective_stop_exchange_algo" if stop_fill else
                      close_attempt["reason"] if (strategy_fill or external_fill) and close_attempt.get("reason") else
                      "external_close_exchange_trades_verified" if external_fill else
                      "exchange_flat_unverified")
            state.portfolio.last_exit_at[symbol] = closed_at
            state.portfolio.last_exit_category[symbol] = _mr_exit_category(reason, net)
            state.portfolio.last_exit_source[symbol] = MEAN_REVERSION
            _invalidate_demo_stop_audit(state)
            close_event = {"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                           "side": position.side.value, "at": closed_at.isoformat(), "reason": reason,
                           "quantity": (verified_fill.get("quantity", verified_fill.get("filled"))
                                        if verified_fill else position.quantity),
                           "price": verified_fill.get("average") if verified_fill else None,
                           "gross_pnl": gross, "funding_paid": funding, "net_pnl": net,
                           "pnl_recorded": net is not None, "source": MEAN_REVERSION,
                           "execution_source": _execution_source(state),
                           "orderId": (stop_fill.get("order_id") if stop_fill else
                                       strategy_fill.get("id") if strategy_fill else
                                       external_fill.get("order_ids", [None])[0]
                                       if external_fill and len(external_fill.get("order_ids", [])) == 1
                                       else None),
                           "algoId": stop_fill.get("algo_id") if stop_fill else None,
                           **observations}
            if fees is not None:
                close_event["fees"] = fees
            _manual_close_event_totals(close_event, position, manual_meta)
            if external_fill:
                close_event["external_order_ids"] = list(external_fill.get("order_ids") or [])
            elif strategy_fill is None:
                close_event["external_close_lookup_at"] = at.isoformat()
            state.events.append(close_event)
    local = set(state.portfolio.positions)
    pending_symbols = {row["symbol"] for row in mr_scan.unresolved_attempts(state)}
    unknown = (live - local) - pending_symbols
    blocked_adoptions = []
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
                    "scope": "manual_position_change", "blocked_symbols": [symbol],
                }
                state.events.append({
                    "event": "manual_position_adoption_blocked", "at": at.isoformat(),
                    "symbol": symbol, "reason": _manual_sync_error(exc),
                    "execution_source": _execution_source(state),
                })
                logger.error("Lead manual position adoption blocked for {}: {}",
                             symbol, _manual_sync_error(exc))
                blocked_adoptions.append({"symbol": symbol, "reason": _manual_sync_error(exc)})
        local = set(state.portfolio.positions)
    if blocked_adoptions:
        blocked_symbols = [row["symbol"] for row in blocked_adoptions]
        mismatches = [
            f"manual position adoption blocked for {row['symbol']}: {row['reason']}"
            for row in blocked_adoptions
        ]
        state.execution_status = "inconsistent"
        state.inconsistencies = mismatches
        state.protective_stop_audit = {
            "at": at.isoformat(), "status": "error",
            "position_count": len(state.portfolio.positions),
            "mismatch_count": len(mismatches), "mismatches": mismatches,
            "scope": "manual_position_change", "blocked_symbols": blocked_symbols,
        }
        return False
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
    pending_closes = [symbol for symbol in local
                      if state.exchange_orders_by_symbol.get(symbol, {}).get("close_attempt")]
    if pending_closes:
        state.execution_status = "close_reconciliation_required"
        state.inconsistencies = [f"close reconciliation required for {symbol}" for symbol in pending_closes]
        for symbol in pending_closes:
            orders = state.exchange_orders_by_symbol[symbol]
            attempt = orders["close_attempt"]
            try:
                execution.close(state.portfolio.positions[symbol], stop=orders.get("stop"),
                                reason=attempt.get("reason"))
            except Exception as exc:
                logger.warning("Close reconciliation pending for {}: {}", symbol, type(exc).__name__)
        return False  # Refresh the signed snapshot before recording any flat outcome.
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
    _reconcile_demo_unverified_closes(execution, state, at)
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


def _prepare_lead_manual_position(execution, symbol, exchange_position, at, *,
                                  recovered=None, previous_position=None, previous_stop=None):
    settings = execution.settings
    if not execution.exchange.markets:
        execution.exchange.load_markets()
    market_id = str(execution.exchange.market(symbol).get("id") or "").upper()
    if not market_id or market_id not in getattr(execution, "lead_market_ids", set()):
        raise RuntimeError("manual position is outside the verified Lead contract whitelist")
    recovered = recovered or execution.reconstruct_open_position(symbol, exchange_position, at=at)
    side = Side(recovered["side"])
    opened_at = recovered["opened_at"]
    atr_required = settings.lookback.atr_len + 1
    frame1 = _adoption_closed_frame(
        execution.exchange, symbol, opened_at, "1h", HOUR_MS,
        max(atr_required, settings.stops.swing_lookback_1h),
        120,
    )
    regime_required = max(
        settings.symbol_regime.ema_slow_bars,
        settings.lookback.min_regime_bars_4h,
        settings.regime.atrpct_lookback,
        settings.lookback.range_bars_4h + 1,
        settings.regime.slow_lookback_4h + 1,
        settings.lookback.adx_len * 3,
    )
    frame4 = _adoption_closed_frame(
        execution.exchange, symbol, opened_at, "4h", FOUR_H_MS, regime_required,
        max(regime_required, 100),
    )
    ema4 = float(frame4["close"].ewm(span=settings.symbol_regime.ema_slow_bars,
                 adjust=False, min_periods=settings.symbol_regime.ema_slow_bars).mean().iloc[-1])
    _, _, threshold = _atr_deviation_threshold(frame4, settings)
    if threshold is None or not math.isfinite(ema4) or ema4 <= 0:
        raise RuntimeError("entry-time deviation context is unavailable")
    entry_price = float(recovered["entry_price"])
    adverse = (ema4 - entry_price) / ema4 if side is Side.LONG else (entry_price - ema4) / ema4
    tag = "MR" if adverse >= threshold else "VSB"
    frozen_strategy = {"strategy_tag": tag}
    if tag == "VSB":
        # A retracement entry may fill up to three hours after the signal.
        for offset in range(4):
            squeeze = squeeze_context(frame1.iloc[:len(frame1) - offset])
            if squeeze["squeeze_complete"] and 6 <= squeeze["sqz_bars"] <= 36:
                frozen_strategy.update(strategy_metadata({"strategy_tag": tag, **squeeze}))
                break
        else:
            raise RuntimeError("entry-time VSB squeeze box is unavailable")
    else:
        frame1 = frame1.tail(60)  # Preserve the original adoption ATR seed.
    recovered = {**recovered, **frozen_strategy}
    atr_series = atr(frame1, settings.lookback.atr_len)
    entry_atr = float(atr_series.iloc[-1])
    if tag == "VSB":
        entry_atr = effective_entry_atr(frame1, entry_atr)
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
    checked_ms = int(_utc_now().timestamp() * 1000)
    market_price = _ticker_price(
        {symbol: ticker}, symbol, checked_ms, max_age_ms=DEMO_POSITION_QUOTE_MAX_AGE_MS,
    )
    if market_price is None:
        raise RuntimeError("fresh contract price is unavailable")
    observed_at = _ticker_observed_at({symbol: ticker}, symbol)
    if observed_at is None or observed_at > _utc_now() + timedelta(seconds=30):
        raise RuntimeError("contract price timestamp is invalid")

    candidate_position = Position(
        symbol, side, float(recovered["quantity"]), float(recovered["entry_price"]),
        float(strategy_stop), opened_at, "manual_position_adopted", regime_value, MEAN_REVERSION,
    )
    if previous_position is None:
        stop = execution.adopt_manual_position_stop(
            candidate_position, strategy_stop=strategy_stop, market_price=market_price,
        )
    else:
        stop = execution.reconcile_manual_position_stop(
            candidate_position, previous_position, previous_stop,
            stop_price=strategy_stop, market_price=market_price, new_cycle=True,
        )
    position = Position(
        symbol, side, float(recovered["quantity"]), float(recovered["entry_price"]),
        float(stop["stop_price"]), opened_at, "manual_position_adopted", regime_value, MEAN_REVERSION,
    )
    return (position, stop, recovered, entry_atr, swing_low, swing_high, market_price, observed_at)


def _install_lead_manual_position(execution, state, symbol, at, prepared):
    position, stop, recovered, entry_atr, swing_low, swing_high, market_price, observed_at = prepared
    opened_at, regime_value, side = position.opened_at, position.regime, position.side
    state.portfolio.positions[symbol] = position
    state.exchange_orders_by_symbol[symbol] = {"stop": stop}
    state.entry_costs[symbol] = {"entry_fee": float(recovered["entry_fee"])}
    state.mark_prices[symbol] = market_price
    _initialize_entry_metadata(
        state, symbol, position, True, market_price, float(stop["stop_price"]),
        entry_atr, swing_low, swing_high, observed_price=market_price,
        observed_at=observed_at, telemetry_history_complete=False,
    )
    if recovered.get("cursor"):
        state.exit_management_by_symbol[symbol]["manual_sync"] = dict(recovered["cursor"])
    state.exit_management_by_symbol[symbol].update(strategy_metadata(recovered))
    _invalidate_demo_stop_audit(state)
    state.events.append({
        "event": "manual_position_adopted", "at": at.isoformat(), "symbol": symbol,
        "side": side.value, "quantity": position.quantity,
        "entry_price": position.entry_price, "opened_at": opened_at.isoformat(),
        "price": position.entry_price,
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


def _adopt_lead_manual_position(execution, state, symbol, exchange_position, at):
    if not getattr(execution, "adopt_manual_positions", False):
        raise RuntimeError("manual adoption is disabled")
    if state.execution_mode != "lead_usdm" or symbol in state.portfolio.positions:
        raise RuntimeError("manual adoption is restricted to untracked Lead positions")
    _install_lead_manual_position(
        execution, state, symbol, at,
        _prepare_lead_manual_position(execution, symbol, exchange_position, at),
    )


def _manual_sync_error(exc):
    message = str(exc)
    safe = ("manual_sync_", "manual position trade history", "stop_too_",
            "strategy stop is invalid", "structure level is on the wrong side",
            "insufficient completed", "entry-time ")
    return message if message.startswith(safe) else type(exc).__name__


def _sync_lead_position_changes(execution, state, snapshot, at):
    """Commit verified manual changes only after the resulting position is protected."""
    blocked = []
    pending = {row["symbol"] for row in mr_scan.unresolved_attempts(state)}
    for symbol, position in list(state.portfolio.positions.items()):
        orders = state.exchange_orders_by_symbol.get(symbol, {})
        if symbol in pending or orders.get("close_attempt"):
            continue  # Existing durable bot order reconciliation owns these fills.
        meta = state.exit_management_by_symbol.setdefault(symbol, {})
        expected = snapshot["positions"].get(symbol)
        cursor = meta.get("manual_sync")
        stage = "fills"
        try:
            plan = execution.reconcile_manual_position(
                position, expected, at=at, cursor=cursor,
                entry_fee=(state.entry_costs.get(symbol) or {}).get("entry_fee"),
            )
            prepared = None
            remaining = plan["remaining"]
            if plan["changes"]:
                stage = "protection"
                if remaining is None:
                    execution._verify_manual_position_snapshot(symbol, None)
                    execution.cancel_saved_stop_if_open(symbol, orders.get("stop"))
                    execution._verify_manual_position_snapshot(symbol, None)
                elif plan["closed_cycles"]:
                    recovered = {**remaining, "cursor": plan["cursor"]}
                    prepared = _prepare_lead_manual_position(
                        execution, symbol, expected, at, recovered=recovered,
                        previous_position=position, previous_stop=orders.get("stop"),
                    )
                else:
                    ticker = execution.exchange.fetch_ticker(symbol)
                    checked_ms = int(_utc_now().timestamp() * 1000)
                    price = _ticker_price({symbol: ticker}, symbol, checked_ms,
                                          max_age_ms=DEMO_POSITION_QUOTE_MAX_AGE_MS)
                    if price is None:
                        raise RuntimeError("manual_sync_fresh_price_unavailable")
                    tracked_price = float((orders.get("stop") or {}).get("stop_price", position.stop_price))
                    recorded_price = float(meta.get("protective_stop", tracked_price))
                    current_stop = (max(tracked_price, recorded_price) if position.side is Side.LONG
                                    else min(tracked_price, recorded_price))
                    # Position.stop_price is an initial risk reference; the active
                    # trailing stop may already be on the profitable side of cost.
                    risk = float(meta.get("initial_risk") or abs(position.entry_price - position.stop_price))
                    reference = remaining["entry_price"] + (risk if position.side is Side.SHORT else -risk)
                    updated = replace(position, quantity=remaining["quantity"],
                                      entry_price=remaining["entry_price"], stop_price=reference)
                    saved_stop = execution.reconcile_manual_position_stop(
                        updated, position, orders.get("stop"), stop_price=current_stop, market_price=price,
                    )
                    prepared = (updated, saved_stop, price)
            # No mutation above this line: failures retain the previous ledger and
            # cursor, making a retry replay exactly the same confirmed changes.
            previous_realized = dict(meta.get("manual_realized", {}))
            for index, cycle in enumerate(plan["closed_cycles"]):
                totals = dict(cycle)
                reason = "external_close_exchange_trades_verified"
                resolver = getattr(execution, "resolve_protective_stop_algo_fill", None)
                if index == 0 and callable(resolver):
                    try:
                        stop_fill = resolver(symbol, orders.get("stop"), cycle["quantity"])
                        if stop_fill and _verified_fill_timestamp(stop_fill) == _aware_datetime(cycle["closed_at"]):
                            reason = "protective_stop_exchange_algo"
                    except Exception:
                        pass  # Fills are verified; an unavailable stop lookup cannot invent ownership.
                if index == 0:
                    for key in ("gross_pnl", "fees", "quantity", "notional"):
                        totals[key] += previous_realized.get(key, 0.0)
                funding_start = _aware_datetime(cycle["opened_at"])
                funding_end = _aware_datetime(cycle["closed_at"]) - timedelta(milliseconds=1)
                funding = (0.0 if funding_end < funding_start else _finite_pnl_value(
                    execution.funding_fee(symbol, funding_start, funding_end)))
                net = totals["gross_pnl"] - totals["fees"] + funding if funding is not None else None
                state.events.append({
                    "event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                    "side": cycle["side"], "at": cycle["closed_at"], "closed_at": cycle["closed_at"],
                    "opened_at": cycle["opened_at"], "exit_time_verified": True,
                    "reason": reason, "quantity": totals["quantity"],
                    "price": totals["notional"] / totals["quantity"], "gross_pnl": totals["gross_pnl"],
                    "entry_price": ((totals["notional"] - (1 if cycle["side"] == "long" else -1)
                                     * totals["gross_pnl"]) / totals["quantity"]),
                    "fees": totals["fees"], "funding_paid": funding, "net_pnl": net,
                    "funding_end_exclusive": True,
                    "pnl_recorded": net is not None, "source": MEAN_REVERSION,
                    "execution_source": _execution_source(state), "strategy_sample_eligible": False,
                    "sample_exclusion_reason": "manual_position_changed",
                })
                state.portfolio.last_exit_at[symbol] = _aware_datetime(cycle["closed_at"])
                state.portfolio.last_exit_category[symbol] = _mr_exit_category(reason, net)
                state.portfolio.last_exit_source[symbol] = MEAN_REVERSION
                state.exit_management_by_symbol.setdefault("__manual_entry_blocks__", {})[symbol] = cycle["closed_at"]
            for change in plan["changes"]:
                state.events.append({"event": "manual_position_changed", "symbol": symbol,
                                     **change, "execution_source": _execution_source(state),
                                     "strategy_sample_eligible": False})
            if remaining is None:
                state.portfolio.positions.pop(symbol, None)
                state.exchange_orders_by_symbol.pop(symbol, None)
                state.exit_management_by_symbol.pop(symbol, None)
                state.entry_costs.pop(symbol, None)
            elif plan["closed_cycles"]:
                _install_lead_manual_position(execution, state, symbol, at, prepared)
                meta = state.exit_management_by_symbol[symbol]
            elif plan["changes"]:
                updated, saved_stop, price = prepared
                state.portfolio.positions[symbol] = updated
                state.exchange_orders_by_symbol[symbol] = {"stop": saved_stop}
                state.entry_costs[symbol] = {"entry_fee": remaining["entry_fee"]}
                meta.update(protective_stop=float(saved_stop["stop_price"]),
                            best_price=max(updated.entry_price, price) if updated.side is Side.LONG else min(updated.entry_price, price),
                            worst_price=min(updated.entry_price, price) if updated.side is Side.LONG else max(updated.entry_price, price),
                            telemetry_history_complete=False, manual_position_changed=True)
            if remaining is not None:
                realized = dict(plan["realized"])
                if not plan["closed_cycles"]:
                    for key in realized:
                        realized[key] += previous_realized.get(key, 0.0)
                meta["manual_realized"] = realized
                meta["manual_sync"] = dict(plan["cursor"])
                meta.pop("manual_sync_error", None)
                state.entry_costs[symbol] = {"entry_fee": remaining["entry_fee"]}
            if plan["changes"]:
                _invalidate_demo_stop_audit(state)
        except Exception as exc:
            reason = _manual_sync_error(exc)
            if meta.get("manual_sync_error") != (stage, reason):
                state.events.append({"event": "manual_position_change_blocked", "at": at.isoformat(),
                                     "symbol": symbol, "stage": stage, "reason": reason,
                                     "execution_source": _execution_source(state)})
            meta["manual_sync_error"] = (stage, reason)
            blocked.append(symbol)
    if blocked:
        state.execution_status = "inconsistent"
        state.inconsistencies = [f"manual position change unresolved: {symbol}" for symbol in blocked]
        state.protective_stop_audit = {"at": at.isoformat(), "status": "error",
                                       "scope": "manual_position_change", "blocked_symbols": blocked,
                                       "mismatch_count": len(blocked), "mismatches": state.inconsistencies}
        return False
    if state.protective_stop_audit.get("scope") == "manual_position_change":
        _invalidate_demo_stop_audit(state)
    return True


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
        mismatches.extend(f"{symbol}: close reconciliation required"
                          for symbol in state.portfolio.positions
                          if state.exchange_orders_by_symbol.get(symbol, {}).get("close_attempt"))
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
            if state.protective_stop_audit.get("scope") != "manual_position_change":
                logger.error("{} position monitor paused: exchange state is not synchronized",
                             _execution_label(execution, state))
                return True

    blocked_symbols = set(state.protective_stop_audit.get("blocked_symbols", []))
    isolated = state.protective_stop_audit.get("scope") == "manual_position_change"
    if isolated:
        if not _demo_snapshot_is_fresh(state, now):
            return True
        ready = {symbol: position for symbol, position in state.portfolio.positions.items()
                 if symbol not in blocked_symbols}
        for symbol, position in ready.items():
            observed = state.exchange_snapshot.get("positions", {}).get(symbol)
            if (observed is None or observed.get("side") != position.side.value
                    or not math.isclose(float(observed["quantity"]), position.quantity, rel_tol=1e-8)
                    or not math.isclose(float(observed["entry_price"]), position.entry_price, rel_tol=1e-6)):
                return True
        audit = state.protective_stop_audit
        verified_at = _aware_datetime(audit.get("unaffected_verified_at"))
        if verified_at is None or (now - verified_at).total_seconds() >= 60:
            try:
                if execution.audit_active_protective_stops(ready, state.exchange_orders_by_symbol):
                    return True
            except Exception:
                return True
            audit["unaffected_verified_at"] = now.isoformat()
    else:
        if state.execution_status != "synchronized":
            if recent_audit_error or not _audit_demo_protective_stops(execution, state, now):
                return True
        if _demo_stop_audit_due(state, now) and not _audit_demo_protective_stops(execution, state, now):
            return True
        if state.execution_status != "synchronized":
            return True

    dirty = False
    if not state.portfolio.positions:
        return dirty
    if not exchange.markets:
        exchange.load_markets()

    stops = settings.stops
    for symbol, position in list(state.portfolio.positions.items()):
        if symbol in blocked_symbols:
            continue
        try:
            ticker = exchange.fetch_ticker(symbol)
            quote_error = None
        except Exception as exc:
            ticker = {}
            quote_error = type(exc).__name__

        checked_at = _utc_now()
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
        if price is None or not _position_sample_is_eligible(position, observed_at):
            state.exit_management_by_symbol[symbol] = meta
            if _apply_max_hold_exit(True, execution, state, symbol, position, None,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue

        _update_demo_excursions(meta, position, price=price, observed_at=observed_at)
        state.mark_prices[symbol] = price
        entry = float(position.entry_price)
        best_price = float(meta.get("best_price", entry))
        worst_price = float(meta.get("worst_price", entry))
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
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
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
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue
        improved = (proposed_stop > current_stop if position.side is Side.LONG
                    else proposed_stop < current_stop)
        if not improved:
            if _mark_trailing_started(
                    state, symbol, position, meta, progress_r, current_stop, proposed_stop,
                    settings.mr_candidate_exit.trailing_trigger_r, checked_at):
                dirty = True
            state.exit_management_by_symbol[symbol] = meta
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue
        if ((position.side is Side.LONG and proposed_stop >= price)
                or (position.side is Side.SHORT and proposed_stop <= price)):
            state.exit_management_by_symbol[symbol] = meta
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue

        first_breakeven_move = _stop_is_less_protective(position.side, current_stop, locked_stop)
        if not first_breakeven_move:
            last_update = _aware_datetime(meta.get("last_fast_stop_update_at"))
            if (last_update is not None
                    and (checked_at - last_update).total_seconds()
                    < DEMO_STOP_UPDATE_MIN_INTERVAL_SECONDS):
                state.exit_management_by_symbol[symbol] = meta
                if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                        checked_at, settings, fee_rate, slip_rate):
                    dirty = True
                continue

        if state.protective_stop_audit.get("status") != "verified":
            state.exit_management_by_symbol[symbol] = meta
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue
        try:
            updated = execution.replace_stop(position, stop=tracked_order,
                                             stop_price=proposed_stop)
        except Exception as exc:
            _invalidate_demo_stop_audit(state)
            logger.warning("{} stop move failed for {}: {}",
                           _execution_label(execution, state), symbol, type(exc).__name__)
            state.exit_management_by_symbol[symbol] = meta
            if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                    checked_at, settings, fee_rate, slip_rate):
                dirty = True
            continue

        state.exchange_orders_by_symbol.setdefault(symbol, {})["stop"] = updated
        actual_stop = float(updated.get("stop_price", proposed_stop))
        meta["protective_stop"] = actual_stop
        meta["last_fast_stop_update_at"] = checked_at.isoformat()
        if _mark_trailing_started(
                state, symbol, position, meta, progress_r, actual_stop, proposed_stop,
                settings.mr_candidate_exit.trailing_trigger_r, checked_at):
            dirty = True
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
        if _apply_max_hold_exit(True, execution, state, symbol, position, price,
                                checked_at, settings, fee_rate, slip_rate):
            dirty = True

    return dirty


# --- universe + bars ------------------------------------------------------------------

def _closed(frame_rows, timeframe_ms, now_ms):
    if active_clock() is not None:
        now_ms = min(now_ms, int(active_clock().lower_now().timestamp() * 1000))
    return [r for r in frame_rows if len(r) >= 6 and int(r[0]) + timeframe_ms <= now_ms]


def _daily_direction_gate(exchange, symbol, signal_at, span, *, fail_on_market_error=False,
                          prefetched=None, atr_period=14, minimum_bars=55):
    """Return a fail-closed direction from completed UTC daily candles only."""
    signal_at = signal_at.replace(tzinfo=timezone.utc) if signal_at.tzinfo is None else signal_at
    signal_at = signal_at.astimezone(timezone.utc)
    signal_ms = int(signal_at.timestamp() * 1000)
    try:
        request = (symbol, "1d", max(span + 1, 100))
        rows = prefetched_read(prefetched if prefetched is not None else {}, request,
            lambda: exchange.fetch_ohlcv(symbol, "1d", limit=request[2]))
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
        if len(frame) < max(span + 1, minimum_bars):
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
        daily_atr = float(atr(frame, atr_period).iloc[-1])
        if not math.isfinite(daily_atr) or daily_atr < 0:
            raise ValueError("invalid daily ATR")
        previous_days = frame.iloc[-8:-1]
        previous_quote_volumes = (
            previous_days["volume"]
            * (previous_days["high"] + previous_days["low"] + previous_days["close"]) / 3
        )
        if len(previous_quote_volumes) != 7 or not all(
                math.isfinite(float(value)) and float(value) > 0
                for value in previous_quote_volumes):
            raise ValueError("invalid daily quote volume baseline")
        daily_metrics = {
            "daily_bars": len(frame), "daily_close": daily_close,
            "daily_ema50": daily_ema, "daily_atr": daily_atr,
            "daily_ema_distance_atr": (
                abs(daily_close - daily_ema) / daily_atr if daily_atr > 0 else None
            ),
            "daily_quote_volume_7d_avg": float(previous_quote_volumes.mean()),
            "daily_bar_close_time": (frame.index[-1].to_pydatetime()
                                     + timedelta(days=1)).isoformat(),
        }
        # Evidence used by the Lead structure pool. Existing direction callers
        # retain their original result; an incomplete pool input stays pending.
        try:
            validate_ohlcv(frame)
            daily_metrics["daily_history_valid"] = bool(
                (frame[["open", "high", "low", "close"]] > 0).all().all()
                and (frame.index.to_series().diff().dropna() == timedelta(days=1)).all())
        except (TypeError, ValueError):
            daily_metrics["daily_history_valid"] = False
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
            "reason": "daily_close_equals_ema50", **daily_metrics,
            "signal_at": signal_at.isoformat(),
        }

    return {
        "ready": True, "direction": direction, "reason": None,
        **daily_metrics,
        "signal_at": signal_at.isoformat(),
    }


def _cached_daily_direction_gate(exchange, symbol, signal_at, span, cache, **kwargs):
    if cache is None:
        return _daily_direction_gate(exchange, symbol, signal_at, span, **kwargs)
    key = (int(signal_at.timestamp() * 1000) // DAY_MS, span,
           kwargs.get("atr_period", 14), kwargs.get("minimum_bars", 55))
    saved = cache.daily_direction.get(symbol)
    if saved is not None and saved[0] == key:
        return {**saved[1], "signal_at": signal_at.isoformat()}
    result = _daily_direction_gate(exchange, symbol, signal_at, span, **kwargs)
    if result.get("daily_history_valid") is not False and result.get("reason") not in {
            "daily_data_unavailable", "daily_data_invalid", "daily_data_stale"}:
        cache.daily_direction[symbol] = (key, result)
    return result


def _cached_four_hour(exchange, symbol, now_ms, cache, *, prefetched=None):
    key = now_ms // FOUR_H_MS
    saved = cache.four_hour.get(symbol) if cache is not None else None
    if saved is not None and saved[0] == key:
        return saved[1]
    rows = _closed(prefetched_read(prefetched if prefetched is not None else {}, (symbol, "4h", 100),
        lambda: exchange.fetch_ohlcv(symbol, "4h", limit=100)), FOUR_H_MS, now_ms)
    if not rows or int(rows[-1][0]) + FOUR_H_MS != key * FOUR_H_MS:
        return None
    try:
        frame = ohlcv_frame(rows)
        validate_ohlcv(frame)
        if ((frame[["open", "high", "low", "close"]] <= 0).any().any()
                or not (frame.index.to_series().diff().dropna() == timedelta(hours=4)).all()):
            return None
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
        "daily_atr", "daily_ema_distance_atr", "daily_quote_volume_7d_avg",
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
    now = now or _utc_now()
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


def _lead_structure(frame1, frame4, gate, settings, at):
    """Structure only: no candle colour, EMA9, time filter or account gates."""
    try:
        for frame, hours, minimum in ((frame1, 1, 25),
                                      (frame4, 4, settings.symbol_regime.ema_slow_bars)):
            validate_ohlcv(frame)
            boundary = pd.Timestamp(at).floor(f"{hours}h")
            if (len(frame) < minimum or (frame[["open", "high", "low", "close"]] <= 0).any().any()
                    or frame.index[-1] + timedelta(hours=hours) != boundary
                    or not (frame.index.to_series().diff().dropna() == timedelta(hours=hours)).all()):
                raise ValueError("incomplete closed candles")
        close = float(frame1["close"].iloc[-1])
        ema = float(frame4["close"].ewm(span=settings.symbol_regime.ema_slow_bars,
                    adjust=False, min_periods=settings.symbol_regime.ema_slow_bars).mean().iloc[-1])
        atr4, atr_pct, threshold = _atr_deviation_threshold(frame4, settings)
        if threshold is None or not math.isfinite(ema) or ema <= 0:
            raise ValueError("structure indicators unavailable")
        squeeze = squeeze_context(frame1)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, OverflowError):
        return {"status": "pending", "reason": "structure_data_unavailable"}
    if gate.get("daily_history_valid") is False:
        return {"status": "pending", "reason": "daily_history_invalid"}
    if not gate.get("ready"):
        reason = gate.get("reason") or "daily_data_unavailable"
        return {"status": "removed" if reason == "daily_close_equals_ema50" else "pending",
                "reason": reason}
    side = Side(gate["direction"])
    adverse = (ema - close) / ema if side is Side.LONG else (close - ema) / ema
    details = {"close": close, "closed_ema50": ema, "deviation_pct": close / ema - 1,
               "closed_4h_atr": atr4, "closed_4h_atr_pct": atr_pct,
               "deviation_threshold_pct": threshold, "adverse_dev": adverse,
               **squeeze, **_daily_gate_status_fields(gate)}
    if not getattr(settings.signals, f"allow_{side.value}"):
        return {**details, "status": "removed", "reason": "daily_structure_direction_changed"}
    # A released candle has zero *current* squeeze bars. Its preceding box is
    # retained for this signal hour only, then disappears on the next scan.
    current_sqz = squeeze["sqz_bars"] if squeeze["squeeze_on"] else 0
    mr_side = deviation_structure_side(ema, close, settings, deviation_pct=threshold)
    near_ema = (close <= ema * (1 + 1.2 * threshold) if side is Side.LONG
                else close >= ema * (1 - 1.2 * threshold))
    if mr_side is side and current_sqz < 6:
        tag = "MR"
    elif (mr_side is not side and near_ema and squeeze["squeeze_complete"]
          and 6 <= squeeze["sqz_bars"] <= 36):
        tag = "VSB"
    else:
        return {**details, "status": "removed", "reason": "deviation_structure_not_met"}
    return {**details, "status": "active", "side": side.value, "strategy_tag": tag, "reason": None}


def _update_lead_pool(state, observations, at):
    """Keep identity/history through read failures, ending only verified structures."""
    counts = dict(added=0, retained=0, removed=0, pending=0)
    bar = (at.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)).isoformat()
    previous = state.mr_candidate_pool
    pool = {}
    for symbol in sorted(set(previous) | set(observations)):
        old = previous.get(symbol)
        result = observations.get(symbol, {"status": "pending", "reason": "universe_data_missing"})
        status = result["status"]
        changed = old and status == "active" and (
            old.get("side") != result.get("side")
            or old.get("strategy_tag", "MR") != result.get("strategy_tag", "MR"))
        if old and (status == "removed" or changed):
            counts["removed"] += 1
            state.events.append({"event": "mr_candidate_removed", "at": at.isoformat(),
                                 "symbol": symbol, "side": old.get("side"),
                                 "first_seen_at": old.get("first_seen_at"),
                                 "bar": bar, "reason": "route_or_direction_changed" if changed else result["reason"]})
        if status == "removed":
            continue
        if status == "pending":
            counts["pending"] += 1
            _set_symbol_status(state, symbol, reason=result["reason"], at=at)
            state.symbol_status[symbol]["candidate_status"] = "pending"
            # There is no verified candidate identity for a never-seen failed read.
            if old is None:
                continue
        base = {} if changed or old is None else old
        first = base.get("first_seen_at") or at.isoformat()
        pool[symbol] = {**base, **result, "strategy_tag": result.get("strategy_tag", base.get("strategy_tag", "MR")), "symbol": symbol,
                        "first_seen_at": first, "checked_at": at.isoformat(), "bar": bar,
                        "pool_age_hours": max(0, (at - datetime.fromisoformat(first)).total_seconds() / 3600)}
        if status == "active":
            counts["retained" if base else "added"] += 1
    state.mr_candidate_pool = pool
    return counts


def _lead_scan_recoveries(scan, symbols):
    return [symbol for symbol in symbols if symbol not in scan.get("symbols", [])
            or (scan.get("outcomes", {}).get(symbol, {}).get("status") == "blocked"
                and scan["outcomes"][symbol].get("reason") == "candidate_structure_unverified")]


def _build_lead_universe(exchange, state, settings, now, *, lead_market_ids,
                         cache, candle_reader=None, on_progress=None):
    """Scan every eligible Lead contract; the old score never caps coverage."""
    started = time.monotonic()
    at = now.replace(minute=0, second=0, microsecond=0)
    now_ms = int(at.timestamp() * 1000)
    bar = (at - timedelta(hours=1)).isoformat()
    # Process-local cache is never trusted across a runner restart.
    if cache.lead_universe_at is not None and cache.lead_universe_at.replace(
            minute=0, second=0, microsecond=0) == at:
        last = cache.lead_universe[-1]
        if not (last.get("pending", 0) and _utc_now() <= at + timedelta(
                seconds=settings.execution.confirm_delay_sec_max)
                and (now - cache.lead_universe_at).total_seconds() >= 5):
            return cache.lead_universe[:6]
    # A failed or incomplete global response suspends eligibility; it never
    # turns absent symbols into a proven structural failure.
    observations = {}
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
    except ccxt.BaseError:
        _update_lead_pool(state, {}, at)
        cache.lead_universe_at = None
        raise
    candidates = _filter_lead_market_candidates(
        filter_tradable_crypto_universe(markets, tickers, settings, now=at), markets, lead_market_ids)
    eligible = {symbol for symbol, _ in candidates}
    for symbol in state.mr_candidate_pool:
        if symbol not in eligible:
            observations[symbol] = {"status": "pending", "reason": "universe_eligibility_unverified"}
            # Only explicit, complete market+ticker information can confirm
            # that an existing candidate has lost basic trading eligibility.
            ticker = tickers.get(symbol) or {}
            try:
                volume = float(ticker["quoteVolume"])
                market = markets.get(symbol) or {}
                info = market.get("info") or {}
                verified = (math.isfinite(volume) and volume >= 0
                            and {"id", "active", "swap", "quote", "settle"}.issubset(market)
                            and {"contractType", "underlyingType", "underlyingSubType", "onboardDate"}.issubset(info))
            except (KeyError, TypeError, ValueError, OverflowError):
                verified = False
            if verified:
                observations[symbol] = {"status": "removed", "reason": "trading_eligibility_lost"}
    span = settings.symbol_regime.ema_slow_bars
    daily_key = (now_ms // DAY_MS, span, settings.lookback.atr_len, MIN_DAILY_UNIVERSE_BARS)
    requests, hourly_limits = [], {}
    for symbol, _ in candidates:
        saved = cache.lead_squeeze_hourly.get(symbol)
        limit = (2 if saved is not None and len(saved) >= 119
                 and saved.index[-1] == pd.Timestamp(at - timedelta(hours=2)) else 120)
        hourly_limits[symbol] = limit
        requests.append((symbol, "1h", limit))
        if cache.daily_direction.get(symbol, (None,))[0] != daily_key:
            requests.append((symbol, "1d", max(span + 1, 100)))
        if cache.four_hour.get(symbol, (None,))[0] != now_ms // FOUR_H_MS:
            requests.append((symbol, "4h", 100))
    try:
        prefetched = candle_reader.read_many(requests, _public_read_deadline(now, settings),
                                            monitor=on_progress) if candle_reader is not None else {}
    except ccxt.BaseError:
        _update_lead_pool(state, {}, now)
        cache.lead_universe_at = None
        raise
    one_hour, current_hour = {}, {}
    for symbol, volume in candidates:
        if on_progress is not None:
            on_progress()
        try:
            limit = hourly_limits[symbol]
            rows = prefetched_read(prefetched, (symbol, "1h", limit),
                                   lambda: exchange.fetch_ohlcv(symbol, "1h", limit=limit))
            frame1 = ohlcv_frame(_closed(rows, HOUR_MS, now_ms))
            has_unfinished = any(int(row[0]) >= now_ms for row in rows)
            if limit == 2:
                frame1 = pd.concat([cache.lead_squeeze_hourly[symbol], frame1])
                closed_count = 119 if has_unfinished else 120
                frame1 = frame1.loc[~frame1.index.duplicated(keep="last")].sort_index().tail(closed_count)
            frame1.attrs["mr_window_bars"] = 59 if has_unfinished else 60
            frame4 = _cached_four_hour(exchange, symbol, now_ms, cache, prefetched=prefetched)
            gate = _cached_daily_direction_gate(exchange, symbol, at, span, cache,
                                                prefetched=prefetched, atr_period=settings.lookback.atr_len,
                                                fail_on_market_error=True)
            result = _lead_structure(frame1, frame4, gate, settings, at)
            if frame4 is not None and len(frame4) < span:
                cache.four_hour.pop(symbol, None)
            if not gate.get("ready"):
                cache.daily_direction.pop(symbol, None)
            if result["reason"] != "structure_data_unavailable":
                cache.lead_squeeze_hourly[symbol] = frame1
                # Preserve Wilder ATR and EMA seeds of the existing MR window.
                cache.lead_hourly[symbol] = frame1.tail(59 if has_unfinished else 60)
            one_hour[symbol] = (frame1 if result.get("strategy_tag") == "VSB"
                                else frame1.tail(59 if has_unfinished else 60))
            observations[symbol] = {**result, "quote_volume_24h": volume}
        except ccxt.BaseError as exc:
            if isinstance(exc, (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.AuthenticationError)):
                _update_lead_pool(state, {}, now)
                cache.lead_universe_at = None
                raise
            observations[symbol] = {"status": "pending", "reason": type(exc).__name__}
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            observations[symbol] = {"status": "pending", "reason": "structure_data_unavailable"}
    prepared = time.monotonic()
    counts = _update_lead_pool(state, observations, now)
    hotlist = [dict(row, rank=index) for index, row in enumerate(
        sorted(state.mr_candidate_pool.values(), key=lambda row: row["symbol"]), 1)]
    summary = {"event": "mr_candidate_pool_refreshed", "at": now.isoformat(), "bar": bar,
               "eligible": len(candidates), "active": sum(row["status"] == "active" for row in hotlist),
               **counts, "data_prepare_seconds": round(prepared - started, 4),
               "pool_refresh_seconds": round(time.monotonic() - prepared, 4),
               "within_entry_deadline": _utc_now() <= at + timedelta(seconds=settings.execution.confirm_delay_sec_max)}
    state.events.append(summary)
    logger.info("Lead MR candidate pool {}", json.dumps(summary, ensure_ascii=False))
    cache.lead_universe = (markets, hotlist, one_hour, current_hour, tickers, [], counts)
    cache.lead_universe_at = now
    return cache.lead_universe[:6]


def _build_universe(exchange, settings, now, now_ms, *, fail_on_market_error=False,
                    on_progress=None, retry_reads=False, candle_reader=None,
                    closed_market_cache=None, lead_market_ids=None):
    deadline = now.replace(minute=0, second=0, microsecond=0) + timedelta(
        seconds=settings.execution.confirm_delay_sec_max)
    read_kwargs = {"retry": retry_reads, "deadline": deadline, "monitor": on_progress}
    markets = _read_entry_data(exchange.load_markets, phase="universe_markets", **read_kwargs)
    tickers = _read_entry_data(exchange.fetch_tickers, phase="universe_tickers", **read_kwargs)
    basic_candidates = filter_tradable_crypto_universe(markets, tickers, settings, now=now)
    candidates = (basic_candidates if lead_market_ids is None else
                  _filter_lead_market_candidates(basic_candidates, markets, lead_market_ids))
    span = settings.symbol_regime.ema_slow_bars
    day_key = int(now_ms) // DAY_MS
    four_hour_key = int(now_ms) // FOUR_H_MS
    daily_cache = closed_market_cache.daily_direction if closed_market_cache is not None else {}
    four_hour_cache = (closed_market_cache.universe_four_hour
                       if closed_market_cache is not None else {})
    requests = [(symbol, "1h", 60) for symbol, _ in basic_candidates]
    for symbol, _ in candidates:
        daily_saved = daily_cache.get(symbol)
        daily_key = (day_key, span, settings.lookback.atr_len, MIN_DAILY_UNIVERSE_BARS)
        if daily_saved is None or daily_saved[0] != daily_key:
            requests.append((symbol, "1d", max(span + 1, 100)))
        four_saved = four_hour_cache.get(symbol)
        if four_saved is None or four_saved[0] != four_hour_key:
            requests.append((symbol, "4h", 15))
    prefetched = (candle_reader.read_many(
        requests, _public_read_deadline(now, settings), monitor=on_progress,
    ) if candle_reader is not None else {})
    one_hour: dict[str, pd.DataFrame] = {}
    current_hour: dict[str, dict] = {}
    activity: dict[str, dict] = {}
    for symbol, _q in basic_candidates:
        if on_progress is not None:
            on_progress()
        try:
            rows = _read_entry_data(
                lambda: prefetched_read(prefetched, (symbol, "1h", 60),
                    lambda: exchange.fetch_ohlcv(symbol, "1h", limit=60)),
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

    # Keep the former ranking for the Lead whitelist comparison event.
    legacy_scored = score_hot_candidates(basic_candidates, tickers, activity)
    legacy_hotlist = [{**row, "rank": rank}
                      for rank, row in enumerate(legacy_scored[:TOP_N], start=1)]

    try:
        book_tickers = exchange.fetch_bids_asks([symbol for symbol, _ in candidates]) if candidates else {}
    except (AttributeError, NotImplementedError):
        book_tickers = {}
    except ccxt.BaseError as exc:
        _raise_market_error_if_needed(exc, fail_on_market_error)
        logger.warning("Universe best-bid/ask data unavailable; spread scores will be missing: {}",
                       type(exc).__name__)
        book_tickers = {}

    def read_rows(symbol, timeframe, limit):
        request = (symbol, timeframe, limit)
        return _read_entry_data(
            lambda: prefetched_read(prefetched, request,
                                    lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=limit)),
            phase=f"universe_{timeframe}", symbol=symbol, **read_kwargs,
        )

    daily_ready = {}
    rejection_counts = {}
    four_hour_metrics = {}
    for symbol, _quote_volume in candidates:
        if on_progress is not None:
            on_progress()
        gate = _cached_daily_direction_gate(
            exchange, symbol, now, span, closed_market_cache,
            fail_on_market_error=fail_on_market_error,
            prefetched=prefetched, atr_period=settings.lookback.atr_len,
            minimum_bars=MIN_DAILY_UNIVERSE_BARS,
        )
        if not gate.get("ready"):
            reason = gate.get("reason") or "daily_data_unavailable"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        daily_ready[symbol] = gate

        cached_four = four_hour_cache.get(symbol)
        if cached_four is not None and cached_four[0] == four_hour_key:
            four_metrics = cached_four[1]
        else:
            try:
                rows4 = read_rows(symbol, "4h", 15)
                closed4 = _closed(rows4, FOUR_H_MS, now_ms)
                frame4 = ohlcv_frame(closed4)
                validate_ohlcv(frame4)
                if len(frame4) < settings.lookback.atr_len:
                    raise ValueError("four-hour history insufficient")
                atr4 = float(atr(frame4, settings.lookback.atr_len).iloc[-1])
                close4 = float(frame4["close"].iloc[-1])
                if not math.isfinite(atr4) or atr4 <= 0 or not math.isfinite(close4) or close4 <= 0:
                    raise ValueError("four-hour ATR unavailable")
                four_metrics = {"natr_4h": atr4 / close4}
                four_hour_cache[symbol] = (four_hour_key, four_metrics)
            except (ccxt.BaseError, KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
                _raise_market_error_if_needed(exc, fail_on_market_error)
                rejection_counts["four_hour_data_unavailable"] = (
                    rejection_counts.get("four_hour_data_unavailable", 0) + 1
                )
                continue
        four_hour_metrics[symbol] = four_metrics

    metrics = []
    for symbol, quote_volume in candidates:
        gate = daily_ready.get(symbol)
        frame1 = one_hour.get(symbol)
        four_metrics = four_hour_metrics.get(symbol)
        if gate is None or frame1 is None or four_metrics is None or len(frame1) < 24:
            if gate is not None:
                rejection_counts["hourly_range_unavailable"] = (
                    rejection_counts.get("hourly_range_unavailable", 0) + 1
                )
            continue
        window = frame1.iloc[-24:]
        first_open = float(window["open"].iloc[0])
        last_close = float(window["close"].iloc[-1])
        range_24h = float(window["high"].max() - window["low"].min())
        if (not all(math.isfinite(value) for value in (first_open, last_close, range_24h))
                or first_open <= 0 or range_24h <= 0):
            rejection_counts["hourly_range_invalid"] = rejection_counts.get("hourly_range_invalid", 0) + 1
            continue

        ticker = tickers.get(symbol) or {}
        info = ticker.get("info") or {}
        try:
            raw_count = info.get("count", ticker.get("count"))
            trade_count = float(raw_count) if raw_count is not None else None
        except (TypeError, ValueError, OverflowError):
            trade_count = None
        if trade_count is not None and (not math.isfinite(trade_count) or trade_count < 0):
            trade_count = None

        book = book_tickers.get(symbol) or {}
        try:
            bid = float(book.get("bid"))
            ask = float(book.get("ask"))
            bid_volume = float(book.get("bidVolume"))
            ask_volume = float(book.get("askVolume"))
            if (not all(math.isfinite(value) and value > 0
                        for value in (bid, ask, bid_volume, ask_volume)) or bid >= ask):
                raise ValueError("invalid best bid/ask")
            spread_pct = (ask - bid) / ((ask + bid) / 2)
            top_book_depth = min(bid * bid_volume, ask * ask_volume)
        except (TypeError, ValueError, OverflowError):
            spread_pct = top_book_depth = None

        daily_volume_average = gate.get("daily_quote_volume_7d_avg")
        hourly_activity = activity.get(symbol, {})
        metrics.append({
            "symbol": symbol,
            "quote_volume_24h": quote_volume,
            "trade_count_24h": trade_count,
            "volume_burst_ratio": hourly_activity.get("volume_burst_ratio"),
            "return_4h_pct": hourly_activity.get("return_4h_pct"),
            "return_24h_pct": hourly_activity.get("return_24h_pct"),
            "volume_ratio_24h_7d": (
                quote_volume / daily_volume_average if daily_volume_average else None
            ),
            "natr_4h": four_metrics["natr_4h"],
            "range_24h": range_24h,
            "net_move_24h": abs(last_close - first_open),
            "spread_pct": spread_pct,
            "top_book_depth_usdt": top_book_depth,
            "daily_ema_distance_atr": gate.get("daily_ema_distance_atr"),
            "daily_bars": gate.get("daily_bars"),
            "daily_direction": gate.get("direction"),
        })

    scored = rank_mean_reversion_candidates(metrics)
    hotlist = [{**row, "rank": rank}
               for rank, row in enumerate(scored[:TOP_N], start=1)]
    daily_distance_shadow_top_n = [
        {"symbol": row["symbol"], "rank": row["rank"],
         "distance_atr": row["daily_ema_distance_atr"]}
        for row in hotlist
        if row.get("daily_ema_distance_atr") is not None
        and row["daily_ema_distance_atr"] <= 0.25
    ]
    logger.info("MR universe funnel {}", json.dumps({
        "basic_tradable": len(basic_candidates),
        "lead_whitelist": len(candidates),
        "daily_and_volatility_ready": len(daily_ready),
        "book_quote_ready": sum(
            1 for row in metrics if row["spread_pct"] is not None
            and row["top_book_depth_usdt"] is not None
        ),
        "rankable": len(metrics),
        "selected_top_n": len(hotlist),
        "rejections": rejection_counts,
        "daily_distance_shadow_rejections": sum(
            1 for row in metrics if row.get("daily_ema_distance_atr") is not None
            and row["daily_ema_distance_atr"] <= 0.25
        ),
        "daily_distance_shadow_top_n": daily_distance_shadow_top_n,
    }, ensure_ascii=False, sort_keys=True))
    return markets, hotlist, one_hour, current_hour, tickers, legacy_hotlist


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
    except (ccxt.BaseError, AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ccxt.BaseError):
            _raise_market_error_if_needed(exc, fail_on_market_error)
        logger.warning("funding rate unavailable for {}; entry will fail closed: {}",
                       symbol, type(exc).__name__)
        return None


def _ema9_reclaim_side(frame1):
    if frame1 is None or len(frame1) < EMA9_RECLAIM_PERIOD + 1:
        return None
    closes = frame1["close"].astype(float)
    ema9 = closes.ewm(span=EMA9_RECLAIM_PERIOD, adjust=False,
                      min_periods=EMA9_RECLAIM_PERIOD).mean()
    previous_close, current_close = float(closes.iloc[-2]), float(closes.iloc[-1])
    previous_ema, current_ema = float(ema9.iloc[-2]), float(ema9.iloc[-1])
    if not all(math.isfinite(value) for value in
               (previous_close, current_close, previous_ema, current_ema)):
        return None
    if previous_close <= previous_ema and current_close > current_ema:
        return "long"
    if previous_close >= previous_ema and current_close < current_ema:
        return "short"
    return None


def _entry_execution_plan(candle, atr_value, bar_close_at, side):
    candle_range = float(candle["high"]) - float(candle["low"])
    candle_range_atr = candle_range / atr_value if atr_value > 0 else float("inf")
    close = float(candle["close"])
    body = abs(close - float(candle["open"]))
    limit_price = None
    expires_at = None
    if candle_range_atr > DYNAMIC_LIMIT_RANGE_ATR:
        direction = 1 if side is Side.LONG else -1
        limit_price = close - direction * DYNAMIC_LIMIT_BODY_RETRACE * body
        expires_at = bar_close_at + timedelta(hours=DYNAMIC_LIMIT_TTL_HOURS)
    return {
        "candle_range_atr": candle_range_atr,
        "order_type": "limit" if limit_price is not None else "market",
        "limit_price": limit_price,
        "limit_expires_at": expires_at.isoformat() if expires_at else None,
        "limit_body_retrace": DYNAMIC_LIMIT_BODY_RETRACE if limit_price is not None else None,
        "signal_bar": bar_close_at.isoformat(),
    }


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
                                 flags, settings, now, *, scan_row=None):
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
        if scan_row is not None:
            scan_row.update(reason=event["reason"], phase="time_filter")
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
                    state.exit_management_by_symbol[symbol].update(strategy_metadata(meta))
                    state.exit_management_by_symbol[symbol]["telemetry_history_complete"] = False
                state.exchange_orders_by_symbol[symbol] = {"stop": result["stop"]}
                attempt["status"] = "completed"
        elif status == "not_filled" and symbol not in state.portfolio.positions:
            attempt["status"] = "not_filled"
        elif status == "pending":
            attempt["status"] = "limit_pending"
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
           checkpoint=None, candle_reader=None):
    now_ms = int(now.timestamp() * 1000)
    fee_rate = settings.fees.taker_bps / 10_000
    slip_rate = settings.fees.slippage_bps / 10_000
    leverage = int(settings.sizing.leverage)
    lead_market_ids = (getattr(execution, "lead_market_ids", set())
                       if state.execution_mode == "lead_usdm" else None)
    if lead_market_ids is not None and closed_market_cache is None:
        closed_market_cache = ClosedMarketCache()
    if is_demo:
        _reconcile_pending_entries(execution, state, checkpoint)
    resume_symbols = (state.mr_entry_scan.get("symbols", [])
                      if is_demo and state.mr_entry_scan
                      and not mr_scan.expired(state.mr_entry_scan, _utc_now()) else [])
    local_pending_symbols = sorted({
        row.get("symbol") for row in state.mr_entry_attempts.values()
        if state.execution_mode == "local_paper"
        and row.get("status") == "paper_limit_pending" and row.get("symbol")
    })

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
            markets, hotlist, one_hour, current_hour, _universe_tickers, legacy_hotlist = _build_universe(
                exchange, settings, now, now_ms, fail_on_market_error=True,
                closed_market_cache=closed_market_cache, lead_market_ids=lead_market_ids,
            )
            market_cache.markets = markets
            market_cache.one_hour.update(one_hour)
            market_cache.current_hour.update(current_hour)
        else:
            markets = market_cache.markets
            hotlist = state.watchlist
            legacy_hotlist = hotlist

        top_symbols = [row["symbol"] for row in hotlist]
        track = sorted(set(top_symbols) | set(state.portfolio.positions)
                       | set(resume_symbols) | set(local_pending_symbols))
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
        track = sorted(set(top_symbols) | set(state.portfolio.positions)
                       | set(resume_symbols) | set(local_pending_symbols))
        one_hour, current_hour, four_hour, regimes = _fetch_selected_bars(
            exchange, track, settings, now_ms, fail_on_market_error=True,
        )
    else:
        if lead_market_ids is not None:
            markets, hotlist, one_hour, current_hour, tickers, legacy_hotlist = _build_lead_universe(
                exchange, state, settings, now, lead_market_ids=lead_market_ids,
                cache=closed_market_cache, candle_reader=candle_reader, on_progress=position_monitor,
            )
        else:
            batch_kwargs = {"candle_reader": candle_reader} if candle_reader is not None else {}
            markets, hotlist, one_hour, current_hour, tickers, legacy_hotlist = _build_universe(
                exchange, settings, now, now_ms, fail_on_market_error=intrabar_entries,
                on_progress=position_monitor, retry_reads=is_demo,
                closed_market_cache=closed_market_cache, **batch_kwargs,
            )
        top_symbols = [row["symbol"] for row in hotlist]
        track = sorted(set(top_symbols) | set(state.portfolio.positions)
                       | set(resume_symbols) | set(local_pending_symbols))
        four_hour: dict[str, pd.DataFrame] = {}
        regimes: dict[str, pd.DataFrame] = {}
        prefetched = {}
        if candle_reader is not None and is_demo:
            requests = [(symbol, "4h", 100) for symbol in track
                        if not closed_market_cache or not closed_market_cache.four_hour.get(symbol)
                        or closed_market_cache.four_hour[symbol][0] != now_ms // FOUR_H_MS]
            prefetched = candle_reader.read_many(requests, _public_read_deadline(now, settings),
                                                  monitor=position_monitor)
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
                    kwargs = {"prefetched": prefetched} if candle_reader is not None else {}
                    frame4 = _cached_four_hour(exchange, symbol, now_ms, closed_market_cache, **kwargs)
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

    if local_pending_symbols:
        for symbol in local_pending_symbols:
            try:
                tickers[symbol] = exchange.fetch_ticker(symbol)
            except ccxt.BaseError as exc:
                logger.warning("Local dynamic limit quote unavailable for {}: {}",
                               symbol, type(exc).__name__)

    ranked_hotlist = legacy_hotlist if state.execution_mode == "lead_usdm" else hotlist
    state.watchlist = hotlist
    if intrabar_entries and not _hotlist_cache_is_fresh(state, now):
        state.exit_management_by_symbol[HOTLIST_CACHE_KEY] = {"refreshed_at": now.isoformat()}

    top_symbols = [row["symbol"] for row in hotlist
                   if lead_market_ids is None or row.get("status") == "active"]
    if candle_reader is not None and is_demo and closed_market_cache is not None:
        span = settings.symbol_regime.ema_slow_bars
        key = (now_ms // DAY_MS, span)
        daily_cache_key = (key[0], span, settings.lookback.atr_len,
                           MIN_DAILY_UNIVERSE_BARS)
        requests = [(symbol, "1d", max(span + 1, 100)) for symbol in top_symbols
                    if not closed_market_cache.daily_direction.get(symbol)
                    or closed_market_cache.daily_direction[symbol][0] != daily_cache_key]
        prefetched = candle_reader.read_many(requests, _public_read_deadline(now, settings),
                                              monitor=position_monitor)
        for symbol, _, _ in requests:
            _cached_daily_direction_gate(exchange, symbol, now, span, closed_market_cache,
                                          prefetched=prefetched)
    decision_ms = (int(_utc_now().timestamp() * 1000) if is_demo else now_ms)
    max_quote_age_ms = (DEMO_POSITION_QUOTE_MAX_AGE_MS if is_demo
                        else LIVE_TICKER_MAX_AGE_MS)
    current_prices = {
        symbol: _ticker_price(tickers, symbol, decision_ms, max_age_ms=max_quote_age_ms)
        for symbol in track
    }
    if not is_demo and local_pending_symbols:
        _process_local_dynamic_limits(
            state, current_prices, settings, now, fee_rate, slip_rate,
        )
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
    if lead_market_ids is not None:
        entry_wait_reason = "均值回归结构候选；等待本小时 1H 方向与 EMA9 确认及风控审批"

    for row in hotlist:
        symbol = row["symbol"]
        regime_value = "UNKNOWN"
        reg_frame = regimes.get(symbol)
        if reg_frame is not None and len(reg_frame):
            regime = reg_frame.iloc[-1]["regime"]
            if isinstance(regime, Regime):
                regime_value = regime.value
        _set_symbol_status(state, symbol, reason=(row.get("reason") or entry_wait_reason), at=now,
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
    if lead_market_ids is not None:
        latest_bar = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    scan = state.mr_entry_scan
    entry_due = (latest_bar is not None and
                 ((is_demo and (scan.get("bar") != latest_bar.isoformat()
                                or scan.get("status") not in {"completed", "expired"}))
                  or (not is_demo and new_exit_bar)))
    if lead_market_ids is not None and scan.get("bar") == latest_bar.isoformat():
        entry_due = entry_due or (not mr_scan.expired(scan, _utc_now())
                                  and bool(_lead_scan_recoveries(scan, top_symbols)))
    if entry_due:
        if state.execution_mode == "lead_usdm" and ranked_hotlist:
            try:
                _record_lead_whitelist_observation(
                    state, execution, ranked_hotlist, hotlist, markets, latest_bar, now,
                )
            except Exception as exc:  # Observability must not affect entry evaluation.
                logger.warning("Lead whitelist shadow unavailable: {}", type(exc).__name__)
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
        is_vsb = meta.get("strategy_tag", "MR") == "VSB"
        if not is_vsb:
            frame = frame.tail(frame.attrs.get("mr_window_bars", len(frame)))
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
            _update_demo_excursions(meta, position, frame=frame, as_of=now)
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
        best_price = float(meta.get("best_price", entry))
        worst_price = float(meta.get("worst_price", entry))
        if _position_bar_is_eligible(position, frame, now):
            best_price = max(best_price, current_high) if side is Side.LONG else min(best_price, current_low)
            worst_price = min(worst_price, current_low) if side is Side.LONG else max(worst_price, current_high)
        meta["best_price"] = best_price
        meta["worst_price"] = worst_price

        # 3. Bollinger mid exit
        mid = float(mid_band(frame["close"], MID_PERIOD, True).iloc[-1])
        ms = MidExitState(reached_mid=bool(meta.get("mid_reached")),
                          hold_count=int(meta.get("mid_hold", 0)),
                          fired=bool(meta.get("mid_fired")))
        fire, ms = ((False, ms) if is_vsb else mid_exit_check(
            side, entry, mid, current_close, buffer=0.0, hold_bars=0, state=ms))
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
        if is_vsb:
            fade_event.update(triggered=False, decision="skipped", skip_reason="vsb_strategy_exempt",
                              reason="vsb_regime_fade_exempt", strategy_tag="VSB")
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

        # Early VSB falsification uses distinct post-fill hourly closes, including
        # the first partial holding hour. Frozen metadata survives limit recovery.
        closed_at = frame.index[-1].to_pydatetime() + timedelta(hours=1)
        if (is_vsb and position.opened_at < closed_at <= now
                and meta.get("vsb_last_check_bar") != closed_at.isoformat()):
            checks = int(meta.get("vsb_hourly_checks", 0)) + 1
            meta.update(vsb_hourly_checks=checks, vsb_last_check_bar=closed_at.isoformat())
            box_mid = meta.get("box_mid")
            holding_hour = (closed_at - position.opened_at.replace(
                minute=0, second=0, microsecond=0)).total_seconds() / 3600
            if (checks <= 4 and 1 <= holding_hour <= 4 and box_mid is not None
                    and (current_close < float(box_mid) if side is Side.LONG
                         else current_close > float(box_mid))):
                _close_position(is_demo, execution, state, symbol, current_close,
                                "vsb_false_breakout", now, fee_rate, slip_rate)
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
        _mark_trailing_started(
            state, symbol, position, meta, progress_r, new_stop, new_stop,
            settings.mr_candidate_exit.trailing_trigger_r, now,
        )
        if abs(new_stop - current_stop) > 1e-12:
            meta["protective_stop"] = new_stop

        state.exit_management_by_symbol[symbol] = meta
        if _apply_max_hold_exit(False, execution, state, symbol, position, current_close,
                                now, settings, fee_rate, slip_rate):
            continue


def _entry_priority_sort_key(candidate):
    priority = candidate[1]
    if "tier_bucket" in priority:
        return tuple(-float(priority[key]) for key in (
            "tier_bucket", "is_market_order", "bar_quality", "space_r")) + (candidate[0],)
    return (
        -int(priority["deviation_zone"]),
        -float(priority["reversal_score"]),
        -min(float(priority["ema_space_r"]), 3.0),
        abs(float(priority["deviation_atr"]) - 1.6),
        candidate[0],
    )


def _entry_priority_metrics(side, candle, close, stop, closed_ema, deviation_pct, atr_pct):
    deviation_atr = abs(deviation_pct) / atr_pct
    if 1.0 <= deviation_atr <= 2.2:
        zone = 2
    elif 2.2 < deviation_atr <= 3.0:
        zone = 1
    else:
        zone = 0
    candle_range = float(candle["high"] - candle["low"])
    if candle_range > 0:
        close_location = ((close - float(candle["low"])) / candle_range
                          if side is Side.LONG else
                          (float(candle["high"]) - close) / candle_range)
        body_ratio = abs(close - float(candle["open"])) / candle_range
        close_location = min(1.0, max(0.0, close_location))
        body_ratio = min(1.0, max(0.0, body_ratio))
    else:
        close_location = body_ratio = 0.0
    reversal_score = (close_location + body_ratio) / 2
    ema_space = (closed_ema - close if side is Side.LONG else close - closed_ema)
    stop_distance = abs(close - stop)
    ema_space_r = max(0.0, ema_space) / stop_distance if stop_distance > 0 else 0.0
    return {
        "deviation_atr": deviation_atr,
        "deviation_zone": zone,
        "reversal_score": reversal_score,
        "candle_close_location": close_location,
        "candle_body_ratio": body_ratio,
        "ema_space_r": ema_space_r,
    }


def _entry_candidate(exchange, state, is_demo, symbol, one_hour, four_hour, regimes,
                     settings, now, structure_reentry_states, *, daily_direction_gate=False,
                     position_monitor=None, closed_market_cache=None, scan_row=None,
                     deviation_pct=None):
    span = settings.symbol_regime.ema_slow_bars
    if position_monitor is not None:
        position_monitor()
    if symbol in state.portfolio.positions:
        if scan_row is not None:
            scan_row["reason"] = "position_already_open"
        return None
    frame1 = one_hour.get(symbol)
    frame4 = four_hour.get(symbol)
    route = state.mr_candidate_pool.get(symbol, {})
    strategy_tag = route.get("strategy_tag", "MR")
    is_vsb = strategy_tag == "VSB"
    if frame1 is not None and not is_vsb:
        frame1 = frame1.tail(frame1.attrs.get("mr_window_bars", len(frame1)))
    if frame1 is None or frame4 is None or len(frame1) < settings.lookback.atr_len + 1 or len(frame4) < span:
        if scan_row is not None:
            scan_row["reason"] = "entry_history_insufficient"
        return None
    evaluated_at = _utc_now() if is_demo else now
    max_delay = (timedelta(seconds=settings.execution.confirm_delay_sec_max)
                 if is_demo else ENTRY_BAR_MAX_DELAY)
    bar_close_time, bar_rejection = _entry_bar_close_time(
        frame1, evaluated_at, max_delay=max_delay,
    )
    if bar_rejection is not None:
        _record_entry_bar_rejection(state, symbol, frame1, bar_rejection, evaluated_at)
        if scan_row is not None:
            scan_row["reason"] = bar_rejection
        return None
    manual_closed = _aware_datetime(
        state.exit_management_by_symbol.get("__manual_entry_blocks__", {}).get(symbol))
    if manual_closed is not None and bar_close_time <= manual_closed:
        _record_entry_bar_rejection(state, symbol, frame1, "manual_close_consumed_signal", evaluated_at)
        if scan_row is not None:
            scan_row["reason"] = "manual_close_consumed_signal"
        return None
    try:
        validate_ohlcv(frame1)
    except (TypeError, ValueError):
        _record_entry_bar_rejection(state, symbol, frame1, "invalid_ohlcv", now)
        if scan_row is not None:
            scan_row["reason"] = "invalid_ohlcv"
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
    squeeze = {}
    if is_vsb:
        try:
            squeeze = squeeze_context(frame1)
            candidate_signal = breakout_signal(frame1, Side(route["side"]), squeeze,
                                                float(atr_series.iloc[-1]))
        except (KeyError, TypeError, ValueError, IndexError):
            candidate_signal = Signal(None, Action.NONE, "vsb_data_unavailable", 0.0)
    else:
        candidate_signal = deviation_entry_signal(
            frame4["close"], candle, settings, candidate_flags, deviation_pct=DEV_PCT,
        )
    if candidate_signal.action is not Action.OPEN:
        if scan_row is not None:
            scan_row["reason"] = candidate_signal.reason
        return None
    if scan_row is not None:
        scan_row["side"] = candidate_signal.side.value
    if not _record_time_filter_decision(
            state, symbol, candidate_signal.side, frame1, bar_close_time, flags,
            settings, evaluated_at, scan_row=scan_row):
        return None
    sig = (candidate_signal if is_vsb else deviation_entry_signal(
        frame4["close"], candle, settings, flags, deviation_pct=DEV_PCT))
    if sig.action is not Action.OPEN:
        if scan_row is not None:
            scan_row["reason"] = sig.reason
        return None
    entry_atr = float(atr_series.iloc[-1])
    if is_vsb:
        try:
            entry_atr = effective_entry_atr(frame1, entry_atr)
        except ValueError:
            if scan_row is not None:
                scan_row["reason"] = "vsb_atr60_unavailable"
            return None
    close = float(candle["close"])
    swing_low = float(frame1["low"].tail(settings.stops.swing_lookback_1h).min())
    swing_high = float(frame1["high"].tail(settings.stops.swing_lookback_1h).max())
    structure = swing_low if sig.side is Side.LONG else swing_high
    try:
        stop, _mult = calculate_adaptive_stop(sig.side, close, entry_atr, structure, settings)
    except ValueError as exc:
        raw_distance = ((close - structure) if sig.side is Side.LONG else (structure - close))
        raw_distance += settings.stops.structure_buffer_atr * entry_atr
        stop_reason = str(exc) if str(exc) in {"stop_too_tight", "stop_too_wide"} else "entry_stop_invalid"
        stop_details = {"price": close, "atr": entry_atr, "structure_price": structure,
                        "raw_stop_distance": raw_distance,
                        "raw_stop_atr": raw_distance / entry_atr if entry_atr > 0 else None}
        stop_details = {key: value if value is None or math.isfinite(value) else None
                        for key, value in stop_details.items()}
        _record_entry_rejection(
            state, symbol, sig.side, stop_reason, "signal_stop", now,
            bar_close_at=bar_close_time, scan_row=scan_row,
            details=stop_details,
        )
        return None
    closed_ema = float(frame4["close"].ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1])
    dev = close / closed_ema - 1.0
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
            if scan_row is not None:
                scan_row["reason"] = daily_reason
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
            if scan_row is not None:
                scan_row["reason"] = "structure_reentry_confirmation_required"
            return None
    ema9_side = sig.side.value if is_vsb else _ema9_reclaim_side(frame1)
    if ema9_side != sig.side.value:
        _record_entry_rejection(
            state, symbol, sig.side, "ema9_reclaim_unconfirmed", "entry_confirmation", now,
            bar_close_at=bar_close_time,
            details={"ema9_reclaim_side": ema9_side, "required_side": sig.side.value},
            scan_row=scan_row,
        )
        return None
    atr4, atr4_pct, atr_threshold = _atr_deviation_threshold(frame4, settings)
    # The explicit override is used only by frozen offline research hypotheses.
    threshold = atr_threshold if deviation_pct is None else deviation_pct
    if state.execution_mode == "lead_usdm" and scan_row is not None and not is_vsb:
        try:
            _record_atr_deviation_shadow(
                state, symbol, frame1, frame4, bar_close_time, evaluated_at,
                closed_ema, close, flags, settings, sig, stop, entry_atr,
                regime_value, daily_gate, scan_row, threshold,
            )
        except Exception as exc:  # Observability must not affect the live candidate.
            logger.warning("Lead ATR deviation shadow unavailable: {}", type(exc).__name__)
    deviation_details = {"deviation_threshold_pct": threshold,
                         "closed_4h_atr": atr4, "closed_4h_atr_pct": atr4_pct,
                         "deviation_pct": dev}
    if threshold is None or not math.isfinite(threshold) or threshold <= 0:
        _record_entry_rejection(
            state, symbol, sig.side, "entry_deviation_unavailable", "entry_deviation", now,
            bar_close_at=bar_close_time, scan_row=scan_row, details=deviation_details,
        )
        return None
    adverse = (closed_ema - close) / closed_ema if sig.side is Side.LONG else (close - closed_ema) / closed_ema
    active_signal = (sig if -1.2 * threshold <= adverse < threshold else
                     Signal(None, Action.NONE, "vsb_deviation_not_met", 0.0)) if is_vsb else deviation_entry_signal_at_ema(
        closed_ema, float(candle["open"]), close, settings, flags, deviation_pct=threshold)
    if active_signal.action is not Action.OPEN or active_signal.side is not sig.side:
        _record_entry_rejection(
            state, symbol, sig.side, "atr_deviation_not_met", "entry_deviation", now,
            bar_close_at=bar_close_time, scan_row=scan_row, details=deviation_details,
        )
        return None
    context = EntrySignalContext(
        symbol, sig.side, bar_close_time, now, evaluated_at,
        float(candle["open"]), close, closed_ema, entry_atr, swing_low, swing_high,
        threshold,
        strategy_tag,
        ((squeeze["box_high"] + 0.15 * float(atr_series.iloc[-1])) if sig.side is Side.LONG
         else (squeeze["box_low"] - 0.15 * float(atr_series.iloc[-1]))) if is_vsb else None,
    ) if is_demo else None
    execution_plan = _entry_execution_plan(candle, entry_atr, bar_close_time, sig.side)
    priority = _entry_priority_metrics(
        sig.side, candle, close, stop, closed_ema, dev, atr4_pct,
    )
    if state.execution_mode == "lead_usdm" or is_vsb:
        z_dev = adverse / threshold
        tier = (2 if 1.2 <= z_dev <= 2.2 else 1 if 1 <= z_dev < 1.2 else 0)
        if is_vsb:
            tier = 2 if 10 <= squeeze["sqz_bars"] <= 24 and abs(adverse) <= 0.5 * threshold else 1
        baseline_volume = (squeeze["squeeze_volume_mean"] if is_vsb else
                           float(frame1["volume"].iloc[:-1].tail(20).mean()))
        volume = float(candle["volume"])
        volume_ratio = volume / baseline_volume if baseline_volume > 0 and math.isfinite(volume) else 0.0
        priority.update(tier_bucket=tier, is_market_order=int(execution_plan["order_type"] == "market"),
                        bar_quality=(priority["candle_close_location"] + min(1.0, volume_ratio / 3.0)) / 2,
                        space_r=2.5 * atr4 / abs(close - stop) if is_vsb else priority["ema_space_r"])
    execution_plan["candidate_priority"] = priority
    execution_plan.update(ema9_reclaim_side=ema9_side, deviation_threshold_pct=threshold,
                          closed_4h_atr_pct=atr4_pct, **strategy_metadata({"strategy_tag": strategy_tag, **squeeze}))
    return (symbol, priority, sig.side, close, stop, entry_atr,
                       swing_low, swing_high, regime_value, sig.reason, daily_gate,
                       context, flags, execution_plan)


def _initialize_entry_metadata(state, symbol, opened, is_demo, close, stop,
                               entry_atr, swing_low, swing_high, *, observed_price=None,
                               observed_at=None, telemetry_history_complete=True):
    if observed_price is not None and not _position_sample_is_eligible(opened, observed_at):
        observed_price = observed_at = None
        telemetry_history_complete = False
    initial_risk = (abs(opened.entry_price - opened.stop_price)
                    if is_demo else abs(close - stop))
    state.exit_management_by_symbol[symbol] = {
        "strategy_tag": "MR",
        "entry_atr": entry_atr,
        "enabled": True,
        "protective_stop": float(opened.stop_price),
        "structure_low": swing_low,
        "structure_high": swing_high,
        "initial_risk": initial_risk,
        "trailing_started": False,
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
    if len(candidate) == 14:
        if isinstance(candidate[-1], dict):
            candidate = (*candidate, None)
        else:
            # Older callers supplied the funding value but had no execution plan.
            candidate = (*candidate[:13], {"order_type": "market"}, candidate[-1])
    (symbol, _strength, side, close, stop, entry_atr, swing_low, swing_high,
     regime_value, reason, daily_gate, context, flags, execution_plan, funding_rate) = candidate
    max_delay = timedelta(seconds=settings.execution.confirm_delay_sec_max)
    market_risk_state = market_risk.state.value
    def reject(reason, phase, details=None):
        _record_entry_rejection(state, symbol, side, reason, phase, _utc_now(),
                                bar_close_at=context.bar_close_at if context else None,
                                details=details, scan_row=scan_row)
        return None
    order_type = execution_plan.get("order_type", "market")
    limit_entry_price = execution_plan.get("limit_price") if order_type == "limit" else None
    limit_expires_at = execution_plan.get("limit_expires_at") if order_type == "limit" else None
    if order_type == "limit":
        try:
            close = float(limit_entry_price)
            structure = swing_low if side is Side.LONG else swing_high
            stop, _ = calculate_adaptive_stop(side, close, entry_atr, structure, settings)
        except (TypeError, ValueError, OverflowError):
            return reject("dynamic_limit_stop_invalid", "signal_stop")
        if (not math.isfinite(close) or close <= 0
                or (side is Side.LONG and stop >= close)
                or (side is Side.SHORT and stop <= close)):
            return reject("dynamic_limit_stop_invalid", "signal_stop")
        stop = float(stop)
    if is_demo:
        try:
            _require_entry_clock()
        except ExchangeClockError:
            return reject("exchange_clock_stale_or_unavailable", "account_gate")
    market_risk_multiplier = (settings.market_risk.elevated_size_multiplier
                              if market_risk.state is MarketRiskState.RISK_ELEVATED else 1.0)
    if (state.execution_mode == "lead_usdm"
            and str((markets.get(symbol) or {}).get("id", "")).upper()
            not in getattr(execution, "lead_market_ids", set())):
        return reject("lead_symbol_not_allowed", "account_gate")
    if is_demo and len(state.portfolio.positions) >= max_pos:
        return reject("max_positions_reached", "account_gate", {"max_positions": max_pos})
    if position_monitor is not None:
        position_monitor()
    if is_demo:
        if state.execution_status != "synchronized" or mr_scan.unresolved_attempts(state):
            return reject("account_not_synchronized", "account_gate")
        if _utc_now() > context.bar_close_at + max_delay:
            return reject("hourly_bar_stale", "signal_expiry")
        funding_rate = _fetch_entry_funding_rate(exchange, symbol, fail_on_market_error=True)
        funding_reason = funding_rate_rejection_reason(funding_rate, side, settings)
        if funding_reason is not None:
            _record_funding_entry_block(state, symbol, side, regime_value,
                                        funding_rate, funding_reason, settings, _utc_now())
            if scan_row is not None:
                scan_row["reason"] = funding_reason
            return None
    step_size = 1e-8
    minimum_quantity = 0.0
    if is_demo:
        step_size, minimum_quantity = _entry_order_limits(markets.get(symbol) or {}, close)
    if state.execution_mode == "lead_usdm" and (
            not _demo_snapshot_is_fresh(state, _utc_now())
            or set(state.exchange_snapshot.get("positions", {})) != set(state.portfolio.positions)
            or state.protective_stop_audit.get("status") != "verified"):
        # A preceding fill invalidates the snapshot/audit; refresh before sizing the next candidate.
        if (not _demo_sync_stopped(execution, state, _utc_now())
                or not _audit_demo_protective_stops(execution, state, _utc_now())):
            return reject("account_not_synchronized", "account_gate")
    target_margin = _target_margin_per_symbol(state, settings)
    try:
        quantity_limits, approval_limits = _entry_capital_limits(state, settings, side, _utc_now())
    except (KeyError, TypeError, ValueError, OverflowError):
        return reject("lead_entry_risk_inputs_unavailable", "quantity")
    quantity_audit = {}
    qty = calculate_order_quantity(state.current_equity, close, state.portfolio, settings,
                                   step_size=step_size, minimum_quantity=minimum_quantity,
                                   size_multiplier=market_risk_multiplier,
                                   stop_price=stop,
                                   gross_notional_cap_pct=(settings.sizing.max_gross_notional_pct
                                                           * market_risk_multiplier),
                                   target_margin_usdt=target_margin, audit=quantity_audit, **quantity_limits)
    if qty <= 0:
        return reject(quantity_audit.get("reason") or "entry_quantity_unavailable",
                      "quantity", quantity_audit)
    intent = OrderIntent(symbol, side, Action.OPEN, qty, close, reason, Regime(regime_value),
                         stop, MEAN_REVERSION)
    approval = assess_entry(intent, state.portfolio, state.current_equity,
                             funding_rate, now, settings,
                             gross_notional_cap_pct=(settings.sizing.max_gross_notional_pct
                                                     * market_risk_multiplier),
                             market_risk_state=market_risk_state,
                             per_symbol_margin_usdt=target_margin, **approval_limits)
    if not approval.allowed:
        return reject(approval.reason, "risk_approval",
                      {"quantity_risk": quantity_audit, "approval": approval.details})
    if order_type == "limit" and not is_demo:
        try:
            ticker = exchange.fetch_ticker(symbol)
        except ccxt.BaseError as exc:
            _raise_market_error_if_needed(exc, False)
            return reject("dynamic_limit_quote_unavailable", "entry_confirmation")
        quote_price = _ticker_price(
            {symbol: ticker}, int(_utc_now().timestamp() * 1000),
        )
        if quote_price is None:
            return reject("dynamic_limit_quote_stale", "entry_confirmation")
        already_touched = (quote_price <= close if side is Side.LONG
                           else quote_price >= close)
        if already_touched:
            return reject("dynamic_limit_price_already_touched", "entry_confirmation",
                          {"quote_price": quote_price, "limit_price": close})
        pending_key = "paper_limit|{}|{}|{}".format(
            symbol, side.value, execution_plan.get("signal_bar") or "unknown",
        )
        state.mr_entry_attempts[pending_key] = {
            "symbol": symbol, "side": side.value, "status": "paper_limit_pending",
            "order_type": "limit", "signal_bar": execution_plan.get("signal_bar"),
            "reason": reason, "regime": regime_value,
            "limit_price": float(close), "stop_price": float(stop),
            "quantity": float(qty), "expires_at": limit_expires_at,
            "ema9_reclaim_side": execution_plan.get("ema9_reclaim_side"),
            "deviation_threshold_pct": execution_plan.get("deviation_threshold_pct"),
            "meta": {**strategy_metadata(execution_plan), "entry_atr": float(entry_atr),
                     "structure_low": float(swing_low),
                     "structure_high": float(swing_high)},
            "created_at": _utc_now().isoformat(),
        }
        state.events.append({"event": "dynamic_limit_entry_pending", "at": _utc_now().isoformat(),
                             "symbol": symbol, "side": side.value,
                             "price": float(close), "stop_price": float(stop),
                             "expires_at": limit_expires_at, "clientOrderId": pending_key,
                             "source": MEAN_REVERSION, "execution_mode": state.execution_mode})
        return {"pending_limit": True, "clientOrderId": pending_key,
                "price": float(close), "expires_at": limit_expires_at}
    execution_kwargs = {}
    if is_demo:
        if scan_row is not None:
            scan_row.update(phase="demo_prepare", endpoint="demo")
        audit = {}
        audit.update(strategy_metadata(execution_plan))
        audit.update({"order_type": order_type,
                      "limit_price": limit_entry_price,
                      "limit_expires_at": limit_expires_at,
                      "ema9_reclaim_side": execution_plan.get("ema9_reclaim_side"),
                      "signal_range_atr": execution_plan.get("candle_range_atr")})
        guard, deadline = _entry_preflight(
            exchange, execution, state, settings, markets.get(symbol) or {},
            context, flags, funding_rate, market_risk, audit,
            propagate_read_errors=scan_row is not None,
            scan_row=scan_row, limit_entry_price=limit_entry_price,
        )
        execution_kwargs = {"entry_preflight": guard, "submission_deadline": deadline,
                            "entry_audit": audit, "order_type": order_type,
                            "limit_expires_at": limit_expires_at}
        if scan_row is not None:
            execution_kwargs.update(scan_row=scan_row, checkpoint=checkpoint,
                                    recovery_meta={**strategy_metadata(execution_plan),
                                                   "entry_atr": entry_atr, "structure_low": swing_low,
                                                   "structure_high": swing_high, "last_close": close})
    opened = _open_position(is_demo, execution, state, symbol, side, close, stop, qty,
                            int(settings.sizing.leverage), fee_rate, slip_rate, reason,
                            regime_value, now, **execution_kwargs)
    if isinstance(opened, dict) and opened.get("pending_limit"):
        state.events.append({"event": "dynamic_limit_entry_pending", "at": _utc_now().isoformat(),
                             "symbol": symbol, "side": side.value, "price": close,
                             "stop_price": stop, "expires_at": limit_expires_at,
                             "clientOrderId": opened.get("clientOrderId"),
                             "source": MEAN_REVERSION})
        return opened
    if opened is not None:
        _initialize_entry_metadata(state, symbol, opened, is_demo, close, stop,
                                   entry_atr, swing_low, swing_high)
        state.exit_management_by_symbol[symbol].update(strategy_metadata(execution_plan))
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
    evaluation_started = time.monotonic()
    signal_seconds = execution_seconds = 0.0
    pending_phase = None
    bars = [frame.index[-1] for frame in one_hour.values() if frame is not None and len(frame)]
    lead = state.execution_mode == "lead_usdm"
    if not bars and not lead:
        return
    target_bar = (pd.Timestamp(now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
                  if lead else max(bars))
    scan = mr_scan.scan_for(state, target_bar, top_symbols,
                            settings.execution.confirm_delay_sec_max, checkpoint)
    if lead and not mr_scan.expired(scan, _utc_now()):
        recovered = _lead_scan_recoveries(scan, top_symbols)
        if recovered:
            for symbol in recovered:
                if symbol not in scan["symbols"]:
                    scan["symbols"].append(symbol)
                else:
                    scan["outcomes"][symbol].update(status="retry")
                    scan["outcomes"][symbol].pop("reason", None)
            scan["status"] = "pending"
            mr_scan.checkpoint(checkpoint)
    if scan["status"] in {"completed", "expired"}:
        return
    _reconcile_structure_invalidation_cooldowns(state)
    reentry = _structure_reentry_states(state.events, scan["symbols"])

    def finish(symbol, status, **details):
        # Pure signal discovery is checkpointed as a batch before execution.
        # Every submission/uncertain result still uses its existing durable gate.
        save = (None if lead and status in {"evaluating", "candidate", "evaluated", "blocked", "expired"}
                else checkpoint)
        return mr_scan.outcome(state, scan, symbol, status, _utc_now(), save, **details)

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
            signal_started = time.monotonic()
            pending_phase = "signal"
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
                if lead and symbol not in top_symbols:
                    finish(symbol, "blocked", reason="candidate_structure_unverified")
                    continue
                if len(state.portfolio.positions) >= max_pos:
                    finish(symbol, "blocked", reason="position_limit")
                    continue
                attempts = (row or {}).get("attempts", 0) + 1
                if row is not None:
                    row.pop("candidate_priority", None)
                row = finish(symbol, "evaluating", attempts=attempts,
                             phase="candidate", endpoint="public", side=None, reason=None,
                             order_may_have_been_sent=False)
                try:
                    candidate = _entry_candidate(
                        exchange, state, True, symbol, one_hour, four_hour, regimes, settings,
                        now, reentry, position_monitor=position_monitor,
                        closed_market_cache=closed_market_cache, scan_row=row,
                    )
                    if candidate is None:
                        finish(symbol, "evaluated", reason=row.get("reason") or "no_eligible_signal")
                    else:
                        finish(symbol, "candidate", side=candidate[2].value,
                               candidate_priority=candidate[1])
                        candidates.append(candidate)
                except mr_scan.ScanCheckpointError:
                    raise
                except Exception as exc:
                    if not failure(symbol, exc):
                        scan["status"] = "paused"
                        return
            candidates.sort(key=_entry_priority_sort_key)
            if lead:
                mr_scan.checkpoint(checkpoint)
            signal_seconds += time.monotonic() - signal_started
            execution_started = time.monotonic()
            pending_phase = "execution"
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
                    if opened is not None and not (isinstance(opened, dict)
                                                   and opened.get("pending_limit")):
                        client_id = mr_scan.client_order_id(symbol, candidate[2].value, scan["bar"])
                        state.mr_entry_attempts[client_id]["status"] = "completed"
                        mr_scan.attempt_outcome(
                            state, state.mr_entry_attempts[client_id], "completed", _utc_now(),
                            phase="fill_confirmed", order_may_have_been_sent=True,
                        )
                    finish(symbol, "pending_limit" if isinstance(opened, dict)
                           and opened.get("pending_limit")
                           else "opened" if opened is not None else "rejected")
                except mr_scan.ScanCheckpointError:
                    raise
                except Exception as exc:
                    if not failure(symbol, exc):
                        scan["status"] = "paused"
                        return
            execution_seconds += time.monotonic() - execution_started
            pending_phase = None
        scan["status"] = ("expired" if any(row["status"] == "expired"
                                          for row in scan["outcomes"].values()) else "completed")
    finally:
        if pending_phase == "signal":
            signal_seconds += time.monotonic() - signal_started
        elif pending_phase == "execution":
            execution_seconds += time.monotonic() - execution_started
        state.events.append({"event": "entry_scan_finished", "at": _utc_now().isoformat(),
                             "bar": scan["bar"], "deadline": scan["deadline"],
                             "status": scan["status"], "symbols": len(scan["symbols"]),
                             "signal_check_seconds": round(signal_seconds, 4),
                             "execution_prepare_seconds": round(execution_seconds, 4),
                             "total_seconds": round(time.monotonic() - evaluation_started, 4),
                             "within_entry_deadline": not mr_scan.expired(scan, _utc_now()),
                             "valid_signals": sum("candidate_priority" in row for row in scan["outcomes"].values()),
                             "reasons": {s: r.get("reason") for s, r in scan["outcomes"].items()
                                         if r.get("reason")},
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
    if any(row.get("status") == "paper_limit_pending"
           for row in state.mr_entry_attempts.values()):
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

    candidates.sort(key=_entry_priority_sort_key)
    logger.info("MR entry candidate priority {}", json.dumps([
        {"symbol": candidate[0], **candidate[1]} for candidate in candidates
    ], ensure_ascii=False, sort_keys=True))
    eligible_candidates = []
    for candidate in candidates:
        if position_monitor is not None:
            position_monitor()
        if len(eligible_candidates) >= slots:
            break
        (symbol, _strength, side, _close, _stop, _entry_atr, _low, _high,
         regime_value, _reason, _daily_gate, _context, _flags, _execution_plan) = candidate
        funding_rate = _fetch_entry_funding_rate(
            exchange, symbol, fail_on_market_error=daily_direction_gate,
        )
        funding_reason = funding_rate_rejection_reason(funding_rate, side, settings)
        if funding_reason is not None:
            _record_funding_entry_block(state, symbol, side, regime_value,
                                        funding_rate, funding_reason, settings, now)
            continue
        eligible_candidates.append((*candidate, funding_rate))

    for candidate in eligible_candidates:
        opened = _submit_entry_candidate(
            exchange, state, execution, is_demo, markets, candidate,
            settings, max_pos, fee_rate, slip_rate, now, market_risk,
            position_monitor=position_monitor,
        )
        if isinstance(opened, dict) and opened.get("pending_limit"):
            break


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
    settings = settings_for_execution(settings, mode.name)
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

    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 5000})
    configure_system_proxy(exchange)
    account_exchange = None
    execution = None
    clock = None
    candle_reader = None
    try:
        state = MultiPaperState.load(state_path, initial_equity=args.equity)
        if state_path_existed and state.execution_mode != mode.name:
            raise RuntimeError(
                f"state file execution mode is {state.execution_mode}; refusing to reuse it for {mode.name}"
            )
        if not state_path_existed:
            state.execution_mode = mode.name
        checkpoint = _RunnerCheckpoint(state, state_path)
        if mode.name == "lead_usdm":
            state.events.append({"event": "lead_risk_policy_loaded", "at": _utc_now().isoformat(),
                                 **settings.lead_risk.model_dump(),
                                 "max_single_trade_risk_pct": settings.risk.max_single_trade_risk_pct,
                                 "max_gross_notional_pct": settings.sizing.max_gross_notional_pct,
                                 "max_directional_notional_pct": settings.sizing.max_directional_notional_pct,
                                 "max_symbols": settings.sizing.max_symbols})
        if is_exchange_execution:
            cooldown_event_recorded = False
            while rate_limit_cooldown_remaining() > 0:
                remaining = rate_limit_cooldown_remaining()
                now = _utc_now()
                if not cooldown_event_recorded:
                    state.events.append({
                        "event": "exchange_rate_limit_cooldown",
                        "at": now.isoformat(),
                        "until_ms": int((now.timestamp() + remaining) * 1000),
                    })
                    cooldown_event_recorded = True
                state.record_equity(now)
                checkpoint.try_save()
                logger.warning("Binance IP rate-limit cooldown active; startup paused for {}s",
                               int(max(1, remaining)))
                time.sleep(min(max(1.0, float(args.poll)), max(1.0, remaining)))
            if mode.name == "lead_usdm":
                account_exchange = create_lead_usdm(settings.runtime)
                execution = LeadPortfolioExecutionService(account_exchange, settings)
                execution.adopt_manual_positions = bool(args.adopt_manual_positions)
            else:
                account_exchange = create_demo_usdm(settings.runtime)
                execution = DemoExecutionService(account_exchange, settings)
            execution.configure_close_recovery(state.exchange_orders_by_symbol, checkpoint)
            try:
                clock = _initialize_exchange_clock(account_exchange, state)
                state.exchange_snapshot = _initialize_demo(execution)
                _observe_account_equity(state, state.exchange_snapshot, settings, _utc_now())
            except (ccxt.BaseError, ExchangeClockError) as exc:
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
            if entries_reconciled and not _demo_sync_stopped(execution, state, _utc_now()):
                if not _lead_manual_position_isolation_active(state):
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
                                 "at": _utc_now().isoformat(),
                                 "previous": state.strategy_variant,
                                 "current": candidate_variant})
            state.strategy_variant = candidate_variant
        state.events.append({"event": "entry_deviation_policy_loaded", "at": _utc_now().isoformat(),
                             "rule": DEV_RULE, "minimum_deviation_pct": DEV_PCT,
                             "atr_multiplier": DEV_ATR_MULTIPLIER, "timeframe": "4h",
                             "price_source": "last_closed_4h_close", "runner_build_id": RUNNER_BUILD_ID})
        checkpoint.try_save()

        logger.info("Starting MR candidate: mode={} strategy={} max_positions={} risk_per_trade_pct={} poll={} intrabar_entries={} adopt_manual_positions={} runner_build_id={} time_filter_build_id={} state={}",
                    mode.name, candidate_variant, args.max_positions,
                    settings.risk.max_single_trade_risk_pct * 100, args.poll,
                    args.intrabar_entries, bool(args.adopt_manual_positions),
                    RUNNER_BUILD_ID, TIME_FILTER_BUILD_ID, state_path)
        market_cache = IntrabarMarketCache() if args.intrabar_entries else None
        closed_market_cache = ClosedMarketCache()
        candle_reader = CandleBatchReader(exchange, _utc_now)
        retry_after = _persisted_market_retry_after(state) if args.intrabar_entries else None
        market_failure_attempt = 0
        last_position_monitor_at = 0.0

        def monitor_demo_positions(*, force=False):
            nonlocal last_position_monitor_at
            if (not is_demo or not state.portfolio.positions
                    or rate_limit_cooldown_remaining() > 0):
                return False
            monotonic_now = time.monotonic()
            if (not force and monotonic_now - last_position_monitor_at
                    < DEMO_POSITION_MONITOR_POLL_SECONDS):
                return False
            last_position_monitor_at = monotonic_now
            monitor_at = _utc_now()
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
                state.record_equity(_utc_now())
                checkpoint.try_save()
            return dirty

        while True:
            cooldown_remaining = rate_limit_cooldown_remaining() if is_exchange_execution else 0.0
            if cooldown_remaining > 0:
                now = _utc_now()
                state.record_equity(now)
                checkpoint.try_save()
                time.sleep(min(max(1.0, float(args.poll)), max(1.0, cooldown_remaining)))
                continue
            if clock is not None:
                _refresh_exchange_clock(clock, account_exchange, state)
                _wait_for_closed_hour()
            now = _utc_now()
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
                       closed_market_cache=closed_market_cache, checkpoint=checkpoint,
                       candle_reader=candle_reader)
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
                    failed_at = _utc_now()
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
            if (is_demo and not checkpoint.failed and rate_limit_cooldown_remaining() <= 0
                    and _background_reads_allowed(_utc_now())):
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
                remaining = max(1.0, (retry_after - _utc_now()).total_seconds())
                time.sleep(min(float(args.poll), remaining))
    finally:
        if candle_reader is not None:
            candle_reader.close()
        if clock is not None and active_clock() is clock:
            activate_clock(None)
        exchange.close()
        if account_exchange is not None:
            account_exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
