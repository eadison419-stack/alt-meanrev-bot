"""Public-market polling and portfolio-wide simulation for a hot-symbol universe."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import threading
import time
from types import SimpleNamespace

import ccxt
import pandas as pd
from loguru import logger

from bot.data.klines import (TIMEFRAME_MS, filter_tradable_crypto_universe, hot_activity_metrics,
                             ohlcv_frame, score_hot_candidates)
from bot.exchange.network import configure_system_proxy
from bot.live.bollinger_outcomes import BollingerOutcomeTracker
from bot.live.manual_close import ManualCloseCommandStore
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import PaperTrader, effective_max_hold_hours
from bot.live.trend_entry_shadow import TrendEntryShadowTracker
from bot.market_risk import MarketRiskAssessment, MarketRiskState, fetch_market_risk_status
from bot.mean_reversion_match import evaluate_mean_reversion_match
from bot.symbol_regime import SymbolRegimeAssessment, classify_symbol_regime
from bot.strategy_router import MEAN_REVERSION, TREND
from bot.trend_match import evaluate_trend_match

HOUR_MS = TIMEFRAME_MS["1h"]
PULLBACK_QUOTE_MAX_AGE_SECONDS = 15


def _ordered_tracked_symbols(position_symbols, pending_symbols, ranked_symbols):
    """Keep risk competition deterministic: held, pending, then hotlist rank order."""
    return list(dict.fromkeys([*position_symbols, *pending_symbols, *ranked_symbols]))


def _symbol_regime_summary(observations: list[dict]) -> dict | None:
    if not observations:
        return None
    state_counts = Counter(row["state"] for row in observations)
    uncertain_symbols: dict[str, list[str]] = defaultdict(list)
    failed_condition_counts: Counter[str] = Counter()
    near_miss_condition_counts: Counter[str] = Counter()
    near_trend_count = 0
    near_range_count = 0
    near_range_structure_only_count = 0
    confirmation_not_met = 0
    for row in observations:
        if row["state"] != "UNCERTAIN":
            continue
        reason = row["reason"]
        uncertain_symbols[reason].append(row["symbol"])
        evidence = row.get("evidence", {})
        if reason == "confirmation_not_met":
            confirmation_not_met += 1
        for strategy, failures in evidence.get("condition_failures", {}).items():
            failed_condition_counts.update(f"{strategy}:{failure}" for failure in failures)
        if evidence.get("near_trend"):
            near_trend_count += 1
        if evidence.get("near_range"):
            near_range_count += 1
        if evidence.get("near_range_structure_only"):
            near_range_structure_only_count += 1
        for strategy, conditions in evidence.get("near_miss_conditions", {}).items():
            near_miss_condition_counts.update(f"{strategy}:{condition}" for condition in conditions)

    closed_bars = [row["last_closed_bar_at"] for row in observations if row.get("last_closed_bar_at")]
    return {
        "event": "symbol_regime_summary",
        "checked_at": max(row["checked_at"] for row in observations),
        "latest_closed_bar_at": max(closed_bars) if closed_bars else None,
        "classified_count": len(observations),
        "state_counts": dict(sorted(state_counts.items())),
        "uncertain_count": state_counts.get("UNCERTAIN", 0),
        "uncertain_reason_counts": dict(sorted(
            (reason, len(symbols)) for reason, symbols in uncertain_symbols.items()
        )),
        "uncertain_symbols_by_reason": {
            reason: sorted(symbols) for reason, symbols in sorted(uncertain_symbols.items())
        },
        "condition_failure_counts": dict(sorted(failed_condition_counts.items())),
        "confirmation_not_met_count": confirmation_not_met,
        "near_trend_count": near_trend_count,
        "near_range_count": near_range_count,
        "near_range_structure_only_count": near_range_structure_only_count,
        "near_miss_condition_counts": dict(sorted(near_miss_condition_counts.items())),
    }


def _current_market_price(ticker: dict | None) -> float | None:
    if not ticker:
        return None
    value = ticker.get("last") or ticker.get("close")
    if value is None:
        return None
    price = float(value)
    return price if math.isfinite(price) and price > 0 else None


def _valid_quote_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _finite_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 100 else None


def _low_position_watch_candidate(assessment: dict, activity: dict, policy) -> dict | None:
    """Return a log-only low-position setup using completed 4H metrics."""
    if not policy.enabled or assessment.get("status") != "scored":
        return None
    metrics = assessment.get("metrics") or {}
    try:
        values = {
            "range_position_pct": float(metrics["range_position_pct"]),
            "ema50_distance_pct": float(metrics["ema50_distance_pct"]),
            "ema50_slope_pct": float(metrics["ema50_slope_pct"]),
            "recent_return_pct": float(metrics["recent_return_pct"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            return None
        structure = str(metrics["structure"]).upper()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    if not policy.range_position_min_pct <= values["range_position_pct"] <= policy.range_position_max_pct:
        return None
    if not policy.ema50_distance_min_pct <= values["ema50_distance_pct"] <= policy.ema50_distance_max_pct:
        return None
    if not policy.recent_return_min_pct <= values["recent_return_pct"] <= policy.recent_return_max_pct:
        return None
    if values["ema50_slope_pct"] < policy.ema50_slope_min_pct:
        return None
    if structure not in {"NEUTRAL", "UP"}:
        return None

    reasons = ["low_to_mid_48bar_position", "price_near_or_below_ema50", "recent_move_not_extended"]
    reasons.append("structure_stabilizing" if structure == "NEUTRAL" else "structure_turning_up")
    reasons.append("ema50_slope_stabilizing")
    try:
        volume_burst = float(activity.get("volume_burst_ratio"))
    except (TypeError, ValueError, OverflowError):
        volume_burst = math.nan
    if math.isfinite(volume_burst) and volume_burst >= 1.2:
        reasons.append("closed_1h_volume_expansion")

    return {
        "candidate_type": "low_position_observation",
        "symbol": assessment.get("symbol"),
        "last_closed_bar_at": assessment.get("last_closed_bar_at"),
        **{key: round(value, 4) for key, value in values.items()},
        "structure": structure,
        "adx": metrics.get("adx"),
        "volume_burst_ratio": round(volume_burst, 4) if math.isfinite(volume_burst) else None,
        "trend_match_score": assessment.get("trend_match_score"),
        "mean_reversion_match_score": assessment.get("mean_reversion_match_score"),
        "direction_bias": assessment.get("direction_bias", "neutral"),
        "reasons": reasons,
    }


def _legacy_directional_observation(assessment: dict, policy) -> dict:
    """Evaluate the existing low-position rules and their symmetric short mirror."""
    if not policy.enabled:
        return {"status": "disabled", "directions": {}}
    if assessment.get("status") != "scored":
        return {"status": assessment.get("status", "unavailable"), "directions": {}}
    metrics = assessment.get("metrics") or {}
    try:
        values = {
            "range_position_pct": float(metrics["range_position_pct"]),
            "ema50_distance_pct": float(metrics["ema50_distance_pct"]),
            "ema50_slope_pct": float(metrics["ema50_slope_pct"]),
            "recent_return_pct": float(metrics["recent_return_pct"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non_finite_metrics")
        structure = str(metrics["structure"]).upper()
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": assessment.get("status", "unavailable"), "directions": {}}

    long_checks = {
        "range_position": policy.range_position_min_pct <= values["range_position_pct"] <= policy.range_position_max_pct,
        "ema50_distance": policy.ema50_distance_min_pct <= values["ema50_distance_pct"] <= policy.ema50_distance_max_pct,
        "recent_return": policy.recent_return_min_pct <= values["recent_return_pct"] <= policy.recent_return_max_pct,
        "ema50_slope_stable": values["ema50_slope_pct"] >= policy.ema50_slope_min_pct,
        "structure_stable_or_up": structure in {"NEUTRAL", "UP"},
    }
    short_checks = {
        "range_position": 100 - policy.range_position_max_pct <= values["range_position_pct"] <= 100 - policy.range_position_min_pct,
        "ema50_distance": -policy.ema50_distance_max_pct <= values["ema50_distance_pct"] <= -policy.ema50_distance_min_pct,
        "recent_return": -policy.recent_return_max_pct <= values["recent_return_pct"] <= -policy.recent_return_min_pct,
        "ema50_slope_stable": values["ema50_slope_pct"] <= -policy.ema50_slope_min_pct,
        "structure_stable_or_down": structure in {"NEUTRAL", "DOWN"},
    }
    return {
        "status": assessment.get("status", "unavailable"),
        "symbol": assessment.get("symbol"),
        "last_closed_bar_at": assessment.get("last_closed_bar_at"),
        "metrics": {**{key: round(value, 4) for key, value in values.items()}, "structure": structure},
        "directions": {
            direction: {"matched": all(checks.values()), "failed_conditions": [key for key, passed in checks.items() if not passed]}
            for direction, checks in (("long", long_checks), ("short", short_checks))
        },
    }


def _bollinger_directional_observation(frame, activity: dict, policy) -> dict:
    """Score four log-only Bollinger setups from completed 4H candles."""
    if not policy.enabled or frame is None or "close" not in frame:
        return {"status": "unavailable", "unavailable_reason": "closed_4h_bars_unavailable", "signals": {}}
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame else None
    required = policy.period + policy.width_lookback_bars + max(
        policy.middle_slope_lookback_bars, policy.expansion_lookback_bars, 3,
    )
    if len(close) < required:
        return {"status": "unavailable", "unavailable_reason": "insufficient_closed_4h_bars", "signals": {}}

    middle = close.rolling(policy.period, min_periods=policy.period).mean()
    deviation = close.rolling(policy.period, min_periods=policy.period).std(ddof=0) * policy.standard_deviations
    upper, lower = middle + deviation, middle - deviation
    band_width = (upper - lower) / middle.replace(0, math.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, math.nan)
    current_width = band_width.iloc[-1]
    prior_widths = band_width.iloc[-policy.width_lookback_bars - 1:-1].dropna()
    slope_base = middle.iloc[-1 - policy.middle_slope_lookback_bars]
    slope_pct = (middle.iloc[-1] / slope_base - 1) * 100 if pd.notna(slope_base) and slope_base else math.nan
    current_bands_valid = all(pd.notna(value) and math.isfinite(float(value)) for value in
                              (current_width, pct_b.iloc[-1], slope_pct))
    if not current_bands_valid or len(prior_widths) < policy.width_lookback_bars:
        return {"status": "unavailable", "unavailable_reason": "invalid_bollinger_metrics", "signals": {}}

    low_width_threshold = float(prior_widths.quantile(policy.width_low_quantile))
    low_width = float(current_width) <= low_width_threshold
    width_previous = band_width.iloc[-2]
    width_lookback = band_width.iloc[-1 - policy.expansion_lookback_bars]
    expanding = bool(pd.notna(width_previous) and pd.notna(width_lookback)
                     and current_width > width_previous and current_width > width_lookback)
    closes_above_middle = bool((close.iloc[-3:] > middle.iloc[-3:]).all())
    closes_below_middle = bool((close.iloc[-3:] < middle.iloc[-3:]).all())
    long_middle_ok = float(slope_pct) >= -policy.middle_flat_tolerance_pct
    short_middle_ok = float(slope_pct) <= policy.middle_flat_tolerance_pct
    volume_ratio = None
    volume_above_mean = None
    if volume is not None and len(volume) >= 21:
        average_volume = volume.iloc[-21:-1].mean()
        if pd.notna(average_volume) and average_volume > 0 and pd.notna(volume.iloc[-1]):
            volume_ratio = float(volume.iloc[-1] / average_volume)
            volume_above_mean = volume_ratio > 1
    volume_ok = (not policy.volume_filter_enabled or volume_above_mean is True)

    conditions = {
        "long_start": {
            "pct_b_low": float(pct_b.iloc[-1]) <= 0.25,
            "bbw_low": low_width,
            "middle_flat_or_up": long_middle_ok,
            "volume_optional": volume_ok,
        },
        "short_start": {
            "pct_b_high": float(pct_b.iloc[-1]) >= 0.75,
            "bbw_low": low_width,
            "middle_flat_or_down": short_middle_ok,
            "volume_optional": volume_ok,
        },
        "long_trend": {
            "three_closes_above_middle": closes_above_middle,
            "middle_slope_up": float(slope_pct) > 0,
            "bbw_expanding": expanding,
        },
        "short_trend": {
            "three_closes_below_middle": closes_below_middle,
            "middle_slope_down": float(slope_pct) < 0,
            "bbw_expanding": expanding,
        },
    }
    current_close = float(close.iloc[-1])
    signals = {
        name: {"matched": all(checks.values()), "failed_conditions": [key for key, passed in checks.items() if not passed]}
        for name, checks in conditions.items()
    }
    signals["long_trend"]["optional_pct_b_confirmation"] = float(pct_b.iloc[-1]) > 0.8
    signals["short_trend"]["optional_pct_b_confirmation"] = float(pct_b.iloc[-1]) < 0.2
    return {
        "status": "scored",
        "signal_close": round(current_close, 8),
        "last_closed_bar_at": frame.index[-1].isoformat() if hasattr(frame.index[-1], "isoformat") else str(frame.index[-1]),
        "pct_b": round(float(pct_b.iloc[-1]), 4),
        "bbw": round(float(current_width), 6),
        "bbw_low_threshold": round(low_width_threshold, 6),
        "bbw_low": low_width,
        "bbw_expanding": expanding,
        "middle_slope_pct": round(float(slope_pct), 4),
        "middle_direction": "up" if slope_pct > policy.middle_flat_tolerance_pct else
                            "down" if slope_pct < -policy.middle_flat_tolerance_pct else "flat",
        "volume_ratio_vs_20": round(volume_ratio, 4) if volume_ratio is not None else None,
        "volume_above_20_mean": volume_above_mean,
        "signals": signals,
    }


def _quote_timestamp(ticker: dict, received_at: datetime) -> tuple[datetime, str]:
    """Return an exchange quote timestamp when available, else the response receipt time."""
    raw = ticker.get("timestamp")
    if raw is None:
        return received_at, "exchange.fetch_ticker.received_at"
    try:
        numeric = float(raw)
        if math.isfinite(numeric):
            seconds = numeric / 1000 if abs(numeric) >= 10_000_000_000 else numeric
            return datetime.fromtimestamp(seconds, timezone.utc), "exchange.fetch_ticker.timestamp"
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    try:
        parsed = pd.Timestamp(raw).to_pydatetime()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), "exchange.fetch_ticker.timestamp"
    except (TypeError, ValueError, OverflowError):
        return received_at, "exchange.fetch_ticker.received_at"


def _fetch_pullback_quote(exchange, symbol: str) -> tuple[float | None, bool, dict]:
    """Fetch and age-check the single-symbol quote used to trigger a Paper fill."""
    ticker = exchange.fetch_ticker(symbol) or {}
    received_at = datetime.now(timezone.utc)
    quote_at, source = _quote_timestamp(ticker, received_at)
    price = _current_market_price(ticker)
    age = (received_at - quote_at).total_seconds()
    fresh = price is not None and -2 <= age <= PULLBACK_QUOTE_MAX_AGE_SECONDS
    return price, fresh, {
        "quote_at": quote_at.isoformat(),
        "quote_received_at": received_at.isoformat(),
        "quote_age_sec": round(age, 3),
        "quote_price": price,
        "quote_bid": _valid_quote_price(ticker.get("bid")),
        "quote_ask": _valid_quote_price(ticker.get("ask")),
        "quote_source": source,
    }


def _lot_filters(market: dict, *, market_order: bool = False) -> tuple[float, float]:
    filters = (market.get("info") or {}).get("filters", [])
    selected = None
    if market_order:
        selected = next((item for item in filters if item.get("filterType") == "MARKET_LOT_SIZE"), None)
    selected = selected or next((item for item in filters if item.get("filterType") == "LOT_SIZE"), None)
    if selected:
        return float(selected.get("stepSize", 1e-8)), float(selected.get("minQty", 0.0))
    return 1e-8, 0.0


class MultiPaperRunner:
    def __init__(self, exchange, settings, state: MultiPaperState, *, top_n: int = 20,
                 universe_refresh_seconds: int = 300,
                 manual_close_store: ManualCloseCommandStore | None = None, execution=None,
                 checkpoint=None, wait_pullback: bool = False,
                 trend_entry_shadow: TrendEntryShadowTracker | None = None,
                 bollinger_outcome_tracker: BollingerOutcomeTracker | None = None,
                 fetch_workers: int = 1,
                 strategy_mode: str = "regime_routed"):
        if not 1 <= top_n <= 20 or universe_refresh_seconds <= 0:
            raise ValueError("top_n must be between 1 and 20 and universe refresh interval must be positive")
        if fetch_workers < 1:
            raise ValueError("fetch_workers must be at least 1")
        if strategy_mode not in {"mean_reversion_only", "regime_routed"}:
            raise ValueError("strategy_mode must be mean_reversion_only or regime_routed")
        self.exchange = exchange
        self.settings = settings
        self.state = state
        self.top_n = top_n
        self.universe_refresh_seconds = universe_refresh_seconds
        self.fetch_workers = fetch_workers
        self.manual_close_store = manual_close_store
        self.execution = execution
        self.checkpoint = checkpoint
        self.wait_pullback = bool(wait_pullback and execution is None)
        self.strategy_mode = strategy_mode
        self.trend_entry_shadow = trend_entry_shadow
        self.bollinger_outcome_tracker = bollinger_outcome_tracker
        self.markets = exchange.load_markets()
        self._migrate_open_positions()
        self._last_universe_refresh = 0.0
        self._ranked: list[dict] = []
        self._startup_scan_pending = self.execution is not None
        self._priority_structure_close_by_symbol: dict[str, str] = {}
        self._last_symbol_regime_log_key: dict[str, str] = {}
        self._cycle_symbol_regime_observations: list[dict] = []

    def _disable_trend_entry_shadow(self, exc: Exception) -> None:
        logger.error("Trend entry shadow disabled after {}", type(exc).__name__)
        self.trend_entry_shadow = None

    def _disable_bollinger_outcome_tracker(self, exc: Exception) -> None:
        logger.error("Bollinger outcome tracking disabled after {}; official strategy continues unchanged",
                     type(exc).__name__)
        self.bollinger_outcome_tracker = None

    def _observe_symbol_regime(self, symbol: str, bars: pd.DataFrame | None, now: datetime,
                               *, fetch_error: str | None = None) -> SymbolRegimeAssessment:
        assessment = classify_symbol_regime(symbol, bars, self.settings, now=now, fetch_error=fetch_error)
        status = assessment.to_dict()
        log_key = status["last_closed_bar_at"] or f"unavailable:{now.replace(minute=0, second=0, microsecond=0).isoformat()}"
        if self._last_symbol_regime_log_key.get(symbol) == log_key:
            return assessment
        self._last_symbol_regime_log_key[symbol] = log_key
        self._cycle_symbol_regime_observations.append(status)
        logger.info("Symbol regime observation {}", json.dumps(status, ensure_ascii=False, sort_keys=True))
        return assessment

    def _log_symbol_regime_summary(self) -> None:
        observations = self._cycle_symbol_regime_observations
        self._cycle_symbol_regime_observations = []
        summary = _symbol_regime_summary(observations)
        if summary is None:
            return
        logger.info("Symbol regime summary {}", json.dumps(summary, ensure_ascii=False, sort_keys=True))

    def _update_market_risk(self, now: datetime) -> MarketRiskAssessment:
        assessment = fetch_market_risk_status(self.exchange, self.settings, now=now)
        status = assessment.to_dict()
        previous = self.state.market_risk_status
        if (previous.get("state"), previous.get("reason")) != (status["state"], status["reason"]):
            self.state.events.append({"event": "market_risk_observed", "at": status["checked_at"],
                                      **status})
        level = "warning" if assessment.state is MarketRiskState.PAUSE_NEW else "info"
        getattr(logger, level)("Market risk observation {}",
                               json.dumps(status, ensure_ascii=False, sort_keys=True))
        self.state.market_risk_status = status
        return assessment

    def _close_startup_stale_positions(self, now: datetime, tickers: dict) -> bool:
        """Close locally tracked Demo positions whose exchange snapshot confirms they remain open."""
        if not self._startup_scan_pending:
            return False
        if self.execution is None:
            self._startup_scan_pending = False
            return False
        if not self.settings.exits.startup_stale_exit:
            self._startup_scan_pending = False
            return False
        positions = self.state.exchange_snapshot.get("positions", {})
        changed = False
        retry_scan = False
        for symbol, position in list(self.state.portfolio.positions.items()):
            live = positions.get(symbol)
            if live is None:
                continue
            opened_at = position.opened_at
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            exit_state = self.state.exit_management_by_symbol.get(symbol) or {}
            max_hold_hours = effective_max_hold_hours(
                position.side, position.entry_price, position.stop_price, exit_state, self.settings.exits,
            )
            if max_hold_hours is None or (now - opened_at).total_seconds() / 3600 <= max_hold_hours:
                continue
            price = _current_market_price(tickers.get(symbol))
            if price is None:
                logger.error("Startup stale position {} exceeds {}h but no public close price is available",
                             symbol, max_hold_hours)
                retry_scan = True
                continue
            trader = PaperTrader(self.state.for_symbol(symbol), self.settings, execution=self.execution)
            trader.margin_spec = self._margin_spec(symbol)
            if trader._close(price, now, "startup_stale_exit"):
                logger.warning("Startup stale position closed for {} after {:.2f}h",
                               symbol, (now - opened_at).total_seconds() / 3600)
                changed = True
                if self.checkpoint is not None:
                    self.checkpoint()
            else:
                retry_scan = True
        self._startup_scan_pending = retry_scan
        return changed

    def _margin_spec(self, symbol: str) -> dict[str, float] | None:
        info = self.markets.get(symbol, {}).get("info", {})
        try:
            return {
                # Binance exchangeInfo exposes a public baseline rate, not the
                # private/user-specific notional bracket table.
                "maintenance_margin_rate": float(info["maintMarginPercent"]) / 100,
                "maintenance_amount": 0.0,
                "liquidation_fee_rate": float(info["liquidationFee"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _migrate_open_positions(self) -> None:
        if self.state.execution_mode == "demo_usdm":
            return
        leverage = float(self.settings.sizing.leverage)
        for symbol, position in self.state.portfolio.positions.items():
            if symbol in self.state.isolated_margin_by_symbol:
                continue
            spec = self._margin_spec(symbol)
            if spec is None:
                logger.error("Cannot migrate {} to isolated paper margin: public Binance margin metadata is incomplete", symbol)
                continue
            margin = position.quantity * position.entry_price / leverage
            if margin > self.state.cash:
                logger.error("Cannot reserve isolated margin for {}: insufficient paper cash", symbol)
                continue
            self.state.cash -= margin
            self.state.isolated_margin_by_symbol[symbol] = {**spec, "initial_margin": margin, "leverage": leverage}
            logger.info("Migrated existing paper position {} to isolated margin {:.4f} USDT; position and stop unchanged",
                        symbol, margin)

    def _entry_execution_allowed(self) -> bool:
        if self.state.execution_mode == "demo_usdm":
            return self.execution is not None and self.state.execution_status == "synchronized"
        return self.execution is None

    def _entry_allowed(self, symbol: str) -> bool:
        if symbol not in {item["symbol"] for item in self._ranked}:
            return False
        return self._entry_execution_allowed()

    def _build_fetch_worker_exchange(self):
        """Build a public exchange for parallel fetching, reusing loaded markets."""
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        configure_system_proxy(exchange)
        exchange.set_markets(self.markets)
        return exchange

    def _fetch_ohlcv_map(self, symbols: list[str], timeframe: str,
                         limits: dict[str, int]) -> dict[str, tuple]:
        """Fetch one timeframe for many symbols, returning {symbol: (rows, error)}.

        Serial when parallelism is disabled or the workload is trivial; threaded
        otherwise. Only per-symbol failures are downgraded to errors so the caller
        can reproduce the original fallback semantics exactly.
        """
        if self.fetch_workers <= 1 or len(symbols) <= 1:
            return self._fetch_ohlcv_serial(symbols, timeframe, limits)
        return self._fetch_ohlcv_threaded(symbols, timeframe, limits)

    def _fetch_ohlcv_serial(self, symbols: list[str], timeframe: str,
                            limits: dict[str, int]) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for symbol in symbols:
            try:
                rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limits[symbol])
                out[symbol] = (rows, None)
            except (ccxt.BaseError, KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
                out[symbol] = (None, exc)
        return out

    def _fetch_ohlcv_threaded(self, symbols: list[str], timeframe: str,
                              limits: dict[str, int]) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        created: list = []
        lock = threading.Lock()
        thread_local = threading.local()

        def worker(symbol: str) -> tuple[str, tuple]:
            exchange = getattr(thread_local, "exchange", None)
            if exchange is None:
                try:
                    exchange = self._build_fetch_worker_exchange()
                except Exception as exc:
                    return symbol, (None, exc)
                with lock:
                    created.append(exchange)
                thread_local.exchange = exchange
            try:
                rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limits[symbol])
                return symbol, (rows, None)
            except (ccxt.BaseError, KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
                return symbol, (None, exc)

        try:
            with ThreadPoolExecutor(max_workers=self.fetch_workers) as pool:
                for symbol, result in pool.map(worker, symbols):
                    out[symbol] = result
        finally:
            for exchange in created:
                try:
                    exchange.close()
                except Exception:
                    pass
        return out

    def _refresh_universe(self, tickers: dict, now: datetime) -> bool:
        elapsed = time.monotonic() - self._last_universe_refresh
        if self._last_universe_refresh and elapsed < self.universe_refresh_seconds:
            return False
        # Score every basically tradable contract before applying the final top-N cap.
        candidates = filter_tradable_crypto_universe(self.markets, tickers, self.settings, now=now)
        now_ms = int(now.timestamp() * 1000)
        activity = {}
        closed_1h_frames = {}
        match_assessments = {}
        trend_assessments = {}
        closed_4h_frames = {}
        outcome_active_symbols = set()
        if self.bollinger_outcome_tracker is not None:
            try:
                outcome_active_symbols = self.bollinger_outcome_tracker.active_symbols
            except Exception as exc:
                self._disable_bollinger_outcome_tracker(exc)
        symbols = [symbol for symbol, _ in candidates]
        one_hour_rows = self._fetch_ohlcv_map(
            symbols, "1h",
            {symbol: (60 if symbol in outcome_active_symbols else 26) for symbol in symbols},
        )
        four_hour_rows = self._fetch_ohlcv_map(
            symbols, "4h", {symbol: 100 for symbol in symbols},
        )
        for symbol in symbols:
            rows, fetch_error = one_hour_rows[symbol]
            if fetch_error is not None:
                closed_1h_frames[symbol] = None
                logger.warning("Hotness bars unavailable for {}; ranking will use remaining features: {}", symbol, fetch_error)
            else:
                try:
                    activity[symbol] = hot_activity_metrics(rows, now_ms=now_ms)
                    closed_1h = [row for row in rows
                                 if len(row) >= 6 and int(row[0]) + HOUR_MS <= now_ms]
                    closed_1h_frames[symbol] = ohlcv_frame(closed_1h) if closed_1h else None
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    closed_1h_frames[symbol] = None
                    logger.warning("Hotness bars unavailable for {}; ranking will use remaining features: {}", symbol, exc)

            rows_4h, fetch_error_4h = four_hour_rows[symbol]
            if fetch_error_4h is not None:
                closed_4h_frames[symbol] = None
                match_assessments[symbol] = evaluate_mean_reversion_match(symbol, None, self.settings, now=now)
                trend_assessments[symbol] = evaluate_trend_match(symbol, None, self.settings, now=now)
                logger.warning("Match-score bars unavailable for {}: {}", symbol, type(fetch_error_4h).__name__)
            else:
                try:
                    closed_4h = [row for row in rows_4h
                                 if len(row) >= 6 and int(row[0]) + TIMEFRAME_MS["4h"] <= now_ms]
                    frame_4h = ohlcv_frame(closed_4h) if closed_4h else None
                    closed_4h_frames[symbol] = frame_4h
                    match_assessments[symbol] = evaluate_mean_reversion_match(symbol, frame_4h, self.settings, now=now)
                    trend_assessments[symbol] = evaluate_trend_match(symbol, frame_4h, self.settings, now=now)
                except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
                    closed_4h_frames[symbol] = None
                    match_assessments[symbol] = evaluate_mean_reversion_match(symbol, None, self.settings, now=now)
                    trend_assessments[symbol] = evaluate_trend_match(symbol, None, self.settings, now=now)
                    logger.warning("Match-score bars unavailable for {}: {}", symbol, type(exc).__name__)
        if self.bollinger_outcome_tracker is not None:
            for symbol in sorted(outcome_active_symbols - set(closed_1h_frames)):
                try:
                    rows = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=60)
                    closed_1h = [row for row in rows
                                 if len(row) >= 6 and int(row[0]) + HOUR_MS <= now_ms]
                    closed_1h_frames[symbol] = ohlcv_frame(closed_1h) if closed_1h else None
                except (ccxt.BaseError, KeyError, TypeError, ValueError, IndexError) as exc:
                    closed_1h_frames[symbol] = None
                    logger.warning("Bollinger outcome bars unavailable for {}: {}", symbol, type(exc).__name__)
        hot_scored = score_hot_candidates(candidates, tickers, activity)
        scored_candidates = []
        for row in hot_scored:
            mean_assessment = match_assessments[row["symbol"]]
            trend_assessment = trend_assessments[row["symbol"]]
            scored_candidates.append({
                **row,
                "mean_reversion_match_score": _finite_score(mean_assessment.get("score")),
                "trend_long_match_score": _finite_score(trend_assessment.get("trend_long_match_score")),
                "trend_short_match_score": _finite_score(trend_assessment.get("trend_short_match_score")),
                "trend_match_score": _finite_score(trend_assessment.get("trend_match_score")),
                "direction_bias": trend_assessment.get("direction_bias") or "neutral",
            })

        # Tradeable universe = hotness top-N only (adaptive strategy pools removed).
        hot_top_n = scored_candidates[:self.top_n]
        ranked = hot_top_n
        legacy_candidate_rows = []
        bollinger_candidate_rows = []
        legacy_failure_counts = {"long": {}, "short": {}}
        bollinger_failure_counts = {name: {} for name in ("long_start", "short_start", "long_trend", "short_trend")}
        legacy_direction_counts = {"long": 0, "short": 0}
        bollinger_signal_counts = {name: 0 for name in bollinger_failure_counts}
        legacy_matched_symbols = set()
        bollinger_matched_symbols = {"long": set(), "short": set()}
        comparison_scored_count = 0
        comparison_unavailable_count = 0
        comparison_available_symbols = set()
        low_position_watch_candidates = []
        for row in scored_candidates:
            symbol = row["symbol"]
            assessment = trend_assessments[symbol]
            legacy = _legacy_directional_observation(assessment, self.settings.low_position_watch)
            bollinger = _bollinger_directional_observation(
                closed_4h_frames.get(symbol), activity.get(symbol, {}), self.settings.bollinger_observation,
            )
            if legacy.get("status") == "scored" and bollinger.get("status") == "scored":
                comparison_scored_count += 1
                comparison_available_symbols.add(symbol)
            else:
                comparison_unavailable_count += 1
            directions = legacy.get("directions", {})
            matched_directions = [direction for direction in ("long", "short")
                                  if directions.get(direction, {}).get("matched")]
            if matched_directions:
                legacy_matched_symbols.add(symbol)
                for direction in matched_directions:
                    legacy_direction_counts[direction] += 1
                    legacy_matched_symbols.add(symbol)
                metrics = legacy.get("metrics", {})
                legacy_candidate_rows.append({
                    "symbol": symbol,
                    "current_price": _current_market_price(tickers.get(symbol)),
                    "signal_close": (closed_4h_frames[symbol]["close"].iloc[-1]
                                     if closed_4h_frames.get(symbol) is not None else None),
                    "last_closed_bar_at": legacy.get("last_closed_bar_at"),
                    "matched_directions": matched_directions,
                    "metrics": metrics,
                    "conditions": directions,
                })
                low_position_watch_candidates.append({
                    "candidate_type": "legacy_directional_observation",
                    "symbol": symbol,
                    "matched_directions": matched_directions,
                    **metrics,
                    "last_closed_bar_at": legacy.get("last_closed_bar_at"),
                })
            for direction, outcome in directions.items():
                for failed in outcome.get("failed_conditions", []):
                    legacy_failure_counts[direction][failed] = legacy_failure_counts[direction].get(failed, 0) + 1

            bb_signals = bollinger.get("signals", {})
            matched_bb = [name for name, outcome in bb_signals.items()
                          if name in bollinger_signal_counts and outcome.get("matched")]
            if matched_bb:
                for name in matched_bb:
                    bollinger_signal_counts[name] += 1
                    direction = "long" if name.startswith("long") else "short"
                    bollinger_matched_symbols[direction].add(symbol)
                bollinger_candidate_rows.append({
                    "symbol": symbol,
                    "current_price": _current_market_price(tickers.get(symbol)),
                    "signal_types": matched_bb,
                    **{key: value for key, value in bollinger.items() if key not in {"signals", "status"}},
                    "signal_conditions": {name: bb_signals[name] for name in matched_bb},
                })
            for name, outcome in bb_signals.items():
                if name not in bollinger_failure_counts:
                    continue
                for failed in outcome.get("failed_conditions", []):
                    bollinger_failure_counts[name][failed] = bollinger_failure_counts[name].get(failed, 0) + 1

        outcome_tracking = {
            "enabled": False,
            "reason": "tracker_unavailable",
            "rules_version": "bollinger_observer_v1",
        }
        if self.bollinger_outcome_tracker is not None:
            try:
                outcome_tracking = self.bollinger_outcome_tracker.process_refresh(
                    bollinger_candidate_rows, closed_1h_frames, now=now,
                )
            except Exception as exc:
                self._disable_bollinger_outcome_tracker(exc)
                outcome_tracking = {
                    "enabled": False,
                    "reason": type(exc).__name__,
                    "rules_version": "bollinger_observer_v1",
                }

        legacy_long_symbols = {row["symbol"] for row in legacy_candidate_rows if "long" in row["matched_directions"]}
        legacy_short_symbols = {row["symbol"] for row in legacy_candidate_rows if "short" in row["matched_directions"]}
        legacy_long_comparable = legacy_long_symbols & comparison_available_symbols
        legacy_short_comparable = legacy_short_symbols & comparison_available_symbols
        legacy_any_symbols = (legacy_long_symbols | legacy_short_symbols) & comparison_available_symbols
        bollinger_any_symbols = (bollinger_matched_symbols["long"] | bollinger_matched_symbols["short"]) & comparison_available_symbols
        directional_overlap = {
            "long": len(legacy_long_comparable & bollinger_matched_symbols["long"]),
            "short": len(legacy_short_comparable & bollinger_matched_symbols["short"]),
        }
        if not ranked:
            logger.warning("No crypto contracts passed the basic tradability filters")
        old = {item["symbol"] for item in self.state.watchlist}
        current_hour_ms = int(now.timestamp() * 1000) // HOUR_MS * HOUR_MS
        first_eligible_close = datetime.fromtimestamp((current_hour_ms + HOUR_MS) / 1000, timezone.utc).isoformat()
        for item in ranked:
            symbol = item["symbol"]
            if symbol not in old:
                self.state.entry_after[symbol] = first_eligible_close
        self._ranked = ranked
        self.state.watchlist = [{**row, "rank": rank} for rank, row in enumerate(ranked, start=1)]
        self._last_universe_refresh = time.monotonic()
        selected = [{
            "rank": rank,
            "symbol": row["symbol"],
            "hot_score": row["hot_score"],
            "mean_reversion_match_score": row["mean_reversion_match_score"],
            "trend_long_match_score": row["trend_long_match_score"],
            "trend_short_match_score": row["trend_short_match_score"],
            "trend_match_score": row["trend_match_score"],
            "direction_bias": row["direction_bias"],
            "candidate_type": "hot_top_n_reference",
            "quote_volume_24h": row["quote_volume_24h"],
            "trade_count_24h": row["trade_count_24h"],
            "volume_burst_ratio": row["volume_burst_ratio"],
            "return_4h_pct": row["return_4h_pct"],
            "return_24h_pct": row["return_24h_pct"],
        } for rank, row in enumerate(ranked, start=1)]
        evaluated = [{
            "symbol": row["symbol"],
            "hot_score": row["hot_score"],
            "mean_reversion_match_score": match_assessments[row["symbol"]]["score"],
            "match_status": match_assessments[row["symbol"]]["status"],
            "unavailable_reason": match_assessments[row["symbol"]]["unavailable_reason"],
            "route_state": match_assessments[row["symbol"]]["route_state"],
            "legacy_regime": match_assessments[row["symbol"]]["legacy_regime"],
            "component_scores": match_assessments[row["symbol"]]["component_scores"],
            "penalties": match_assessments[row["symbol"]]["penalties"],
            "reasons": match_assessments[row["symbol"]]["reasons"],
        } for row in scored_candidates]
        trend_evaluated = [{
            "symbol": row["symbol"],
            "hot_score": row["hot_score"],
            "mean_reversion_match_score": match_assessments[row["symbol"]]["score"],
            "trend_long_match_score": trend_assessments[row["symbol"]]["trend_long_match_score"],
            "trend_short_match_score": trend_assessments[row["symbol"]]["trend_short_match_score"],
            "trend_match_score": trend_assessments[row["symbol"]]["trend_match_score"],
            "direction_bias": trend_assessments[row["symbol"]]["direction_bias"],
            "score_aggregation": trend_assessments[row["symbol"]]["score_aggregation"],
            "match_status": trend_assessments[row["symbol"]]["status"],
            "unavailable_reason": trend_assessments[row["symbol"]]["unavailable_reason"],
            "last_closed_bar_at": trend_assessments[row["symbol"]]["last_closed_bar_at"],
            "route_state": match_assessments[row["symbol"]]["route_state"],
            "metrics": trend_assessments[row["symbol"]]["metrics"],
            "long_component_scores": trend_assessments[row["symbol"]]["long"]["component_scores"],
            "short_component_scores": trend_assessments[row["symbol"]]["short"]["component_scores"],
            "long_weighted_contributions": trend_assessments[row["symbol"]]["long"]["weighted_contributions"],
            "short_weighted_contributions": trend_assessments[row["symbol"]]["short"]["weighted_contributions"],
            "long_deductions": trend_assessments[row["symbol"]]["long"]["deductions"],
            "short_deductions": trend_assessments[row["symbol"]]["short"]["deductions"],
            "long_reasons": trend_assessments[row["symbol"]]["long"]["reasons"],
            "short_reasons": trend_assessments[row["symbol"]]["short"]["reasons"],
            "long_deduction_reasons": trend_assessments[row["symbol"]]["long"]["deduction_reasons"],
            "short_deduction_reasons": trend_assessments[row["symbol"]]["short"]["deduction_reasons"],
            "common_deductions": trend_assessments[row["symbol"]]["common_deductions"],
            "component_scores": trend_assessments[row["symbol"]]["component_scores"],
            "weighted_contributions": trend_assessments[row["symbol"]]["weighted_contributions"],
            "deductions": trend_assessments[row["symbol"]]["deductions"],
            "reasons": trend_assessments[row["symbol"]]["reasons"],
            "deduction_reasons": trend_assessments[row["symbol"]]["deduction_reasons"],
        } for row in scored_candidates]
        refresh_log = {
            "event": "hot_universe_refreshed",
            "at": now.isoformat(),
            "basic_tradable_count": len(candidates),
            "activity_metrics_available_count": sum(
                any(value is not None for value in activity.get(symbol, {}).values())
                for symbol, _ in candidates
            ),
            "top_n": self.top_n,
            "ranking_basis": "hot_score_desc, quote_volume_24h_desc, symbol_asc",
            "candidate_selection_mode": "hot_top_n",
            "hot_score_weights": {
                "closed_1h_volume_burst": 0.40,
                "closed_1h_abs_4h_and_24h_moves": 0.30,
                "ticker_24h_quote_volume_and_trade_count": 0.30,
            },
            "selected_count": len(ranked),
            "selected_candidates": selected,
            "hot_top_n_count": len(hot_top_n),
            "hot_top_n_candidates": [{
                "rank": rank,
                "symbol": row["symbol"],
                "hot_score": row["hot_score"],
                "candidate_type": "hot_top_n_reference",
                "mean_reversion_match_score": row["mean_reversion_match_score"],
                "trend_long_match_score": row["trend_long_match_score"],
                "trend_short_match_score": row["trend_short_match_score"],
                "trend_match_score": row["trend_match_score"],
                "direction_bias": row["direction_bias"],
            } for rank, row in enumerate(hot_top_n, start=1)],
            "low_position_watch": {
                "enabled": self.settings.low_position_watch.enabled,
                "mode": "observation_only",
                "affects_candidate_selection": False,
                "criteria": self.settings.low_position_watch.model_dump(exclude={"enabled"}),
                "candidate_count": len(low_position_watch_candidates),
                "candidates": low_position_watch_candidates,
            },
            "observation_comparison": {
                "rules_version": "bollinger_observer_v1",
                "mode": "independent_log_only_comparison",
                "timeframe": "4h",
                "uses_closed_bars_only": True,
                "affects_candidate_selection": False,
                "evaluated_count": comparison_scored_count,
                "unavailable_count": comparison_unavailable_count,
                "legacy_rules": {
                    "enabled": self.settings.low_position_watch.enabled,
                    "direction_policy": "existing_long_rules_plus_symmetric_short_mirror",
                    "long_candidate_count": legacy_direction_counts["long"],
                    "short_candidate_count": legacy_direction_counts["short"],
                    "failure_condition_counts": legacy_failure_counts,
                    "candidates": legacy_candidate_rows,
                },
                "bollinger": {
                    "enabled": self.settings.bollinger_observation.enabled,
                    "parameters": self.settings.bollinger_observation.model_dump(exclude={"enabled"}),
                    "signal_counts": bollinger_signal_counts,
                    "failure_condition_counts": bollinger_failure_counts,
                    "candidates": bollinger_candidate_rows,
                    "outcome_tracking": outcome_tracking,
                },
                "overlap": {
                    "same_direction": directional_overlap,
                    "legacy_only_symbols": len(legacy_any_symbols - bollinger_any_symbols),
                    "bollinger_only_symbols": len(bollinger_any_symbols - legacy_any_symbols),
                    "both_rule_families_symbols": len(legacy_any_symbols & bollinger_any_symbols),
                    "neither_rule_family_symbols": max(
                        0, comparison_scored_count - len(legacy_any_symbols | bollinger_any_symbols),
                    ),
                },
            },
            "mean_reversion_match": {
                "mode": "observation_only",
                "weights": self.settings.mean_reversion_match.weights.model_dump(),
                "penalties": self.settings.mean_reversion_match.penalties.model_dump(),
                "evaluated_count": len(evaluated),
                "scored_count": sum(row["match_status"] == "scored" for row in evaluated),
                "candidates": evaluated,
            },
            "trend_match": {
                "mode": "observation_only",
                "score_aggregation": self.settings.trend_match.score_aggregation,
                "direction_bias_min_score": self.settings.trend_match.direction_bias_min_score,
                "direction_bias_min_delta": self.settings.trend_match.direction_bias_min_delta,
                "weights": self.settings.trend_match.weights.model_dump(),
                "parameters": self.settings.trend_match.model_dump(exclude={"weights"}),
                "evaluated_count": len(trend_evaluated),
                "scored_count": sum(row["match_status"] == "scored" for row in trend_evaluated),
                "candidates": trend_evaluated,
            },
        }
        logger.info("Hot crypto universe refresh {}", json.dumps(refresh_log, ensure_ascii=False, sort_keys=True))
        return True

    def _process_manual_close_requests(self, tickers: dict, now: datetime, hour_open_key: str) -> bool:
        if self.manual_close_store is None:
            return False
        changed = False
        for request in self.manual_close_store.pending():
            request_id, symbol = request["request_id"], request["symbol"]
            if symbol not in self.state.portfolio.positions:
                live = (self.state.exchange_snapshot.get("positions", {}).get(symbol)
                        if self.execution is not None else None)
                if live is not None:
                    try:
                        from bot.signals import Side
                        position = SimpleNamespace(symbol=symbol, side=Side(live["side"]))
                        result = self.execution.close(position, stop=None, reason="manual_close_external_position")
                        fill = result.get("fill")
                        if fill is None:
                            raise RuntimeError("Demo close was not confirmed filled")
                        entry = float(live["entry_price"])
                        direction = 1 if live["side"] == "long" else -1
                        gross = float(fill["filled"]) * (float(fill["average"]) - entry) * direction
                        fee = fill.get("fee_cost")
                        if fee is None:
                            fee = float(fill["filled"]) * float(fill["average"]) * self.settings.fees.taker_bps / 10_000
                        self.state.events.append({
                            "event": "external_position_reduce_only_close", "at": now.isoformat(),
                            "symbol": symbol, "side": live["side"], "quantity": fill["filled"],
                            "price": fill["average"], "orderId": fill["id"], "host": result.get("host"),
                            "execution_source": "binance_usdm_demo", "reason": "manual_close",
                            "gross_pnl_estimate": gross, "close_fee": fee,
                            "warning": "开仓费用与资金费未知，此记录为估算且不计入完整策略盈亏",
                        })
                        self.state.exchange_snapshot = self.execution.sync_snapshot()
                        if self.state.exchange_snapshot.get("available_usdt") is not None:
                            self.state.cash = float(self.state.exchange_snapshot["available_usdt"])
                        self.state.execution_status = "inconsistent"
                        self.state.inconsistencies = [
                            item for item in self.state.inconsistencies if not item.startswith(f"{symbol}:")
                        ]
                        self.state.last_processed_by_symbol[symbol] = hour_open_key
                        self.manual_close_store.complete(request_id, {
                            "status": "closed", "symbol": symbol, "price": fill["average"],
                            "net_pnl": None, "fees": fee, "reason": "manual_close",
                            "message": "Demo 仓位已按 reduceOnly 平仓；因缺少机器人开仓费用／资金费记录，未计算完整净盈亏。仍需同步账户后恢复开仓。",
                        })
                        changed = True
                        continue
                    except Exception as exc:
                        self.manual_close_store.complete(request_id, {
                            "status": "rejected", "symbol": symbol,
                            "message": f"Demo 平仓未确认：{type(exc).__name__}，请检查交易所持仓",
                        })
                        continue
                self.manual_close_store.complete(request_id, {
                    "status": "rejected", "symbol": symbol, "message": "该标的已无持仓",
                })
                continue
            price = _current_market_price(tickers.get(symbol))
            if price is None:
                # Keep the request queued until a fresh public last-trade price is available.
                continue
            trader = PaperTrader(self.state.for_symbol(symbol), self.settings, execution=self.execution)
            trader.margin_spec = self._margin_spec(symbol)
            if not trader.manual_close(price, now, request_id=request_id):
                self.manual_close_store.complete(request_id, {
                    "status": "rejected", "symbol": symbol, "message": "该标的已无持仓",
                })
                continue
            # Do not let an older queued signal re-open or reverse the manually closed leg
            # on the same 1H decision boundary.
            self.state.last_processed_by_symbol[symbol] = hour_open_key
            close_fill = next(event for event in reversed(self.state.events)
                              if event.get("event") == "paper_fill"
                              and event.get("action") == "CLOSE"
                              and event.get("symbol") == symbol
                              and event.get("reason") == "manual_close")
            self.manual_close_store.complete(request_id, {
                "status": "closed", "symbol": symbol, "price": close_fill["price"],
                "net_pnl": close_fill["net_pnl"], "fees": close_fill["fees"],
                "reason": "manual_close",
            })
            logger.info("Manual paper close completed for {} at latest public price {}", symbol, price)
            changed = True
        return changed

    def _reconcile_external_flat_positions(self, snapshot: dict, now: datetime) -> tuple[set[str], set[str]]:
        """Drop stale local legs only after a post-entry external fill and flat Demo snapshot."""
        if self.execution is None or not callable(
                getattr(self.execution, "cancel_saved_stop_if_open", None)):
            return set(), set()
        live_symbols = set((snapshot.get("positions") or {}).keys())
        reconciled_symbols: set[str] = set()
        resolved_order_ids: set[str] = set()
        for symbol, position in list(self.state.portfolio.positions.items()):
            if symbol in live_symbols:
                continue
            opened_at = position.opened_at
            candidates = []
            for event in self.state.events:
                if (event.get("event") != "foreign_exchange_fill_warning"
                        or event.get("symbol") != symbol):
                    continue
                try:
                    filled_at = datetime.fromisoformat(event["at"])
                except (KeyError, TypeError, ValueError):
                    continue
                if filled_at.tzinfo is not None and filled_at >= opened_at:
                    candidates.append((filled_at, event))
            if not candidates:
                continue

            order_data = self.state.exchange_orders_by_symbol.get(symbol, {})
            stop = order_data.get("stop") if isinstance(order_data, dict) else None
            verified_stop_fill = None
            resolver = getattr(self.execution, "resolve_protective_stop_fill", None)
            if callable(resolver):
                for _, warning in sorted(candidates, key=lambda item: item[0]):
                    order_id = warning.get("orderId")
                    if not order_id:
                        continue
                    try:
                        verified_stop_fill = resolver(symbol, str(order_id), stop, position.quantity)
                    except Exception as exc:
                        logger.warning("Cannot verify Demo protective stop fill for {}: {}",
                                       symbol, type(exc).__name__)
                    if verified_stop_fill:
                        break
            try:
                self.execution.cancel_saved_stop_if_open(symbol, stop)
            except Exception as exc:
                logger.error("Cannot confirm stale Demo stop cleanup for {}; position remains locked: {}",
                             symbol, type(exc).__name__)
                continue

            close_at = (datetime.fromisoformat(verified_stop_fill["closed_at"])
                        if verified_stop_fill else max(now, opened_at))
            if close_at.tzinfo is None:
                close_at = close_at.replace(tzinfo=timezone.utc)
            exit_state = self.state.exit_management_by_symbol.get(symbol) or {}
            entry_cost = self.state.entry_costs.get(symbol) or {}
            if verified_stop_fill:
                if "entry_fee" in entry_cost and "funding_paid" in entry_cost:
                    gross_pnl = float(exit_state.get("realized_gross", 0.0)) + verified_stop_fill["gross_pnl"]
                    fees = (float(entry_cost["entry_fee"]) + verified_stop_fill["exit_fee"]
                            + float(exit_state.get("realized_exit_fees", 0.0))
                            + float(exit_state.get("realized_entry_fee", 0.0)))
                    funding = (float(entry_cost["funding_paid"])
                               + float(exit_state.get("realized_funding", 0.0)))
                    net_pnl = gross_pnl - fees - funding
                else:
                    gross_pnl = fees = funding = net_pnl = None
                close_reason = "protective_stop_exchange_algo"
            else:
                gross_pnl = fees = funding = net_pnl = None
                close_reason = "external_fill_after_entry_and_exchange_position_flat"
            self.state.portfolio.register_close(
                symbol, close_at, reason=close_reason,
                net_pnl=net_pnl,
            )
            for mapping in (self.state.entry_costs, self.state.isolated_margin_by_symbol,
                            self.state.exit_management_by_symbol, self.state.pending_by_symbol,
                            self.state.exchange_orders_by_symbol, self.state.mark_prices):
                mapping.pop(symbol, None)
            self.state.last_processed_by_symbol[symbol] = now.replace(
                minute=0, second=0, microsecond=0).isoformat()
            order_ids = [str(item[1].get("orderId")) for item in candidates
                         if item[1].get("orderId")]
            if verified_stop_fill:
                resolved_order_ids.add(verified_stop_fill["order_id"])
            else:
                resolved_order_ids.update(order_ids)
            self.state.foreign_order_ids = [
                str(order_id) for order_id in self.state.foreign_order_ids
                if str(order_id) not in resolved_order_ids
            ]
            self.state.events.append({
                "event": "external_position_reconciled_flat",
                "at": close_at.isoformat() if verified_stop_fill else now.isoformat(),
                "closed_at": close_at.isoformat() if verified_stop_fill else None,
                "reconciled_at": now.isoformat(),
                "symbol": symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "external_order_ids": order_ids,
                "orderId": verified_stop_fill["order_id"] if verified_stop_fill else None,
                "algoId": verified_stop_fill["algo_id"] if verified_stop_fill else None,
                "execution_source": "binance_usdm_demo" if verified_stop_fill else None,
                "close_price": verified_stop_fill["average"] if verified_stop_fill else None,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "funding_paid": funding,
                "net_pnl": net_pnl,
                "pnl_recorded": verified_stop_fill is not None and net_pnl is not None,
                "reason": close_reason,
                "warning": ("已通过机器人 Algo 止损单与币安逐笔成交双重核验并计入平仓统计"
                            if verified_stop_fill and net_pnl is not None else
                            "已确认机器人保护止损成交；本地入场成本或资金费记录不完整，净盈亏仍未记账"
                            if verified_stop_fill else
                            "按 Demo 账户空仓核对并清除本地残仓；成交价、完整盈亏未从历史成交复原"),
            })
            reconciled_symbols.add(symbol)
            if verified_stop_fill:
                logger.info("Reconciled {} bot protective stop fill at {} with net PnL {}",
                            symbol, verified_stop_fill["average"], net_pnl)
            else:
                logger.warning("Reconciled {} local position to flat Demo snapshot after external fill; PnL left unrecorded",
                               symbol)
        return reconciled_symbols, resolved_order_ids

    def _retry_protective_stop_cleanup(self, snapshot: dict, now: datetime) -> tuple[set[str], set[str]]:
        cleaned_symbols: set[str] = set()
        pending_symbols: set[str] = set()
        reconcile_stops = getattr(self.execution, "reconcile_reduced_position_stops", None)
        if reconcile_stops is None:
            return cleaned_symbols, pending_symbols
        local_positions = self.state.portfolio.positions
        exchange_positions = snapshot.get("positions", {})
        for symbol, order_data in list(self.state.exchange_orders_by_symbol.items()):
            if not order_data.get("stale_stops") and not order_data.get("snapshot_reconciliation_required"):
                continue
            position = local_positions.get(symbol)
            live = exchange_positions.get(symbol)
            if position is None or live is None:
                continue
            step, _, _ = self.execution._market_rules(symbol)
            if (live["side"] != position.side.value
                    or abs(float(live["quantity"]) - position.quantity) > step + 1e-12):
                pending_symbols.add(symbol)
                continue
            try:
                stale = reconcile_stops(symbol, position.side.value,
                                        order_data.get("stop"), position.quantity)
            except Exception as exc:
                stale = order_data.get("stale_stops", [])
                logger.warning("Protective stop cleanup retry failed for {}: {}", symbol, type(exc).__name__)
            if stale:
                order_data["stale_stops"] = stale
                pending_symbols.add(symbol)
            else:
                order_data.pop("stale_stops", None)
                order_data.pop("snapshot_reconciliation_required", None)
                cleaned_symbols.add(symbol)
                self.state.events.append({
                    "event": "protective_stop_cleanup_recovered", "at": now.isoformat(),
                    "symbol": symbol, "remaining_quantity": position.quantity,
                })
        return cleaned_symbols, pending_symbols

    @staticmethod
    def _is_protective_stop_audit_mismatch(item: str) -> bool:
        value = item.lower()
        return (value.startswith("protective stop audit:")
                or "active protective stop id missing" in value
                or "saved protective stop is not active" in value
                or "extra active bot protective stop" in value
                or "active protective stop audit failed" in value)

    def _audit_active_protective_stops(self, now: datetime) -> list[str]:
        """Read and record a fresh Demo stop audit without placing or changing orders."""
        audit_stops = getattr(self.execution, "audit_active_protective_stops", None)
        try:
            if not callable(audit_stops):
                raise RuntimeError("audit unavailable")
            mismatches = list(audit_stops(self.state.portfolio.positions,
                                          self.state.exchange_orders_by_symbol) or [])
            status = "mismatch" if mismatches else "verified"
        except Exception as exc:
            mismatches = [f"protective stop audit: {type(exc).__name__}"]
            status = "error"
        self.state.protective_stop_audit = {
            "at": now.isoformat(), "status": status,
            "position_count": len(self.state.portfolio.positions),
            "mismatch_count": len(mismatches), "mismatches": mismatches,
        }
        return mismatches

    def _recover_unexplained_execution_lock(self, mismatches: list[str],
                                            prior_inconsistencies: list[str],
                                            now: datetime,
                                            stop_audit_mismatches: list[str] | None = None) -> list[str]:
        """Clear only an empty-reason lock after a fresh position and stop audit."""
        if (self.state.execution_status != "inconsistent" or mismatches or prior_inconsistencies
                or self.state.foreign_order_ids):
            return []
        audit_stops = getattr(self.execution, "audit_active_protective_stops", None)
        if not callable(audit_stops):
            return ["active protective stop audit unavailable; execution lock retained"]
        if stop_audit_mismatches is not None:
            stop_mismatches = stop_audit_mismatches
        else:
            try:
                stop_mismatches = audit_stops(self.state.portfolio.positions,
                                              self.state.exchange_orders_by_symbol)
            except Exception as exc:
                return [f"active protective stop audit failed: {type(exc).__name__}"]
        if stop_mismatches:
            return list(stop_mismatches)
        self.state.execution_status = "synchronized"
        self.state.inconsistencies = []
        self.state.events.append({
            "event": "execution_status_recovered",
            "at": now.isoformat(),
            "reason": "fresh_positions_and_protective_stops_verified_without_unresolved_mismatches",
            "position_count": len(self.state.portfolio.positions),
        })
        return []

    def cycle(self) -> bool:
        self._cycle_symbol_regime_observations = []
        now_ms = self.exchange.milliseconds()
        now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
        tickers = self.exchange.fetch_tickers()
        changed = False
        if self.execution is not None:
            prior_inconsistencies = list(self.state.inconsistencies)
            snapshot = self.execution.sync_snapshot()
            self.state.exchange_snapshot = snapshot
            if snapshot.get("available_usdt") is not None:
                self.state.cash = float(snapshot["available_usdt"])
            exchange_positions = snapshot.get("positions", {})
            local_positions = self.state.portfolio.positions
            mismatches = []
            stale_stop_symbols, stop_cleanup_pending = self._retry_protective_stop_cleanup(snapshot, now)
            changed = bool(stale_stop_symbols) or changed
            for symbol in sorted(set(exchange_positions) | set(local_positions)):
                live = exchange_positions.get(symbol)
                local = local_positions.get(symbol)
                if live is None or local is None:
                    mismatch = f"{symbol}: local/exchange position missing on one side"
                    mismatches.append(mismatch)
                    if live is not None and local is None and not any(
                        event.get("symbol") == symbol and event.get("execution_source") == "binance_usdm_demo"
                        for event in self.state.events
                    ) and not any(event.get("symbol") == symbol for event in self.state.events
                                  if event.get("event") == "foreign_exchange_fill_warning"):
                        warning = {"event": "foreign_exchange_fill_warning", "at": snapshot["at"],
                                   "symbol": symbol, "reason": "exchange_position_without_robot_order_history",
                                   "warning": "未知来源仓位，以交易所仓位为准"}
                        self.state.events.append(warning)
                    continue
                step, _, _ = self.execution._market_rules(symbol)
                if (live["side"] != local.side.value
                        or abs(float(live["quantity"]) - local.quantity) > step + 1e-12
                        or not math.isclose(float(live["entry_price"]), local.entry_price, rel_tol=1e-6, abs_tol=1e-10)):
                    mismatches.append(f"{symbol}: side/quantity/average differs")
            mismatches.extend(f"{symbol}: stale bot protective stop cleanup pending"
                              for symbol in sorted(stop_cleanup_pending))
            if hasattr(self.execution, "audit_recent_fills"):
                tracked_symbols = set(exchange_positions) | set(local_positions)
                tracked_symbols.update(event.get("symbol") for event in self.state.events
                                       if event.get("execution_source") == "binance_usdm_demo" and event.get("symbol"))
                known_order_ids = {str(event.get("orderId")) for event in self.state.events
                                   if event.get("execution_source") == "binance_usdm_demo" and event.get("orderId")}
                known_order_ids.update(str(order.get("id")) for data in self.state.exchange_orders_by_symbol.values()
                                       for order in [data.get("stop", {})] if order.get("id"))
                foreign_ids = set(map(str, self.state.foreign_order_ids))
                foreign_ids.update(
                    str(order_id)
                    for event in self.state.events
                    if event.get("event") == "external_position_reconciled_flat"
                    for order_id in event.get("external_order_ids", [])
                )
                for warning in self.execution.audit_recent_fills(tracked_symbols, known_order_ids, foreign_ids):
                    foreign_ids.add(warning["orderId"])
                    self.state.foreign_order_ids.append(warning["orderId"])
                    self.state.events.append({"event": "foreign_exchange_fill_warning", "at": now.isoformat(),
                                              **warning, "warning": "发现非机器人 orderId 成交，以交易所仓位为准"})
                    mismatches.append(f"{warning['symbol']}: foreign orderId {warning['orderId']} fill detected")

            reconciled_symbols, resolved_order_ids = self._reconcile_external_flat_positions(snapshot, now)
            reconciled_symbols.update(stale_stop_symbols)
            changed = bool(reconciled_symbols) or changed
            if reconciled_symbols:
                mismatches = [item for item in mismatches
                              if not any(item.startswith(f"{symbol}:") for symbol in reconciled_symbols)]
            if resolved_order_ids:
                mismatches = [item for item in mismatches
                              if not any(item.endswith(f"foreign orderId {order_id} fill detected")
                                         for order_id in resolved_order_ids)]
            resolved_ids_in_state = {
                str(order_id)
                for event in self.state.events
                if event.get("event") == "external_position_reconciled_flat"
                for order_id in event.get("external_order_ids", [])
            }
            resolved_ids_in_state.update(resolved_order_ids)
            self.state.foreign_order_ids = [
                str(order_id) for order_id in self.state.foreign_order_ids
                if str(order_id) not in resolved_ids_in_state
            ]
            for order_id in self.state.foreign_order_ids:
                if str(order_id) in resolved_ids_in_state:
                    continue
                warning = next((event for event in reversed(self.state.events)
                                if event.get("event") == "foreign_exchange_fill_warning"
                                and str(event.get("orderId")) == str(order_id)), None)
                if warning:
                    mismatch = f"{warning.get('symbol')}: foreign orderId {order_id} fill detected"
                    if mismatch not in mismatches:
                        mismatches.append(mismatch)
            previous_stop_audit = self.state.protective_stop_audit
            stop_audit_mismatches = self._audit_active_protective_stops(now)
            changed = (self.state.protective_stop_audit != previous_stop_audit) or changed
            mismatches.extend(stop_audit_mismatches)
            prior_non_audit_inconsistencies = [
                item for item in prior_inconsistencies
                if not self._is_protective_stop_audit_mismatch(item)
            ]
            mismatches.extend(self._recover_unexplained_execution_lock(
                mismatches, prior_non_audit_inconsistencies, now, stop_audit_mismatches,
            ))
            if mismatches:
                self.state.execution_status = "inconsistent"
                self.state.inconsistencies = mismatches
                logger.error("Exchange/local state inconsistent; new entries disabled: {}", "; ".join(mismatches))
            elif (self.state.execution_status != "inconsistent"
                  or (reconciled_symbols and prior_inconsistencies
                      and all(any(item.startswith(f"{symbol}:") for symbol in reconciled_symbols)
                              or any(f"foreign orderId {order_id} fill detected" in item
                                     for order_id in resolved_ids_in_state)
                              for item in prior_inconsistencies))):
                self.state.execution_status = "synchronized"
                self.state.inconsistencies = []
            startup_closed = self._close_startup_stale_positions(now, tickers)
            changed = startup_closed or changed
        hour_open_ms = now_ms // HOUR_MS * HOUR_MS
        hour_open_at = datetime.fromtimestamp(hour_open_ms / 1000, timezone.utc)
        hour_open_key = pd.Timestamp(hour_open_at).isoformat()

        positions = list(self.state.portfolio.positions)
        self._priority_structure_close_by_symbol = {
            symbol: close_key
            for symbol, checked_key in self._priority_structure_close_by_symbol.items()
            if symbol in positions and checked_key == close_key
        }
        priority_1h_rows: dict[str, list] = {}
        priority_structure_checked = {
            symbol for symbol, checked_key in self._priority_structure_close_by_symbol.items()
            if checked_key == close_key
        }
        for symbol in positions:
            view = self.state.for_symbol(symbol)
            if (view.last_processed_close == hour_open_key and view.pending is None
                    or symbol in priority_structure_checked):
                continue
            try:
                raw_1h = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
                priority_1h_rows[symbol] = raw_1h
                closed_1h = [row for row in raw_1h if int(row[0]) + HOUR_MS <= now_ms]
                if (not closed_1h
                        or int(closed_1h[-1][0]) + HOUR_MS != hour_open_ms):
                    continue
                price = _current_market_price(tickers.get(symbol))
                if price is None:
                    continue

                trader = PaperTrader(view, self.settings, execution=self.execution)
                trader.margin_spec = self._margin_spec(symbol)
                candle_close = float(closed_1h[-1][4])
                checked, closed = trader.close_on_structure_signal(
                    candle_close=candle_close,
                    current_price=price,
                    now=now,
                    close_key=hour_open_key,
                )
                if checked:
                    self._priority_structure_close_by_symbol[symbol] = hour_open_key
                    priority_structure_checked.add(symbol)
                if closed:
                    changed = True
                    if self.checkpoint is not None:
                        self.checkpoint()
            except (ccxt.BaseError, KeyError, TypeError, ValueError, IndexError) as exc:
                logger.warning("Priority structure check failed for {}; main cycle will retry: {}",
                               symbol, type(exc).__name__)

        market_risk = self._update_market_risk(now)
        changed = self._refresh_universe(tickers, now) or changed
        eligible = [item["symbol"] for item in self._ranked]
        eligible_symbols = set(eligible)
        manual_close_changed = self._process_manual_close_requests(tickers, now, hour_open_key)
        changed = manual_close_changed or changed
        if manual_close_changed and self.checkpoint is not None:
            self.checkpoint()

        # Retain open symbols for stop/exit management even after they leave the hot ranking.
        positions = list(self.state.portfolio.positions.keys())
        pending = list(self.state.pending_by_symbol.keys())
        shadow_symbols = []
        if self.trend_entry_shadow is not None:
            try:
                shadow_symbols = sorted(self.trend_entry_shadow.active_symbols)
            except Exception as exc:
                self._disable_trend_entry_shadow(exc)
        tracked = _ordered_tracked_symbols(positions, pending, [*shadow_symbols, *eligible])
        watch_symbols = {row["symbol"] for row in self.state.watchlist}
        for symbol in self.state.portfolio.positions:
            if symbol not in watch_symbols:
                self.state.watchlist.append({"symbol": symbol, "rank": None,
                                             "quote_volume_24h": float((tickers.get(symbol) or {}).get("quoteVolume") or 0),
                                             "held_outside_top": True})

        # Mark all currently held legs first so every new order sees portfolio equity at market.
        for symbol in self.state.portfolio.positions:
            price = _current_market_price(tickers.get(symbol))
            if price is not None:
                self.state.mark_prices[symbol] = price

        mark_prices: dict[str, float] = {}
        for symbol in self.state.portfolio.positions:
            try:
                result = self.exchange.fetch_mark_price(symbol)
                value = result.get("markPrice") or result.get("mark")
                if value is not None and math.isfinite(float(value)) and float(value) > 0:
                    mark_prices[symbol] = float(value)
            except (ccxt.BaseError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Mark price unavailable for {}; Binance liquidation simulation skipped this poll: {}", symbol, exc)

        for symbol in tracked:
            event_count_before = len(self.state.events)
            position_before = self.state.portfolio.positions.get(symbol)
            update_called = False
            symbol_changed = False
            symbol_regime_observed = False
            ticker = tickers.get(symbol)
            price = _current_market_price(ticker) or self.state.mark_prices.get(symbol)
            if price is None:
                logger.warning("No current public price for {}; paper monitoring skipped this cycle", symbol)
                continue
            self.state.mark_prices[symbol] = price
            view = self.state.for_symbol(symbol)
            shadow_active = False
            if self.trend_entry_shadow is not None:
                try:
                    shadow_active = symbol in self.trend_entry_shadow.active_symbols
                except Exception as exc:
                    self._disable_trend_entry_shadow(exc)
            trader = PaperTrader(view, self.settings, execution=self.execution)
            trader.margin_spec = self._margin_spec(symbol)
            watch = next((row for row in self.state.watchlist if row.get("symbol") == symbol), None)
            market_status = self.state.symbol_status.get(symbol, {})
            observation_context = {
                "hotlist_rank": watch.get("rank") if watch else None,
                "hotlist_score": watch.get("hot_score") if watch else None,
                "outside_hotlist": bool(watch and watch.get("held_outside_top")),
                "market_regime": market_status.get("regime"),
                "market_state_at": market_status.get("at"),
                "signal_action": market_status.get("action"),
                "signal_reason": market_status.get("reason"),
            }
            monitor_changed = trader.monitor_tick(
                current_price=price, mark_price=mark_prices.get(symbol), now=now,
                observation_context=observation_context,
            )
            changed = monitor_changed or changed
            if monitor_changed and self.checkpoint is not None:
                self.checkpoint()

            def check_hold_timer() -> bool:
                nonlocal changed
                timer_changed = trader.check_max_hold(price, now)
                changed = timer_changed or changed
                if timer_changed and self.checkpoint is not None:
                    self.checkpoint()
                return timer_changed

            if trader.margin_spec is None:
                self.state.symbol_status[symbol] = {"regime": "MARGIN_DATA_UNAVAILABLE", "action": "NONE",
                                                    "reason": "public_maintenance_margin_metadata_missing", "at": now.isoformat()}

            # Signals are evaluated only once the current hour begins, using completed bars.
            if (view.last_processed_close == hour_open_key and view.pending is None
                    and not shadow_active):
                check_hold_timer()
                continue
            try:
                raw_1h = priority_1h_rows.pop(symbol, None)
                if raw_1h is None:
                    raw_1h = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
                raw_4h = self.exchange.fetch_ohlcv(symbol, timeframe="4h", limit=300)
                closed_4h = [row for row in raw_4h if int(row[0]) + TIMEFRAME_MS["4h"] <= now_ms]
                frame_4h = ohlcv_frame(closed_4h)
                symbol_regime_state = None
                symbol_regime_evidence = None
                if symbol in eligible_symbols:
                    assessment = self._observe_symbol_regime(symbol, frame_4h, now)
                    symbol_regime_state = assessment.state.value
                    assessment_evidence = getattr(assessment, "evidence", None)
                    if assessment_evidence is not None:
                        last_closed_bar_at = getattr(assessment, "last_closed_bar_at", None)
                        symbol_regime_evidence = {
                            **assessment_evidence,
                            "assessment_reason": getattr(assessment, "reason", None),
                            "last_closed_bar_at": (last_closed_bar_at.isoformat()
                                                    if last_closed_bar_at else None),
                        }
                    symbol_regime_observed = True
                current = next((row for row in reversed(raw_1h or []) if int(row[0]) == hour_open_ms), None)
                if current is None:
                    logger.warning("No current 1H open bar for {}; will retry next poll", symbol)
                    check_hold_timer()
                    continue
                closed_1h = [row for row in raw_1h if int(row[0]) + HOUR_MS <= now_ms]
                frame_1h = ohlcv_frame(closed_1h)
                if len(frame_1h) < 48 or len(frame_4h) < self.settings.lookback.min_regime_bars_4h:
                    self.state.symbol_status[symbol] = {"regime": "WAITING_HISTORY", "action": "NONE",
                                                        "reason": "insufficient_closed_bars", "at": now.isoformat()}
                    changed = True
                    check_hold_timer()
                    continue
                try:
                    funding = self.exchange.fetch_funding_rate(symbol)
                    rate = float(funding["fundingRate"]) if funding and funding.get("fundingRate") is not None else None
                except (ccxt.BaseError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Funding rate unavailable for {}; new entries will fail closed: {}", symbol, exc)
                    rate = None
                process_now = now
                pullback_quote_fresh = True
                pullback_quote_metadata = None
                pending = view.pending
                trend_pullback_pending = (
                    pending is not None
                    and pending.get("execution_type") == "pullback"
                    and pending.get("source") == "TREND"
                )
                actual_pullback_flow = self.wait_pullback or trend_pullback_pending
                if self.trend_entry_shadow is not None:
                    try:
                        registered = self.trend_entry_shadow.register_pending(pending)
                        if registered:
                            self.trend_entry_shadow.save()
                    except Exception as exc:
                        self._disable_trend_entry_shadow(exc)
                shadow_needs_quote = False
                if self.trend_entry_shadow is not None:
                    try:
                        shadow_needs_quote = (shadow_active
                                              or self.trend_entry_shadow.needs_quote(symbol))
                    except Exception as exc:
                        self._disable_trend_entry_shadow(exc)
                shadow_quote_fresh = False
                shadow_quote_metadata = None
                if actual_pullback_flow or shadow_needs_quote:
                    if actual_pullback_flow:
                        process_now = datetime.now(timezone.utc)
                    if ((actual_pullback_flow and pending is not None
                         and pending.get("execution_type") == "pullback")
                            or shadow_needs_quote):
                        try:
                            quote_price, quote_fresh, quote_metadata = _fetch_pullback_quote(
                                self.exchange, symbol,
                            )
                            shadow_quote_fresh = quote_fresh
                            shadow_quote_metadata = quote_metadata
                            if (actual_pullback_flow and pending is not None
                                    and pending.get("execution_type") == "pullback"):
                                pullback_quote_fresh = quote_fresh
                                pullback_quote_metadata = {
                                    key: value for key, value in quote_metadata.items()
                                    if key not in {"quote_bid", "quote_ask"}
                                }
                                process_now = datetime.fromisoformat(quote_metadata["quote_received_at"])
                            if (actual_pullback_flow and pending is not None
                                    and pending.get("execution_type") == "pullback"
                                    and quote_fresh and quote_price is not None):
                                price = quote_price
                                self.state.mark_prices[symbol] = quote_price
                            elif not quote_fresh:
                                if actual_pullback_flow and pending is not None:
                                    logger.warning("PULLBACK_QUOTE_STALE symbol={} age_sec={} source={}; pending fill skipped",
                                                   symbol, quote_metadata.get("quote_age_sec"),
                                                   quote_metadata.get("quote_source"))
                                else:
                                    logger.warning("TREND_ENTRY_SHADOW_QUOTE_STALE symbol={} age_sec={} source={}",
                                                   symbol, quote_metadata.get("quote_age_sec"),
                                                   quote_metadata.get("quote_source"))
                        except Exception as exc:
                            if (actual_pullback_flow and pending is not None
                                    and pending.get("execution_type") == "pullback"):
                                pullback_quote_fresh = False
                            shadow_quote_fresh = False
                            quote_error_metadata = {
                                "quote_at": None,
                                "quote_received_at": datetime.now(timezone.utc).isoformat(),
                                "quote_age_sec": None,
                                "quote_price": None,
                                "quote_source": f"exchange.fetch_ticker.error:{type(exc).__name__}",
                            }
                            if (actual_pullback_flow and pending is not None
                                    and pending.get("execution_type") == "pullback"):
                                pullback_quote_metadata = quote_error_metadata
                            shadow_quote_metadata = quote_error_metadata
                            logger.warning("Fresh market quote unavailable for {}; {}",
                                           symbol, type(exc).__name__)
                step_size, minimum = _lot_filters(self.markets.get(symbol, {}), market_order=self.execution is not None)
                cutoff_raw = self.state.entry_after.get(symbol)
                cutoff = datetime.fromisoformat(cutoff_raw) if cutoff_raw else None
                gross_notional_cap_pct = (
                    0.0 if market_risk.state is MarketRiskState.PAUSE_NEW else None
                )
                update_called = True
                symbol_changed = trader.update(
                    frame_1h, frame_4h,
                    current_open_at=datetime.fromtimestamp(int(current[0]) / 1000, timezone.utc),
                    current_open=float(current[1]), current_price=price, funding_rate=rate, now=process_now,
                    allow_entries=self._entry_allowed(symbol),
                    symbol_in_hot_universe=symbol in eligible_symbols,
                    allowed_entry_sources=None,
                    entry_execution_allowed=self._entry_execution_allowed(),
                    step_size=step_size, minimum_quantity=minimum,
                    entry_not_before=cutoff, mark_price=mark_prices.get(symbol),
                    wait_pullback=self.wait_pullback,
                    mean_reversion_only=self.strategy_mode == "mean_reversion_only",
                    pullback_quote_fresh=pullback_quote_fresh,
                    pullback_quote_metadata=pullback_quote_metadata,
                    symbol_regime_state=symbol_regime_state,
                    symbol_regime_evidence=symbol_regime_evidence,
                    market_risk_state=market_risk.state.value,
                    gross_notional_cap_pct=gross_notional_cap_pct,
                    skip_structure_signal_close_key=(
                        hour_open_key if symbol in priority_structure_checked else None
                    ),
                )
                if self.trend_entry_shadow is not None:
                    try:
                        shadow_changed = self.trend_entry_shadow.register_pending(view.pending)
                        shadow_price = (
                            shadow_quote_metadata.get("quote_price")
                            if shadow_quote_fresh and shadow_quote_metadata else price
                        )
                        shadow_changed = self.trend_entry_shadow.observe_symbol(
                            symbol, now=process_now, current_price=shadow_price,
                            quote_fresh=shadow_quote_fresh, quote_metadata=shadow_quote_metadata,
                            bars_1h=frame_1h, market_risk_state=market_risk.state.value,
                        ) or shadow_changed
                        if shadow_changed:
                            self.trend_entry_shadow.save()
                    except Exception as exc:
                        self._disable_trend_entry_shadow(exc)
                changed = symbol_changed or changed
                check_hold_timer()
                if view.last_processed_close == hour_open_key:
                    self.state.entry_after.pop(symbol, None)
            except (ccxt.BaseError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Public market cycle failed for {}: {}", symbol, exc)
                if symbol in eligible_symbols and not symbol_regime_observed:
                    self._observe_symbol_regime(symbol, None, now,
                                                fetch_error=f"{type(exc).__name__}: {str(exc)[:300]}")
                self.state.symbol_status[symbol] = {"regime": "DATA_UNAVAILABLE", "action": "NONE",
                                                    "reason": type(exc).__name__, "at": now.isoformat()}
                changed = True
                check_hold_timer()
                symbol_changed = (update_called and (
                    len(self.state.events) != event_count_before
                    or self.state.portfolio.positions.get(symbol) != position_before
                ))
            if symbol_changed and self.checkpoint is not None:
                self.checkpoint()

        self._log_symbol_regime_summary()
        return changed
