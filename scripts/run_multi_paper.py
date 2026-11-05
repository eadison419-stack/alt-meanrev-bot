"""Run a portfolio-wide paper trader over Binance's hottest crypto perpetuals."""

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.exchange.binance_usdm import create_demo_usdm, resolve_execution_mode
from bot.exchange.network import configure_system_proxy
from bot.execution import DemoExecutionService
from bot.live.bollinger_outcomes import BollingerOutcomeTracker
from bot.live.manual_close import ManualCloseCommandStore, command_directory_for_state
from bot.live.multi_paper import MultiPaperState
from bot.live.multi_runner import MultiPaperRunner
from bot.live.trend_entry_shadow import TrendEntryShadowTracker


# Binance API failures that are transient (network-level) and safe to retry.
# In ccxt 4.x, NetworkError is the base of ExchangeNotAvailable, RateLimitExceeded,
# DDoSProtection, and RequestTimeout. Logical ExchangeError subclasses (auth,
# bad request, insufficient funds) are intentionally NOT retried: repeating them
# would just repeat a permanent mistake.
_RETRYABLE_API_ERRORS = (ccxt.NetworkError,)

# Binance IP bans surface as a 418 whose message contains "banned until <ms>".
_BAN_UNTIL_RE = re.compile(r"banned until\s+(\d+)")

_MAX_STARTUP_ATTEMPTS = 12
_MAX_RETRY_DELAY_SECONDS = 300.0


def _is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_API_ERRORS)


def _ban_remaining_seconds(exc: BaseException) -> float | None:
    match = _BAN_UNTIL_RE.search(str(exc))
    if not match:
        return None
    try:
        return float(match.group(1)) / 1000.0 - time.time()
    except (TypeError, ValueError):
        return None


def _retry_delay_seconds(exc: BaseException, attempt: int, poll: int) -> float:
    banned_for = _ban_remaining_seconds(exc)
    if banned_for is not None:
        return max(0.0, banned_for + 1.0)
    return min(_MAX_RETRY_DELAY_SECONDS, poll * (2 ** max(0, attempt - 1)))


def _run_startup_with_retry(fn, *, poll: int, label: str):
    """Run a startup step, retrying only transient network failures."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable_api_error(exc):
                raise
            attempt += 1
            if attempt >= _MAX_STARTUP_ATTEMPTS:
                logger.error("{} failed {} times on transient API errors; giving up: {}",
                             label, attempt, exc)
                raise
            delay = _retry_delay_seconds(exc, attempt, poll)
            logger.warning("{} transient API failure (attempt {}/{}); retrying in {:.1f}s: {}",
                           label, attempt, _MAX_STARTUP_ATTEMPTS, delay, type(exc).__name__)
            time.sleep(delay)


def _initialize_demo_execution(settings):
    """Create the Demo client and take a first snapshot, closing on failure."""
    demo_exchange = create_demo_usdm(settings.runtime)
    try:
        execution = DemoExecutionService(demo_exchange, settings)
        snapshot = execution.initialize()
        return demo_exchange, execution, snapshot
    except Exception:
        demo_exchange.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=60, help="market monitor polling interval in seconds")
    parser.add_argument("--top", type=int, default=20, help="maximum number of hot crypto contracts to scan (1-20)")
    parser.add_argument("--universe-refresh", type=int, default=300, help="refresh volume ranking interval in seconds")
    parser.add_argument("--fetch-workers", type=int, default=8,
                        help="parallel OHLCV fetch worker count during universe refresh (1 disables)")
    parser.add_argument("--equity", type=float, default=10_000.0, help="starting virtual USDT balance if no prior state exists")
    parser.add_argument("--state", type=Path, default=Path("data/paper/portfolio.json"), help="portfolio-wide JSON state")
    parser.add_argument("--strategy-mode", choices=("mean_reversion_only", "regime_routed"),
                        default="mean_reversion_only",
                        help="entry mode; defaults to the standard mean-reversion strategy")
    parser.add_argument("--legacy-state", type=Path, default=Path("data/paper/state.json"), help="read-only seed from prior ADA paper state")
    parser.add_argument("--once", action="store_true", help="run one public-data cycle and exit")
    parser.add_argument("--live", action="store_true", help="deprecated safety flag; all live trading is refused")
    parser.add_argument("--sync-exchange", action="store_true",
                        help="explicitly discard stale local-only positions when the Demo account is flat")
    parser.add_argument("--sync-only", action="store_true",
                        help="with --sync-exchange, save the flat Demo snapshot and exit before strategy evaluation")
    args = parser.parse_args()
    if (args.poll <= 0 or args.universe_refresh <= 0 or args.equity <= 0
            or not 1 <= args.top <= 20 or not 1 <= args.fetch_workers <= 16):
        parser.error("poll, universe refresh, and equity must be positive; "
                     "--top must be 1-20; --fetch-workers must be 1-16")
    if args.sync_only and not args.sync_exchange:
        parser.error("--sync-only requires --sync-exchange")

    def path(value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    state_path, legacy_path = path(args.state), path(args.legacy_state)
    settings = load_settings()
    try:
        mode = resolve_execution_mode(bot_mode=settings.runtime.bot_mode,
                                      testnet=settings.runtime.binance_testnet, live_flag=args.live)
    except RuntimeError as exc:
        logger.error("Execution mode refused: {}", exc)
        return 2
    if args.sync_exchange and mode.name != "demo_usdm":
        parser.error("--sync-exchange is available only in paper + BINANCE_TESTNET=true mode")
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    demo_exchange = None
    try:
        state = MultiPaperState.load(state_path, initial_equity=args.equity,
                                     legacy_path=legacy_path)
        shadow_path = state_path.with_name(f"{state_path.stem}.trend-shadow.json")
        try:
            trend_entry_shadow = TrendEntryShadowTracker(shadow_path, settings)
            logger.info("Trend entry shadow enabled: path={} guards_atr={}",
                        shadow_path, trend_entry_shadow.state["guards_atr"])
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Trend entry shadow unavailable after {}; official strategy continues unchanged",
                         type(exc).__name__)
            trend_entry_shadow = None
        outcome_path = state_path.with_name(f"{state_path.stem}.bollinger-outcomes.json")
        try:
            bollinger_outcome_tracker = BollingerOutcomeTracker(outcome_path, settings)
            logger.info("Bollinger outcome tracking enabled: path={} horizons_hours={}",
                        outcome_path, bollinger_outcome_tracker.state["horizons_hours"])
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Bollinger outcome tracking unavailable after {}; official strategy continues unchanged",
                         type(exc).__name__)
            bollinger_outcome_tracker = None
        prior_demo_snapshot = state.exchange_snapshot if state.execution_mode == "demo_usdm" else None
        state.execution_mode = mode.name
        execution = None
        if mode.name == "demo_usdm":
            demo_exchange, execution, snapshot = _run_startup_with_retry(
                lambda: _initialize_demo_execution(settings),
                poll=args.poll, label="Demo execution initialization",
            )
            state.exchange_snapshot = snapshot
            if prior_demo_snapshot is None:
                state.initial_equity = float(snapshot["equity"])
            if args.sync_exchange:
                if snapshot["positions"]:
                    raise RuntimeError("Demo account has open positions; explicit sync is refused without matching saved Demo order metadata")
                stale_symbols = list(state.portfolio.positions)
                state.portfolio.positions.clear()
                state.entry_costs.clear()
                state.isolated_margin_by_symbol.clear()
                state.exit_management_by_symbol.clear()
                state.exchange_orders_by_symbol.clear()
                state.pending_by_symbol.clear()
                state.mark_prices.clear()
                state.execution_status = "synchronized"
                state.inconsistencies = []
                state.events.append({"event": "exchange_state_synchronized", "at": snapshot["at"],
                                     "host": snapshot["host"], "discarded_local_only_positions": stale_symbols})
                state.cash = float(snapshot["available_usdt"])
                if args.sync_only:
                    state.save(state_path)
                    logger.info("Demo state synchronized to {}; strategy runner not started", state_path)
                    return 0
        runner = _run_startup_with_retry(
            lambda: MultiPaperRunner(
                exchange, settings, state, top_n=args.top,
                universe_refresh_seconds=args.universe_refresh,
                fetch_workers=args.fetch_workers,
                manual_close_store=ManualCloseCommandStore(command_directory_for_state(state_path)),
                execution=execution,
                trend_entry_shadow=trend_entry_shadow,
                bollinger_outcome_tracker=bollinger_outcome_tracker,
                checkpoint=lambda: state.save(state_path),
                strategy_mode=args.strategy_mode,
            ),
            poll=args.poll, label="Runner initialization",
        )
        logger.info("Starting multi-symbol Paper: mode={} testnet={} host={} strategy_mode={} top={} max_positions={}",
                    mode.name, mode.testnet, execution.host if execution else "local-only",
                    args.strategy_mode, args.top, settings.sizing.max_symbols)
        logger.info("Entry timing mode=immediate")
        last_boundary_probe = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        while True:
            cycle_started = datetime.now(timezone.utc)
            try:
                changed = runner.cycle()
            except ccxt.BaseError as exc:
                if args.once:
                    return 1
                delay = _retry_delay_seconds(exc, 1, args.poll)
                logger.warning("Binance public market data request failed; will retry in {:.1f}s: {}", delay, exc)
                time.sleep(delay)
                continue
            # Save each polling cycle so live marks, hot ranking, and drawdown state survive restart.
            state.save(state_path)
            logger.info("Paper portfolio equity={}, open_positions={}, watched={}, changed={}",
                        round(state.current_equity, 4), len(state.portfolio.positions),
                        len(state.watchlist), changed)
            if args.once:
                break
            after_cycle = datetime.now(timezone.utc)
            boundary = after_cycle.replace(minute=0, second=0, microsecond=0)
            if boundary > last_boundary_probe:
                probe_at = boundary + timedelta(seconds=5)
                if cycle_started >= probe_at:
                    last_boundary_probe = boundary
                    next_delay = args.poll
                else:
                    next_delay = max(0.5, (probe_at - after_cycle).total_seconds())
            else:
                next_boundary = last_boundary_probe + timedelta(hours=1)
                probe_at = next_boundary + timedelta(seconds=5)
                next_delay = min(args.poll, max(0.5, (probe_at - after_cycle).total_seconds()))
            time.sleep(next_delay)
    except ccxt.BaseError as exc:
        logger.error("Binance public market data request failed: {}", exc)
        return 1
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Multi-symbol paper cycle failed safely: {}", exc)
        return 1
    finally:
        exchange.close()
        if demo_exchange is not None:
            demo_exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
