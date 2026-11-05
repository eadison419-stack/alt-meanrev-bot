"""Run the legacy LOW/HIGH mean-reversion strategy in an isolated local Paper account.

The runner uses the hot Top-20 universe, the standard closed-bar LOW/HIGH entry
signals, BTC market-risk gate, shared portfolio risk controls, and EMA20 mid-band
rejection exit. It creates no private exchange client and sends no orders.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import ohlcv_frame
from bot.exchange.network import configure_system_proxy
from bot.live.multi_paper import MultiPaperState
from bot.live.multi_runner import MultiPaperRunner
from bot.live.paper import PaperTrader
from bot.mr_candidate import MidExitState, mid_band, mid_exit_check
from bot.strategy_router import MEAN_REVERSION

HOUR_MS = 60 * 60 * 1000
MID_PERIOD = 20
STRATEGY_VARIANT = "legacy_low_high_ema20_mid_exit"
DEFAULT_STATE = Path("data/paper/legacy_low_high_portfolio.json")
DEFAULT_LOG = Path("logs/legacy_low_high_paper.log")


def _closed_one_hour(exchange, symbol: str, now_ms: int):
    rows = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
    closed = [row for row in rows if int(row[0]) + HOUR_MS <= now_ms]
    return ohlcv_frame(closed) if closed else None


def _apply_ema20_mid_exit(exchange, state: MultiPaperState, settings,
                          now: datetime) -> bool:
    """Apply the legacy backtest's EMA20 mid-band rejection exit to open MR legs."""
    changed = False
    now_ms = int(now.timestamp() * 1000)
    for symbol, position in list(state.portfolio.positions.items()):
        if position.source != MEAN_REVERSION:
            continue
        try:
            frame = _closed_one_hour(exchange, symbol, now_ms)
        except ccxt.BaseError as exc:
            logger.warning("EMA20 exit data unavailable for {}: {}", symbol, type(exc).__name__)
            continue
        if frame is None or len(frame) < MID_PERIOD:
            continue

        bar_open = frame.index[-1]
        bar_close = bar_open + timedelta(hours=1)
        if bar_close <= position.opened_at:
            continue
        meta = state.exit_management_by_symbol.setdefault(symbol, {})
        bar_key = bar_open.isoformat()
        if meta.get("legacy_mid_last_bar") == bar_key:
            continue

        current_close = float(frame["close"].iloc[-1])
        mid = float(mid_band(frame["close"], MID_PERIOD, True).iloc[-1])
        if not math.isfinite(mid):
            continue
        mid_state = MidExitState(
            reached_mid=bool(meta.get("legacy_mid_reached", False)),
            hold_count=0,
            fired=False,
        )
        fire, mid_state = mid_exit_check(
            position.side, float(position.entry_price), mid, current_close,
            buffer=0.0, hold_bars=0, state=mid_state,
        )
        meta["legacy_mid_reached"] = mid_state.reached_mid
        meta["legacy_mid_last_bar"] = bar_key
        if not fire:
            continue

        trader = PaperTrader(state.for_symbol(symbol), settings)
        if trader._close(current_close, bar_close.to_pydatetime(), "bollinger_mid_rejection"):
            state.exit_management_by_symbol.pop(symbol, None)
            logger.info("EMA20 mid-band exit {} side={} close={} mid={} bar={}",
                        symbol, position.side.value, current_close, mid, bar_key)
            changed = True
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=60, help="market polling interval in seconds")
    parser.add_argument("--top", type=int, default=20, help="hot-universe size (1-20)")
    parser.add_argument("--universe-refresh", type=int, default=300,
                        help="hot-universe refresh interval in seconds")
    parser.add_argument("--fetch-workers", type=int, default=8,
                        help="parallel OHLCV fetch workers (1 disables parallel fetch)")
    parser.add_argument("--equity", type=float, default=5_000.0,
                        help="starting virtual USDT balance when the isolated state is new")
    parser.add_argument("--once", action="store_true", help="run one market cycle and exit")
    args = parser.parse_args()
    if (args.poll <= 0 or args.universe_refresh <= 0 or args.equity <= 0
            or not 1 <= args.top <= 20 or not 1 <= args.fetch_workers <= 16):
        parser.error("poll, universe refresh, and equity must be positive; top must be 1-20; workers 1-16")

    settings = load_settings()
    state_path = PROJECT_ROOT / DEFAULT_STATE
    log_path = PROJECT_ROOT / DEFAULT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, rotation="10 MB", retention="30 days", encoding="utf-8")

    state = MultiPaperState.load(state_path, initial_equity=args.equity)
    if state.execution_mode != "local_paper":
        raise RuntimeError(f"isolated Paper state has unexpected execution mode: {state.execution_mode}")
    if (state.events or state.portfolio.positions or state.last_processed_by_symbol) \
            and state.strategy_variant != STRATEGY_VARIANT:
        raise RuntimeError(f"isolated state belongs to another strategy: {state.strategy_variant}")
    if not settings.market_risk.enabled:
        raise RuntimeError("BTC market-risk gate is disabled in configs/default.yaml")
    state.strategy_variant = STRATEGY_VARIANT
    state.execution_status = "local"

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    try:
        runner = MultiPaperRunner(
            exchange, settings, state,
            top_n=args.top,
            universe_refresh_seconds=args.universe_refresh,
            fetch_workers=args.fetch_workers,
            checkpoint=lambda: state.save(state_path),
            strategy_mode="mean_reversion_only",
        )
        logger.info(
            "Starting isolated legacy MR Paper: state={} initial_equity={} top={} max_positions={} "
            "risk_per_trade={} BTC_gate=enabled EMA20_mid_exit=enabled",
            state_path, state.initial_equity, args.top, settings.sizing.max_symbols,
            settings.risk.max_single_trade_risk_pct,
        )
        while True:
            try:
                runner.cycle()
                now = datetime.now(timezone.utc)
                _apply_ema20_mid_exit(exchange, state, settings, now)
                state.save(state_path)
                logger.info("Isolated Paper equity={} open_positions={} watched={}",
                            round(state.current_equity, 4), len(state.portfolio.positions),
                            len(state.watchlist))
            except ccxt.BaseError as exc:
                logger.warning("Public market data request failed: {}", type(exc).__name__)
                if args.once:
                    return 1
            except (RuntimeError, ValueError, OSError) as exc:
                logger.error("Legacy MR Paper cycle stopped safely: {}", exc)
                return 1
            if args.once:
                return 0
            time.sleep(args.poll)
    finally:
        exchange.close()


if __name__ == "__main__":
    raise SystemExit(main())
