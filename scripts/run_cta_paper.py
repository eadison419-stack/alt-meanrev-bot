"""Run the Donchian breakout long-only strategy (bot.trend_follow) on paper/demo.

Independent entry point for the new CTA breakout strategy. Two modes, same as
run_multi_paper.py:
  BINANCE_TESTNET=false -> local_paper  (local virtual fills, no orders, no key)
  BINANCE_TESTNET=true  -> demo_usdm   (Binance USD-M testnet, needs API key)

Strategy (validated in scripts/diagnose_cta_proto.py): enter long when a closed
4H bar closes above the prior N-bar high; exit on a reverse M-bar low break or a
2x ATR protective stop. Long-only. The protective stop is placed on the exchange
in demo mode and tracked locally in local_paper mode.

Run (Windows, from project root):
  python scripts/run_cta_paper.py --once          # local paper (default), one cycle
  python scripts/run_cta_paper.py                 # local paper, continuous

  python scripts/run_cta_paper.py --demo          # Binance testnet demo (needs API key env)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import ohlcv_frame
from bot.exchange.binance_usdm import create_demo_usdm, resolve_execution_mode
from bot.exchange.network import configure_system_proxy
from bot.execution import DemoExecutionService
from bot.live.multi_paper import MultiPaperState
from bot.portfolio import Position
from bot.regime import Regime
from bot.risk import OrderIntent
from bot.signals import Action, Side
from bot.strategy_router import TREND
from bot.trend_follow import (BreakoutConfig, breadth_ok, breakeven_level,
                              indicators, long_entry, long_exit,
                              market_breadth, position_size)

FOUR_H_MS = 4 * 3600 * 1000
RISK_PCT = 0.01
DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "1000PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]


def _closed_4h(exchange, symbols, now_ms):
    frames = {}
    for s in symbols:
        try:
            rows = exchange.fetch_ohlcv(s, "4h", limit=60)
        except ccxt.BaseError as exc:
            logger.warning("4H fetch failed for {}: {}", s, type(exc).__name__)
            continue
        closed = [r for r in rows if r[0] + FOUR_H_MS <= now_ms]
        if len(closed) >= 30:
            frames[s] = ohlcv_frame(closed)
    return frames


def _local_open(state, symbol, price, stop, qty, leverage, fee_rate, slip_rate, reason, at):
    entry_fill = price * (1 + slip_rate)
    margin = qty * entry_fill / leverage
    entry_fee = qty * entry_fill * fee_rate
    if state.cash < margin + entry_fee:
        return None
    state.cash -= margin + entry_fee
    state.isolated_margin_by_symbol[symbol] = {"initial_margin": margin, "leverage": leverage}
    state.entry_costs[symbol] = {"entry_fee": entry_fee}
    position = Position(symbol, Side.LONG, qty, entry_fill, stop, at, reason, "TREND", TREND)
    state.portfolio.positions[symbol] = position
    state.mark_prices[symbol] = entry_fill
    state.events.append({"event": "paper_fill", "action": "OPEN", "symbol": symbol,
                         "side": "long", "at": at.isoformat(), "quantity": qty,
                         "price": entry_fill, "reason": reason, "source": TREND})
    return position


def _local_close(state, symbol, price, reason, at, fee_rate, slip_rate):
    position = state.portfolio.positions.get(symbol)
    if position is None:
        return None
    exit_fill = price * (1 - slip_rate)
    gross = position.quantity * (exit_fill - position.entry_price)
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
                         "side": "long", "at": at.isoformat(), "quantity": position.quantity,
                         "price": exit_fill, "reason": reason, "net_pnl": net, "source": TREND})
    return net


def _demo_open(execution, state, symbol, price, stop, qty, reason, at):
    intent = OrderIntent(symbol, Side.LONG, Action.OPEN, qty, price, reason, Regime.MID, stop, TREND)
    result = execution.open(intent, stop_price=stop)
    state.exchange_orders_by_symbol[symbol] = {"stop": result["stop"]}
    pos = result["position"]
    position = Position(symbol, Side.LONG, float(pos["quantity"]), float(pos["entry_price"]),
                        float(result["stop"]["stop_price"]), at, reason, "TREND", TREND)
    state.portfolio.positions[symbol] = position
    state.events.append({"event": "paper_fill", "action": "OPEN", "symbol": symbol,
                         "side": "long", "at": at.isoformat(), "quantity": position.quantity,
                         "price": position.entry_price, "reason": reason, "source": TREND,
                         "execution_source": "binance_usdm_demo"})
    return position


def _demo_close(execution, state, symbol, reason, at):
    position = state.portfolio.positions.get(symbol)
    if position is None:
        return
    stop = state.exchange_orders_by_symbol.get(symbol, {}).get("stop")
    execution.close(position, stop=stop, reason=reason)
    state.portfolio.positions.pop(symbol, None)
    state.exchange_orders_by_symbol.pop(symbol, None)
    state.exit_management_by_symbol.pop(symbol, None)
    state.events.append({"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                         "side": "long", "at": at.isoformat(), "quantity": position.quantity,
                         "reason": reason, "source": TREND, "execution_source": "binance_usdm_demo"})


def _demo_sync_stopped(execution, state, at):
    """Detect positions the exchange protective stop already closed."""
    snapshot = execution.sync_snapshot()
    state.exchange_snapshot = snapshot
    live = set(snapshot["positions"])
    for symbol in list(state.portfolio.positions):
        if symbol not in live:
            state.portfolio.positions.pop(symbol, None)
            state.exchange_orders_by_symbol.pop(symbol, None)
            state.exit_management_by_symbol.pop(symbol, None)
            state.events.append({"event": "paper_fill", "action": "CLOSE", "symbol": symbol,
                                 "side": "long", "at": at.isoformat(), "reason": "protective_stop",
                                 "source": TREND, "execution_source": "binance_usdm_demo"})


def _cycle(exchange, state, execution, settings, symbols, cfg, max_pos, is_demo, now):
    now_ms = int(now.timestamp() * 1000)
    fee_rate = settings.fees.taker_bps / 10_000
    slip_rate = settings.fees.slippage_bps / 10_000
    leverage = int(settings.sizing.leverage)
    frames = _closed_4h(exchange, symbols, now_ms)
    breadth_val = market_breadth(frames)

    if is_demo:
        _demo_sync_stopped(execution, state, now)

    # exits (reverse breakout; local stop also closed here)
    for s in list(state.portfolio.positions):
        frame = frames.get(s)
        if frame is None or len(frame) < 2:
            continue
        position = state.portfolio.positions[s]
        meta = state.exit_management_by_symbol.get(s)
        stop = position.stop_price
        if is_demo:
            stop = state.exchange_orders_by_symbol.get(s, {}).get("stop", {}).get("stop_price", stop)
        elif cfg.be_mult is not None and meta is not None:
            level = breakeven_level(float(position.entry_price), float(meta["entry_atr"]),
                                    float(meta["highest_high"]), cfg.be_mult)
            if level is not None:
                stop = max(float(stop), level)
        triggered, reason = long_exit(frame, cfg, float(stop))
        if triggered:
            if reason == "stop" and not is_demo:
                _local_close(state, s, float(frame["low"].iloc[-1]), "protective_stop", now, fee_rate, slip_rate)
            elif reason == "reverse":
                if is_demo:
                    _demo_close(execution, state, s, "reverse_breakout", now)
                else:
                    _local_close(state, s, float(frame["close"].iloc[-1]), "reverse_breakout", now, fee_rate, slip_rate)
            continue
        # advance the breakeven tracker from this closed bar (applies from the next bar)
        if cfg.be_mult is not None and meta is not None:
            high = float(frame["high"].iloc[-1])
            meta["highest_high"] = max(float(meta["highest_high"]), high)
            if is_demo:
                level = breakeven_level(float(position.entry_price), float(meta["entry_atr"]),
                                        float(meta["highest_high"]), cfg.be_mult)
                if level is not None:
                    old_stop = state.exchange_orders_by_symbol.get(s, {}).get("stop")
                    old_price = float((old_stop or {}).get("stop_price", 0.0) or 0.0)
                    if old_stop and 0.0 < old_price < level and float(frame["close"].iloc[-1]) > level:
                        try:
                            new_stop = execution.replace_stop(position, stop=old_stop, stop_price=level)
                            state.exchange_orders_by_symbol[s]["stop"] = new_stop
                        except Exception as exc:
                            logger.warning("breakeven stop move failed for {}: {}", s, type(exc).__name__)

    # entries (long breakout, ranked by breakout strength, capped by max_pos)
    slot = state.current_equity / max_pos
    if breadth_ok(breadth_val, cfg.breadth_min):
        cands = []
        for s in symbols:
            if s in state.portfolio.positions:
                continue
            frame = frames.get(s)
            if frame is None:
                continue
            triggered, stop = long_entry(frame, cfg)
            if not triggered:
                continue
            ind = indicators(frame, cfg)
            i = len(frame) - 1
            atr_val = float(ind.atr[i])
            close = float(frame["close"].iloc[-1])
            upper = float(ind.upper[i])
            if atr_val <= 0:
                continue
            strength = (close - upper) / atr_val
            cands.append((s, strength, close, stop, atr_val))
        cands.sort(key=lambda x: -x[1])
        for s, _strength, close, stop, atr_val in cands[: max(0, max_pos - len(state.portfolio.positions))]:
            qty = position_size(slot, RISK_PCT, cfg.atr_mult, atr_val)
            if qty <= 0:
                continue
            if is_demo:
                opened = _demo_open(execution, state, s, close, stop, qty, "donchian_breakout", now)
            else:
                opened = _local_open(state, s, close, stop, qty, leverage, fee_rate, slip_rate, "donchian_breakout", now)
            if opened is not None and cfg.be_mult is not None:
                state.exit_management_by_symbol[s] = {"entry_atr": atr_val, "highest_high": close}
    elif cfg.breadth_min is not None:
        logger.info("breadth gate blocks entries: breadth={} < {}", round(breadth_val, 3) if breadth_val is not None else None, cfg.breadth_min)

    # refresh marks
    for s in list(state.portfolio.positions):
        frame = frames.get(s)
        if frame is not None:
            state.mark_prices[s] = float(frame["close"].iloc[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=300, help="poll interval in seconds")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--state", type=Path, default=Path("data/paper/cta_portfolio.json"))
    parser.add_argument("--demo", action="store_true",
                        help="run on Binance USDT-M testnet demo (default: local paper, no key)")
    parser.add_argument("--be-mult", type=float, default=None,
                        help="breakeven stop: raise the long stop to entry after price moves "
                             "+N*ATR(entry) in our favour (default: off)")
    parser.add_argument("--breadth-min", type=float, default=None,
                        help="market breadth gate: only open new longs when the fraction of "
                             "symbols closing above their 4H EMA50 is at least this value "
                             "(default: off)")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    def path(value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    state_path = path(args.state)
    settings = load_settings()

    # Background-friendly logging: write to a file so the runner can run under
    # pythonw.exe (no console) and still leave a readable trail.
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(log_dir / "cta_paper.log", rotation="10 MB", retention=7,
               enqueue=True, level="INFO")
    pid_path = PROJECT_ROOT / "data" / "paper" / "cta_paper.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    mode = resolve_execution_mode(bot_mode=settings.runtime.bot_mode,
                                  testnet=args.demo, live_flag=False)
    is_demo = mode.name == "demo_usdm"

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    demo_exchange = None
    execution = None
    try:
        state = MultiPaperState.load(state_path, initial_equity=args.equity)
        state.execution_mode = mode.name
        if is_demo:
            demo_exchange = create_demo_usdm(settings.runtime)
            execution = DemoExecutionService(demo_exchange, settings)
            state.exchange_snapshot = execution.initialize()
            state.initial_equity = float(state.exchange_snapshot["equity"])

        cfg = BreakoutConfig(be_mult=args.be_mult, breadth_min=args.breadth_min)
        logger.info("Starting CTA breakout paper: mode={} symbols={} max_positions={} be_mult={} breadth_min={}",
                    mode.name, len(symbols), args.max_positions, args.be_mult, args.breadth_min)
        while True:
            now = datetime.now(timezone.utc)
            try:
                _cycle(exchange, state, execution, settings, symbols, cfg, args.max_positions,
                       is_demo, now)
            except ccxt.BaseError as exc:
                logger.warning("public market data failed; will retry: {}", type(exc).__name__)
            state.record_equity(now)
            state.save(state_path)
            logger.info("CTA paper equity={} open_positions={}",
                        round(state.current_equity, 4), len(state.portfolio.positions))
            if args.once:
                break
            time.sleep(args.poll)
    finally:
        exchange.close()
        if demo_exchange is not None:
            demo_exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
