"""Run the local paper trader against public Binance USDT-M market data."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.klines import TIMEFRAME_MS, normalize_symbol, ohlcv_frame, resolve_symbols
from bot.exchange.network import configure_system_proxy
from bot.live.paper import PaperAccount, PaperTrader
from bot.market_risk import MarketRiskState, fetch_market_risk_status


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cycle(exchange, trader: PaperTrader, symbol: str) -> bool:
    now_ms = exchange.milliseconds()
    now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    bars: dict[str, list[list[float]]] = {}
    for timeframe in ("1h", "4h"):
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
        if not rows:
            raise RuntimeError(f"Binance returned no {timeframe} candles for {symbol}")
        bars[timeframe] = rows
    rows_1h = bars["1h"]
    current_hour_ms = now_ms // TIMEFRAME_MS["1h"] * TIMEFRAME_MS["1h"]
    current = next((row for row in reversed(rows_1h) if int(row[0]) == current_hour_ms), None)
    if current is None:
        raise RuntimeError("latest public 1H candle is unavailable; refusing stale paper fills")
    closed_1h = [row for row in rows_1h if int(row[0]) + TIMEFRAME_MS["1h"] <= now_ms]
    closed_4h = [row for row in bars["4h"] if int(row[0]) + TIMEFRAME_MS["4h"] <= now_ms]
    frame_1h, frame_4h = ohlcv_frame(closed_1h), ohlcv_frame(closed_4h)
    if len(frame_1h) < 48 or len(frame_4h) < 80:
        logger.info("Waiting for sufficient closed candle history: 1H={}, 4H={}", len(frame_1h), len(frame_4h))
        return False
    ticker = exchange.fetch_ticker(symbol)
    price = ticker.get("last") or ticker.get("close")
    if price is None:
        raise RuntimeError("public ticker has no last price; stop monitoring is unavailable")
    mark_price = None
    if trader.account.position is not None:
        try:
            mark = exchange.fetch_mark_price(symbol)
            value = mark.get("markPrice") or mark.get("mark")
            mark_price = float(value) if value is not None else None
        except (ccxt.BaseError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Mark price unavailable; Binance-style liquidation trigger skipped this poll: {}", exc)
    try:
        funding = exchange.fetch_funding_rate(symbol)
        rate = float(funding["fundingRate"]) if funding and funding.get("fundingRate") is not None else None
    except (ccxt.BaseError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Funding rate unavailable; new entries will fail closed: {}", exc)
        rate = None
    assessment = fetch_market_risk_status(exchange, trader.settings, now=now)
    status = assessment.to_dict()
    prior_status = getattr(trader, "last_market_risk_status", None)
    risk_changed = prior_status != (assessment.state.value, assessment.reason)
    if risk_changed:
        trader._event("market_risk_observed", **status)
        trader.last_market_risk_status = (assessment.state.value, assessment.reason)
    level = "warning" if assessment.state is MarketRiskState.PAUSE_NEW else "info"
    getattr(logger, level)("Market risk observation {}",
                           json.dumps(status, ensure_ascii=False, sort_keys=True))
    changed = trader.update(
        frame_1h, frame_4h, current_open_at=datetime.fromtimestamp(int(current[0]) / 1000, timezone.utc),
        current_open=float(current[1]), current_price=float(price), funding_rate=rate, now=now,
        mark_price=mark_price,
    )
    return trader.check_max_hold(float(price), now) or changed or risk_changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ADAUSDT", help="one USDT-M swap symbol, e.g. ADAUSDT")
    parser.add_argument("--poll", type=int, default=60, help="public market polling interval in seconds")
    parser.add_argument("--equity", type=float, default=10_000.0, help="starting virtual USDT balance for a new state")
    parser.add_argument("--state", type=Path, default=Path("data/paper/state.json"), help="local JSON paper state")
    parser.add_argument("--once", action="store_true", help="run one public-data cycle and exit")
    args = parser.parse_args()
    if args.poll <= 0 or args.equity <= 0:
        parser.error("--poll and --equity must be positive")

    settings = load_settings()
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    try:
        symbols = resolve_symbols(exchange, args.symbol, settings)
        if len(symbols) != 1 or normalize_symbol(symbols[0]) != normalize_symbol(args.symbol):
            parser.error("paper mode currently accepts exactly one explicitly named symbol")
        state_path = _project_path(args.state)
        account = PaperAccount.load(state_path, symbol=symbols[0], initial_equity=args.equity)
        trader = PaperTrader(account, settings)
        market_info = exchange.market(symbols[0]).get("info", {})
        try:
            trader.margin_spec = {
                "maintenance_margin_rate": float(market_info["maintMarginPercent"]) / 100,
                "maintenance_amount": 0.0,
                "liquidation_fee_rate": float(market_info["liquidationFee"]),
            }
        except (KeyError, TypeError, ValueError):
            trader.margin_spec = {}
        if account.position is not None and account.isolated_margin is None and trader.margin_spec:
            leverage = float(settings.sizing.leverage)
            allocated = account.position.notional / leverage
            if allocated <= account.cash:
                account.cash -= allocated
                account.isolated_margin = {**trader.margin_spec, "initial_margin": allocated, "leverage": leverage}
                account.events.append({"event": "isolated_margin_migrated", "symbol": symbols[0],
                                       "at": datetime.now(timezone.utc).isoformat(), "initial_margin": allocated})
                account.save(state_path)
                logger.info("Migrated existing paper position to isolated margin {:.4f} USDT; position and stop unchanged", allocated)
        logger.info("Starting local paper mode for {} with virtual equity {}; public market data only", symbols[0], account.initial_equity)
        while True:
            changed = run_cycle(exchange, trader, symbols[0])
            if changed:
                account.save(state_path)
                logger.info("Paper state saved: cash={}, open_position={}, events={}", round(account.cash, 4), account.position is not None, len(account.events))
            if args.once:
                break
            time.sleep(args.poll)
    except ccxt.BaseError as exc:
        logger.error("Binance public market data request failed: {}", exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        logger.error("Paper cycle failed safely: {}", exc)
        return 1
    finally:
        exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
