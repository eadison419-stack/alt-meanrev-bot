"""Download public Binance USDT-M OHLCV into the local SQLite cache."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.cache import KlineCache
from bot.data.klines import fetch_closed_ohlcv, filter_tradable_crypto_universe, resolve_symbols
from bot.exchange.network import configure_system_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="AUTO", help="AUTO, HOT (all current eligible crypto perpetuals), or comma-separated USDT-M swaps")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--db", type=Path, default=Path("data/market.sqlite"))
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")

    settings = load_settings()
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    try:
        if args.symbols.strip().upper() == "HOT":
            markets, tickers = exchange.load_markets(), exchange.fetch_tickers()
            symbols = [symbol for symbol, _ in filter_tradable_crypto_universe(markets, tickers, settings)]
        else:
            symbols = resolve_symbols(exchange, args.symbols, settings)
        logger.info("Downloading {} eligible symbols for rolling hot-universe research", len(symbols))
        now_ms = exchange.milliseconds()
        since_ms = int((datetime.fromtimestamp(now_ms / 1000, timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
        db_path = args.db if args.db.is_absolute() else PROJECT_ROOT / args.db
        cache = KlineCache(db_path)
        for symbol in symbols:
            for timeframe in (settings.timeframes.exec, settings.timeframes.regime):
                rows = fetch_closed_ohlcv(exchange, symbol, timeframe, since_ms=since_ms, until_ms=now_ms)
                cache.upsert(symbol, timeframe, rows)
                logger.info("Cached {} {} bars for {} in {}", len(rows), timeframe, symbol, db_path)
    except ccxt.BaseError as exc:
        logger.error("Binance public market data request failed: {}", exc)
        return 1
    finally:
        exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
