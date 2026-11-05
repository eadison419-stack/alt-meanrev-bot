"""Download public Binance USDT-M funding settlements into a local CSV cache."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
from loguru import logger

from bot.config import PROJECT_ROOT, load_settings
from bot.data.funding import fetch_funding_rates, upsert_funding_csv
from bot.data.klines import filter_tradable_crypto_universe, normalize_symbol, resolve_symbols
from bot.exchange.network import configure_system_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="One explicit USDT-M perpetual symbol")
    parser.add_argument("--symbols", help="HOT for all current eligible crypto perpetuals, or comma-separated symbols")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")

    settings = load_settings()
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    try:
        if args.symbol and args.symbols:
            parser.error("use either --symbol or --symbols")
        requested = args.symbols or args.symbol or "ADAUSDT"
        if requested.strip().upper() == "HOT":
            markets, tickers = exchange.load_markets(), exchange.fetch_tickers()
            symbols = [symbol for symbol, _ in filter_tradable_crypto_universe(markets, tickers, settings)]
        else:
            symbols = resolve_symbols(exchange, requested, settings)
        now_ms = exchange.milliseconds()
        since_ms = int((datetime.fromtimestamp(now_ms / 1000, timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
        logger.info("Downloading funding history for {} symbols", len(symbols))
        for symbol in symbols:
            if args.output and len(symbols) > 1:
                parser.error("--output is only supported for one symbol")
            output = args.output or Path(f"data/funding_{normalize_symbol(symbol)}.csv")
            output_path = output if output.is_absolute() else PROJECT_ROOT / output
            rows = fetch_funding_rates(exchange, symbol, since_ms=since_ms, until_ms=now_ms)
            total = upsert_funding_csv(output_path, rows)
            logger.info("Cached {} funding settlements for {} in {}", total, symbol, output_path)
    except ccxt.BaseError as exc:
        logger.error("Binance public funding data request failed: {}", exc)
        return 1
    finally:
        exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
