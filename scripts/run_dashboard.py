"""Serve the paper monitoring dashboard on localhost only."""

import argparse
import json
from pathlib import Path
import webbrowser

import ccxt

from bot.config import PROJECT_ROOT
from bot.exchange.network import configure_system_proxy
from bot.live.dashboard import serve_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="symbol matching a legacy single-symbol paper state")
    parser.add_argument("--state", type=Path, default=Path("data/paper/portfolio.json"), help="paper JSON state file")
    parser.add_argument("--port", type=int, default=8765, help="localhost dashboard port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    state_path = args.state if args.state.is_absolute() else PROJECT_ROOT / args.state
    if not state_path.is_file():
        parser.error(f"paper state not found: {state_path}; start the paper runner first")
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    symbol = None
    if state_data.get("version") == 1:
        requested = args.symbol or state_data.get("symbol")
        if not requested:
            parser.error("--symbol is required for a legacy single-symbol paper state")
    elif args.symbol:
        parser.error("--symbol cannot be used with a portfolio-wide state")
    else:
        requested = None

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    server = None
    try:
        if requested:
            markets = exchange.load_markets()
            normalized = requested.upper().replace("/", "").replace(":USDT", "").replace(":", "")
            symbol = next((key for key, market in markets.items()
                           if market.get("swap") and market.get("quote") == "USDT"
                           and market.get("settle") == "USDT"
                           and key.upper().replace("/", "").replace(":USDT", "").replace(":", "") == normalized), None)
            if symbol is None:
                parser.error(f"unknown Binance USDT-M swap symbol: {requested}")
        server = serve_dashboard(state_path, symbol, exchange, port=args.port)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Paper dashboard listening on {url}", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        if server is not None:
            server.server_close()
        exchange.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
