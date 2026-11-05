"""Hot-universe exemption A/B: does the live "out-of-rank TREND" path add or lose?

Live paper.py grants a "hot-universe exemption": a symbol outside the ranked hot
universe (symbol_in_hot_universe is False) that independently classifies as a
confirmed TREND (trend_confirmation_pass=True) with a valid pullback may still
open a TREND position (hot_universe_exempted=True). The backtest, by default,
skips out-of-rank symbols entirely, so it systematically undercounts live/paper
TREND trades and exposure.

This script replays the portfolio with the exemption off (baseline, production
backtest behavior) vs on (simulate the live exemption) to measure the edge and
magnitude of the exempted trades, with a separate PnL roll-up for exempted fills.

Run:  python scripts/diagnose_hot_universe_exemption_ab.py --window 2
      python scripts/diagnose_hot_universe_exemption_ab.py --window all
      python scripts/diagnose_hot_universe_exemption_ab.py --window 3 --variant exempt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache

from run_range_ab import DEFAULT_SYMBOLS, load_or_fetch_funding


def _report(label: str, r, t0: float, extra_trades=()) -> None:
    m = r.metrics
    tr = [t for t in r.trades if (t.source or "") == "TREND"]
    n_long = sum(1 for t in tr if t.side.value == "long")
    n_short = sum(1 for t in tr if t.side.value == "short")
    print(f"[{label}] ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  trades={m.get('trade_count')}  TREND n={len(tr)} (L{n_long}/S{n_short})  "
          f"net={m.get('net_pnl'):.2f}  return={m.get('total_return_pct')}%  "
          f"maxDD={m.get('max_drawdown_pct')}%  PF={m.get('profit_factor')}", flush=True)
    for t in extra_trades:
        print(f"      EXEMPT  {t['symbol']:<16} {t['opened_at'][:16]}  side={t['side']:<5} "
              f"exit={t['exit_reason']:<28} pnl={t['net_pnl']:>9.2f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "all"))
    parser.add_argument("--variant", default="all", help="baseline, exempt, or all")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--funding-cache", default="data/ab_funding.pkl")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in ",".join(DEFAULT_SYMBOLS).split(",") if s.strip()]

    cache = KlineCache(args.cache)
    funding = load_or_fetch_funding(symbols, 2 * args.window_days, Path(args.funding_cache))
    bars_1h = {s: cache.load(s, "1h") for s in symbols}
    bars_4h = {s: cache.load(s, "4h") for s in symbols}
    market_bars_4h = cache.load("BTCUSDT", "4h")

    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    end = pd.Timestamp.utcnow().floor("h")
    windows = [int(w) for w in (("1", "2", "3") if args.window == "all" else (args.window,))]
    variants = ["baseline", "exempt"] if args.variant == "all" else [args.variant]

    for w in windows:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=args.window_days), end
        else:
            start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)
        print(f"=== window {w} {start.date()}..{stop.date()} ===", flush=True)
        for name in variants:
            settings = load_settings(env_file=None)
            t0 = time.perf_counter()
            r = run_portfolio_backtest(
                bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
                start=start, end=stop, top_n=20, initial_equity=10_000.0,
                enable_trend_routing=True, market_bars_4h=market_bars_4h,
                mr_entry_mode="legacy", mr_deviation_pct=0.02,
                hot_universe_exemption=(name == "exempt"),
            )
            _report(name, r, t0)
            if name == "exempt":
                records = list(r.hot_universe_exemption_records)
                total = sum(float(x["net_pnl"]) for x in records)
                print(f"    exempted trades n={len(records)}  pnl={total:.2f}", flush=True)
                for x in records:
                    print(f"      EXEMPT  {x['symbol']:<16} {x['opened_at'][:16]}  side={x['side']:<5} "
                          f"exit={x['exit_reason']:<28} pnl={x['net_pnl']:>9.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
