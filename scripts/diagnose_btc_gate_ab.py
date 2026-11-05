"""BTC bull-regime long gate A/B (Route A, option 1).

The TREND engine is long-alt-beta: trailing_stop continuation winners live in BTC
bull phases, while structure_stop "chase" losers bleed in chop/downtrend. Gate the
long side on a BTC 4H regime — allow long TREND only when BTC 4H close is above
its EMA50 (btc_bull_long_only=True) — and see whether it cuts the structure_stop
bleed while keeping the trailing_stop winners.

Run:  python scripts/diagnose_btc_gate_ab.py --window 2
      python scripts/diagnose_btc_gate_ab.py --window all
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


def _report(label: str, r, t0: float) -> None:
    m = r.metrics
    tr = [t for t in r.trades if (t.source or "") == "TREND"]
    n_long = sum(1 for t in tr if t.side.value == "long")
    n_short = sum(1 for t in tr if t.side.value == "short")
    ss = sum(1 for t in tr if (t.exit_reason or "").startswith("trend_structure_stop"))
    trl = sum(1 for t in tr if (t.exit_reason or "").startswith("trend_trailing_stop"))
    ss_pnl = sum(float(t.net_pnl) for t in tr if (t.exit_reason or "").startswith("trend_structure_stop"))
    trl_pnl = sum(float(t.net_pnl) for t in tr if (t.exit_reason or "").startswith("trend_trailing_stop"))
    print(f"[{label}] ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  trades={m.get('trade_count')}  TREND n={len(tr)} (L{n_long}/S{n_short})  "
          f"net={m.get('net_pnl'):.2f}  return={m.get('total_return_pct')}%  "
          f"maxDD={m.get('max_drawdown_pct')}%  PF={m.get('profit_factor')}", flush=True)
    print(f"  structure_stop n={ss} pnl={ss_pnl:.2f}   trailing_stop n={trl} pnl={trl_pnl:.2f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "all"))
    parser.add_argument("--variant", default="all", help="baseline, gate, or all")
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
    variants = ["baseline", "gate"] if args.variant == "all" else [args.variant]

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
                btc_bull_long_only=(name == "gate"),
            )
            _report(name, r, t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
