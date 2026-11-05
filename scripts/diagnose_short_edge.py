"""Short-side edge probe (Route A, option 2): do TREND_DOWN shorts have any edge?

Long-alt-beta dominates; shorts look like they should win in BTC downtrends. But
the W2 downtrend shorts (exemption-released) lost in the late half. This script
replays 3 windows (hot universe + hot-universe exemption, so downtrend symbols
are released back in and the short sample is real) and groups every short by
(a) exit reason, (b) BTC 4H regime at entry (close vs EMA50, and EMA50 6-bar
slope) to find whether any regime hosts a positive short edge.

Run:  python scripts/diagnose_short_edge.py --window all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache

from run_range_ab import DEFAULT_SYMBOLS, load_or_fetch_funding

FOUR_HOURS = pd.Timedelta(4, unit="h")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "all"))
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

    settings = load_settings(env_file=None)
    span = settings.symbol_regime.ema_slow_bars
    btc_ema = market_bars_4h["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    btc_slope = btc_ema / btc_ema.shift(6) - 1.0

    def btc_regime(ts: pd.Timestamp) -> str:
        idx = int(np.searchsorted(market_bars_4h.index.asi8 + FOUR_HOURS.value, ts.value, side="right") - 1)
        if idx < 0:
            return "?"
        pos = "above" if float(market_bars_4h["close"].iloc[idx]) > float(btc_ema.iloc[idx]) else "below"
        s = btc_slope.iloc[idx]
        slope = "up" if (np.isfinite(s) and s > 0) else "down"
        return f"{pos}_{slope}"

    end = pd.Timestamp.utcnow().floor("h")
    windows = [int(w) for w in (("1", "2", "3") if args.window == "all" else (args.window,))]

    for w in windows:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=args.window_days), end
        else:
            start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)
        t0 = time.perf_counter()
        r = run_portfolio_backtest(
            bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
            start=start, end=stop, top_n=20, initial_equity=10_000.0,
            enable_trend_routing=True, market_bars_4h=market_bars_4h,
            mr_entry_mode="legacy", mr_deviation_pct=0.02,
            hot_universe_exemption=True,
        )
        m = r.metrics
        shorts = [t for t in r.trades if (t.source or "") == "TREND" and t.side.value == "short"]
        longs = [t for t in r.trades if (t.source or "") == "TREND" and t.side.value == "long"]
        print(f"\n=== window {w} {start.date()}..{stop.date()} ({time.perf_counter() - t0:.0f}s) ===", flush=True)
        print(f"  net={m.get('net_pnl'):.2f} PF={m.get('profit_factor')} maxDD={m.get('max_drawdown_pct')}  "
              f"longs={len(longs)} shorts={len(shorts)}", flush=True)

        def stat(trades):
            if not trades:
                return "  (none)"
            pnl = sum(float(t.net_pnl) for t in trades)
            wins = sum(1 for t in trades if float(t.net_pnl) > 0)
            return f"n={len(trades)} pnl={pnl:.2f} win={wins}/{len(trades)} ({100*wins/len(trades):.0f}%)"

        print(f"  SHORT total:  {stat(shorts)}", flush=True)
        print(f"  LONG  total:  {stat(longs)}", flush=True)

        # shorts by exit reason
        by_exit = {}
        for t in shorts:
            key = (t.exit_reason or "?").replace("trend_", "")
            by_exit.setdefault(key, []).append(t)
        print("  SHORT by exit:", flush=True)
        for key in sorted(by_exit):
            print(f"      {key:<24} {stat(by_exit[key])}", flush=True)

        # shorts by BTC regime at entry
        by_regime = {}
        for t in shorts:
            by_regime.setdefault(btc_regime(pd.Timestamp(t.opened_at)), []).append(t)
        print("  SHORT by BTC regime at entry (close_vs_ema50 + slope):", flush=True)
        for key in sorted(by_regime):
            print(f"      {key:<12} {stat(by_regime[key])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
