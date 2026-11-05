"""Diagnose the biggest TREND loss engine: trend_structure_stop_hit.

The regime diagnosis showed trend_structure_stop_hit is the persistent bleeder
(W1 -962/25 trades, W2 -1065/26), while trend_trailing_stop_hit is the winner
engine. This script buckets TREND trades by exit reason and profiles the
structure-stop losers: hold time (entry immediately reversed?), side, and net.

Run:  python scripts/diagnose_structure_stop.py --window 2
      python scripts/diagnose_structure_stop.py --window 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache

from run_range_ab import DEFAULT_SYMBOLS, WINDOW_DAYS, load_or_fetch_funding


def held_h(t):
    return (t.closed_at - t.opened_at).total_seconds() / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True, choices=(1, 2))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    cache = KlineCache(args.cache)
    funding = load_or_fetch_funding(symbols, 2 * args.window_days, Path(args.funding_cache))
    bars_1h, bars_4h = {}, {}
    for sym in symbols:
        bars_1h[sym] = cache.load(sym, "1h")
        bars_4h[sym] = cache.load(sym, "4h")
    market_bars_4h = cache.load("BTCUSDT", "4h")

    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    settings = load_settings(env_file=None)
    end = pd.Timestamp.utcnow().floor("h")
    if args.window == 1:
        start = end - pd.Timedelta(days=2 * args.window_days)
        stop = end - pd.Timedelta(days=args.window_days)
    else:
        start = end - pd.Timedelta(days=args.window_days)
        stop = end

    result = run_portfolio_backtest(
        bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
        start=start, end=stop, top_n=20, initial_equity=10_000.0,
        enable_trend_routing=True, market_bars_4h=market_bars_4h,
        mr_entry_mode="legacy", mr_deviation_pct=0.02,
        record_trend_funnel=True,
    )
    m = result.metrics
    print(f"=== window {args.window} {start.date()}..{stop.date()} ===")
    print(f"  net={m.get('net_pnl'):.2f}  PF={m.get('profit_factor')}  trades={m.get('trade_count')}")

    tr = [t for t in result.trades if (t.source or "") == "TREND"]
    rows = [{"exit": t.exit_reason, "side": t.side.value, "net": float(t.net_pnl),
             "held_h": held_h(t)} for t in tr]
    frame = pd.DataFrame(rows)
    print("\n  by exit_reason:")
    for reason, sub in frame.groupby("exit"):
        wins = (sub["net"] > 0).sum()
        print(f"    {reason:<32} n={len(sub):>3}  net={sub['net'].sum():>9.2f}  "
              f"win_rate={wins/len(sub)*100:5.1f}%  "
              f"held_med={sub['held_h'].median():>5.1f}h  held_avg={sub['held_h'].mean():>5.1f}h")

    # Profile the structure-stop losers specifically.
    ss = frame[frame["exit"].str.startswith("trend_structure_stop")].copy()
    if len(ss):
        print(f"\n  trend_structure_stop profile (n={len(ss)}):")
        for side, sub in ss.groupby("side"):
            print(f"    {side:<5} n={len(sub):>3}  net={sub['net'].sum():>8.2f}  "
                  f"held_med={sub['held_h'].median():>5.1f}h")
        bins = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 10**9)]
        labels = ["<2h", "2-6h", "6-12h", "12-24h", ">24h"]
        print("    hold-time distribution:")
        for (lo, hi), lab in zip(bins, labels):
            sub = ss[(ss["held_h"] >= lo) & (ss["held_h"] < hi)]
            if len(sub):
                print(f"      {lab:<8} n={len(sub):>3}  net={sub['net'].sum():>8.2f}  "
                      f"avg={sub['net'].mean():>7.2f}")
    # Stop distance (initial_risk % of entry) by exit reason, from opt-in records.
    stops = pd.DataFrame(result.trend_stops)
    if len(stops):
        print("\n  stop distance (initial risk % of entry) by exit_reason:")
        for reason, sub in stops.groupby("exit_reason"):
            print(f"    {reason:<32} n={len(sub):>3}  "
                  f"med={sub['stop_distance_pct'].median():.2f}%  "
                  f"avg={sub['stop_distance_pct'].mean():.2f}%  "
                  f"p10={sub['stop_distance_pct'].quantile(0.1):.2f}%  "
                  f"p90={sub['stop_distance_pct'].quantile(0.9):.2f}%")
        sss = stops[stops["exit_reason"].str.startswith("trend_structure_stop")]
        if len(sss):
            print("  trend_structure_stop stop-distance histogram:")
            for lo, hi, lab in [(0, 1, "<1%"), (1, 1.5, "1-1.5%"), (1.5, 2, "1.5-2%"),
                                (2, 3, "2-3%"), (3, 5, "3-5%"), (5, 10**9, ">5%")]:
                sub = sss[(sss["stop_distance_pct"] >= lo) & (sss["stop_distance_pct"] < hi)]
                if len(sub):
                    print(f"      {lab:<8} n={len(sub):>3}  net={sub['net_pnl'].sum():>8.2f}  "
                          f"avg={sub['net_pnl'].mean():>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
