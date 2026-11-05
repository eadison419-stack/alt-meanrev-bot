"""Entry-timing feature probe: what distinguishes structure-stop losers from
trailing-stop winners at entry?

The structure-stop diagnosis showed the biggest TREND bleed is追高 entries that
reverse within hours. Before designing an entry-timing filter, measure whether
those losers differ from winners at entry on: distance from EMA50
(|close_to_ema50_pct|), ADX, and trend confirmation streak. This tells us which
filter dimension has signal (without changing any strategy).

Run:  python scripts/diagnose_entry_timing.py --window 2
      python scripts/diagnose_entry_timing.py --window 1
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
from bot.symbol_regime import classify_symbol_regime

from run_range_ab import DEFAULT_SYMBOLS, WINDOW_DAYS, load_or_fetch_funding

FOUR_HOURS = pd.Timedelta(4, unit="h")


def features(bars_4h, symbol, opened_at, settings):
    try:
        a = classify_symbol_regime(symbol, bars_4h[symbol], settings,
                                   now=opened_at.to_pydatetime())
    except Exception:
        return None
    ev = a.evidence or {}
    dev = ev.get("close_to_slow_ema_pct")
    return {
        "dev": dev,
        "abs_dev": abs(dev) if dev is not None else None,
        "adx": ev.get("adx"),
        "streak": ev.get("trend_confirmation_current_streak"),
        "structure": ev.get("structure"),
    }


def _med(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


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
    )

    rows = []
    for t in result.trades:
        if (t.source or "") != "TREND":
            continue
        f = features(bars_4h, t.symbol, t.opened_at, settings)
        if f is None:
            continue
        rows.append({"exit": t.exit_reason, "side": t.side.value,
                     "net": float(t.net_pnl), "abs_dev": f["abs_dev"],
                     "dev": f["dev"], "adx": f["adx"], "streak": f["streak"]})
    frame = pd.DataFrame(rows)
    print(f"=== window {args.window} {start.date()}..{stop.date()}  (n={len(frame)}) ===")

    def grp(tag):
        return frame[frame["exit"].str.startswith(tag)]

    ss = grp("trend_structure_stop")
    tr = grp("trend_trailing_stop")
    ema = grp("trend_key_ema")
    for label, sub in (("structure_stop", ss), ("trailing_stop", tr), ("key_ema", ema)):
        if not len(sub):
            continue
        print(f"  {label:<16} n={len(sub):>3}  |dev_ema| med={sub['abs_dev'].median():.2f}%  "
              f"avg={sub['abs_dev'].mean():.2f}%  adx med={sub['adx'].median():.1f}  "
              f"streak med={sub['streak'].median():.1f}")

    # Deviation buckets for structure-stop losers (追高?): long |dev|, short |dev|.
    if len(ss):
        print("  structure_stop |dev_ema| histogram:")
        for lo, hi, lab in [(0, 2, "<2%"), (2, 4, "2-4%"), (4, 6, "4-6%"),
                            (6, 10, "6-10%"), (10, 10**9, ">10%")]:
            sub = ss[(ss["abs_dev"] >= lo) & (ss["abs_dev"] < hi)]
            if len(sub):
                print(f"      {lab:<8} n={len(sub):>3}  net={sub['net'].sum():>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
