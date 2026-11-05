"""Entry-timing feature probe v2: which entry feature separates trend-continuation
winners (trailing_stop) from trend-exhaustion losers (structure_stop)?

v1 found ADX differs (losers hotter) but the ADX cap died on the W3 OOS window
(a chop filter, not entry-timing alpha). v2 adds "how far has the trend already
run" features — confirmation streak and cumulative move since the streak start —
which are the natural axes for early-trend vs late-trend. Diagnostic only.

Run:  python scripts/diagnose_entry_timing2.py --window 1
      python scripts/diagnose_entry_timing2.py --window 3
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

from run_range_ab import DEFAULT_SYMBOLS, load_or_fetch_funding

FOUR_HOURS = pd.Timedelta(4, unit="h")


def features(bars_4h, symbol, opened_at, settings):
    try:
        a = classify_symbol_regime(symbol, bars_4h[symbol], settings,
                                   now=opened_at.to_pydatetime())
    except Exception:
        return None
    ev = a.evidence or {}
    dev = ev.get("close_to_slow_ema_pct")
    streak = ev.get("trend_confirmation_current_streak")
    c = bars_4h[symbol]
    closes = c["close"].to_numpy(dtype=float)
    idx = int(c.index.searchsorted(opened_at - FOUR_HOURS, side="right") - 1)
    cum_move = None
    if streak is not None and idx >= 0:
        n = int(streak)
        base_idx = idx - (n - 1)
        if base_idx >= 0 and closes[base_idx] > 0:
            cum_move = (closes[idx] / closes[base_idx] - 1.0) * 100
    return {
        "abs_dev": abs(dev) if dev is not None else None,
        "adx": ev.get("adx"),
        "streak": streak,
        "slope": ev.get("slow_ema_slope_pct"),
        "cum_move": cum_move,
    }


def _med(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--funding-cache", default="data/ab_funding.pkl")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in ",".join(DEFAULT_SYMBOLS).split(",") if s.strip()]

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
    elif args.window == 2:
        start = end - pd.Timedelta(days=args.window_days)
        stop = end
    else:
        start = end - pd.Timedelta(days=3 * args.window_days)
        stop = end - pd.Timedelta(days=2 * args.window_days)

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
        # signed move in the trade's own direction
        d = f["cum_move"] if t.side.value == "long" else (-f["cum_move"] if f["cum_move"] is not None else None)
        rows.append({"exit": t.exit_reason, "side": t.side.value, "net": float(t.net_pnl),
                     "abs_dev": f["abs_dev"], "adx": f["adx"], "streak": f["streak"],
                     "slope": f["slope"], "dir_move": d})
    frame = pd.DataFrame(rows)
    print(f"=== window {args.window} {start.date()}..{stop.date()}  (n={len(frame)}) ===")

    def grp(tag):
        return frame[frame["exit"].str.startswith(tag)]

    ss, tr, ema = grp("trend_structure_stop"), grp("trend_trailing_stop"), grp("trend_key_ema")
    for label, sub in (("structure_stop(输家)", ss), ("trailing_stop(赢家)", tr), ("key_ema", ema)):
        if not len(sub):
            continue
        print(f"  {label:<20} n={len(sub):>3}  net={sub['net'].sum():>8.2f}  "
              f"|dev|med={sub['abs_dev'].median():.2f}  adx med={sub['adx'].median():.1f}  "
              f"streak med={sub['streak'].median():.1f}  slope med={sub['slope'].median():.3f}  "
              f"dir_move med={sub['dir_move'].median():.2f}%")

    # streak buckets: does the loser share concentrate at high streak (late trend)?
    if len(ss) or len(tr):
        print("  streak 分桶 (structure_stop vs trailing_stop):")
        for lo, hi in [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 10**9)]:
            s = ss[(ss["streak"] >= lo) & (ss["streak"] <= hi)]
            t = tr[(tr["streak"] >= lo) & (tr["streak"] <= hi)]
            lab = f"{lo}" if lo == hi else f"{lo}+"
            if len(s) or len(t):
                print(f"      streak={lab:<4}  SS n={len(s):>3} net={s['net'].sum():>8.2f}   "
                      f"TR n={len(t):>3} net={t['net'].sum():>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
