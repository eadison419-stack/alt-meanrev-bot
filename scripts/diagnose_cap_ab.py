"""Counterfactual A/B: loosen the two notional caps and watch short-side fills.

The funnel diagnosis showed the 2:1 long bias is introduced at the fill stage by
two *symmetric* notional caps culling tight-stop short entries:

  per_symbol_notional_cap    = margin_per_symbol_usdt * leverage  (default 1000*3 = $3000)
  directional_exposure_cap   = equity * max_directional_notional_pct  (default 0.9)

TREND sizes by risk (0.5% equity ~= $50), so notional ~= $50 / stop_distance_pct,
and short signals carry ~2.8x larger desired notional (tight structure stops).
This script replays the same window with the caps loosened and reports whether
the short entries those caps were culling would have been profitable.

Variants:
  baseline           margin 1000, directional 0.9   (production)
  per_symbol_loose   margin 20000 (cap $60k, never binds), directional 0.9
  dir_loose          margin 1000, directional 1.8   (= gross cap, no extra direction limit)
  both_loose         margin 20000, directional 1.8

Run:  python scripts/diagnose_cap_ab.py --window 2 --variant baseline,both_loose
      python scripts/diagnose_cap_ab.py --window 2 --variant all
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

from run_range_ab import DEFAULT_SYMBOLS, WINDOW_DAYS, load_or_fetch_funding

VARIANTS = {
    "baseline": {"margin": 1000, "dir": 0.9},
    "per_symbol_loose": {"margin": 20000, "dir": 0.9},
    "dir_loose": {"margin": 1000, "dir": 1.8},
    "both_loose": {"margin": 20000, "dir": 1.8},
}


def _run(symbols, bars_1h, bars_4h, funding, market_bars_4h, settings, args,
         variant: str):
    s = settings.model_copy(deep=True)
    s.sizing.margin_per_symbol_usdt = VARIANTS[variant]["margin"]
    s.sizing.max_directional_notional_pct = VARIANTS[variant]["dir"]

    end = pd.Timestamp.utcnow().floor("h")
    if args.window == 1:
        start = end - pd.Timedelta(days=2 * args.window_days)
        stop = end - pd.Timedelta(days=args.window_days)
    else:
        start = end - pd.Timedelta(days=args.window_days)
        stop = end

    t0 = time.perf_counter()
    result = run_portfolio_backtest(
        bars_1h, bars_4h, funding, {s: 0 for s in symbols}, s,
        start=start, end=stop, top_n=20, initial_equity=10_000.0,
        enable_trend_routing=True, market_bars_4h=market_bars_4h,
        mr_entry_mode="legacy", mr_deviation_pct=0.02,
        record_trend_funnel=True,
    )
    elapsed = time.perf_counter() - t0
    m = result.metrics

    tr = [t for t in result.trades if (t.source or "") == "TREND"]
    longs = [t for t in tr if t.side.value == "long"]
    shorts = [t for t in tr if t.side.value == "short"]
    short_net = sum(t.net_pnl for t in shorts)
    short_wins = sum(1 for t in shorts if t.net_pnl > 0)

    # cap rejections from the funnel fill-rejection records
    cap_rej = {"per_symbol_notional_cap": {"long": 0, "short": 0},
               "directional_exposure_cap": {"long": 0, "short": 0}}
    for f in result.trend_funnel_fill_rejections:
        if f["reason"] in cap_rej:
            cap_rej[f["reason"]][f["side"]] += 1

    print(f"\n[{variant} | window {args.window} {start.date()}..{stop.date()}] "
          f"({elapsed:.0f}s)", flush=True)
    print(f"  TREND n_long={len(longs)}  n_short={len(shorts)}  "
          f"short_win_rate={short_wins/len(shorts)*100 if shorts else 0:.0f}%", flush=True)
    print(f"  net={m.get('net_pnl'):.2f}  return={m.get('total_return_pct')}%  "
          f"maxDD={m.get('max_drawdown_pct')}%  PF={m.get('profit_factor')}", flush=True)
    print(f"  short_net_total={short_net:.2f}  "
          f"short_avg={short_net/len(shorts) if shorts else 0:.2f}", flush=True)
    print(f"  cap rejections: per_symbol L{cap_rej['per_symbol_notional_cap']['long']}"
          f"/S{cap_rej['per_symbol_notional_cap']['short']}  "
          f"directional L{cap_rej['directional_exposure_cap']['long']}"
          f"/S{cap_rej['directional_exposure_cap']['short']}", flush=True)
    if shorts:
        print("  shorts:", flush=True)
        for t in sorted(shorts, key=lambda x: x.net_pnl):
            held_h = (t.closed_at - t.opened_at).total_seconds() / 3600
            print(f"    {t.symbol:<12} {t.opened_at.date()} net={t.net_pnl:>8.2f} "
                  f"held={held_h:>5.1f}h exit={t.exit_reason}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True, choices=(1, 2))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
    parser.add_argument("--variant", default="all")
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
    variants = (list(VARIANTS) if args.variant == "all"
                else [v.strip() for v in args.variant.split(",") if v.strip()])
    for v in variants:
        _run(symbols, bars_1h, bars_4h, funding, market_bars_4h, settings, args, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
