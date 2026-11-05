"""Pre-check for direction #2: does BTC 4H direction predict TREND long PnL?

The funnel/cap work showed TREND is long alt-beta and its PnL is BTC-direction
dominated. If TREND longs systematically lose when BTC 4H is below its EMA50
(or BTC is falling over the last 24h), then a BTC-direction filter on long
entries has data support worth A/B-testing. This script runs the baseline
backtest and buckets TREND trades by the BTC 4H state at entry time.

Run:  python scripts/diagnose_btc_direction.py --window 2
      python scripts/diagnose_btc_direction.py --window 1
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

FOUR_HOURS = pd.Timedelta(4, unit="h")


def btc_state(market_bars_4h, opened_at, settings):
    """BTC 4H state at a trade's open: close vs EMA_slow, 24h and 4h return."""
    closed = market_bars_4h.loc[market_bars_4h.index + FOUR_HOURS <= opened_at]
    span = settings.symbol_regime.ema_slow_bars
    if len(closed) < span + 1:
        return None
    ema = closed["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    close = float(closed["close"].iloc[-1])
    above = close > float(ema.iloc[-1])
    ret_24h = close / float(closed["close"].iloc[-7]) - 1.0 if len(closed) >= 7 else None
    ret_4h = close / float(closed["close"].iloc[-2]) - 1.0 if len(closed) >= 2 else None
    return {"above_ema": above, "ret_24h": ret_24h, "ret_4h": ret_4h}


def _bucket(trades, market_bars_4h, settings, side, label):
    rows = []
    for t in trades:
        if (t.source or "") != "TREND" or t.side.value != side:
            continue
        st = btc_state(market_bars_4h, t.opened_at, settings)
        if st is None:
            continue
        rows.append({"net": float(t.net_pnl), "above_ema": st["above_ema"],
                     "ret_24h": st["ret_24h"]})
    if not rows:
        print(f"\n  {side} (no trades)")
        return
    frame = pd.DataFrame(rows)
    print(f"\n  {side.upper()} by BTC 4H close vs EMA50 at entry (n={len(frame)}):")
    for above, sub in frame.groupby("above_ema"):
        tag = "above_ema" if above else "below_ema"
        wins = (sub["net"] > 0).sum()
        print(f"    {tag:<10} n={len(sub):>3}  net={sub['net'].sum():>9.2f}  "
              f"win_rate={wins/len(sub)*100:5.1f}%  avg={sub['net'].mean():>7.2f}")
    print(f"  {side.upper()} by BTC 24h return at entry:")
    up = frame[frame["ret_24h"] >= 0]
    down = frame[frame["ret_24h"] < 0]
    for tag, sub in (("btc_24h_up", up), ("btc_24h_down", down)):
        if not len(sub):
            continue
        wins = (sub["net"] > 0).sum()
        print(f"    {tag:<14} n={len(sub):>3}  net={sub['net'].sum():>9.2f}  "
              f"win_rate={wins/len(sub)*100:5.1f}%  avg={sub['net'].mean():>7.2f}")


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
    m = result.metrics
    print(f"=== window {args.window} {start.date()}..{stop.date()} ===")
    print(f"  overall net={m.get('net_pnl'):.2f}  PF={m.get('profit_factor')}  "
          f"trades={m.get('trade_count')}")
    _bucket(result.trades, market_bars_4h, settings, "long", "long")
    _bucket(result.trades, market_bars_4h, settings, "short", "short")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
