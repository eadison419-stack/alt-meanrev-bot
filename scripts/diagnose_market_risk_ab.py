"""Market-risk gate A/B: does the direction-neutral volatility scaler help or hurt?

The market_risk entry gate classifies BTC 4H into NORMAL / RISK_ELEVATED /
PAUSE_NEW and scales new-entry size by elevated_size_multiplier (0.5) when
RISK_ELEVATED. Because it keys on *absolute* 24h moves and single-bar impulse
(volatility, not direction), it fires in both bull and bear stretches. This
script replays the portfolio with the gate on (baseline) vs off (mr_off,
always NORMAL => full size) to measure whether the scaler is net-helpful.

Run:  python scripts/diagnose_market_risk_ab.py --window 2 --variant mr_off
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3"))
    parser.add_argument("--variant", default="all", help="baseline, mr_off, or all")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
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
    w = int(args.window)
    if w == 1:
        start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
    elif w == 2:
        start, stop = end - pd.Timedelta(days=args.window_days), end
    else:
        start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)

    variants = ["baseline", "mr_off"] if args.variant == "all" else [args.variant]
    for name in variants:
        settings = load_settings(env_file=None)
        if name == "mr_off":
            settings.market_risk.enabled = False
        t0 = time.perf_counter()
        r = run_portfolio_backtest(
            bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
            start=start, end=stop, top_n=20, initial_equity=10_000.0,
            enable_trend_routing=True, market_bars_4h=market_bars_4h,
            mr_entry_mode="legacy", mr_deviation_pct=0.02,
        )
        m = r.metrics
        tr = [t for t in r.trades if (t.source or "") == "TREND"]
        n_long = sum(1 for t in tr if t.side.value == "long")
        n_short = sum(1 for t in tr if t.side.value == "short")
        print(f"[{name} | window {w} {start.date()}..{stop.date()}] ({time.perf_counter() - t0:.0f}s)", flush=True)
        print(f"  TREND n={len(tr)} (L{n_long}/S{n_short})  "
              f"net={m.get('net_pnl'):.2f}  return={m.get('total_return_pct')}%  "
              f"maxDD={m.get('max_drawdown_pct')}%  PF={m.get('profit_factor')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
