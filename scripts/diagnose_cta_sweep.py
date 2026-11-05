"""CTA prototype parameter robustness sweep (long-only).

A single textbook param set (N=20/M=10/2xATR) gave a suspiciously clean long
edge (W3 PF 13.89). That is exactly the overfitting signature that killed the
ADX filter. This sweep answers: does the long breakout edge survive parameter
variation, or is it a single-parameter fluke?

Sweeps N (breakout), M (exit), ATR multiple; long-only, $10k/30 symbols, 3 windows.
Robustness read: the share of configs that are net-positive in each window, and
the median/min/max net across the grid.

Run:  python scripts/diagnose_cta_sweep.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from bot.data.cache import KlineCache

from run_range_ab import DEFAULT_SYMBOLS
from diagnose_cta_proto import run_symbol

FOUR_HOURS = pd.Timedelta(4, unit="h")


def _maxdd(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float(((curve - peak) / peak).min() * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--total-equity", type=float, default=10_000.0)
    parser.add_argument("--funding-cache", default=None,
                        help="path to funding pickle; omit to ignore funding")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in ",".join(DEFAULT_SYMBOLS).split(",") if s.strip()]
    per_symbol = args.total_equity / len(symbols)

    funding = None
    if args.funding_cache:
        with open(args.funding_cache, "rb") as fh:
            funding = pickle.load(fh)

    cache = KlineCache(args.cache)
    end = pd.Timestamp.utcnow().floor("h")
    windows = [
        (1, end - pd.Timedelta(days=60), end - pd.Timedelta(days=30)),
        (2, end - pd.Timedelta(days=30), end),
        (3, end - pd.Timedelta(days=90), end - pd.Timedelta(days=60)),
    ]

    fee, slip = 5 / 10_000, 3 / 10_000
    grid = [(N, M, a) for N in (10, 20, 55) for M in (10, 20) for a in (1.5, 2.0, 3.0)]

    # Pre-load frames once.
    frames = {}
    for s in symbols:
        frames[s] = cache.load(s, "4h")

    results = []
    for (N, M, atr_mult) in grid:
        row = {"N": N, "M": M, "atr": atr_mult}
        for (w, start, stop) in windows:
            net = 0.0
            dd = []
            for s in symbols:
                f = frames[s].loc[(frames[s].index + FOUR_HOURS >= start)
                                  & (frames[s].index + FOUR_HOURS <= stop)]
                if len(f) < 60:
                    continue
                trades, curve = run_symbol(f, per_symbol, N, M, atr_mult, 0.01, fee, slip,
                                           long_only=True, allow_entry=None,
                                           funding=funding.get(s) if funding else None)
                net += sum(t["net"] for t in trades)
                dd.append(_maxdd(pd.Series(curve)))
            row[f"w{w}"] = net / args.total_equity * 100
            row[f"dd{w}"] = float(np.mean(dd)) if dd else 0.0
        row["total"] = row["w1"] + row["w2"] + row["w3"]
        results.append(row)
        print(f"N={N:>2} M={M:>2} atr={atr_mult:.1f}  w1={row['w1']:>6.2f}%  w2={row['w2']:>6.2f}%  "
              f"w3={row['w3']:>6.2f}%  total={row['total']:>6.2f}%", flush=True)

    df = pd.DataFrame(results)
    print("\n=== robustness summary (long-only, 3 windows) ===", flush=True)
    for w in ("w1", "w2", "w3"):
        pos = (df[w] > 0).mean() * 100
        print(f"  {w}: {pos:.0f}% of {len(df)} configs net-positive  "
              f"(min {df[w].min():.2f}% / med {df[w].median():.2f}% / max {df[w].max():.2f}%)", flush=True)
    pos_total = (df["total"] > 0).mean() * 100
    print(f"  total: {pos_total:.0f}% net-positive  (min {df['total'].min():.2f}% / "
          f"med {df['total'].median():.2f}% / max {df['total'].max():.2f}%)", flush=True)
    # Is the edge concentrated in the default (20,10,2.0)?
    d = df[(df["N"] == 20) & (df["M"] == 10) & (df["atr"] == 2.0)].iloc[0]
    print(f"  default(20,10,2.0) total={d['total']:.2f}% vs grid median {df['total'].median():.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
