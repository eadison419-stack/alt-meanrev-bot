"""Deep dive on the trend_pullback rule: long/short legs + breadth gate.

Splits the closest-to-surviving walk-forward signal into its long leg and short
leg per window, then tests whether a breadth gate (long only when universe
breadth > 0.6, short only when breadth < 0.4) removes the w1 loss.

Frozen rules, no tuning. 5 consecutive 60-day windows (w5..w2) + full period.

Run: python scripts/diagnose_trend_pullback_deep.py
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from diagnose_mr_trend_ab import (DEFAULT_SYMBOLS, FEE, FOUR_HOURS, SLIP,
                                  atr as _atr, daily_regime, ema, load, stats)
from walk_forward_longshort import (REV_BARS, STOP_ATR, compute_signal_series,
                                    run_bidirectional)

DB = "data/market.sqlite"
FUNDING = "data/ab_funding.pkl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--funding", default=FUNDING)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--window-days", type=int, default=60)
    args = ap.parse_args()

    with open(args.funding, "rb") as fh:
        funding = pickle.load(fh)
    b4, b1 = load(args.db)
    sig, _ = compute_signal_series(b4, b1)
    tp = sig["trend_pullback"]

    atr_map = {s: _atr(b4[s], 14) for s in b4}
    exit_lower_map = {s: b4[s]["low"].rolling(REV_BARS).min().shift(1) for s in b4}
    exit_upper_map = {s: b4[s]["high"].rolling(REV_BARS).max().shift(1) for s in b4}

    # breadth series (full 4H index)
    above = pd.DataFrame({s: b4[s]["close"] > ema(b4[s]["close"], 50) for s in b4})
    breadth = above.mean(axis=1, skipna=True)

    end = pd.Timestamp.utcnow().floor("h")
    d = args.window_days
    windows = [
        ("w5", end - pd.Timedelta(days=5 * d), end - pd.Timedelta(days=4 * d)),
        ("w4", end - pd.Timedelta(days=4 * d), end - pd.Timedelta(days=3 * d)),
        ("w3", end - pd.Timedelta(days=3 * d), end - pd.Timedelta(days=2 * d)),
        ("w1", end - pd.Timedelta(days=2 * d), end - pd.Timedelta(days=d)),
        ("w2", end - pd.Timedelta(days=d), end),
    ]
    labels = [w[0] for w in windows]

    def run(frames, mode):
        idx_per = {s: frames[s].index for s in frames}
        long_arr = {s: tp[s]["long"].reindex(idx_per[s]).fillna(False).to_numpy() for s in frames}
        short_arr = {s: tp[s]["short"].reindex(idx_per[s]).fillna(False).to_numpy() for s in frames}
        a_arr = {s: atr_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
        el = {s: exit_lower_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
        eu = {s: exit_upper_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
        if mode == "long":
            short_arr = {s: np.zeros_like(v) for s, v in short_arr.items()}
        elif mode == "short":
            long_arr = {s: np.zeros_like(v) for s, v in long_arr.items()}
        elif mode == "breadth":
            # long only when breadth>0.6, short only when breadth<0.4
            for s in frames:
                bv = breadth.reindex(idx_per[s]).fillna(0.5).to_numpy()
                long_arr[s] = long_arr[s] & (bv > 0.6)
                short_arr[s] = short_arr[s] & (bv < 0.4)
        trades, curve = run_bidirectional(frames, long_arr, short_arr, a_arr, el, eu,
                                          funding, args.equity, args.max_positions)
        return stats(trades, args.equity, curve)

    modes = ["long", "short", "both", "breadth"]
    rows = {m: {} for m in modes}
    for label, start, stop in windows:
        frames = {s: b4[s].loc[(b4[s].index + FOUR_HOURS >= start) & (b4[s].index + FOUR_HOURS <= stop)]
                  for s in b4}
        frames = {s: f for s, f in frames.items() if len(f) >= 80}
        for m in modes:
            rows[m][label] = run(frames, m)

    full = {s: b4[s] for s in b4}
    full_rows = {m: run(full, m) for m in modes}

    print(f"\n{'mode':<10}" + "".join(f"{w:>9}" for w in labels) + f"{'full':>9}")
    print("-" * 80)
    for m in modes:
        rets = [rows[m][w]["ret"] for w in labels]
        n = [rows[m][w]["n"] for w in labels]
        print(f"{m:<10}" + "".join(f"{r:>8.2f}%" for r in rets) + f"{full_rows[m]['ret']:>8.2f}%")
        print(f"{'':<10}" + "".join(f"{x:>8d} " for x in n) + f"{full_rows[m]['n']:>8d} trades")
        print(f"{'':<10}" + "".join(f"L{rows[m][w]['long_net']:>6.0f}/S{rows[m][w]['short_net']:>5.0f}" for w in labels)
              + f"   L{full_rows[m]['long_net']:.0f}/S{full_rows[m]['short_net']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
