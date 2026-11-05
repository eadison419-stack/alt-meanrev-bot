"""CTA breakout long-only at PORTFOLIO level (shared cash + max positions).

The proto measured the breakout edge with 30 independent $333 books. Real
deployment shares one $10k book and caps positions (max_symbols=5). This script
replays the same Donchian long-only strategy with cross-symbol coordination and a
position cap, to check whether the clean long edge survives concentration.

Params (textbook): breakout N=20, exit reverse M=10, stop 2x ATR(14), 1% risk,
taker 5bp + slippage 3bp, funding included.

Run:  python scripts/diagnose_cta_portfolio.py --window all
      python scripts/diagnose_cta_portfolio.py --window all --max-positions 30
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

FOUR_HOURS = pd.Timedelta(4, unit="h")


def _indicators(frame, N, M):
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    l = frame["low"].to_numpy(float)
    c = frame["close"].to_numpy(float)
    prev_c = np.empty(len(frame)); prev_c[0] = o[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    upper = pd.Series(h).rolling(N).max().shift(1).to_numpy()
    lower = pd.Series(l).rolling(N).min().shift(1).to_numpy()
    exit_lower = pd.Series(l).rolling(M).min().shift(1).to_numpy()
    exit_upper = pd.Series(h).rolling(M).max().shift(1).to_numpy()
    return o, h, l, c, atr, upper, lower, exit_lower, exit_upper


def run_portfolio(frames, funding_map, total_equity, max_pos, N, M, atr_mult,
                  risk_pct, fee, slip):
    inds = {s: _indicators(f, N, M) for s, f in frames.items()}
    # time -> per-symbol bar index (4H close time t maps to bar with close==t)
    close_to_idx = {}
    for s, f in frames.items():
        m = {f.index[i] + FOUR_HOURS: i for i in range(len(f))}
        close_to_idx[s] = m
    all_times = sorted({t for m in close_to_idx.values() for t in m})

    equity = total_equity
    positions = {}  # symbol -> dict(side, entry_px, stop, qty, entry_time, entry_idx)
    trades = []
    curve = []

    def close(s, px, reason, t, idx):
        nonlocal equity
        p = positions.pop(s)
        sign = 1.0
        entry_fill = p["entry_px"] * (1 + slip)
        exit_fill = px * (1 - slip)
        gross = (exit_fill - entry_fill) * p["qty"]
        fees = p["qty"] * (entry_fill + exit_fill) * fee
        fc = 0.0
        fund = funding_map.get(s)
        if fund is not None and not fund.empty:
            sub = fund.loc[(fund.index > p["entry_time"]) & (fund.index <= t)]
            for _, row in sub.iterrows():
                mp = float(row.get("mark_price", 0.0) or 0.0)
                if mp > 0:
                    fc += p["qty"] * mp * float(row["funding_rate"])
        net = gross - fees - fc
        equity += net
        trades.append({"symbol": s, "reason": reason, "net": net})

    for t in all_times:
        # 1. exits
        for s in list(positions):
            p = positions[s]
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, atr, upper, lower, exit_lower, exit_upper = inds[s]
            if np.isfinite(atr[idx]) and l[idx] <= p["stop"]:
                close(s, min(o[idx], p["stop"]), "stop", t, idx)
            elif np.isfinite(exit_lower[idx]) and c[idx] < exit_lower[idx]:
                close(s, c[idx], "reverse", t, idx)
        # 2. entries (long only), ranked by breakout strength
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, atr, upper, lower, exit_lower, exit_upper = inds[s]
            if np.isfinite(upper[idx]) and np.isfinite(atr[idx]) and atr[idx] > 0 and c[idx] > upper[idx]:
                strength = (c[idx] - upper[idx]) / atr[idx]
                cands.append((s, strength, idx, c[idx], atr[idx]))
        cands.sort(key=lambda x: -x[1])
        for s, _strength, idx, px, atr_val in cands[: max(0, max_pos - len(positions))]:
            # fixed total risk: each position slot gets equity/max_pos, risk 1% of that
            # slot, so total book risk stays ~1% regardless of max_pos (fair
            # concentration-vs-diversification comparison).
            risk = (equity / max_pos) * risk_pct
            stop = px - atr_mult * atr_val
            qty = risk / (atr_mult * atr_val)
            positions[s] = {"entry_px": px, "stop": stop, "qty": qty,
                            "entry_time": t, "entry_idx": idx}
        curve.append(equity)

    # force-close open at end
    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        close(s, float(frames[s]["close"].iloc[idx]), "end",
              frames[s].index[idx] + FOUR_HOURS, idx)
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times))


def _maxdd(curve):
    peak = curve.cummax()
    return float(((curve - peak) / peak).min() * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "all"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--funding-cache", default="data/ab_funding.pkl")
    parser.add_argument("--total-equity", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in ",".join(DEFAULT_SYMBOLS).split(",") if s.strip()]

    with open(args.funding_cache, "rb") as fh:
        funding = pickle.load(fh)
    cache = KlineCache(args.cache)
    end = pd.Timestamp.utcnow().floor("h")
    windows = [int(w) for w in (("1", "2", "3") if args.window == "all" else (args.window,))]

    N, M, atr_mult, risk_pct, fee, slip = 20, 10, 2.0, 0.01, 5 / 10_000, 3 / 10_000

    for w in windows:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=args.window_days), end
        else:
            start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)
        frames = {}
        for s in symbols:
            f = cache.load(s, "4h")
            f = f.loc[(f.index + FOUR_HOURS >= start) & (f.index + FOUR_HOURS <= stop)]
            if len(f) >= 60:
                frames[s] = f
        trades, curve = run_portfolio(frames, funding, args.total_equity, args.max_positions,
                                      N, M, atr_mult, risk_pct, fee, slip)
        net = sum(t["net"] for t in trades)
        wins = sum(1 for t in trades if t["net"] > 0)
        gw = sum(t["net"] for t in trades if t["net"] > 0)
        gl = -sum(t["net"] for t in trades if t["net"] < 0)
        pf = (gw / gl) if gl > 0 else float("inf")
        print(f"=== window {w} {start.date()}..{stop.date()}  max_pos={args.max_positions} ===", flush=True)
        print(f"  net={net:.2f} (ret {net/args.total_equity*100:.2f}%)  trades={len(trades)}  "
              f"win={wins/len(trades)*100:.0f}%  PF={pf:.2f}  maxDD={_maxdd(curve):.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
