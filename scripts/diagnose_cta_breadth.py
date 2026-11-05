"""CTA Donchian long-only + cross-sectional breadth gate.

Hypothesis: the full-period loss (-28.5%) comes from fake breakouts in chop.
Gate long entries on market breadth — the fraction of the universe whose own
4H close is above its 4H EMA50. Only open a breakout when breadth >= threshold.

Sweep thresholds {0 (baseline), 0.4, 0.5, 0.6, 0.7} over the FULL period, then
optionally per 60-day window for robustness.

Run:
  python scripts/diagnose_cta_breadth.py
  python scripts/diagnose_cta_breadth.py --window all --window-days 60
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

from diagnose_mr_trend_ab import (
    DEFAULT_SYMBOLS, FEE, FOUR_HOURS, SLIP,
    run_cta_baseline, stats,
)

DB = "data/market.sqlite"
FUNDING = "data/ab_funding.pkl"


def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def load_4h(db, symbols):
    con = sqlite3.connect(db, timeout=30)
    frames = {}
    for s in symbols:
        df = pd.read_sql_query(
            "SELECT timestamp_ms, open, high, low, close, volume FROM klines "
            "WHERE symbol=? AND timeframe='4h' ORDER BY timestamp_ms",
            con, params=(s,),
        )
        df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=["timestamp_ms"])
        frames[s] = df
    con.close()
    return frames


def breadth_series(frames):
    above = pd.DataFrame({s: frames[s]["close"] > ema(frames[s]["close"], 50)
                          for s in frames})
    return above.mean(axis=1, skipna=True)


def run_cta_breadth(frames, funding, equity0, max_pos, breadth, threshold,
                    N=20, M=10, atr_mult=2.0, risk_pct=0.01):
    """Long-only CTA with a breadth gate on entries. breadth indexed by bar open time."""
    def inds(f):
        h, l, c = f["high"].to_numpy(float), f["low"].to_numpy(float), f["close"].to_numpy(float)
        o = f["open"].to_numpy(float)
        pc = np.empty(len(f)); pc[0] = o[0]; pc[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        a = pd.Series(tr).rolling(14).mean().to_numpy()
        upper = pd.Series(h).rolling(N).max().shift(1).to_numpy()
        exit_lower = pd.Series(l).rolling(M).min().shift(1).to_numpy()
        return o, h, l, c, a, upper, exit_lower
    ind = {s: inds(f) for s, f in frames.items()}
    close_to_idx = {s: {f.index[i] + FOUR_HOURS: i for i in range(len(f))} for s, f in frames.items()}
    all_times = sorted({t for m in close_to_idx.values() for t in m})
    equity = equity0
    positions = {}
    trades = []
    curve = []
    blocked = 0

    def close(s, px, reason, t):
        nonlocal equity
        p = positions.pop(s)
        entry_fill = p["entry_px"] * (1 + SLIP)
        exit_fill = px * (1 - SLIP)
        gross = (exit_fill - entry_fill) * p["qty"]
        fees = p["qty"] * (entry_fill + exit_fill) * FEE
        fund = funding.get(s)
        fc = 0.0
        if fund is not None and not fund.empty:
            sub = fund.loc[(fund.index > p["entry_time"]) & (fund.index <= t)]
            for _, row in sub.iterrows():
                mp = float(row.get("mark_price", 0.0) or 0.0)
                if mp > 0:
                    fc += p["qty"] * mp * float(row["funding_rate"])
        net = gross - fees - fc
        equity += net
        trades.append({"symbol": s, "side": "LONG", "reason": reason, "net": net})

    for t in all_times:
        for s in list(positions):
            p = positions[s]
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, a, upper, exit_lower = ind[s]
            if np.isfinite(a[idx]) and l[idx] <= p["stop"]:
                close(s, min(o[idx], p["stop"]), "stop", t)
            elif np.isfinite(exit_lower[idx]) and c[idx] < exit_lower[idx]:
                close(s, c[idx], "reverse", t)
        # breadth at the current decision bar (open-time indexed)
        ts = None
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            if ts is None:
                ts = frames[s].index[idx]
            o, h, l, c, a, upper, exit_lower = ind[s]
            if np.isfinite(upper[idx]) and np.isfinite(a[idx]) and a[idx] > 0 and c[idx] > upper[idx]:
                b = float(breadth.loc[ts]) if ts in breadth.index else np.nan
                if np.isfinite(b) and b < threshold:
                    blocked += 1
                    continue
                cands.append((s, (c[idx] - upper[idx]) / a[idx], idx, c[idx], a[idx]))
        cands.sort(key=lambda x: -x[1])
        for s, _strength, idx, px, atr_val in cands[: max(0, max_pos - len(positions))]:
            risk = (equity / max_pos) * risk_pct
            stop = px - atr_mult * atr_val
            qty = risk / (atr_mult * atr_val)
            positions[s] = {"entry_px": px, "stop": stop, "qty": qty, "entry_time": t}
        curve.append(equity)

    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        close(s, float(frames[s]["close"].iloc[idx]), "end", frames[s].index[idx] + FOUR_HOURS)
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times)), blocked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--funding", default=FUNDING)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--window", choices=("full", "1", "2", "3", "4", "5", "all"), default="full")
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--thresholds", default="0.0,0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    with open(args.funding, "rb") as fh:
        funding = pickle.load(fh)
    full_frames = load_4h(args.db, DEFAULT_SYMBOLS)
    end = pd.Timestamp.utcnow().floor("h")
    ths = [float(x) for x in args.thresholds.split(",")]

    if args.window == "full":
        runs = [("full", full_frames, 0.0, None)]
    elif args.window == "all":
        runs = []
        for w in (1, 2, 3, 4, 5):
            d = args.window_days
            if w == 1:
                start, stop = end - pd.Timedelta(days=2 * d), end - pd.Timedelta(days=d)
            elif w == 2:
                start, stop = end - pd.Timedelta(days=d), end
            elif w == 3:
                start, stop = end - pd.Timedelta(days=3 * d), end - pd.Timedelta(days=2 * d)
            elif w == 4:
                start, stop = end - pd.Timedelta(days=4 * d), end - pd.Timedelta(days=3 * d)
            else:
                start, stop = end - pd.Timedelta(days=5 * d), end - pd.Timedelta(days=4 * d)
            fr = {s: f.loc[(f.index + FOUR_HOURS >= start) & (f.index + FOUR_HOURS <= stop)]
                  for s, f in full_frames.items()}
            fr = {s: f for s, f in fr.items() if len(f) >= 80}
            runs.append((f"w{w}", fr, start, stop))
    else:
        w = int(args.window)
        d = args.window_days
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * d), end - pd.Timedelta(days=d)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=d), end
        elif w == 3:
            start, stop = end - pd.Timedelta(days=3 * d), end - pd.Timedelta(days=2 * d)
        elif w == 4:
            start, stop = end - pd.Timedelta(days=4 * d), end - pd.Timedelta(days=3 * d)
        else:
            start, stop = end - pd.Timedelta(days=5 * d), end - pd.Timedelta(days=4 * d)
        fr = {s: f.loc[(f.index + FOUR_HOURS >= start) & (f.index + FOUR_HOURS <= stop)]
              for s, f in full_frames.items()}
        fr = {s: f for s, f in fr.items() if len(f) >= 80}
        runs = [(f"w{w}", fr, start, stop)]

    for label, frames, _start, _stop in runs:
        breadth = breadth_series(frames)
        print(f"\n=== {label}  symbols={len(frames)}  breadth mean={breadth.mean():.2f} ===", flush=True)
        for th in ths:
            trades, curve, blocked = run_cta_breadth(frames, funding, args.equity,
                                                     args.max_positions, breadth, th)
            st = stats(trades, args.equity, curve)
            print(f"  th={th:.1f}: net={st['net']:8.2f} ({st['ret']:7.2f}%) n={st['n']:3d} "
                  f"win={st['win']:4.0f}% PF={st['pf']:5.2f} DD={st['maxdd']:6.2f}% "
                  f"blocked={blocked}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
