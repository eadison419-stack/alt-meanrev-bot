"""Narrow proposition: short pullbacks inside a daily DOWN trend.

Hypothesis: only SHORT, only when the daily regime is DOWN (close < EMA50(1D)
and 3-day EMA50 slope < 0); enter when a 1H up-ripple closes >= k*ATR14 above
the 1H EMA50 and that 1H bar is bearish (fade the bounce).

Grid:
  regime gate  = {down} vs {down_or_neutral}   (never short an UP trend)
  k            = 1.0 / 1.5 / 2.0
  exit         = base (BE +1.0 ATR, trail 0.8) vs relaxed (BE +1.5, trail 2.0)

Reuses the signal/loader/indicator helpers from diagnose_mr_trend_ab; only the
entry gate and the short-only engine differ. 60-day x 5 windows (W3..W5 are the
previously untouched holdouts).

Run: python scripts/diagnose_mr_trend_short.py --window all --window-days 60
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from diagnose_mr_trend_ab import (
    DEFAULT_SYMBOLS, FEE, FOUR_HOURS, SLIP,
    build_signals, load, stats,
)

DB = "data/market.sqlite"
FUNDING = "data/ab_funding.pkl"


def run_short_engine(frames, signals, funding, equity0, max_pos, regime_gate, exit_mode, risk_pct=0.01):
    """Short-only engine. regime_gate in {'down', 'down_or_neutral'}."""
    allowed = ("DOWN",) if regime_gate == "down" else ("DOWN", "NEUTRAL")
    close_to_idx = {s: {f.index[i] + FOUR_HOURS: i for i in range(len(f))} for s, f in frames.items()}
    all_times = sorted({t for m in close_to_idx.values() for t in m})
    equity = equity0
    positions = {}
    trades = []
    curve = []
    funnel = {"cand": 0, "pass": 0, "trigger": 0, "filled": 0, "cut_neutral": 0, "cut_up": 0}

    if exit_mode == "base":
        be_r, trail_after, trail_mult = 1.0, 0.8, 0.8
    else:
        be_r, trail_after, trail_mult = 1.5, 2.0, 2.0

    def realize(s, px, reason, t):
        nonlocal equity
        p = positions.pop(s)
        entry_fill = p["entry_px"] * (1 + SLIP)
        exit_fill = px * (1 - SLIP)
        gross = (entry_fill - exit_fill) * p["qty"]
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
        trades.append({"symbol": s, "side": "SHORT", "reason": reason, "net": net})

    for t in all_times:
        for s in list(positions):
            p = positions[s]
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            f = frames[s]
            o, h, l, c = (f["open"].iloc[idx], f["high"].iloc[idx],
                          f["low"].iloc[idx], f["close"].iloc[idx])
            atr_val = float(signals[s]["atr4h"].loc[f.index[idx]])
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            if h >= p["stop"]:
                realize(s, max(o, p["stop"]), "stop", t); continue
            r = (p["entry_px"] - l) / p["entry_atr"]
            if r >= be_r:
                p["stop"] = min(p["stop"], p["entry_px"])
            p["best_low"] = min(p["best_low"], l)
            if r >= trail_after:
                p["stop"] = min(p["stop"], p["best_low"] + trail_mult * atr_val)
            bars_held = idx - p["entry_idx"]
            if bars_held >= 6:
                prog = (p["entry_px"] - c) / p["entry_atr"]
                if prog < 0.4:
                    realize(s, c, "no_progress", t); continue
            if bars_held >= 24:
                realize(s, c, "max_hold", t); continue
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            sig = signals[s]
            ts = frames[s].index[idx]
            regime = sig["regime"].loc[ts]
            strig = int(sig["short_trigger"].loc[ts])
            if not strig:
                continue
            funnel["cand"] += 1
            if regime == "UP":
                funnel["cut_up"] += 1
                continue
            if regime not in allowed:
                funnel["cut_neutral"] += 1
                continue
            funnel["pass"] += 1
            funnel["trigger"] += 1
            cands.append((s, idx, float(frames[s]["close"].iloc[idx]), float(sig["atr4h"].loc[ts])))
        for s, idx, px, atr_val in cands[: max(0, max_pos - len(positions))]:
            slot = equity / max_pos
            risk = slot * risk_pct
            stop = px + 2.0 * atr_val
            qty = risk / (2.0 * atr_val)
            positions[s] = {"side": "SHORT", "entry_px": px, "stop": stop, "qty": qty,
                            "entry_atr": atr_val, "entry_time": t, "entry_idx": idx,
                            "best_low": px}
            funnel["filled"] += 1
        curve.append(equity)

    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        realize(s, float(frames[s]["close"].iloc[idx]), "end", frames[s].index[idx] + FOUR_HOURS)
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times)), funnel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", required=True, choices=("1", "2", "3", "4", "5", "all"))
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--funding", default=FUNDING)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--k-list", default="1.0,1.5,2.0")
    args = ap.parse_args()

    with open(args.funding, "rb") as fh:
        funding = pickle.load(fh)
    b4, b1 = load(args.db)
    end = pd.Timestamp.utcnow().floor("h")
    windows = [int(w) for w in (("1", "2", "3", "4", "5") if args.window == "all" else (args.window,))]
    ks = [float(x) for x in args.k_list.split(",")]

    for w in windows:
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
        frames = {}
        for s in b4:
            f = b4[s].loc[(b4[s].index + FOUR_HOURS >= start) & (b4[s].index + FOUR_HOURS <= stop)]
            if len(f) >= 80:
                frames[s] = f
        print(f"\n=== window {w}  {start.date()}..{stop.date()}  symbols={len(frames)}  max_pos={args.max_positions} ===", flush=True)
        for k in ks:
            sig = build_signals(b4, b1, k)
            for gate in ("down", "down_or_neutral"):
                for emode in ("base", "relaxed"):
                    trades, curve, funnel = run_short_engine(frames, sig, funding, args.equity,
                                                             args.max_positions, gate, emode)
                    st = stats(trades, args.equity, curve)
                    print(f"  [gate={gate} k={k} {emode}] net={st['net']:7.2f} ({st['ret']:6.2f}%) "
                          f"n={st['n']:3d} win={st['win']:4.0f}% PF={st['pf']:5.2f} DD={st['maxdd']:5.2f}% "
                          f"fill={funnel['filled']}/{funnel['trigger']} cut_up={funnel['cut_up']} cut_neu={funnel['cut_neutral']}",
                          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
