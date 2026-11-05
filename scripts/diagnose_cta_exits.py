"""CTA breakout exit-management A/B: take-profit, trailing stop, breakeven, time stop.

Baseline = Donchian long-only at PORTFOLIO level (shared cash, max_positions),
N=20 entry, M=10 reverse exit, 2xATR(14) fixed stop, 1% risk/slot, taker 5bp +
slippage 3bp, real funding from data/ab_funding.pkl. This is a byte-for-byte
reimplementation of `diagnose_cta_portfolio.py`'s run_portfolio, with opt-in
exit rules layered on top; the `baseline` variant reproduces that script's numbers.

Question under test: does layering a profit-taking / stop-management rule on top
of the textbook breakout improve the long edge, or does it cap / whipsaw the
trend-continuation trades that are the only known positive edge?

Exit rules (each evaluated against the protective/reverse stops in the priority
order: time -> stop -> tp -> reverse; trailing/breakeven update the stop for the
NEXT bar only, never self-trigger on the bar that raises them):
  tp{N}    : exit when high >= entry + N * ATR(entry)            (fill at target)
  trail{N} : Chandelier stop = max(stop, highest_high - N*ATR(current))
  be{N}    : once highest_high >= entry + N*ATR(entry), raise stop to entry (breakeven)
  time{N}  : force-close at bar close after N bars if still open

Run:  python3 scripts/diagnose_cta_exits.py --variant all
      python3 scripts/diagnose_cta_exits.py --variant tp3
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from bot.data.cache import KlineCache

FOUR_HOURS = pd.Timedelta(4, unit="h")

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]

VARIANTS = {
    "baseline": {},
    "tp2": {"tp_mult": 2.0},
    "tp3": {"tp_mult": 3.0},
    "tp4": {"tp_mult": 4.0},
    "tp5": {"tp_mult": 5.0},
    "trail2": {"trail_mult": 2.0},
    "trail3": {"trail_mult": 3.0},
    "trail4": {"trail_mult": 4.0},
    "be1": {"be_mult": 1.0},
    "be1.5": {"be_mult": 1.5},
    "be2": {"be_mult": 2.0},
    "be2.5": {"be_mult": 2.5},
    "be3": {"be_mult": 3.0},
    "time12": {"time_bars": 12},
    "time24": {"time_bars": 24},
}


def _indicators(frame, N, M):
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    l = frame["low"].to_numpy(float)
    c = frame["close"].to_numpy(float)
    prev_c = np.empty(len(frame)); prev_c[0] = o[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    upper = pd.Series(h).rolling(N).max().shift(1).to_numpy()
    exit_lower = pd.Series(l).rolling(M).min().shift(1).to_numpy()
    return o, h, l, c, atr, upper, exit_lower


def run_portfolio(frames, funding_map, total_equity, max_pos, N, M, atr_mult,
                  risk_pct, fee, slip, tp_mult=None, trail_mult=None,
                  be_mult=None, time_bars=None):
    inds = {s: _indicators(f, N, M) for s, f in frames.items()}
    close_to_idx = {}
    for s, f in frames.items():
        close_to_idx[s] = {f.index[i] + FOUR_HOURS: i for i in range(len(f))}
    all_times = sorted({t for m in close_to_idx.values() for t in m})

    equity = total_equity
    positions = {}
    trades = []
    curve = []

    def close(s, px, reason, t):
        nonlocal equity
        p = positions.pop(s)
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
            o, h, l, c, atr, upper, exit_lower = inds[s]
            reason = None
            px = None
            if time_bars and (idx - p["entry_idx"]) >= time_bars:
                reason, px = "time", c[idx]
            if reason is None and np.isfinite(atr[idx]) and l[idx] <= p["stop"]:
                reason, px = "stop", min(o[idx], p["stop"])
            if reason is None and tp_mult and h[idx] >= p["target"]:
                reason, px = "tp", p["target"]
            if reason is None and np.isfinite(exit_lower[idx]) and c[idx] < exit_lower[idx]:
                reason, px = "reverse", c[idx]
            if reason is not None:
                close(s, px, reason, t)
                continue
            # still open: tighten stop for the NEXT bar (never self-trigger)
            if trail_mult is not None and np.isfinite(atr[idx]):
                p["hi"] = max(p["hi"], h[idx])
                p["stop"] = max(p["stop"], p["hi"] - trail_mult * atr[idx])
            if be_mult is not None and np.isfinite(atr[idx]):
                p["hi"] = max(p["hi"], h[idx])
                if p["hi"] >= p["entry_px"] + be_mult * p["atr_entry"]:
                    p["stop"] = max(p["stop"], p["entry_px"])
        # 2. entries (long only), ranked by breakout strength
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, atr, upper, exit_lower = inds[s]
            if np.isfinite(upper[idx]) and np.isfinite(atr[idx]) and atr[idx] > 0 and c[idx] > upper[idx]:
                strength = (c[idx] - upper[idx]) / atr[idx]
                cands.append((s, strength, idx, c[idx], atr[idx]))
        cands.sort(key=lambda x: -x[1])
        for s, _strength, idx, px, atr_val in cands[: max(0, max_pos - len(positions))]:
            risk = (equity / max_pos) * risk_pct
            stop = px - atr_mult * atr_val
            qty = risk / (atr_mult * atr_val)
            positions[s] = {
                "entry_px": px, "stop": stop, "qty": qty,
                "entry_time": t, "entry_idx": idx,
                "hi": px, "atr_entry": atr_val,
                "target": (px + tp_mult * atr_val) if tp_mult else float("inf"),
            }
        curve.append(equity)

    # force-close open at end
    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        close(s, float(frames[s]["close"].iloc[idx]), "end",
              frames[s].index[idx] + FOUR_HOURS)
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times))


def _maxdd(curve):
    peak = curve.cummax()
    return float(((curve - peak) / peak).min() * 100)


def _stats(trades, curve, total_equity):
    net = sum(t["net"] for t in trades)
    n = len(trades)
    wins = sum(1 for t in trades if t["net"] > 0)
    gw = sum(t["net"] for t in trades if t["net"] > 0)
    gl = -sum(t["net"] for t in trades if t["net"] < 0)
    pf = (gw / gl) if gl > 0 else float("inf")
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {
        "net": net, "ret": net / total_equity * 100, "n": n,
        "win": wins / n * 100 if n else 0.0, "pf": pf,
        "dd": _maxdd(curve), "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="all",
                        help="comma-separated variant names, or 'all' (default)")
    parser.add_argument("--window", default="all", choices=("1", "2", "3", "4", "all"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--funding-cache", default="data/ab_funding.pkl")
    parser.add_argument("--total-equity", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    args = parser.parse_args()
    symbols = DEFAULT_SYMBOLS

    with open(args.funding_cache, "rb") as fh:
        funding = pickle.load(fh)
    cache = KlineCache(args.cache)
    end = pd.Timestamp.utcnow().floor("h")
    wins = [int(w) for w in (("1", "2", "3", "4") if args.window == "all" else (args.window,))]
    variants = list(VARIANTS) if args.variant == "all" else [v.strip() for v in args.variant.split(",")]

    N, M, atr_mult, risk_pct, fee, slip = 20, 10, 2.0, 0.01, 5 / 10_000, 3 / 10_000

    # pre-load frames once per window
    window_frames = {}
    for w in wins:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=args.window_days), end
        elif w == 3:
            start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)
        else:
            start, stop = end - pd.Timedelta(days=4 * args.window_days), end - pd.Timedelta(days=3 * args.window_days)
        frames = {}
        for s in symbols:
            f = cache.load(s, "4h")
            f = f.loc[(f.index + FOUR_HOURS >= start) & (f.index + FOUR_HOURS <= stop)]
            if len(f) >= 60:
                frames[s] = f
        window_frames[w] = (start, stop, frames)

    rows = []
    for vname in variants:
        opts = VARIANTS[vname]
        row = {"variant": vname}
        for w in wins:
            start, stop, frames = window_frames[w]
            t0 = time.perf_counter()
            trades, curve = run_portfolio(
                frames, funding, args.total_equity, args.max_positions,
                N, M, atr_mult, risk_pct, fee, slip, **opts)
            st = _stats(trades, curve, args.total_equity)
            row[f"w{w}"] = st
            reasons = " ".join(f"{k}:{v}" for k, v in sorted(st["reasons"].items()))
            print(f"[{vname:<8} w{w} {start.date()}..{stop.date()} {time.perf_counter()-t0:.1f}s] "
                  f"ret={st['ret']:>6.2f}%  n={st['n']:>3}  win={st['win']:>3.0f}%  "
                  f"PF={st['pf']:>5.2f}  maxDD={st['dd']:>5.2f}%  [{reasons}]", flush=True)
        rows.append(row)

    print("\n=== summary (long-only, max_pos={}, funding incl.) ===".format(args.max_positions), flush=True)
    order = [k for k in ("w4", "w3", "w1", "w2") if k in rows[0]]
    print(f"{'variant':<9} " + " ".join(f"{k.upper():>8}" for k in order) + f"  {'total':>8}  "
          + " ".join(f"{k+'dd':>6}" for k in order), flush=True)
    for row in rows:
        vals = [row[k]["ret"] for k in order]
        dds = [row[k]["dd"] for k in order]
        tot = sum(vals)
        print(f"{row['variant']:<9} " + " ".join(f"{v:>7.2f}%" for v in vals) + f"  {tot:>7.2f}%  "
              + " ".join(f"{d:>5.2f}%" for d in dds), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
