"""Cross-sectional relative-strength leg prototype vs the CTA Donchian baseline.

STATUS (2026-10-24): FALSIFIED. The standalone relative-strength leg shows no
robust edge -- the momentum-lookback response is noisy and non-monotonic (rs18
is an isolated spike), breakeven (be2) is harmful here, and every variant is
still long-beta (loses in the W2 downtrend window). Kept as an evidence trail
only; not wired into production. See DIAGNOSIS_2026-10-23_funnel_cap.md section
18 (十八).

The CTA breakout strategy enters long on a per-symbol Donchian breakout (close
above the prior N-bar high). This script tests a *different* entry signal: a
standalone cross-sectional relative-strength leg that ranks every symbol by its
trailing momentum and holds the top-K strongest. Everything else is identical to
the CTA framework: long-only, 2xATR(14) fixed stop, M-bar reverse-breakout exit,
optional breakeven (be2), 1% risk/slot, taker 5bp + slippage 3bp, real funding
from data/ab_funding.pkl, max_positions shared cash.

Baseline reproduction: the `donchian` variant uses the same warmup handling as
the RS variants (full-frame indicators + momentum lookback window, trading only
inside [start, stop]), so the two are on equal footing. NOTE this differs from
diagnose_cta_exits.py, which computes indicators on the window frame only (and
so loses the first ~14 bars to ATR warmup); the memory baseline (+12.39% total,
be2 +14.96%) is the window-frame number. The warmup-corrected `donchian` here is
reported alongside for apples-to-apples comparison with the RS leg.

Warmup caveat: the cache starts ~2026-06-25, and w4 (Jun26-Jul26) sits right at
that boundary, so RS momentum (which needs `lookback` bars of history) is not
computable at the very start of w4 -- longer lookbacks lose more of w4. Weigh
w3/w1/w2 (clean warmup) over w4.

Run:  python3 scripts/diagnose_cta_rs.py --variant all
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

# Variants: mode=donchian|rs; lookback in 4H bars (18=3d, 42=7d, 84=14d).
VARIANTS = {
    "donchian": {"mode": "donchian"},
    "donchian_be2": {"mode": "donchian", "be_mult": 2.0},
    "rs6": {"mode": "rs", "lookback": 6},
    "rs12": {"mode": "rs", "lookback": 12},
    "rs18": {"mode": "rs", "lookback": 18},
    "rs24": {"mode": "rs", "lookback": 24},
    "rs30": {"mode": "rs", "lookback": 30},
    "rs42": {"mode": "rs", "lookback": 42},
    "rs84": {"mode": "rs", "lookback": 84},
    "rs84_pos": {"mode": "rs", "lookback": 84, "min_mom": 0.0},
    "rs42_k3": {"mode": "rs", "lookback": 42, "max_pos": 3},
    "rs18_k3": {"mode": "rs", "lookback": 18, "max_pos": 3},
    "rs18_be2": {"mode": "rs", "lookback": 18, "be_mult": 2.0},
    "rs84_be2": {"mode": "rs", "lookback": 84, "be_mult": 2.0},
}

MAX_LOOKBACK = max(v.get("lookback", 0) for v in VARIANTS.values())


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


def _momentum(c, lookback):
    mom = np.full(len(c), np.nan)
    if lookback < len(c):
        mom[lookback:] = c[lookback:] / c[:-lookback] - 1.0
    return mom


def run_portfolio(frames, funding_map, total_equity, max_pos, N, M, atr_mult,
                  risk_pct, fee, slip, start, stop, mode="donchian",
                  lookback=42, be_mult=None, min_mom=None):
    inds = {}
    for s, f in frames.items():
        o, h, l, c, atr, upper, exit_lower = _indicators(f, N, M)
        mom = _momentum(c, lookback) if mode == "rs" else None
        inds[s] = (o, h, l, c, atr, upper, exit_lower, mom)
    close_to_idx = {s: {f.index[i] + FOUR_HOURS: i for i in range(len(f))}
                    for s, f in frames.items()}
    all_times = sorted({t for m in close_to_idx.values() for t in m
                        if start <= t <= stop})

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

    def open_pos(s, idx, px, atr_val):
        risk = (equity / max_pos) * risk_pct
        stop = px - atr_mult * atr_val
        qty = risk / (atr_mult * atr_val)
        positions[s] = {
            "entry_px": px, "stop": stop, "qty": qty,
            "entry_time": t, "entry_idx": idx,
            "hi": px, "atr_entry": atr_val,
        }

    for t in all_times:
        # 1. exits (stop -> reverse; breakeven raises stop for the NEXT bar)
        for s in list(positions):
            p = positions[s]
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, atr, upper, exit_lower, mom = inds[s]
            reason = px = None
            if np.isfinite(atr[idx]) and l[idx] <= p["stop"]:
                reason, px = "stop", min(o[idx], p["stop"])
            if reason is None and np.isfinite(exit_lower[idx]) and c[idx] < exit_lower[idx]:
                reason, px = "reverse", c[idx]
            if reason is not None:
                close(s, px, reason, t)
                continue
            if be_mult is not None and np.isfinite(atr[idx]):
                p["hi"] = max(p["hi"], h[idx])
                if p["hi"] >= p["entry_px"] + be_mult * p["atr_entry"]:
                    p["stop"] = max(p["stop"], p["entry_px"])

        # 2. entries (long only)
        slots = max(0, max_pos - len(positions))
        if mode == "donchian":
            cands = []
            for s in frames:
                if s in positions:
                    continue
                idx = close_to_idx[s].get(t)
                if idx is None:
                    continue
                o, h, l, c, atr, upper, exit_lower, mom = inds[s]
                if np.isfinite(upper[idx]) and np.isfinite(atr[idx]) and atr[idx] > 0 \
                        and c[idx] > upper[idx]:
                    strength = (c[idx] - upper[idx]) / atr[idx]
                    cands.append((s, strength, idx, c[idx], atr[idx]))
            cands.sort(key=lambda x: -x[1])
            for s, _strength, idx, px, atr_val in cands[:slots]:
                open_pos(s, idx, px, atr_val)
        else:  # relative strength: rank by trailing momentum, long top-K
            ranked = []
            for s in frames:
                if s in positions:
                    continue
                idx = close_to_idx[s].get(t)
                if idx is None:
                    continue
                o, h, l, c, atr, upper, exit_lower, mom = inds[s]
                m = mom[idx]
                if np.isfinite(m) and np.isfinite(atr[idx]) and atr[idx] > 0 \
                        and (min_mom is None or m > min_mom):
                    ranked.append((s, m, idx, c[idx], atr[idx]))
            ranked.sort(key=lambda x: -x[1])
            for s, _m, idx, px, atr_val in ranked[:slots]:
                open_pos(s, idx, px, atr_val)

        curve.append(equity)

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
    warmup = pd.Timedelta((MAX_LOOKBACK + 20) * 4, unit="h")

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
            f = f.loc[(f.index + FOUR_HOURS >= start - warmup) & (f.index + FOUR_HOURS <= stop)]
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
            mp = opts.get("max_pos", args.max_positions)
            kw = {k: v for k, v in opts.items() if k != "max_pos"}
            trades, curve = run_portfolio(
                frames, funding, args.total_equity, mp,
                N, M, atr_mult, risk_pct, fee, slip, start, stop, **kw)
            st = _stats(trades, curve, args.total_equity)
            row[f"w{w}"] = st
            reasons = " ".join(f"{k}:{v}" for k, v in sorted(st["reasons"].items()))
            print(f"[{vname:<12} w{w} {start.date()}..{stop.date()} {time.perf_counter()-t0:.1f}s] "
                  f"ret={st['ret']:>6.2f}%  n={st['n']:>3}  win={st['win']:>3.0f}%  "
                  f"PF={st['pf']:>5.2f}  maxDD={st['dd']:>5.2f}%  [{reasons}]", flush=True)
        rows.append(row)

    print("\n=== summary (long-only, funding incl.) ===", flush=True)
    order = [k for k in ("w4", "w3", "w1", "w2") if k in rows[0]]
    print(f"{'variant':<13} " + " ".join(f"{k.upper():>8}" for k in order) + f"  {'total':>8}  "
          + " ".join(f"{k+'dd':>6}" for k in order), flush=True)
    for row in rows:
        vals = [row[k]["ret"] for k in order]
        dds = [row[k]["dd"] for k in order]
        tot = sum(vals)
        print(f"{row['variant']:<13} " + " ".join(f"{v:>7.2f}%" for v in vals) + f"  {tot:>7.2f}%  "
              + " ".join(f"{d:>5.2f}%" for d in dds), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
