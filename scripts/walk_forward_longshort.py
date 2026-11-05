"""Walk-forward validation of a FROZEN long/short rule space.

Pre-registered: 7 bidirectional signals x ONE fixed exit scheme. No parameter
search — every number below is fixed in advance. Windows are the five
consecutive 60-day slices in chronological order (w5..w2); a signal "survives"
only if it is net-positive in ALL five forward windows it never influenced.

Exit (identical for all): 2x ATR14(4H) protective stop, reverse M=10 breakout,
24x4H max hold. Sizing: 1% risk per slot, max_pos=5, taker 5bp + slip 3bp,
real funding (correct sign: shorts receive positive funding).

Run: python scripts/walk_forward_longshort.py
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

from diagnose_mr_trend_ab import (DEFAULT_SYMBOLS, FEE, FOUR_HOURS, SLIP,
                                  atr as _atr, daily_regime, ema, load, stats)

DB = "data/market.sqlite"
FUNDING = "data/ab_funding.pkl"
RISK_PCT = 0.01
STOP_ATR = 2.0
REV_BARS = 10
MAX_HOLD = 24


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_signal_series(b4, b1):
    """Return sig[signal][symbol] = dict(long=Series, short=Series) on the full 4H index."""
    names = ["donchian", "momentum", "ma_cross", "ema_slope", "trend_pullback",
             "rsi_reversion", "breakout_breadth"]
    sig = {n: {} for n in names}
    above = pd.DataFrame({s: b4[s]["close"] > ema(b4[s]["close"], 50) for s in b4})
    breadth = above.mean(axis=1, skipna=True)

    for s in b4:
        f = b4[s]
        h = f["high"]; l = f["low"]; c = f["close"]
        prev_h = h.rolling(20).max().shift(1)
        prev_l = l.rolling(20).min().shift(1)
        e20 = ema(c, 20); e50 = ema(c, 50)
        slope3 = e50 / e50.shift(3) - 1
        ret12 = c / c.shift(12) - 1
        r = rsi(c, 14)
        sig["donchian"][s] = {"long": c > prev_h, "short": c < prev_l}
        sig["momentum"][s] = {"long": ret12 > 0.04, "short": ret12 < -0.04}
        cross_up = (e20 > e50) & (e20.shift(1) <= e50.shift(1))
        cross_dn = (e20 < e50) & (e20.shift(1) >= e50.shift(1))
        sig["ma_cross"][s] = {"long": cross_up, "short": cross_dn}
        sig["ema_slope"][s] = {"long": (c > e50) & (slope3 > 0), "short": (c < e50) & (slope3 < 0)}
        sig["rsi_reversion"][s] = {"long": r < 35, "short": r > 65}
        sig["breakout_breadth"][s] = {
            "long": (c > prev_h) & (breadth > 0.6),
            "short": (c < prev_l) & (breadth < 0.4),
        }
        f1 = b1[s]
        e50_1h = ema(f1["close"], 50)
        a1 = _atr(f1, 14)
        dev = f1["close"] / e50_1h - 1
        thr = 1.5 * a1 / e50_1h
        long_1h = (dev <= -thr) & (f1["close"] > f1["open"])
        short_1h = (dev >= thr) & (f1["close"] < f1["open"])
        bar_open = (f1.index + pd.Timedelta(1, unit="h")).floor("4h") - FOUR_HOURS

        def agg(flag):
            s_ = pd.Series(flag.astype(int).values, index=bar_open)
            return s_.groupby(s_.index).sum().reindex(f.index).fillna(0).astype(bool)

        daily = daily_regime(f)
        # use the PREVIOUS completed daily bar only: a daily bar labelled D
        # aggregates 4H bars up to D 20:00 and closes at D+1 00:00, so at a 4H
        # close time t it is not yet complete — shift(1) avoids look-ahead.
        regime = daily.shift(1).reindex(f.index + FOUR_HOURS, method="ffill")
        regime.index = f.index
        sig["trend_pullback"][s] = {
            "long": (regime == "UP") & agg(long_1h),
            "short": (regime == "DOWN") & agg(short_1h),
        }
    return sig, names


def run_bidirectional(frames, long_arr, short_arr, atr_arr, exit_lower, exit_upper,
                      funding, equity0, max_pos):
    close_to_idx = {s: {f.index[i] + FOUR_HOURS: i for i in range(len(f))} for s, f in frames.items()}
    all_times = sorted({t for m in close_to_idx.values() for t in m})
    equity = equity0
    positions = {}
    trades = []
    curve = []

    def realize(s, px, reason, t):
        nonlocal equity
        p = positions.pop(s)
        sign = 1.0 if p["side"] == "LONG" else -1.0
        entry_fill = p["entry_px"] * (1 + SLIP)
        exit_fill = px * (1 - SLIP)
        gross = (exit_fill - entry_fill) * p["qty"] * sign
        fees = p["qty"] * (entry_fill + exit_fill) * FEE
        fund = funding.get(s)
        fc = 0.0
        if fund is not None and not fund.empty:
            sub = fund.loc[(fund.index > p["entry_time"]) & (fund.index <= t)]
            for _, row in sub.iterrows():
                mp = float(row.get("mark_price", 0.0) or 0.0)
                if mp > 0:
                    fc += p["qty"] * mp * float(row["funding_rate"]) * sign
        net = gross - fees - fc
        equity += net
        trades.append({"symbol": s, "side": p["side"], "reason": reason, "net": net})

    for t in all_times:
        for s in list(positions):
            p = positions[s]
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            f = frames[s]
            o, h, l, c = (f["open"].iloc[idx], f["high"].iloc[idx],
                          f["low"].iloc[idx], f["close"].iloc[idx])
            a = atr_arr[s][idx]
            if not np.isfinite(a) or a <= 0:
                continue
            if p["side"] == "LONG":
                if l <= p["stop"]:
                    realize(s, min(o, p["stop"]), "stop", t); continue
                if np.isfinite(exit_lower[s][idx]) and c < exit_lower[s][idx]:
                    realize(s, c, "reverse", t); continue
            else:
                if h >= p["stop"]:
                    realize(s, max(o, p["stop"]), "stop", t); continue
                if np.isfinite(exit_upper[s][idx]) and c > exit_upper[s][idx]:
                    realize(s, c, "reverse", t); continue
            if idx - p["entry_idx"] >= MAX_HOLD:
                realize(s, c, "max_hold", t); continue
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            lng = long_arr[s][idx]
            sht = short_arr[s][idx]
            if lng and sht:
                continue
            a = atr_arr[s][idx]
            c = float(frames[s]["close"].iloc[idx])
            if not np.isfinite(a) or a <= 0:
                continue
            if lng:
                cands.append((s, "LONG", idx, c, a))
            elif sht:
                cands.append((s, "SHORT", idx, c, a))
        for s, side, idx, px, a in cands[: max(0, max_pos - len(positions))]:
            slot = equity / max_pos
            risk = slot * RISK_PCT
            stop = px - STOP_ATR * a if side == "LONG" else px + STOP_ATR * a
            qty = risk / (STOP_ATR * a)
            positions[s] = {"side": side, "entry_px": px, "stop": stop, "qty": qty,
                            "entry_time": t, "entry_idx": idx}
        curve.append(equity)

    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        realize(s, float(frames[s]["close"].iloc[idx]), "end", frames[s].index[idx] + FOUR_HOURS)
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times))


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
    sig, names = compute_signal_series(b4, b1)

    atr_map = {s: _atr(b4[s], 14) for s in b4}
    exit_lower_map = {s: b4[s]["low"].rolling(REV_BARS).min().shift(1) for s in b4}
    exit_upper_map = {s: b4[s]["high"].rolling(REV_BARS).max().shift(1) for s in b4}

    end = pd.Timestamp.utcnow().floor("h")
    d = args.window_days
    windows = [
        ("w5", end - pd.Timedelta(days=5 * d), end - pd.Timedelta(days=4 * d)),
        ("w4", end - pd.Timedelta(days=4 * d), end - pd.Timedelta(days=3 * d)),
        ("w3", end - pd.Timedelta(days=3 * d), end - pd.Timedelta(days=2 * d)),
        ("w1", end - pd.Timedelta(days=2 * d), end - pd.Timedelta(days=d)),
        ("w2", end - pd.Timedelta(days=d), end),
    ]

    results = {n: {} for n in names}
    full_results = {}

    for label, start, stop in windows:
        frames = {}
        for s in b4:
            f = b4[s].loc[(b4[s].index + FOUR_HOURS >= start) & (b4[s].index + FOUR_HOURS <= stop)]
            if len(f) >= 80:
                frames[s] = f
        idx_per = {s: frames[s].index for s in frames}
        for n in names:
            long_arr = {s: sig[n][s]["long"].reindex(idx_per[s]).fillna(False).to_numpy() for s in frames}
            short_arr = {s: sig[n][s]["short"].reindex(idx_per[s]).fillna(False).to_numpy() for s in frames}
            a_arr = {s: atr_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
            el = {s: exit_lower_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
            eu = {s: exit_upper_map[s].reindex(idx_per[s]).to_numpy() for s in frames}
            trades, curve = run_bidirectional(frames, long_arr, short_arr, a_arr, el, eu,
                                              funding, args.equity, args.max_positions)
            results[n][label] = stats(trades, args.equity, curve)

    idx_per = {s: b4[s].index for s in b4}
    for n in names:
        long_arr = {s: sig[n][s]["long"].reindex(idx_per[s]).fillna(False).to_numpy() for s in b4}
        short_arr = {s: sig[n][s]["short"].reindex(idx_per[s]).fillna(False).to_numpy() for s in b4}
        a_arr = {s: atr_map[s].reindex(idx_per[s]).to_numpy() for s in b4}
        el = {s: exit_lower_map[s].reindex(idx_per[s]).to_numpy() for s in b4}
        eu = {s: exit_upper_map[s].reindex(idx_per[s]).to_numpy() for s in b4}
        trades, curve = run_bidirectional(b4, long_arr, short_arr, a_arr, el, eu,
                                          funding, args.equity, args.max_positions)
        full_results[n] = stats(trades, args.equity, curve)

    print(f"\n{'signal':<16}" + "".join(f"{w:>9}" for w in ["w5", "w4", "w3", "w1", "w2"]) + f"{'full':>9}  verdict")
    print("-" * 90)
    for n in names:
        cells = [results[n][w] for w in ["w5", "w4", "w3", "w1", "w2"]]
        rets = [c["ret"] for c in cells]
        row = f"{n:<16}" + "".join(f"{r:>8.2f}%" for r in rets) + f"{full_results[n]['ret']:>8.2f}%"
        verdict = "PASS(all>0)" if all(r > 0 for r in rets) else ("pos " + str(sum(r > 0 for r in rets)) + "/5")
        print(f"{row}  {verdict}")
    print("\nNOTE: % = net return on $10k; real funding (correct short sign); frozen rules, no tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
