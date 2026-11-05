"""Trend-aligned mean-reversion A/B backtest (Route A vs B), self-contained.

Strategy under test:
  trend gate (daily) = close vs EMA50(1D) and 3-day EMA50 slope direction,
  optional 4H confirmed-RANGE gate (Route A only),
  trigger = 1H close deviates from 1H EMA50 by k*ATR14 with a same-direction 1H bar,
  exit   = stop 2.0 ATR + breakeven + trailing stop + no-progress + max-hold.
           base:    BE at +1.0 ATR, trail 0.8 ATR after +0.8 ATR
           relaxed: BE at +1.5 ATR, trail 2.0 ATR after +2.0 ATR

Portfolio level, shared $10k cash, max_pos=5, risk-based sizing (1% of slot),
long/short symmetric, taker 5bp + slippage 3bp, real funding from the pickle.
(Original MR TP1/TP2 partial exits are deferred to a later exit-variant pass.)

Also prints the CTA Donchian long-only baseline (N=20/M=10/2 ATR stop) on the
same window for a same-frame comparison.

Run:
  python scripts/diagnose_mr_trend_ab.py --window 2 --window-days 30
  python scripts/diagnose_mr_trend_ab.py --window all --window-days 30
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

DB = "data/market.sqlite"
FUNDING = "data/ab_funding.pkl"
DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]
FOUR_HOURS = pd.Timedelta(4, unit="h")
FEE = 5 / 10_000
SLIP = 3 / 10_000


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = frame["high"], frame["low"], frame["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = frame["high"], frame["low"], frame["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    up = c.diff()
    dn = -c.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=frame.index)
    a0 = a.replace(0, np.nan)
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / a0
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / a0
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def structure_states(frame: pd.DataFrame, lookback: int) -> pd.Series:
    h, l = frame["high"], frame["low"]
    rh = h.rolling(lookback, min_periods=lookback).max()
    ph = h.shift(lookback).rolling(lookback, min_periods=lookback).max()
    rl = l.rolling(lookback, min_periods=lookback).min()
    pl = l.shift(lookback).rolling(lookback, min_periods=lookback).min()
    hh, lh, hl, ll = rh > ph, rh < ph, rl > pl, rl < pl
    st = pd.Series("NEUTRAL", index=frame.index, dtype=object)
    st[hh & hl] = "UP"
    st[lh & ll] = "DOWN"
    st[(hh & ll) | (lh & hl)] = "CONFLICT"
    return st


def range_confirmed(frame: pd.DataFrame, mode: str = "not_conflict") -> pd.Series:
    e50 = ema(frame["close"], 50)
    slope = e50 / e50.shift(3) - 1
    dist = frame["close"] / e50 - 1
    struct = structure_states(frame, 4)
    ok = (adx(frame, 14) <= 18) & (slope.abs() <= 0.0005) & (dist.abs() <= 0.015)
    if mode == "not_conflict":
        ok = ok & struct.isin(["NEUTRAL", "UP", "DOWN"])
    else:
        ok = ok & (struct == "NEUTRAL")
    return (ok.rolling(2).sum() == 2)


def daily_regime(frame: pd.DataFrame) -> pd.Series:
    daily = frame.resample("24h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    e = ema(daily["close"], 50)
    slope = e / e.shift(3) - 1
    out = pd.Series("NEUTRAL", index=daily.index, dtype=object)
    out[(daily["close"] > e) & (slope > 0)] = "UP"
    out[(daily["close"] < e) & (slope < 0)] = "DOWN"
    return out


def load(db: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    con = sqlite3.connect(db, timeout=30)
    b4, b1 = {}, {}
    for s in DEFAULT_SYMBOLS:
        for tf, target in (("4h", b4), ("1h", b1)):
            df = pd.read_sql_query(
                "SELECT timestamp_ms, open, high, low, close, volume FROM klines "
                "WHERE symbol=? AND timeframe=? ORDER BY timestamp_ms",
                con, params=(s, tf),
            )
            df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.set_index("ts").drop(columns=["timestamp_ms"])
            target[s] = df
    con.close()
    return b4, b1


def build_signals(b4: dict[str, pd.DataFrame], b1: dict[str, pd.DataFrame], k: float):
    """Per-symbol series aligned to 4H bar index (open ts)."""
    sig = {}
    for s in b4:
        f4 = b4[s]
        daily = daily_regime(f4)
        close_times = f4.index + FOUR_HOURS
        # use only the PREVIOUS completed daily bar: a daily bar labelled D
        # covers D 00:00..D 20:00 and closes at D+1 00:00, so at a 4H close
        # time t it is not yet complete — shift(1) avoids the look-ahead.
        mapped = daily.shift(1).reindex(close_times, method="ffill")
        mapped.index = f4.index
        rng = range_confirmed(f4, "not_conflict").fillna(False)
        f1 = b1[s]
        e50 = ema(f1["close"], 50)
        a = atr(f1, 14)
        dev = f1["close"] / e50 - 1
        thr = k * a / e50
        long_t = (dev <= -thr) & (f1["close"] > f1["open"])
        short_t = (dev >= thr) & (f1["close"] < f1["open"])
        bar_open = (f1.index + pd.Timedelta(1, unit="h")).floor("4h") - FOUR_HOURS

        def agg(flag: pd.Series) -> pd.Series:
            s_ = pd.Series(flag.astype(int).values, index=bar_open)
            return s_.groupby(s_.index).sum().reindex(f4.index).fillna(0).astype(int)

        sig[s] = {
            "regime": mapped.fillna("NEUTRAL"),
            "range": rng,
            "long_trigger": agg(long_t),
            "short_trigger": agg(short_t),
            "atr4h": atr(f4, 14),
        }
    return sig


def run_engine(frames, signals, funding, equity0, max_pos, variant, exit_mode, risk_pct=0.01):
    close_to_idx = {s: {f.index[i] + FOUR_HOURS: i for i in range(len(f))} for s, f in frames.items()}
    all_times = sorted({t for m in close_to_idx.values() for t in m})
    equity = equity0
    positions = {}
    trades = []
    curve = []
    funnel = {"universe": 0, "trend_pass": 0, "range_pass": 0, "trigger": 0, "filled": 0}
    cut_long = cut_short = 0
    cut_candidates = []  # (symbol, side, idx) blocked by the trend gate

    if exit_mode == "base":
        be_r, trail_after, trail_mult = 1.0, 0.8, 0.8
    else:
        be_r, trail_after, trail_mult = 1.5, 2.0, 2.0

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
        # exits
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
            sign = 1.0 if p["side"] == "LONG" else -1.0
            if p["side"] == "LONG":
                if l <= p["stop"]:
                    realize(s, min(o, p["stop"]), "stop", t); continue
                r = (h - p["entry_px"]) / p["entry_atr"]
                if r >= be_r:
                    p["stop"] = max(p["stop"], p["entry_px"])
                p["best_high"] = max(p["best_high"], h)
                if r >= trail_after:
                    p["stop"] = max(p["stop"], p["best_high"] - trail_mult * atr_val)
            else:
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
                prog = (c - p["entry_px"]) * sign / p["entry_atr"]
                if prog < 0.4:
                    realize(s, c, "no_progress", t); continue
            if bars_held >= 24:
                realize(s, c, "max_hold", t); continue
        # entries
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            sig = signals[s]
            ts = frames[s].index[idx]
            funnel["universe"] += 1
            regime = sig["regime"].loc[ts]
            ltrig = int(sig["long_trigger"].loc[ts])
            strig = int(sig["short_trigger"].loc[ts])
            if regime not in ("UP", "DOWN"):
                if ltrig:
                    cut_long += 1
                    cut_candidates.append((s, "LONG", idx))
                if strig:
                    cut_short += 1
                    cut_candidates.append((s, "SHORT", idx))
                continue
            funnel["trend_pass"] += 1
            if variant == "A" and not bool(sig["range"].loc[ts]):
                continue
            funnel["range_pass"] += 1
            if regime == "UP" and ltrig:
                funnel["trigger"] += 1
                cands.append((s, "LONG", idx, float(frames[s]["close"].iloc[idx]),
                              float(sig["atr4h"].loc[ts])))
            elif regime == "DOWN" and strig:
                funnel["trigger"] += 1
                cands.append((s, "SHORT", idx, float(frames[s]["close"].iloc[idx]),
                              float(sig["atr4h"].loc[ts])))
        for s, side, idx, px, atr_val in cands[: max(0, max_pos - len(positions))]:
            slot = equity / max_pos
            risk = slot * risk_pct
            stop = px - 2.0 * atr_val if side == "LONG" else px + 2.0 * atr_val
            qty = risk / (2.0 * atr_val)
            positions[s] = {"side": side, "entry_px": px, "stop": stop, "qty": qty,
                            "entry_atr": atr_val, "entry_time": t, "entry_idx": idx,
                            "best_high": px, "best_low": px}
            funnel["filled"] += 1
        curve.append(equity)

    for s in list(positions):
        p = positions[s]
        idx = len(frames[s]) - 1
        realize(s, float(frames[s]["close"].iloc[idx]), "end", frames[s].index[idx] + FOUR_HOURS)
    cut_pnl = {"LONG": 0.0, "SHORT": 0.0}
    for s, side, idx in cut_candidates:
        pnl, _ = counterfactual_pnl(frames, signals, funding, s, side, idx, exit_mode)
        cut_pnl[side] += pnl
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times)), funnel, (cut_long, cut_short), cut_pnl


def _maxdd(curve):
    peak = curve.cummax()
    return float(((curve - peak) / peak).min() * 100)


def counterfactual_pnl(frames, signals, funding, symbol, side, idx, exit_mode):
    """Net PnL of a trade the trend gate would otherwise have blocked, using the
    same exit rules as run_engine (approximation: runs independently, no cap
    interaction)."""
    f = frames[symbol]
    sig = signals[symbol]
    entry_px = float(f["close"].iloc[idx])
    entry_atr = float(sig["atr4h"].loc[f.index[idx]])
    if not np.isfinite(entry_atr) or entry_atr <= 0:
        return 0.0, "bad_atr"
    if exit_mode == "base":
        be_r, trail_after, trail_mult = 1.0, 0.8, 0.8
    else:
        be_r, trail_after, trail_mult = 1.5, 2.0, 2.0
    sign = 1.0 if side == "LONG" else -1.0
    stop = entry_px - 2.0 * entry_atr if side == "LONG" else entry_px + 2.0 * entry_atr
    qty = 1.0 / (2.0 * entry_atr)  # normalized to $1 risk, report per-$1-risk
    best_hi, best_lo = entry_px, entry_px
    entry_time = f.index[idx] + FOUR_HOURS
    reason = "end"
    exit_px = entry_px
    for j in range(idx + 1, len(f)):
        o, h, l, c = f["open"].iloc[j], f["high"].iloc[j], f["low"].iloc[j], f["close"].iloc[j]
        a = float(sig["atr4h"].loc[f.index[j]])
        if not np.isfinite(a) or a <= 0:
            continue
        t = f.index[j] + FOUR_HOURS
        if side == "LONG":
            if l <= stop:
                exit_px, reason = min(o, stop), "stop"; break
            r = (h - entry_px) / entry_atr
            if r >= be_r:
                stop = max(stop, entry_px)
            best_hi = max(best_hi, h)
            if r >= trail_after:
                stop = max(stop, best_hi - trail_mult * a)
        else:
            if h >= stop:
                exit_px, reason = max(o, stop), "stop"; break
            r = (entry_px - l) / entry_atr
            if r >= be_r:
                stop = min(stop, entry_px)
            best_lo = min(best_lo, l)
            if r >= trail_after:
                stop = min(stop, best_lo + trail_mult * a)
        bars_held = j - idx
        if bars_held >= 6:
            prog = (c - entry_px) * sign / entry_atr
            if prog < 0.4:
                exit_px, reason = c, "no_progress"; break
        if bars_held >= 24:
            exit_px, reason = c, "max_hold"; break
    entry_fill = entry_px * (1 + SLIP)
    exit_fill = exit_px * (1 - SLIP)
    gross = (exit_fill - entry_fill) * qty * sign
    fees = qty * (entry_fill + exit_fill) * FEE
    fund = funding.get(symbol)
    fc = 0.0
    if fund is not None and not fund.empty:
        sub = fund.loc[(fund.index > entry_time) & (fund.index <= f.index[len(f) - 1] + FOUR_HOURS)]
        for _, row in sub.iterrows():
            mp = float(row.get("mark_price", 0.0) or 0.0)
            if mp > 0:
                fc += qty * mp * float(row["funding_rate"]) * sign
    return gross - fees - fc, reason


def stats(trades, equity0, curve):
    net = sum(t["net"] for t in trades)
    wins = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] <= 0]
    gw = sum(t["net"] for t in wins)
    gl = -sum(t["net"] for t in losses)
    pf = gw / gl if gl > 0 else float("inf")
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    return {
        "net": net, "ret": net / equity0 * 100, "n": len(trades),
        "win": len(wins) / len(trades) * 100 if trades else 0,
        "pf": pf, "maxdd": _maxdd(curve),
        "long_n": len(longs), "long_net": sum(t["net"] for t in longs),
        "short_n": len(shorts), "short_net": sum(t["net"] for t in shorts),
    }


def run_cta_baseline(frames, funding, equity0, max_pos, N=20, M=10, atr_mult=2.0, risk_pct=0.01):
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
        cands = []
        for s in frames:
            if s in positions:
                continue
            idx = close_to_idx[s].get(t)
            if idx is None:
                continue
            o, h, l, c, a, upper, exit_lower = ind[s]
            if np.isfinite(upper[idx]) and np.isfinite(a[idx]) and a[idx] > 0 and c[idx] > upper[idx]:
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
    return trades, pd.Series(curve, index=pd.DatetimeIndex(all_times)), None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", required=True, choices=("1", "2", "3", "4", "5", "all"))
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--funding", default=FUNDING)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--k-list", default="1.0,1.5,2.0")
    ap.add_argument("--skip-cta", action="store_true")
    args = ap.parse_args()

    with open(args.funding, "rb") as fh:
        funding = pickle.load(fh)
    b4, b1 = load(args.db)
    end = pd.Timestamp.utcnow().floor("h")
    windows = [int(w) for w in (("1", "2", "3") if args.window == "all" else (args.window,))]
    ks = [float(x) for x in args.k_list.split(",")]

    for w in windows:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * args.window_days), end - pd.Timedelta(days=args.window_days)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=args.window_days), end
        elif w == 3:
            start, stop = end - pd.Timedelta(days=3 * args.window_days), end - pd.Timedelta(days=2 * args.window_days)
        elif w == 4:
            start, stop = end - pd.Timedelta(days=4 * args.window_days), end - pd.Timedelta(days=3 * args.window_days)
        else:
            start, stop = end - pd.Timedelta(days=5 * args.window_days), end - pd.Timedelta(days=4 * args.window_days)
        frames = {}
        for s in b4:
            f = b4[s].loc[(b4[s].index + FOUR_HOURS >= start) & (b4[s].index + FOUR_HOURS <= stop)]
            if len(f) >= 80:
                frames[s] = f
        print(f"\n=== window {w}  {start.date()}..{stop.date()}  symbols={len(frames)}  max_pos={args.max_positions} ===", flush=True)

        for k in ks:
            sig = build_signals(b4, b1, k)
            for variant in ("A", "B"):
                for emode in ("base", "relaxed"):
                    trades, curve, funnel, cuts, cut_pnl = run_engine(frames, sig, funding, args.equity,
                                                                       args.max_positions, variant, emode)
                    st = stats(trades, args.equity, curve)
                    print(f"  [{variant} k={k} {emode}] net={st['net']:7.2f} ({st['ret']:6.2f}%) "
                          f"n={st['n']:3d} win={st['win']:4.0f}% PF={st['pf']:5.2f} DD={st['maxdd']:5.2f}% "
                          f"L={st['long_n']}/{st['long_net']:.0f} S={st['short_n']}/{st['short_net']:.0f} "
                          f"fill={funnel['filled']}/{funnel['trigger']} "
                          f"cutL={cuts[0]}({cut_pnl['LONG']:.0f}) cutS={cuts[1]}({cut_pnl['SHORT']:.0f})", flush=True)
        if not args.skip_cta:
            trades, curve, _, _ = run_cta_baseline(frames, funding, args.equity, args.max_positions)
            st = stats(trades, args.equity, curve)
            print(f"  [CTA] net={st['net']:7.2f} ({st['ret']:6.2f}%) n={st['n']:3d} "
                  f"win={st['win']:4.0f}% PF={st['pf']:5.2f} DD={st['maxdd']:5.2f}% "
                  f"L={st['long_n']}/{st['long_net']:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
