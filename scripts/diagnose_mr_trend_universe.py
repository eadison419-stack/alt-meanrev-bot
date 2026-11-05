"""Pre-check: quantify the 1D-trend gate x 4H-RANGE gate intersection.

Answers the two gating numbers BEFORE writing any strategy code:

  1. What fraction of 4H bars fall in each daily regime (UP / DOWN / NEUTRAL)?
  2. How many candidate mean-reversion entries survive, for
       Route A = daily trend gate  AND  4H confirmed-RANGE,
       Route B = daily trend gate alone?

If Route A produces < 30 combined candidates over the ~400-day sample, it is
declared dead per the agreed kill-line and we go Route-B-only.

Trend gate (daily, resampled from 4H at UTC day boundaries):
  UP   = close > EMA50  and  EMA50 3-day slope > 0
  DOWN = close < EMA50  and  EMA50 3-day slope < 0
  else NEUTRAL (no trade)

4H RANGE gate: replicate the production classifier's raw RANGE candidate with
structure mode `not_conflict` (the validated unlock), then require the last 2
raw candidates to both be RANGE (range_confirm_bars=2), matching
classify_symbol_regime. A second line reports the strict `neutral` mode for
contrast.

Trigger proxy: 1H close deviates from 1H EMA50 by >= k * ATR14(1H), with a
bullish 1H bar for longs and a bearish bar for shorts (k=1.5 default).

Self-contained: stdlib + pandas + numpy only, reads data/market.sqlite directly.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

DB = "data/market.sqlite"
K = 1.5  # ATR multiple for the deviation trigger


def load_4h(db: str) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(db, timeout=30)
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM klines WHERE timeframe='4h' ORDER BY symbol"
    ).fetchall()]
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = pd.read_sql_query(
            "SELECT timestamp_ms, open, high, low, close, volume FROM klines "
            "WHERE symbol=? AND timeframe='4h' ORDER BY timestamp_ms",
            con, params=(s,),
        )
        df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=["timestamp_ms"])
        out[s] = df
    con.close()
    return out


def load_1h(db: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(db, timeout=30)
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = pd.read_sql_query(
            "SELECT timestamp_ms, open, high, low, close, volume FROM klines "
            "WHERE symbol=? AND timeframe='1h' ORDER BY timestamp_ms",
            con, params=(s,),
        )
        df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=["timestamp_ms"])
        out[s] = df
    con.close()
    return out


def adx_series(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    up = close.diff()
    down = -close.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    atr_s = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_series(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def structure_states(frame: pd.DataFrame, lookback: int) -> pd.Series:
    high, low = frame["high"], frame["low"]
    recent_high = high.rolling(lookback, min_periods=lookback).max()
    prior_high = high.shift(lookback).rolling(lookback, min_periods=lookback).max()
    recent_low = low.rolling(lookback, min_periods=lookback).min()
    prior_low = low.shift(lookback).rolling(lookback, min_periods=lookback).min()
    hh = recent_high > prior_high
    lh = recent_high < prior_high
    hl = recent_low > prior_low
    ll = recent_low < prior_low
    states = pd.Series("NEUTRAL", index=frame.index, dtype=object)
    states[hh & hl] = "UP"
    states[lh & ll] = "DOWN"
    states[(hh & ll) | (lh & hl)] = "CONFLICT"
    return states


def range_raw_candidates(frame: pd.DataFrame, mode: str) -> pd.Series:
    """Replicate _raw_candidates' RANGE flag (not_conflict or neutral)."""
    adx = adx_series(frame, 14)
    ema_slow = ema(frame["close"], 50)
    slope = ema_slow / ema_slow.shift(3) - 1
    dist = frame["close"] / ema_slow - 1
    struct = structure_states(frame, 4)
    range_adx = adx <= 18
    range_slope = slope.abs() <= 0.0005
    range_dist = dist.abs() <= 0.015
    if mode == "not_conflict":
        range_struct = struct.isin(["NEUTRAL", "UP", "DOWN"])
    else:
        range_struct = struct == "NEUTRAL"
    return range_adx & range_slope & range_dist & range_struct


def daily_trend(frame: pd.DataFrame) -> pd.Series:
    daily = frame.resample("24h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    e = ema(daily["close"], 50)
    slope = e / e.shift(3) - 1
    out = pd.Series("NEUTRAL", index=daily.index, dtype=object)
    out[(daily["close"] > e) & (slope > 0)] = "UP"
    out[(daily["close"] < e) & (slope < 0)] = "DOWN"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--k", type=float, default=K)
    args = ap.parse_args()

    bars4 = load_4h(args.db)
    bars1 = load_1h(args.db, list(bars4))

    print(f"symbols: {len(bars4)}")
    print(f"{'symbol':<12}{'4h_bars':>8}{'1h_bars':>8}{'d_UP%':>8}{'d_DOWN%':>8}{'A_4h%':>8}{'B_4h%':>8}{'A_trig':>8}{'B_trig':>8}")
    totals = {"4h": 0, "1h": 0, "up": 0, "down": 0, "A": 0, "B": 0, "Atrig": 0, "Btrig": 0}
    per_symbol = []
    for s, f4 in bars4.items():
        f1 = bars1[s]
        dt = daily_trend(f4)
        # map each 4h bar to the daily regime of the most recent COMPLETED daily
        # bar. A daily bar labelled D aggregates 4H bars up to D 20:00 and only
        # closes at D+1 00:00, so at a 4H close time t it is not yet complete —
        # shift(1) avoids the look-ahead (matches walk_forward_longshort).
        close_times = f4.index + pd.Timedelta(4, unit="h")
        regime = dt.shift(1).reindex(close_times, method="ffill")
        regime.index = f4.index
        regime = regime.fillna("NEUTRAL")
        up = (regime == "UP").sum()
        down = (regime == "DOWN").sum()
        neu = (regime == "NEUTRAL").sum()
        r_nc = range_raw_candidates(f4, "not_conflict")
        r_neutral = range_raw_candidates(f4, "neutral")
        # confirmed RANGE: last 2 raw candidates both RANGE
        confirmed_nc = r_nc.rolling(2).sum() == 2
        # route gates per 4h bar
        A_mask = confirmed_nc & regime.isin(["UP", "DOWN"])
        B_mask = regime.isin(["UP", "DOWN"])
        # trigger proxy on 1h
        e50_1h = ema(f1["close"], 50)
        atr_1h = atr_series(f1, 14)
        dev = f1["close"] / e50_1h - 1
        bullish = f1["close"] > f1["open"]
        bearish = f1["close"] < f1["open"]
        long_trig = (dev <= -args.k * atr_1h / e50_1h) & bullish
        short_trig = (dev >= args.k * atr_1h / e50_1h) & bearish
        trig_1h = (long_trig | short_trig).astype(int)
        # attribute each 1h trigger to its 4h bar, then sum where gate holds
        trig_4h = f1.index.floor("4h")
        tri = pd.Series(trig_1h.values, index=trig_4h)
        tri = tri.groupby(tri.index).sum()
        A_trig = int(tri.reindex(A_mask.index).fillna(0)[A_mask].sum())
        B_trig = int(tri.reindex(B_mask.index).fillna(0)[B_mask].sum())

        n4 = len(f4); n1 = len(f1)
        up_pct = 100 * up / n4 if n4 else 0
        down_pct = 100 * down / n4 if n4 else 0
        A_pct = 100 * A_mask.sum() / n4 if n4 else 0
        B_pct = 100 * B_mask.sum() / n4 if n4 else 0
        print(f"{s:<12}{n4:>8}{n1:>8}{up_pct:>7.1f}%{down_pct:>7.1f}%{A_pct:>7.2f}%{B_pct:>7.2f}%{A_trig:>8}{B_trig:>8}")
        per_symbol.append((s, A_trig, B_trig))
        totals["4h"] += n4; totals["1h"] += n1
        totals["up"] += up; totals["down"] += down
        totals["A"] += int(A_mask.sum()); totals["B"] += int(B_mask.sum())
        totals["Atrig"] += A_trig; totals["Btrig"] += B_trig

    print("-" * 88)
    print(f"{'TOTAL':<12}{totals['4h']:>8}{totals['1h']:>8}"
          f"{100*totals['up']/totals['4h']:>7.1f}%{100*totals['down']/totals['4h']:>7.1f}%"
          f"{100*totals['A']/totals['4h']:>7.2f}%{100*totals['B']/totals['4h']:>7.2f}%"
          f"{totals['Atrig']:>8}{totals['Btrig']:>8}")
    print("\nKill-line (Route A < 30 combined triggers => dead):")
    print(f"  Route A combined triggers = {totals['Atrig']}  -> {'DEAD, go Route-B-only' if totals['Atrig'] < 30 else 'viable'}")
    print(f"  Route B combined triggers = {totals['Btrig']}")
    print("\nTop Route-A symbols by triggers:")
    for s, a, b in sorted(per_symbol, key=lambda x: -x[1])[:10]:
        print(f"  {s:<12} A={a}  B={b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
