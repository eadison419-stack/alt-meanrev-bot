"""CTA trend-following prototype: Donchian breakout vs the confirm+pullback TREND engine.

First-pass result (per-symbol $10k each, 30 symbols = $300k) showed breakout long
edge is clean (W3 long PF 5.26) but shorts still bleed and downtrends still lose —
same long-beta structure as the current engine. This version normalizes capital to
a single --total-equity (spread across symbols, default $10k so it is comparable to
the current TREND engine), fixes the maxDD metric, and adds --long-only to isolate
the long breakout edge (shorts were a drag in all three windows).

Textbook params (NOT tuned): entry breakout N=20 (4H), exit reverse M=10, stop 2x
ATR(14), 1% risk/trade, taker 5bp + slippage 3bp, funding ignored.

Run:  python scripts/diagnose_cta_proto.py --window all
      python scripts/diagnose_cta_proto.py --window all --long-only
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

from run_range_ab import DEFAULT_SYMBOLS

FOUR_HOURS = pd.Timedelta(4, unit="h")


def run_symbol(frame: pd.DataFrame, initial_equity: float, N: int, M: int,
               atr_mult: float, risk_pct: float, fee: float, slip: float,
               long_only: bool, allow_entry=None, funding=None) -> tuple[list[dict], list[float]]:
    o = frame["open"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    n = len(frame)

    prev_c = np.empty(n); prev_c[0] = o[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()

    upper = pd.Series(h).rolling(N).max().shift(1).to_numpy()
    lower = pd.Series(l).rolling(N).min().shift(1).to_numpy()
    exit_lower = pd.Series(l).rolling(M).min().shift(1).to_numpy()
    exit_upper = pd.Series(h).rolling(M).max().shift(1).to_numpy()

    equity = initial_equity
    pos = 0
    entry_px = 0.0
    stop = 0.0
    qty = 0.0
    entry_time = None
    trades: list[dict] = []
    curve: list[float] = []

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal equity, pos, entry_time
        sign = 1.0 if pos == 1 else -1.0
        entry_fill = entry_px * (1 + slip) if pos == 1 else entry_px * (1 - slip)
        exit_fill = px * (1 - slip) if pos == 1 else px * (1 + slip)
        gross = (exit_fill - entry_fill) * qty * sign
        fees = qty * (entry_fill + exit_fill) * fee
        fc = 0.0
        if funding is not None and not funding.empty and entry_time is not None:
            exit_time = frame.index[i] + FOUR_HOURS
            sub = funding.loc[(funding.index > entry_time) & (funding.index <= exit_time)]
            for _, row in sub.iterrows():
                mp = float(row.get("mark_price", 0.0) or 0.0)
                if mp > 0:
                    fc += qty * mp * float(row["funding_rate"])
        net = gross - fees - (fc if pos == 1 else -fc)
        equity += net
        trades.append({"side": "long" if pos == 1 else "short", "reason": reason, "net": net})
        pos = 0
        entry_time = None

    for i in range(n):
        cur_atr = atr[i]
        if pos == 1:
            if np.isfinite(cur_atr) and l[i] <= stop:
                close_trade(i, min(o[i], stop), "stop")
            elif np.isfinite(exit_lower[i]) and c[i] < exit_lower[i]:
                close_trade(i, c[i], "reverse")
        elif pos == -1:
            if np.isfinite(cur_atr) and h[i] >= stop:
                close_trade(i, max(o[i], stop), "stop")
            elif np.isfinite(exit_upper[i]) and c[i] > exit_upper[i]:
                close_trade(i, c[i], "reverse")

        if pos == 0:
            allowed = allow_entry is None or bool(allow_entry[i])
            if allowed and np.isfinite(upper[i]) and c[i] > upper[i] and np.isfinite(cur_atr) and cur_atr > 0:
                risk = equity * risk_pct
                stop = c[i] - atr_mult * cur_atr
                qty = risk / (atr_mult * cur_atr)
                entry_px = c[i]
                entry_time = frame.index[i] + FOUR_HOURS
                pos = 1
            elif allowed and (not long_only) and np.isfinite(lower[i]) and c[i] < lower[i] \
                    and np.isfinite(cur_atr) and cur_atr > 0:
                risk = equity * risk_pct
                stop = c[i] + atr_mult * cur_atr
                qty = risk / (atr_mult * cur_atr)
                entry_px = c[i]
                entry_time = frame.index[i] + FOUR_HOURS
                pos = -1
        curve.append(equity)

    if pos != 0:
        close_trade(n - 1, c[-1], "end")
        curve[-1] = equity
    return trades, curve


def _maxdd(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float(((curve - peak) / peak).min() * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "all"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--total-equity", type=float, default=10_000.0,
                        help="total capital spread across all symbols (comparable to the TREND engine)")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--btc-gate", action="store_true",
                        help="allow entries only when BTC 4H close > EMA50 (long-only regime gate)")
    parser.add_argument("--funding-cache", default=None,
                        help="path to funding pickle (data/ab_funding.pkl); omit to ignore funding")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in ",".join(DEFAULT_SYMBOLS).split(",") if s.strip()]
    per_symbol = args.total_equity / len(symbols)

    funding = None
    if args.funding_cache:
        with open(args.funding_cache, "rb") as fh:
            funding = pickle.load(fh)

    cache = KlineCache(args.cache)
    btc_bull = None
    if args.btc_gate:
        btc = cache.load("BTCUSDT", "4h")
        btc_ema = btc["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        btc_bull = (btc["close"] > btc_ema)
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
        all_trades = []
        curves = []
        t0 = time.perf_counter()
        for s in symbols:
            frame = cache.load(s, "4h")
            frame = frame.loc[(frame.index + FOUR_HOURS >= start) & (frame.index + FOUR_HOURS <= stop)]
            if len(frame) < 60:
                continue
            allow = None
            if btc_bull is not None:
                allow = btc_bull.reindex(frame.index).ffill().fillna(False).to_numpy()
            trades, curve = run_symbol(frame, per_symbol, N, M, atr_mult, risk_pct, fee, slip,
                                       args.long_only, allow, funding.get(s) if funding else None)
            all_trades.extend(trades)
            curves.append(pd.Series(curve, index=frame.index))

        net = sum(t["net"] for t in all_trades)
        wins = sum(1 for t in all_trades if t["net"] > 0)
        gw = sum(t["net"] for t in all_trades if t["net"] > 0)
        gl = -sum(t["net"] for t in all_trades if t["net"] < 0)
        pf = (gw / gl) if gl > 0 else float("inf")
        per_symbol_dd = [ _maxdd(s) for s in curves ]
        mean_dd = float(np.mean(per_symbol_dd)) if per_symbol_dd else 0.0
        combined = pd.concat(curves, axis=1).sum(axis=1)
        combined_dd = _maxdd(combined)
        longs = [t for t in all_trades if t["side"] == "long"]
        shorts = [t for t in all_trades if t["side"] == "short"]
        print(f"=== window {w} {start.date()}..{stop.date()} ({time.perf_counter() - t0:.0f}s) "
              f"long_only={args.long_only} ===", flush=True)
        print(f"  net={net:.2f} (ret {net/args.total_equity*100:.2f}%)  trades={len(all_trades)}  "
              f"win={wins/len(all_trades)*100:.0f}%  PF={pf:.2f}  maxDD(avg per-symbol)={mean_dd:.2f}%  "
              f"maxDD(combined)={combined_dd:.2f}%", flush=True)
        print(f"    long  n={len(longs)} net={sum(t['net'] for t in longs):.2f}  "
              f"short n={len(shorts)} net={sum(t['net'] for t in shorts):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
