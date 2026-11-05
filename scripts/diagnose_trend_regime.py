"""Diagnose TREND regime dependence across two OOS windows.

Runs the baseline (TREND-only, RANGE gate left at production neutral) backtest on
one window with real funding, then breaks the TREND trades down by win rate,
long/short, exit reason, and per-symbol contribution, and characterizes the BTC
4H market regime (return, realized vol, ADX, drawdown) over the same window.

Run:  python scripts/diagnose_trend_regime.py --window 1
      python scripts/diagnose_trend_regime.py --window 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache
from bot.indicators import indicator_frame
from bot.market_risk import classify_market_risk

from run_range_ab import DEFAULT_SYMBOLS, WINDOW_DAYS, load_or_fetch_funding


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def trade_rows(trades):
    rows = []
    for trade in trades:
        if (trade.source or "") != "TREND":
            continue
        rows.append({
            "symbol": trade.symbol,
            "side": trade.side.value,
            "opened": trade.opened_at,
            "closed": trade.closed_at,
            "held_h": (trade.closed_at - trade.opened_at).total_seconds() / 3600.0,
            "gross": float(trade.gross_pnl),
            "fees": float(trade.fees),
            "funding": float(trade.funding_paid),
            "net": float(trade.net_pnl),
            "exit": trade.exit_reason,
        })
    return rows


def report_trend(rows, label: str):
    print(f"\n=== TREND trades: {label} (n={len(rows)}) ===")
    if not rows:
        print("  (no TREND trades)")
        return
    frame = pd.DataFrame(rows)
    wins = frame[frame["net"] > 0]
    losses = frame[frame["net"] <= 0]
    net = frame["net"].sum()
    gross = frame["gross"].sum()
    fees = frame["fees"].sum()
    funding = frame["funding"].sum()
    print(f"  net={net:.2f}  gross={gross:.2f}  fees={fees:.2f}  funding={funding:.2f}")
    print(f"  win_rate={len(wins)/len(frame)*100:.1f}%  avg_win={wins['net'].mean() if len(wins) else 0:.2f}  "
          f"avg_loss={losses['net'].mean() if len(losses) else 0:.2f}  "
          f"PF={wins['net'].sum()/abs(losses['net'].sum()) if len(losses) and losses['net'].sum() < 0 else float('inf'):.2f}")
    print(f"  avg_held_h={frame['held_h'].mean():.1f}  median_held_h={frame['held_h'].median():.1f}")
    for side in ("long", "short"):
        sub = frame[frame["side"] == side]
        print(f"  {side:<6} n={len(sub):>3}  net={sub['net'].sum():>9.2f}  "
              f"win_rate={len(sub[sub['net']>0])/len(sub)*100 if len(sub) else 0:.0f}%")
    print("  by exit_reason:")
    for reason, sub in frame.groupby("exit"):
        print(f"    {reason:<34} n={len(sub):>3}  net={sub['net'].sum():>9.2f}")
    print("  by symbol (top |net|):")
    by_sym = frame.groupby("symbol")["net"].agg(["count", "sum"]).sort_values("sum", key=abs, ascending=False)
    for sym, row in by_sym.head(12).iterrows():
        print(f"    {sym:<16} n={int(row['count']):>3}  net={row['sum']:>9.2f}")


def report_btc(frame: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp, label: str):
    window = frame.loc[(frame.index >= start) & (frame.index < stop)]
    if window.empty:
        print(f"\n=== BTC regime: {label} ===  (no bars)")
        return
    close = window["close"].astype(float)
    ret = float(close.iloc[-1] / close.iloc[0] - 1.0) * 100
    logret = np.log(close / close.shift(1)).dropna()
    vol = float(logret.std() * math.sqrt(6 * 365) * 100)  # 4h bars -> annualized %
    adx = indicator_frame(window, 14, 14)["adx"]
    adx_mean = float(adx.dropna().mean())
    dd = float((close / close.cummax() - 1).min()) * 100
    up = int((window["close"] > window["open"]).sum())
    down = int((window["close"] < window["open"]).sum())
    print(f"\n=== BTC regime: {label} ===")
    print(f"  return={ret:.2f}%  realized_vol_ann={vol:.1f}%  adx_mean={adx_mean:.1f}  "
          f"maxDD={dd:.2f}%  up/down_bars={up}/{down}")


def report_market_risk(result, market_bars_4h, settings, start, stop, label):
    print(f"\n=== market_risk gate: {label} ===")
    summary = result.market_risk_execution_summary or {}
    for state in ("NORMAL", "RISK_ELEVATED", "PAUSE_NEW"):
        row = summary.get(state)
        if row:
            print(f"  {state:<14} opened={row['opened_count']:>3}  notional={row['opened_notional']:>10.0f}  "
                  f"mult={row['configured_size_multiplier']}")
        else:
            print(f"  {state:<14} opened=  0")
    four_h = pd.Timedelta(hours=4)
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for ts in market_bars_4h.index:
        if not (start <= ts < stop):
            continue
        history = market_bars_4h.loc[market_bars_4h.index + four_h <= ts]
        try:
            assessment = classify_market_risk(history, settings, now=ts.to_pydatetime())
        except Exception as exc:
            counts["ERR"] = counts.get("ERR", 0) + 1
            continue
        counts[assessment.state.value] = counts.get(assessment.state.value, 0) + 1
        reasons[assessment.reason] = reasons.get(assessment.reason, 0) + 1
    print("  BTC 4h risk-state distribution over window:")
    for state in ("NORMAL", "RISK_ELEVATED", "PAUSE_NEW"):
        print(f"    {state:<14} {counts.get(state, 0)} bars")
    print("  by reason:")
    for reason, n in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"    {reason:<38} {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True, choices=(1, 2))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    cache = KlineCache(args.cache)
    funding = load_or_fetch_funding(symbols, 2 * args.window_days, Path(args.funding_cache))

    bars_1h, bars_4h = {}, {}
    for s in symbols:
        bars_1h[s] = cache.load(s, "1h")
        bars_4h[s] = cache.load(s, "4h")
    market_bars_4h = cache.load("BTCUSDT", "4h")

    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    settings = load_settings(env_file=None)
    end = pd.Timestamp.utcnow().floor("h")
    if args.window == 1:
        start = end - pd.Timedelta(days=2 * args.window_days)
        stop = end - pd.Timedelta(days=args.window_days)
    else:
        start = end - pd.Timedelta(days=args.window_days)
        stop = end
    label = f"window {args.window} {start.date()}..{stop.date()}"

    result = run_portfolio_backtest(
        bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
        start=start, end=stop, top_n=20, initial_equity=10_000.0,
        enable_trend_routing=True, market_bars_4h=market_bars_4h,
        mr_entry_mode="legacy", mr_deviation_pct=0.02,
    )
    report_btc(market_bars_4h, start, stop, label)
    report_market_risk(result, market_bars_4h, settings, start, stop, label)
    report_trend(trade_rows(result.trades), label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
