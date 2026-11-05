"""Analyze the live CTA paper OOS run against the backtest baseline.

Reads the CTA runner's state file (data/paper/cta_portfolio.json) and reports:

  1. The mark-to-market equity curve (return %, max drawdown %) reconstructed
     from ``equity_history`` (appended once per cycle by the runner).
  2. The trade-by-trade record reconstructed from ``events`` (OPEN/CLOSE
     pairs), broken down by exit reason, with win rate and PnL per reason.
  3. The breakeven-rescue count: ``protective_stop`` exits whose fill is near
     the entry price (the signature of the be2 breakeven stop). This is the
     single most important OOS check -- the backtest attributed be2's edge to
     ~22 such rescues over 4 months (~5.5/month).
  4. A side-by-side against the backtest baseline (max_pos=5, real funding).

Self-contained (stdlib only) so it runs anywhere the state file is available.

Usage:
  python scripts/analyze_cta_oos.py [--state data/paper/cta_portfolio.json]
                                    [--breakeven-tol 0.005]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Backtest baseline for the CTA breakout long-only strategy (max_pos=5, real
# funding, 30 symbols). These are the numbers the live OOS run is being
# compared against. Sourced from the in-sample backtest (diagnose_cta_exits.py
# / diagnose_cta_portfolio.py) and recorded in project memory on 2026-10-24.
BACKTEST_BASELINE = {
    "windows": "4x30d (Jun26-Jul26, Jul25-Aug24, Aug24-Sep23, Sep23-Oct23)",
    "baseline_return_total_pct": +12.39,   # no exit management
    "be2_return_total_pct": +14.96,        # breakeven be_mult=2.0
    "baseline_by_window_pct": {"W4": -0.41, "W3": +11.02, "W1": +8.03, "W2": -6.25},
    "be2_by_window_pct": {"W4": -0.09, "W3": +12.47, "W1": +8.32, "W2": -5.74},
    "be2_rescue_count_per_4mo": 22,        # ~5.5 breakeven rescues / month
}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def reconstruct_trades(events: list[dict]) -> list[dict]:
    """Pair OPEN/CLOSE paper_fill events into per-trade records."""
    trades: list[dict] = []
    open_by_symbol: dict[str, dict] = {}
    for ev in events:
        if ev.get("event") != "paper_fill":
            continue
        action = ev.get("action")
        symbol = ev.get("symbol")
        if action == "OPEN":
            open_by_symbol[symbol] = ev
        elif action == "CLOSE":
            entry = open_by_symbol.pop(symbol, None)
            if entry is None:
                continue  # CLOSE without a matching OPEN in the window
            entry_price = float(entry["price"])
            exit_price = float(ev["price"])
            quantity = float(ev.get("quantity") or entry.get("quantity"))
            net_pnl = ev.get("net_pnl")
            if net_pnl is None:
                # demo-mode CLOSE events omit net_pnl; approximate without fees.
                net_pnl = (exit_price - entry_price) * quantity
            trades.append({
                "symbol": symbol,
                "entry_at": _parse_ts(entry["at"]),
                "exit_at": _parse_ts(ev["at"]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "net_pnl": float(net_pnl),
                "reason": ev.get("reason", "unknown"),
            })
    return trades


def equity_stats(equity_history: list[dict], initial_equity: float) -> dict | None:
    if not equity_history:
        return None
    equities = [float(e["equity"]) for e in equity_history]
    last = equities[-1]
    ret_pct = (last / initial_equity - 1.0) * 100.0 if initial_equity else 0.0
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return {
        "n_points": len(equities),
        "first_at": equity_history[0]["at"],
        "last_at": equity_history[-1]["at"],
        "return_pct": ret_pct,
        "max_dd_pct": max_dd * 100.0,
    }


def realized_equity_stats(trades: list[dict], initial_equity: float) -> dict | None:
    """Fallback curve from realized PnL only (no MTM; understates drawdown)."""
    if not trades:
        return None
    points = []
    running = initial_equity
    for t in trades:
        running += t["net_pnl"]
        points.append(running)
    return {
        "n_points": len(points),
        "first_at": trades[0]["entry_at"].isoformat(),
        "last_at": trades[-1]["exit_at"].isoformat(),
        "return_pct": (points[-1] / initial_equity - 1.0) * 100.0 if initial_equity else 0.0,
        "max_dd_pct": max_drawdown_pct(points),
    }


def max_drawdown_pct(equities: list[float]) -> float:
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return max_dd * 100.0


def breakeven_rescues(trades: list[dict], tol: float) -> list[dict]:
    out = []
    for t in trades:
        if t["reason"] != "protective_stop":
            continue
        ratio = t["exit_price"] / t["entry_price"] - 1.0 if t["entry_price"] else 0.0
        if abs(ratio) <= tol:
            out.append({**t, "exit_vs_entry_pct": ratio * 100.0})
    return out


def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path,
                        default=Path("data/paper/cta_portfolio.json"),
                        help="path to the CTA runner's state file")
    parser.add_argument("--breakeven-tol", type=float, default=0.005,
                        help="|exit/entry - 1| below which a protective_stop is "
                             "counted as a breakeven rescue (default 0.5%%)")
    args = parser.parse_args()

    if not args.state.exists():
        print(f"[ERROR] state file not found: {args.state}")
        return 1
    data = json.loads(args.state.read_text(encoding="utf-8"))

    initial_equity = float(data.get("initial_equity", 10_000.0))
    events = data.get("events", [])
    equity_history = data.get("equity_history", [])
    trades = reconstruct_trades(events)

    print("=" * 72)
    print("CTA paper OOS forward-validation")
    print("=" * 72)
    print(f"state file      : {args.state}")
    print(f"execution_mode  : {data.get('execution_mode', '?')}")
    print(f"initial_equity  : ${initial_equity:,.2f}")
    print(f"events          : {len(events)}")
    print(f"trades (closed) : {len(trades)}")

    # 1) equity curve
    print("\n-- equity curve --")
    stats = equity_stats(equity_history, initial_equity)
    if stats is not None:
        print(f"points          : {stats['n_points']} "
              f"({stats['first_at']} .. {stats['last_at']})")
        print(f"return          : {_fmt_pct(stats['return_pct'])}")
        print(f"max drawdown    : {_fmt_pct(-stats['max_dd_pct'])}")
    else:
        print("no equity_history yet (runner started before 2026-10-24 code "
              "update). Falling back to realized-PnL-only curve:")
        realized = realized_equity_stats(trades, initial_equity)
        if realized is not None:
            print(f"return (realized): {_fmt_pct(realized['return_pct'])}")
            print(f"max drawdown (realized-only, understates DD): "
                  f"{_fmt_pct(-realized['max_dd_pct'])}")
        else:
            print("no trades either -- run has just started.")

    # 2) trades by exit reason
    print("\n-- trades by exit reason --")
    if not trades:
        print("(none yet)")
    else:
        reason_stats = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
        for t in trades:
            s = reason_stats[t["reason"]]
            s["n"] += 1
            s["pnl"] += t["net_pnl"]
            if t["net_pnl"] > 0:
                s["wins"] += 1
        print(f"{'reason':<20} {'n':>4} {'win%':>7} {'net_pnl':>12}")
        for reason, s in sorted(reason_stats.items(), key=lambda kv: -kv[1]["pnl"]):
            win_pct = 100.0 * s["wins"] / s["n"] if s["n"] else 0.0
            print(f"{reason:<20} {s['n']:>4} {win_pct:>6.1f}% {s['pnl']:>+12.2f}")
        total = sum(t["net_pnl"] for t in trades)
        total_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        print(f"{'TOTAL':<20} {len(trades):>4} "
              f"{100.0 * total_wins / len(trades):>6.1f}% {total:>+12.2f}")

    # 3) breakeven rescues
    print("\n-- breakeven rescues (be2 signature) --")
    rescues = breakeven_rescues(trades, args.breakeven_tol)
    protective = [t for t in trades if t["reason"] == "protective_stop"]
    if protective:
        print(f"protective_stop exits : {len(protective)}")
        print(f"breakeven rescues     : {len(rescues)} "
              f"(|exit/entry - 1| <= {args.breakeven_tol:.1%})")
        for t in rescues:
            print(f"  {t['symbol']:<14} {t['exit_vs_entry_pct']:+.3f}% "
                  f"net {t['net_pnl']:+.2f}")
    else:
        print("no protective_stop exits yet.")

    # 4) backtest baseline reference
    print("\n-- backtest baseline (in-sample, for reference) --")
    print(f"windows        : {BACKTEST_BASELINE['windows']}")
    print(f"return (no be) : "
          f"{_fmt_pct(BACKTEST_BASELINE['baseline_return_total_pct'])}")
    print(f"return (be2)   : "
          f"{_fmt_pct(BACKTEST_BASELINE['be2_return_total_pct'])}")
    print(f"be2 rescues    : ~{BACKTEST_BASELINE['be2_rescue_count_per_4mo']} "
          f"per 4 months (~5.5/month)")
    print("\nNOTE: the live period will differ from the in-sample windows, so "
          "compare direction and the breakeven-rescue rate, not the exact "
          "return figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
