"""Diagnose where the ~2:1 long bias enters the TREND funnel (pure diagnosis).

The TREND classification layer is direction-balanced at the 4H universe level
(W2: TREND_UP ~466 vs TREND_DOWN ~441, ~1.06x), yet execution opens ~2:1 long
(W2: 40 long / 21 short). This script counts each funnel layer per direction so
the skew can be attributed to a specific gate rather than guessed:

  L0  universe 4H classification (all symbols, every 4H bar in the window)
  L1  top-20-eligible TREND symbol-hours (the only symbols the backtest evaluates)
  L2  1H entry signal OPEN (long/short), plus the blocked reason by direction
  L3  route-passed / queued (outer route block, e.g. market_risk_paused)
  L4  opened trades (long/short), from the backtest's actual fills
  L5  fill-stage rejections (guard_signal / margin / capacity) by side

This is read-only diagnosis: it only adds an opt-in ``record_trend_funnel``
collection flag to the backtest and does not change any decision output.

Run:  python scripts/diagnose_trend_funnel.py --window 2
      python scripts/diagnose_trend_funnel.py --window 1   (contrast)
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache
from bot.symbol_regime import _raw_candidates, classify_symbol_regime

from run_range_ab import DEFAULT_SYMBOLS, WINDOW_DAYS, load_or_fetch_funding

FOUR_HOURS = pd.Timedelta(4, unit="h")


def universe_classification(bars_4h, settings, start: pd.Timestamp,
                            stop: pd.Timestamp) -> tuple[int, int, int]:
    """Count TREND_UP / TREND_DOWN / other over every 4H bar in the window.

    Uses the same per-bar candidate sequence ``_raw_candidates`` produces and
    the same confirm_bars confirmation rule as ``classify_symbol_regime``, but
    vectorized (one pass per symbol) so it runs in seconds instead of minutes.
    Every bar inside the window is far past indicator warmup, so this matches
    the per-bar classifier exactly here.
    """
    policy = settings.symbol_regime
    cb = int(policy.confirm_bars)
    rcb = int(policy.range_confirm_bars)
    up = down = other = 0
    for symbol, frame in bars_4h.items():
        candidates, _raw = _raw_candidates(frame, policy)
        for i, ts in enumerate(frame.index):
            close_t = ts + FOUR_HOURS
            if not (start <= close_t < stop):
                continue
            if i + 1 >= cb:
                win = candidates[i - cb + 1:i + 1]
                if len(set(win)) == 1 and win[0] == "TREND_UP":
                    up += 1
                    continue
                if len(set(win)) == 1 and win[0] == "TREND_DOWN":
                    down += 1
                    continue
            if i + 1 >= rcb:
                rwin = candidates[i - rcb + 1:i + 1]
                if len(set(rwin)) == 1 and rwin[0] == "RANGE":
                    other += 1
                    continue
            other += 1
    return up, down, other


def _ratio(a: int, b: int) -> str:
    if b == 0:
        return "inf" if a else "n/a"
    return f"{a / b:.2f}"


def report_funnel(events, fill_rejections, trades, label: str) -> None:
    print(f"\n=== TREND entry funnel: {label} ===")
    by_dir = {
        "up": [e for e in events if e["direction"] == "up"],
        "down": [e for e in events if e["direction"] == "down"],
    }
    layers = {}
    for d in ("up", "down"):
        es = by_dir[d]
        l1 = len(es)
        l2 = sum(1 for e in es if e["entry_action"] == "OPEN")
        l3 = sum(1 for e in es if e["final_action"] == "OPEN")
        layers[d] = {"L1_classified": l1, "L2_1h_open": l2, "L3_queued": l3}

    print(f"  {'layer':<26} {'up':>6} {'down':>6} {'up:down':>9}")
    for label_, up_key, down_key in (
        ("L1 top-20 classified TREND", "L1_classified", "L1_classified"),
        ("L2 1H entry OPEN", "L2_1h_open", "L2_1h_open"),
        ("L3 route-passed / queued", "L3_queued", "L3_queued"),
    ):
        u, d = layers["up"][up_key], layers["down"][down_key]
        print(f"  {label_:<26} {u:>6} {d:>6} {_ratio(u, d):>9}")

    tr = [t for t in trades if (t.source or "") == "TREND"]
    opened = {"long": sum(1 for t in tr if t.side.value == "long"),
              "short": sum(1 for t in tr if t.side.value == "short")}
    print(f"  {'L4 opened trades (long/short)':<26} {opened['long']:>6} {opened['short']:>6} "
          f"{_ratio(opened['long'], opened['short']):>9}")

    # Survival rates make the per-direction pass rate explicit.
    print("  per-direction survival (classified -> 1h open -> queued):")
    for d in ("up", "down"):
        s = layers[d]
        s12 = s["L2_1h_open"] / s["L1_classified"] if s["L1_classified"] else float("nan")
        s23 = s["L3_queued"] / s["L2_1h_open"] if s["L2_1h_open"] else float("nan")
        print(f"    {d:<5} L1->L2 {s12*100:5.1f}%   L2->L3 {s23*100:5.1f}%")

    # Why does L1 -> L2 drop? (1H entry gate, by direction)
    print("  L1 -> L2 blocked reasons (1H entry gate), by direction:")
    blocked_reasons = {"up": collections.Counter(), "down": collections.Counter()}
    for d in ("up", "down"):
        for e in by_dir[d]:
            if e["entry_action"] != "OPEN" and e["route_block"] is None:
                blocked_reasons[d][e["entry_reason"]] += 1
    all_reasons = sorted({r for d in ("up", "down") for r in blocked_reasons[d]})
    for r in all_reasons:
        u, dd = blocked_reasons["up"].get(r, 0), blocked_reasons["down"].get(r, 0)
        print(f"    {r:<42} up={u:>5}  down={dd:>5}")

    # Why does L2 -> L3 drop? (outer route block)
    print("  L2 -> L3 route blocks (outer route), by direction:")
    route_blocks = {"up": collections.Counter(), "down": collections.Counter()}
    for d in ("up", "down"):
        for e in by_dir[d]:
            if e["entry_action"] == "OPEN" and e["final_action"] != "OPEN":
                route_blocks[d][e["final_reason"]] += 1
    for r in sorted({r for d in ("up", "down") for r in route_blocks[d]}):
        u, dd = route_blocks["up"].get(r, 0), route_blocks["down"].get(r, 0)
        print(f"    {r:<42} up={u:>5}  down={dd:>5}")

    # Fill-stage rejections (risk/capacity/margin) by side.
    print("  L5 fill-stage rejections (risk/capacity), by side:")
    fill = {"long": collections.Counter(), "short": collections.Counter()}
    for f in fill_rejections:
        fill[f["side"]][f["reason"]] += 1
    for r in sorted({r for side in ("long", "short") for r in fill[side]}):
        l, s = fill["long"].get(r, 0), fill["short"].get(r, 0)
        print(f"    {r:<42} long={l:>5}  short={s:>5}")
    if not fill["long"] and not fill["short"]:
        print("    (none)")

    # Opened TREND notional by side: is the short sizing systematically larger?
    print("  opened TREND notional (quantity*entry) by side:")
    for side in ("long", "short"):
        ns = [t.quantity * t.entry_price for t in tr if t.side.value == side]
        if ns:
            avg = sum(ns) / len(ns)
            over3k = sum(1 for n in ns if n > 3000)
            print(f"    {side:<5} n={len(ns):>3}  avg_notional={avg:>9.0f}  n>3000={over3k}")
        else:
            print(f"    {side:<5} n=  0")

    # Notional detail for the two symmetric caps that rejected mostly shorts.
    print("  cap rejection notional (order vs cap):")
    for reason in ("per_symbol_notional_cap", "directional_exposure_cap"):
        for side in ("long", "short"):
            rows = [f for f in fill_rejections if f["reason"] == reason and f["side"] == side]
            if not rows:
                continue
            on = [f["details"].get("order_notional") for f in rows]
            on = [x for x in on if isinstance(x, (int, float))]
            mn = [f["details"].get("max_notional") for f in rows]
            mn = [x for x in mn if isinstance(x, (int, float))]
            if on:
                print(f"    {reason:<28} {side:<5} n={len(rows):>3}  order "
                      f"min/avg/max={min(on):.0f}/{sum(on)/len(on):.0f}/{max(on):.0f}  "
                      f"cap={mn[0]:.0f}" if mn else
                      f"    {reason:<28} {side:<5} n={len(rows):>3}  order "
                      f"min/avg/max={min(on):.0f}/{sum(on)/len(on):.0f}/{max(on):.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True, choices=(1, 2))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--window-days", type=int, default=30,
                        help="days per window (sandbox default 30; use 60 locally)")
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
    parser.add_argument("--skip-universe", action="store_true",
                        help="skip L0 universe classification (faster)")
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

    if not args.skip_universe:
        up, down, other = universe_classification(bars_4h, settings, start, stop)
        print(f"\n=== L0 universe 4H classification: {label} ===")
        print(f"  TREND_UP={up}  TREND_DOWN={down}  other={other}  up:down={_ratio(up, down)}")

    result = run_portfolio_backtest(
        bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
        start=start, end=stop, top_n=20, initial_equity=10_000.0,
        enable_trend_routing=True, market_bars_4h=market_bars_4h,
        mr_entry_mode="legacy", mr_deviation_pct=0.02,
        record_trend_funnel=True,
    )
    report_funnel(result.trend_funnel, result.trend_funnel_fill_rejections,
                  result.trades, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
