"""Quantify the new-RANGE vs legacy-LOW/HIGH routing contradiction (read-only).

Offline diagnostic. For a set of symbols and their 4H bars, classify every
completed bar with BOTH the new symbol-regime classifier and the legacy
mean-reversion classifier, then cross-tabulate the exact questions that decide
whether a mean-reversion (MR) entry can ever fire:

  1. When the NEW classifier confirms RANGE, what does the LEGACY classifier
     say?  MR entries need legacy LOW/HIGH; legacy SIDEWAYS blocks entry.
  2. How often "close within 1.5% of EMA50" (new RANGE distance gate) coincides
     with "range position outside 30%-70%" (legacy LOW/HIGH edge gate)?
  3. How often ADX<=18 coincides with ATR% percentile > 30 (the only way a
     new-RANGE bar escapes the legacy SIDEWAYS block)?

No orders, no state mutation, no private credentials. Data is read from Binance's
public market-data mirror (data-api.binance.vision, spot 4H) unless --cache points
at an existing KlineCache.

Run:  python scripts/diagnose_regime_conflict.py [--symbols SOL,XRP,...] [--days 120]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.config import load_settings
from bot.data.cache import KlineCache
from bot.data.klines import TIMEFRAME_MS, ohlcv_frame
from bot.regime import Regime, classify_regimes
from bot.symbol_regime import _raw_candidates, _structure

FOUR_HOURS_MS = TIMEFRAME_MS["4h"]

# Liquid alt perpetuals, also listed on Binance spot, in base-quote "XUSDT" form.
DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]


def _fetch_bars_data_api(symbol: str, *, days: int) -> list[list[float]]:
    """Fetch closed 4H bars from Binance's public market-data mirror."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    url = (
        "https://data-api.binance.vision/api/v3/klines"
        f"?symbol={symbol}&interval=4h&startTime={start_ms}&limit=1000"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode())
    rows = []
    for row in payload:
        open_time = int(row[0])
        if open_time + FOUR_HOURS_MS > now_ms:  # drop the still-forming bar
            continue
        rows.append([open_time, *map(float, row[1:6])])
    return rows


def _confirmed_states(candidates: list[str], *, confirm_bars: int,
                      range_confirm_bars: int) -> list[str]:
    """Reconstruct per-bar confirmed states the same way classify_symbol_regime does."""
    states = []
    for i, raw in enumerate(candidates):
        if raw in ("TREND_UP", "TREND_DOWN"):
            if i + 1 >= confirm_bars and len(set(candidates[i - confirm_bars + 1:i + 1])) == 1:
                states.append("TREND")
            else:
                states.append("UNCERTAIN")
        elif raw == "RANGE":
            if i + 1 >= range_confirm_bars and all(
                c == "RANGE" for c in candidates[i - range_confirm_bars + 1:i + 1]
            ):
                states.append("RANGE")
            else:
                states.append("UNCERTAIN")
        else:
            states.append("UNCERTAIN")
    return states


def analyze_symbol(symbol: str, closed: pd.DataFrame, settings) -> dict:
    """Cross-tabulate new vs legacy classifier over one symbol's closed 4H bars."""
    policy = settings.symbol_regime
    candidates, _ = _raw_candidates(closed, policy)
    states = _confirmed_states(
        candidates, confirm_bars=policy.confirm_bars,
        range_confirm_bars=policy.range_confirm_bars,
    )

    legacy = classify_regimes(closed, settings)
    old_regime = legacy["regime"].map(lambda value: value.value if value is not None else None)
    adx = legacy["adx"]
    atr_rank = legacy["atr_percentile_rank"]
    range_position = legacy["range_position"]

    ema_slow = closed["close"].ewm(
        span=policy.ema_slow_bars, adjust=False, min_periods=policy.ema_slow_bars,
    ).mean()
    distance_pct = closed["close"] / ema_slow - 1.0

    n = len(closed)
    stats = {
        "symbol": symbol,
        "bars": n,
        "new_state": {"TREND": 0, "RANGE": 0, "UNCERTAIN": 0},
        "range_old_counts": {},
        "range_old_edge": 0,          # old LOW or HIGH under new RANGE
        "trend_old_counts": {},
        # Contradiction A: near EMA50 vs at range edge
        "near_ema": 0, "near_ema_edge": 0,
        # Contradiction B: ADX<=18 vs ATR% rank > 30
        "low_adx": 0, "low_adx_high_atr_rank": 0,
    }
    for i in range(n):
        if old_regime.iloc[i] is None:
            continue
        new_state = states[i]
        stats["new_state"][new_state] += 1
        old = old_regime.iloc[i]

        if new_state == "RANGE":
            stats["range_old_counts"][old] = stats["range_old_counts"].get(old, 0) + 1
            if old in ("LOW", "HIGH"):
                stats["range_old_edge"] += 1
        elif new_state == "TREND":
            stats["trend_old_counts"][old] = stats["trend_old_counts"].get(old, 0) + 1

        # Contradiction A (all bars with valid distance + range_position)
        d = float(distance_pct.iloc[i])
        rp = float(range_position.iloc[i]) if pd.notna(range_position.iloc[i]) else math.nan
        if math.isfinite(d):
            near = abs(d) <= policy.range_max_distance_pct
            if near:
                stats["near_ema"] += 1
                if math.isfinite(rp) and (rp <= 0.30 or rp >= 0.70):
                    stats["near_ema_edge"] += 1

        # Contradiction B (all bars with valid adx + atr_rank)
        a = float(adx.iloc[i]) if pd.notna(adx.iloc[i]) else math.nan
        ar = float(atr_rank.iloc[i]) if pd.notna(atr_rank.iloc[i]) else math.nan
        if math.isfinite(a) and a <= policy.range_adx_max:
            stats["low_adx"] += 1
            if math.isfinite(ar) and ar > 30:
                stats["low_adx_high_atr_rank"] += 1

    # RANGE gate breakdown: which of the four conditions is the binding constraint.
    gate = {
        "adx_eval": 0, "adx_fail": 0,
        "slope_eval": 0, "slope_fail": 0,
        "distance_eval": 0, "distance_fail": 0,
        "structure_eval": 0, "structure_fail": 0,
        "all_eval": 0, "all_pass": 0,
    }
    for i in range(n):
        a = float(adx.iloc[i]) if pd.notna(adx.iloc[i]) else math.nan
        d = float(distance_pct.iloc[i]) if pd.notna(distance_pct.iloc[i]) else math.nan
        slope = math.nan
        if i >= policy.slope_lookback_bars:
            ref = float(ema_slow.iloc[i - policy.slope_lookback_bars])
            cur = float(ema_slow.iloc[i])
            if math.isfinite(ref) and math.isfinite(cur) and ref > 0:
                slope = cur / ref - 1.0
        structure_state = None
        if i >= 2 * policy.structure_lookback_bars - 1:
            structure_state, _ = _structure(closed, i, policy.structure_lookback_bars)

        if math.isfinite(a):
            gate["adx_eval"] += 1
            if not a <= policy.range_adx_max:
                gate["adx_fail"] += 1
        if math.isfinite(slope):
            gate["slope_eval"] += 1
            if not abs(slope) <= policy.range_slope_max_pct:
                gate["slope_fail"] += 1
        if math.isfinite(d):
            gate["distance_eval"] += 1
            if not abs(d) <= policy.range_max_distance_pct:
                gate["distance_fail"] += 1
        if structure_state is not None:
            gate["structure_eval"] += 1
            if structure_state != "NEUTRAL":
                gate["structure_fail"] += 1
        if all(math.isfinite(x) for x in (a, d, slope)) and structure_state is not None:
            gate["all_eval"] += 1
            if (a <= policy.range_adx_max and abs(slope) <= policy.range_slope_max_pct
                    and abs(d) <= policy.range_max_distance_pct
                    and structure_state == "NEUTRAL"):
                gate["all_pass"] += 1
    stats["range_gate"] = gate

    return stats


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="comma-separated 'XUSDT' base-quote symbols")
    parser.add_argument("--days", type=int, default=120, help="history length in days")
    parser.add_argument("--cache", type=str, default=None,
                        help="optional KlineCache sqlite path to read/write")
    parser.add_argument("--json-out", type=str, default=None,
                        help="optional path to write raw per-symbol stats as JSON")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")

    settings = load_settings(env_file=None)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    cache = KlineCache(args.cache) if args.cache else None

    per_symbol = []
    for symbol in symbols:
        try:
            if cache is not None:
                since_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
                frame = cache.load(symbol, "4h", since_ms=since_ms)
                if frame.empty:
                    rows = _fetch_bars_data_api(symbol, days=args.days)
                    cache.upsert(symbol, "4h", rows)
                    frame = ohlcv_frame(rows)
            else:
                rows = _fetch_bars_data_api(symbol, days=args.days)
                frame = ohlcv_frame(rows)
        except Exception as exc:
            print(f"  [skip] {symbol}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        if len(frame) < 120:
            print(f"  [skip] {symbol}: only {len(frame)} bars")
            continue
        stats = analyze_symbol(symbol, frame, settings)
        per_symbol.append(stats)

    if not per_symbol:
        print("No symbols produced usable data.")
        return 1

    # Aggregate
    agg = {
        "bars": 0, "new_state": {"TREND": 0, "RANGE": 0, "UNCERTAIN": 0},
        "range_old_counts": {}, "range_old_edge": 0, "trend_old_counts": {},
        "near_ema": 0, "near_ema_edge": 0, "low_adx": 0, "low_adx_high_atr_rank": 0,
        "range_gate": {k: 0 for k in (
            "adx_eval", "adx_fail", "slope_eval", "slope_fail",
            "distance_eval", "distance_fail", "structure_eval", "structure_fail",
            "all_eval", "all_pass",
        )},
    }
    for s in per_symbol:
        agg["bars"] += s["bars"]
        for k in ("TREND", "RANGE", "UNCERTAIN"):
            agg["new_state"][k] += s["new_state"][k]
        for k, v in s["range_old_counts"].items():
            agg["range_old_counts"][k] = agg["range_old_counts"].get(k, 0) + v
        agg["range_old_edge"] += s["range_old_edge"]
        for k, v in s["trend_old_counts"].items():
            agg["trend_old_counts"][k] = agg["trend_old_counts"].get(k, 0) + v
        agg["near_ema"] += s["near_ema"]
        agg["near_ema_edge"] += s["near_ema_edge"]
        agg["low_adx"] += s["low_adx"]
        agg["low_adx_high_atr_rank"] += s["low_adx_high_atr_rank"]
        for k, v in s["range_gate"].items():
            agg["range_gate"][k] += v

    total = agg["bars"]
    print("=" * 72)
    print(f"Regime-conflict diagnostic  (symbols={len(per_symbol)}, bars={total}, "
          f"days={args.days}, source=Binance data mirror 4H)")
    print("=" * 72)

    print("\n[1] New classifier confirmed-state distribution")
    for k in ("TREND", "RANGE", "UNCERTAIN"):
        print(f"    {k:>9}: {agg['new_state'][k]:>7}  ({_pct(agg['new_state'][k], total)})")

    n_range = agg["new_state"]["RANGE"]
    print(f"\n[2] MR path: when NEW confirms RANGE (n={n_range}), what does LEGACY say?")
    for old in sorted(agg["range_old_counts"], key=lambda k: -agg["range_old_counts"][k]):
        cnt = agg["range_old_counts"][old]
        print(f"    {old:>10}: {cnt:>7}  ({_pct(cnt, n_range)})")
    print(f"    -> legacy LOW|HIGH (would reach MR signal gate): {agg['range_old_edge']} "
          f"({_pct(agg['range_old_edge'], n_range)})")
    print(f"    -> legacy SIDEWAYS (blocks entry):                "
          f"{agg['range_old_counts'].get('SIDEWAYS', 0)} "
          f"({_pct(agg['range_old_counts'].get('SIDEWAYS', 0), n_range)})")

    n_trend = agg["new_state"]["TREND"]
    print(f"\n[3] TREND path (informational): NEW TREND (n={n_trend}) -> legacy regime")
    for old in sorted(agg["trend_old_counts"], key=lambda k: -agg["trend_old_counts"][k]):
        print(f"    {old:>10}: {agg['trend_old_counts'][old]:>7}  "
              f"({_pct(agg['trend_old_counts'][old], n_trend)})")

    print("\n[4] Contradiction A: 'close within 1.5% of EMA50' vs 'range edge (<=30% or >=70%)'")
    print(f"    near-EMA50 bars:            {agg['near_ema']}")
    print(f"    ... of which also at edge:  {agg['near_ema_edge']} "
          f"({_pct(agg['near_ema_edge'], agg['near_ema'])})")

    print("\n[5] Contradiction B: 'ADX<=18' vs 'ATR% percentile > 30 (escapes SIDEWAYS)'")
    print(f"    ADX<=18 bars:                       {agg['low_adx']}")
    print(f"    ... of which ATR% rank > 30:        {agg['low_adx_high_atr_rank']} "
          f"({_pct(agg['low_adx_high_atr_rank'], agg['low_adx'])})")

    print("\n[6] Why RANGE is so rare: per-condition failure rate (evaluable bars only)")
    gate = agg["range_gate"]
    for key, label in (
        ("adx", "ADX <= 18 (range_adx_max)"),
        ("slope", "|EMA50 slope| <= 0.05% (range_slope_max_pct)"),
        ("distance", "|close/EMA50 - 1| <= 1.5% (range_max_distance_pct)"),
        ("structure", "structure == NEUTRAL"),
    ):
        ev, fail = gate[f"{key}_eval"], gate[f"{key}_fail"]
        print(f"    {label:<48} pass={ev - fail:>7}/{ev}  fail={_pct(fail, ev)}")
    print(f"    ALL FOUR pass (raw RANGE candidate):       "
          f"{gate['all_pass']}/{gate['all_eval']}  ({_pct(gate['all_pass'], gate['all_eval'])})")

    print("\n[7] Per-symbol RANGE counts (top 15)")
    ranked = sorted(per_symbol, key=lambda s: -s["new_state"]["RANGE"])
    for s in ranked[:15]:
        print(f"    {s['symbol'][:16]:>16}: RANGE={s['new_state']['RANGE']:>5}  "
              f"edge={s['range_old_edge']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"aggregate": agg, "per_symbol": per_symbol}, handle,
                      ensure_ascii=False, indent=2, default=str)
        print(f"\nWrote raw stats to {args.json_out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
