"""Single-variable sensitivity sweep of the RANGE gate (read-only, offline).

For each of the four RANGE conditions, relax it one at a time and measure:
  - how many bars become RANGE candidates / confirmed RANGE (2 consecutive bars), and
  - of those, what fraction the LEGACY classifier labels LOW/HIGH (the only legacy
    states that can generate a mean-reversion entry), vs SIDEWAYS (blocks entry).

This answers "which condition to loosen, and by how much", without touching
production code. The four conditions are computed bar-by-bar from completed 4H
candles; only the threshold/structure-mode varies.

Run:  python scripts/diagnose_range_sensitivity.py [--symbols ...] [--days 120]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

import numpy as np
import pandas as pd

from bot.config import load_settings
from bot.data.klines import TIMEFRAME_MS, ohlcv_frame
from bot.indicators import indicator_frame
from bot.regime import classify_regimes
from bot.symbol_regime import _structure

FOUR_HOURS_MS = TIMEFRAME_MS["4h"]

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]

# structure_mode: which _structure states satisfy the RANGE gate.
#   "neutral"       -> only NEUTRAL (current production)
#   "not_conflict"  -> NEUTRAL/UP/DOWN (anything except CONFLICT)
#   "any"           -> ignore structure entirely
STRUCTURE_MODES = ("neutral", "not_conflict", "any")


def _fetch_4h(symbol: str, *, days: int) -> list[list[float]]:
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
        if open_time + FOUR_HOURS_MS > now_ms:
            continue
        rows.append([open_time, *map(float, row[1:6])])
    return rows


def _structure_ok(state: str | None, mode: str) -> bool:
    if state is None:
        return False
    if mode == "neutral":
        return state == "NEUTRAL"
    if mode == "not_conflict":
        return state in ("NEUTRAL", "UP", "DOWN")
    return True  # "any"


def _per_symbol_arrays(closed: pd.DataFrame, settings) -> dict:
    """Compute the four parameter-independent condition inputs per bar."""
    policy = settings.symbol_regime
    adx = indicator_frame(closed, policy.adx_period, policy.adx_period)["adx"].to_numpy(dtype=float)
    close = closed["close"].to_numpy(dtype=float)
    ema_slow = closed["close"].ewm(
        span=policy.ema_slow_bars, adjust=False, min_periods=policy.ema_slow_bars,
    ).mean().to_numpy(dtype=float)

    n = len(closed)
    slope = np.full(n, np.nan)
    for i in range(policy.slope_lookback_bars, n):
        ref, cur = ema_slow[i - policy.slope_lookback_bars], ema_slow[i]
        if math.isfinite(ref) and math.isfinite(cur) and ref > 0:
            slope[i] = cur / ref - 1.0
    distance = close / ema_slow - 1.0
    structure = [None] * n
    for i in range(2 * policy.structure_lookback_bars - 1, n):
        structure[i] = _structure(closed, i, policy.structure_lookback_bars)[0]

    legacy = classify_regimes(closed, settings)
    legacy_regime = legacy["regime"].map(lambda v: v.value if v is not None else None).to_numpy()

    valid = np.ones(n, dtype=bool)
    valid &= np.isfinite(adx)
    valid &= np.isfinite(slope)
    valid &= np.isfinite(distance)
    valid &= np.array([s is not None for s in structure])
    return {"adx": adx, "slope": slope, "distance": distance, "structure": structure,
            "legacy_regime": legacy_regime, "valid": valid}


def _range_confirmed(arrays: dict, *, adx_max: float, slope_max: float,
                     distance_max: float, structure_mode: str) -> np.ndarray:
    adx_ok = arrays["adx"] <= adx_max
    slope_ok = np.abs(arrays["slope"]) <= slope_max
    distance_ok = np.abs(arrays["distance"]) <= distance_max
    structure_ok = np.array([_structure_ok(s, structure_mode) for s in arrays["structure"]])
    candidate = arrays["valid"] & adx_ok & slope_ok & distance_ok & structure_ok
    confirmed = candidate.copy()
    confirmed[1:] &= candidate[:-1]  # 2 consecutive candidate bars
    return confirmed


def _evaluate(confirmed: np.ndarray, arrays: dict) -> dict:
    idx = np.flatnonzero(confirmed)
    n_confirmed = int(confirmed.sum())
    if n_confirmed == 0:
        return {"confirmed": 0, "low_high": 0, "sideways": 0, "mid": 0}
    regs = arrays["legacy_regime"][idx]
    regs = regs[regs != None]  # noqa: E711
    low_high = int(np.isin(regs, ["LOW", "HIGH"]).sum())
    sideways = int((regs == "SIDEWAYS").sum())
    mid = int((regs == "MID").sum())
    return {"confirmed": n_confirmed, "low_high": low_high, "sideways": sideways, "mid": mid}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()
    settings = load_settings(env_file=None)
    policy = settings.symbol_regime
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]

    arrays_list = []
    total_bars = 0
    for symbol in symbols:
        try:
            frame = ohlcv_frame(_fetch_4h(symbol, days=args.days))
        except Exception as exc:
            print(f"  [skip] {symbol}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        if len(frame) < 120:
            continue
        arrays_list.append(_per_symbol_arrays(frame, settings))
        total_bars += len(frame)
    if not arrays_list:
        print("No symbols produced usable data.")
        return 1

    def merge(arrays_attr: str) -> np.ndarray:
        return np.concatenate([a[arrays_attr] for a in arrays_list])

    merged = {
        "adx": merge("adx"), "slope": merge("slope"), "distance": merge("distance"),
        "structure": merge("structure"), "legacy_regime": merge("legacy_regime"),
        "valid": merge("valid"),
    }

    base = dict(
        adx_max=float(policy.range_adx_max),
        slope_max=float(policy.range_slope_max_pct),
        distance_max=float(policy.range_max_distance_pct),
        structure_mode="neutral",
    )

    def run_and_fmt(overrides: dict) -> tuple[dict, str]:
        params = {**base, **overrides}
        confirmed = _range_confirmed(merged, **params)
        result = _evaluate(confirmed, merged)
        n = result["confirmed"]
        lh_pct = 100 * result["low_high"] / n if n else 0.0
        label = ", ".join(
            f"{k.replace('structure_mode', 'structure')}={v}" for k, v in overrides.items()
        ) or "baseline"
        return result, (f"{label:<34} RANGE={n:>6}  LOW/HIGH={result['low_high']:>5} "
                        f"({lh_pct:>5.1f}%)  SIDEWAYS={result['sideways']:>5}  MID={result['mid']:>5}")

    print("=" * 96)
    print(f"RANGE sensitivity sweep  (symbols={len(arrays_list)}, bars={total_bars}, days={args.days})")
    print("    'RANGE' = bars confirmed RANGE (2 consecutive candidate bars);")
    print("    'LOW/HIGH' = the only legacy states that can open an MR entry; SIDEWAYS blocks.")
    print("=" * 96)

    print("\n[baseline] current config")
    _, line = run_and_fmt({})
    print("   ", line)

    print("\n[structure] relax structure gate (other conditions at baseline)")
    for mode in STRUCTURE_MODES:
        _, line = run_and_fmt({"structure_mode": mode})
        print("   ", line)

    print("\n[slope] relax EMA50-slope gate (other conditions at baseline)")
    for sm in (0.001, 0.002, 0.005, 0.01):
        _, line = run_and_fmt({"slope_max": sm})
        print("   ", line)

    print("\n[distance] relax close-to-EMA50 gate (other conditions at baseline)")
    for dm in (0.02, 0.03, 0.05, 0.08):
        _, line = run_and_fmt({"distance_max": dm})
        print("   ", line)

    print("\n[adx] relax ADX upper bound (other conditions at baseline)")
    for am in (20, 22, 25):
        _, line = run_and_fmt({"adx_max": am})
        print("   ", line)

    print("\n[combined] promising two-way relaxations")
    for overrides in (
        {"structure_mode": "not_conflict", "slope_max": 0.002},
        {"structure_mode": "not_conflict", "slope_max": 0.005},
        {"structure_mode": "not_conflict", "distance_max": 0.03},
        {"structure_mode": "any", "slope_max": 0.005},
        {"slope_max": 0.005, "adx_max": 20},
    ):
        _, line = run_and_fmt(overrides)
        print("   ", line)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
