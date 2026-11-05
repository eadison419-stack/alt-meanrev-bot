"""Confirm-bar sensitivity for the TREND regime gate (read-only, offline).

For confirm_bars in {2,3,4,5}, count how many bars confirm TREND and measure how
stable those confirms are (do they still confirm on the next bar?). The review's
Priority 2 asks exactly this before committing to a PnL replay: if lowering
confirm_bars only adds states that flip straight back, reject the change.

Run:  python scripts/diagnose_confirm_bars.py [--symbols ...] [--days 120]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import pandas as pd

from bot.config import load_settings
from bot.data.klines import TIMEFRAME_MS, ohlcv_frame
from bot.symbol_regime import _raw_candidates

FOUR_HOURS_MS = TIMEFRAME_MS["4h"]

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]

TREND_UP, TREND_DOWN = "TREND_UP", "TREND_DOWN"


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


def _trend_confirmed(candidates: list[str], *, n: int) -> list[bool]:
    """bar i confirms TREND when the last n candidates are one same TREND direction."""
    out = []
    for i, _ in enumerate(candidates):
        if i + 1 < n:
            out.append(False)
            continue
        window = candidates[i - n + 1:i + 1]
        out.append(len(set(window)) == 1 and window[0] in (TREND_UP, TREND_DOWN))
    return out


def _continuation(confirmed: list[bool]) -> float | None:
    """P(confirmed at bar i+1 | confirmed at bar i)."""
    numerator = denominator = 0
    for i in range(len(confirmed) - 1):
        if confirmed[i]:
            denominator += 1
            if confirmed[i + 1]:
                numerator += 1
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()
    settings = load_settings(env_file=None)
    policy = settings.symbol_regime
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]

    per_symbol = []
    for symbol in symbols:
        try:
            rows = _fetch_4h(symbol, days=args.days)
            frame = ohlcv_frame(rows)
        except Exception as exc:
            print(f"  [skip] {symbol}: {type(exc).__name__} {str(exc)[:80]}")
            continue
        if len(frame) < 120:
            continue
        candidates, _ = _raw_candidates(frame, policy)
        per_symbol.append({"symbol": symbol, "candidates": candidates})

    if not per_symbol:
        print("No symbols produced usable data.")
        return 1

    windows = (2, 3, 4, 5)
    totals = {n: 0 for n in windows}
    cont_sums = {n: 0.0 for n in windows}
    cont_counts = {n: 0 for n in windows}
    # Marginal bars unlocked by going 4 -> 3, and their next-bar continuation.
    unlocked_3 = 0
    unlocked_3_continue = 0
    total_bars = 0

    for item in per_symbol:
        cand = item["candidates"]
        total_bars += len(cand)
        confirmed = {}
        for n in windows:
            confirmed[n] = _trend_confirmed(cand, n=n)
            count = sum(confirmed[n])
            totals[n] += count
            for i in range(len(confirmed[n]) - 1):
                if confirmed[n][i]:
                    cont_counts[n] += 1
                    if confirmed[n][i + 1]:
                        cont_sums[n] += 1
        for i in range(len(cand)):
            if confirmed[3][i] and not confirmed[4][i]:
                unlocked_3 += 1
                if i + 1 < len(cand) and confirmed[3][i + 1]:
                    unlocked_3_continue += 1

    print("=" * 72)
    print(f"Confirm-bar sensitivity (symbols={len(per_symbol)}, bars={total_bars}, "
          f"days={args.days}, source=Binance data mirror 4H)")
    print("=" * 72)
    print(f"\n{'confirm_bars':>12} {'TREND-confirmed bars':>22} {'next-bar continuation':>22}")
    for n in windows:
        rate = (cont_sums[n] / cont_counts[n] * 100) if cont_counts[n] else None
        print(f"{n:>12} {totals[n]:>22} {('%.2f%%' % rate) if rate is not None else 'n/a':>22}")

    base = totals[4]
    new3 = totals[3]
    print(f"\nMarginal effect of 4 -> 3:")
    print(f"    TREND-confirmed bars: {base} -> {new3}  (+{new3 - base}, "
          f"+{(new3 - base) / base * 100:.1f}%)" if base else "    no TREND bars at 4")
    if unlocked_3:
        cont = unlocked_3_continue / unlocked_3 * 100
        print(f"    bars newly unlocked by 4->3: {unlocked_3}")
        print(f"    ... of which still TREND-confirmed(3) next bar: {unlocked_3_continue} ({cont:.1f}%)")
        print(f"    (baseline next-bar continuation at n=4: "
              f"{cont_sums[4] / cont_counts[4] * 100:.1f}%)" if cont_counts[4] else "")
    else:
        print("    (no bars change between n=4 and n=3)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
