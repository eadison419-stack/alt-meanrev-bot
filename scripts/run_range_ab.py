"""RANGE redesign A/B with real funding (fapi.binance.com) and two OOS windows.

Compares four variants across two non-overlapping 60-day windows over the full
30-symbol universe. Klines come from the local spot mirror cache; funding rates
are real, fetched from Binance USDT-M futures and cached to a pickle.

Variants (all inherit confirm_bars=3, the adopted production value):
  baseline  : range_structure=neutral, slope=0.05%, trend_slope=0.10%, legacy MR
  A         : range_structure=not_conflict, slope=0.20%, trend_slope=0.30%, ema_deviation(2%)
  A_legacy  : same range/trend as A, but legacy LOW/HIGH MR trigger (isolates the trigger)
  B         : range_structure=not_conflict, slope=0.08%, trend_slope=0.10%, ema_deviation(2%)
              (conservative: keeps trend threshold untouched, passes validator)
  B+        : same as B, but ema_deviation threshold relaxed 2% -> 1% to raise MR frequency
  B++       : same as B, but ema_deviation threshold relaxed to 0.5% (frequency probe)
  B+_adx20  : same as B+, but range_adx_max 18 -> 20 (aligns with legacy SIDEWAYS<20; probe)

Run:  python scripts/run_range_ab.py --window 1 --variant A
      python scripts/run_range_ab.py --window 2 --variant all
      python scripts/run_range_ab.py --funding-only

Findings so far (2026-10-23, 30 symbols x 2 non-overlapping 30-day windows, real fapi funding):
  - Trend dominates PnL and is strongly regime-dependent: W1 (Aug24-Sep23) TREND +934,
    W2 (Sep23-Oct23) TREND -582. MR is a rounding error so far (1-7 trades vs 60-77 TREND).
  - trend_slope_min 0.10% -> 0.30% is nearly free: TREND count 77->76 (W1), 60->59 (W2).
  - slope 0.08% vs 0.20% is trade-identical: A and B produce byte-for-byte the same MR trades,
    so 0.20% adds RANGE bars (450->982) but no trades, and requires breaking the
    trend>range slope validator. Drop it; keep slope <= 0.08%, never touch trend_slope.
  - ema_deviation beats legacy LOW/HIGH in both windows (W2: +78 on 2 trades vs -54 on 7).
  - B+ (deviation 1%) tripled MR frequency vs B (2%): 3 -> 9 MR trades over both windows,
    without touching TREND (76/60 same as B). It is the only config with positive MR in
    both windows (+69 / +77), but W1's +69 rests on one ADA short (+97.30); strip that and
    W1 MR turns negative. Still a hint, not evidence: MR n << 30 OOS trades.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import urllib.request
from pathlib import Path

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]

WINDOW_DAYS = 60

VARIANTS = {
    "baseline": {
        "range_structure_mode": "neutral",
        "range_slope_max_pct": 0.0005,
        "trend_slope_min_pct": 0.001,
        "mr_entry_mode": "legacy",
    },
    "A": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.002,
        "trend_slope_min_pct": 0.003,
        "mr_entry_mode": "ema_deviation",
        "mr_deviation_pct": 0.02,
    },
    "A_legacy": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.002,
        "trend_slope_min_pct": 0.003,
        "mr_entry_mode": "legacy",
    },
    "B": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.0008,
        "trend_slope_min_pct": 0.001,
        "mr_entry_mode": "ema_deviation",
        "mr_deviation_pct": 0.02,
    },
    "B+": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.0008,
        "trend_slope_min_pct": 0.001,
        "mr_entry_mode": "ema_deviation",
        "mr_deviation_pct": 0.01,
    },
    "B++": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.0008,
        "trend_slope_min_pct": 0.001,
        "mr_entry_mode": "ema_deviation",
        "mr_deviation_pct": 0.005,
    },
    "B+_adx20": {
        "range_structure_mode": "not_conflict",
        "range_slope_max_pct": 0.0008,
        "trend_slope_min_pct": 0.001,
        "range_adx_max": 20,
        "mr_entry_mode": "ema_deviation",
        "mr_deviation_pct": 0.01,
    },
}


def _urlopen(url: str) -> list:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    delay = 2.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read().decode())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_funding(symbol: str, days: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days + 2) * 24 * 3600_000
    payload = _urlopen(
        f"https://fapi.binance.com/fapi/v1/fundingRate"
        f"?symbol={symbol}&startTime={start_ms}&limit=1000"
    )
    rows = [
        (int(item["fundingTime"]), float(item["fundingRate"]), float(item["markPrice"]))
        for item in payload
    ]
    frame = pd.DataFrame(rows, columns=["ts", "funding_rate", "mark_price"])
    if frame.empty:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return frame.set_index("ts").sort_index()


def load_or_fetch_funding(symbols: list[str], days: int, path: Path) -> dict[str, pd.DataFrame]:
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    funding: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            funding[symbol] = fetch_funding(symbol, days)
            print(f"  funding {symbol}: {len(funding[symbol])} settlements", flush=True)
        except Exception as exc:
            print(f"  funding ERR {symbol}: {type(exc).__name__} {str(exc)[:90]}", flush=True)
            funding[symbol] = pd.DataFrame(columns=["funding_rate", "mark_price"])
        time.sleep(0.25)
    with path.open("wb") as handle:
        pickle.dump(funding, handle)
    return funding


def _by_source(trades):
    out: dict[str, dict[str, float | int]] = {}
    for trade in trades:
        row = out.setdefault(trade.source or "?", {"n": 0, "pnl": 0.0})
        row["n"] = int(row["n"]) + 1
        row["pnl"] = float(row["pnl"]) + float(trade.net_pnl)
    return out


def run_window(symbols, bars_1h, bars_4h, funding, market_bars_4h, settings_base,
               window_index: int, end: pd.Timestamp, variant_label: str,
               window_days: int):
    settings = settings_base.model_copy(deep=True)
    overrides = VARIANTS[variant_label]
    settings.symbol_regime.confirm_bars = 3
    settings.symbol_regime.range_structure_mode = overrides["range_structure_mode"]
    settings.symbol_regime.range_slope_max_pct = overrides["range_slope_max_pct"]
    settings.symbol_regime.trend_slope_min_pct = overrides["trend_slope_min_pct"]
    if "range_adx_max" in overrides:
        settings.symbol_regime.range_adx_max = overrides["range_adx_max"]

    if window_index == 1:
        # Older window.
        start = end - pd.Timedelta(days=2 * window_days)
        stop = end - pd.Timedelta(days=window_days)
    else:
        # Most recent window.
        start = end - pd.Timedelta(days=window_days)
        stop = end
    onboard_ms = {s: 0 for s in symbols}

    result = run_portfolio_backtest(
        bars_1h, bars_4h, funding, onboard_ms, settings,
        start=start, end=stop, top_n=20, initial_equity=10_000.0,
        enable_trend_routing=True, market_bars_4h=market_bars_4h,
        mr_entry_mode=overrides.get("mr_entry_mode", "legacy"),
        mr_deviation_pct=float(overrides.get("mr_deviation_pct", 0.02)),
    )
    m = result.metrics
    src = _by_source(result.trades)
    print(f"[{variant_label} | window {window_index} {start.date()}..{stop.date()}]", flush=True)
    print(f"    trades={m.get('trade_count')}  net_pnl={m.get('net_pnl'):.2f}  "
          f"return={m.get('total_return_pct')}%  maxDD={m.get('max_drawdown_pct')}%  "
          f"PF={m.get('profit_factor')}", flush=True)
    for source, stat in sorted(src.items()):
        print(f"        {source:<16} n={stat['n']:>4}  pnl={stat['pnl']:>10.2f}", flush=True)
    for trade in result.trades:
        if (trade.source or "") == "MEAN_REVERSION":
            print(f"        MR  {trade.symbol:<16} {trade.opened_at.date()}  "
                  f"side={trade.side.value:<5} pnl={trade.net_pnl:>8.2f}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--window", type=int, default=1, choices=(1, 2),
                        help="1 = older 60 days, 2 = most recent 60 days")
    parser.add_argument("--variant", default="all")
    parser.add_argument("--cache", default="/tmp/ab_market.sqlite")
    parser.add_argument("--funding-cache", default="/tmp/ab_funding.pkl")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS,
                        help="days per non-overlapping window (default 60; use 30 in sandbox)")
    parser.add_argument("--funding-only", action="store_true")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    cache = KlineCache(args.cache)
    funding = load_or_fetch_funding(symbols, 2 * args.window_days, Path(args.funding_cache))
    if args.funding_only:
        return 0

    bars_1h, bars_4h = {}, {}
    for s in symbols:
        bars_1h[s] = cache.load(s, "1h")
        bars_4h[s] = cache.load(s, "4h")
    market_bars_4h = cache.load("BTCUSDT", "4h")

    # Suppress the per-symbol-hour evidence JSON; it is log I/O bound.
    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    settings = load_settings(env_file=None)
    end = pd.Timestamp.utcnow().floor("h")
    variants = (["baseline", "A", "A_legacy", "B", "B+", "B++", "B+_adx20"] if args.variant == "all"
                else [v.strip() for v in args.variant.split(",") if v.strip()])

    for label in variants:
        t0 = time.perf_counter()
        run_window(symbols, bars_1h, bars_4h, funding, market_bars_4h, settings,
                   args.window, end, label, args.window_days)
        if args.time:
            print(f"    (elapsed {time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
