"""A/B replay for confirm_bars and the RANGE/MR redesign (funding=0 approximation).

Runs the shared portfolio backtest over spot 1H/4H candles from Binance's public
market-data mirror. Funding is approximated as zero (the sandbox cannot reach the
futures funding endpoint), so ABSOLUTE PnL is optimistic; the A/B deltas between
variants are what matter. The legacy mean-reversion trigger is compared against an
experimental EMA50-deviation trigger.

Run:  python scripts/run_ab_experiment.py [--symbols ...] [--days 120]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache
from bot.data.klines import ohlcv_frame

STEP_MS = {"1h": 3_600_000, "4h": 14_400_000}

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]


def _urlopen(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    delay = 2.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read().decode())
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _fetch(symbol: str, interval: str, days: int) -> list[list[float]]:
    step = STEP_MS[interval]
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600_000
    rows: dict[int, list[float]] = {}
    cursor = start_ms
    while cursor < now_ms:
        url = (
            "https://data-api.binance.vision/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&startTime={cursor}&limit=1000"
        )
        payload = _urlopen(url)
        if not payload:
            break
        for row in payload:
            timestamp = int(row[0])
            if timestamp + step > now_ms:
                continue
            if timestamp >= cursor:
                rows[timestamp] = [timestamp, *map(float, row[1:6])]
        next_cursor = int(payload[-1][0]) + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.4)
    return [rows[key] for key in sorted(rows)]


def _frame(cache: KlineCache, symbol: str, interval: str, days: int) -> pd.DataFrame:
    frame = cache.load(symbol, interval)
    if len(frame) < 100:
        cache.upsert(symbol, interval, _fetch(symbol, interval, days))
        frame = cache.load(symbol, interval)
    return frame


def _synth_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start - pd.Timedelta(days=1), end, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": 0.0, "mark_price": 0.0}, index=index)


def _by_source(trades):
    out: dict[str, dict[str, float | int]] = {}
    for trade in trades:
        row = out.setdefault(trade.source or "?", {"n": 0, "pnl": 0.0})
        row["n"] = int(row["n"]) + 1
        row["pnl"] = float(row["pnl"]) + float(trade.net_pnl)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--cache", default="data/market_spot_ab.sqlite")
    parser.add_argument("--variants", default="all",
                        help="comma-separated subset of: baseline,confirm3,range_relaxed,range_relaxed+mr_dev")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    cache = KlineCache(args.cache)

    bars_1h, bars_4h = {}, {}
    for symbol in symbols:
        try:
            bars_1h[symbol] = _frame(cache, symbol, "1h", args.days)
            bars_4h[symbol] = _frame(cache, symbol, "4h", args.days)
            time.sleep(0.3)
        except Exception as exc:
            print(f"  [skip] {symbol}: {type(exc).__name__} {str(exc)[:80]}")
    if not bars_1h:
        print("Not enough symbols fetched.")
        return 1
    market_bars_4h = _frame(cache, "BTCUSDT", "4h", args.days)
    onboard_ms = {symbol: 0 for symbol in bars_1h}

    end = pd.Timestamp.utcnow().floor("h")
    start = end - pd.Timedelta(days=args.days)
    funding = {symbol: _synth_funding(start, end) for symbol in bars_1h}

    # The backtest emits one huge evidence JSON per symbol-hour; suppress it so the
    # replay runs at compute speed instead of log I/O speed.
    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    variants = [
        ("baseline", {}),
        ("confirm3", {"confirm_bars": 3}),
        ("range_relaxed", {"range_structure_mode": "not_conflict", "range_slope_max_pct": 0.002}),
        ("range_relaxed+mr_dev", {
            "range_structure_mode": "not_conflict", "range_slope_max_pct": 0.002,
            "mr_entry_mode": "ema_deviation", "mr_deviation_pct": 0.02,
        }),
    ]
    if args.variants.strip().lower() != "all":
        wanted = {item.strip() for item in args.variants.split(",") if item.strip()}
        variants = [(label, over) for label, over in variants if label in wanted]

    print("=" * 100)
    print(f"A/B replay  (symbols={len(bars_1h)}, window={start.date()}..{end.date()}, "
          f"days={args.days}, funding=0 approximation)")
    print("=" * 100)
    rows = []
    for label, overrides in variants:
        settings = load_settings(env_file=None)
        settings.symbol_regime.confirm_bars = int(overrides.get("confirm_bars", settings.symbol_regime.confirm_bars))
        if "range_structure_mode" in overrides:
            settings.symbol_regime.range_structure_mode = overrides["range_structure_mode"]
        if "range_slope_max_pct" in overrides:
            settings.symbol_regime.range_slope_max_pct = overrides["range_slope_max_pct"]
        mr_mode = overrides.get("mr_entry_mode", "legacy")
        mr_dev = float(overrides.get("mr_deviation_pct", 0.02))
        result = run_portfolio_backtest(
            bars_1h, bars_4h, funding, onboard_ms, settings,
            start=start, end=end, top_n=20, initial_equity=10_000.0,
            enable_trend_routing=True, market_bars_4h=market_bars_4h,
            mr_entry_mode=mr_mode, mr_deviation_pct=mr_dev,
        )
        m = result.metrics
        src = _by_source(result.trades)
        rows.append((label, m, src))
        print(f"\n[{label}]")
        print(f"    trades={m.get('trade_count')}  net_pnl={m.get('net_pnl'):.2f}  "
              f"return={m.get('total_return_pct')}%  maxDD={m.get('max_drawdown_pct')}%  "
              f"PF={m.get('profit_factor')}")
        for source, stat in sorted(src.items()):
            print(f"        {source:<16} n={stat['n']:>4}  pnl={stat['pnl']:>10.2f}")
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
