"""One-shot data extension to 400 days for the mr-trend mean-reversion project.

Self-contained (stdlib + pandas only) so it runs in the sandbox without the
project venv. Two jobs:

  1. Backfill daily (1h, 4h) klines for the existing 31-symbol universe into
     data/market.sqlite, paginating the public Binance spot klines endpoint
     (data-api.binance.vision) so the cache spans ~400 days.
  2. Backfill funding settlements from the public USDT-M fapi endpoint and
     store them in data/ab_funding.pkl using the exact format consumed by
     scripts/run_range_ab.py (dict[symbol -> DataFrame indexed by UTC ts with
     columns funding_rate, mark_price]).

Symbol mapping: the spot cache uses PEPEUSDT, but the USDT-M perpetual is
1000PEPEUSDT (a 1000x divisor). Funding is pulled for the futures symbol and
stored under the spot symbol so it joins the spot cache on the same key.

Purely additive and idempotent: kline rows upsert by primary key, funding is
extended and re-sorted with duplicate-timestamp drop. Safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

DB = "data/market.sqlite"
FUNDING_PKL = "data/ab_funding.pkl"
DAYS = 400
KLINES_BASE = "https://data-api.binance.vision/api/v3/klines"
FUNDING_BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
PAGE = 1000

# futures symbol for a few spot names that differ (1000x divisor contracts).
FUTURES_ALIAS = {
    "PEPEUSDT": "1000PEPEUSDT",
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def urlopen_json(url: str, retries: int = 4) -> list | dict:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                if resp.status == 429 or resp.status == 418:
                    time.sleep(2 * (attempt + 1))
                    continue
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - transient network
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"urlopen failed for {url}: {last}")


def fetch_klines(symbol: str, interval: str, days: int) -> list[list]:
    """Return kline rows [[ts_ms,o,h,l,c,vol],...] oldest->newest over `days`."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    out: list[list] = []
    cursor = end_ms
    while True:
        # endTime-only walks back PAGE bars from `cursor` (most recent first);
        # passing startTime makes the endpoint return the oldest PAGE bars instead.
        url = f"{KLINES_BASE}?symbol={symbol}&interval={interval}&endTime={cursor}&limit={PAGE}"
        rows = urlopen_json(url)
        if not isinstance(rows, list) or not rows:
            break
        out.extend(rows)
        oldest = int(rows[0][0])
        if oldest <= start_ms or len(rows) < PAGE:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    # dedupe + sort by ts
    seen: dict[int, list] = {}
    for r in out:
        seen[int(r[0])] = r
    return [seen[k] for k in sorted(seen)]


def upsert_klines(symbol: str, timeframe: str, rows: list[list], db: str = DB) -> int:
    if not rows:
        return 0
    con = sqlite3.connect(db, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    con.executemany(
        """
        INSERT INTO klines(symbol, timeframe, timestamp_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, timestamp_ms) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """,
        [(symbol, timeframe, int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows],
    )
    con.commit()
    con.close()
    return len(rows)


def fetch_funding(fut_symbol: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows: list[tuple[int, float, float]] = []
    cursor = end_ms
    while True:
        # endTime-only walks back PAGE settlements from `cursor` (most recent
        # first); passing startTime makes the endpoint return the oldest PAGE.
        url = f"{FUNDING_BASE}?symbol={fut_symbol}&endTime={cursor}&limit={PAGE}"
        payload = urlopen_json(url)
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            rows.append((int(item["fundingTime"]), float(item["fundingRate"]), float(item["markPrice"])))
        oldest = int(payload[0]["fundingTime"])
        if oldest <= start_ms or len(payload) < PAGE:
            break
        cursor = oldest - 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])
    frame = pd.DataFrame(rows, columns=["ts", "funding_rate", "mark_price"])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return frame.set_index("ts").sort_index()


def merge_funding(frame: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return new
    merged = pd.concat([frame, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="AUTO", help="comma-separated symbols or AUTO (all cached symbols)")
    parser.add_argument("--timeframes", default="1h,4h", help="comma-separated timeframes to backfill")
    parser.add_argument("--skip-funding", action="store_true", help="do not pull funding")
    parser.add_argument("--db", default=DB)
    parser.add_argument("--funding-pkl", default=FUNDING_PKL)
    args = parser.parse_args(argv)

    con = sqlite3.connect(args.db)
    all_symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM klines WHERE timeframe='4h' ORDER BY symbol"
    ).fetchall()]
    con.close()
    if args.symbols.strip().upper() == "AUTO":
        symbols = all_symbols
    else:
        requested = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbols = [s for s in requested if s in all_symbols]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    log(f"universe {len(symbols)} symbols -> {symbols}; timeframes {timeframes}")

    funding: dict[str, pd.DataFrame] = {}
    if not args.skip_funding:
        import os
        if os.path.exists(args.funding_pkl):
            with open(args.funding_pkl, "rb") as fh:
                funding = pickle.load(fh)
            log(f"loaded funding pickle with {len(funding)} symbols")

    for i, sym in enumerate(symbols, 1):
        # klines backfill
        for tf in timeframes:
            try:
                rows = fetch_klines(sym, tf, DAYS)
                n = upsert_klines(sym, tf, rows, db=args.db)
                log(f"[{i}/{len(symbols)}] klines {sym} {tf}: +{n} rows (now cached)")
            except Exception as exc:  # noqa: BLE001
                log(f"[{i}/{len(symbols)}] klines {sym} {tf} ERR {type(exc).__name__}: {str(exc)[:120]}")

        # funding backfill
        if not args.skip_funding:
            fut = FUTURES_ALIAS.get(sym, sym)
            try:
                new = fetch_funding(fut, DAYS)
                old = funding.get(sym, pd.DataFrame(columns=["funding_rate", "mark_price"]))
                funding[sym] = merge_funding(old, new)
                log(f"[{i}/{len(symbols)}] funding {sym} ({fut}): {len(new)} new -> total {len(funding[sym])}")
            except Exception as exc:  # noqa: BLE001
                log(f"[{i}/{len(symbols)}] funding {sym} ({fut}) ERR {type(exc).__name__}: {str(exc)[:120]}")

    if not args.skip_funding:
        with open(args.funding_pkl, "wb") as fh:
            pickle.dump(funding, fh)
        log(f"wrote {args.funding_pkl} with {len(funding)} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
