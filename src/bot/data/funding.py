"""Historical USDT-M funding-rate download and CSV cache helpers."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd


class FundingExchange(Protocol):
    def fetch_funding_rate_history(self, symbol: str, since: int | None = None, limit: int | None = None): ...


FUNDING_COLUMNS = ["timestamp_ms", "funding_rate", "mark_price"]


def fetch_funding_rates(
    exchange: FundingExchange,
    symbol: str,
    *,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
) -> list[dict[str, float | int | None]]:
    """Fetch timestamp-ordered, de-duplicated historical settlements in [since, until)."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    end = until_ms if until_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = int(since_ms)
    rows: dict[int, dict[str, float | int | None]] = {}
    while cursor < end:
        page = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=limit)
        if not page:
            break
        timestamps = [int(item["timestamp"]) for item in page]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("exchange returned unordered or duplicate funding timestamps")
        for item in page:
            timestamp = int(item["timestamp"])
            if cursor <= timestamp < end:
                info = item.get("info") or {}
                raw_mark = item.get("markPrice", info.get("markPrice"))
                rows[timestamp] = {
                    "timestamp_ms": timestamp,
                    "funding_rate": float(item["fundingRate"]),
                    "mark_price": float(raw_mark) if raw_mark not in (None, "") else None,
                }
        next_cursor = timestamps[-1] + 1
        if next_cursor <= cursor:
            raise ValueError("funding pagination did not advance")
        if timestamps[-1] >= end - 1 or len(page) < limit:
            break
        cursor = next_cursor
    return [rows[timestamp] for timestamp in sorted(rows)]


def upsert_funding_csv(path: str | Path, rows: list[dict[str, float | int | None]]) -> int:
    """Merge funding rows by timestamp, preserving prior cached records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[int, dict[str, float | int | None]] = {}
    if destination.exists():
        with destination.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                timestamp = int(row["timestamp_ms"])
                merged[timestamp] = {
                    "timestamp_ms": timestamp,
                    "funding_rate": float(row["funding_rate"]),
                    "mark_price": float(row["mark_price"]) if row.get("mark_price") else None,
                }
    for row in rows:
        timestamp = int(row["timestamp_ms"])
        merged[timestamp] = {
            "timestamp_ms": timestamp,
            "funding_rate": float(row["funding_rate"]),
            "mark_price": float(row["mark_price"]) if row.get("mark_price") is not None else None,
        }
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FUNDING_COLUMNS)
        writer.writeheader()
        writer.writerows(merged[timestamp] for timestamp in sorted(merged))
    return len(merged)


def load_funding_csv(path: str | Path) -> pd.DataFrame:
    """Load a funding CSV into a UTC-indexed frame with unique ordered timestamps."""
    frame = pd.read_csv(path)
    missing = set(FUNDING_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"funding CSV is missing columns: {sorted(missing)}")
    if frame["timestamp_ms"].duplicated().any():
        raise ValueError("funding timestamps must be unique")
    frame["timestamp_ms"] = frame["timestamp_ms"].astype("int64")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame["mark_price"] = pd.to_numeric(frame["mark_price"], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("funding timestamps must be strictly increasing and unique")
    return frame
