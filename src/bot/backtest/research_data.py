"""Read-only legacy inspection and immutable public research snapshots.

No credentials, private endpoints, database writes, or production state here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

import ccxt
import numpy as np
import pandas as pd

from bot.indicators import validate_ohlcv

HOUR_MS = 3_600_000
STREAMS = {
    "contract_1h": ("/fapi/v1/klines", "fapiPublicGetKlines", "1h", HOUR_MS),
    "contract_4h": ("/fapi/v1/klines", "fapiPublicGetKlines", "4h", 4 * HOUR_MS),
    "mark_1h": ("/fapi/v1/markPriceKlines", "fapiPublicGetMarkPriceKlines", "1h", HOUR_MS),
    "funding": ("/fapi/v1/fundingRate", "fapiPublicGetFundingRate", None, None),
}
PUBLIC_ROOT = "https://fapi.binance.com"
OHLCV = ["open", "high", "low", "close", "volume"]


def utc(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("an explicit timezone is required")
    return result.tz_convert("UTC")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_once(path, payload):
    path = Path(path)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def legacy_inventory(path):
    """Never instantiate KlineCache: its constructor writes the schema."""
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT symbol,timeframe,COUNT(*),MIN(timestamp_ms),MAX(timestamp_ms) "
            "FROM klines GROUP BY symbol,timeframe ORDER BY symbol,timeframe"
        ).fetchall()
    return {"path": str(path), "sha256": digest_file(path), "provenance": "unrecorded",
            "datasets": [{"symbol": s, "timeframe": tf, "rows": n,
                          "first_open": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
                          "last_open": pd.Timestamp(last, unit="ms", tz="UTC").isoformat()}
                         for s, tf, n, first, last in rows]}


def legacy_frame(path, symbol, timeframe):
    with sqlite3.connect(Path(path).resolve(strict=True).as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT timestamp_ms,open,high,low,close,volume FROM klines "
            "WHERE symbol=? AND timeframe=? ORDER BY timestamp_ms", (symbol, timeframe),
        ).fetchall()
    return ohlcv_frame(rows)


def ohlcv_frame(rows):
    frame = pd.DataFrame([row[:6] for row in rows], columns=["timestamp", *OHLCV])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for name in OHLCV:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame = frame.set_index("timestamp")
    if rows and len(rows[0]) >= 9:
        frame["quote_volume"] = [float(row[7]) for row in rows]
        frame["trade_count"] = [int(row[8]) for row in rows]
    return frame


def profile_bars(frame, timeframe_ms, *, start=None, end=None):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("bar opens must have explicit UTC timestamps")
    index = frame.index
    errors = []
    if index.has_duplicates:
        errors.append("duplicate_bar_open")
    if not index.is_monotonic_increasing:
        errors.append("unsorted_bar_open")
    if len(index) and (index.as_unit("ms").asi8 % timeframe_ms != 0).any():
        errors.append("misaligned_bar_open")
    delta = pd.Timedelta(timeframe_ms, unit="ms")
    gaps = int(((index[1:] - index[:-1]) != delta).sum())
    if gaps:
        errors.append("bar_gap")
    try:
        validate_ohlcv(frame)
        values = frame[OHLCV].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError("nonpositive price or invalid volume")
        for name in ("quote_volume", "trade_count"):
            if name in frame:
                values = frame[name].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"invalid {name}")
    except (TypeError, ValueError):
        errors.append("invalid_ohlcv")
    if not len(index):
        errors.append("empty")
    if start is not None and (not len(index) or index[0] != utc(start)):
        errors.append("start_not_covered")
    if end is not None and (not len(index) or index[-1] + delta != utc(end)):
        errors.append("end_not_covered")
    return {"rows": len(frame), "first_open": index[0].isoformat() if len(index) else None,
            "end_exclusive": (index[-1] + delta).isoformat() if len(index) else None,
            "duplicate_rows": int(index.duplicated().sum()), "gap_count": gaps,
            "errors": errors, "valid": not errors}


def aggregate_closed_bars(frame, source_hours, target_hours, *, as_of):
    """Reject partial/gapped UTC buckets; never forward-fill a missing candle."""
    if target_hours % source_hours or target_hours < source_hours:
        raise ValueError("target interval must be an integer multiple of source interval")
    source_ms = source_hours * HOUR_MS
    if not profile_bars(frame, source_ms)["valid"]:
        raise ValueError("source candles must be valid, unique, aligned and continuous")
    frame = frame.loc[frame.index + pd.Timedelta(source_hours, unit="h") <= utc(as_of)]
    rule = f"{target_hours}h"
    groups = frame.resample(rule, origin="epoch", closed="left", label="left")
    result = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    complete = groups["close"].count() == target_hours // source_hours
    result = result.loc[complete & (result.index + pd.Timedelta(target_hours, unit="h") <= utc(as_of))]
    return result


def compare_frames(left, right):
    common = left.index.intersection(right.index)
    if not len(common):
        return {"compared_rows": 0, "different_rows": 0, "comparable": False}
    matches = np.isclose(left.loc[common, OHLCV].to_numpy(float),
                         right.loc[common, OHLCV].to_numpy(float), rtol=1e-8, atol=1e-8).all(axis=1)
    differing = common[~matches]
    return {"compared_rows": len(common), "different_rows": int((~matches).sum()),
            "comparable": True, "first_difference": differing[0].isoformat() if len(differing) else None}


def _public_read(client, method, params):
    for attempt in range(3):
        try:
            return getattr(client, method)(params)
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection):
            raise
        except ccxt.NetworkError:
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


def fetch_public_rows(client, stream, symbol, start, end):
    _endpoint, method, interval, duration = STREAMS[stream]
    cursor, end_ms = int(utc(start).timestamp() * 1000), int(utc(end).timestamp() * 1000)
    rows, seen = [], {}
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000}
        if interval:
            params["interval"] = interval
        page = _public_read(client, method, params)
        if not isinstance(page, list):
            raise ValueError("public response must be an array")
        if not page:
            break
        latest = cursor - 1
        for row in page:
            timestamp = int(row[0] if interval else row["fundingTime"])
            if not cursor <= timestamp < end_ms:
                raise ValueError("public response lies outside its requested window")
            if not interval and row.get("symbol") != symbol:
                raise ValueError("funding symbol differs from the request")
            if timestamp in seen and seen[timestamp] != row:
                raise ValueError("conflicting public rows")
            if timestamp not in seen:
                rows.append(row)
                seen[timestamp] = row
            latest = max(latest, timestamp)
        if latest < cursor:
            raise ValueError("public pagination made no progress")
        cursor = latest + (duration or 1)
    return sorted(rows, key=lambda row: int(row[0] if interval else row["fundingTime"]))


def collect_snapshot(client, directory, symbols, start, end, *, progress=print):
    directory = Path(directory)
    start, end = utc(start), utc(end)
    if end <= start or start != start.floor("D") or end != end.floor("D"):
        raise ValueError("snapshot bounds must be ordered UTC midnights")
    if end > pd.Timestamp.now(tz="UTC").floor("h"):
        raise ValueError("snapshot must exclude incomplete and future candles")
    directory.mkdir(parents=True, exist_ok=True)
    symbols = sorted(set(symbols))
    if not symbols or any(re.fullmatch(r"[A-Z0-9]{1,30}USDT", symbol) is None for symbol in symbols):
        raise ValueError("research symbols must be explicit Binance USD-M identifiers")
    plan = {"schema": 1, "root": PUBLIC_ROOT, "symbols": symbols,
            "start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat()}
    plan_path = directory / "collection_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing snapshot plan differs; choose a new version directory")
    else:
        write_json_once(plan_path, plan)
    if (directory / "manifest.json").exists():
        return load_manifest(directory)
    metadata = _public_read(client, "fapiPublicGetExchangeInfo", {})
    markets = [row for row in metadata["symbols"] if row["symbol"] in plan["symbols"]]
    available = sorted({row["symbol"] for row in markets})
    excluded = {symbol: "absent_from_public_exchange_info"
                for symbol in plan["symbols"] if symbol not in available}
    if not available:
        raise ValueError("none of the requested symbols are in public exchange metadata")
    files = []
    for symbol in available:
        for stream in STREAMS:
            path = directory / f"{symbol}_{stream}.json.gz"
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload["symbol"] != symbol or payload["stream"] != stream or payload["plan"] != plan:
                    raise ValueError("saved public stream does not match collection plan")
            else:
                rows = fetch_public_rows(client, stream, symbol, start, end)
                payload = {"symbol": symbol, "stream": stream, "plan": plan,
                           "source": PUBLIC_ROOT + STREAMS[stream][0],
                           "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}
                # Write only after a whole stream succeeds. Never overwrite a version.
                with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                    handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
            files.append({"path": path.name, "symbol": symbol, "stream": stream,
                          "rows": len(payload["rows"]), "sha256": digest_file(path)})
            progress(f"public snapshot {len(files)}/{len(available) * len(STREAMS)} {symbol} {stream} {len(payload['rows'])}")
    metadata_path = directory / "exchange_info.json"
    if not metadata_path.exists():
        write_json_once(metadata_path, {"source": PUBLIC_ROOT + "/fapi/v1/exchangeInfo",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "symbols": markets})
    files.append({"path": metadata_path.name, "stream": "exchange_info", "sha256": digest_file(metadata_path)})
    manifest = {**plan, "symbols": available, "requested_symbols": plan["symbols"], "excluded": excluded,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": files, "universe_scope": "fixed_existing_cache_symbols; current metadata; survivorship bias"}
    write_json_once(directory / "manifest.json", manifest)
    return manifest


def load_manifest(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = {(s, stream) for s in manifest["symbols"] for stream in STREAMS}
    actual = {(row.get("symbol"), row["stream"]) for row in manifest["files"] if row["stream"] in STREAMS}
    if actual != expected or len(manifest["files"]) != len(expected) + 1:
        raise ValueError("snapshot manifest is incomplete or contains duplicate streams")
    expected_paths = {f"{symbol}_{stream}.json.gz" for symbol, stream in expected}
    expected_paths.add("exchange_info.json")
    if {row["path"] for row in manifest["files"]} != expected_paths:
        raise ValueError("snapshot manifest file names do not match expected streams")
    for row in manifest["files"]:
        if row["stream"] in STREAMS and row["path"] != f"{row['symbol']}_{row['stream']}.json.gz":
            raise ValueError("snapshot manifest maps a stream to the wrong file")
        path = (directory / row["path"]).resolve(strict=True)
        if path.parent != directory or digest_file(path) != row["sha256"]:
            raise ValueError("snapshot path or SHA256 verification failed")
    return manifest


def load_stream(directory, symbol, stream):
    with gzip.open(Path(directory) / f"{symbol}_{stream}.json.gz", "rt", encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    if stream != "funding":
        return ohlcv_frame(rows)
    frame = pd.DataFrame([{"timestamp": int(row["fundingTime"]),
                           "funding_rate": float(row["fundingRate"]),
                           "mark_price": float(row["markPrice"])} for row in rows],
                         columns=["timestamp", "funding_rate", "mark_price"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    if (frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not np.isfinite(frame.to_numpy()).all() or (frame["mark_price"] <= 0).any()):
        raise ValueError("funding history is incomplete or invalid")
    return frame


def chronological_splits(start, end, *, embargo_hours=48):
    start, end = utc(start), utc(end)
    if start != start.floor("D") or end != end.floor("D") or end - start < pd.Timedelta(30, unit="D"):
        raise ValueError("research requires at least 30 days with UTC midnight boundaries")
    if embargo_hours < 24:
        raise ValueError("embargo must cover at least the 24-hour maximum hold")
    days = (end - start).days
    cut1, cut2 = start + pd.Timedelta(days // 2, unit="D"), start + pd.Timedelta(days * 3 // 4, unit="D")
    embargo = pd.Timedelta(embargo_hours, unit="h")
    windows = [("train", start, cut1), ("validation", cut1 + embargo, cut2),
               ("test", cut2 + embargo, end)]
    if any(right <= left for _, left, right in windows):
        raise ValueError("embargo leaves an empty split")
    return [{"name": name, "start": left.isoformat(), "end_exclusive": right.isoformat(),
             "embargo_hours": embargo_hours, "genuinely_unseen": False}
            for name, left, right in windows]
