"""SQLite-backed, idempotent cache for exchange OHLCV bars."""

import sqlite3
from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class KlineCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp_ms)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_klines_lookup ON klines(symbol, timeframe, timestamp_ms)")

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def upsert(self, symbol: str, timeframe: str, rows: list[list[float]]) -> int:
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany("""
                INSERT INTO klines(symbol, timeframe, timestamp_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, timestamp_ms) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume
            """, [(symbol, timeframe, int(row[0]), *map(float, row[1:6])) for row in rows])
        return len(rows)

    def load(self, symbol: str, timeframe: str, *, since_ms: int | None = None, until_ms: int | None = None) -> pd.DataFrame:
        clauses, params = ["symbol = ?", "timeframe = ?"], [symbol, timeframe]
        if since_ms is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(int(since_ms))
        if until_ms is not None:
            clauses.append("timestamp_ms < ?")
            params.append(int(until_ms))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT timestamp_ms, open, high, low, close, volume FROM klines WHERE {' AND '.join(clauses)} ORDER BY timestamp_ms",
                params,
            ).fetchall()
        frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        if frame.empty:
            frame = pd.DataFrame(columns=OHLCV_COLUMNS[1:])
            frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        return frame.set_index("timestamp")
