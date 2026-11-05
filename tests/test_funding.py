import pandas as pd
import pytest

from bot.data.funding import fetch_funding_rates, load_funding_csv, upsert_funding_csv


class FakeFundingExchange:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        self.calls.append((symbol, since, limit))
        return [row for row in self.rows if row["timestamp"] >= since][:limit]


def item(timestamp, rate=0.0001, mark="100"):
    return {"timestamp": timestamp, "fundingRate": rate, "info": {"markPrice": mark}}


def test_fetch_funding_rates_paginates_and_applies_half_open_range():
    exchange = FakeFundingExchange([item(100), item(200), item(300), item(400), item(500)])

    rows = fetch_funding_rates(exchange, "ADA/USDT:USDT", since_ms=100, until_ms=450, limit=3)

    assert [row["timestamp_ms"] for row in rows] == [100, 200, 300, 400]
    assert len(exchange.calls) == 2
    assert exchange.calls[1][1] == 301
    assert rows[0]["mark_price"] == 100.0


def test_fetch_funding_rates_rejects_unordered_pages():
    exchange = FakeFundingExchange([item(200), item(100)])
    with pytest.raises(ValueError, match="unordered"):
        fetch_funding_rates(exchange, "ADA/USDT:USDT", since_ms=0, until_ms=300)


def test_funding_csv_merge_is_idempotent_and_loads_utc_timestamps(tmp_path):
    path = tmp_path / "funding.csv"
    first = [
        {"timestamp_ms": 1000, "funding_rate": 0.0001, "mark_price": 100},
        {"timestamp_ms": 2000, "funding_rate": -0.0002, "mark_price": None},
    ]
    assert upsert_funding_csv(path, first) == 2
    assert upsert_funding_csv(path, [dict(first[0], funding_rate=0.0003)]) == 2
    frame = load_funding_csv(path)
    assert len(frame) == 2
    assert str(frame.index.tz) == "UTC"
    assert frame["funding_rate"].iloc[0] == pytest.approx(0.0003)
    assert pd.isna(frame["mark_price"].iloc[1])
