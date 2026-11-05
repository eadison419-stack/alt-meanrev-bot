from datetime import datetime, timezone

import pytest

from bot.config import load_settings
from bot.data.cache import KlineCache
from bot.data.klines import (fetch_closed_ohlcv, normalize_symbol, rank_hot_candidates,
                             resolve_symbols, score_hot_candidates, select_universe)


class FakeExchange:
    def __init__(self, markets=None, tickers=None, pages=None):
        self.markets = markets or {}
        self.tickers = tickers or {}
        self.pages = pages or []
        self.calls = []

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return self.tickers

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        for page in self.pages:
            if page and page[0][0] >= since:
                return page
        return []


def make_market(symbol, onboard, *, swap=True, active=True, quote="USDT", settle="USDT"):
    return {"symbol": symbol, "swap": swap, "active": active, "quote": quote,
            "settle": settle, "info": {"onboardDate": onboard}}


def test_auto_universe_filters_contract_status_age_volume_and_exclusions():
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    recent = int(datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp() * 1000)
    old = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
    markets = {
        "ADA/USDT:USDT": make_market("ADA/USDT:USDT", old),
        "BTC/USDT:USDT": make_market("BTC/USDT:USDT", old),
        "NEW/USDT:USDT": make_market("NEW/USDT:USDT", recent),
        "UNKNOWN/USDT:USDT": make_market("UNKNOWN/USDT:USDT", None),
        "SPOT/USDT": make_market("SPOT/USDT", old, swap=False),
        "INACTIVE/USDT:USDT": make_market("INACTIVE/USDT:USDT", old, active=False),
        "OTHER/USDC:USDC": make_market("OTHER/USDC:USDC", old, quote="USDC", settle="USDC"),
    }
    tickers = {symbol: {"quoteVolume": 25_000_000} for symbol in markets}
    tickers["LOW/USDT:USDT"] = {"quoteVolume": 1}
    exchange = FakeExchange(markets, tickers)
    settings = load_settings(env_file=None)
    assert select_universe(exchange, settings, now=now) == ["ADA/USDT:USDT"]
    assert resolve_symbols(exchange, "AUTO", settings, now=now) == ["ADA/USDT:USDT"]


def test_explicit_symbol_resolution_and_exclusions():
    exchange = FakeExchange({"ADA/USDT:USDT": make_market("ADA/USDT:USDT", 1)})
    settings = load_settings(env_file=None)
    assert resolve_symbols(exchange, "adausdt", settings) == ["ADA/USDT:USDT"]
    with pytest.raises(ValueError, match="excluded"):
        resolve_symbols(exchange, "BTCUSDT", settings)
    with pytest.raises(ValueError, match="unknown"):
        resolve_symbols(exchange, "MISSINGUSDT", settings)
    assert normalize_symbol("ADA/USDT:USDT") == "ADAUSDT"


def test_full_hot_scores_preserve_existing_top_n_order():
    candidates = [("AAA/USDT:USDT", 30_000_000), ("BBB/USDT:USDT", 25_000_000),
                  ("CCC/USDT:USDT", 20_000_000)]
    tickers = {
        "AAA/USDT:USDT": {"info": {"count": 100}},
        "BBB/USDT:USDT": {"info": {"count": 200}},
        "CCC/USDT:USDT": {"info": {"count": 300}},
    }
    activity = {
        symbol: {"volume_burst_ratio": float(index), "return_4h_pct": index,
                 "return_24h_pct": index * 2}
        for index, (symbol, _) in enumerate(candidates, start=1)
    }

    full = score_hot_candidates(candidates, tickers, activity)
    top_two = rank_hot_candidates(candidates, tickers, activity, limit=2)

    assert top_two == full[:2]


def test_fetch_paginates_and_discards_open_bar_and_out_of_range_rows():
    hour = 3_600_000
    start = 10 * hour
    end = start + 4 * hour + 5_000
    rows = [[timestamp, 1, 2, 0.5, 1.5, 10] for timestamp in (start, start + hour, start + 2 * hour, start + 3 * hour, start + 4 * hour)]
    exchange = FakeExchange(pages=[rows[:2], rows[2:]])
    result = fetch_closed_ohlcv(exchange, "ADA/USDT:USDT", "1h", since_ms=start, until_ms=end, limit=2)
    assert [row[0] for row in result] == [start, start + hour, start + 2 * hour, start + 3 * hour]
    assert len(exchange.calls) == 2
    assert exchange.calls[1][2] == start + 2 * hour


def test_fetch_stops_cleanly_on_empty_page():
    exchange = FakeExchange(pages=[])
    assert fetch_closed_ohlcv(exchange, "ADA/USDT:USDT", "4h", since_ms=0, until_ms=10_000) == []


def test_fetch_rejects_unordered_exchange_rows():
    exchange = FakeExchange(pages=[[[3_600_000, 1, 2, 0.5, 1.5, 10], [0, 1, 2, 0.5, 1.5, 10]]])
    with pytest.raises(ValueError, match="unordered"):
        fetch_closed_ohlcv(exchange, "ADA/USDT:USDT", "1h", since_ms=0, until_ms=20_000_000)


def test_sqlite_cache_upserts_and_filters_time_range(tmp_path):
    cache = KlineCache(tmp_path / "data" / "klines.sqlite")
    rows = [[0, 1, 2, 0.5, 1.5, 10], [3_600_000, 2, 3, 1, 2.5, 20]]
    assert cache.upsert("ADA/USDT:USDT", "1h", rows) == 2
    assert cache.upsert("ADA/USDT:USDT", "1h", rows) == 2
    frame = cache.load("ADA/USDT:USDT", "1h", since_ms=1)
    assert len(frame) == 1
    assert str(frame.index.tz) == "UTC"
    assert frame.iloc[0]["close"] == 2.5
    assert list(cache.load("MISSING", "1h").columns) == ["open", "high", "low", "close", "volume"]
