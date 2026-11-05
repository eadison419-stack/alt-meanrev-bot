from datetime import datetime, timezone

import pytest

from bot.config import load_settings
from bot.data.klines import (
    TIMEFRAME_MS,
    _percentile_scores,
    hot_activity_metrics,
    rank_hot_candidates,
    filter_tradable_crypto_universe,
)


def market(symbol, onboard, *, contract="PERPETUAL", underlying="COIN", subtypes=None,
           active=True, swap=True, quote="USDT", settle="USDT"):
    return {
        "active": active, "swap": swap, "quote": quote, "settle": settle,
        "info": {"contractType": contract, "underlyingType": underlying,
                 "underlyingSubType": subtypes or ["Crypto"], "onboardDate": onboard},
    }


def test_tradable_crypto_filter_returns_all_basic_eligible_without_hot_rank_cap():
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    old_listing = 1_700_000_000_000
    new_listing = int(datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp() * 1000)
    markets = {
        "AAA/USDT:USDT": market("AAA", old_listing),
        "BBB/USDT:USDT": market("BBB", old_listing),
        "XAU/USDT:USDT": market("XAU", old_listing, contract="TRADIFI_PERPETUAL", underlying="COMMODITY", subtypes=["TradFi"]),
        "STOCK/USDT:USDT": market("STOCK", old_listing, underlying="EQUITY", subtypes=["TradFi"]),
        "NEW/USDT:USDT": market("NEW", new_listing),
        "BTC/USDT:USDT": market("BTC", old_listing),
    }
    tickers = {symbol: {"quoteVolume": volume} for symbol, volume in {
        "AAA/USDT:USDT": 25_000_000, "BBB/USDT:USDT": 80_000_000,
        "XAU/USDT:USDT": 3_000_000_000, "STOCK/USDT:USDT": 2_000_000_000,
        "NEW/USDT:USDT": 900_000_000, "BTC/USDT:USDT": 5_000_000_000,
    }.items()}

    result = filter_tradable_crypto_universe(markets, tickers, load_settings(env_file=None), now=now)

    assert result == [("BBB/USDT:USDT", 80_000_000.0), ("AAA/USDT:USDT", 25_000_000.0)]


def test_hot_crypto_universe_enforces_hard_filters_even_if_config_changes():
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    old_listing = 1_700_000_000_000
    exactly_14_days = int(datetime(2026, 9, 15, tzinfo=timezone.utc).timestamp() * 1000)
    too_new = int(datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp() * 1000)
    markets = {
        "ADA/USDT:USDT": market("ADA", exactly_14_days),
        "BTC/USDT:USDT": market("BTC", old_listing),
        "ETH/USDT:USDT": market("ETH", old_listing),
        "LOW/USDT:USDT": market("LOW", old_listing),
        "NEW/USDT:USDT": market("NEW", too_new),
        "INACTIVE/USDT:USDT": market("INACTIVE", old_listing, active=False),
        "SPOT/USDT": market("SPOT", old_listing, swap=False),
        "USDC/USDC:USDC": market("USDC/USDC:USDC", old_listing, quote="USDC", settle="USDC"),
        "TRADFI/USDT:USDT": market("TRADFI", old_listing, underlying="EQUITY", subtypes=["TradFi"]),
    }
    tickers = {
        "ADA/USDT:USDT": {"quoteVolume": 20_000_000},
        "BTC/USDT:USDT": {"quoteVolume": 5_000_000_000},
        "ETH/USDT:USDT": {"quoteVolume": 5_000_000_000},
        "LOW/USDT:USDT": {"quoteVolume": 19_999_999},
        "NEW/USDT:USDT": {"quoteVolume": 30_000_000},
        "INACTIVE/USDT:USDT": {"quoteVolume": 30_000_000},
        "SPOT/USDT": {"quoteVolume": 30_000_000},
        "USDC/USDC:USDC": {"quoteVolume": 30_000_000},
        "TRADFI/USDT:USDT": {"quoteVolume": 30_000_000},
    }
    settings = load_settings(env_file=None)
    settings.universe.exclude_symbols = []

    result = filter_tradable_crypto_universe(markets, tickers, settings, now=now)

    assert result == [("ADA/USDT:USDT", 20_000_000.0)]


def test_hot_crypto_universe_discards_invalid_quote_volume():
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    old_listing = 1_700_000_000_000
    markets = {f"{symbol}/USDT:USDT": market(symbol, old_listing)
               for symbol in ("NAN", "INFINITE", "TEXT", "MISSING")}
    tickers = {
        "NAN/USDT:USDT": {"quoteVolume": float("nan")},
        "INFINITE/USDT:USDT": {"quoteVolume": float("inf")},
        "TEXT/USDT:USDT": {"quoteVolume": "not-a-number"},
    }

    result = filter_tradable_crypto_universe(markets, tickers, load_settings(env_file=None), now=now)

    assert result == []


def test_hot_activity_metrics_use_only_closed_bars_and_measure_relative_burst():
    hour = TIMEFRAME_MS["1h"]
    rows = []
    for index in range(26):
        close = 104.0 if index == 24 else 100.0
        volume = 300.0 if index == 24 else 100.0
        rows.append([index * hour, 100.0, max(100.0, close), min(100.0, close), close, volume])

    metrics = hot_activity_metrics(rows, now_ms=25 * hour + 1_000)

    assert round(metrics["volume_burst_ratio"], 2) == 3.06
    assert round(metrics["return_4h_pct"], 2) == 4.0
    assert round(metrics["return_24h_pct"], 2) == 4.0


def test_hot_candidates_prioritize_burst_and_symmetric_price_activity():
    candidates = [("LIQUID/USDT:USDT", 80_000_000.0), ("ACTIVE/USDT:USDT", 25_000_000.0)]
    tickers = {
        "LIQUID/USDT:USDT": {"info": {"count": "100000"}},
        "ACTIVE/USDT:USDT": {"info": {"count": "10000"}},
    }
    activity = {
        "LIQUID/USDT:USDT": {"volume_burst_ratio": 1.0, "return_4h_pct": 0.5, "return_24h_pct": 1.0},
        "ACTIVE/USDT:USDT": {"volume_burst_ratio": 3.0, "return_4h_pct": -4.0, "return_24h_pct": -12.0},
    }

    ranked = rank_hot_candidates(candidates, tickers, activity, limit=2)

    assert ranked[0]["symbol"] == "ACTIVE/USDT:USDT"
    assert ranked[0]["hot_score"] > ranked[1]["hot_score"]
    assert ranked[0]["trade_count_24h"] == 10_000


def test_hot_candidates_preserve_missing_metrics_and_have_stable_tie_break():
    candidates = [("ZZZ/USDT:USDT", 25_000_000.0), ("AAA/USDT:USDT", 25_000_000.0)]
    ranked = rank_hot_candidates(
        candidates,
        {"ZZZ/USDT:USDT": {"count": 10}, "AAA/USDT:USDT": {"count": 10}},
        {"ZZZ/USDT:USDT": {"volume_burst_ratio": None, "return_4h_pct": None,
                            "return_24h_pct": None}},
        limit=2,
    )

    assert [row["symbol"] for row in ranked] == ["AAA/USDT:USDT", "ZZZ/USDT:USDT"]
    assert ranked[1]["return_4h_pct"] is None
    assert ranked[1]["return_24h_pct"] is None
    assert ranked[1]["trade_count_24h"] == 10


def test_hot_candidate_rescore_is_capped_at_twenty():
    with pytest.raises(ValueError, match="between 1 and 20"):
        rank_hot_candidates([], {}, {}, limit=21)


def test_percentile_scores_keep_midrank_for_ties_and_zero_for_missing():
    scores = _percentile_scores({"low": 1.0, "tie_a": 2.0, "tie_b": 2.0,
                                 "high": 4.0, "missing": None})

    assert scores == {"low": 0.125, "tie_a": 0.5, "tie_b": 0.5,
                      "high": 0.875, "missing": 0.0}
