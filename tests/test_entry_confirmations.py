from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.signals import Side
from scripts import run_mr_candidate_paper as runner


def _frame(closes):
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes,
    }, index=pd.date_range("2026-11-01", periods=len(closes), freq="h", tz="UTC"))


def test_ema9_reclaim_detects_closed_bar_cross_in_both_directions():
    long_frame = _frame([100.0] * 10 + [99.0, 101.0])
    short_frame = _frame([100.0] * 10 + [101.0, 99.0])
    no_cross_frame = _frame([100.0] * 10 + [101.0, 102.0])

    assert runner._ema9_reclaim_side(long_frame) == Side.LONG.value
    assert runner._ema9_reclaim_side(short_frame) == Side.SHORT.value
    assert runner._ema9_reclaim_side(no_cross_frame) is None


def test_dynamic_limit_plan_uses_body_retrace_and_three_hour_expiry():
    candle = {"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0}
    closed_at = datetime(2026, 11, 1, 1, tzinfo=timezone.utc)

    long_plan = runner._entry_execution_plan(candle, 2.0, closed_at, Side.LONG)
    short_plan = runner._entry_execution_plan(candle, 2.0, closed_at, Side.SHORT)

    assert long_plan["candle_range_atr"] == 2.5
    assert long_plan["order_type"] == "limit"
    assert long_plan["limit_price"] == 103.0 - 0.441 * 3.0
    assert datetime.fromisoformat(long_plan["limit_expires_at"]) == closed_at + timedelta(hours=3)
    assert short_plan["limit_price"] == 103.0 + 0.441 * 3.0


def test_normal_range_keeps_market_execution():
    candle = {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}
    closed_at = datetime(2026, 11, 1, 1, tzinfo=timezone.utc)

    plan = runner._entry_execution_plan(candle, 2.0, closed_at, Side.LONG)

    assert plan["candle_range_atr"] == 1.5
    assert plan["order_type"] == "market"
    assert plan["limit_price"] is None and plan["limit_expires_at"] is None


def test_funding_flip_uses_latest_fresh_settled_pair():
    now = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
    stamp = int(now.timestamp() * 1000)

    class Exchange:
        def fetch_funding_rate_history(self, _symbol, limit=2):
            assert limit == 2
            return [
                {"timestamp": stamp - 8 * 60 * 60 * 1000, "fundingRate": 0.0001},
                {"timestamp": stamp - 60_000, "fundingRate": -0.0001},
            ]

    side, details = runner._funding_flip_side(Exchange(), "ADA/USDT:USDT", now)

    assert side == Side.LONG.value
    assert details["previous_rate"] > 0 > details["current_rate"]


def test_stale_funding_flip_is_not_accepted():
    now = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
    stamp = int((now - timedelta(hours=20)).timestamp() * 1000)

    class Exchange:
        def fetch_funding_rate_history(self, _symbol, limit=2):
            return [
                {"timestamp": stamp - 8 * 60 * 60 * 1000, "fundingRate": -0.0001},
                {"timestamp": stamp, "fundingRate": 0.0001},
            ]

    side, details = runner._funding_flip_side(Exchange(), "ADA/USDT:USDT", now)

    assert side is None
    assert details["reason"] == "funding_flip_history_stale"
