from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from bot.config import load_settings
from bot.live.trend_entry_shadow import (
    TrendEntryShadowTracker,
    adverse_chase_atr,
    quote_execution_price,
    summarize_shadow_state,
)
from bot.live.multi_runner import _fetch_pullback_quote
from bot.signals import Side
from bot.strategy_router import TREND


NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def pending(side=Side.LONG):
    return {
        "source": TREND,
        "execution_type": "pullback",
        "symbol": "AAA/USDT:USDT",
        "side": side.value,
        "signal_time": NOW.isoformat(),
        "confirmed_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=45)).isoformat(),
        "reference_price": 99.6 if side is Side.LONG else 100.4,
        "signal_reference_price": 100.0,
        "pullback_price": 99.6 if side is Side.LONG else 100.4,
        "atr": 1.0,
        "stop_price": 98.2 if side is Side.LONG else 101.8,
        "structure_break_price": 97.5 if side is Side.LONG else 102.5,
        "regime": "TREND_UP" if side is Side.LONG else "TREND_DOWN",
    }


def bars_1h():
    index = pd.date_range(end=pd.Timestamp(NOW), periods=12, freq="1h", name="timestamp")
    return pd.DataFrame(
        [(100.0, 101.0, 99.0, 100.0, 100.0)] * len(index),
        columns=["open", "high", "low", "close", "volume"], index=index,
    )


@pytest.mark.parametrize(
    ("side", "last", "bid", "ask", "expected", "basis"),
    [
        (Side.LONG, 100.0, 99.9, 100.1, 100.1, "ask"),
        (Side.SHORT, 100.0, 99.9, 100.1, 99.9, "bid"),
        (Side.LONG, 100.0, 99.9, None, 100.0, "last_proxy"),
    ],
)
def test_quote_execution_price_uses_side_specific_quote(side, last, bid, ask, expected, basis):
    price, source = quote_execution_price(side, last=last, bid=bid, ask=ask)

    assert price == expected
    assert source == basis


@pytest.mark.parametrize(
    ("side", "market_price", "reference_price", "expected"),
    [
        (Side.LONG, 100.2, 100.0, 0.2),
        (Side.SHORT, 99.8, 100.0, 0.2),
        (Side.LONG, 99.8, 100.0, 0.0),
        (Side.SHORT, 100.2, 100.0, 0.0),
    ],
)
def test_adverse_chase_atr_is_directional(side, market_price, reference_price, expected):
    assert adverse_chase_atr(
        side, market_price=market_price, reference_price=reference_price, atr_value=1.0,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_hybrid_guards_use_signal_reference_and_keep_pullback_baseline(tmp_path, side):
    tracker = TrendEntryShadowTracker(tmp_path / "shadow.json", load_settings(env_file=None))
    candidate_pending = pending(side)
    assert tracker.register_pending(candidate_pending)

    if side is Side.LONG:
        last, bid, ask = 100.05, 100.0, 100.15
    else:
        last, bid, ask = 99.95, 99.85, 100.0
    changed = tracker.observe_symbol(
        "AAA/USDT:USDT", now=NOW + timedelta(seconds=5), current_price=last,
        quote_fresh=True,
        quote_metadata={"quote_price": last, "quote_bid": bid, "quote_ask": ask,
                        "quote_age_sec": 0.1},
        bars_1h=bars_1h(), market_risk_state="NORMAL",
    )

    candidate = tracker.state["candidates"]["AAA/USDT:USDT|" + NOW.isoformat()]
    scenarios = candidate["scenarios"]
    assert changed
    assert scenarios["pullback"]["status"] == "waiting_pullback"
    assert scenarios["hybrid_0.1atr"]["status"] == "waiting_pullback"
    assert scenarios["hybrid_0.2atr"]["status"] == "open"
    assert scenarios["hybrid_0.3atr"]["status"] == "open"
    assert scenarios["hybrid_0.2atr"]["entry_mode"] == "immediate_within_guard"
    assert scenarios["hybrid_0.2atr"]["initial_adverse_deviation_atr_from_signal"] == pytest.approx(0.15)

    tracker.save()
    summary = summarize_shadow_state(tracker.state)
    assert summary["candidate_count"] == 1
    assert summary["scenarios"]["hybrid_0.2atr"]["statuses"] == {"open": 1}


def test_pullback_touch_opens_all_scenarios_at_first_fresh_quote(tmp_path):
    tracker = TrendEntryShadowTracker(tmp_path / "shadow.json", load_settings(env_file=None))
    tracker.register_pending(pending())

    tracker.observe_symbol(
        "AAA/USDT:USDT", now=NOW + timedelta(seconds=5), current_price=99.5,
        quote_fresh=True,
        quote_metadata={"quote_price": 99.5, "quote_bid": 99.4, "quote_ask": 99.6},
        bars_1h=bars_1h(), market_risk_state="NORMAL",
    )

    scenarios = tracker.state["candidates"]["AAA/USDT:USDT|" + NOW.isoformat()]["scenarios"]
    assert all(scenario["status"] == "open" for scenario in scenarios.values())
    assert all(scenario["entry_mode"] == "pullback_touch" for scenario in scenarios.values())


def test_quote_fetch_preserves_bid_ask_for_shadow_without_marking_stale_quote_fresh():
    class Exchange:
        @staticmethod
        def fetch_ticker(symbol):
            assert symbol == "AAA/USDT:USDT"
            return {"last": 100.0, "bid": 99.9, "ask": 100.1}

    price, fresh, metadata = _fetch_pullback_quote(Exchange(), "AAA/USDT:USDT")

    assert price == 100.0
    assert fresh
    assert metadata["quote_bid"] == 99.9
    assert metadata["quote_ask"] == 100.1
