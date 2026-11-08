from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
import os

import pandas as pd
import pytest

from bot.config import load_settings
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.portfolio import Position
from bot.regime import Regime
from bot.signals import Side
from bot.strategy_router import MEAN_REVERSION
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner


UNAVAILABLE_PNL_VALUES = [
    pytest.param(None, id="missing"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive_infinity"),
    pytest.param(float("-inf"), id="negative_infinity"),
    pytest.param("NaN", id="nan_text"),
    pytest.param("Infinity", id="infinity_text"),
    pytest.param("unavailable", id="invalid_text"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(10 ** 400, id="overflowing_integer"),
]


class _FundingExchange:
    @staticmethod
    def fetch_funding_rate(_symbol):
        return {"fundingRate": 0.0}


def _daily_rows(closes, *, start=datetime(2026, 9, 5, tzinfo=timezone.utc)):
    rows = []
    for offset, close in enumerate(closes):
        opened = start + timedelta(days=offset)
        rows.append([int(opened.timestamp() * 1000), close, close, close, close, 1.0])
    return rows


def _market_risk(state: MarketRiskState, now: datetime) -> MarketRiskAssessment:
    return MarketRiskAssessment(state, state.value.lower(), now, "BTC/USDT:USDT", "4h", {})


def test_background_live_confirmation_requires_launcher_token(monkeypatch):
    args = SimpleNamespace(live=True, background_confirmed=True)
    monkeypatch.setenv("BOT_LEAD_BACKGROUND_CONFIRMATION", "LEAD")

    assert runner._confirm_live(args, runner.argparse.ArgumentParser()) is True
    assert "BOT_LEAD_BACKGROUND_CONFIRMATION" not in os.environ


def test_background_live_confirmation_rejects_missing_token(monkeypatch):
    args = SimpleNamespace(live=True, background_confirmed=True)
    monkeypatch.delenv("BOT_LEAD_BACKGROUND_CONFIRMATION", raising=False)

    with pytest.raises(SystemExit):
        runner._confirm_live(args, runner.argparse.ArgumentParser())


def test_background_confirmation_cannot_enable_nonlive_mode():
    args = SimpleNamespace(live=False, background_confirmed=True)

    with pytest.raises(SystemExit):
        runner._confirm_live(args, runner.argparse.ArgumentParser())


def test_exchange_target_margin_tracks_current_equity_and_local_paper_stays_fixed():
    settings = load_settings(env_file=None)
    lead = MultiPaperState(initial_equity=1_000, cash=1_000, execution_mode="lead_usdm",
                           exchange_snapshot={"equity": 1_000})
    demo = MultiPaperState(initial_equity=1_000, cash=1_000, execution_mode="demo_usdm",
                           exchange_snapshot={"equity": 1_000})
    paper = MultiPaperState(initial_equity=1_000, cash=1_000)

    assert runner._target_margin_per_symbol(lead, settings) is None
    lead.exchange_snapshot["equity"] = 1_250
    assert runner._target_margin_per_symbol(lead, settings) is None
    assert runner._target_margin_per_symbol(demo, settings) == pytest.approx(134.2)
    demo.exchange_snapshot["equity"] = 1_250
    assert runner._target_margin_per_symbol(demo, settings) == pytest.approx(167.75)
    assert runner._target_margin_per_symbol(paper, settings) == 1_000


@pytest.mark.parametrize(("mode", "expected"), [
    ("demo_usdm", 60), ("lead_usdm", 60), ("local_paper", 300),
])
def test_default_poll_matches_execution_mode(mode, expected):
    assert runner._default_poll_for_mode(mode) == expected


@pytest.mark.parametrize("mode", ["demo_usdm", "lead_usdm"])
def test_exchange_modes_are_labeled_with_the_daily_gate_strategy(mode):
    assert runner._strategy_variant_for_mode(mode) == "ema50_daily_gate_closed_h1_mean_reversion"


def test_local_paper_strategy_variant_tracks_entry_mode():
    assert runner._strategy_variant_for_mode("local_paper") == "ema50_deviation_candidate"
    assert runner._strategy_variant_for_mode("local_paper", intrabar_entries=True) == (
        "ema50_daily_gate_closed_h1_mean_reversion"
    )


def test_lead_candidate_filter_keeps_only_portfolio_whitelisted_usdt_symbols():
    execution = SimpleNamespace(lead_market_ids={"BTCUSDT"})
    hotlist = [{"symbol": "BTC/USDT:USDT"}, {"symbol": "ETH/USDT:USDT"}]
    markets = {
        "BTC/USDT:USDT": {"id": "BTCUSDT"},
        "ETH/USDT:USDT": {"id": "ETHUSDT"},
    }

    assert runner._filter_lead_candidates(hotlist, markets, execution) == [hotlist[0]]
    with pytest.raises(RuntimeError, match="whitelist"):
        runner._filter_lead_candidates(hotlist, markets, SimpleNamespace(lead_market_ids=set()))


def _cached_intrabar_fixture(now, symbol="BEAT/USDT:USDT"):
    state = MultiPaperState(initial_equity=5_000, cash=4_000)
    position = Position(symbol, Side.SHORT, 1.0, 100.0, 102.0,
                        now - timedelta(hours=1), "test_entry", Regime.LOW.value)
    state.portfolio.positions[symbol] = position
    state.mark_prices[symbol] = 100.0
    state.watchlist = [{"symbol": symbol, "rank": 1}]
    state.exit_management_by_symbol[symbol] = {
        "enabled": True, "protective_stop": 102.0,
    }
    one_hour, four_hour = _frames()
    state.exit_management_by_symbol[runner.BAR_KEY] = str(one_hour.index[-1])
    cache = runner.IntrabarMarketCache(
        markets={}, one_hour={symbol: one_hour}, four_hour={symbol: four_hour},
        selected_symbols={symbol}, bars_hour_key=int(now.timestamp() * 1000) // runner.HOUR_MS,
    )
    return state, cache, one_hour, four_hour, symbol


def test_ticker_price_requires_a_fresh_timestamp():
    now_ms = int(datetime(2026, 10, 26, 5, 15, tzinfo=timezone.utc).timestamp() * 1000)

    assert runner._ticker_price(
        {"BEAT/USDT:USDT": {"last": 100.0, "timestamp": now_ms}},
        "BEAT/USDT:USDT", now_ms,
    ) == 100.0
    assert runner._ticker_price(
        {"BEAT/USDT:USDT": {"last": 100.0}}, "BEAT/USDT:USDT", now_ms,
    ) is None
    assert runner._ticker_price(
        {"BEAT/USDT:USDT": {"last": 100.0, "timestamp": now_ms - runner.LIVE_TICKER_MAX_AGE_MS - 1}},
        "BEAT/USDT:USDT", now_ms,
    ) is None


def test_market_retry_delay_honors_binance_ban_expiry():
    banned_until_ms = 1_793_439_479_968
    now = datetime.fromtimestamp((banned_until_ms - 10_000_000) / 1000, timezone.utc)
    exc = runner.ccxt.DDoSProtection(f"418 banned until {banned_until_ms}")

    assert runner._market_retry_delay_seconds(exc, poll=15, attempt=1, now=now) == pytest.approx(10_003)


def _structure_close(symbol, side, at, *, reason=None, net_pnl=-10.0):
    return {
        "event": "paper_fill", "action": "CLOSE", "source": MEAN_REVERSION,
        "symbol": symbol, "side": side.value, "at": at.isoformat(),
        "reason": reason or f"{side.value}_structure_invalid", "net_pnl": net_pnl,
    }


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
@pytest.mark.parametrize("net_pnl, expected", [
    (-1.0, "stop_loss"), (0.0, "stop_loss"), (1.0, "strategy_exit"),
    ("-1.0", "stop_loss"), ("0", "stop_loss"), ("1.0", "strategy_exit"),
])
def test_structure_invalidation_losses_use_six_hour_cooldown_category(side, net_pnl, expected):
    assert runner._mr_exit_category(f"{side.value}_structure_invalid", net_pnl) == expected


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
@pytest.mark.parametrize("net_pnl", UNAVAILABLE_PNL_VALUES)
def test_unavailable_structure_pnl_restores_cooldown_and_failure_count(side, net_pnl):
    symbol = "JCT/USDT:USDT"
    at = datetime(2026, 11, 2, 6, tzinfo=timezone.utc)
    state = MultiPaperState()
    state.portfolio.last_exit_at[symbol] = at
    state.portfolio.last_exit_category[symbol] = "strategy_exit"
    state.events.extend([
        _structure_close(symbol, side, at - timedelta(hours=6)),
        _structure_close(symbol, side, at, net_pnl=net_pnl),
    ])
    assert runner._mr_exit_category(f"{side.value}_structure_invalid", net_pnl) == "stop_loss"

    # Legacy JSON may contain non-finite values; restore it without losing the gate.
    restored = MultiPaperState.from_dict(json.loads(json.dumps(state.to_dict())))
    runner._reconcile_structure_invalidation_cooldowns(restored)
    runner._reconcile_structure_invalidation_cooldowns(restored)

    assert restored.portfolio.last_exit_category[symbol] == "stop_loss"
    assert restored.portfolio.last_exit_at[symbol] == at
    assert runner._structure_reentry_state(restored.events, symbol, side) == (
        2, at.isoformat(), True,
    )
    updates = [e for e in restored.events if e.get("event") == "entry_cooldown_policy_updated"]
    assert len(updates) == 1
    assert updates[0]["net_pnl"] is None
    json.dumps(updates[0], allow_nan=False)


def test_two_consecutive_same_side_structure_losses_require_reentry_confirmation():
    symbol = "JCT/USDT:USDT"
    first = datetime(2026, 11, 1, 1, tzinfo=timezone.utc)
    second = first + timedelta(hours=6)
    events = [_structure_close(symbol, Side.LONG, first),
              _structure_close(symbol, Side.LONG, second)]

    assert runner._structure_reentry_state(events, symbol, Side.LONG) == (
        2, second.isoformat(), True,
    )
    assert runner._structure_reentry_state(events, symbol, Side.SHORT) == (0, None, False)

    confirmed = {
        "event": "structure_reentry_confirmed", "source": MEAN_REVERSION,
        "symbol": symbol, "side": Side.LONG.value,
        "after_structure_loss_at": second.isoformat(),
    }
    assert runner._structure_reentry_state([*events, confirmed], symbol, Side.LONG) == (
        2, second.isoformat(), False,
    )

    third = second + timedelta(hours=1)
    renewed_losses = [*events, confirmed, _structure_close(symbol, Side.LONG, third)]
    assert runner._structure_reentry_state(renewed_losses, symbol, Side.LONG) == (
        3, third.isoformat(), True,
    )


def test_other_exit_or_profitable_invalidation_breaks_consecutive_failure_run():
    symbol = "JCT/USDT:USDT"
    at = datetime(2026, 11, 1, tzinfo=timezone.utc)
    first = _structure_close(symbol, Side.LONG, at)
    profit = _structure_close(symbol, Side.LONG, at + timedelta(hours=1), net_pnl=2.0)
    other_exit = _structure_close(
        symbol, Side.LONG, at + timedelta(hours=2),
        reason="bollinger_mid_rejection", net_pnl=-2.0,
    )

    assert runner._structure_reentry_state([first, profit], symbol, Side.LONG) == (
        0, None, False,
    )
    assert runner._structure_reentry_state([first, first, other_exit], symbol, Side.LONG) == (
        0, None, False,
    )


def test_structure_confirmation_records_first_post_loss_closed_bar_once():
    symbol = "JCT/USDT:USDT"
    one_hour, _four_hour = _frames()
    one_hour.loc[one_hour.index[-2], "high"] = 100.5
    one_hour.loc[one_hour.index[-1], "close"] = 100.6
    bar_close = one_hour.index[-1] + timedelta(hours=1)
    loss_at = (bar_close - timedelta(hours=2)).to_pydatetime()
    state = MultiPaperState()
    state.events.extend([
        _structure_close(symbol, Side.LONG, loss_at - timedelta(hours=1)),
        _structure_close(symbol, Side.LONG, loss_at),
    ])
    now = bar_close.to_pydatetime() + timedelta(minutes=2)

    reentry_states = runner._structure_reentry_states(state.events, {symbol})
    runner._record_structure_reentry_confirmations(state, symbol, one_hour, now, reentry_states)
    runner._record_structure_reentry_confirmations(state, symbol, one_hour, now, reentry_states)

    confirmations = [e for e in state.events
                     if e.get("event") == "structure_reentry_confirmed"]
    assert len(confirmations) == 1
    assert confirmations[0]["bar_close_at"] == bar_close.isoformat()
    assert confirmations[0]["after_structure_loss_at"] == loss_at.isoformat()
    assert runner._structure_reentry_state(state.events, symbol, Side.LONG)[2] is False
    restored = MultiPaperState.from_dict(state.to_dict())
    assert runner._structure_reentry_state(restored.events, symbol, Side.LONG)[2] is False


def test_existing_structure_loss_cooldown_is_reclassified_for_demo_state():
    symbol = "JCT/USDT:USDT"
    at = datetime(2026, 11, 2, 6, tzinfo=timezone.utc)
    state = MultiPaperState()
    state.portfolio.last_exit_at[symbol] = at
    state.portfolio.last_exit_category[symbol] = "strategy_exit"
    state.events.append(_structure_close(symbol, Side.LONG, at))

    runner._reconcile_structure_invalidation_cooldowns(state)
    runner._reconcile_structure_invalidation_cooldowns(state)

    assert state.portfolio.last_exit_category[symbol] == "stop_loss"
    assert len([e for e in state.events
                if e.get("event") == "entry_cooldown_policy_updated"]) == 1


def test_structure_confirmation_does_not_end_six_hour_cooldown():
    settings = load_settings(env_file=None)
    symbol = "JCT/USDT:USDT"
    one_hour, _four_hour = _frames()
    one_hour.loc[one_hour.index[-2], "high"] = 100.5
    one_hour.loc[one_hour.index[-1], "close"] = 100.6
    now = (one_hour.index[-1] + timedelta(hours=1)).to_pydatetime()
    loss_at = now - timedelta(hours=2)
    state = MultiPaperState()
    state.portfolio.last_exit_at[symbol] = loss_at
    state.portfolio.last_exit_category[symbol] = "strategy_exit"
    state.events.extend([
        _structure_close(symbol, Side.LONG, loss_at - timedelta(hours=6), net_pnl=None),
        _structure_close(symbol, Side.LONG, loss_at, net_pnl=None),
    ])
    runner._reconcile_structure_invalidation_cooldowns(state)
    reentry_states = runner._structure_reentry_states(state.events, {symbol})
    runner._record_structure_reentry_confirmations(state, symbol, one_hour, now, reentry_states)
    restored = MultiPaperState.from_dict(json.loads(json.dumps(state.to_dict(), allow_nan=False)))
    assert runner._structure_reentry_state(restored.events, symbol, Side.LONG) == (
        2, loss_at.isoformat(), False,
    )
    intent = runner.OrderIntent(symbol, Side.LONG, runner.Action.OPEN, 1, 100,
                                "test_entry", Regime.LOW, 95, MEAN_REVERSION)
    just_before_end = loss_at + timedelta(hours=6) - timedelta(seconds=1)
    assert runner.assess_entry(intent, restored.portfolio, 5_000, 0, just_before_end,
                               settings).reason == "cooldown_active"
    assert runner.assess_entry(intent, restored.portfolio, 5_000, 0,
                               loss_at + timedelta(hours=6), settings).allowed


def test_structure_confirmation_gate_blocks_candidate_before_funding(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_status = "synchronized"
    symbol = "JCT/USDT:USDT"
    one_hour, four_hour = _frames()
    now = datetime(2026, 10, 26, 5, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(runner, "_utc_now", lambda: now)
    loss_at = now - timedelta(hours=2)
    state.portfolio.last_exit_at[symbol] = loss_at
    state.portfolio.last_exit_category[symbol] = "strategy_exit"
    state.events.extend([
        _structure_close(symbol, Side.LONG, loss_at - timedelta(hours=6)),
        _structure_close(symbol, Side.LONG, loss_at),
    ])
    flags = TimeFilterFlags(True, True, False, False)

    class NoFundingExchange:
        @staticmethod
        def fetch_ohlcv(_symbol, timeframe, *, limit):
            assert timeframe == "1d"
            return _daily_rows(list(range(100, 151)) + [1.0])

        @staticmethod
        def fetch_funding_rate(_symbol):
            raise AssertionError("confirmation-blocked entry must not reach funding")

    monkeypatch.setattr(runner, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *args, **kwargs: flags)
    monkeypatch.setattr(
        runner, "deviation_entry_signal",
        lambda *_args, **_kwargs: SimpleNamespace(
            action=runner.Action.OPEN, side=Side.LONG, reason="range_deviation_long",
        ),
    )
    monkeypatch.setattr(
        runner, "calculate_adaptive_stop",
        lambda side, price, *_args, **_kwargs: (price - 0.2, 1.0),
    )

    runner._evaluate_entries(
        NoFundingExchange(), state, None, True, {}, [symbol], {symbol: one_hour},
        {symbol: four_hour}, {}, settings, 5, 0.0005, 0.0003, now,
        _market_risk(MarketRiskState.NORMAL, now),
    )

    assert not state.portfolio.positions
    assert state.portfolio.last_exit_category[symbol] == "stop_loss"
    assert state.symbol_status[symbol]["reason"] == "structure_reentry_confirmation_required"
    assert any(e.get("event") == "entry_signal_blocked"
               and e.get("reason") == "structure_reentry_confirmation_required"
               for e in state.events)


def test_intrabar_cache_only_polls_open_position_tickers_between_bar_refreshes(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 10, 26, 5, 15, tzinfo=timezone.utc)
    state, cache, one_hour, four_hour, symbol = _cached_intrabar_fixture(now)
    calls = {"tickers": 0, "ohlcv": 0, "position_tickers": 0}

    class Exchange:
        def __init__(self):
            self.now_ms = int(now.timestamp() * 1000)

        def fetch_tickers(self):
            calls["tickers"] += 1
            raise AssertionError("the hotlist cache is still fresh")

        def fetch_ohlcv(self, *_args, **_kwargs):
            calls["ohlcv"] += 1
            raise AssertionError("closed bars should be reused between hour refreshes")

        def fetch_ticker(self, requested_symbol):
            assert requested_symbol == symbol
            calls["position_tickers"] += 1
            return {"last": 100.0, "timestamp": self.now_ms}

    exchange = Exchange()
    monkeypatch.setattr(runner, "_update_market_risk",
                        lambda *_args, **_kwargs: _market_risk(MarketRiskState.NORMAL, now))
    monkeypatch.setattr(runner, "_hotlist_cache_is_fresh", lambda *_args: True)
    monkeypatch.setattr(runner, "_evaluate_exits", lambda *_args: None)
    monkeypatch.setattr(runner, "_evaluate_entries", lambda *_args, **_kwargs: None)

    runner._cycle(exchange, None, state, settings, now, False, 5,
                  intrabar_entries=True, market_cache=cache)
    second_now = now + timedelta(seconds=15)
    exchange.now_ms = int(second_now.timestamp() * 1000)
    runner._cycle(exchange, None, state, settings, second_now, False, 5,
                  intrabar_entries=True, market_cache=cache)

    assert calls == {"tickers": 0, "ohlcv": 0, "position_tickers": 2}
    assert state.exit_management_by_symbol[symbol]["live_stop_monitor"]["status"] == "active"


def test_stale_position_quote_pauses_entries_and_marks_stop_monitor_unavailable(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 10, 26, 5, 15, tzinfo=timezone.utc)
    state, cache, _one_hour, _four_hour, symbol = _cached_intrabar_fixture(now)
    state.exit_management_by_symbol[runner.BAR_KEY] = "not-yet-processed"
    now_ms = int(now.timestamp() * 1000)

    class Exchange:
        @staticmethod
        def fetch_ticker(_symbol):
            return {"last": 100.0, "timestamp": now_ms - runner.LIVE_TICKER_MAX_AGE_MS - 1}

    monkeypatch.setattr(runner, "_update_market_risk",
                        lambda *_args, **_kwargs: _market_risk(MarketRiskState.NORMAL, now))
    monkeypatch.setattr(runner, "_hotlist_cache_is_fresh", lambda *_args: True)
    monkeypatch.setattr(runner, "_evaluate_exits", lambda *_args: None)
    observed_risk = []
    monkeypatch.setattr(runner, "_evaluate_entries",
                        lambda *args, **_kwargs: observed_risk.append(args[-1].state))

    runner._cycle(Exchange(), None, state, settings, now, False, 5,
                  intrabar_entries=True, market_cache=cache)

    assert symbol in state.portfolio.positions
    assert state.market_risk_status["state"] == "PAUSE_NEW"
    assert state.market_risk_status["reason"] == "live_price_unavailable"
    assert state.exit_management_by_symbol[symbol]["live_stop_monitor"]["status"] == "unavailable"
    assert observed_risk == [MarketRiskState.PAUSE_NEW]


def _frames():
    index_1h = pd.date_range(end="2026-10-26T04:00:00Z", periods=80, freq="1h")
    index_4h = pd.date_range(end="2026-10-26T00:00:00Z", periods=80, freq="4h")
    close_1h = pd.Series(100.0, index=index_1h)
    close_4h = pd.Series(100.0, index=index_4h)
    one_hour = pd.DataFrame({"open": close_1h, "high": close_1h + 1,
                             "low": close_1h - 1, "close": close_1h,
                             "volume": 1.0}, index=index_1h)
    four_hour = pd.DataFrame({"open": close_4h, "high": close_4h + 1,
                              "low": close_4h - 1, "close": close_4h,
                              "volume": 1.0}, index=index_4h)
    return one_hour, four_hour


def test_entry_bar_close_validation_rejects_open_and_stale_candles():
    one_hour, _four_hour = _frames()

    close_at, reason = runner._entry_bar_close_time(
        one_hour, datetime(2026, 10, 26, 4, 59, tzinfo=timezone.utc),
    )
    assert close_at is None
    assert reason == "hourly_bar_not_closed"

    close_at, reason = runner._entry_bar_close_time(
        one_hour, datetime(2026, 10, 26, 5, 10, 1, tzinfo=timezone.utc),
    )
    assert close_at is None
    assert reason == "hourly_bar_stale"

    close_at, reason = runner._entry_bar_close_time(
        one_hour, datetime(2026, 10, 26, 5, 10, tzinfo=timezone.utc),
    )
    assert close_at == datetime(2026, 10, 26, 5, tzinfo=timezone.utc)
    assert reason is None


def test_daily_direction_gate_uses_only_completed_utc_candles():
    class Exchange:
        def __init__(self):
            self.limit = None

        def fetch_ohlcv(self, _symbol, timeframe, *, limit):
            assert timeframe == "1d"
            self.limit = limit
            return _daily_rows(list(range(100, 151)) + [1.0])

    exchange = Exchange()
    gate = runner._daily_direction_gate(
        exchange, "APR/USDT:USDT", datetime(2026, 10, 26, 1, tzinfo=timezone.utc), 50,
    )

    assert exchange.limit == 100
    assert gate["ready"] is True
    assert gate["direction"] == "long"
    assert gate["daily_bars"] == 51
    assert gate["daily_close"] == 150.0
    assert gate["daily_bar_close_time"] == "2026-10-26T00:00:00+00:00"
    assert runner._daily_direction_rejection_reason(gate, Side.LONG) is None

    class FallingExchange:
        @staticmethod
        def fetch_ohlcv(*_args, **_kwargs):
            return _daily_rows(list(range(151, 100, -1)))

    falling = runner._daily_direction_gate(
        FallingExchange(),
        "APR/USDT:USDT", datetime(2026, 10, 26, 1, tzinfo=timezone.utc), 50,
    )
    assert falling["direction"] == "short"
    assert runner._daily_direction_rejection_reason(falling, Side.SHORT) is None


def test_daily_direction_gate_blocks_insufficient_history_and_equal_ema():
    class Exchange:
        def __init__(self, closes):
            self.rows = _daily_rows(closes)

        def fetch_ohlcv(self, *_args, **_kwargs):
            return self.rows

    signal_at = datetime(2026, 10, 26, 1, tzinfo=timezone.utc)
    short_history = runner._daily_direction_gate(
        Exchange(list(range(100, 150))), "APR/USDT:USDT", signal_at, 50,
    )
    equal = runner._daily_direction_gate(
        Exchange([100.0] * 51), "APR/USDT:USDT", signal_at, 50,
    )

    assert short_history["reason"] == "daily_history_insufficient"
    assert short_history["daily_bars"] == 50
    assert equal["reason"] == "daily_close_equals_ema50"
    assert equal["daily_close"] == equal["daily_ema50"] == 100.0


def test_daily_direction_gate_fails_closed_when_market_data_is_unavailable():
    class Exchange:
        @staticmethod
        def fetch_ohlcv(*_args, **_kwargs):
            raise runner.ccxt.NetworkError("offline")

    gate = runner._daily_direction_gate(
        Exchange(), "APR/USDT:USDT", datetime(2026, 10, 26, 1, tzinfo=timezone.utc), 50,
    )

    assert gate["ready"] is False
    assert gate["reason"] == "daily_data_unavailable"
    assert gate["error_type"] == "NetworkError"


def test_demo_entry_daily_gate_blocks_counter_direction_before_funding(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_status = "synchronized"
    symbol = "APR/USDT:USDT"
    now = datetime(2026, 10, 26, 5, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(runner, "_utc_now", lambda: now)
    one_hour, four_hour = _frames()
    flags = TimeFilterFlags(True, True, False, False)

    class Exchange:
        def fetch_ohlcv(self, requested_symbol, timeframe, *, limit):
            assert requested_symbol == symbol
            assert timeframe == "1d"
            return _daily_rows(list(range(100, 151)) + [1.0])

        @staticmethod
        def fetch_funding_rate(_symbol):
            raise AssertionError("daily-rejected entry must not reach funding check")

    monkeypatch.setattr(runner, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *args, **kwargs: flags)
    monkeypatch.setattr(
        runner, "deviation_entry_signal",
        lambda *_args, **_kwargs: SimpleNamespace(
            action=runner.Action.OPEN, side=Side.SHORT, reason="range_deviation_short",
        ),
    )
    monkeypatch.setattr(
        runner, "calculate_adaptive_stop",
        lambda side, price, *_args, **_kwargs: (price + 0.2, 1.0),
    )

    runner._evaluate_entries(
        Exchange(), state, None, True, {}, [symbol], {symbol: one_hour}, {symbol: four_hour},
        {}, settings, 5, 0.0005, 0.0003, now,
        _market_risk(MarketRiskState.NORMAL, now),
    )

    assert not state.portfolio.positions
    assert state.symbol_status[symbol]["reason"] == "daily_direction_long_only"
    assert state.symbol_status[symbol]["daily_direction"] == "long"
    assert state.symbol_status[symbol]["daily_bars"] == 51
    assert any(event.get("event") == "entry_signal_blocked"
               and event.get("reason") == "daily_direction_long_only" for event in state.events)


def test_morning_dump_time_filter_blocks_rez_like_short_before_funding_or_order():
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    symbol = "REZ/USDT:USDT"
    now = datetime(2026, 10, 26, 2, 1, tzinfo=timezone.utc)

    index_1h = pd.date_range(end="2026-10-26T01:00:00Z", periods=80, freq="1h")
    one_hour = pd.DataFrame({
        "open": 0.004775,
        "high": 0.004800,
        "low": 0.004750,
        "close": 0.004775,
        "volume": 1.0,
    }, index=index_1h)
    one_hour.loc[index_1h[-1], ["open", "high", "low", "close"]] = [
        0.004775, 0.004800, 0.004490, 0.004560,
    ]
    index_4h = pd.date_range(end="2026-10-26T00:00:00Z", periods=80, freq="4h")
    four_hour = pd.DataFrame({
        "open": 0.004400,
        "high": 0.004450,
        "low": 0.004350,
        "close": 0.004400,
        "volume": 1.0,
    }, index=index_4h)

    class NoFundingExchange:
        @staticmethod
        def fetch_funding_rate(_symbol):
            raise AssertionError("time-filter-rejected entry must not reach funding")

    runner._evaluate_entries(
        NoFundingExchange(), state, None, False, {}, [symbol], {symbol: one_hour},
        {symbol: four_hour}, {}, settings, 5, 0.0005, 0.0003, now,
        _market_risk(MarketRiskState.NORMAL, now),
    )

    assert not state.portfolio.positions
    assert not [event for event in state.events if event.get("event") == "paper_fill"
                and event.get("action") == "OPEN"]
    decision = next(event for event in state.events
                    if event.get("event") == "entry_time_filter_decision")
    assert decision["decision"] == "BLOCK"
    assert decision["reason"] == "morning_down_blocks_short"
    assert decision["side"] == "short"
    assert decision["bar_close_at"] == "2026-10-26T02:00:00+00:00"
    assert decision["bar_close_local"] == "2026-10-26T10:00:00+08:00"
    assert decision["recent_move_atr"] >= settings.signals.impulse_atr_mult
    assert decision["time_filter_build_id"] == runner.TIME_FILTER_BUILD_ID


def test_8767_uses_closed_h1_entry_with_daily_gate_and_keeps_live_stop(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    symbol = "APR/USDT:USDT"
    now = datetime(2026, 10, 26, 5, 5, tzinfo=timezone.utc)
    one_hour, four_hour = _frames()
    one_hour.loc[one_hour.index[-1], ["open", "high", "low", "close"]] = [98.2, 98.8, 98.0, 98.6]
    flags = TimeFilterFlags(True, True, False, False)
    gate_calls = []
    monkeypatch.setattr(runner, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *args, **kwargs: flags)
    monkeypatch.setattr(
        runner, "_daily_direction_gate",
        lambda _exchange, _symbol, signal_at, span, **_kwargs: (
            gate_calls.append((signal_at, span))
            or {"ready": True, "direction": "long", "reason": None,
                "daily_bars": 51, "daily_close": 101.0, "daily_ema50": 100.0,
                "daily_bar_close_time": "2026-10-26T00:00:00+00:00",
                "signal_at": signal_at.isoformat()}
        ),
    )
    monkeypatch.setattr(
        runner,
        "calculate_adaptive_stop",
        lambda side, price, *_args, **_kwargs: (price - 0.2 if side is Side.LONG else price + 0.2, 1.0),
    )
    monkeypatch.setattr(runner, "calculate_order_quantity", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(runner, "assess_entry", lambda *_args, **_kwargs: SimpleNamespace(allowed=True))

    runner._evaluate_entries(
        _FundingExchange(), state, None, False, {}, [symbol], {symbol: one_hour},
        {symbol: four_hour}, {}, settings, 5, 0.0, 0.0, now,
        _market_risk(MarketRiskState.NORMAL, now), daily_direction_gate=True,
    )

    position = state.portfolio.positions[symbol]
    assert position.side is Side.LONG
    assert position.source == MEAN_REVERSION
    assert position.entry_price == pytest.approx(98.6)
    assert gate_calls == [(datetime(2026, 10, 26, 5, tzinfo=timezone.utc), 50)]
    assert state.symbol_status[symbol]["daily_direction"] == "long"
    assert len([event for event in state.events if event.get("event") == "paper_fill"
                and event.get("action") == "OPEN"]) == 1

    runner._evaluate_live_stop_exits(state, None, False, {symbol: position.stop_price - 0.1},
                                     0.0, 0.0, now + timedelta(seconds=5))
    assert symbol not in state.portfolio.positions
    assert state.events[-1]["reason"] == "protective_stop"


def test_8767_closed_h1_entry_respects_btc_pause_new(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    symbol = "APR/USDT:USDT"
    now = datetime(2026, 10, 26, 5, 15, tzinfo=timezone.utc)
    one_hour, four_hour = _frames()
    state.symbol_status[symbol] = {"regime": "LOW"}

    runner._evaluate_entries(
        _FundingExchange(), state, None, False, {}, [symbol], {symbol: one_hour},
        {symbol: four_hour}, {}, settings, 5, 0.0005, 0.0003, now,
        _market_risk(MarketRiskState.PAUSE_NEW, now), daily_direction_gate=True,
    )

    assert not state.portfolio.positions


def test_8767_cycle_dispatches_closed_h1_entries_once_with_daily_gate(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    symbol = "APR/USDT:USDT"
    state.watchlist = [{"symbol": symbol, "rank": 1}]
    one_hour, four_hour = _frames()
    now = datetime(2026, 10, 26, 5, 15, tzinfo=timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    dispatched = []

    class Exchange:
        @staticmethod
        def load_markets():
            return {}

        @staticmethod
        def fetch_tickers():
            return {symbol: {"last": 98.6, "timestamp": now_ms}}

    monkeypatch.setattr(runner, "_update_market_risk",
                        lambda *_args, **_kwargs: _market_risk(MarketRiskState.NORMAL, now))
    monkeypatch.setattr(runner, "_hotlist_cache_is_fresh", lambda *_args: True)
    monkeypatch.setattr(
        runner, "_fetch_selected_bars",
        lambda *_args, **_kwargs: (
            {symbol: one_hour},
            {symbol: {"open": 98.2, "hour_key": "2026-10-26T05:00:00+00:00"}},
            {symbol: four_hour},
            {},
        ),
    )
    monkeypatch.setattr(runner, "_evaluate_exits", lambda *_args: None)
    monkeypatch.setattr(
        runner, "_evaluate_entries",
        lambda *args, **kwargs: dispatched.append(kwargs.get("daily_direction_gate")),
    )

    runner._cycle(Exchange(), None, state, settings, now, False, 5, intrabar_entries=True)
    runner._cycle(Exchange(), None, state, settings, now + timedelta(seconds=15),
                  False, 5, intrabar_entries=True)

    assert dispatched == [True]


def _short_exit_fixture(*, is_demo=False):
    settings = load_settings(env_file=None)
    symbol = "APR/USDT:USDT"
    opened_at = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    now = opened_at + timedelta(hours=2)
    position = Position(symbol, Side.SHORT, 1.0, 100.0, 102.0, opened_at,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_mode = "demo_usdm" if is_demo else "local_paper"
    state.execution_status = "synchronized" if is_demo else "local"
    state.portfolio.positions[symbol] = position
    state.exit_management_by_symbol[symbol] = {
        "entry_atr": 1.0, "initial_risk": 2.0,
        "structure_low": 90.0, "structure_high": 110.0,
        "best_price": 100.0, "worst_price": 101.0,
        "last_close": 100.0, "no_progress_count": 0,
        "mid_reached": False, "mid_hold": 0, "mid_fired": False,
    }
    one_hour, _ = _frames()
    index = one_hour.index[-1]
    one_hour.loc[index, ["open", "high", "low", "close"]] = [100.0, 100.0, 97.0, 98.0]
    if is_demo:
        state.protective_stop_audit = {
            "at": now.isoformat(), "status": "verified", "position_count": 1,
            "mismatch_count": 0, "mismatches": [],
        }
        state.exchange_orders_by_symbol[symbol] = {
            "stop": {"id": "old-stop", "stop_price": 102.0, "amount": 1.0},
        }
    return settings, state, symbol, one_hour, now


def test_local_candidate_runner_ratchets_stop_without_mutating_frozen_position():
    settings, state, symbol, one_hour, now = _short_exit_fixture()

    runner._evaluate_exits(state, None, False, {symbol: one_hour}, {}, settings,
                           0.0, 0.0, now)

    position = state.portfolio.positions[symbol]
    active_stop = state.exit_management_by_symbol[symbol]["protective_stop"]
    assert position.stop_price == 102.0  # immutable initial risk remains available
    assert active_stop == 99.0  # 1.5R excursion locks 0.5R with the current MR rule.
    runner._evaluate_live_stop_exits(state, None, False, {symbol: 100.1},
                                     0.0, 0.0, now + timedelta(minutes=1))
    assert symbol not in state.portfolio.positions
    assert state.events[-1]["reason"] == "protective_stop"


def test_demo_candidate_runner_uses_exchange_stop_and_does_not_repeat_replacement():
    settings, state, symbol, _one_hour, _fixture_now = _short_exit_fixture(is_demo=True)
    now = datetime.now(timezone.utc)
    state.exchange_snapshot = {"at": now.isoformat()}
    state.portfolio.positions[symbol] = replace(
        state.portfolio.positions[symbol], opened_at=now - timedelta(hours=2),
    )

    class Exchange:
        markets = {symbol: {}}

        @staticmethod
        def fetch_ticker(_symbol):
            return {
                "last": 99.0,
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }

    class Execution:
        def __init__(self):
            self.replacements = []
            self.exchange = SimpleNamespace(price_to_precision=lambda _symbol, price: str(price))

        @staticmethod
        def audit_active_protective_stops(_positions, _orders):
            return []

        def replace_stop(self, position, *, stop, stop_price):
            self.replacements.append((stop, stop_price))
            return {"id": "new-stop", "stop_price": float(stop_price), "amount": 1.0}

    execution = Execution()
    runner._manage_demo_positions(Exchange(), execution, state, settings, now, 0.0, 0.0)
    runner._manage_demo_positions(
        Exchange(), execution, state, settings, now + timedelta(seconds=1), 0.0, 0.0,
    )

    assert len(execution.replacements) == 1
    assert execution.replacements[0][0]["id"] == "old-stop"
    assert state.portfolio.positions[symbol].stop_price == 102.0
    assert state.exchange_orders_by_symbol[symbol]["stop"]["id"] == "new-stop"
    assert state.exchange_orders_by_symbol[symbol]["stop"]["stop_price"] == 99.6
    assert state.exit_management_by_symbol[symbol]["protective_stop"] == 99.6
    assert state.protective_stop_audit["status"] == "verified"
    updates = [event for event in state.events if event.get("event") == "protective_stop_updated"]
    assert len(updates) == 1
    assert updates[0]["previous_stop"] == 102.0
    assert updates[0]["stop_price"] == 99.6  # 1 ATR locks 0.2R; the exit rule is unchanged.


def test_demo_position_sync_preserves_stop_high_watermark_and_flags_regression():
    settings, state, symbol, _one_hour, now = _short_exit_fixture(is_demo=True)
    state.exit_management_by_symbol[symbol]["protective_stop"] = 100.0

    class Execution:
        @staticmethod
        def sync_snapshot():
            return {"positions": {symbol: {"side": "short"}}}

    assert runner._demo_sync_stopped(Execution(), state, now)

    meta = state.exit_management_by_symbol[symbol]
    assert meta["protective_stop"] == 100.0
    assert meta["stop_regression"] == {"previous_stop": 100.0, "observed_stop": 102.0}
    assert state.execution_status == "inconsistent"
    assert state.protective_stop_audit["status"] == "mismatch"
    assert "less protective" in state.protective_stop_audit["mismatches"][0]
    assert state.events[-1]["event"] == "protective_stop_regression_detected"

    class AuditExecution:
        @staticmethod
        def audit_active_protective_stops(_positions, _orders):
            return []

    assert not runner._audit_demo_protective_stops(AuditExecution(), state, now)
    assert state.protective_stop_audit["status"] == "mismatch"


def test_demo_protective_stop_audit_refreshes_and_blocks_on_mismatch():
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_mode = "demo_usdm"
    now = datetime(2026, 10, 26, 5, tzinfo=timezone.utc)

    class Execution:
        result = []

        def audit_active_protective_stops(self, _positions, _orders):
            return self.result

    execution = Execution()
    assert runner._demo_stop_audit_due(state, now)
    assert runner._audit_demo_protective_stops(execution, state, now)
    assert state.protective_stop_audit["status"] == "verified"
    assert not runner._demo_stop_audit_due(state, now + timedelta(seconds=299))

    execution.result = ["APR/USDT:USDT: active protective stop id missing"]
    assert not runner._audit_demo_protective_stops(execution, state, now + timedelta(minutes=5))
    assert state.execution_status == "inconsistent"
    assert state.protective_stop_audit["status"] == "mismatch"
    assert runner._demo_stop_audit_due(state, now + timedelta(minutes=5, seconds=1))


def test_demo_position_sync_uses_tracked_exchange_stop_for_dashboard():
    settings, state, symbol, _one_hour, now = _short_exit_fixture(is_demo=True)

    class Execution:
        @staticmethod
        def sync_snapshot():
            return {"positions": {symbol: {"side": "short"}}}

    assert runner._demo_sync_stopped(Execution(), state, now)
    assert state.exit_management_by_symbol[symbol]["protective_stop"] == 102.0
    assert state.exit_management_by_symbol[symbol]["enabled"] is True


def test_manual_position_adoption_is_explicit_and_lead_only(monkeypatch):
    symbol = "KOMA/USDT:USDT"
    now = datetime(2026, 11, 8, 3, tzinfo=timezone.utc)
    live_position = {"side": "short", "quantity": 10_000.0, "entry_price": 0.02003}
    adopted = []

    class Execution:
        adopt_manual_positions = False

        def sync_snapshot(self):
            return {"positions": {symbol: live_position}}

    def install(_execution, state, requested_symbol, _exchange_position, at):
        assert requested_symbol == symbol and at == now
        position = Position(symbol, Side.SHORT, 10_000.0, 0.02003, 0.021,
                            now - timedelta(hours=1), "manual_position_adopted", Regime.MID.value)
        state.portfolio.positions[symbol] = position
        state.exchange_orders_by_symbol[symbol] = {
            "stop": {"id": "algo-1", "stop_price": 0.021},
        }
        runner._invalidate_demo_stop_audit(state)
        adopted.append(requested_symbol)

    execution = Execution()
    state = MultiPaperState(initial_equity=1_000, cash=1_000)
    state.execution_mode = "lead_usdm"
    assert not runner._demo_sync_stopped(execution, state, now)
    assert adopted == []
    assert state.execution_status == "inconsistent"

    monkeypatch.setattr(runner, "_adopt_lead_manual_position", install)
    execution.adopt_manual_positions = True
    assert runner._demo_sync_stopped(execution, state, now)
    assert adopted == [symbol]
    assert state.execution_status == "synchronized"
    assert state.portfolio.positions[symbol].reason == "manual_position_adopted"

    demo = MultiPaperState(initial_equity=1_000, cash=1_000)
    demo.execution_mode = "demo_usdm"
    assert not runner._demo_sync_stopped(execution, demo, now)
    assert adopted == [symbol]


def test_adopt_manual_positions_flag_requires_live_mode(monkeypatch):
    monkeypatch.setattr(runner.sys, "argv", ["run_mr_candidate_paper.py", "--adopt-manual-positions"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_demo_strategy_close_records_exchange_fill_costs_and_pnl():
    symbol = "APR/USDT:USDT"
    opened_at = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    closed_at = opened_at + timedelta(hours=2)
    position = Position(symbol, Side.LONG, 2.0, 100.0, 95.0, opened_at,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_mode = "demo_usdm"
    state.portfolio.positions[symbol] = position
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "algo-stop"}}

    class Execution:
        @staticmethod
        def resolve_entry_fee(requested_symbol, side, quantity, entry_price, entry_time):
            assert (requested_symbol, side, quantity, entry_price, entry_time) == (
                symbol, Side.LONG, 2.0, 100.0, opened_at,
            )
            return 0.2

        def close(self, requested_position, *, stop, reason):
            assert requested_position is position
            assert stop["id"] == "algo-stop"
            assert reason == "test_exit"
            return {"fill": {"id": "close-1", "filled": 2.0, "average": 105.0,
                              "fee_cost": 0.21, "realized_pnl": 10.0,
                              "closed_at": closed_at.isoformat()}}

        @staticmethod
        def funding_fee(_symbol, start, end):
            assert start == opened_at
            assert end == closed_at
            return 0.1

    runner._close_position(True, Execution(), state, symbol, 105.0, "test_exit", closed_at,
                           0.0, 0.0)

    event = state.events[-1]
    assert event["action"] == "CLOSE"
    assert event["orderId"] == "close-1"
    assert event["price"] == 105.0
    assert event["fees"] == pytest.approx(0.41)
    assert event["funding_paid"] == pytest.approx(0.1)
    assert event["net_pnl"] == pytest.approx(9.69)
    assert event["pnl_recorded"] is True
    assert symbol not in state.portfolio.positions
    assert symbol not in state.entry_costs


@pytest.mark.parametrize("field", ["gross_pnl", "entry_fee", "exit_fee", "funding"])
@pytest.mark.parametrize("invalid", UNAVAILABLE_PNL_VALUES)
def test_demo_close_with_unavailable_pnl_records_unknown_and_stop_cooldown(field, invalid):
    symbol = "APR/USDT:USDT"
    opened_at = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    closed_at = opened_at + timedelta(hours=2)
    position = Position(symbol, Side.LONG, 2.0, 100.0, 95.0, opened_at,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState()
    state.portfolio.positions[symbol] = position
    values = dict(gross_pnl=10.0, entry_fee=0.2, exit_fee=0.21, funding=0.1)
    values[field] = invalid
    state.entry_costs[symbol] = {"entry_fee": values["entry_fee"]}
    fill = {"filled": 2.0, "average": 105.0, "gross_pnl": values["gross_pnl"],
            "exit_fee": values["exit_fee"], "closed_at": closed_at.isoformat()}
    if field == "gross_pnl" and invalid is None:
        fill["average"] = None  # No verified price from which to derive gross PnL.
    execution = SimpleNamespace(
        close=lambda *args, **kwargs: {"fill": fill},
        funding_fee=lambda *args: values["funding"],
    )

    runner._close_position(True, execution, state, symbol, 105.0,
                           "long_structure_invalid", closed_at, 0.0, 0.0)

    event = state.events[-1]
    assert event["net_pnl"] is None
    assert event["pnl_recorded"] is False
    assert symbol not in state.portfolio.positions
    assert state.portfolio.last_exit_at[symbol] == closed_at
    assert state.portfolio.last_exit_category[symbol] == "stop_loss"
    restored = MultiPaperState.from_dict(json.loads(json.dumps(state.to_dict(), allow_nan=False)))
    assert restored.events[-1]["net_pnl"] is None
    assert runner._structure_reentry_state(restored.events, symbol, Side.LONG) == (
        1, closed_at.isoformat(), False,
    )


@pytest.mark.parametrize("field", ["average", "filled"])
@pytest.mark.parametrize("invalid", UNAVAILABLE_PNL_VALUES)
def test_demo_close_pnl_fallback_requires_finite_fill_values(field, invalid):
    now = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    position = Position("APR/USDT:USDT", Side.LONG, 2, 100, 95, now,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState()
    state.entry_costs[position.symbol] = {"entry_fee": 0.2}
    fill = {"average": 105.0, "filled": 2.0, "exit_fee": 0.21, field: invalid}
    execution = SimpleNamespace(funding_fee=lambda *args: 0.1)

    gross, fees, funding, net = runner._demo_close_values(execution, state, position, fill, now)

    assert gross is None
    assert fees == pytest.approx(0.41)
    assert funding == 0.1
    assert net is None


@pytest.mark.parametrize("overrides", [
    {"entry_fee": 1e308, "exit_fee": 1e308},
    {"gross_pnl": 1e308, "funding": 1e308},
    {"gross_pnl": None, "average": 1e308},
])
def test_demo_close_pnl_checks_arithmetic_overflow(overrides):
    now = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    position = Position("APR/USDT:USDT", Side.LONG, 2, 100, 95, now,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState()
    values = dict(gross_pnl=10.0, entry_fee=0.2, exit_fee=0.21, funding=0.1, average=105.0)
    values.update(overrides)
    state.entry_costs[position.symbol] = {"entry_fee": values["entry_fee"]}
    execution = SimpleNamespace(funding_fee=lambda *args: values["funding"])
    fill = {"gross_pnl": values["gross_pnl"], "exit_fee": values["exit_fee"],
            "average": values["average"], "filled": 2.0}

    result = runner._demo_close_values(execution, state, position, fill, now)

    assert result[-1] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("side, expected", [(Side.LONG, 9.69), (Side.SHORT, -10.31)])
def test_demo_close_pnl_accepts_finite_numeric_strings(side, expected):
    now = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    position = Position("APR/USDT:USDT", side, 2, 100, 95 if side is Side.LONG else 110,
                        now, "test_entry", Regime.LOW.value)
    state = MultiPaperState()
    state.entry_costs[position.symbol] = {"entry_fee": "0.2"}
    execution = SimpleNamespace(funding_fee=lambda *args: "0.1")
    fill = {"average": "105", "filled": "2", "exit_fee": "0.21"}

    assert runner._demo_close_values(execution, state, position, fill, now)[-1] == pytest.approx(expected)


def test_demo_candidate_sync_records_verified_protective_stop_fill():
    symbol = "APR/USDT:USDT"
    opened_at = datetime(2026, 10, 26, 3, tzinfo=timezone.utc)
    closed_at = opened_at + timedelta(hours=2)
    position = Position(symbol, Side.LONG, 2.0, 100.0, 95.0, opened_at,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_mode = "demo_usdm"
    state.portfolio.positions[symbol] = position
    state.entry_costs[symbol] = {"entry_fee": 0.2}
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "algo-stop"}}

    class Execution:
        @staticmethod
        def sync_snapshot():
            return {"positions": {}}

        @staticmethod
        def resolve_protective_stop_algo_fill(requested_symbol, stop, quantity):
            assert (requested_symbol, stop["id"], quantity) == (symbol, "algo-stop", 2.0)
            return {"order_id": "stop-fill-1", "algo_id": "algo-stop", "quantity": 2.0,
                    "average": 95.0, "gross_pnl": -10.0, "exit_fee": 0.1,
                    "closed_at": closed_at.isoformat()}

        @staticmethod
        def cancel_saved_stop_if_open(requested_symbol, stop):
            assert (requested_symbol, stop["id"]) == (symbol, "algo-stop")

        @staticmethod
        def funding_fee(_symbol, start, end):
            assert start == opened_at
            assert end == closed_at
            return -0.05

    assert runner._demo_sync_stopped(Execution(), state, closed_at)
    event = state.events[-1]
    assert event["reason"] == "protective_stop_exchange_algo"
    assert event["orderId"] == "stop-fill-1"
    assert event["algoId"] == "algo-stop"
    assert event["price"] == 95.0
    assert event["fees"] == pytest.approx(0.3)
    assert event["funding_paid"] == pytest.approx(-0.05)
    assert event["net_pnl"] == pytest.approx(-10.35)
    assert event["pnl_recorded"] is True


def test_demo_sync_cancels_orphaned_bot_stops_after_flat_snapshot():
    state = MultiPaperState(initial_equity=5_000, cash=5_000)
    state.execution_mode = "demo_usdm"
    now = datetime(2026, 10, 26, 5, tzinfo=timezone.utc)

    class Execution:
        @staticmethod
        def sync_snapshot():
            return {"positions": {}}

        @staticmethod
        def cancel_orphaned_bot_stops(live_symbols):
            assert live_symbols == set()
            return [{"symbol": "APR/USDT:USDT", "algo_id": "orphan-stop"}]

    assert runner._demo_sync_stopped(Execution(), state, now)
    assert state.events[-1] == {
        "event": "demo_orphan_stop_cancelled", "at": now.isoformat(),
        "symbol": "APR/USDT:USDT", "algoId": "orphan-stop",
        "execution_source": "binance_usdm_demo",
    }
