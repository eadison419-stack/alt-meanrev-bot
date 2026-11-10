from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bot.config import load_settings
from bot.live import paper
from bot.live.bollinger_outcomes import BollingerOutcomeTracker
from bot.live.multi_paper import MultiPaperState
from bot.live.multi_runner import MultiPaperRunner
from bot.live import multi_runner as multi_runner_module
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.portfolio import Position
from bot.regime import Regime
from bot.signals import Action, Side, Signal
from bot.strategy_router import MEAN_REVERSION
from bot.time_filter import TimeFilterFlags


def test_tracked_symbols_use_positions_pending_then_ranked_hotlist_order():
    assert multi_runner_module._ordered_tracked_symbols(
        ["held", "shared"], ["pending", "held"], ["rank1", "shared", "rank2"],
    ) == ["held", "shared", "pending", "rank1", "rank2"]


def test_symbol_regime_summary_groups_uncertain_reasons_and_condition_failures():
    summary = multi_runner_module._symbol_regime_summary([
        {
            "checked_at": "2026-10-19T12:00:00+00:00",
            "last_closed_bar_at": "2026-10-19T12:00:00+00:00",
            "symbol": "AAA/USDT:USDT", "state": "UNCERTAIN", "reason": "structure_conflict",
            "evidence": {"condition_failures": {"TREND_UP": ["structure_not_up"],
                                                  "RANGE": ["structure_not_neutral"]},
                        "near_trend": True, "near_range_structure_only": True,
                        "near_miss_conditions": {
                            "TREND_UP": ["structure_not_up"],
                        }},
        },
        {
            "checked_at": "2026-10-19T12:00:01+00:00",
            "last_closed_bar_at": "2026-10-19T12:00:00+00:00",
            "symbol": "BBB/USDT:USDT", "state": "UNCERTAIN", "reason": "confirmation_not_met",
            "evidence": {"condition_failures": {}, "near_range": True,
                         "near_miss_conditions": {"RANGE": ["range_confirmation_not_met"]}},
        },
        {
            "checked_at": "2026-10-19T12:00:02+00:00",
            "last_closed_bar_at": "2026-10-19T12:00:00+00:00",
            "symbol": "CCC/USDT:USDT", "state": "TREND", "reason": "trend_confirmed_up",
            "evidence": {},
        },
    ])

    assert summary["state_counts"] == {"TREND": 1, "UNCERTAIN": 2}
    assert summary["uncertain_reason_counts"] == {"confirmation_not_met": 1, "structure_conflict": 1}
    assert summary["uncertain_symbols_by_reason"]["structure_conflict"] == ["AAA/USDT:USDT"]
    assert summary["condition_failure_counts"] == {
        "RANGE:structure_not_neutral": 1, "TREND_UP:structure_not_up": 1,
    }
    assert summary["confirmation_not_met_count"] == 1
    assert summary["near_trend_count"] == 1
    assert summary["near_range_count"] == 1
    assert summary["near_range_structure_only_count"] == 1
    assert summary["near_miss_condition_counts"] == {
        "RANGE:range_confirmation_not_met": 1,
        "TREND_UP:structure_not_up": 1,
    }


class FakeExchange:
    def __init__(self, start_ms):
        self.now_ms = start_ms
        self.symbol = "AAA/USDT:USDT"
        self.markets = {self.symbol: {
            "active": True, "swap": True, "quote": "USDT", "settle": "USDT",
            "info": {"contractType": "PERPETUAL", "underlyingType": "COIN", "underlyingSubType": ["Crypto"],
                     "onboardDate": 1_600_000_000_000,
                     "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"}]},
        }}

    def load_markets(self):
        return self.markets

    def milliseconds(self):
        return self.now_ms

    def fetch_tickers(self):
        return {self.symbol: {"last": 100.0, "close": 100.0, "quoteVolume": 50_000_000,
                              "info": {"count": "1000"}}}

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0}

    def fetch_mark_price(self, symbol):
        return {"markPrice": 100.0}

    def fetch_ohlcv(self, symbol, *, timeframe, limit):
        step = 3_600_000 if timeframe == "1h" else 14_400_000
        end = self.now_ms // step * step
        start = end - (limit - 1) * step
        result = []
        for index, timestamp in enumerate(range(start, end + 1, step)):
            close = 100 + np.sin(index / 9) * 0.1
            result.append([timestamp, close, close + 1, close - 1, close, 1])
        return result


def mock_candidate_match_scores(monkeypatch, scores):
    original_mean = multi_runner_module.evaluate_mean_reversion_match
    original_trend = multi_runner_module.evaluate_trend_match

    def evaluate_mean(symbol, *args, **kwargs):
        result = original_mean(symbol, *args, **kwargs)
        result["score"] = scores[symbol]["mean_reversion"]
        result["status"] = "scored"
        return result

    def evaluate_trend(symbol, *args, **kwargs):
        result = original_trend(symbol, *args, **kwargs)
        result.update({
            "trend_long_match_score": scores[symbol]["trend_long"],
            "trend_short_match_score": scores[symbol]["trend_short"],
            "trend_match_score": scores[symbol]["trend"],
            "direction_bias": scores[symbol]["bias"],
            "status": "scored",
        })
        return result

    monkeypatch.setattr(multi_runner_module, "evaluate_mean_reversion_match", evaluate_mean)
    monkeypatch.setattr(multi_runner_module, "evaluate_trend_match", evaluate_trend)


def test_runner_waits_for_next_hour_for_new_hot_symbols(monkeypatch):
    settings = load_settings(env_file=None)
    first = int(datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc).timestamp() * 1000)
    exchange = FakeExchange(first)
    state = MultiPaperState()

    def fake_regimes(frame, _settings):
        return pd.DataFrame({
            "atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
            "range_position": 0.2, "new_low": False, "new_high": False,
            "regime": Regime.LOW,
        }, index=frame.index)

    monkeypatch.setattr(paper, "classify_regimes", fake_regimes)
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(paper, "evaluate_time_filter", lambda *args, **kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(paper, "generate_signal", lambda *args, **kwargs: Signal(Side.LONG, Action.OPEN, "synthetic_hot_entry", 1.0))

    checkpoints = []
    runner = MultiPaperRunner(exchange, settings, state, top_n=1, universe_refresh_seconds=300,
                              checkpoint=lambda: checkpoints.append(set(state.portfolio.positions)))
    monkeypatch.setattr(runner, "_observe_symbol_regime", lambda *_args, **_kwargs: SimpleNamespace(
        state=SimpleNamespace(value="RANGE"),
    ))
    runner.cycle()
    symbol = exchange.symbol
    assert symbol == state.watchlist[0]["symbol"]
    assert state.watchlist[0]["hot_score"] == 50.0
    assert state.watchlist[0]["volume_burst_ratio"] is not None
    assert state.watchlist[0]["return_4h_pct"] is not None
    assert symbol not in state.portfolio.positions
    assert any(event.get("reason") == "symbol_outside_hot_universe" for event in state.events)

    exchange.now_ms = int(datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc).timestamp() * 1000)
    runner.cycle()

    assert symbol in state.portfolio.positions
    assert state.portfolio.positions[symbol].opened_at == datetime(2026, 1, 5, 11, tzinfo=timezone.utc)
    assert state.portfolio.positions[symbol].strategy_source == MEAN_REVERSION
    accepted = next(event for event in state.events
                    if event.get("event") == "strategy_route_evaluated"
                    and event.get("result_action") == Action.OPEN.value)
    assert accepted["symbol"] == symbol
    assert accepted["direction"] == Side.LONG.value
    assert accepted["signal_source"] == MEAN_REVERSION
    assert accepted["candidate_source"] == MEAN_REVERSION
    assert accepted["signal_timestamp"]
    assert any(symbol in saved_positions for saved_positions in checkpoints)
    assert math.isclose(state.portfolio.positions[symbol].quantity / 0.001,
                        round(state.portfolio.positions[symbol].quantity / 0.001), abs_tol=1e-7)

    exchange.now_ms = int(datetime(2026, 1, 5, 23, 10, tzinfo=timezone.utc).timestamp() * 1000)
    runner.cycle()
    checkpoint = next(event for event in state.events
                      if event.get("event") == "management_observation_checkpoint")
    assert checkpoint["checkpoint"] == "12h"
    assert checkpoint["hotlist_rank"] == 1
    assert checkpoint["market_regime"] == "LOW"
    assert checkpoint["entry_reason"] == "synthetic_hot_entry"


def test_universe_refresh_parallel_matches_serial(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    second = "BBB/USDT:USDT"
    exchange.markets[second] = {**exchange.markets[exchange.symbol], "info": {
        **exchange.markets[exchange.symbol]["info"], "onboardDate": 1_600_000_000_000,
    }}
    tickers = {
        exchange.symbol: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "1000"}},
        second: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "2000"}},
    }
    monkeypatch.setattr(exchange, "fetch_tickers", lambda: tickers)

    runner = MultiPaperRunner(exchange, settings, MultiPaperState(), top_n=1, fetch_workers=4)
    # Each worker thread builds its own exchange via this factory; FakeExchange is
    # stateless, so the threaded path must reproduce the serial ranking exactly.
    monkeypatch.setattr(runner, "_build_fetch_worker_exchange",
                        lambda: FakeExchange(int(now.timestamp() * 1000)))

    assert runner._refresh_universe(tickers, now)
    assert [row["symbol"] for row in runner._ranked] == [second]


def test_universe_refresh_scores_all_tradable_symbols_before_selecting_top_n(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    second = "BBB/USDT:USDT"
    exchange.markets[second] = {**exchange.markets[exchange.symbol], "info": {
        **exchange.markets[exchange.symbol]["info"], "onboardDate": 1_600_000_000_000,
    }}
    tickers = {
        exchange.symbol: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "1000"}},
        second: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "2000"}},
    }
    monkeypatch.setattr(exchange, "fetch_tickers", lambda: tickers)
    fetched = []
    fetch_ohlcv = exchange.fetch_ohlcv

    def record_fetch(symbol, **kwargs):
        fetched.append((symbol, kwargs["timeframe"]))
        return fetch_ohlcv(symbol, **kwargs)

    monkeypatch.setattr(exchange, "fetch_ohlcv", record_fetch)
    logged = []
    monkeypatch.setattr(multi_runner_module.logger, "info", lambda message, *args: logged.append((message, args)))
    runner = MultiPaperRunner(exchange, settings, MultiPaperState(), top_n=1)

    assert runner._refresh_universe(tickers, now)

    assert set(fetched) == {
        (symbol, timeframe) for symbol in (exchange.symbol, second) for timeframe in ("1h", "4h")
    }
    assert [row["symbol"] for row in runner._ranked] == [second]
    refresh = next(json.loads(args[0]) for message, args in logged
                   if message == "Hot crypto universe refresh {}")
    assert refresh["basic_tradable_count"] == 2
    assert refresh["selected_count"] == 1
    assert refresh["selected_candidates"][0]["symbol"] == second
    assert refresh["selected_candidates"][0]["hot_score"] == runner._ranked[0]["hot_score"]
    evaluated = refresh["mean_reversion_match"]["candidates"]
    assert refresh["mean_reversion_match"]["mode"] == "observation_only"
    assert refresh["mean_reversion_match"]["evaluated_count"] == 2
    assert {row["symbol"] for row in evaluated} == {exchange.symbol, second}
    assert all("mean_reversion_match_score" in row and "hot_score" in row for row in evaluated)
    trend_evaluated = refresh["trend_match"]["candidates"]
    assert refresh["trend_match"]["mode"] == "observation_only"
    assert refresh["trend_match"]["evaluated_count"] == 2
    assert refresh["trend_match"]["scored_count"] == 2
    assert {row["symbol"] for row in trend_evaluated} == {exchange.symbol, second}
    assert all("trend_match_score" in row and "metrics" in row for row in trend_evaluated)
    assert all("trend_long_match_score" in row for row in trend_evaluated)
    assert all("trend_short_match_score" in row and "direction_bias" in row for row in trend_evaluated)
    assert all("long_component_scores" in row and "short_component_scores" in row for row in trend_evaluated)
    assert all(row["last_closed_bar_at"] for row in trend_evaluated)
    assert all("mean_reversion_match_score" in row and "hot_score" in row for row in trend_evaluated)
    assert all("reasons" in row and "component_scores" in row for row in evaluated)
    assert refresh["candidate_selection_mode"] == "hot_top_n"
    assert refresh["hot_top_n_candidates"][0]["symbol"] == second


def test_universe_refresh_records_bollinger_hits_to_isolated_outcome_state(monkeypatch, tmp_path):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    state = MultiPaperState()
    outcome_path = tmp_path / "portfolio.bollinger-outcomes.json"
    tracker = BollingerOutcomeTracker(outcome_path, settings)
    logged = []
    monkeypatch.setattr(multi_runner_module.logger, "info", lambda message, *args: logged.append((message, args)))

    def matched_observation(*_args):
        signals = {
            name: {"matched": name == "long_start", "failed_conditions": []}
            for name in ("long_start", "short_start", "long_trend", "short_trend")
        }
        return {
            "status": "scored", "signal_close": 99.5,
            "last_closed_bar_at": "2026-01-05T04:00:00+00:00",
            "pct_b": 0.2, "bbw": 0.01, "signals": signals,
        }

    monkeypatch.setattr(multi_runner_module, "_bollinger_directional_observation", matched_observation)
    runner = MultiPaperRunner(exchange, settings, state, top_n=1, bollinger_outcome_tracker=tracker)
    tickers = exchange.fetch_tickers()

    assert runner._refresh_universe(tickers, now)

    assert len(tracker.state["samples"]) == 1
    sample = next(iter(tracker.state["samples"].values()))
    assert sample["signal_type"] == "long_start"
    assert sample["entry_reference"] == "first_tracker_observation_ticker_last"
    assert sample["entry_price"] == 100.0
    assert sample["entry_timing_status"] == "delayed"
    assert tracker.active_symbols == {exchange.symbol}
    assert state.portfolio.positions == {}
    assert state.pending_by_symbol == {}
    assert outcome_path.is_file()

    refresh = next(json.loads(args[0]) for message, args in logged
                   if message == "Hot crypto universe refresh {}")
    tracking = refresh["observation_comparison"]["bollinger"]["outcome_tracking"]
    assert tracking["enabled"] is True
    assert tracking["new_samples"] == 1
    assert tracking["summary"]["sample_count"] == 1
    assert tracking["summary"]["eligible_sample_count"] == 0
    assert tracking["summary"]["delayed_sample_count"] == 1





def test_observation_comparison_covers_assigned_pool_members_without_changing_selection(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    formal = "BBB/USDT:USDT"
    exchange.markets[formal] = {**exchange.markets[exchange.symbol], "info": {
        **exchange.markets[exchange.symbol]["info"], "onboardDate": 1_600_000_000_000,
    }}
    tickers = {
        exchange.symbol: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "1000"}},
        formal: {"last": 100.0, "quoteVolume": 25_000_000, "info": {"count": "2000"}},
    }
    mock_candidate_match_scores(monkeypatch, {
        exchange.symbol: {"mean_reversion": 20, "trend_long": 30, "trend_short": 25,
                          "trend": 30, "bias": "neutral"},
        formal: {"mean_reversion": 20, "trend_long": 80, "trend_short": 30,
                 "trend": 80, "bias": "long"},
    })
    original = multi_runner_module.evaluate_trend_match

    def add_low_base_metrics(symbol, *args, **kwargs):
        result = original(symbol, *args, **kwargs)
        result["metrics"] = {
            "range_position_pct": 30.0,
            "ema50_distance_pct": -2.0,
            "ema50_slope_pct": -0.1,
            "recent_return_pct": 4.0,
            "structure": "NEUTRAL",
            "adx": 18.0,
        }
        result["status"] = "scored"
        result["last_closed_bar_at"] = "2026-01-05T08:00:00+00:00"
        return result

    monkeypatch.setattr(multi_runner_module, "evaluate_trend_match", add_low_base_metrics)
    logged = []
    monkeypatch.setattr(multi_runner_module.logger, "info", lambda message, *args: logged.append((message, args)))
    runner = MultiPaperRunner(exchange, settings, MultiPaperState(), top_n=1)

    assert runner._refresh_universe(tickers, now)
    refresh = next(json.loads(args[0]) for message, args in logged
                   if message == "Hot crypto universe refresh {}")
    watch = refresh["low_position_watch"]

    assert watch["mode"] == "observation_only"
    assert watch["affects_candidate_selection"] is False
    assert {row["symbol"] for row in watch["candidates"]} == {exchange.symbol, formal}
    assert watch["candidate_count"] == 2
    comparison = refresh["observation_comparison"]
    assert comparison["mode"] == "independent_log_only_comparison"
    assert comparison["affects_candidate_selection"] is False
    assert comparison["legacy_rules"]["long_candidate_count"] == 2
    assert comparison["bollinger"]["enabled"] is True
    assert comparison["evaluated_count"] == 2
    assert {row["symbol"] for row in runner._ranked} == {formal}


def test_legacy_observer_adds_symmetric_short_rules():
    settings = load_settings(env_file=None)
    assessment = {
        "status": "scored",
        "symbol": "AAA/USDT:USDT",
        "last_closed_bar_at": "2026-01-05T08:00:00+00:00",
        "metrics": {
            "range_position_pct": 70.0,
            "ema50_distance_pct": 2.0,
            "ema50_slope_pct": 0.1,
            "recent_return_pct": -4.0,
            "structure": "DOWN",
        },
    }

    result = multi_runner_module._legacy_directional_observation(assessment, settings.low_position_watch)

    assert result["directions"]["short"] == {"matched": True, "failed_conditions": []}
    assert result["directions"]["long"]["matched"] is False


def test_bollinger_observer_uses_closed_bars_and_reports_long_start():
    settings = load_settings(env_file=None)
    index = np.arange(101)
    close = 100 + 0.5 * np.sin(2 * np.pi * index / 20 - np.pi / 2)
    frame = pd.DataFrame({"close": close, "volume": np.full(len(close), 100.0)},
                         index=pd.date_range("2026-01-01", periods=len(close), freq="4h", tz="UTC"))

    result = multi_runner_module._bollinger_directional_observation(
        frame, {}, settings.bollinger_observation,
    )

    assert result["status"] == "scored"
    assert result["signals"]["long_start"]["matched"] is True
    assert result["signals"]["short_start"]["matched"] is False
    assert result["middle_direction"] == "flat"
    assert result["volume_above_20_mean"] is False


@pytest.mark.parametrize("metric_update", [
    {"range_position_pct": 46.0},
    {"ema50_distance_pct": 9.0},
    {"recent_return_pct": 16.0},
    {"ema50_slope_pct": -0.6},
    {"structure": "DOWN"},
])
def test_low_position_watch_rejects_extended_or_weak_setups(metric_update):
    settings = load_settings(env_file=None)
    assessment = {
        "status": "scored",
        "symbol": "AAA/USDT:USDT",
        "metrics": {
            "range_position_pct": 30.0,
            "ema50_distance_pct": -2.0,
            "ema50_slope_pct": -0.1,
            "recent_return_pct": 4.0,
            "structure": "NEUTRAL",
        },
    }
    assessment["metrics"].update(metric_update)

    assert multi_runner_module._low_position_watch_candidate(
        assessment, {}, settings.low_position_watch,
    ) is None



def test_runner_logs_btc_market_risk_observation_without_budget(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    state = MultiPaperState()
    runner = MultiPaperRunner(exchange, settings, state, top_n=1)
    calls = []
    original = exchange.fetch_ohlcv

    def track(symbol, *, timeframe, limit):
        calls.append((symbol, timeframe, limit))
        return original(symbol, timeframe=timeframe, limit=limit)

    monkeypatch.setattr(exchange, "fetch_ohlcv", track)
    assessment = runner._update_market_risk(now)

    assert calls == [(settings.market_risk.symbol, "4h", 100)]
    assert assessment.state in MarketRiskState
    assert state.market_risk_status["mode"] == "entry_gate"
    assert state.market_risk_status["symbol"] == settings.market_risk.symbol
    assert state.market_risk_status["checked_at"] == now.isoformat()
    assert "gross_notional_cap_pct" not in state.market_risk_status
    assert any(event["event"] == "market_risk_observed" for event in state.events)


def test_hot_symbol_regime_is_logged_once_per_closed_bar_without_state_mutation(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    state = MultiPaperState()
    runner = MultiPaperRunner(exchange, settings, state, top_n=1)
    index = pd.date_range(end="2026-01-05 04:00:00+00:00", periods=100, freq="4h", name="timestamp")
    bars = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0,
    }, index=index)
    observations = []
    monkeypatch.setattr(multi_runner_module.logger, "info",
                        lambda _message, payload: observations.append(json.loads(payload)))

    runner._observe_symbol_regime("AAA/USDT:USDT", bars, now)
    runner._observe_symbol_regime("AAA/USDT:USDT", bars, now + timedelta(minutes=1))

    assert len(observations) == 1
    assert observations[0]["mode"] == "strategy_routing"
    assert observations[0]["state"] == "RANGE"
    assert observations[0]["evidence"]["adx"] is not None
    thresholds = observations[0]["evidence"]["thresholds"]
    assert thresholds["trend_adx_min"] == 25
    assert thresholds["confirm_bars"] == 3
    assert thresholds["range_confirm_bars"] == 2
    assert state.events == []
    assert state.symbol_status == {}


def test_pause_new_blocks_new_entries_without_changing_position_management(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    state = MultiPaperState()
    runner = MultiPaperRunner(exchange, settings, state, top_n=1)
    monkeypatch.setattr(runner, "_update_market_risk", lambda _now: MarketRiskAssessment(
        MarketRiskState.PAUSE_NEW, "synthetic_pause", now, "BTC/USDT:USDT", "4h", {},
    ))
    monkeypatch.setattr(paper, "classify_regimes", lambda frame, _settings: pd.DataFrame({
        "atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
        "range_position": 0.2, "new_low": False, "new_high": False, "regime": Regime.LOW,
    }, index=frame.index))
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(paper, "evaluate_time_filter",
                        lambda *args, **kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(paper, "generate_signal", lambda *args, **kwargs: Signal(
        Side.LONG, Action.OPEN, "synthetic_entry", 1.0,
    ))

    runner.cycle()
    exchange.now_ms = int(datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc).timestamp() * 1000)
    runner.cycle()

    assert exchange.symbol not in state.portfolio.positions
    routed = [event for event in state.events if event.get("event") == "strategy_route_evaluated"]
    assert routed
    assert all(event["market_risk_state"] == "PAUSE_NEW" for event in routed)
    assert all(event["result_action"] != Action.OPEN.value for event in routed)
    assert any(event.get("entry_block_reason") == "market_risk_paused" for event in routed)


def test_pause_new_cancels_an_already_queued_open(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc)
    exchange = FakeExchange(int(now.timestamp() * 1000))
    state = MultiPaperState()
    symbol = exchange.symbol
    execute_at = now.replace(minute=0, second=0, microsecond=0)
    state.pending_by_symbol[symbol] = {
        "execute_at": execute_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "queued_before_pause", "strength": 1.0, "close_first": False,
        "size_multiplier": 1.0, "atr": 1.0, "regime": Regime.LOW.value,
        "source": MEAN_REVERSION, "signal_time": (execute_at - timedelta(hours=1)).isoformat(),
    }
    runner = MultiPaperRunner(exchange, settings, state, top_n=1)
    monkeypatch.setattr(runner, "_update_market_risk", lambda _now: MarketRiskAssessment(
        MarketRiskState.PAUSE_NEW, "synthetic_pause", now, "BTC/USDT:USDT", "4h", {},
    ))

    runner.cycle()

    assert symbol not in state.portfolio.positions
    assert symbol not in state.pending_by_symbol
    assert any(event.get("reason") == "market_risk_paused" for event in state.events)


def test_runner_keeps_pending_symbols_tracked_outside_hotlist(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc)
    hot_symbol = "AAA/USDT:USDT"
    pending_symbol = "BBB/USDT:USDT"
    exchange = FakeExchange(int(now.timestamp() * 1000))
    exchange.markets[pending_symbol] = exchange.markets[hot_symbol]
    ticker = {"last": 100.0, "close": 100.0, "quoteVolume": 50_000_000,
              "info": {"count": "1000"}}
    monkeypatch.setattr(exchange, "fetch_tickers", lambda: {hot_symbol: ticker, pending_symbol: ticker})

    hour_open = now.replace(minute=0, second=0, microsecond=0)
    hour_key = pd.Timestamp(hour_open).isoformat()
    signal_time = now - timedelta(minutes=20)
    state = MultiPaperState()
    state.pending_by_symbol[pending_symbol] = {
        "execution_type": "pullback", "execute_at": signal_time.isoformat(), "action": "OPEN",
        "side": "short", "reason": "expired_test", "signal_time": signal_time.isoformat(),
        "confirmed_at": signal_time.isoformat(), "expires_at": (now - timedelta(minutes=5)).isoformat(),
        "reference_price": 100.0, "pullback_price": 101.0, "entry_structure_high": 99.0,
    }
    state.last_processed_by_symbol[hot_symbol] = hour_key
    state.last_processed_by_symbol[pending_symbol] = hour_key
    state.watchlist = [{"symbol": hot_symbol, "rank": 1}]

    runner = MultiPaperRunner(exchange, settings, state, top_n=1)
    runner._ranked = [{"symbol": hot_symbol}]
    monkeypatch.setattr(runner, "_refresh_universe", lambda *_args: False)

    runner.cycle()

    assert pending_symbol not in runner._ranked
    assert pending_symbol not in state.pending_by_symbol
    assert state.portfolio.positions == {}
    cancelled = next(event for event in state.events
                     if event.get("event") == "pending_entry_cancelled" and event.get("symbol") == pending_symbol)
    assert cancelled["reason"] == "timeout"


def test_exchange_inconsistency_blocks_new_entries_until_synchronized():
    settings = load_settings(env_file=None)
    start = int(datetime(2026, 1, 5, 10, 10, tzinfo=timezone.utc).timestamp() * 1000)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent")
    runner = MultiPaperRunner(FakeExchange(start), settings, state, top_n=1)
    runner._ranked = [{"symbol": "AAA/USDT:USDT"}]
    runner.execution = object()

    assert runner._entry_allowed("AAA/USDT:USDT") is False
    state.execution_status = "synchronized"
    assert runner._entry_allowed("AAA/USDT:USDT") is True


def test_startup_closes_exchange_confirmed_stale_position_with_reduce_only_intent(monkeypatch):
    symbol = "AAA/USDT:USDT"
    now = datetime(2026, 1, 6, 12, tzinfo=timezone.utc)
    opened = now - timedelta(hours=25)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="synchronized")
    state.portfolio.positions[symbol] = Position(
        symbol, Side.LONG, 1.0, 100.0, 95.0, opened, "entry", Regime.LOW.value,
    )
    state.exchange_snapshot = {"positions": {symbol: {"side": "long", "quantity": 1.0, "entry_price": 100.0}}}

    class StartupExecution:
        def __init__(self):
            self.close_calls = []

        def close(self, position, *, stop, reason):
            self.close_calls.append((position.side, stop, reason))
            return {"fill": {"filled": 1.0, "average": 101.0, "fee_cost": 0.0, "id": "reduce-only-id"},
                    "host": "demo-fapi"}

        def sync_snapshot(self):
            return {"available_usdt": 5_000.0, "positions": {}}

    execution = StartupExecution()
    runner = MultiPaperRunner(FakeExchange(int(now.timestamp() * 1000)), load_settings(env_file=None), state,
                              execution=execution)
    assert runner._close_startup_stale_positions(now, {symbol: {"last": 101.0}}) is True
    assert execution.close_calls == [(Side.LONG, None, "startup_stale_exit")]
    intent = next(event for event in state.events if event.get("event") == "order_intent")
    assert intent["decision"] == "demo_reduce_only_close"
    assert intent["intent"]["reason"] == "startup_stale_exit"
    assert state.portfolio.positions == {}
    assert runner._close_startup_stale_positions(now, {symbol: {"last": 101.0}}) is False


def test_startup_does_not_close_confirmed_2r_trailing_runner_for_time_alone():
    symbol = "AAA/USDT:USDT"
    now = datetime(2026, 1, 6, 12, tzinfo=timezone.utc)
    opened = now - timedelta(hours=25)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="synchronized")
    position = Position(symbol, Side.LONG, 0.2, 100.0, 95.0, opened, "entry", Regime.LOW.value)
    state.portfolio.positions[symbol] = position
    state.exit_management_by_symbol[symbol] = {
        "trailing_started": True, "trailing_trigger_r": 2.0,
        "protective_stop": 102.0,
    }
    state.exchange_snapshot = {"positions": {symbol: {"side": "long", "quantity": 0.2,
                                                          "entry_price": 100.0}}}

    class StartupExecution:
        def __init__(self):
            self.close_calls = []

        def close(self, position, *, stop, reason):
            self.close_calls.append(reason)
            return {"fill": {"filled": position.quantity, "average": 100.0,
                              "fee_cost": 0.0, "id": "reduce-only-id"}, "host": "demo-fapi"}

    execution = StartupExecution()
    runner = MultiPaperRunner(FakeExchange(int(now.timestamp() * 1000)), load_settings(env_file=None), state,
                              execution=execution)

    assert runner._close_startup_stale_positions(now, {symbol: {"last": 101.0}}) is False
    assert execution.close_calls == []
    assert state.portfolio.positions[symbol] is position


def test_unexplained_execution_lock_recovers_only_after_protective_stop_audit():
    from datetime import datetime, timezone

    symbol = "AAA/USDT:USDT"
    at = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent")
    state.portfolio.positions[symbol] = Position(
        symbol, Side.LONG, 2.0, 100.0, 95.0, at, "entry", Regime.LOW.value,
    )
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "active-stop", "stop_price": 95.0}}

    class StopAuditExecution:
        def __init__(self):
            self.mismatches = []
            self.audited = False

        def audit_active_protective_stops(self, positions, tracked_orders):
            self.audited = True
            assert list(positions) == [symbol]
            assert tracked_orders[symbol]["stop"]["id"] == "active-stop"
            return self.mismatches

    execution = StopAuditExecution()
    runner = MultiPaperRunner(FakeExchange(int(at.timestamp() * 1000)), load_settings(env_file=None),
                              state, execution=execution)

    errors = runner._recover_unexplained_execution_lock([], [], at)

    assert errors == []
    assert execution.audited is True
    assert state.execution_status == "synchronized"
    assert state.events[-1]["event"] == "execution_status_recovered"

    state.execution_status = "inconsistent"
    execution.mismatches = [f"{symbol}: stop quantity differs"]
    errors = runner._recover_unexplained_execution_lock([], [], at)
    assert errors == execution.mismatches
    assert state.execution_status == "inconsistent"


def test_demo_cycle_audits_active_stops_and_clears_only_resolved_audit_lock(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    hour_key = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0)).isoformat()
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent",
                            inconsistencies=[f"{symbol}: saved protective stop is not active at the saved side/quantity/price"])
    state.portfolio.positions[symbol] = Position(
        symbol, Side.LONG, 1.0, 100.0, 95.0, now - timedelta(hours=1), "entry", Regime.LOW.value,
    )
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "active-stop", "stop_price": 95.0}}
    state.last_processed_by_symbol[symbol] = hour_key

    class PeriodicAuditExecution:
        def __init__(self):
            self.mismatches = []
            self.audit_calls = 0

        def sync_snapshot(self):
            return {"available_usdt": 5000.0, "positions": {
                symbol: {"side": "long", "quantity": 1.0, "entry_price": 100.0},
            }, "at": now.isoformat()}

        def _market_rules(self, _symbol):
            return (0.001, 0.001, 0.0)

        def audit_recent_fills(self, *_args):
            return []

        def audit_active_protective_stops(self, positions, tracked_orders):
            self.audit_calls += 1
            assert list(positions) == [symbol]
            assert tracked_orders[symbol]["stop"]["id"] == "active-stop"
            return self.mismatches

    execution = PeriodicAuditExecution()
    exchange = FakeExchange(int(now.timestamp() * 1000))
    runner = MultiPaperRunner(exchange, settings, state, execution=execution)
    runner._ranked = [{"symbol": symbol}]
    runner._startup_scan_pending = False
    monkeypatch.setattr(runner, "_refresh_universe", lambda *_args: False)

    runner.cycle()

    assert execution.audit_calls == 1
    assert state.protective_stop_audit["status"] == "verified"
    assert state.protective_stop_audit["at"] == now.isoformat()
    assert state.execution_status == "synchronized"
    assert state.inconsistencies == []

    execution.mismatches = [f"{symbol}: saved protective stop is not active at the saved side/quantity/price"]
    runner.cycle()
    assert execution.audit_calls == 2
    assert state.protective_stop_audit["status"] == "mismatch"
    assert state.execution_status == "inconsistent"
    assert state.inconsistencies == execution.mismatches
    assert runner._entry_allowed(symbol) is False


def test_runner_checks_max_hold_on_regular_poll_without_waiting_for_next_hour(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 11, 0, 10, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    hour_key = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0)).isoformat()
    state = MultiPaperState()
    state.portfolio.positions[symbol] = Position(
        symbol, Side.LONG, 1.0, 100.0, 90.0, now - timedelta(hours=25), "entry", Regime.LOW.value,
    )
    state.last_processed_by_symbol[symbol] = hour_key
    runner = MultiPaperRunner(FakeExchange(int(now.timestamp() * 1000)), settings, state, top_n=1)
    runner._ranked = [{"symbol": symbol}]
    monkeypatch.setattr(runner, "_refresh_universe", lambda *_args: False)

    runner.cycle()

    assert symbol not in state.portfolio.positions
    close = next(event for event in state.events
                 if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close["reason"] == "max_hold_24h"


def test_confirmed_partial_fill_is_recorded_when_stop_cleanup_is_pending():
    from types import SimpleNamespace

    symbol = "ADA/USDT:USDT"
    at = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    settings = load_settings(env_file=None)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="synchronized")
    state.portfolio.positions[symbol] = Position(
        symbol, Side.SHORT, 6.0, 100.0, 110.0, at, "entry", Regime.HIGH.value,
    )
    state.entry_costs[symbol] = {"entry_fee": 0.6, "entry_slippage": 0.0, "funding_paid": 0.0}
    state.exit_management_by_symbol[symbol] = {
        "policy": "proposal_v1", "entry_atr": 1.0, "initial_quantity": 10.0,
        "initial_entry_fee": 1.0, "tp1_taken": True, "tp2_taken": False,
        "no_progress_reduced": False, "best_price": 90.0, "worst_price": 105.0,
        "protective_stop": 90.8, "latest_regime": "MID", "realized_gross": 0.0,
        "realized_exit_fees": 0.0, "realized_entry_fee": 0.4, "realized_funding": 0.0,
        "realized_exit_quantity": 4.0, "realized_exit_notional": 380.0,
    }
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "old-stop", "stop_price": 90.8}}

    class FailedStopCleanupExecution:
        host = "https://demo-fapi.binance.com"

        def __init__(self):
            self.reduce_calls = 0

        def _market_rules(self, _symbol):
            return (1.0, 1.0, 0.0)

        def reduce(self, position, quantity, *, stop, reason):
            self.reduce_calls += 1
            assert quantity == 4.0 and reason == "take_profit_2"
            assert position.quantity == 6.0
            return {
                "fill": {"id": "tp2-order", "filled": 4.0, "average": 88.0, "fee_cost": 0.1},
                "position": {"side": "short", "quantity": 2.0},
                "stop": {"id": "replacement-stop", "stop_price": 90.8, "amount": 2.0},
                "stale_stops": [{"algoId": "orphan-stop", "quantity": "10"}],
                "stop_cleanup_error": None,
            }

        def sync_snapshot(self):
            return {"available_usdt": 5000.0, "positions": {}}

    execution = FailedStopCleanupExecution()
    trader = paper.PaperTrader(state.for_symbol(symbol), settings, execution=execution)

    assert trader._manage_proposal_tick(88.0, at.replace(hour=11)) is True

    assert state.portfolio.positions[symbol].quantity == 2.0
    assert state.exit_management_by_symbol[symbol]["tp2_taken"] is True
    assert state.execution_status == "inconsistent"
    assert state.exchange_orders_by_symbol[symbol]["stale_stops"][0]["algoId"] == "orphan-stop"
    reduction = next(event for event in state.events if event.get("orderId") == "tp2-order")
    assert reduction["event"] == "paper_partial_fill"
    assert reduction["remaining_quantity"] == 2.0
    assert reduction["execution_source"] == "binance_usdm_demo"
    trader._manage_proposal_tick(88.0, at.replace(hour=11))
    assert execution.reduce_calls == 1


def test_runner_retries_stale_bot_stop_cleanup_and_tracks_unresolved_orders():
    symbol = "AAA/USDT:USDT"
    at = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent")
    state.portfolio.positions[symbol] = Position(
        symbol, Side.SHORT, 6.0, 100.0, 110.0, at, "entry", Regime.HIGH.value,
    )
    state.exchange_orders_by_symbol[symbol] = {
        "stop": {"id": "active-stop", "amount": 6.0},
        "stale_stops": [{"algoId": "orphan-stop", "quantity": "10"}],
    }

    class StopReconciliationExecution:
        def __init__(self):
            self.remaining = [{"algoId": "orphan-stop"}]
            self.calls = 0

        def _market_rules(self, _symbol):
            return (0.001, 0.001, 0.0)

        def reconcile_reduced_position_stops(self, requested_symbol, side, stop, quantity):
            self.calls += 1
            assert (requested_symbol, side, stop["id"], quantity) == (symbol, "short", "active-stop", 6.0)
            return self.remaining

    execution = StopReconciliationExecution()
    runner = MultiPaperRunner(FakeExchange(int(at.timestamp() * 1000)), load_settings(env_file=None),
                              state, execution=execution)
    snapshot = {"positions": {symbol: {"side": "short", "quantity": 6.0}}}

    cleaned, pending = runner._retry_protective_stop_cleanup(snapshot, at)
    assert cleaned == set()
    assert pending == {symbol}
    assert state.exchange_orders_by_symbol[symbol]["stale_stops"] == execution.remaining

    execution.remaining = []
    mismatched = {"positions": {symbol: {"side": "short", "quantity": 5.0}}}
    cleaned, pending = runner._retry_protective_stop_cleanup(mismatched, at)
    assert cleaned == set()
    assert pending == {symbol}
    assert execution.calls == 1

    cleaned, pending = runner._retry_protective_stop_cleanup(snapshot, at)
    assert cleaned == {symbol}
    assert pending == set()
    assert "stale_stops" not in state.exchange_orders_by_symbol[symbol]
    assert state.events[-1]["event"] == "protective_stop_cleanup_recovered"

    state.exchange_orders_by_symbol[symbol]["snapshot_reconciliation_required"] = True
    cleaned, pending = runner._retry_protective_stop_cleanup(snapshot, at)
    assert cleaned == {symbol}
    assert pending == set()
    assert "snapshot_reconciliation_required" not in state.exchange_orders_by_symbol[symbol]


def test_external_fill_and_flat_exchange_position_reconcile_stale_local_leg():
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 12, 10, tzinfo=timezone.utc)
    opened_at = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    order_id = "external-123"
    position = Position(symbol, Side.SHORT, 10.0, 100.0, 110.0, opened_at,
                        "test_entry", Regime.HIGH.value)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent")
    state.portfolio.positions[symbol] = position
    state.entry_costs[symbol] = {"entry_fee": 1.0}
    state.isolated_margin_by_symbol[symbol] = {"initial_margin": 100.0}
    state.exit_management_by_symbol[symbol] = {"enabled": True}
    state.pending_by_symbol[symbol] = {"action": "OPEN"}
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "stop-456"}}
    state.mark_prices[symbol] = 99.0
    state.foreign_order_ids.append(order_id)
    state.events.append({"event": "foreign_exchange_fill_warning", "at": now.isoformat(),
                         "symbol": symbol, "orderId": order_id})

    class FakeExecution:
        def __init__(self):
            self.cancelled = []

        def cancel_saved_stop_if_open(self, requested_symbol, stop):
            self.cancelled.append((requested_symbol, stop["id"]))

    execution = FakeExecution()
    runner = MultiPaperRunner(FakeExchange(int(now.timestamp() * 1000)), settings, state,
                              execution=execution)
    reconciled, resolved = runner._reconcile_external_flat_positions(
        {"positions": {}, "at": now.isoformat()}, now)

    assert reconciled == {symbol}
    assert resolved == {order_id}
    assert state.foreign_order_ids == []
    assert execution.cancelled == [(symbol, "stop-456")]
    assert symbol not in state.portfolio.positions
    assert state.portfolio.last_exit_at[symbol] == now
    assert symbol not in state.entry_costs
    assert symbol not in state.isolated_margin_by_symbol
    assert symbol not in state.exit_management_by_symbol
    assert symbol not in state.pending_by_symbol
    assert symbol not in state.exchange_orders_by_symbol
    assert symbol not in state.mark_prices
    event = state.events[-1]
    assert event["event"] == "external_position_reconciled_flat"
    assert event["close_price"] is None
    assert event["net_pnl"] is None
    assert event["pnl_recorded"] is False


def test_bot_stop_fill_is_recorded_only_after_algo_and_trade_resolution():
    settings = load_settings(env_file=None)
    now = datetime(2026, 1, 5, 12, 10, tzinfo=timezone.utc)
    opened_at = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    order_id = "bot-stop-order"
    position = Position(symbol, Side.LONG, 2.0, 100.0, 95.0, opened_at,
                        "test_entry", Regime.LOW.value)
    state = MultiPaperState(execution_mode="demo_usdm", execution_status="inconsistent")
    state.portfolio.positions[symbol] = position
    state.entry_costs[symbol] = {"entry_fee": 0.2, "funding_paid": 0.1}
    state.exit_management_by_symbol[symbol] = {"realized_gross": 1.0, "realized_exit_fees": 0.05,
                                               "realized_entry_fee": 0.1, "realized_funding": 0.02}
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "bot-algo-stop"}}
    state.foreign_order_ids.append(order_id)
    state.events.append({"event": "foreign_exchange_fill_warning", "at": now.isoformat(),
                         "symbol": symbol, "orderId": order_id})

    class FakeExecution:
        def cancel_saved_stop_if_open(self, requested_symbol, stop):
            assert requested_symbol == symbol

        def resolve_protective_stop_fill(self, requested_symbol, requested_order_id, stop, quantity):
            assert (requested_symbol, requested_order_id, stop["id"], quantity) == (
                symbol, order_id, "bot-algo-stop", 2.0)
            return {"order_id": order_id, "algo_id": "bot-algo-stop", "average": 95.0,
                    "gross_pnl": -10.0, "exit_fee": 0.1, "closed_at": now.isoformat()}

    runner = MultiPaperRunner(FakeExchange(int(now.timestamp() * 1000)), settings, state,
                              execution=FakeExecution())
    reconciled, resolved = runner._reconcile_external_flat_positions({"positions": {}}, now)
    event = state.events[-1]
    assert reconciled == {symbol}
    assert resolved == {order_id}
    assert event["reason"] == "protective_stop_exchange_algo"
    assert event["closed_at"] == now.isoformat()
    assert event["close_price"] == 95.0
    assert event["fees"] == pytest.approx(0.45)
    assert event["net_pnl"] == pytest.approx(-9.57)
    assert event["pnl_recorded"] is True
