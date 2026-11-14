"""Offline acceptance for the Lead structure pool and the real runner path."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import time

import ccxt
import pandas as pd
import pytest

from bot.config import load_settings
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner


AT = datetime(2026, 11, 14, 5, tzinfo=timezone.utc)


class PublicMarket:
    """Deterministic closed and unfinished bars; no private API methods."""

    def __init__(self, count=25):
        self.at = AT
        self.calls = []
        self.failures = {}
        self.cross = {count - 1}
        self.missing = set()
        self.offset = 0.0
        self.symbols = [f"COIN{i:03}/USDT:USDT" for i in range(count)]
        self.markets = {symbol: {
            "id": symbol.split("/")[0] + "USDT", "active": True, "swap": True,
            "quote": "USDT", "settle": "USDT",
            "info": {"contractType": "PERPETUAL", "underlyingType": "COIN",
                     "underlyingSubType": ["Crypto"], "onboardDate": 1500000000000},
        } for symbol in self.symbols}

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {symbol: {"quoteVolume": 100_000_000 - i * 1_000_000,
                         "last": 97., "timestamp": int(self.at.timestamp() * 1000)}
                for i, symbol in enumerate(self.symbols) if symbol not in self.missing}

    def fetch_ohlcv(self, symbol, timeframe, *, limit):
        self.calls.append((symbol, timeframe, limit))
        if (symbol, timeframe) in self.failures:
            raise self.failures[symbol, timeframe]("offline read failure")
        hours = {"1h": 1, "4h": 4, "1d": 24}[timeframe]
        end = pd.Timestamp(self.at).floor(f"{hours}h")
        rows = []
        for i, stamp in enumerate(pd.date_range(end=end, periods=limit, freq=f"{hours}h")):
            if timeframe == "1d":
                close, opening, width = 100 + i * .1, 100 + i * .1, 1.
            elif timeframe == "4h":
                close, opening, width = 100., 100., 1.
            else:
                close, opening, width = 96.8 + self.offset, 96.8 + self.offset, .4
                if self.symbols.index(symbol) in self.cross:
                    if i == limit - 3:
                        close = 96.6 + self.offset
                    if i == limit - 2:
                        close = 97.0 + self.offset
            rows.append([int(stamp.timestamp() * 1000), opening,
                         max(opening, close) + width, min(opening, close) - width, close, 1000.])
        return rows


@pytest.fixture
def env(monkeypatch):
    market = PublicMarket()
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    settings = load_settings(env_file=None)
    cache = runner.ClosedMarketCache()
    clock = [AT + timedelta(seconds=1)]
    monkeypatch.setattr(runner, "_utc_now", lambda: clock[0])
    ids = {row["id"] for row in market.markets.values()}

    def refresh():
        market.at = clock[0]
        return runner._build_lead_universe(market, state, settings, clock[0],
                                           lead_market_ids=ids, cache=cache)

    return SimpleNamespace(market=market, state=state, settings=settings, cache=cache,
                           clock=clock, ids=ids, refresh=refresh)


def test_real_cycle_covers_candidate_after_old_top20_and_consumes_signal_once(env, monkeypatch, tmp_path):
    e = env
    seen, monitored = [], []
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(runner, "_update_market_risk", lambda *a: MarketRiskAssessment(
        MarketRiskState.NORMAL, "normal", e.clock[0], "BTC/USDT:USDT", "4h", {}))
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *a: True)
    monkeypatch.setattr(runner, "_demo_stop_audit_due", lambda *a: False)
    monkeypatch.setattr(runner, "_submit_entry_candidate", lambda *a, **kw: seen.append(a[5][0]))
    execution = SimpleNamespace(lead_market_ids=e.ids)
    path = tmp_path / "ledger.json"

    def cycle():
        runner._cycle(e.market, execution, e.state, e.settings, e.clock[0], True, 5,
                      closed_market_cache=e.cache, position_monitor=lambda: monitored.append(1),
                      checkpoint=lambda: e.state.save(path))

    cycle()
    assert len(e.state.mr_candidate_pool) == 25
    assert len(e.state.mr_entry_scan["symbols"]) == 25
    first = e.market.symbols[0]
    outcome = e.state.mr_entry_scan["outcomes"][first]
    assert outcome["status"] == "evaluated"
    assert outcome["reason"] == "hourly_candle_direction_mismatch"
    assert outcome["bar"] == (AT - timedelta(hours=1)).isoformat()
    assert seen == [e.market.symbols[-1]]
    assert monitored
    assert len(e.state.portfolio.positions) == 0
    e.state = MultiPaperState.load(path)
    assert e.state.mr_entry_scan["outcomes"][first] == outcome
    e.cache = runner.ClosedMarketCache()  # Restart rechecks every pool member.
    cycle()
    assert seen == [e.market.symbols[-1]]
    assert len(e.state.mr_candidate_pool) == 25


@pytest.mark.parametrize("data_ready,expected", [
    (True, "afternoon_dump_blocks_long"),
    (False, "hourly_history_gap"),
])
def test_lead_scan_keeps_time_filter_rejection(env, monkeypatch, data_ready, expected):
    e = env
    markets, _, one, *_ = e.refresh()
    symbol = e.market.symbols[-1]
    four = {symbol: e.cache.four_hour[symbol][1]}
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(
        False, True, False, False, afternoon_dump_today=True, data_ready=data_ready,
        data_reason=None if data_ready else "hourly_history_gap",
    ))
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})
    runner._evaluate_entries(e.market, e.state, None, True, markets, [symbol], one,
                             four, {}, e.settings, 5, 0, 0, e.clock[0], risk)
    outcome = e.state.mr_entry_scan["outcomes"][symbol]
    assert outcome["status"] == "evaluated"
    assert outcome["phase"] == "time_filter"
    assert outcome["reason"] == expected
    assert "candidate_priority" not in outcome


def test_retains_first_seen_across_hours_and_incrementally_reads_hourly_data(env):
    e = env
    e.refresh()
    symbol = e.market.symbols[0]
    first = e.state.mr_candidate_pool[symbol]["first_seen_at"]
    e.market.calls.clear()
    e.clock[0] += timedelta(hours=1)
    e.refresh()
    row = e.state.mr_candidate_pool[symbol]
    assert row["first_seen_at"] == first and row["pool_age_hours"] == 1
    assert row["bar"] == (AT).isoformat()
    assert e.market.calls == [(s, "1h", 2) for s in e.market.symbols]


def test_read_failure_suspends_and_recovery_preserves_identity(env):
    e = env
    e.refresh()
    symbol = e.market.symbols[0]
    first = e.state.mr_candidate_pool[symbol]["first_seen_at"]
    e.clock[0] += timedelta(hours=1)
    e.market.failures[symbol, "1h"] = ccxt.RequestTimeout
    e.refresh()
    assert e.state.mr_candidate_pool[symbol]["status"] == "pending"
    assert e.state.mr_candidate_pool[e.market.symbols[1]]["status"] == "active"
    e.market.failures.clear()
    e.clock[0] += timedelta(seconds=6)
    e.refresh()
    assert e.state.mr_candidate_pool[symbol]["status"] == "active"
    assert e.state.mr_candidate_pool[symbol]["first_seen_at"] == first


def test_partial_universe_does_not_remove_missing_candidates(env):
    e = env
    e.refresh()
    symbol = e.market.symbols[0]
    e.market.missing.add(symbol)
    e.clock[0] += timedelta(hours=1)
    e.refresh()
    assert e.state.mr_candidate_pool[symbol]["status"] == "pending"
    assert not any(row.get("event") == "mr_candidate_removed" for row in e.state.events)


def test_verified_structure_failure_removes_but_does_not_touch_position_state(env):
    e = env
    e.refresh()
    e.state.exchange_orders_by_symbol["sentinel"] = {"stop": {"id": "protected"}}
    e.market.offset = 3.
    e.clock[0] += timedelta(hours=1)
    e.refresh()
    assert not e.state.mr_candidate_pool
    assert e.state.exchange_orders_by_symbol["sentinel"]["stop"]["id"] == "protected"
    assert sum(row.get("event") == "mr_candidate_removed" for row in e.state.events) == 25


def test_direction_change_and_equal_daily_close_end_old_identity(env):
    e = env
    observations = {"COIN": {"status": "active", "side": "long", "close": 97}}
    runner._update_lead_pool(e.state, observations, AT)
    observations["COIN"].update(side="short", close=103)
    runner._update_lead_pool(e.state, observations, AT + timedelta(hours=1))
    assert e.state.mr_candidate_pool["COIN"]["side"] == "short"
    assert e.state.mr_candidate_pool["COIN"]["first_seen_at"] == (AT + timedelta(hours=1)).isoformat()
    runner._update_lead_pool(e.state, {"COIN": {"status": "removed", "reason": "daily_close_equals_ema50"}},
                             AT + timedelta(hours=2))
    assert not e.state.mr_candidate_pool


def test_old_ledger_does_not_promote_legacy_watchlist_without_structure_verification(env):
    old = env.state.to_dict()
    old.pop("mr_candidate_pool")
    old["watchlist"] = [{"symbol": "OLD", "rank": 1}]
    restored = MultiPaperState.from_dict(old)
    assert restored.mr_candidate_pool == {}
    assert restored.watchlist[0]["symbol"] == "OLD"


@pytest.mark.parametrize("side,close", [(Side.LONG, 97.), (Side.SHORT, 103.)])
def test_structure_uses_entry_formula_without_requiring_hourly_reversal(env, side, close):
    e = env
    rows1 = e.market.fetch_ohlcv(e.market.symbols[0], "1h", limit=60)
    frame1 = runner.ohlcv_frame(runner._closed(rows1, runner.HOUR_MS, int(AT.timestamp() * 1000)))
    frame1.loc[:, ["open", "high", "low", "close"]] = [close, close + 1, close - 1, close]
    rows4 = e.market.fetch_ohlcv(e.market.symbols[0], "4h", limit=100)
    frame4 = runner.ohlcv_frame(runner._closed(rows4, runner.FOUR_H_MS, int(AT.timestamp() * 1000)))
    gate = {"ready": True, "direction": side.value}
    structure = runner._lead_structure(frame1, frame4, gate, e.settings, AT)
    assert structure["side"] == side.value and structure["status"] == "active"
    flags = TimeFilterFlags(True, True, False, False)
    flat = runner.deviation_entry_signal_at_ema(100, close, close, e.settings, flags,
                                               deviation_pct=structure["deviation_threshold_pct"])
    assert flat.action is Action.NONE
    opening = close - .1 if side is Side.LONG else close + .1
    actual = runner.deviation_entry_signal_at_ema(100, opening, close, e.settings, flags,
                                                 deviation_pct=structure["deviation_threshold_pct"])
    assert actual.action is Action.OPEN and actual.side is side
    stale = frame1.iloc[:-1]
    assert runner._lead_structure(stale, frame4, gate, e.settings, AT)["status"] == "pending"


def test_deadline_remains_original_after_slow_refresh(env, monkeypatch):
    e = env
    e.refresh()
    e.clock[0] = AT + timedelta(seconds=61)
    called = []
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: called.append(1))
    result = e.refresh()
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})
    runner._evaluate_entries(e.market, e.state, None, True, result[0], e.market.symbols,
                             result[2], {}, {}, e.settings, 5, 0, 0, e.clock[0], risk)
    assert called == []
    assert e.state.mr_entry_scan["deadline"] == (AT + timedelta(seconds=60)).isoformat()
    assert e.state.mr_entry_scan["status"] == "expired"


def test_old_cross_cannot_be_replayed_in_the_following_hour(env):
    e = env
    e.refresh()
    symbol = e.market.symbols[-1]
    assert runner._ema9_reclaim_side(e.cache.lead_hourly[symbol]) == "long"
    e.clock[0] += timedelta(hours=1)
    e.market.cross.clear()
    e.market.offset = .2
    e.refresh()
    assert runner._ema9_reclaim_side(e.cache.lead_hourly[symbol]) is None


def test_offline_coverage_timing_reports_actual_scope(env):
    start = time.monotonic()
    env.refresh()
    summary = env.state.events[-1]
    assert summary["eligible"] == summary["active"] == 25
    assert summary["data_prepare_seconds"] > 0
    assert summary["data_prepare_seconds"] <= time.monotonic() - start + .001


def test_recovered_symbol_joins_same_bar_without_replaying_other_outcomes(env, monkeypatch):
    e = env
    e.refresh()
    result = e.cache.lead_universe
    symbols = e.market.symbols
    called = []
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: called.append(a[3]))
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})

    def scan(selected):
        runner._evaluate_entries(e.market, e.state, None, True, result[0], selected,
                                 result[2], {}, {}, e.settings, 5, 0, 0, e.clock[0], risk)

    scan(symbols[:-1])
    original_deadline = e.state.mr_entry_scan["deadline"]
    e.clock[0] += timedelta(seconds=5)
    scan(symbols)
    assert called == symbols
    assert e.state.mr_entry_scan["deadline"] == original_deadline
    scan(symbols)
    assert called == symbols


@pytest.mark.parametrize("status", ["submitting", "uncertain", "limit_pending", "protected"])
def test_unresolved_orders_including_partial_limits_prevent_duplicate_entry(env, monkeypatch, status):
    e = env
    e.state.mr_entry_attempts["existing"] = {"status": status, "symbol": "COIN000/USDT:USDT",
                                              "filled": .5, "quantity": 1.0}
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: pytest.fail("must reconcile first"))
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})
    runner._evaluate_entries(e.market, e.state, None, True, {}, ["COIN000/USDT:USDT"],
                             {}, {}, {}, e.settings, 5, 0, 0, e.clock[0], risk)
    assert e.state.mr_entry_scan["status"] == "paused"


def test_fatal_batch_failure_suspends_old_candidates_instead_of_leaving_them_active(env):
    e = env
    e.refresh()
    e.clock[0] += timedelta(hours=1)

    class FailedBatch:
        def read_many(self, *args, **kwargs):
            raise ccxt.RateLimitExceeded("offline ban")

    with pytest.raises(ccxt.RateLimitExceeded):
        runner._build_lead_universe(e.market, e.state, e.settings, e.clock[0],
                                    lead_market_ids=e.ids, cache=e.cache, candle_reader=FailedBatch())
    assert len(e.state.mr_candidate_pool) == 25
    assert all(row["status"] == "pending" for row in e.state.mr_candidate_pool.values())


def test_pool_rejects_daily_gaps_and_explicit_equal_direction(env):
    e = env
    result = e.refresh()
    symbol = e.market.symbols[0]
    one, four = result[2][symbol], e.cache.four_hour[symbol][1]
    invalid = {"ready": True, "direction": "long", "daily_history_valid": False}
    assert runner._lead_structure(one, four, invalid, e.settings, AT)["status"] == "pending"
    equal = {"ready": False, "reason": "daily_close_equals_ema50"}
    assert runner._lead_structure(one, four, equal, e.settings, AT)["status"] == "removed"


def test_pool_does_not_drop_real_qualifying_position_or_generate_a_top_up(env):
    from bot.portfolio import Position
    e = env
    result = e.refresh()
    symbol = e.market.symbols[-1]
    position = Position(symbol, Side.LONG, 1, 97, 95, AT, "existing", "MID")
    e.state.portfolio.positions[symbol] = position
    assert runner._entry_candidate(e.market, e.state, True, symbol, result[2],
                                   {symbol: e.cache.four_hour[symbol][1]}, {},
                                   e.settings, e.clock[0], {}) is None
    assert e.state.portfolio.positions[symbol] is position


def test_lead_batches_read_only_progress_but_checkpoints_before_execution(env, monkeypatch):
    e = env
    saves = []
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: None)
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})
    runner._evaluate_entries(e.market, e.state, None, True, {}, e.market.symbols,
                             {}, {}, {}, e.settings, 5, 0, 0, e.clock[0], risk,
                             checkpoint=lambda: saves.append(e.state.to_dict()))
    assert len(saves) <= 5
    assert len(e.state.mr_entry_scan["outcomes"]) == 25
    assert all(row["status"] == "evaluated" for row in e.state.mr_entry_scan["outcomes"].values())


def test_incremental_window_matches_full_read_when_no_unfinished_candle_exists(env):
    e = env
    original = e.market.fetch_ohlcv

    def closed_only(symbol, timeframe, *, limit):
        rows = original(symbol, timeframe, limit=limit + (timeframe == "1h"))
        if timeframe == "1h":
            cutoff = int(pd.Timestamp(e.market.at).floor("h").timestamp() * 1000)
            return [row for row in rows if row[0] < cutoff][-limit:]
        return rows

    e.market.fetch_ohlcv = closed_only
    e.refresh()
    symbol = e.market.symbols[0]
    assert len(e.cache.lead_hourly[symbol]) == 60
    e.clock[0] += timedelta(hours=1)
    e.refresh()
    expected = runner.ohlcv_frame(closed_only(symbol, "1h", limit=60))
    pd.testing.assert_frame_equal(e.cache.lead_hourly[symbol], expected)


def test_restart_suspension_can_recover_only_unsubmitted_symbols(env, monkeypatch):
    e = env
    symbols = e.market.symbols[:2]
    bar = pd.Timestamp(AT - timedelta(hours=1))
    e.state.mr_entry_scan = {"bar": bar.isoformat(), "deadline": (AT + timedelta(seconds=60)).isoformat(),
                             "symbols": symbols, "status": "completed", "outcomes": {
                                 symbols[0]: {"status": "blocked", "reason": "candidate_structure_unverified", "attempts": 0},
                                 symbols[1]: {"status": "opened", "attempts": 1},
                             }}
    called = []
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: called.append(a[3]))
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", e.clock[0], "BTC", "4h", {})
    runner._evaluate_entries(e.market, e.state, None, True, {}, symbols,
                             {}, {}, {}, e.settings, 5, 0, 0, e.clock[0], risk)
    assert called == [symbols[0]]
    assert e.state.mr_entry_scan["outcomes"][symbols[1]]["status"] == "opened"


def test_invalid_four_hour_cache_can_recover_in_the_same_period(env):
    e = env
    original = e.market.fetch_ohlcv
    symbol = e.market.symbols[0]

    def broken(requested, timeframe, *, limit):
        rows = original(requested, timeframe, limit=limit)
        if requested == symbol and timeframe == "4h":
            for row in rows:
                row[1:5] = [0., 0., 0., 0.]
        return rows

    e.market.fetch_ohlcv = broken
    e.refresh()
    assert symbol not in e.cache.four_hour
    assert symbol not in e.state.mr_candidate_pool
    e.market.fetch_ohlcv = original
    e.clock[0] += timedelta(seconds=6)
    e.refresh()
    assert e.state.mr_candidate_pool[symbol]["status"] == "active"
