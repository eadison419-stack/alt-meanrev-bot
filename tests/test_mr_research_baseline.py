from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.backtest.research_data import (
    aggregate_closed_bars, chronological_splits, fetch_public_rows, load_manifest, profile_bars,
)
from bot.config import load_settings
from bot.entry_quality import execution_quote
from bot.exchange.clock import activate_clock, active_clock
from bot.lead_risk import lead_entry_exposure, settings_for_execution
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.mr_candidate import deviation_entry_signal
from bot.portfolio import Position
from bot.regime import Regime
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner
from scripts.replay_mr_baseline import (
    ReplayClock, ReplayExecution, ReplayPublic, _ClosedRegimes, _check_protective_stops,
    _closed_prefix, _has_raw_signal, _rows, _with_entry_regime,
)


def _frame(index, prices):
    prices = [float(price) for price in prices]
    return pd.DataFrame({"open": prices, "high": [p + 1 for p in prices],
                         "low": [p - 1 for p in prices], "close": prices,
                         "volume": 10.0}, index=index)


@pytest.mark.parametrize("now,count", [("2026-10-01T23:59:59Z", 0), ("2026-10-02T00:00:00Z", 1)])
def test_utc_daily_requires_six_fully_closed_four_hour_candles(now, count):
    times = pd.date_range("2026-10-01", periods=12, freq="4h", tz="UTC")
    frame = _frame(times, list(range(100, 112)))
    result = aggregate_closed_bars(frame, 4, 24, as_of=now)
    assert len(result) == count
    if count:
        assert result.iloc[0].to_dict() == dict(open=100, high=106, low=99, close=105, volume=60)
    with pytest.raises(ValueError, match="continuous"):
        aggregate_closed_bars(frame.drop(times[4]), 4, 24, as_of=now)


def test_funding_pagination_cannot_cross_research_cutoff():
    start = pd.Timestamp("2026-10-01T00:00:00Z")
    end = start + pd.Timedelta(1, unit="D")
    first = int(start.timestamp() * 1000)
    last = int(end.timestamp() * 1000) - 1
    requests = []

    def funding(params):
        requests.append(dict(params))
        return ([{"symbol": "BTCUSDT", "fundingTime": first, "fundingRate": ".0001", "markPrice": "100"}]
                if len(requests) == 1 else [])

    rows = fetch_public_rows(SimpleNamespace(fapiPublicGetFundingRate=funding), "funding", "BTCUSDT", start, end)
    assert len(rows) == 1
    assert [(r["startTime"], r["endTime"]) for r in requests] == [(first, last), (first + 1, last)]


def test_manifest_rejects_tampered_public_file(tmp_path):
    files = [{"path": f"BTCUSDT_{stream}.json.gz", "symbol": "BTCUSDT", "stream": stream}
             for stream in ("contract_1h", "contract_4h", "mark_1h", "funding")]
    files.append({"path": "exchange_info.json", "stream": "exchange_info"})
    from bot.backtest.research_data import digest_file
    for file in files:
        path = tmp_path / file["path"]
        path.write_bytes(b"original")
        file["sha256"] = digest_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"symbols": ["BTCUSDT"], "files": files}), encoding="utf-8")
    load_manifest(tmp_path)
    (tmp_path / files[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_manifest(tmp_path)


def test_split_embargo_and_hourly_prefix_exclude_boundary_future():
    splits = chronological_splits("2026-05-01T00:00:00Z", "2026-11-10T00:00:00Z")
    for a, b in zip(splits, splits[1:]):
        assert pd.Timestamp(b["start"]) - pd.Timestamp(a["end_exclusive"]) == pd.Timedelta(48, unit="h")
        assert not b["genuinely_unseen"]
    index = pd.date_range("2026-05-01", periods=6, freq="h", tz="UTC")
    frame = _frame(index, [100, 101, 102, 103, 900, 999])
    assert _closed_prefix(frame, index[4], pd.Timedelta(1, unit="h"), 59)["close"].tolist() == [100, 101, 102, 103]


def _public(now):
    symbol = "TEST/USDT:USDT"
    one = _frame(pd.date_range(end=now.floor("h"), periods=60, freq="h"), [100] * 59 + [150])
    daily = _frame(pd.date_range(end=now.floor("D") - pd.Timedelta(1, unit="D"), periods=100, freq="D"),
                   list(range(100, 200)))
    funding = pd.DataFrame({"funding_rate": [.0001, -.004], "mark_price": [105, 109]},
                           index=pd.DatetimeIndex([now - pd.Timedelta(1, unit="h"), now + pd.Timedelta(1, unit="h")]))
    frame = {symbol: {"contract_1h": one, "mark_1h": one.copy(), "daily": daily, "funding": funding}}
    filters = [{"filterType": "LOT_SIZE", "minQty": "0.1", "stepSize": "0.1"},
               {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
               {"filterType": "MIN_NOTIONAL", "notional": "5"}]
    return symbol, ReplayPublic(frame, [{"symbol": "TESTUSDT", "filters": filters}], ReplayClock(now))


def test_replay_feeds_production_daily_gate_and_valid_synthetic_quote():
    now = pd.Timestamp("2026-10-01T05:00:01Z")
    symbol, public = _public(now)
    saved = active_clock()
    activate_clock(public.clock)
    try:
        reference = SimpleNamespace(fetch_ohlcv=lambda *args, **kwargs:
            _rows(public.frames[symbol]["daily"].tail(99)))
        expected = runner._daily_direction_gate(reference, symbol, now, 50)
        assert runner._daily_direction_gate(public, symbol, now, 50) == expected
        assert expected["ready"] and expected["direction"] == "long"
        quote, reason = execution_quote(public.fetch_order_book(symbol, limit=5), Side.LONG, now, max_age_sec=15)
        assert reason is None and quote["price"] == pytest.approx(150.01)
        assert public.fetch_funding_rate(symbol) == {"fundingRate": .0001}
        public.clock.window_end = now.floor("h")
        assert public.contract_price(symbol) == 100
    finally:
        activate_clock(saved)


def test_replay_account_uses_mark_to_active_stop_and_actual_funding_sign():
    now = pd.Timestamp("2026-10-01T05:00:01Z")
    symbol, public = _public(now)
    settings = settings_for_execution(load_settings(env_file=None), "lead_usdm")
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    execution = ReplayExecution(public, state, settings, 1000)
    opened = datetime(2026, 10, 1, 3, tzinfo=timezone.utc)
    position = Position(symbol, Side.LONG, 1, 100, 90, opened, "test", Regime.MID.value)
    state.portfolio.positions[symbol] = position
    execution.positions[symbol] = {"side": "long", "quantity": 1, "entry_price": 100}
    stop = {"id": "s", "amount": 1, "stop_price": 95}
    execution.stops[symbol] = stop
    state.exchange_orders_by_symbol[symbol] = {"stop": stop}
    execution.history[(symbol, opened.isoformat())] = {"side": "long", "quantity": 1}
    state.exchange_snapshot = execution.sync_snapshot()
    state.protective_stop_audit = {"status": "verified", "position_count": 1, "mismatch_count": 0}
    exposure = lead_entry_exposure(state, settings, now)
    assert exposure.long_notional == 150
    assert exposure.capital_risk == pytest.approx(5.39)
    assert exposure.giveback_risk == pytest.approx(50.0)
    assert exposure.weighted_giveback_risk == pytest.approx(7.5)
    assert exposure.stop_risk == pytest.approx(12.89)
    assert execution.funding_fee(symbol, opened, now) == pytest.approx(-.0105)
    assert profile_bars(public.frames[symbol]["contract_1h"], 3600000)["valid"]


@pytest.mark.parametrize("open_price,close", [(96, 98), (104, 102), (99, 100), (98, 97)])
def test_fast_raw_signal_screen_keeps_runner_direction_and_deviation(open_price, close):
    settings = load_settings(env_file=None)
    four = pd.Series([100.0] * 99)
    candle = {"open": open_price, "close": close}
    expected = deviation_entry_signal(four, candle, settings,
        TimeFilterFlags(True, True, False, False), deviation_pct=runner.DEV_PCT)
    assert _has_raw_signal(candle, 100, settings) is (expected.action is Action.OPEN)


def test_replay_open_gap_exits_at_adverse_open_before_stop_can_move():
    now = pd.Timestamp("2026-10-01T05:00:01Z")
    symbol, public = _public(now)
    public.frames[symbol]["contract_1h"].loc[now.floor("h"), "open"] = 90
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    execution = ReplayExecution(public, state, load_settings(env_file=None), 1000)
    opened = datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc)
    position = Position(symbol, Side.LONG, 1, 100, 95, opened, "test", Regime.MID.value)
    state.portfolio.positions[symbol] = position
    state.entry_costs[symbol] = {"entry_fee": .05}
    execution.positions[symbol] = {"side": "long", "quantity": 1, "entry_price": 100}
    stop = {"id": "s", "amount": 1, "stop_price": 95}
    execution.stops[symbol] = stop
    state.exchange_orders_by_symbol[symbol] = {"stop": stop}
    execution.history[(symbol, opened.isoformat())] = {"side": "long", "quantity": 1}
    _check_protective_stops(state, execution, public, {}, public.clock, .0005, .0003, open_only=True)
    assert not state.portfolio.positions and not execution.stops
    event = state.events[-1]
    assert event["reason"] == "protective_stop_exchange_algo"
    assert event["price"] == pytest.approx(90 * (1 - .0003))


def test_same_input_candidate_regime_stop_quantity_and_fill_reconciliation(monkeypatch):
    now = pd.Timestamp("2026-10-01T00:00:01Z")
    symbol, public = _public(now)
    hourly = public.frames[symbol]["contract_1h"]
    hourly.loc[hourly.index[-2], ["open", "high", "low", "close"]] = [97.5, 98.5, 97, 98]
    hourly.loc[hourly.index[-1], ["open", "high", "low", "close"]] = [98.05, 99, 97.9, 98.2]
    four_frame = _frame(pd.date_range(end="2026-09-30T20:00:00Z", periods=99, freq="4h"), [100] * 99)
    one, four = {symbol: hourly.iloc[:-1]}, {symbol: four_frame}
    settings = settings_for_execution(load_settings(env_file=None), "lead_usdm")
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    execution = ReplayExecution(public, state, settings, 1000)
    monkeypatch.setattr(runner, "_ema9_reclaim_side", lambda _frame: "long")
    saved_clock = active_clock()
    activate_clock(public.clock)
    try:
        regimes = _ClosedRegimes(four, settings)
        direct = runner._entry_candidate(public, state, True, symbol, one, four, regimes, settings, now, {})
        deferred = runner._entry_candidate(public, state, True, symbol, one, four, {}, settings, now, {})
        assert direct is not None and _with_entry_regime(deferred, regimes) == direct
        assert direct[2] is Side.LONG
        state.exchange_snapshot = execution.sync_snapshot()
        assert runner._audit_demo_protective_stops(execution, state, now)
        market_risk = MarketRiskAssessment(MarketRiskState.NORMAL, "fixture", now, "BTC/USDT:USDT", "4h", {})
        position = runner._submit_entry_candidate(public, state, execution, True, public.markets,
            (*direct, None), settings, 5, .0005, .0003, now, market_risk)
        assert position is not None and position.quantity == 6.1
        audit = next(event for event in state.events if event.get("event") == "entry_preflight_decision")
        assert audit["planned_quantity"] == position.quantity and audit["planned_stop"] == position.stop_price
        assert audit["planned_price"] == pytest.approx(98.06)
        assert position.stop_price < position.entry_price
        public.clock.at = now + pd.Timedelta(3, unit="min")
        execution.exit_override[symbol] = 101
        runner._close_position(True, execution, state, symbol, 101, "test_close", public.clock.now(), .0005, .0003)
        assert state.events[-1]["net_pnl"] == pytest.approx(execution.wallet - 1000)
    finally:
        activate_clock(saved_clock)
