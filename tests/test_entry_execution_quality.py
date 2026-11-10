from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.config import load_settings
from bot.entry_quality import EntrySignalContext, entry_liquidity_observation, execution_quote
from bot.exchange.binance_usdm import FAPI_URLS
from bot.execution import DemoExecutionService, EntryPreflightRejected
from bot.live.multi_paper import MultiPaperState
from bot.portfolio import Position
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.regime import Regime
from bot.risk import OrderIntent, calculate_order_quantity
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 3, 5, 0, 30, tzinfo=timezone.utc)
SYMBOL = "TEST/USDT:USDT"


def _book(at=NOW, *, bid=98.1, ask=98.2):
    return {"timestamp": int(at.timestamp() * 1000),
            "bids": [[bid, 1000.0]], "asks": [[ask, 1000.0]]}


def _guard(monkeypatch, *, side=Side.LONG, book=None):
    settings = load_settings(env_file=None)
    clock = [NOW]
    monkeypatch.setattr(runner, "_utc_now", lambda: clock[0])
    state = MultiPaperState(initial_equity=5000, cash=5000)
    context = EntrySignalContext(
        SYMBOL, side, NOW.replace(second=0), NOW-timedelta(seconds=5), NOW,
        97.5 if side is Side.LONG else 102.5,
        98.0 if side is Side.LONG else 102.0,
        100.0, 1.0, 97.0, 103.0,
    )
    planned_stop = 96.75 if side is Side.LONG else 103.25
    intent = OrderIntent(SYMBOL, side, Action.OPEN, 1.0, context.signal_close,
                         "range_deviation_"+side.value, Regime.MID, planned_stop, "MEAN_REVERSION")

    class QuoteClient:
        calls = 0

        @staticmethod
        def price_to_precision(symbol, price):
            return str(price)

        def fetch_order_book(self, symbol, *, limit):
            assert symbol == SYMBOL and limit == 5
            self.calls += 1
            if book is not None:
                return book
            return _book(clock[0], bid=98.1 if side is Side.LONG else 101.8,
                         ask=98.2 if side is Side.LONG else 101.9)

    client = QuoteClient()
    execution = SimpleNamespace(exchange=client)
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW, "BTC/USDT:USDT", "4h", {})
    audit = {}
    market = {"precision": {"amount": 0.1}, "limits": {"cost": {"min": 5.0}}}
    guard, deadline = runner._entry_preflight(
        client, execution, state, settings, market, context,
        TimeFilterFlags(True, True, False, False), 0.0, risk, audit,
    )
    return guard, deadline, audit, intent, state, settings, clock, client


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_fresh_executable_quote_rechecks_size_and_only_observes_atr_caps(monkeypatch, side):
    guard, _, audit, intent, state, settings, _, _ = _guard(monkeypatch, side=side)
    refreshed, stop = guard(intent, intent.stop_price)

    assert refreshed.price == pytest.approx(98.2 if side is Side.LONG else 101.8)
    assert refreshed.stop_price == stop
    assert refreshed.quantity == calculate_order_quantity(
        state.current_equity, refreshed.price, state.portfolio, settings,
        step_size=0.1, minimum_quantity=5.0/refreshed.price, size_multiplier=1.0,
        stop_price=stop, gross_notional_cap_pct=settings.sizing.max_gross_notional_pct,
    )
    assert refreshed.quantity != intent.quantity
    assert audit["decision"] == "ALLOW"
    assert audit["adverse_quote_atr"] == pytest.approx(0.2)
    assert all(item["would_block"] for item in audit["atr_threshold_observations"])
    assert audit["observe_only"] is True
    assert "entry_liquidity" not in audit


def test_lead_quote_guard_refreshes_account_and_uses_active_stops(monkeypatch):
    guard, _, audit, intent, state, settings, clock, client = _guard(monkeypatch)
    state.execution_mode = "lead_usdm"
    state.execution_status = "synchronized"
    state.exchange_snapshot = {"at": NOW.isoformat(), "equity": 1000, "positions": {
        "OLD": {"side": "long", "quantity": 1, "mark_price": 120}}}
    state.portfolio.positions["OLD"] = Position("OLD", Side.LONG, 1, 100, 90, NOW, "old", "MID")
    state.exchange_orders_by_symbol["OLD"] = {"stop": {"id": "s", "amount": 1, "stop_price": 90.2}}
    state.protective_stop_audit = {"status": "verified", "mismatch_count": 0, "position_count": 1}
    calls = []
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *args: calls.append("account") or True)
    monkeypatch.setattr(runner, "_audit_demo_protective_stops", lambda *args: calls.append("stops") or True)
    refreshed, stop = guard(intent, intent.stop_price)
    assert calls == ["account", "stops"]
    assert audit["existing_portfolio_capital_risk"] == pytest.approx(10.1804)
    assert audit["existing_portfolio_giveback_risk"] == pytest.approx(20.0)
    assert audit["existing_portfolio_weighted_giveback_risk"] == pytest.approx(3.0)
    assert audit["existing_portfolio_stop_risk"] == pytest.approx(13.1804)
    assert audit["existing_portfolio_stop_risk"] + audit["new_planned_risk"] <= 40
    assert refreshed.notional <= 600 and stop == 96.6
    assert state.portfolio.positions["OLD"].stop_price == 90


def test_final_preflight_rejection_keeps_risk_budgets_and_client_identity(monkeypatch):
    guard, _, audit, intent, state, _, _, _ = _guard(monkeypatch)
    state.execution_mode = "lead_usdm"
    state.execution_status = "synchronized"
    state.exchange_snapshot = {"at": NOW.isoformat(), "equity": 1000, "positions": {
        "OLD": {"side": "long", "quantity": 1, "mark_price": 400}}}
    state.portfolio.positions["OLD"] = Position("OLD", Side.LONG, 1, 100, 90, NOW, "old", "MID")
    state.exchange_orders_by_symbol["OLD"] = {"stop": {"id": "s", "amount": 1, "stop_price": 95}}
    state.protective_stop_audit = {"status": "verified", "mismatch_count": 0, "position_count": 1}
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *args: True)
    monkeypatch.setattr(runner, "_audit_demo_protective_stops", lambda *args: True)
    audit["clientOrderId"] = "mrb_same_attempt"
    with pytest.raises(EntryPreflightRejected, match="portfolio_stop_risk_limit"):
        guard(intent, intent.stop_price)
    event = state.events[-1]
    assert event["event"] == "entry_preflight_decision" and event["decision"] == "BLOCK"
    assert event["clientOrderId"] == "mrb_same_attempt"
    assert event["planned_quantity"] == 0 and event["planned_stop"] > 0
    assert event["quantity_risk"]["existing_portfolio_stop_risk"] > 40
    assert event["quantity_risk"]["max_portfolio_stop_risk"] == 40


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_liquidity_observation_walks_both_sides_and_floors_suggested_size(side):
    book = {
        "asks": [[100, 2], [100.1, 4], [100.2, 100]],
        "bids": [[99.9, 3], [99.85, 2], [99.7, 100]],
    }
    result = entry_liquidity_observation(book, side, 5, slippage_limit=0.0015,
                                         step_size=1, minimum_quantity=1)

    assert result["status"] == "estimated"
    ask, bid = (result["entry"], result["reverse"]) if side is Side.LONG else (result["reverse"], result["entry"])
    assert ask["planned_vwap"] == pytest.approx(100.06)
    assert bid["planned_vwap"] == pytest.approx(99.88)
    assert ask["planned_slippage_pct"] == pytest.approx(0.0006)
    assert bid["planned_slippage_pct"] == pytest.approx((99.9 - 99.88) / 99.9)
    assert ask["quantity_within_limit"] == 6
    assert bid["quantity_within_limit"] == 5
    assert result["spread_pct"] == pytest.approx(0.1 / 99.95)
    assert result["suggested_quantity"] == 2 and result["suggested_action"] == "reduce"


@pytest.mark.parametrize("depth", ["too_thin", "invalid_tail", "unexpected_error"])
def test_lead_liquidity_observation_cannot_change_order_intent(monkeypatch, depth):
    book = _book()
    book["asks"] = [[98.2, 0.15]]
    book["bids"] = [[98.1, 0.15]]
    if depth == "invalid_tail":
        book["bids"].append([98.11, 0.1])
    if depth == "unexpected_error":
        def broken_observation(*args, **kwargs):
            raise RuntimeError("diagnostic failed")
        monkeypatch.setattr(runner, "entry_liquidity_observation", broken_observation)
    guard, _, audit, intent, state, _, _, client = _guard(monkeypatch, book=book)
    state.execution_mode = "lead_usdm"
    state.execution_status = "synchronized"
    state.exchange_snapshot = {"at": NOW.isoformat(), "equity": 1000, "positions": {}}
    state.protective_stop_audit = {"status": "verified", "mismatch_count": 0, "position_count": 0}
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *args: True)
    monkeypatch.setattr(runner, "_audit_demo_protective_stops", lambda *args: True)

    refreshed, stop = guard(intent, intent.stop_price)

    assert client.calls == 2  # The same public and execution quote requests.
    assert audit["decision"] == "ALLOW" and refreshed.quantity == 6.1 and stop == 96.6
    observation = audit["entry_liquidity"]
    assert observation["observe_only"] is True
    if depth == "too_thin":
        assert observation["entry"]["planned_vwap"] is None
        assert observation["suggested_quantity"] == 0 and observation["suggested_action"] == "skip"
        position = Position(SYMBOL, Side.LONG, refreshed.quantity, 98.3, stop, NOW,
                            intent.reason, Regime.MID.value, "MEAN_REVERSION")
        runner._record_entry_execution_fill(state, position, {"fill": {}}, audit)
        result = MultiPaperState.from_dict(state.to_dict()).events[-1]["entry_liquidity"]
        assert result["actual_slippage_pct"] == pytest.approx(98.3 / 98.2 - 1)
        assert result["suggested_action"] == "skip"
    else:
        assert observation["status"] == "unavailable"


@pytest.mark.parametrize("age,reason", [(15.001, "entry_quote_stale"), (-0.001, "entry_quote_from_future")])
def test_quote_age_rejects_stale_and_future_timestamps(age, reason):
    quote, error = execution_quote(_book(NOW-timedelta(seconds=age)), Side.LONG,
                                   NOW, max_age_sec=15)
    assert quote is None and error == reason


@pytest.mark.parametrize("book", [
    {}, {"timestamp": 1, "bids": [], "asks": []},
    _book(bid=float("nan")), _book(ask=float("inf")), _book(bid=99, ask=98),
])
def test_incomplete_or_invalid_order_books_fail_closed(book):
    assert execution_quote(book, Side.LONG, NOW, max_age_sec=15) == (None, "entry_quote_invalid")


def test_quote_age_and_signal_deadline_include_the_exact_boundary(monkeypatch):
    assert execution_quote(_book(NOW-timedelta(seconds=15)), Side.LONG,
                           NOW, max_age_sec=15)[1] is None
    guard, deadline, audit, intent, _, _, clock, _ = _guard(monkeypatch)
    clock[0] = deadline
    guard(intent, intent.stop_price)
    assert audit["decision"] == "ALLOW"


def test_quantity_uses_exchange_rounded_stop_price(monkeypatch):
    guard, _, audit, intent, state, settings, _, client = _guard(monkeypatch)
    settings.risk.max_single_trade_risk_pct = 0.001
    monkeypatch.setattr(client, "price_to_precision", lambda symbol, price: "96.5")
    refreshed, stop = guard(intent, intent.stop_price)
    assert stop == 96.5 and refreshed.stop_price == 96.5
    assert refreshed.quantity == calculate_order_quantity(
        state.current_equity, refreshed.price, state.portfolio, settings,
        step_size=0.1, minimum_quantity=5/refreshed.price, size_multiplier=1,
        stop_price=96.5, gross_notional_cap_pct=settings.sizing.max_gross_notional_pct,
    )
    assert audit["planned_stop"] == 96.5


def test_signal_expired_during_preparation_never_fetches_quotes(monkeypatch):
    guard, deadline, audit, intent, _, _, clock, client = _guard(monkeypatch)
    clock[0] = deadline + timedelta(microseconds=1)
    with pytest.raises(EntryPreflightRejected, match="entry_signal_expired"):
        guard(intent, intent.stop_price)
    assert client.calls == 0
    assert audit["decision"] == "BLOCK"


@pytest.mark.parametrize("bid,ask,stop,reason", [
    (99.1, 99.2, 96.75, "entry_signal_no_longer_valid"),
    (96.5, 96.6, 96.75, "entry_structure_invalid"),
    (96.7, 96.8, 96.9, "entry_price_beyond_planned_stop"),
])
def test_quote_must_preserve_signal_structure_and_original_stop(monkeypatch, bid, ask, stop, reason):
    guard, _, audit, intent, state, _, _, _ = _guard(monkeypatch, book=_book(bid=bid, ask=ask))
    with pytest.raises(EntryPreflightRejected, match=reason):
        guard(intent, stop)
    assert not state.portfolio.positions
    assert audit["reason"] == reason


def test_final_executor_check_rejects_expiry_after_guard_and_rounding(monkeypatch):
    from bot import execution as execution_module
    guard, deadline, audit, intent, state, settings, clock, _ = _guard(monkeypatch)

    class Client:
        urls = {"api": dict(FAPI_URLS)}
        orders = []

        def amount_to_precision(self, symbol, amount):
            clock[0] = deadline + timedelta(seconds=1)
            return str(amount)

        def price_to_precision(self, symbol, price):
            return str(price)

    client = Client()
    service = DemoExecutionService(client, settings)
    monkeypatch.setattr(execution_module, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_position", lambda symbol: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda symbol: None)
    monkeypatch.setattr(service, "_market_rules", lambda symbol: (0.1, 0.1, 5.0))
    monkeypatch.setattr(service, "_market_fill", lambda *args, **kwargs: client.orders.append(args))
    result = runner._open_position(
        True, service, state, SYMBOL, Side.LONG, intent.price, intent.stop_price, 1, 3,
        0, 0, intent.reason, Regime.MID.value, NOW,
        entry_preflight=guard, submission_deadline=deadline, entry_audit=audit,
    )
    assert result is None and client.orders == []
    assert audit["decision"] == "BLOCK" and audit["reason"] == "entry_signal_expired"


def test_executor_invokes_guard_after_account_preparation(monkeypatch):
    settings = load_settings(env_file=None)
    client = SimpleNamespace(urls={"api": dict(FAPI_URLS)})
    service = DemoExecutionService(client, settings)
    calls = []
    monkeypatch.setattr(service, "_ensure_one_way", lambda: calls.append("mode"))
    monkeypatch.setattr(service, "_position", lambda symbol: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda symbol: calls.append("prepare"))

    def guard(intent, stop):
        calls.append("guard")
        raise EntryPreflightRejected("blocked")

    intent = OrderIntent(SYMBOL, Side.LONG, Action.OPEN, 1, 98, "test", Regime.MID, 97)
    with pytest.raises(EntryPreflightRejected):
        service.open(intent, stop_price=97, before_submit=guard)
    assert calls == ["mode", "prepare", "guard"]


def test_demo_poll_wakes_at_the_hour_without_changing_local_poll():
    late = NOW.replace(minute=59, second=50)
    assert runner._next_poll_delay(late, 60, align_hourly=True) == 10
    assert runner._next_poll_delay(NOW, 60, align_hourly=True) == 60
    assert runner._next_poll_delay(late, 60) == 60
    assert runner._next_poll_delay(late+timedelta(seconds=20), 60, align_hourly=True,
                                    cycle_started_at=late) == 0


def test_four_hour_cache_matches_uncached_frames_and_expires_at_close():
    clock = [NOW]

    class Client:
        calls = 0

        def fetch_ohlcv(self, symbol, timeframe, *, limit):
            assert timeframe == "4h" and limit == 100
            self.calls += 1
            current = clock[0].replace(hour=(clock[0].hour//4)*4, minute=0, second=0, microsecond=0)
            return [[int((current-timedelta(hours=4*(99-i))).timestamp()*1000), 100, 101, 99, 100+i/100, 1]
                    for i in range(100)]

    client, cache = Client(), runner.ClosedMarketCache()
    now_ms = int(NOW.timestamp()*1000)
    first = runner._cached_four_hour(client, SYMBOL, now_ms, cache)
    cached = runner._cached_four_hour(client, SYMBOL, now_ms+3600000, cache)
    assert first is cached and client.calls == 1
    reference = runner.ohlcv_frame(runner._closed(client.fetch_ohlcv(SYMBOL, "4h", limit=100), runner.FOUR_H_MS, now_ms))
    pd.testing.assert_frame_equal(cached, reference)
    clock[0] = NOW.replace(hour=8, minute=0, second=0)
    fresh = runner._cached_four_hour(client, SYMBOL, int(clock[0].timestamp()*1000), cache)
    assert fresh.index[-1] == pd.Timestamp(clock[0]-timedelta(hours=4))
    assert client.calls == 3


def test_four_hour_boundary_never_falls_back_to_previous_cached_period():
    cache = runner.ClosedMarketCache()

    class Client:
        def fetch_ohlcv(self, *args, **kwargs):
            return [[int(NOW.replace(hour=0, minute=0, second=0).timestamp()*1000), 100, 101, 99, 100, 1]]

    assert runner._cached_four_hour(Client(), SYMBOL, int(NOW.timestamp()*1000), cache) is not None
    boundary = NOW.replace(hour=8, minute=0, second=0)
    assert runner._cached_four_hour(Client(), SYMBOL, int(boundary.timestamp()*1000), cache) is None


def test_daily_cache_uses_closed_utc_period_and_invalidates_at_midnight():
    clock = [NOW]

    class Client:
        calls = 0

        def fetch_ohlcv(self, symbol, timeframe, *, limit):
            assert timeframe == "1d"
            self.calls += 1
            current = clock[0].replace(hour=0, minute=0, second=0, microsecond=0)
            return [[int((current-timedelta(days=99-i)).timestamp()*1000), 100+i, 100+i, 100+i, 100+i, 1]
                    for i in range(100)]

    client, cache = Client(), runner.ClosedMarketCache()
    first = runner._cached_daily_direction_gate(client, SYMBOL, NOW, 50, cache)
    second = runner._cached_daily_direction_gate(client, SYMBOL, NOW+timedelta(hours=1), 50, cache)
    assert first["ready"] and second["daily_ema50"] == first["daily_ema50"] and client.calls == 1
    clock[0] += timedelta(days=1)
    result = runner._cached_daily_direction_gate(client, SYMBOL, clock[0], 50, cache)
    assert client.calls == 2
    assert result["daily_bar_close_time"] == clock[0].replace(hour=0, minute=0, second=0).isoformat()


@pytest.mark.parametrize("network_error", [False, True])
def test_daily_cache_never_reuses_old_ready_result_at_a_new_utc_day(network_error):
    class Client:
        unavailable = False

        def fetch_ohlcv(self, symbol, timeframe, *, limit):
            if self.unavailable and network_error:
                raise runner.ccxt.NetworkError("offline")
            current = NOW.replace(hour=0, minute=0, second=0)
            rows = [[int((current-timedelta(days=99-i)).timestamp()*1000), 100+i, 100+i, 100+i, 100+i, 1]
                    for i in range(100)]
            # A missing new daily candle excludes the previous in-progress row;
            # leaving it here would supply the new day's valid completed bar.
            return rows[:-1] if self.unavailable else rows

    client, cache = Client(), runner.ClosedMarketCache()
    assert runner._cached_daily_direction_gate(client, SYMBOL, NOW, 50, cache)["ready"]
    client.unavailable = True
    result = runner._cached_daily_direction_gate(client, SYMBOL, NOW+timedelta(days=1), 50, cache)
    assert result["ready"] is False
    assert result["reason"] == ("daily_data_unavailable" if network_error else "daily_data_stale")


def test_fill_quality_components_reconcile_and_survive_state_roundtrip(monkeypatch):
    guard, _, audit, intent, state, _, _, _ = _guard(monkeypatch)
    guard(intent, intent.stop_price)
    from bot.portfolio import Position
    position = Position(SYMBOL, Side.LONG, 1, 98.3, 96.75, NOW, intent.reason,
                        Regime.MID.value, "MEAN_REVERSION")
    runner._record_entry_execution_fill(state, position,
                                        {"order_submitted_at": NOW.isoformat(), "fill": {}}, audit)
    event = state.events[-1]
    assert event["signal_to_fill_atr"] == pytest.approx(
        event["signal_to_public_quote_atr"] + event["public_to_demo_quote_atr"] + event["demo_quote_to_fill_atr"])
    restored = MultiPaperState.from_dict(state.to_dict())
    assert restored.events[-1] == event


def _entry_frames():
    index = pd.date_range(end=NOW.replace(minute=0, second=0)-timedelta(hours=1),
                          periods=60, freq="h", tz="UTC")
    one = pd.DataFrame({"open": 97.5, "high": 99., "low": 97., "close": 98., "volume": 1.}, index=index)
    four_index = pd.date_range(end=NOW.replace(hour=0, minute=0, second=0), periods=100, freq="4h", tz="UTC")
    four = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.}, index=four_index)
    return one, four


def test_rejected_first_candidate_does_not_consume_slot_and_fill_has_quote_audit(monkeypatch, tmp_path):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5000, cash=5000)
    state.execution_status = "synchronized"
    one, four = _entry_frames()
    first, second = "FIRST/USDT:USDT", "SECOND/USDT:USDT"
    monkeypatch.setattr(runner, "_utc_now", lambda: NOW)
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *args, **kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(runner, "_cached_daily_direction_gate", lambda *args, **kwargs: {"ready": True, "direction": "long"})

    class Client:
        @staticmethod
        def price_to_precision(symbol, price):
            return str(price)

        def fetch_order_book(self, symbol, *, limit):
            return {} if symbol == first else _book()

        def fetch_funding_rate(self, symbol):
            return {"fundingRate": 0.0}

    class Execution:
        exchange = Client()

        def open(self, intent, *, stop_price, before_submit, submission_deadline,
                 client_order_id, on_submit):
            refreshed, stop = before_submit(intent, stop_price)
            on_submit(refreshed.quantity, stop, NOW)
            return {"fill": {"id": "fill-2", "closed_at": NOW.isoformat(), "fee_cost": 0.1},
                    "position": {"quantity": refreshed.quantity, "entry_price": refreshed.price},
                    "stop": {"stop_price": stop}, "order_submitted_at": NOW.isoformat()}

    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW, "BTC/USDT:USDT", "4h", {})
    runner._evaluate_entries(
        Client(), state, Execution(), True, {}, [first, second],
        {first: one, second: one}, {first: four, second: four}, {}, settings,
        1, 0.0005, 0.0003, NOW, risk, checkpoint=lambda: state.save(tmp_path / "state.json"),
    )
    assert list(state.portfolio.positions) == [second]
    assert state.symbol_status[first]["reason"] == "entry_quote_invalid"
    event = next(e for e in state.events if e["event"] == "entry_execution_quality")
    assert event["symbol"] == second and event["orderId"] == "fill-2"
    assert event["fill_time_verified"] is True
    pos = state.portfolio.positions[second]
    assert state.exit_management_by_symbol[second]["initial_risk"] == pytest.approx(pos.entry_price-pos.stop_price)


def test_executor_uses_revised_quantity_price_and_stop_before_protecting_fill(monkeypatch):
    from bot import execution as execution_module
    guard, deadline, audit, intent, _, settings, clock, _ = _guard(monkeypatch)

    class Client:
        urls = {"api": dict(FAPI_URLS)}

        def amount_to_precision(self, symbol, amount):
            return str(amount)

        def price_to_precision(self, symbol, price):
            return str(price)

    service = DemoExecutionService(Client(), settings)
    calls = []
    position = {}
    monkeypatch.setattr(execution_module, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_position", lambda symbol: position or None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda symbol: None)
    monkeypatch.setattr(service, "_market_rules", lambda symbol: (0.1, 0.1, 5.0))

    def fill(symbol, side, amount, *, reduce_only):
        assert not reduce_only
        calls.append((side, amount))
        position.update(side="long", quantity=amount, entry_price=98.2)
        return {"id": "entry", "filled": amount}

    def protect(symbol, side, amount, stop):
        assert position and amount == position["quantity"]
        return {"stop_price": stop, "amount": amount}

    monkeypatch.setattr(service, "_market_fill", fill)
    monkeypatch.setattr(service, "_place_stop", protect)
    result = service.open(intent, stop_price=intent.stop_price, before_submit=guard,
                          submission_deadline=deadline)
    assert calls == [("buy", audit["planned_quantity"])]
    assert result["stop"]["stop_price"] == audit["planned_stop"]
    assert result["order_submitted_at"] == NOW.isoformat()


def test_demo_cycle_evaluates_the_same_closed_hour_only_once(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState(initial_equity=5000, cash=5000)
    state.execution_status = "synchronized"
    one, four = _entry_frames()
    calls = []
    monkeypatch.setattr(runner, "_utc_now", lambda: NOW)
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW, "BTC/USDT:USDT", "4h", {})
    monkeypatch.setattr(runner, "_update_market_risk", lambda *args: risk)
    monkeypatch.setattr(runner, "_build_universe", lambda *args, **kwargs: ({}, [{"symbol": SYMBOL}], {SYMBOL: one}, {}, {}))
    monkeypatch.setattr(runner, "_cached_four_hour", lambda *args: four)
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *args: True)
    monkeypatch.setattr(runner, "_demo_stop_audit_due", lambda *args: False)
    monkeypatch.setattr(runner, "_evaluate_exits", lambda *args: None)
    monkeypatch.setattr(runner, "_entry_candidate", lambda *args, **kwargs: None)
    evaluate = runner._evaluate_entries

    def evaluate_entries(*args, **kwargs):
        calls.append("entry")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "_evaluate_entries", evaluate_entries)
    monkeypatch.setattr(runner, "_cached_daily_direction_gate", lambda *args, **kwargs: {"ready": False})
    cache = runner.ClosedMarketCache()
    runner._cycle(None, None, state, settings, NOW, True, 1, closed_market_cache=cache)
    runner._cycle(None, None, state, settings, NOW+timedelta(seconds=5), True, 1, closed_market_cache=cache)
    assert calls == ["entry"]


@pytest.mark.parametrize("field,value", [
    ("entry_quote_max_age_sec", 0),
    ("entry_adverse_atr_observe", []),
    ("entry_adverse_atr_observe", [float("inf")]),
])
def test_entry_quality_settings_reject_invalid_limits(field, value):
    settings = load_settings(env_file=None)
    values = settings.execution.model_dump()
    values[field] = value
    with pytest.raises(ValueError):
        type(settings.execution)(**values)
