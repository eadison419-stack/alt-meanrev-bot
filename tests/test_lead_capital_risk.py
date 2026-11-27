from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bot.config import load_settings
from bot.exchange.binance_usdm import LIVE_FAPI_URLS, LIVE_SAPI_URL
from bot.execution import EntryPreflightRejected, LeadPortfolioExecutionService
from bot.entry_quality import EntrySignalContext
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.lead_risk import (lead_entry_exposure, liquidation_is_safe, liquidation_price,
                           parse_brackets, select_leverage, settings_for_execution)
from bot.live.multi_paper import MultiPaperState
from bot.portfolio import PortfolioState, Position
from bot.regime import Regime
from bot.risk import (EntryExposure, OrderIntent, assess_entry, calculate_adaptive_stop,
                      calculate_order_quantity)
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
SYMBOL = "TEST/USDT:USDT"


@pytest.fixture
def settings():
    return settings_for_execution(load_settings(env_file=None), "lead_usdm")


def _brackets(cap=10000, leverage=20):
    return {"symbol": "TESTUSDT", "brackets": [{"notionalFloor": 0, "notionalCap": cap,
            "initialLeverage": leverage, "maintMarginRatio": 0.0065, "cum": 0}]}


def _intent(side=Side.LONG, stop=None):
    return OrderIntent(SYMBOL, side, Action.OPEN, 6, 100, "test", Regime.MID,
                       stop or (98 if side is Side.LONG else 102), "MEAN_REVERSION")


def _state():
    return MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized",
        exchange_snapshot={"at": NOW.isoformat(), "equity": 1000, "positions": {}},
        protective_stop_audit={"status": "verified", "mismatch_count": 0, "position_count": 0})


def _size(settings, *, exposure=EntryExposure(0, 0, 0), stop=98, side=Side.LONG):
    return calculate_order_quantity(1000, 100, PortfolioState(), settings,
        step_size=0.1, minimum_quantity=0.1, stop_price=stop, side=side,
        target_notional_usdt=600, exposure=exposure, portfolio_risk_cap_pct=0.04)


def test_demo_settings_and_quantity_unchanged_while_lead_ignores_margin_and_leverage(settings):
    demo = load_settings(env_file=None)
    assert settings_for_execution(demo, "demo_usdm") is demo
    assert demo.sizing.leverage == 3 and demo.risk.daily_drawdown_halt_pct == 0.04
    assert settings.risk.daily_drawdown_halt_pct == 0.10
    assert calculate_order_quantity(1000, 100, PortfolioState(), demo, step_size=0.1,
        stop_price=98, target_margin_usdt=134.2) == 4.0
    assert _size(settings) == 6.0
    changed_margin = settings.model_copy(update={"sizing": settings.sizing.model_copy(
        update={"margin_per_symbol_usdt": 1, "live_margin_equity_pct": 0.01, "leverage": 1})})
    assert _size(changed_margin) == 6.0


@pytest.mark.parametrize("exposure,stop,quantity", [
    (EntryExposure(0, 0, 0), 92, 2.3),
    (EntryExposure(850, 0, 0), 98, 0.5),
    (EntryExposure(875, 900, 0), 98, 0.2),
    (EntryExposure(0, 0, 30), 98, 4.1),
    (EntryExposure(0, 0, 40), 98, 0),
    (EntryExposure(0, 0, 41), 98, 0),
])
def test_quantity_clips_all_remaining_budgets_without_changing_stop(settings, exposure, stop, quantity):
    assert _size(settings, exposure=exposure, stop=stop) == quantity
    if quantity:
        decision = assess_entry(replace(_intent(), quantity=quantity, stop_price=stop),
            PortfolioState(), 1000, 0, NOW, settings, exposure=exposure,
            portfolio_risk_cap_pct=0.04, per_symbol_notional_usdt=600)
        assert decision.allowed, decision.reason
        assert decision.intent.stop_price == stop


def test_risk_recheck_rejects_excess_combined_loss(settings):
    result = assess_entry(_intent(), PortfolioState(), 1000, 0, NOW, settings,
        exposure=EntryExposure(0, 0, 30), portfolio_risk_cap_pct=0.04,
        per_symbol_notional_usdt=600)
    assert result.reason == "portfolio_stop_risk_cap"


@pytest.mark.parametrize("exposure,step,minimum,reason", [
    (EntryExposure(0, 0, 55), .1, .1, "portfolio_stop_risk_limit"),
    (EntryExposure(900, 0, 0), .1, .1, "directional_notional_limit"),
    (EntryExposure(900, 900, 0), .1, .1, "gross_notional_limit"),
    (EntryExposure(0, 0, 0), 10, 0, "quantity_step_limit"),
    (EntryExposure(0, 0, 0), .1, 10, "minimum_order_quantity"),
    (EntryExposure(0, 0, 30), .1, .1, None),
])
def test_quantity_audit_explains_limits_without_changing_result(settings, exposure, step, minimum, reason):
    kwargs = dict(step_size=step, minimum_quantity=minimum, stop_price=98, side=Side.LONG,
                  target_notional_usdt=600, exposure=exposure, portfolio_risk_cap_pct=.04)
    audit = {}
    before = calculate_order_quantity(1000, 100, PortfolioState(), settings, **kwargs)
    after = calculate_order_quantity(1000, 100, PortfolioState(), settings, audit=audit, **kwargs)
    assert after == before
    assert audit["reason"] == reason
    assert audit["existing_portfolio_stop_risk"] == exposure.stop_risk
    assert audit["max_portfolio_stop_risk"] == 40
    assert audit["remaining_portfolio_stop_risk"] == max(0, 40 - exposure.stop_risk)


def test_initial_entry_zero_quantity_persists_xmr_like_risk_reason(settings, monkeypatch):
    state = _state()
    old = "OLD/USDT:USDT"
    state.portfolio.positions[old] = Position(old, Side.LONG, 1, 100, 90, NOW, "test", "MID")
    state.exchange_snapshot["positions"][old] = {"side": "long", "quantity": 1, "mark_price": 400}
    state.exchange_orders_by_symbol[old] = {"stop": {"id": "stop", "amount": 1, "stop_price": 95}}
    state.protective_stop_audit["position_count"] = 1
    monkeypatch.setattr(runner, "_utc_now", lambda: NOW)
    monkeypatch.setattr(runner, "_ema9_reclaim_side", lambda _frame: "long")
    context = EntrySignalContext(SYMBOL, Side.LONG, NOW, NOW, NOW, 98, 99, 101, 1, 97, 103, .01)
    candidate = (SYMBOL, .02, Side.LONG, 99, 97, 1, 97, 103, "MID", "test", None,
                 context, TimeFilterFlags(True, True, False, False), 0)
    execution = SimpleNamespace(lead_market_ids={"TESTUSDT"})
    public = SimpleNamespace(fetch_funding_rate=lambda _: {"fundingRate": 0})
    assert runner._submit_entry_candidate(public, state, execution, True,
        {SYMBOL: {"id": "TESTUSDT", "precision": {"amount": .1}}}, candidate,
        settings, 5, 0, 0, NOW, MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW,
                                                   "BTC/USDT:USDT", "4h", {})) is None
    event = state.events[-1]
    assert event["event"] == "entry_risk_rejected" and event["phase"] == "quantity"
    assert event["reason"] == "portfolio_stop_risk_limit"
    assert event["details"]["existing_portfolio_stop_risk"] > 40
    assert event["details"]["existing_portfolio_giveback_risk"] == pytest.approx(300)
    assert event["details"]["portfolio_giveback_risk_weight"] == pytest.approx(.15)
    assert event["details"]["max_portfolio_stop_risk"] == 40


def test_daily_threshold_preserves_baseline_and_existing_latch(settings):
    state = PortfolioState()
    demo = load_settings(env_file=None)
    assert not state.observe_equity(1000, NOW, demo)
    assert state.observe_equity(950, NOW, demo)
    assert state.observe_equity(990, NOW, settings)
    assert state.day_start_equity == 1000
    assert not state.observe_equity(990, NOW + timedelta(days=1), settings)
    assert state.day_start_equity == 990
    fresh = PortfolioState()
    fresh.observe_equity(1000, NOW, settings)
    assert not fresh.observe_equity(901, NOW, settings)
    assert fresh.observe_equity(900, NOW, settings)


def _add_position(state, symbol, side, mark, stop):
    state.portfolio.positions[symbol] = Position(symbol, side, 1, 100,
        90 if side is Side.LONG else 110, NOW, "manual_adoption", "MID")
    state.exchange_snapshot["positions"][symbol] = {"side": side.value, "quantity": 1,
        "info": {"markPrice": mark}}
    state.exchange_orders_by_symbol[symbol] = {"stop": {"id": "stop", "amount": 1,
        "stop_price": stop, "external": True}}
    state.protective_stop_audit["position_count"] += 1


def test_mark_to_active_stop_risk_includes_adopted_profit_positions_without_offset(settings):
    state = _state()
    _add_position(state, "A", Side.LONG, 120, 110)
    _add_position(state, "B", Side.SHORT, 80, 85)
    exposure = lead_entry_exposure(state, settings, NOW)
    assert exposure.long_notional == 120 and exposure.short_notional == 80
    assert exposure.capital_risk == pytest.approx(0.79)
    assert exposure.giveback_risk == pytest.approx(15.0)
    assert exposure.weighted_giveback_risk == pytest.approx(2.25)
    assert exposure.stop_risk == pytest.approx(3.04)
    assert state.portfolio.positions["A"].stop_price == 90


@pytest.mark.parametrize("side,mark,stop", [
    (Side.LONG, 98, 95),
    (Side.SHORT, 102, 105),
])
def test_lead_capital_risk_keeps_loss_from_entry_while_giveback_is_zero(settings, side, mark, stop):
    state = _state()
    _add_position(state, "A", side, mark, stop)
    exposure = lead_entry_exposure(state, settings, NOW)
    expected = abs(100 - stop) + (100 + stop) * (
        settings.fees.taker_bps / 10_000 + settings.risk.max_slippage_pct
    )
    assert exposure.capital_risk == pytest.approx(expected)
    assert exposure.giveback_risk == 0
    assert exposure.weighted_giveback_risk == 0
    assert exposure.stop_risk == pytest.approx(expected)


@pytest.mark.parametrize("fault", ["missing_mark", "stop_crossed", "stale", "pending", "unverified", "foreign"])
def test_unreliable_live_risk_inputs_fail_closed(settings, fault):
    state = _state()
    _add_position(state, "A", Side.LONG, 120, 110)
    if fault == "missing_mark":
        state.exchange_snapshot["positions"]["A"]["info"] = {}
    elif fault == "stop_crossed":
        state.exchange_orders_by_symbol["A"]["stop"]["stop_price"] = 121
    elif fault == "stale":
        state.exchange_snapshot["at"] = (NOW - timedelta(seconds=91)).isoformat()
    elif fault == "pending":
        state.mr_entry_attempts["a"] = {"status": "uncertain"}
    elif fault == "unverified":
        state.protective_stop_audit["status"] = "mismatch"
    else:
        state.exchange_snapshot["positions"]["B"] = {}
    with pytest.raises((ValueError, TypeError, KeyError)):
        lead_entry_exposure(state, settings, NOW)


@pytest.mark.parametrize("side,stop,leverage", [(Side.LONG, 98, 9), (Side.SHORT, 102, 9),
                                               (Side.LONG, 90, 4), (Side.SHORT, 110, 4)])
def test_dynamic_leverage_uses_real_tier_and_preserves_stop(settings, side, stop, leverage):
    brackets = parse_brackets(_brackets(), "TESTUSDT")
    plan = select_leverage(side, 100, 6, stop, brackets, settings, 1000)
    assert plan["leverage"] == leverage
    assert plan["notional"] == 600
    assert liquidation_is_safe(side, 100, stop, plan["estimated_liquidation_price"], settings.lead_risk)


@pytest.mark.parametrize("stop,available,max_leverage", [(80, 1000, 20), (98, 62, 20), (98, 1000, 2)])
def test_unsafe_or_unaffordable_leverage_rejects(settings, stop, available, max_leverage):
    with pytest.raises(ValueError):
        select_leverage(Side.LONG, 100, 6, stop,
            parse_brackets(_brackets(leverage=max_leverage), "TESTUSDT"), settings, available)


def test_liquidation_recalculates_tier_at_liquidation_notional():
    rows = {"symbol": "TESTUSDT", "brackets": [
        {"notionalFloor": 0, "notionalCap": 1050, "initialLeverage": 20, "maintMarginRatio": .005, "cum": 0},
        {"notionalFloor": 1050, "notionalCap": 10000, "initialLeverage": 10, "maintMarginRatio": .01, "cum": 5.25}]}
    brackets = parse_brackets(rows, "TESTUSDT")
    price = liquidation_price(Side.SHORT, 100, 10, 100, brackets)
    assert price == pytest.approx((1100 + 5.25) / 10.1)
    assert price * 10 > 1050
    assert 100 + 10 * (100 - price) == pytest.approx(price * 10 * .01 - 5.25)


@pytest.mark.parametrize("liquidation", [None, 0, float("nan"), float("inf"), 99, 101])
def test_invalid_or_insufficient_liquidation_readback_rejects(settings, liquidation):
    assert not liquidation_is_safe(Side.LONG, 100, 98, liquidation, settings.lead_risk)


class LeadClient:
    urls = {"api": {**LIVE_FAPI_URLS, "sapi": LIVE_SAPI_URL}}

    def __init__(self):
        self.leverage = 3
        self.leverage_calls = []
        self.orders = []
        self.live = None
        self.liquidation = 90.6
        self.available = 1000
        self.auto_margin = False
        self.stop = None

    def market(self, symbol):
        return {"id": "TESTUSDT", "limits": {}, "info": {}}

    def fetch_position_mode(self):
        return {"hedged": False}

    def fetch_positions(self, symbols):
        return [] if self.live is None else [self.live]

    def fapiPrivateGetLeverageBracket(self, params):
        return _brackets()

    def fapiPrivateGetSymbolConfig(self, params):
        return [{"symbol": "TESTUSDT", "leverage": self.leverage, "isAutoAddMargin": self.auto_margin}]

    def fetch_balance(self):
        return {"info": {"availableBalance": self.available}}

    def set_margin_mode(self, mode, symbol):
        assert mode == "isolated"

    def set_leverage(self, leverage, symbol):
        self.leverage = leverage
        self.leverage_calls.append(leverage)
        return {"leverage": leverage, "maxNotionalValue": "10000"}

    def amount_to_precision(self, symbol, value):
        return str(value)

    price_to_precision = amount_to_precision

    def create_order(self, symbol, kind, side, quantity, price, params):
        self.orders.append((kind, side, quantity, params))
        if kind == "STOP_MARKET":
            self.stop = {"algoId": "stop", "clientAlgoId": params["clientOrderId"],
                "symbol": "TESTUSDT", "orderType": "STOP_MARKET", "side": side.upper(),
                "algoStatus": "NEW", "reduceOnly": True, "quantity": quantity,
                "triggerPrice": params["stopPrice"], "workingType": params["workingType"], "positionSide": "BOTH"}
            return {"id": "stop"}
        if params.get("reduceOnly"):
            self.live = None
        else:
            self.live = {"symbol": symbol, "side": "long", "contracts": quantity,
                "entryPrice": 100, "unrealizedPnl": 0, "marginMode": "isolated",
                "markPrice": 100, "liquidationPrice": self.liquidation,
                "info": {"isolatedWallet": quantity * 100 / self.leverage - quantity * 100 * .0005}}
        return {"id": "fill", "filled": quantity, "average": 100, "status": "closed"}

    def fapiPrivateGetOpenAlgoOrders(self, params):
        return [self.stop] if self.stop else []

    def fapiPrivateDeleteAlgoOrder(self, params):
        assert params["algoId"] == "stop"
        self.stop = None


def test_live_open_reserves_costs_above_ten_percent_margin_and_verifies_mark_stop(settings):
    client = LeadClient()
    service = LeadPortfolioExecutionService(client, settings)
    plans, submissions = [], []
    result = service.open(_intent(), stop_price=98, on_safety_plan=plans.append,
        client_order_id="mrb_test", on_submit=lambda *args: submissions.append(args))
    assert client.leverage_calls == [9]
    assert plans[0]["leverage"] == 9 and submissions[0][1] == 98
    assert client.orders[1][-1]["workingType"] == "MARK_PRICE"
    assert result["capital_safety"]["verified"] is True
    assert result["capital_safety"]["actual_liquidation_price"] == 90.6
    assert [row[0] for row in client.orders] == ["market", "STOP_MARKET"]


@pytest.mark.parametrize("liquidation", [97, None, float("nan")])
def test_bad_actual_liquidation_closes_only_new_position_reduce_only(settings, liquidation):
    client = LeadClient()
    client.liquidation = liquidation
    with pytest.raises(RuntimeError, match="reduceOnly closed"):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98)
    assert client.live is None and client.stop is None
    assert client.orders[-1][-1]["reduceOnly"] is True
    assert client.orders[1][-1]["stopPrice"] == 98


def test_existing_position_is_never_releveraged(settings):
    client = LeadClient()
    client.live = {"symbol": SYMBOL, "side": "long", "contracts": 1,
                   "entryPrice": 100, "unrealizedPnl": 0}
    with pytest.raises(RuntimeError, match="already exists"):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98)
    assert client.leverage_calls == [] and client.orders == []


@pytest.mark.parametrize("fault", ["brackets", "auto_margin", "margin"])
def test_pre_entry_failures_never_send_an_order(settings, fault):
    client = LeadClient()
    if fault == "brackets":
        client.fapiPrivateGetLeverageBracket = lambda _: {}
    elif fault == "auto_margin":
        client.auto_margin = True
    else:
        client.available = 60
    with pytest.raises(EntryPreflightRejected):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98)
    assert client.orders == [] and client.leverage_calls == []


def test_final_quote_quantity_is_used_for_leverage_and_margin(settings):
    client = LeadClient()
    result = LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98,
        before_submit=lambda intent, stop: (replace(intent, quantity=3), stop))
    assert result["capital_safety"]["notional"] == 300
    assert client.orders[0][2] == 3


@pytest.mark.parametrize("side,structure", [(Side.LONG, 98), (Side.SHORT, 102)])
def test_same_market_data_produces_same_strategy_stop_under_both_profiles(settings, side, structure):
    demo = load_settings(env_file=None)
    assert calculate_adaptive_stop(side, 100, 2, structure, settings) == calculate_adaptive_stop(
        side, 100, 2, structure, demo)


def test_leverage_preparation_does_not_bypass_quote_expiry_checkpoint(settings, monkeypatch):
    import bot.execution as execution_module
    clock = [NOW]
    monkeypatch.setattr(runner, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(execution_module, "_utc_now", lambda: clock[0])
    state = _state()
    client = LeadClient()
    original = client.set_leverage

    def delayed_leverage(*args):
        result = original(*args)
        clock[0] += timedelta(seconds=16)
        return result

    client.set_leverage = delayed_leverage
    result = runner._open_position(True, LeadPortfolioExecutionService(client, settings), state,
        SYMBOL, Side.LONG, 100, 98, 6, 3, .0005, .0003, "test", "MID", NOW,
        entry_preflight=lambda intent, stop: (intent, stop),
        submission_deadline=NOW + timedelta(seconds=60),
        entry_audit={"symbol": SYMBOL, "side": "long",
                     "quote_valid_until": (NOW + timedelta(seconds=15)).isoformat()},
        scan_row={"bar": NOW.isoformat()}, checkpoint=lambda: None)
    assert result is None and client.orders == []
    attempt = next(iter(state.mr_entry_attempts.values()))
    assert attempt["status"] == "not_submitted"
    assert attempt["audit"]["capital_safety"]["leverage"] == 9


class ProtectionClient(LeadClient):
    def __init__(self, side=Side.LONG, *, entry=100, stop=95, liquidation=80, wallet=30):
        super().__init__()
        self.live = {"symbol": SYMBOL, "side": side.value, "contracts": 2,
                     "entryPrice": entry, "markPrice": entry, "marginMode": "isolated",
                     "liquidationPrice": liquidation, "info": {"isolatedWallet": wallet}}
        self.active = [{"algoId": "old", "clientAlgoId": "mrb_s_old", "symbol": "TESTUSDT",
                        "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL",
                        "reduceOnly": True, "quantity": 2, "triggerPrice": stop,
                        "workingType": "CONTRACT_PRICE"}]
        self.calls = []
        self.fail_cancel = False
        self.return_wrong_trigger = False

    def create_order(self, symbol, kind, side, quantity, price, params):
        assert kind == "STOP_MARKET"
        self.calls.append(("place", quantity, params["stopPrice"], params["workingType"]))
        order_id = f"new-{len(self.calls)}"
        row = {**self.active[0], "algoId": order_id, "clientAlgoId": params["clientOrderId"],
               "side": side.upper(), "quantity": quantity, "triggerPrice": params["stopPrice"],
               "workingType": "CONTRACT_PRICE" if self.return_wrong_trigger else params["workingType"]}
        self.active.append(row)
        return {"id": order_id, "info": row}

    def fapiPrivateGetOpenAlgoOrders(self, params):
        return [dict(row) for row in self.active]

    def fapiPrivateGetAlgoOrder(self, params):
        return next(dict(row) for row in self.active if row["algoId"] == params["algoId"])

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.calls.append(("cancel", params["algoId"]))
        if self.fail_cancel:
            raise RuntimeError("cancellation unavailable")
        self.active = [row for row in self.active if row["algoId"] != params["algoId"]]


def _protection_service(settings, client):
    service = LeadPortfolioExecutionService(client, settings)
    service._market_rules = lambda symbol: (.01, .01, 0)
    pos = Position(SYMBOL, Side(client.live["side"]), float(client.live["contracts"]),
                   client.live["entryPrice"], float(client.active[0]["triggerPrice"]),
                   NOW, "manual_position_adopted", "MID")
    stop = {"id": "old", "clientOrderId": "mrb_s_old", "amount": 2,
            "stop_price": pos.stop_price}
    return service, pos, stop


@pytest.mark.parametrize("side,entry,stop,liquidation,wallet,reason", [
    (Side.LONG, 37.21, 34.95, 35.70656566, 7.442, "liquidation_buffer"),
    (Side.SHORT, 100, 105, 103, 20, "liquidation_buffer"),
    (Side.LONG, 100, 95, 92, 20, "liquidation_buffer"),
    (Side.LONG, 100, 99.9, 99.5, 20, "liquidation_buffer"),
    (Side.LONG, 100, 95, 80, 10, "margin_below"),
    (Side.LONG, 100, 95, 80, None, "margin_below"),
    (Side.LONG, 100, 95, 80, float("nan"), "margin_below"),
    (Side.LONG, 100, 95, None, 20, "liquidation_buffer"),
])
def test_manual_lead_adoption_rejects_unsafe_margin_or_liquidation_without_orders(
        settings, side, entry, stop, liquidation, wallet, reason):
    client = ProtectionClient(side, entry=entry, stop=stop, liquidation=liquidation, wallet=wallet)
    client.active = []
    service = LeadPortfolioExecutionService(client, settings)
    service._market_rules = lambda symbol: (.01, .01, 0)
    pos = Position(SYMBOL, side, 2, entry, stop, NOW, "manual_position_adopted", "MID")
    with pytest.raises(RuntimeError, match=reason):
        service.adopt_manual_position_stop(pos, strategy_stop=stop, market_price=entry)
    assert client.calls == []


def test_adoption_migrates_tighter_legacy_stop_without_widening(settings):
    client = ProtectionClient(stop=96)
    service, pos, stop = _protection_service(settings, client)
    saved = service.adopt_manual_position_stop(pos, strategy_stop=95, market_price=100)
    assert saved["stop_price"] == 96
    assert client.calls == [("place", 2, 96, "MARK_PRICE"), ("cancel", "old")]
    assert len(client.active) == 1
    assert service.audit_active_protective_stops({SYMBOL: pos}, {SYMBOL: {"stop": saved}}) == []


@pytest.mark.parametrize("mode,reason", [
    ("legacy", "trigger_must_be_mark"), ("unsafe_liquidation", "liquidation_buffer"),
    ("low_wallet", "margin_below"), ("missing_mark", "mark_price"),
])
def test_continuous_audit_flags_trigger_margin_and_liquidation_after_adoption(settings, mode, reason):
    client = ProtectionClient()
    if mode != "legacy":
        client.active[0]["workingType"] = "MARK_PRICE"
    if mode == "unsafe_liquidation":
        client.live["liquidationPrice"] = 97
    if mode == "low_wallet":
        client.live["info"]["isolatedWallet"] = 10
    if mode == "missing_mark":
        client.live["markPrice"] = None
    service, pos, stop = _protection_service(settings, client)
    mismatches = service.audit_active_protective_stops({SYMBOL: pos}, {SYMBOL: {"stop": stop}})
    assert len(mismatches) == 1 and reason in mismatches[0]
    assert client.calls == []


def test_manual_add_with_insufficient_wallet_preserves_original_stop(settings):
    client = ProtectionClient(wallet=20)
    service, original, stop = _protection_service(settings, client)
    client.live["contracts"] = 3
    with pytest.raises(RuntimeError, match="margin_below"):
        service.reconcile_manual_position_stop(replace(original, quantity=3), original, stop,
                                               stop_price=95, market_price=100)
    assert client.calls == [] and client.active[0]["algoId"] == "old"


def test_manual_resize_migrates_stop_and_retry_reuses_verified_replacement(settings):
    client = ProtectionClient(wallet=40)
    service, original, stop = _protection_service(settings, client)
    client.live["contracts"] = 3
    client.fail_cancel = True
    updated = replace(original, quantity=3)
    with pytest.raises(RuntimeError, match="cancellation unavailable"):
        service.reconcile_manual_position_stop(updated, original, stop, stop_price=95, market_price=100)
    assert len(client.active) == 2
    client.fail_cancel = False
    saved = service.reconcile_manual_position_stop(updated, original, stop, stop_price=95, market_price=100)
    assert len([c for c in client.calls if c[0] == "place"]) == 1
    assert len(client.active) == 1 and client.active[0]["algoId"] == saved["id"]


def test_same_price_stop_replacement_migrates_trigger_and_preserves_new_stop_on_cancel_failure(settings):
    client = ProtectionClient()
    service, pos, stop = _protection_service(settings, client)
    client.fail_cancel = True
    with pytest.raises(RuntimeError, match="cancellation unavailable"):
        service.replace_stop(pos, stop=stop, stop_price=95)
    assert client.calls == [("place", 2, 95, "MARK_PRICE"), ("cancel", "old")]
    assert len(client.active) == 2 and client.active[-1]["workingType"] == "MARK_PRICE"


def test_wrong_trigger_readback_never_cancels_old_stop(settings):
    client = ProtectionClient()
    client.return_wrong_trigger = True
    service, pos, stop = _protection_service(settings, client)
    with pytest.raises(RuntimeError, match="could not be verified"):
        service.replace_stop(pos, stop=stop, stop_price=96)
    assert client.active[0]["algoId"] == "old" and len(client.active) == 1


@pytest.mark.parametrize("side,stop,mark,liquidation", [
    (Side.LONG, 105, 110, 80), (Side.SHORT, 95, 90, 120),
])
def test_profit_lock_and_existing_cross_position_pass_safety_check(settings, side, stop, mark, liquidation):
    client = ProtectionClient(side, stop=95 if side is Side.LONG else 105, liquidation=liquidation)
    client.live.update(markPrice=mark, marginMode="cross", info={})
    service, pos, _ = _protection_service(settings, client)
    service._verify_stop_safety(pos, service._position(SYMBOL), stop)


def test_higher_margin_floor_selects_lower_leverage_and_config_cannot_weaken_ten_percent(settings):
    from pydantic import ValidationError

    policy = settings.lead_risk.model_copy(update={"min_isolated_margin_pct": .20})
    changed = settings.model_copy(update={"lead_risk": policy})
    plan = select_leverage(Side.LONG, 100, 6, 98, parse_brackets(_brackets(), "TESTUSDT"), changed, 1000)
    assert plan["leverage"] == 4
    data = settings.lead_risk.model_dump()
    data["min_isolated_margin_pct"] = .05
    with pytest.raises(ValidationError):
        type(settings.lead_risk).model_validate(data)


@pytest.mark.parametrize("safe", [True, False])
def test_restart_recovery_rechecks_actual_liquidation_without_resending(settings, safe):
    client = LeadClient()
    service = LeadPortfolioExecutionService(client, settings)
    result = service.open(_intent(), stop_price=98)
    client.live["liquidationPrice"] = 90.6 if safe else 97
    client.fapiPrivateGetOrder = lambda _: {"clientOrderId": "mrb_saved", "symbol": "TESTUSDT",
        "side": "BUY", "reduceOnly": False, "executedQty": 6, "status": "FILLED",
        "orderId": "fill", "avgPrice": 100}
    client.fapiPrivateGetUserTrades = lambda _: [{"orderId": "fill", "qty": 6, "time": 1794139200000}]
    attempt = {"symbol": SYMBOL, "side": "long", "quantity": 6, "stop_price": 98,
        "clientOrderId": "mrb_saved", "submitted_at": NOW.isoformat(),
        "audit": {"capital_safety": result["capital_safety"]}}
    if safe:
        report = service.inspect_entry_attempt(attempt)
        assert report["status"] == "protected"
        assert report["result"]["capital_safety"]["verified"]
        assert len(client.orders) == 2
    else:
        with pytest.raises(RuntimeError, match="reduceOnly closed"):
            service.inspect_entry_attempt(attempt)
        assert client.live is None
        assert len([o for o in client.orders if o[0] == "market" and not o[-1].get("reduceOnly")]) == 1


@pytest.mark.parametrize("field,value", [("initialLeverage", 0), ("notionalCap", float("nan")),
                                         ("maintMarginRatio", 1), ("cum", -1)])
def test_invalid_venue_tiers_do_not_receive_a_fallback(field, value):
    response = _brackets()
    response["brackets"][0][field] = value
    with pytest.raises(ValueError):
        parse_brackets(response, "TESTUSDT")


@pytest.mark.parametrize("confirmation", [None, {"leverage": 10, "maxNotionalValue": "NaN"},
    {"leverage": 10.5, "maxNotionalValue": "10000"}, {"leverage": 10, "maxNotionalValue": "10000"},
    {"leverage": 10, "maxNotionalValue": "500"}])
def test_unconfirmed_venue_leverage_or_notional_limit_prevents_submission(settings, confirmation):
    client = LeadClient()
    client.set_leverage = lambda *args: confirmation
    with pytest.raises(EntryPreflightRejected, match="confirmation_failed"):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98)
    assert client.orders == []


def test_next_candidate_refreshes_snapshot_after_prior_fill(settings, monkeypatch):
    state = _state()
    _add_position(state, "OLD", Side.LONG, 120, 110)
    state.exchange_snapshot["positions"] = {}  # Previous fill not yet in the account snapshot.
    state.protective_stop_audit["status"] = "pending"
    calls = []
    monkeypatch.setattr(runner, "_utc_now", lambda: NOW)
    monkeypatch.setattr(runner, "_ema9_reclaim_side", lambda _frame: "long")

    def refresh(*args):
        calls.append("account")
        state.exchange_snapshot["positions"] = {"OLD": {"side": "long", "quantity": 1,
            "mark_price": 120}}
        return True

    def audit(*args):
        calls.append("stops")
        state.protective_stop_audit["status"] = "verified"
        return True

    monkeypatch.setattr(runner, "_demo_sync_stopped", refresh)
    monkeypatch.setattr(runner, "_audit_demo_protective_stops", audit)
    monkeypatch.setattr(runner, "_fetch_entry_funding_rate", lambda *args, **kwargs: 0)
    monkeypatch.setattr(runner, "_open_position", lambda *args, **kwargs: calls.append("open") or None)
    context = EntrySignalContext(SYMBOL, Side.LONG, NOW, NOW, NOW, 97, 98, 100, 1, 97, 103, .01)
    candidate = (SYMBOL, 1, Side.LONG, 98, 96.6, 1, 97, 103, "MID", "test", None, context, None, 0)
    market_risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW, "BTC/USDT:USDT", "4h", {})
    execution = SimpleNamespace(lead_market_ids={"TESTUSDT"})
    runner._submit_entry_candidate(None, state, execution, True, {SYMBOL: {"id": "TESTUSDT"}},
        candidate, settings, 5, .0005, .0003, NOW, market_risk)
    assert calls == ["account", "stops", "open"]


def test_manual_position_appearing_during_preflight_is_not_releveraged(settings):
    client = LeadClient()

    def preflight(intent, stop):
        client.live = {"symbol": SYMBOL, "side": "long", "contracts": 1,
                       "entryPrice": 100, "unrealizedPnl": 0}
        return intent, stop

    with pytest.raises(EntryPreflightRejected, match="appeared_before_leverage_change"):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98,
                                                            before_submit=preflight)
    assert client.leverage_calls == [] and client.orders == []


def test_safety_exit_failure_keeps_protective_stop_and_requires_reconciliation(settings):
    client = LeadClient()
    client.liquidation = 97
    original = client.create_order

    def fail_close(symbol, kind, side, quantity, price, params):
        if kind == "market" and params.get("reduceOnly"):
            assert client.stop is not None
            raise RuntimeError("simulated close failure")
        return original(symbol, kind, side, quantity, price, params)

    client.create_order = fail_close
    with pytest.raises(RuntimeError, match="emergency close failed; reconciliation required"):
        LeadPortfolioExecutionService(client, settings).open(_intent(), stop_price=98)
    assert client.live is not None and client.stop["triggerPrice"] == 98


def test_exchange_maximum_leverage_is_respected(settings):
    plan = select_leverage(Side.LONG, 100, 6, 98,
        parse_brackets(_brackets(leverage=6), "TESTUSDT"), settings, 1000)
    assert plan["leverage"] == 6 and plan["notional"] == 600
