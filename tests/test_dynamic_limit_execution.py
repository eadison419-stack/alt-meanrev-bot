from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bot.config import RuntimeSettings, load_settings
from bot.exchange.binance_usdm import FAPI_URLS
from bot.execution import DemoExecutionService
from bot.live.multi_paper import MultiPaperState
from bot.regime import Regime
from bot.risk import OrderIntent
from bot.signals import Action, Side


class LimitClient:
    def __init__(self):
        self.urls = {"api": dict(FAPI_URLS)}
        self.active_stops = []
        self.order = {
            "clientOrderId": "mrb_test_entry", "symbol": "ADAUSDT",
            "side": "BUY", "type": "LIMIT", "reduceOnly": False,
            "status": "NEW", "executedQty": "0", "avgPrice": "0", "orderId": 42,
        }
        self.deleted_entry_ids = []
        self.deleted_stop_ids = []
        self.calls = []
        self.position = None

    def market(self, _symbol):
        return {"id": "ADAUSDT"}

    def amount_to_precision(self, _symbol, amount):
        return str(amount)

    def price_to_precision(self, _symbol, price):
        return str(price)

    def fapiPrivateGetOrder(self, params):
        assert params["origClientOrderId"] == "mrb_test_entry"
        return dict(self.order)

    def fapiPrivateGetOpenAlgoOrders(self, _params):
        self.calls.append("query_stops")
        return list(self.active_stops)

    def fapiPrivateDeleteOrder(self, params):
        self.deleted_entry_ids.append(params["origClientOrderId"])
        self.order["status"] = "CANCELED"

    def fapiPrivateGetUserTrades(self, _params):
        return [{"orderId": 42, "qty": "0.5", "commission": "0.01",
                 "commissionAsset": "USDT", "time": 1791676800000}]

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.deleted_stop_ids.append(str(params["algoId"]))
        self.active_stops = [row for row in self.active_stops
                             if str(row["algoId"]) != str(params["algoId"])]

    def create_order(self, symbol, kind, side, amount, price, params):
        self.calls.append("entry")
        return {"id": "42", "info": {"orderId": 42}}


def _service(client):
    service = DemoExecutionService(client, RuntimeSettings())
    service._verify_urls = lambda: None
    service._ensure_one_way = lambda: None
    service._position = lambda _symbol: client.position
    service._prepare_entry = lambda _intent: None
    service._finalize_entry = lambda _intent, _prepared: None
    service._market_rules = lambda _symbol: (0.1, 0.1, 5.0)
    return service


def _attempt(now, *, expires_at=None):
    return {
        "symbol": "ADA/USDT:USDT", "side": "long",
        "clientOrderId": "mrb_test_entry", "order_type": "limit",
        "submitted_at": now.isoformat(),
        "expires_at": (expires_at or now + timedelta(hours=3)).isoformat(),
        "quantity": 1.0, "stop_price": 90.0,
        "stop_client_algo_id": "mrb_s_mrb_test_entry",
        "preprotected_stop": {
            "id": "7", "clientOrderId": "mrb_s_mrb_test_entry",
            "stop_price": 90.0, "amount": None, "close_all": True,
        },
        "audit": {},
    }


def _active_close_all_stop():
    return {
        "algoId": "7", "clientAlgoId": "mrb_s_mrb_test_entry",
        "symbol": "ADAUSDT", "orderType": "STOP_MARKET",
        "algoStatus": "NEW", "side": "SELL", "quantity": "0",
        "triggerPrice": "90.0", "workingType": "CONTRACT_PRICE",
        "closePosition": True, "reduceOnly": False,
    }


def test_open_limit_persists_and_verifies_close_all_stop_before_gtd_entry(monkeypatch):
    now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    client = LimitClient()
    client.active_stops = [_active_close_all_stop()]
    service = _service(client)
    monkeypatch.setattr("bot.execution._utc_now", lambda: now)
    events = []

    def place_stop(*args, **kwargs):
        events.append("stop")
        assert kwargs["close_all"] is True
        assert kwargs["client_order_id"] == "mrb_s_mrb_test_entry"
        return {"id": "7", "clientOrderId": "mrb_s_mrb_test_entry",
                "stop_price": 90.0, "amount": None, "close_all": True}

    service._place_stop = place_stop
    service._saved_stop_matches = lambda *_args: True
    service.inspect_limit_entry_attempt = lambda _attempt: {"status": "pending"}
    client.create_order = lambda *args: (events.append("entry")
                                          or {"id": "42", "info": {"orderId": 42}})

    def on_submit(*_args):
        events.append("submission_checkpoint")

    def on_protection(_stop):
        events.append("protection_checkpoint")

    intent = OrderIntent("ADA/USDT:USDT", Side.LONG, Action.OPEN, 1.0,
                         100.0, "test", Regime.MID, 90.0, "mean_reversion")
    result = service.open_limit(
        intent, stop_price=90.0, limit_expires_at=now + timedelta(hours=3),
        client_order_id="mrb_test_entry", on_submit=on_submit,
        on_protection=on_protection,
    )

    assert result["pending_limit"] is True
    assert events == ["submission_checkpoint", "stop", "protection_checkpoint", "entry"]


def test_pending_limit_remains_open_only_with_matching_protection(monkeypatch):
    now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    client = LimitClient()
    client.active_stops = [_active_close_all_stop()]
    service = _service(client)
    monkeypatch.setattr("bot.execution._utc_now", lambda: now)

    report = service.inspect_limit_entry_attempt(_attempt(now))

    assert report == {"status": "pending", "orderId": "42", "filled": 0.0}
    assert client.deleted_entry_ids == []


def test_partial_limit_fill_cancels_remainder_and_verifies_close_all_protection(monkeypatch):
    now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    client = LimitClient()
    client.order.update(status="PARTIALLY_FILLED", executedQty="0.5", avgPrice="100")
    client.active_stops = [_active_close_all_stop()]
    client.position = {"side": "long", "quantity": 0.5, "entry_price": 100.0}
    service = _service(client)
    monkeypatch.setattr("bot.execution._utc_now", lambda: now)

    report = service.inspect_limit_entry_attempt(_attempt(now))

    assert report["status"] == "protected"
    assert report["result"]["fill"]["filled"] == 0.5
    assert report["result"]["stop"]["close_all"] is True
    assert client.deleted_entry_ids == ["mrb_test_entry"]
    assert report["result"]["position"]["quantity"] == 0.5


def test_expired_zero_fill_cancels_entry_and_then_removes_idle_stop(monkeypatch):
    now = datetime(2026, 11, 11, 14, tzinfo=timezone.utc)
    client = LimitClient()
    client.active_stops = [_active_close_all_stop()]
    service = _service(client)
    monkeypatch.setattr("bot.execution._utc_now", lambda: now)
    canceled = []
    service.cancel_saved_stop_if_open = lambda symbol, stop: canceled.append((symbol, stop["id"]))

    report = service.inspect_limit_entry_attempt(
        _attempt(now - timedelta(hours=3), expires_at=now - timedelta(seconds=1)),
    )

    assert report["status"] == "not_filled"
    assert client.deleted_entry_ids == ["mrb_test_entry"]
    assert canceled == [("ADA/USDT:USDT", "7")]


def _local_attempt(now, *, expires_at=None):
    return {
        "symbol": "ADA/USDT:USDT", "side": "long", "status": "paper_limit_pending",
        "reason": "synthetic_entry", "regime": Regime.MID.value,
        "limit_price": 99.0, "stop_price": 95.0, "quantity": 1.0,
        "expires_at": (expires_at or now + timedelta(hours=3)).isoformat(),
        "ema9_reclaim_side": "long", "funding_flip": {"side": "long"},
        "meta": {"entry_atr": 1.0, "structure_low": 95.2, "structure_high": 101.0},
    }


def test_local_paper_limit_waits_for_touch_then_fills_at_limit():
    from scripts import run_mr_candidate_paper as runner

    now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    state = MultiPaperState(initial_equity=1000, cash=1000)
    state.mr_entry_attempts["paper_limit|ADA|bar"] = _local_attempt(now)
    settings = load_settings(env_file=None)

    runner._process_local_dynamic_limits(
        state, {"ADA/USDT:USDT": 99.5}, settings, now, 0.001, 0.01,
    )
    assert state.portfolio.positions == {}

    runner._process_local_dynamic_limits(
        state, {"ADA/USDT:USDT": 98.5}, settings, now + timedelta(minutes=5), 0.001, 0.01,
    )
    position = state.portfolio.positions["ADA/USDT:USDT"]
    assert position.entry_price == pytest.approx(99.0)
    assert position.stop_price == pytest.approx(95.0)
    assert state.mr_entry_attempts["paper_limit|ADA|bar"]["status"] == "paper_limit_filled"


def test_local_paper_limit_expiry_and_gap_through_stop_are_conservative():
    from scripts import run_mr_candidate_paper as runner

    now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
    settings = load_settings(env_file=None)
    expired = MultiPaperState(initial_equity=1000, cash=1000)
    expired.mr_entry_attempts["paper_limit|expired"] = _local_attempt(
        now, expires_at=now - timedelta(seconds=1),
    )
    runner._process_local_dynamic_limits(
        expired, {"ADA/USDT:USDT": 98.0}, settings, now, 0.001, 0.01,
    )
    assert expired.portfolio.positions == {}
    assert expired.mr_entry_attempts["paper_limit|expired"]["status"] == "paper_limit_expired"

    gapped = MultiPaperState(initial_equity=1000, cash=1000)
    gapped.mr_entry_attempts["paper_limit|gapped"] = _local_attempt(now)
    runner._process_local_dynamic_limits(
        gapped, {"ADA/USDT:USDT": 94.0}, settings, now, 0.001, 0.01,
    )
    assert gapped.portfolio.positions == {}
    actions = [(event.get("event"), event.get("action"), event.get("reason"))
               for event in gapped.events]
    assert ("paper_fill", "OPEN", "synthetic_entry") in actions
    assert ("paper_fill", "CLOSE", "protective_stop") in actions
