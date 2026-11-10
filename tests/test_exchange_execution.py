import pytest
from types import SimpleNamespace

from bot.config import RuntimeSettings
from bot.exchange.binance_usdm import (FAPI_URLS, LIVE_FAPI_URLS, LIVE_SAPI_URL,
                                       create_lead_usdm,
                                       lock_demo_urls, resolve_execution_mode,
                                       verify_demo_urls, verify_live_urls)
from bot.execution import DemoExecutionService, LeadPortfolioExecutionService
from bot.live.paper import PaperAccount, PaperTrader
from bot.regime import Regime
from bot.risk import OrderIntent
from bot.signals import Action, Side


def test_execution_mode_matrix_refuses_live_and_routes_only_testnet_true_to_demo():
    assert resolve_execution_mode(bot_mode="paper", testnet=True).name == "demo_usdm"
    assert resolve_execution_mode(bot_mode="paper", testnet=False).name == "local_paper"
    with pytest.raises(RuntimeError, match="live mode"):
        resolve_execution_mode(bot_mode="paper", testnet=True, live_flag=True)
    with pytest.raises(RuntimeError, match="live mode"):
        resolve_execution_mode(bot_mode="live", testnet=True)
    with pytest.raises(RuntimeError, match="BOT_MODE"):
        resolve_execution_mode(bot_mode="invalid", testnet=False)
    assert resolve_execution_mode(bot_mode="live", testnet=False,
                                  live_flag=True, live_confirmed=True).name == "lead_usdm"
    with pytest.raises(RuntimeError, match="human confirmation"):
        resolve_execution_mode(bot_mode="live", testnet=False, live_flag=True)
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET"):
        resolve_execution_mode(bot_mode="live", testnet=True,
                               live_flag=True, live_confirmed=True)


class FakeClient:
    def __init__(self):
        self.urls = {"api": {key: "https://fapi.binance.com/fapi/v1" for key in FAPI_URLS}}
        self.enabled = False
        self.session = SimpleNamespace(proxies={})

    def enable_demo_trading(self, enabled):
        self.enabled = enabled

    def close(self):
        self.closed = True


def test_demo_client_urls_are_pinned_after_ccxt_demo_mode_and_verified():
    exchange = FakeClient()
    lock_demo_urls(exchange)
    assert exchange.enabled is True
    assert exchange.urls["api"] == FAPI_URLS
    verify_demo_urls(exchange)


def test_host_mismatch_fails_closed_even_when_demo_switch_was_requested():
    exchange = FakeClient()
    lock_demo_urls(exchange)
    exchange.urls["api"]["fapiPrivate"] = "https://fapi.binance.com/fapi/v1"
    with pytest.raises(RuntimeError, match="demo-fapi"):
        verify_demo_urls(exchange)


def test_demo_factory_refuses_missing_testnet_keys_without_network():
    from bot.exchange.binance_usdm import create_demo_usdm

    runtime = RuntimeSettings(_env_file=None, binance_testnet=True, bot_mode="paper")
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_demo_usdm(runtime)


def test_lead_factory_pins_production_urls_without_enabling_demo_mode():
    client = FakeClient()
    captured = {}
    runtime = RuntimeSettings(_env_file=None, binance_testnet=False, bot_mode="live",
                              binance_api_key="lead-key", binance_api_secret="lead-secret")

    def factory(config):
        captured.update(config)
        return client

    assert create_lead_usdm(runtime, exchange_factory=factory) is client
    assert {key: client.urls["api"][key] for key in LIVE_FAPI_URLS} == LIVE_FAPI_URLS
    assert client.urls["api"]["sapi"] == LIVE_SAPI_URL
    assert client.enabled is False
    assert captured["apiKey"] == "lead-key"
    assert captured["secret"] == "lead-secret"
    verify_live_urls(client)


def test_live_execution_verifier_rejects_demo_host_and_never_changes_position_mode():
    class LiveClient:
        urls = {"api": {**LIVE_FAPI_URLS, "sapi": LIVE_SAPI_URL}}

        def __init__(self):
            self.set_position_mode_calls = 0

        def fetch_position_mode(self):
            return {"hedged": True}

        def set_position_mode(self, _hedged):
            self.set_position_mode_calls += 1

    client = LiveClient()
    service = LeadPortfolioExecutionService(client, object())
    with pytest.raises(RuntimeError, match="one-way"):
        service._ensure_one_way()
    assert client.set_position_mode_calls == 0
    client.urls["api"]["fapiPrivate"] = FAPI_URLS["fapiPrivate"]
    with pytest.raises(RuntimeError, match="fapi.binance.com"):
        verify_live_urls(client)


def test_lead_execution_requires_active_lead_identity_and_uses_verified_usdt_whitelist():
    class LeadClient:
        urls = {"api": {**LIVE_FAPI_URLS, "sapi": LIVE_SAPI_URL}}

        def sapiGetCopyTradingFuturesUserStatus(self, _params):
            return {"code": "000000", "success": True,
                    "data": {"isLeadTrader": True}}

        def sapiGetCopyTradingFuturesLeadSymbol(self, _params):
            return {"code": "000000", "success": True, "data": [
                {"symbol": "BTCUSDT", "quoteAsset": "USDT"},
                {"symbol": "ETHUSDT", "quoteAsset": "USDT"},
                {"symbol": "BTCUSDC", "quoteAsset": "USDC"},
            ]}

    service = LeadPortfolioExecutionService(LeadClient(), object())
    service._verify_lead_api_access()
    assert service.lead_market_ids == {"BTCUSDT", "ETHUSDT"}


def test_live_credentials_and_flags_are_not_accepted_by_runtime_matrix():
    with pytest.raises(RuntimeError):
        resolve_execution_mode(bot_mode="live", testnet=False, live_flag=False)
    with pytest.raises(RuntimeError):
        resolve_execution_mode(bot_mode="paper", testnet=True, live_flag=True)


class OrderSpy:
    def __init__(self):
        self.urls = {"api": dict(FAPI_URLS)}
        self.orders = []

    def enable_demo_trading(self, enabled):
        assert enabled is True

    def create_order(self, *args):
        self.orders.append(args)
        return {"id": "1", "status": "closed", "filled": args[3], "average": 100}

    def amount_to_precision(self, symbol, amount):
        return str(amount)

    def price_to_precision(self, symbol, price):
        return str(price)

    def market(self, symbol):
        return {"id": symbol.replace("/", "").replace(":USDT", "")}


def test_open_without_stop_is_rejected_before_order_submission():
    from types import SimpleNamespace

    exchange = OrderSpy()
    service = DemoExecutionService(exchange, SimpleNamespace())
    intent = OrderIntent("ADA/USDT:USDT", Side.LONG, Action.OPEN, 1, 100, "reason", Regime.LOW, 99)
    with pytest.raises(ValueError, match="protective stop"):
        service.open(intent, stop_price=None)
    assert exchange.orders == []


def test_market_close_always_uses_reduce_only_and_demo_host():
    from types import SimpleNamespace

    exchange = OrderSpy()
    service = DemoExecutionService(exchange, SimpleNamespace())
    fill = service._market_fill("ADA/USDT:USDT", "sell", 1, reduce_only=True)
    assert fill["id"] == "1"
    assert exchange.orders[0][-1]["reduceOnly"] is True
    assert service.host == "https://demo-fapi.binance.com"


def test_market_fill_records_verified_fee_realized_pnl_and_exchange_time():
    from types import SimpleNamespace

    class FillClient(OrderSpy):
        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, params):
            assert params == {"symbol": "ADAUSDT", "orderId": "1", "limit": 1000}
            return [{"orderId": "1", "qty": "1", "commission": "0.04",
                     "commissionAsset": "USDT", "realizedPnl": "2.5",
                     "time": 1791000000000}]

    service = DemoExecutionService(FillClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.1, 0.1, 0.0)
    fill = service._market_fill("ADA/USDT:USDT", "sell", 1, reduce_only=True)

    assert fill["trade_data_verified"] is True
    assert fill["fee_cost"] == pytest.approx(0.04)
    assert fill["realized_pnl"] == pytest.approx(2.5)
    assert fill["closed_at"] == "2026-10-03T04:00:00+00:00"


def test_demo_funding_fee_requires_complete_usdt_income_rows():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 10, 2, tzinfo=timezone.utc)

    class IncomeClient(OrderSpy):
        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetIncome(self, params):
            assert params["symbol"] == "ADAUSDT"
            assert params["incomeType"] == "FUNDING_FEE"
            return [{"incomeType": "FUNDING_FEE", "asset": "USDT", "income": "-0.12",
                     "time": 1790870400000}]

    service = DemoExecutionService(IncomeClient(), SimpleNamespace())
    assert service.funding_fee("ADA/USDT:USDT", start, end) == pytest.approx(-0.12)


def test_orphan_bot_stop_cleanup_cancels_only_flat_symbols():
    from types import SimpleNamespace

    class AlgoClient(OrderSpy):
        def __init__(self):
            super().__init__()
            self.markets = {
                "AAA/USDT:USDT": {"id": "AAAUSDT", "symbol": "AAA/USDT:USDT"},
                "BBB/USDT:USDT": {"id": "BBBUSDT", "symbol": "BBB/USDT:USDT"},
            }
            self.rows = [
                {"symbol": "AAAUSDT", "algoId": "orphan", "clientAlgoId": "mrb_s_old",
                 "algoStatus": "NEW", "orderType": "STOP_MARKET", "reduceOnly": True},
                {"symbol": "BBBUSDT", "algoId": "active", "clientAlgoId": "mrb_s_live",
                 "algoStatus": "NEW", "orderType": "STOP_MARKET", "reduceOnly": True},
                {"symbol": "AAAUSDT", "algoId": "foreign", "clientAlgoId": "manual",
                 "algoStatus": "NEW", "orderType": "STOP_MARKET", "reduceOnly": True},
            ]
            self.cancelled = []

        def fapiPrivateGetOpenAlgoOrders(self, params):
            assert params == {}
            return list(self.rows)

        def fapiPrivateDeleteAlgoOrder(self, params):
            self.cancelled.append(params["algoId"])
            self.rows = [row for row in self.rows if str(row["algoId"]) != params["algoId"]]

    exchange = AlgoClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    cancelled = service.cancel_orphaned_bot_stops({"BBB/USDT:USDT"})

    assert cancelled == [{"symbol": "AAA/USDT:USDT", "algo_id": "orphan"}]
    assert exchange.cancelled == ["orphan"]
    assert {row["algoId"] for row in exchange.rows} == {"active", "foreign"}


def test_resolve_entry_fee_requires_one_quantity_price_and_side_match():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    opened_at = datetime(2026, 10, 1, tzinfo=timezone.utc)

    class EntryClient(OrderSpy):
        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, params):
            assert params["symbol"] == "ADAUSDT"
            return [{"orderId": "entry-1", "side": "BUY", "qty": "2", "price": "100",
                     "commission": "0.2", "commissionAsset": "USDT",
                     "time": int((opened_at.timestamp() + 30) * 1000)}]

    service = DemoExecutionService(EntryClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.1, 0.1, 0.0)
    assert service.resolve_entry_fee("ADA/USDT:USDT", Side.LONG, 2.0, 100.0,
                                     opened_at) == pytest.approx(0.2)
    assert service.resolve_entry_fee("ADA/USDT:USDT", Side.SHORT, 2.0, 100.0,
                                     opened_at) is None


def test_readonly_active_stop_audit_checks_saved_order_against_actual_position(monkeypatch):
    from types import SimpleNamespace

    class AuditClient(OrderSpy):
        def __init__(self):
            super().__init__()
            self.open_algo_calls = []
            self.rows = [{
                "algoId": 123, "clientAlgoId": "mrb_s_test", "symbol": "ADAUSDT",
                "orderType": "STOP_MARKET",
                "algoStatus": "NEW", "side": "SELL", "quantity": "2.0",
                "triggerPrice": "95.0", "reduceOnly": True,
            }]

        def market(self, symbol):
            return {"id": symbol.replace("/", "").replace(":USDT", "")}

        def fapiPrivateGetOpenAlgoOrders(self, params):
            self.open_algo_calls.append(params)
            return self.rows

        def price_to_precision(self, symbol, price):
            return str(price)

    exchange = AuditClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (0.1, 0.1, 0.0))
    position = SimpleNamespace(side=Side.LONG, quantity=2.0)
    tracked = {"ADA/USDT:USDT": {"stop": {"id": "123", "stop_price": 95.0}}}

    assert service.audit_active_protective_stops({"ADA/USDT:USDT": position}, tracked) == []
    assert exchange.open_algo_calls == [{"symbol": "ADAUSDT"}]
    assert exchange.orders == []

    exchange.rows[0]["quantity"] = "1.0"
    assert service.audit_active_protective_stops({"ADA/USDT:USDT": position}, tracked) == [
        "ADA/USDT:USDT: saved protective stop is not active at the saved side/quantity/price"
    ]


def test_demo_stop_replacement_places_new_stop_before_canceling_old(monkeypatch):
    from types import SimpleNamespace

    exchange = OrderSpy()
    service = DemoExecutionService(exchange, SimpleNamespace())
    exchange.fapiPrivateGetAlgoOrder = lambda params: {
        "algoId": params["algoId"], "clientAlgoId": "mrb_s_old", "symbol": "ADAUSDT",
        "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL",
        "quantity": "2", "triggerPrice": "95", "reduceOnly": True,
    }
    calls = []
    monkeypatch.setattr(service, "_position", lambda _symbol: {"side": "long", "quantity": 2.0})
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (0.1, 0.1, 0.0))
    monkeypatch.setattr(service, "_place_stop", lambda symbol, side, amount, price:
                        calls.append(("place", amount, price)) or
                        {"id": "new-stop", "stop_price": price})
    monkeypatch.setattr(service, "_cancel_stop", lambda symbol, stop:
                        calls.append(("cancel", stop["id"])))

    result = service.replace_stop(
        SimpleNamespace(symbol="ADA/USDT:USDT", side=Side.LONG),
        stop={"id": "old-stop", "stop_price": 95.0}, stop_price=98.0,
    )

    assert result == {"id": "new-stop", "stop_price": 98.0}
    assert calls == [("place", 2.0, 98.0), ("cancel", "old-stop")]


@pytest.mark.parametrize(
    ("position_side", "order_side", "old_price", "requested"),
    [(Side.LONG, "SELL", 95.0, 94.9), (Side.SHORT, "BUY", 105.0, 105.1)],
)
def test_demo_stop_replacement_rejects_less_protective_prices(
    monkeypatch, position_side, order_side, old_price, requested,
):
    from types import SimpleNamespace

    exchange = OrderSpy()
    service = DemoExecutionService(exchange, SimpleNamespace())
    exchange.fapiPrivateGetAlgoOrder = lambda params: {
        "algoId": params["algoId"], "clientAlgoId": "mrb_s_old", "symbol": "ADAUSDT",
        "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": order_side,
        "quantity": "2", "triggerPrice": str(old_price), "reduceOnly": True,
    }
    calls = []
    monkeypatch.setattr(service, "_position", lambda _symbol: {
        "side": position_side.value, "quantity": 2.0,
    })
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (0.1, 0.1, 0.0))
    monkeypatch.setattr(service, "_place_stop", lambda *args: calls.append(("place", args)) or {})
    monkeypatch.setattr(service, "_cancel_stop", lambda *args: calls.append(("cancel", args)))

    with pytest.raises(ValueError, match="refusing to loosen"):
        service.replace_stop(
            SimpleNamespace(symbol="ADA/USDT:USDT", side=position_side),
            stop={"id": "old-stop", "stop_price": old_price}, stop_price=requested,
        )

    assert calls == []


def test_expired_partial_market_fill_is_registered_and_protected_at_actual_quantity(monkeypatch):
    from types import SimpleNamespace

    class PartialFillClient(OrderSpy):
        def create_order(self, *args):
            self.orders.append(args)
            return {"id": "331508428", "status": "expired", "filled": 14272,
                    "average": 0.01073}

        def price_to_precision(self, symbol, price):
            return str(price)

    exchange = PartialFillClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    positions = iter([None, {"side": "long", "quantity": 14272.0, "entry_price": 0.01073}])
    monkeypatch.setattr(service, "_position", lambda _symbol: next(positions))
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda _symbol: None)
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (1.0, 1.0, 5.0))
    placed_stops = []
    monkeypatch.setattr(service, "_place_stop", lambda symbol, side, amount, price:
                        placed_stops.append((symbol, side, amount, price)) or
                        {"id": "stop-1", "stop_price": price, "amount": amount})

    intent = OrderIntent("US/USDT:USDT", Side.LONG, Action.OPEN, 21658.0, 0.010754,
                         "low_zone_1h_bearish_close", Regime.LOW, 0.009134)
    result = service.open(intent, stop_price=0.009134)

    assert result["fill"]["id"] == "331508428"
    assert result["fill"]["status"] == "expired"
    assert result["fill"]["filled"] == 14272.0
    assert result["position"]["quantity"] == 14272.0
    assert placed_stops == [("US/USDT:USDT", "long", 14272.0, 0.009134)]


def test_demo_partial_reduction_installs_remaining_stop_before_reduce_only_fill(monkeypatch):
    from types import SimpleNamespace

    exchange = OrderSpy()
    service = DemoExecutionService(exchange, SimpleNamespace())
    calls = []
    positions = iter([
        {"side": "long", "quantity": 10.0},
        {"side": "long", "quantity": 6.0},
    ])
    monkeypatch.setattr(service, "_position", lambda _symbol: next(positions))
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (1.0, 1.0, 0.0))
    monkeypatch.setattr(service, "_place_stop", lambda symbol, side, amount, price:
                        calls.append(("place_stop", amount, price)) or {"id": "new-stop", "stop_price": price})
    monkeypatch.setattr(service, "_market_fill", lambda symbol, side, amount, *, reduce_only:
                        calls.append(("market", amount, reduce_only)) or {"id": "reduce", "filled": amount, "average": 101.0})
    monkeypatch.setattr(service, "_retire_superseded_stops",
                        lambda symbol, side, stop, quantity:
                        calls.append(("retire_stops", stop["id"], quantity)) or [])

    result = service.reduce(
        SimpleNamespace(symbol="ADA/USDT:USDT", side=Side.LONG), 4.0,
        stop={"id": "old-stop", "stop_price": 95.0}, reason="take_profit_1",
    )

    assert result["position"]["quantity"] == 6.0
    assert calls == [
        ("place_stop", 6.0, 95.0),
        ("market", 4.0, True),
        ("retire_stops", "new-stop", 6.0),
    ]


def test_demo_partial_fill_is_returned_when_superseded_stop_cancel_fails(monkeypatch):
    from types import SimpleNamespace

    class StopClient(OrderSpy):
        def __init__(self):
            super().__init__()
            self.active = [
                {"algoId": "old-stop", "clientAlgoId": "mrb_s_old", "orderType": "STOP_MARKET",
                 "side": "SELL", "quantity": "10", "algoStatus": "NEW", "reduceOnly": True},
            ]
            self.fail_cancel = True

        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetOpenAlgoOrders(self, params):
            assert params == {"symbol": "ADAUSDT"}
            return list(self.active)

        def fapiPrivateDeleteAlgoOrder(self, params):
            if self.fail_cancel:
                raise RuntimeError("simulated stale-stop cancellation failure")
            self.active = [row for row in self.active if str(row["algoId"]) != str(params["algoId"])]

    exchange = StopClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    positions = iter([{"side": "long", "quantity": 10.0}, {"side": "long", "quantity": 6.0}])
    monkeypatch.setattr(service, "_position", lambda _symbol: next(positions))
    monkeypatch.setattr(service, "_market_rules", lambda _symbol: (1.0, 1.0, 0.0))

    def place_stop(symbol, side, amount, price):
        row = {"algoId": "new-stop", "clientAlgoId": "mrb_s_new", "orderType": "STOP_MARKET",
               "side": "SELL", "quantity": str(amount), "triggerPrice": str(price),
               "algoStatus": "NEW", "reduceOnly": True}
        exchange.active.append(row)
        return {"id": "new-stop", "clientOrderId": "mrb_s_new", "stop_price": price,
                "amount": amount, "info": row}

    monkeypatch.setattr(service, "_place_stop", place_stop)
    monkeypatch.setattr(service, "_market_fill", lambda symbol, side, amount, *, reduce_only:
                        {"id": "reduce-1", "filled": amount, "average": 101.0})

    result = service.reduce(
        SimpleNamespace(symbol="ADA/USDT:USDT", side=Side.LONG), 4.0,
        stop={"id": "already-canceled-stop", "stop_price": 95.0}, reason="take_profit_1",
    )

    assert result["fill"]["id"] == "reduce-1"
    assert result["position"]["quantity"] == 6.0
    assert result["stop_cleanup_error"] is None
    assert [row["algoId"] for row in result["stale_stops"]] == ["old-stop"]

    exchange.fail_cancel = False
    remaining = service.reconcile_reduced_position_stops(
        "ADA/USDT:USDT", "long", result["stop"], 6.0,
    )
    assert remaining == []
    assert [row["algoId"] for row in exchange.active] == ["new-stop"]


def test_fill_audit_warns_on_non_robot_order_ids_for_tracked_symbols():
    from types import SimpleNamespace

    class TradeClient(OrderSpy):
        def fetch_my_trades(self, symbol, since=None, limit=None):
            assert symbol == "ADA/USDT:USDT"
            assert limit == 1000
            return [{"order": "robot-1"}, {"order": "manual-2"}]

    service = DemoExecutionService(TradeClient(), SimpleNamespace())
    warnings = service.audit_recent_fills({"ADA/USDT:USDT"}, {"robot-1"}, set())
    assert warnings == [{"symbol": "ADA/USDT:USDT", "orderId": "manual-2"}]


def test_triggered_bot_stop_fill_is_verified_from_algo_and_user_trades():
    from types import SimpleNamespace

    class StopClient(OrderSpy):
        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetAlgoOrder(self, params):
            assert params == {"algoId": "algo-1"}
            return {"actualOrderId": "close-1", "clientAlgoId": "mrb_s_abc",
                    "reduceOnly": True, "orderType": "STOP_MARKET"}

        def fapiPrivateGetUserTrades(self, params):
            assert params == {"symbol": "ADAUSDT", "orderId": "close-1", "limit": 1000}
            return [{"orderId": "close-1", "qty": "2", "price": "95", "commission": "0.1",
                     "commissionAsset": "USDT", "realizedPnl": "-10", "time": 1791000000000}]

    service = DemoExecutionService(StopClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.1, 0.1, 0.0)
    result = service.resolve_protective_stop_fill(
        "ADA/USDT:USDT", "close-1", {"id": "algo-1"}, 2.0,
    )
    assert result["quantity"] == 2.0
    assert result["average"] == 95.0
    assert result["gross_pnl"] == -10.0
    assert result["exit_fee"] == 0.1
    assert result["order_id"] == "close-1"


def test_exchange_flat_close_fill_aggregates_exact_opposite_side_trades():
    from datetime import datetime, timedelta, timezone

    opened_at = datetime.now(timezone.utc) - timedelta(hours=2)
    observed_flat_at = opened_at + timedelta(hours=1)
    closed_ms = int((opened_at + timedelta(minutes=50)).timestamp() * 1000)
    rows = [
        {"id": 10, "orderId": 20, "symbol": "ADAUSDT", "side": "SELL",
         "positionSide": "BOTH", "qty": "2", "price": "100", "time": closed_ms - 1_000,
         "commission": "0.2", "commissionAsset": "USDT", "realizedPnl": "0"},
        {"id": 11, "orderId": 21, "symbol": "ADAUSDT", "side": "BUY",
         "positionSide": "BOTH", "qty": "0.5", "price": "90", "time": closed_ms,
         "commission": "-0.01", "commissionAsset": "USDT", "realizedPnl": "5"},
        {"id": 12, "orderId": 21, "symbol": "ADAUSDT", "side": "BUY",
         "positionSide": "BOTH", "qty": "1.5", "price": "92", "time": closed_ms + 1_000,
         "commission": "0.03", "commissionAsset": "USDT", "realizedPnl": "12"},
    ]

    class TradeClient(OrderSpy):
        def market(self, symbol):
            assert symbol == "ADA/USDT:USDT"
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, params):
            assert params == {
                "symbol": "ADAUSDT", "startTime": int(opened_at.timestamp() * 1000),
                "endTime": int(observed_flat_at.timestamp() * 1000), "limit": 1000,
            }
            return rows

    service = DemoExecutionService(TradeClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.1, 0.1, 0.0)

    result = service.resolve_exchange_flat_close_fill(
        "ADA/USDT:USDT", Side.SHORT, opened_at, observed_flat_at, 2.0,
    )

    assert result == {
        "order_ids": ["21"], "quantity": 2.0, "average": 91.5,
        "gross_pnl": 17.0, "exit_fee": pytest.approx(0.02),
        "closed_at": datetime.fromtimestamp((closed_ms + 1_000) / 1000, timezone.utc).isoformat(),
    }


@pytest.mark.parametrize("invalid_row", [
    {"id": 11, "orderId": 21, "symbol": "ADAUSDT", "side": "BUY", "positionSide": "BOTH",
     "qty": "1.9", "price": "90", "commission": "0.1", "commissionAsset": "USDT",
     "realizedPnl": "5", "time": 1_800_000_000_000},
    {"id": 11, "orderId": 21, "symbol": "ADAUSDT", "side": "BUY", "positionSide": "BOTH",
     "qty": "2", "price": "90", "commission": "0.1", "commissionAsset": "BNB",
     "realizedPnl": "5", "time": 1_800_000_000_000},
])
def test_exchange_flat_close_fill_keeps_unmatched_quantity_or_fee_unverified(invalid_row):
    from datetime import datetime, timedelta, timezone

    opened_at = datetime.now(timezone.utc) - timedelta(hours=2)
    observed_flat_at = opened_at + timedelta(hours=1)

    class TradeClient(OrderSpy):
        def market(self, _symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, _params):
            invalid_row["time"] = int((opened_at + timedelta(minutes=30)).timestamp() * 1000)
            return [invalid_row]

    service = DemoExecutionService(TradeClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.1, 0.1, 0.0)

    assert service.resolve_exchange_flat_close_fill(
        "ADA/USDT:USDT", Side.SHORT, opened_at, observed_flat_at, 2.0,
    ) is None


def test_unverified_or_incomplete_stop_trade_is_not_recorded():
    from types import SimpleNamespace

    class StopClient(OrderSpy):
        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetAlgoOrder(self, params):
            return {"actualOrderId": "close-1", "clientAlgoId": "manual", "reduceOnly": True,
                    "orderType": "STOP_MARKET"}

    service = DemoExecutionService(StopClient(), SimpleNamespace())
    assert service.resolve_protective_stop_fill(
        "ADA/USDT:USDT", "close-1", {"id": "algo-1"}, 2.0,
    ) is None


def test_manual_position_reconstruction_replays_adds_and_partial_closes():
    from datetime import datetime, timedelta, timezone

    opened = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = [
        {"id": 1, "symbol": "ADAUSDT", "time": int(opened.timestamp() * 1000),
         "side": "BUY", "positionSide": "BOTH", "qty": "2", "price": "100",
         "commission": "0.2", "commissionAsset": "USDT"},
        {"id": 2, "symbol": "ADAUSDT", "time": int((opened + timedelta(hours=1)).timestamp() * 1000),
         "side": "BUY", "positionSide": "BOTH", "qty": "1", "price": "110",
         "commission": "0.11", "commissionAsset": "USDT"},
        {"id": 3, "symbol": "ADAUSDT", "time": int((opened + timedelta(hours=2)).timestamp() * 1000),
         "side": "SELL", "positionSide": "BOTH", "qty": "2", "price": "120",
         "commission": "0.24", "commissionAsset": "USDT"},
    ]

    class TradeClient(OrderSpy):
        def market(self, _symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, params):
            assert params["symbol"] == "ADAUSDT" and params["limit"] == 1000
            return rows

    service = DemoExecutionService(TradeClient(), SimpleNamespace())
    service._market_rules = lambda _symbol: (0.01, 0.01, 0.0)
    current = opened + timedelta(hours=3)
    result = service.reconstruct_open_position(
        "ADA/USDT:USDT",
        {"side": "long", "quantity": 1.0, "entry_price": 310 / 3},
        at=current,
    )

    assert result["side"] == "long"
    assert result["quantity"] == 1.0
    assert result["entry_price"] == pytest.approx(310 / 3)
    assert result["entry_fee"] == pytest.approx(0.31 / 3)
    assert result["opened_at"] == opened


def test_manual_position_reconstruction_rejects_unsupported_commission_currency():
    from datetime import datetime, timezone

    class TradeClient(OrderSpy):
        def market(self, _symbol):
            return {"id": "ADAUSDT"}

        def fapiPrivateGetUserTrades(self, _params):
            return [{"id": 1, "symbol": "ADAUSDT", "time": 1_791_000_000_000,
                     "side": "BUY", "positionSide": "BOTH", "qty": "1", "price": "100",
                     "commission": "0.001", "commissionAsset": "BNB"}]

    service = DemoExecutionService(TradeClient(), SimpleNamespace())
    with pytest.raises(RuntimeError, match="unsupported or invalid"):
        service.reconstruct_open_position(
            "ADA/USDT:USDT", {"side": "long", "quantity": 1, "entry_price": 100},
            at=datetime(2026, 10, 4, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(("existing_stop", "strategy_stop", "expected_stop", "external"), [
    (95.0, 90.0, 95.0, True),
    (90.0, 95.0, 95.0, False),
])
def test_manual_position_adoption_preserves_tighter_stop_or_tightens_it(
        existing_stop, strategy_stop, expected_stop, external):
    from datetime import datetime, timezone
    from bot.portfolio import Position

    class StopClient(OrderSpy):
        def __init__(self):
            super().__init__()
            self.active = [{
                "algoId": "manual-stop", "clientAlgoId": "manual-client-id",
                "symbol": "ADAUSDT", "orderType": "STOP_MARKET", "algoStatus": "NEW",
                "side": "SELL", "quantity": "2", "triggerPrice": str(existing_stop),
                "reduceOnly": True,
            }]

        def market(self, _symbol):
            return {"id": "ADAUSDT"}

        def price_to_precision(self, _symbol, price):
            return str(round(float(price), 2))

        def fapiPrivateGetOpenAlgoOrders(self, _params):
            return list(self.active)

        def fapiPrivateDeleteAlgoOrder(self, params):
            self.active = [row for row in self.active if row["algoId"] != params["algoId"]]

    exchange = StopClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    service._market_rules = lambda _symbol: (0.01, 0.01, 0.0)
    service._position = lambda _symbol: {"side": "long", "quantity": 2.0, "entry_price": 100.0}

    def place(symbol, side, amount, stop_price):
        row = {"algoId": "bot-stop", "clientAlgoId": "mrb_s_new", "symbol": "ADAUSDT",
               "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL",
               "quantity": str(amount), "triggerPrice": str(stop_price), "reduceOnly": True}
        exchange.active.append(row)
        return {"id": "bot-stop", "clientOrderId": "mrb_s_new", "stop_price": stop_price,
                "amount": amount, "info": row}

    service._place_stop = place
    position = Position("ADA/USDT:USDT", Side.LONG, 2.0, 100.0, strategy_stop,
                        datetime(2026, 10, 1, tzinfo=timezone.utc),
                        "manual_position_adopted", Regime.LOW.value)
    saved = service.adopt_manual_position_stop(
        position, strategy_stop=strategy_stop, market_price=100.0,
    )

    assert saved["stop_price"] == expected_stop
    assert saved["external"] is external
    assert len(exchange.active) == 1
    assert exchange.active[0]["algoId"] == saved["id"]


def test_trailing_stop_replacement_validates_and_retires_adopted_manual_stop():
    from datetime import datetime, timezone
    from bot.portfolio import Position

    class StopClient(OrderSpy):
        def __init__(self):
            super().__init__()
            self.active = [{
                "algoId": "manual-stop", "clientAlgoId": "manual-client-id",
                "symbol": "ADAUSDT", "orderType": "STOP_MARKET", "algoStatus": "NEW",
                "side": "SELL", "quantity": "2", "triggerPrice": "95", "reduceOnly": True,
            }]

        def market(self, _symbol):
            return {"id": "ADAUSDT"}

        def price_to_precision(self, _symbol, price):
            return str(round(float(price), 2))

        def fapiPrivateGetAlgoOrder(self, params):
            return next(row for row in self.active if str(row["algoId"]) == params["algoId"])

        def fapiPrivateGetOpenAlgoOrders(self, _params):
            return list(self.active)

        def fapiPrivateDeleteAlgoOrder(self, params):
            self.active = [row for row in self.active if row["algoId"] != params["algoId"]]

    exchange = StopClient()
    service = DemoExecutionService(exchange, SimpleNamespace())
    service._market_rules = lambda _symbol: (0.01, 0.01, 0.0)
    service._position = lambda _symbol: {"side": "long", "quantity": 2.0, "entry_price": 100.0}

    def place(symbol, side, amount, stop_price):
        row = {"algoId": "bot-stop", "clientAlgoId": "mrb_s_new", "symbol": "ADAUSDT",
               "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL",
               "quantity": str(amount), "triggerPrice": str(stop_price), "reduceOnly": True}
        exchange.active.append(row)
        return {"id": "bot-stop", "clientOrderId": "mrb_s_new", "stop_price": stop_price,
                "amount": amount, "info": row}

    service._place_stop = place
    position = Position("ADA/USDT:USDT", Side.LONG, 2.0, 100.0, 95.0,
                        datetime(2026, 10, 1, tzinfo=timezone.utc),
                        "manual_position_adopted", Regime.LOW.value)
    saved = {"id": "manual-stop", "clientOrderId": "manual-client-id", "stop_price": 95.0,
             "amount": 2.0, "external": True, "order_side": "SELL"}

    updated = service.replace_stop(position, stop=saved, stop_price=98.0)

    assert updated["id"] == "bot-stop"
    assert updated["stop_price"] == 98.0
    assert [row["algoId"] for row in exchange.active] == ["bot-stop"]


def test_failed_demo_entry_never_creates_local_paper_position():
    from datetime import datetime, timezone

    settings = __import__("bot.config", fromlist=["load_settings"]).load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }

    class FailedExecution:
        host = "https://demo-fapi.binance.com"

        def open(self, intent, *, stop_price):
            raise RuntimeError("Demo order rejected")

    PaperTrader(account, settings, execution=FailedExecution())._execute_pending(100, at, 0.0)
    assert account.position is None
    assert not any(event.get("event") == "paper_fill" and event.get("action") == "OPEN"
                   for event in account.events)
