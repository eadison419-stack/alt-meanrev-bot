from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.config import load_settings
from bot.execution import DemoExecutionService
from bot.live.manual_positions import replay_position_changes
from bot.live.multi_paper import MultiPaperState
from bot.portfolio import Position
from bot.signals import Side
from scripts import run_mr_candidate_paper as runner


OPENED = datetime(2026, 11, 12, 3, tzinfo=timezone.utc)
SYMBOL = "ADA/USDT:USDT"


def position(side=Side.LONG, quantity=2, price=100, stop=95):
    return Position(SYMBOL, side, quantity, price, stop, OPENED, "manual_position_adopted", "MID")


def trade(number, side, quantity, price, fee=None):
    return (int((OPENED + timedelta(minutes=number - 1)).timestamp() * 1000),
            number, side, quantity, price, quantity * price * .001 if fee is None else fee)


def expected(side="long", quantity=2, price=100):
    return {"side": side, "quantity": quantity, "entry_price": price}


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_replay_add_reduce_and_fee_allocation(side):
    buy, sell = ("BUY", "SELL") if side is Side.LONG else ("SELL", "BUY")
    old = position(side, stop=95 if side is Side.LONG else 105)
    rows = [trade(1, buy, 2, 100), trade(2, buy, 1, 110), trade(3, sell, 1, 120)]
    plan = replay_position_changes(old, rows, expected(side.value, 2, 310 / 3), entry_fee=.2)
    assert plan["remaining"]["entry_price"] == pytest.approx(310 / 3)
    assert plan["remaining"]["entry_fee"] == pytest.approx(.31 * 2 / 3)
    assert plan["remaining"]["opened_at"] == OPENED
    assert plan["realized"]["gross_pnl"] == pytest.approx((120 - 310 / 3) * (1 if side is Side.LONG else -1))
    assert plan["realized"]["fees"] == pytest.approx(.31 / 3 + .12)
    assert [row["action"] for row in plan["changes"]] == ["add", "reduce"]
    retry = replay_position_changes(
        replace(old, quantity=2, entry_price=310 / 3), rows, expected(side.value, 2, 310 / 3),
        cursor=plan["cursor"], entry_fee=plan["remaining"]["entry_fee"],
    )
    assert retry["changes"] == []


def test_same_quantity_and_average_round_trip_is_not_lost():
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110), trade(3, "BUY", 1, 100)]
    plan = replay_position_changes(position(), rows, expected(), entry_fee=.2)
    assert plan["realized"]["gross_pnl"] == 10
    assert len(plan["changes"]) == 2
    assert plan["remaining"]["entry_fee"] == pytest.approx(.2)


def test_fill_cursor_distinguishes_partial_fills_in_the_same_millisecond():
    first = trade(1, "BUY", 2, 100)
    second = (first[0], 2, "BUY", 1, 110, .11)
    plan = replay_position_changes(position(), [first, second], expected(quantity=3, price=310 / 3),
                                   cursor={"time": first[0], "id": 1}, entry_fee=.2)
    assert len(plan["changes"]) == 1
    assert plan["cursor"] == {"time": first[0], "id": 2}


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_single_fill_reversal_splits_old_close_and_new_open(side):
    first, reverse = ("BUY", "SELL") if side is Side.LONG else ("SELL", "BUY")
    old = position(side, stop=95 if side is Side.LONG else 105)
    rows = [trade(1, first, 2, 100), trade(2, reverse, 3, 110)]
    plan = replay_position_changes(old, rows, expected("short" if side is Side.LONG else "long", 1, 110))
    closed = plan["closed_cycles"][0]
    assert closed["quantity"] == 2
    assert closed["fees"] == pytest.approx(.2 + .22)
    assert closed["gross_pnl"] == (20 if side is Side.LONG else -20)
    assert plan["remaining"]["entry_fee"] == pytest.approx(.11)
    assert plan["remaining"]["opened_at"] == OPENED + timedelta(minutes=1)


def test_multiple_reversals_and_full_close_between_snapshots():
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 5, 110), trade(3, "BUY", 4, 105)]
    plan = replay_position_changes(position(), rows, expected("long", 1, 105))
    assert [cycle["side"] for cycle in plan["closed_cycles"]] == ["long", "short"]
    assert [cycle["gross_pnl"] for cycle in plan["closed_cycles"]] == [20, 15]
    assert plan["remaining"]["entry_fee"] == pytest.approx(.105)
    flat = replay_position_changes(position(), rows[:2] + [trade(3, "BUY", 3, 105)], None)
    assert flat["remaining"] is None
    assert len(flat["closed_cycles"]) == 2


def test_flat_then_same_side_reopen_is_a_new_cycle():
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 2, 110), trade(3, "BUY", 2, 100)]
    plan = replay_position_changes(position(), rows, expected())
    assert len(plan["closed_cycles"]) == 1
    assert plan["remaining"]["opened_at"] == OPENED + timedelta(minutes=2)


def test_replay_fails_closed_on_missing_anchor_cursor_or_unmatched_fills():
    with pytest.raises(RuntimeError, match="original_position"):
        replay_position_changes(position(), [trade(2, "BUY", 1, 110)], expected())
    with pytest.raises(RuntimeError, match="cursor"):
        replay_position_changes(position(), [trade(2, "BUY", 1, 110)], expected(),
                                cursor={"time": 1, "id": 1}, entry_fee=.2)
    with pytest.raises(RuntimeError, match="do_not_match"):
        replay_position_changes(position(), [trade(1, "BUY", 2, 100)], expected(quantity=3))


def stop_service(new_position):
    class Exchange:
        def __init__(self):
            self.active = [{"algoId": "old", "clientAlgoId": "mrb_s_old", "symbol": "ADAUSDT",
                            "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL",
                            "quantity": "2", "triggerPrice": "95", "reduceOnly": True}]
            self.calls = []
            self.fail_cancel = False

        def market(self, symbol):
            return {"id": "ADAUSDT"}

        def price_to_precision(self, symbol, price):
            return str(price)

        def fapiPrivateGetOpenAlgoOrders(self, params):
            return [dict(row) for row in self.active]

        def fapiPrivateDeleteAlgoOrder(self, params):
            self.calls.append(("cancel", params["algoId"]))
            if self.fail_cancel:
                raise RuntimeError("cancellation unavailable")
            self.active = [row for row in self.active if row["algoId"] != params["algoId"]]

    service = object.__new__(DemoExecutionService)
    service.exchange = Exchange()
    service._verify_urls = lambda: None
    service._market_rules = lambda symbol: (.01, .01, 0)
    service._position = lambda symbol: expected(new_position.side.value, new_position.quantity, new_position.entry_price)

    def place(symbol, side, amount, price):
        assert any(row["algoId"] == "old" for row in service.exchange.active)
        service.exchange.calls.append(("place", amount, price))
        row = {"algoId": "new", "clientAlgoId": "mrb_s_new", "symbol": "ADAUSDT",
               "orderType": "STOP_MARKET", "algoStatus": "NEW", "side": "SELL" if side == "long" else "BUY",
               "quantity": str(amount), "triggerPrice": str(price), "reduceOnly": True}
        service.exchange.active.append(row)
        return {"id": "new", "clientOrderId": "mrb_s_new", "stop_price": price, "amount": amount}

    service._place_stop = place
    old_stop = {"id": "old", "stop_price": 95, "amount": 2, "order_side": "SELL"}
    return service, old_stop


@pytest.mark.parametrize("quantity", [1, 3])
def test_resize_stop_protects_actual_total_before_canceling_original(quantity):
    updated = replace(position(), quantity=quantity)
    service, stop = stop_service(updated)
    result = service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert result["amount"] == quantity
    assert service.exchange.calls == [("place", quantity, 95), ("cancel", "old")]
    assert [row["algoId"] for row in service.exchange.active] == ["new"]


def test_reversal_replaces_stop_direction():
    updated = position(Side.SHORT, 1, 110, 115)
    service, stop = stop_service(updated)
    service.reconcile_manual_position_stop(updated, position(), stop, stop_price=115, market_price=110)
    assert service.exchange.active[0]["side"] == "BUY"
    assert service.exchange.active[0]["quantity"] == "1"


def test_flat_then_reopen_same_direction_uses_new_cycle_stop():
    new = replace(position(), opened_at=OPENED + timedelta(minutes=1), stop_price=90)
    service, stop = stop_service(new)
    result = service.reconcile_manual_position_stop(
        new, position(), stop, stop_price=90, market_price=100, new_cycle=True)
    assert result["stop_price"] == 90


def test_stop_retry_reuses_verified_replacement_after_cancel_failure():
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    service.exchange.fail_cancel = True
    with pytest.raises(RuntimeError, match="cancellation"):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert len(service.exchange.active) == 2
    service.exchange.fail_cancel = False
    service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert sum(call[0] == "place" for call in service.exchange.calls) == 1
    assert len(service.exchange.active) == 1


def test_further_add_during_retry_retires_intermediate_bot_stop():
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    service.exchange.fail_cancel = True
    with pytest.raises(RuntimeError):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    # The user adds again before the next retry. The earlier replacement is
    # still protective for three units; a new four-unit stop must come first.
    service.exchange.active[-1]["algoId"] = "intermediate"
    service.exchange.fail_cancel = False
    updated = replace(position(), quantity=4)
    service._position = lambda symbol: expected(quantity=4)
    service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert [row["algoId"] for row in service.exchange.active] == ["new"]
    assert service.exchange.active[0]["quantity"] == "4"


def test_missing_new_stop_confirmation_never_cancels_the_old_stop():
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    service._place_stop = lambda *args: {"id": "new", "stop_price": 95, "amount": 3}
    with pytest.raises(RuntimeError, match="replacement_stop_not_verified"):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert [row["algoId"] for row in service.exchange.active] == ["old"]
    assert not service.exchange.calls


def test_untracked_external_stop_is_not_canceled_during_resize():
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    service.exchange.active.append({**service.exchange.active[0], "algoId": "foreign",
                                    "clientAlgoId": "another-system"})
    with pytest.raises(RuntimeError, match="ambiguous"):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert len(service.exchange.active) == 2 and not service.exchange.calls


def test_strategy_close_cannot_submit_against_a_stale_quantity():
    service, stop = stop_service(replace(position(), quantity=3))
    service._close_orders = {}
    service._close_checkpoint = None
    service._market_fill = lambda *args, **kwargs: pytest.fail("must reconcile before submitting a close")
    with pytest.raises(RuntimeError, match="required_before_strategy_close"):
        service.close(position(), stop=stop)


def test_adoption_cannot_hide_a_missing_fill_of_one_quantity_step():
    service, _ = stop_service(position())
    service._manual_position_trades = lambda *args, **kwargs: [trade(1, "BUY", 2, 100)]
    with pytest.raises(RuntimeError, match="do not match"):
        service.reconstruct_open_position(SYMBOL, expected(quantity=2.01), at=OPENED)


def test_unverified_replacement_or_concurrent_change_keeps_old_stop():
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    place = service._place_stop

    def concurrent(*args):
        result = place(*args)
        service._position = lambda symbol: expected(quantity=4)
        return result

    service._place_stop = concurrent
    with pytest.raises(RuntimeError, match="changed_during"):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=95, market_price=100)
    assert any(row["algoId"] == "old" for row in service.exchange.active)
    assert not any(call[0] == "cancel" for call in service.exchange.calls)


@pytest.mark.parametrize("price,reason", [(94, "would_loosen"), (101, "crossed")])
def test_manual_changes_cannot_loosen_or_cross_the_stop(price, reason):
    updated = replace(position(), quantity=3)
    service, stop = stop_service(updated)
    with pytest.raises(RuntimeError, match=reason):
        service.reconcile_manual_position_stop(updated, position(), stop, stop_price=price, market_price=100)
    assert service.exchange.calls == []


def managed_state(monkeypatch, rows, observed):
    now = OPENED + timedelta(minutes=5)
    monkeypatch.setattr(runner, "_utc_now", lambda: now)
    state = MultiPaperState(initial_equity=1000, cash=1000)
    state.execution_mode = "lead_usdm"
    state.portfolio.positions[SYMBOL] = position()
    state.exchange_orders_by_symbol[SYMBOL] = {"stop": {"id": "old", "amount": 2, "stop_price": 95}}
    state.entry_costs[SYMBOL] = {"entry_fee": .2}
    runner._initialize_entry_metadata(state, SYMBOL, position(), True, 100, 95, 2, 90, 110)
    state.exit_management_by_symbol[SYMBOL].update(no_progress_count=3, trailing_started=True, mid_fired=True)

    class Execution:
        adopt_manual_positions = True
        fail_protection = False
        protection_calls = 0
        exchange = SimpleNamespace(fetch_ticker=lambda symbol: {"last": 120, "timestamp": int(now.timestamp() * 1000)})

        def sync_snapshot(self):
            return {"at": now.isoformat(), "positions": {SYMBOL: observed} if observed else {}}

        def reconcile_manual_position(self, old, expected, **kwargs):
            kwargs.pop("at")
            return replay_position_changes(old, rows, expected, **kwargs)

        def reconcile_manual_position_stop(self, new, old, stop, *, stop_price, market_price, new_cycle=False):
            self.protection_calls += 1
            if self.fail_protection:
                raise RuntimeError("manual_sync_replacement_stop_not_verified")
            return {"id": "new", "amount": new.quantity, "stop_price": stop_price}

        def _verify_manual_position_snapshot(self, symbol, expected):
            assert expected == (observed or None)

        def cancel_saved_stop_if_open(self, symbol, stop):
            assert not observed

        def funding_fee(self, *args):
            return -.05

    return state, Execution(), now


def test_runner_updates_add_and_preserves_management_progress(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "BUY", 1, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected(quantity=3, price=310 / 3))
    assert runner._demo_sync_stopped(service, state, now)
    assert state.portfolio.positions[SYMBOL].quantity == 3
    assert state.portfolio.positions[SYMBOL].opened_at == OPENED
    meta = state.exit_management_by_symbol[SYMBOL]
    assert (meta["no_progress_count"], meta["trailing_started"], meta["mid_fired"]) == (3, True, True)
    assert meta["protective_stop"] == 95
    assert state.exchange_orders_by_symbol[SYMBOL]["stop"]["amount"] == 3
    assert state.entry_costs[SYMBOL]["entry_fee"] == pytest.approx(.31)
    assert runner._demo_sync_stopped(service, state, now)
    assert service.protection_calls == 1
    assert len([e for e in state.events if e.get("event") == "manual_position_changed"]) == 1


def test_failed_protection_keeps_ledger_and_cursor_then_retries(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "BUY", 1, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected(quantity=3, price=310 / 3))
    service.fail_protection = True
    assert not runner._demo_sync_stopped(service, state, now)
    assert state.portfolio.positions[SYMBOL].quantity == 2
    assert "manual_sync" not in state.exit_management_by_symbol[SYMBOL]
    assert state.execution_status == "inconsistent"
    service.fail_protection = False
    assert runner._demo_sync_stopped(service, state, now)
    assert state.portfolio.positions[SYMBOL].quantity == 3
    assert state.execution_status == "synchronized"


def test_position_state_and_cursor_survive_serialization_without_duplicate_pnl(monkeypatch):
    import json

    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected(quantity=1))
    assert runner._demo_sync_stopped(service, state, now)
    # These are the dictionaries serialized by MultiPaperState, with no new
    # top-level state schema or custom objects required.
    state.exit_management_by_symbol = json.loads(json.dumps(state.exit_management_by_symbol))
    state.entry_costs = json.loads(json.dumps(state.entry_costs))
    before = len(state.events)
    assert runner._demo_sync_stopped(service, state, now)
    assert len(state.events) == before
    assert state.exit_management_by_symbol[SYMBOL]["manual_realized"]["gross_pnl"] == 10


def test_realized_partial_pnl_is_included_in_later_strategy_close(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected(quantity=1))
    assert runner._demo_sync_stopped(service, state, now)
    values = runner._demo_close_values(service, state, state.portfolio.positions[SYMBOL],
                                       {"average": 120, "filled": 1, "exit_fee": .12}, now)
    assert values[0] == 30
    assert values[1] == pytest.approx(.43)
    assert values[-1] == pytest.approx(29.52)


def test_pending_strategy_flat_reconciliation_keeps_prior_partial_totals(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110)]
    observed = expected(quantity=1)
    state, service, now = managed_state(monkeypatch, rows, observed)
    assert runner._demo_sync_stopped(service, state, now)
    state.exchange_orders_by_symbol[SYMBOL]["close_attempt"] = {"reason": "long_structure_invalid"}
    service.close = lambda *args, **kwargs: {"fill": {"filled": 1, "average": 120, "exit_fee": .12}}
    observed.clear()
    assert runner._demo_sync_stopped(service, state, now)
    closed = next(e for e in state.events if e.get("action") == "CLOSE")
    assert closed["quantity"] == 2 and closed["price"] == 115
    assert closed["entry_price"] == 100
    assert closed["net_pnl"] == pytest.approx(29.52)
    assert not closed["strategy_sample_eligible"]


def test_unresolved_bot_close_owns_its_fills(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected(quantity=1))
    state.exchange_orders_by_symbol[SYMBOL]["close_attempt"] = {"status": "submitted"}
    service.reconcile_manual_position = lambda *args, **kwargs: pytest.fail("pending bot close must reconcile first")
    assert runner._sync_lead_position_changes(service, state, service.sync_snapshot(), now)
    assert state.portfolio.positions[SYMBOL].quantity == 2
    assert not state.events


def test_isolated_manual_error_keeps_other_verified_positions_managed(monkeypatch):
    settings = load_settings(env_file=None)
    now = OPENED + timedelta(minutes=5)
    monkeypatch.setattr(runner, "_utc_now", lambda: now)
    state = MultiPaperState()
    state.execution_mode, state.execution_status = "lead_usdm", "inconsistent"
    state.portfolio.positions[SYMBOL] = position()
    state.portfolio.positions["BAD"] = replace(position(), symbol="BAD")
    for symbol, old in state.portfolio.positions.items():
        state.exchange_orders_by_symbol[symbol] = {"stop": {"id": symbol, "stop_price": 95}}
        runner._initialize_entry_metadata(state, symbol, old, True, 100, 95, 2, 90, 110)
    state.exchange_snapshot = {"at": now.isoformat(), "positions": {SYMBOL: expected(), "BAD": expected(quantity=3)}}
    state.protective_stop_audit = {"at": now.isoformat(), "status": "error",
                                  "scope": "manual_position_change", "blocked_symbols": ["BAD"]}
    tickers, audits = [], []

    def ticker(symbol):
        tickers.append(symbol)
        return {"last": 100, "timestamp": int(now.timestamp() * 1000)}

    def audit(positions, orders):
        audits.append(list(positions))
        return []

    exchange = SimpleNamespace(markets={SYMBOL: {}}, fetch_ticker=ticker,
                               price_to_precision=lambda symbol, price: str(price))
    service = SimpleNamespace(audit_active_protective_stops=audit, exchange=exchange)
    assert runner._manage_demo_positions(exchange, service, state, settings, now, 0, 0)
    assert tickers == [SYMBOL] and audits == [[SYMBOL]]
    assert state.execution_status == "inconsistent"
    assert runner._manage_demo_positions(exchange, service, state, settings, now + timedelta(seconds=5), 0, 0)
    assert len(audits) == 1


def test_runner_reduction_then_full_close_settles_once(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 1, 110)]
    observed = expected(quantity=1)
    state, service, now = managed_state(monkeypatch, rows, observed)
    assert runner._demo_sync_stopped(service, state, now)
    assert state.entry_costs[SYMBOL]["entry_fee"] == pytest.approx(.1)
    assert runner._demo_sync_stopped(service, state, now)
    prior = state.exit_management_by_symbol[SYMBOL]["manual_realized"]
    assert prior["gross_pnl"] == 10
    rows.append(trade(3, "SELL", 1, 120))
    observed.clear()
    assert runner._demo_sync_stopped(service, state, now)
    closes = [e for e in state.events if e.get("action") == "CLOSE"]
    assert len(closes) == 1
    assert closes[0]["gross_pnl"] == 30
    assert closes[0]["fees"] == pytest.approx(.2 + .11 + .12)
    assert closes[0]["net_pnl"] == pytest.approx(30 - .43 - .05)
    assert not state.portfolio.positions
    assert runner._demo_sync_stopped(service, state, now)
    assert len([e for e in state.events if e.get("action") == "CLOSE"]) == 1


def test_runner_reversal_creates_new_context_after_old_settlement(monkeypatch):
    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 3, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected("short", 1, 110))

    def prepare(execution, symbol, observed, at, *, recovered, previous_position, previous_stop):
        assert previous_position.quantity == 2
        new = replace(position(Side.SHORT, 1, 110, 115), opened_at=recovered["opened_at"])
        return new, {"id": "new", "amount": 1, "stop_price": 115}, recovered, 3, 100, 112, 110, now

    monkeypatch.setattr(runner, "_prepare_lead_manual_position", prepare)
    assert runner._demo_sync_stopped(service, state, now)
    new = state.portfolio.positions[SYMBOL]
    assert new.side is Side.SHORT and new.quantity == 1
    assert new.opened_at == OPENED + timedelta(minutes=1)
    meta = state.exit_management_by_symbol[SYMBOL]
    assert not meta["trailing_started"] and meta["no_progress_count"] == 0
    assert meta["entry_atr"] == 3
    assert state.entry_costs[SYMBOL]["entry_fee"] == pytest.approx(.11)
    assert len([e for e in state.events if e.get("action") == "CLOSE"]) == 1


def test_reversal_prepares_completed_candle_context_and_new_stop(monkeypatch):
    from bot.regime import Regime

    rows = [trade(1, "BUY", 2, 100), trade(2, "SELL", 3, 110)]
    state, service, now = managed_state(monkeypatch, rows, expected("short", 1, 110))
    service.settings = load_settings(env_file=None)
    service.lead_market_ids = {"ADAUSDT"}
    service.exchange = SimpleNamespace(markets={SYMBOL: {}}, market=lambda symbol: {"id": "ADAUSDT"},
                                       fetch_ticker=lambda symbol: {"last": 110, "timestamp": int(now.timestamp() * 1000)})
    frame = pd.DataFrame({"open": [110.] * 60, "high": [112.] * 60, "low": [108.] * 60,
                          "close": [110.] * 60, "volume": [100.] * 60},
                         index=pd.date_range(end=OPENED - timedelta(hours=1), periods=60, freq="h"))
    four = frame.copy()
    four.loc[:, ["open", "high", "low", "close"]] -= 10  # short entry retains MR deviation
    monkeypatch.setattr(runner, "_adoption_closed_frame", lambda *args: frame if args[3] == "1h" else four)
    monkeypatch.setattr(runner, "calculate_adaptive_stop", lambda *args: (115, 1))
    monkeypatch.setattr(runner, "classify_regimes", lambda *args: pd.DataFrame({"regime": [Regime.MID]}))
    assert runner._demo_sync_stopped(service, state, now)
    new = state.portfolio.positions[SYMBOL]
    assert new.side is Side.SHORT and new.stop_price == 115
    assert state.exit_management_by_symbol[SYMBOL]["structure_high"] == 112
    assert state.exchange_orders_by_symbol[SYMBOL]["stop"]["amount"] == 1


def test_closed_manual_cycle_dashboard_uses_reconciled_total_and_cost(monkeypatch):
    from bot.live.dashboard import build_order_snapshot

    rows = [trade(1, "BUY", 2, 100), trade(2, "BUY", 1, 110), trade(3, "SELL", 3, 120)]
    state, service, now = managed_state(monkeypatch, rows, None)
    assert runner._demo_sync_stopped(service, state, now)
    result = build_order_snapshot({"execution_mode": "lead_usdm", "events": state.events})
    closed = result["trades"][0]
    assert closed["quantity"] == 3
    assert closed["entry_price"] == pytest.approx(310 / 3)
    assert closed["exit_price"] == 120
    assert closed["opened_at"] == OPENED.isoformat()


def test_manual_close_consumes_the_current_signal(monkeypatch):
    settings = load_settings(env_file=None)
    now = OPENED + timedelta(seconds=30)
    state = MultiPaperState()
    state.exit_management_by_symbol["__manual_entry_blocks__"] = {SYMBOL: (OPENED + timedelta(seconds=10)).isoformat()}
    index1 = pd.date_range(end=OPENED - timedelta(hours=1), periods=60, freq="h")
    index4 = pd.date_range(end=OPENED - timedelta(hours=4), periods=60, freq="4h")
    frame1 = pd.DataFrame({"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000}, index=index1)
    frame4 = pd.DataFrame({"close": 110}, index=index4)
    assert runner._entry_candidate(None, state, False, SYMBOL, {SYMBOL: frame1}, {SYMBOL: frame4}, {},
                                   settings, now, {}) is None
    assert state.events[-1]["reason"] == "manual_close_consumed_signal"
