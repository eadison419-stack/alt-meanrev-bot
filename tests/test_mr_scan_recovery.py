from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import ccxt
import pandas as pd
import pytest

from bot.config import load_settings
from bot.exchange.binance_usdm import FAPI_URLS
from bot.execution import DemoExecutionService, EntryPreflightRejected
from bot.live import mr_scan
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.risk import OrderIntent
from bot.regime import Regime
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 3, 5, 0, 20, tzinfo=timezone.utc)
FIRST, SECOND = "FIRST/USDT:USDT", "SECOND/USDT:USDT"


@pytest.fixture
def scan_env(monkeypatch, tmp_path):
    settings = load_settings(env_file=None)
    settings.sizing.max_directional_notional_pct = settings.sizing.max_gross_notional_pct
    state = MultiPaperState(initial_equity=5000, cash=5000, execution_status="synchronized")
    clock = [NOW]
    calls = []
    faults = {}
    one = pd.DataFrame({"open": 97.5, "high": 99., "low": 97., "close": 98., "volume": 1.},
                       index=pd.date_range(end=NOW.replace(second=0)-timedelta(hours=1), periods=60, freq="h"))
    four = pd.DataFrame({"close": 100.}, index=pd.date_range(end=NOW, periods=100, freq="4h"))
    monkeypatch.setattr(runner, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(runner, "_cached_daily_direction_gate", lambda *a, **kw: {"ready": True, "direction": "long"})

    class Client:
        def price_to_precision(self, symbol, price):
            return str(price)

        def fetch_order_book(self, symbol, *, limit):
            return {"timestamp": int(clock[0].timestamp()*1000), "bids": [[98.1, 100]], "asks": [[98.2, 100]]}

        def fetch_funding_rate(self, symbol):
            return {"fundingRate": 0.0}

    class Execution:
        exchange = Client()

        def open(self, intent, *, stop_price, before_submit, submission_deadline,
                 client_order_id, on_submit):
            calls.append(intent.symbol)
            errors = faults.get(intent.symbol, [])
            if errors:
                error = errors.pop(0)
                if callable(error):
                    error()
                else:
                    raise error
            refreshed, stop = before_submit(intent, stop_price)
            on_submit(refreshed.quantity, stop, clock[0])
            if faults.get("after_submit"):
                if isinstance(faults["after_submit"], Exception):
                    raise faults["after_submit"]
                raise ccxt.RequestTimeout("response lost")
            return {"fill": {"id": client_order_id, "clientOrderId": client_order_id},
                    "position": {"quantity": refreshed.quantity, "entry_price": refreshed.price},
                    "stop": {"id": "stop", "stop_price": stop}, "order_submitted_at": clock[0].isoformat()}

    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", NOW, "BTC/USDT:USDT", "4h", {})
    path = tmp_path / "state.json"
    execution = Execution()

    def run(current=state):
        runner._evaluate_entries(Client(), current, execution, True, {}, [FIRST, SECOND],
                                 {FIRST: one, SECOND: one}, {FIRST: four, SECOND: four}, {}, settings,
                                 5, .0005, .0003, clock[0], risk,
                                 checkpoint=lambda: current.save(path))

    return SimpleNamespace(state=state, clock=clock, calls=calls, faults=faults, run=run,
                           path=path, one=one, four=four, settings=settings, risk=risk)


def test_bad_symbol_does_not_abort_remaining_candidates(scan_env):
    env = scan_env
    env.faults[FIRST] = [ccxt.BadSymbol("unsupported")]
    env.run()
    assert env.calls == [FIRST, SECOND]
    assert list(env.state.portfolio.positions) == [SECOND]
    failure = next(e for e in env.state.events if e.get("error") == "BadSymbol")
    assert (failure["symbol"], failure["endpoint"], failure["order_may_have_been_sent"]) == (FIRST, "demo", False)
    assert env.state.mr_entry_scan["status"] == "completed"


def test_one_read_retry_runs_after_other_candidates_and_never_reruns_completed(scan_env):
    env = scan_env
    env.faults[FIRST] = [ccxt.RequestTimeout("read failed"), ccxt.RequestTimeout("again")]
    env.run()
    assert env.calls == [FIRST, SECOND, FIRST]
    assert env.state.mr_entry_scan["outcomes"][FIRST]["status"] == "error"
    restored = MultiPaperState.load(env.path)
    restored.portfolio.positions.clear()  # Even a filled-then-closed symbol stays consumed.
    env.run(restored)
    assert env.calls == [FIRST, SECOND, FIRST]


def test_retry_does_not_extend_original_deadline(scan_env):
    env = scan_env
    env.faults[FIRST] = [ccxt.RequestTimeout("read failed")]
    env.faults[SECOND] = [lambda: env.clock.__setitem__(0, NOW.replace(second=0)+timedelta(seconds=61))]
    env.run()
    assert env.calls == [FIRST, SECOND]
    assert env.state.mr_entry_scan["outcomes"][FIRST]["status"] == "expired"
    assert not env.state.portfolio.positions


@pytest.mark.parametrize("error", [ccxt.AuthenticationError, ccxt.RateLimitExceeded, RuntimeError])
def test_global_errors_pause_scan_without_attempting_the_next_order(scan_env, error):
    env = scan_env
    env.faults[FIRST] = [error("private detail must not be logged")]
    env.run()
    assert env.calls == [FIRST]
    assert env.state.mr_entry_scan["status"] == "paused"
    assert "private detail" not in env.path.read_text(encoding="utf-8")


def test_lost_submit_response_survives_restart_and_never_resends(scan_env):
    env = scan_env
    env.faults["after_submit"] = True
    env.run()
    restored = MultiPaperState.load(env.path)
    assert env.calls == [FIRST]
    attempt = mr_scan.unresolved_attempts(restored)[0]
    assert attempt["quantity"] > 0 and attempt["stop_price"] > 0
    assert attempt["clientOrderId"].startswith("mrb_")
    env.run(restored)
    assert env.calls == [FIRST]


def test_preflight_bad_request_has_linked_uncertain_outcome_without_raw_response(scan_env):
    env = scan_env
    env.faults["after_submit"] = ccxt.BadRequest("GET ?signature=example-signature")
    env.run()
    audit = next(e for e in env.state.events if e.get("event") == "entry_preflight_decision")
    outcome = next(e for e in env.state.events if e.get("event") == "entry_attempt_outcome"
                   and e["status"] == "uncertain")
    assert audit["clientOrderId"] == outcome["clientOrderId"]
    assert outcome["error"] == "BadRequest" and outcome["order_may_have_been_sent"]
    assert env.calls == [FIRST]
    assert "example-signature" not in env.path.read_text(encoding="utf-8")


def test_successful_attempt_links_preflight_fill_and_completed_outcome(scan_env):
    env = scan_env
    env.run()
    for symbol in (FIRST, SECOND):
        events = [e for e in env.state.events if e.get("symbol") == symbol]
        audit = next(e for e in events if e.get("event") == "entry_preflight_decision")
        fill = next(e for e in events if e.get("event") == "entry_execution_quality")
        outcome = next(e for e in events if e.get("event") == "entry_attempt_outcome"
                       and e["status"] == "completed")
        assert audit["clientOrderId"] == fill["clientOrderId"] == outcome["clientOrderId"]
    assert len(env.state.portfolio.positions) == 2


def test_interrupted_scan_resumes_unfinished_symbols_without_repeating_exits(scan_env, monkeypatch):
    env = scan_env
    # A saved partial scan from before submission; its completed symbol is flat.
    scan = mr_scan.scan_for(env.state, env.one.index[-1], [FIRST, SECOND], 60)
    mr_scan.outcome(env.state, scan, FIRST, "opened", NOW, side="long")
    env.state.exit_management_by_symbol[runner.BAR_KEY] = str(env.one.index[-1])
    env.state.save(env.path)
    restored = MultiPaperState.load(env.path)
    monkeypatch.setattr(runner, "_update_market_risk", lambda *a: env.risk)
    monkeypatch.setattr(runner, "_build_universe", lambda *a, **kw: ({}, [{"symbol": s} for s in [FIRST, SECOND]],
                                                                                {FIRST: env.one, SECOND: env.one}, {}, {}))
    monkeypatch.setattr(runner, "_cached_four_hour", lambda *a: env.four)
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *a: True)
    monkeypatch.setattr(runner, "_demo_stop_audit_due", lambda *a: False)
    monkeypatch.setattr(runner, "_evaluate_exits", lambda *a: pytest.fail("exit replayed"))
    # Exercise real scan recovery while replacing the exchange submission with a rejection.
    monkeypatch.setattr(runner, "_submit_entry_candidate", lambda *a, **kw: env.calls.append(a[5][0]))
    runner._cycle(None, None, restored, env.settings, NOW, True, 5,
                  checkpoint=lambda: restored.save(env.path))
    assert env.calls == [SECOND]
    assert restored.mr_entry_scan["status"] == "completed"


def test_executor_cannot_submit_when_intent_checkpoint_fails(monkeypatch):
    from bot import execution as module
    settings = load_settings(env_file=None)
    client = SimpleNamespace(urls={"api": dict(FAPI_URLS)},
                             amount_to_precision=lambda s, q: str(q), price_to_precision=lambda s, p: str(p))
    service = DemoExecutionService(client, settings)
    monkeypatch.setattr(module, "_utc_now", lambda: NOW)
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_position", lambda s: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda s: None)
    monkeypatch.setattr(service, "_market_rules", lambda s: (.1, .1, 5))
    monkeypatch.setattr(service, "_market_fill", lambda *a, **kw: pytest.fail("submitted without durable intent"))

    def failed_save(*args):
        raise OSError("disk full")

    intent = OrderIntent(FIRST, Side.LONG, Action.OPEN, 1, 98, "test", Regime.MID, 96, "MEAN_REVERSION")
    with pytest.raises(OSError):
        service.open(intent, stop_price=96, client_order_id="mrb_fixed", on_submit=failed_save)


def test_persisting_intent_past_deadline_prevents_market_order(monkeypatch):
    from bot import execution as module
    clock = [NOW]
    client = SimpleNamespace(urls={"api": dict(FAPI_URLS)},
                             amount_to_precision=lambda s, q: str(q), price_to_precision=lambda s, p: str(p))
    service = DemoExecutionService(client, load_settings(env_file=None))
    monkeypatch.setattr(module, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_position", lambda s: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda s: None)
    monkeypatch.setattr(service, "_market_rules", lambda s: (.1, .1, 5))
    monkeypatch.setattr(service, "_market_fill", lambda *a, **kw: pytest.fail("expired signal submitted"))
    intent = OrderIntent(FIRST, Side.LONG, Action.OPEN, 1, 98, "test", Regime.MID, 96, "MEAN_REVERSION")
    deadline = NOW.replace(second=0) + timedelta(seconds=60)
    with pytest.raises(EntryPreflightRejected, match="entry_signal_expired"):
        service.open(intent, stop_price=96, client_order_id="mrb_fixed", submission_deadline=deadline,
                     on_submit=lambda *a: clock.__setitem__(0, deadline+timedelta(microseconds=1)))


def test_quote_expiring_during_checkpoint_is_rejected_before_send(scan_env):
    env = scan_env
    audit = {"symbol": FIRST, "side": "long",
             "quote_valid_until": (NOW+timedelta(seconds=15)).isoformat()}
    row = {"bar": env.one.index[-1].isoformat()}
    sends = []

    class Execution:
        def open(self, intent, *, stop_price, client_order_id, on_submit):
            on_submit(intent.quantity, stop_price, NOW)
            sends.append(client_order_id)

    def save():
        env.state.save(env.path)
        if mr_scan.unresolved_attempts(env.state):
            env.clock[0] = NOW+timedelta(seconds=16)

    opened = runner._open_position(True, Execution(), env.state, FIRST, Side.LONG, 98, 96,
                                   1, 3, 0, 0, "test", "MID", NOW,
                                   entry_audit=audit, scan_row=row, checkpoint=save)
    assert opened is None and sends == []
    assert audit["reason"] == "entry_quote_stale"
    assert not mr_scan.unresolved_attempts(env.state)


@pytest.fixture
def recovery_env():
    attempt = {"symbol": FIRST, "side": "long", "clientOrderId": "mrb_fixed", "quantity": 1.,
               "stop_price": 96., "submitted_at": NOW.isoformat(), "status": "uncertain",
               "bar": (NOW.replace(second=0)-timedelta(hours=1)).isoformat(), "reason": "test",
               "regime": "MID", "meta": {"entry_atr": 1., "structure_low": 97.,
                                          "structure_high": 99., "last_close": 98.}}

    class Client:
        urls = {"api": dict(FAPI_URLS)}
        query_error = None
        live = {"side": "long", "quantity": 1., "entry_price": 98.}
        order = {"symbol": "FIRSTUSDT", "clientOrderId": "mrb_fixed", "orderId": 12,
                 "side": "BUY", "status": "FILLED", "executedQty": "1", "avgPrice": "98", "reduceOnly": False}
        stops = [{"algoId": "stop12", "clientAlgoId": "mrb_s_fixed", "orderType": "STOP_MARKET",
                  "reduceOnly": True, "algoStatus": "NEW", "side": "SELL", "quantity": "1", "triggerPrice": "96"}]

        def market(self, symbol):
            return {"id": "FIRSTUSDT"}

        def fapiPrivateGetOrder(self, params):
            assert params == {"symbol": "FIRSTUSDT", "origClientOrderId": "mrb_fixed"}
            if self.query_error:
                raise self.query_error
            return self.order

        def fapiPrivateGetOpenAlgoOrders(self, params):
            return self.stops

        def fapiPrivateGetUserTrades(self, params):
            return [{"orderId": 12, "qty": "1", "commission": "0.05", "commissionAsset": "USDT",
                     "time": int(NOW.timestamp()*1000)}]

    client = Client()
    service = DemoExecutionService(client, load_settings(env_file=None))
    service._position = lambda symbol: client.live
    state = MultiPaperState(mr_entry_attempts={"mrb_fixed": attempt})
    return state, client, service


def test_confirmed_entry_and_active_stop_are_adopted_once(recovery_env, tmp_path):
    state, client, service = recovery_env
    save = lambda: state.save(tmp_path / "state.json")
    assert runner._reconcile_pending_entries(service, state, save)
    assert state.portfolio.positions[FIRST].entry_price == 98
    assert state.entry_costs[FIRST]["entry_fee"] == .05
    assert state.exchange_orders_by_symbol[FIRST]["stop"]["id"] == "stop12"
    assert runner._reconcile_pending_entries(service, state, save)
    assert len([e for e in state.events if e.get("action") == "OPEN"]) == 1
    assert not state.exit_management_by_symbol[FIRST]["telemetry_history_complete"]


def test_recovery_preserves_saved_preflight_quality_and_order_identity(recovery_env, tmp_path):
    state, client, service = recovery_env
    state.mr_entry_attempts["mrb_fixed"]["audit"] = {
        "signal_atr": 1., "signal_close": 98., "clientOrderId": "mrb_fixed",
        "demo_quote": {"price": 98., "quoted_at": NOW.isoformat()}, "public_quote": None,
    }
    assert runner._reconcile_pending_entries(service, state, lambda: state.save(tmp_path / "state.json"))
    quality = next(e for e in state.events if e.get("event") == "entry_execution_quality")
    assert quality["clientOrderId"] == "mrb_fixed"
    assert quality["fill_price"] == 98 and quality["demo_quote_to_fill_atr"] == 0
    outcome = next(e for e in state.events if e.get("event") == "entry_attempt_outcome")
    assert outcome["status"] == "completed" and outcome["reconciled"]


@pytest.mark.parametrize("case,reason", [("flat", "filled_then_flat"), ("no_stop", "protection_unconfirmed"),
                                         ("not_found", "query_failed")])
def test_ambiguous_recovery_never_assumes_no_fill_or_resends(recovery_env, case, reason, tmp_path):
    state, client, service = recovery_env
    if case in {"flat", "not_found"}:
        client.live = None
    if case == "not_found":
        client.query_error = ccxt.OrderNotFound("missing")
    if case == "no_stop":
        client.stops = []
    assert not runner._reconcile_pending_entries(service, state, lambda: state.save(tmp_path / "state.json"))
    assert not state.portfolio.positions
    assert mr_scan.unresolved_attempts(state)[0]["reconciliation"]["status"] == reason


def test_confirmed_zero_fill_rejection_is_terminal(recovery_env, tmp_path):
    state, client, service = recovery_env
    client.live = None
    client.order = {**client.order, "status": "REJECTED", "executedQty": "0"}
    assert runner._reconcile_pending_entries(service, state, lambda: state.save(tmp_path / "state.json"))
    assert state.mr_entry_attempts["mrb_fixed"]["status"] == "not_filled"
    assert not state.portfolio.positions


def test_pending_symbol_does_not_disable_management_of_known_positions(recovery_env):
    state, client, service = recovery_env
    from bot.portfolio import Position
    state.portfolio.positions[SECOND] = Position(SECOND, Side.LONG, 1, 100, 96, NOW, "test", "MID")
    state.exchange_orders_by_symbol[SECOND] = {"stop": {"id": "known-stop", "stop_price": 96}}
    service.sync_snapshot = lambda: {"at": NOW.isoformat(), "positions": {FIRST: client.live, SECOND: client.live}}
    service.cancel_orphaned_bot_stops = lambda *a: pytest.fail("unresolved protective stops cleaned up")
    assert runner._demo_sync_stopped(service, state, NOW)
    assert state.execution_status == "synchronized"
    assert mr_scan.unresolved_attempts(state)  # Separate new-entry gate remains closed.
    # An unrelated foreign position still stops account synchronization.
    service.sync_snapshot = lambda: {"positions": {FIRST: client.live, SECOND: client.live, "FOREIGN": client.live}}
    assert not runner._demo_sync_stopped(service, state, NOW)


def test_old_ledgers_load_without_scan_fields_and_preserve_exit_progress():
    state = MultiPaperState()
    state.exit_management_by_symbol[runner.BAR_KEY] = "old-bar"
    payload = state.to_dict()
    payload.pop("mr_entry_scan")
    payload.pop("mr_entry_attempts")
    restored = MultiPaperState.from_dict(payload)
    assert restored.mr_entry_scan == {} and restored.mr_entry_attempts == {}
    assert restored.exit_management_by_symbol[runner.BAR_KEY] == "old-bar"


def test_upgrade_recovers_same_hour_open_then_close_from_legacy_events(scan_env):
    env = scan_env
    env.state.events.extend([
        {"event": "paper_fill", "action": "OPEN", "source": "MEAN_REVERSION",
         "symbol": FIRST, "side": "long", "at": NOW.isoformat()},
        {"event": "paper_fill", "action": "CLOSE", "symbol": FIRST, "at": NOW.isoformat()},
    ])
    env.run()
    assert env.calls == [SECOND]
    assert env.state.mr_entry_scan["outcomes"][FIRST]["restored_from"] == "paper_fill"
