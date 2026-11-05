import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.config import load_settings
from bot.live import multi_paper as multi_paper_module
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import PaperTrader
from bot.portfolio import Position
from bot.regime import Regime
from bot.signals import Side
from bot.strategy_router import TREND


def queued_open(symbol, side, at):
    view_state = {
        "execute_at": at.isoformat(), "action": "OPEN", "side": side,
        "reason": "synthetic_hot_coin_signal", "strength": 1.0, "atr": 1.0,
        "regime": (Regime.LOW if side == "long" else Regime.HIGH).value,
    }
    return view_state


def test_multistate_persists_source_and_reads_legacy_strategy_source():
    symbol = "ADA/USDT:USDT"
    position = Position(symbol, Side.LONG, 1.0, 100.0, 95.0,
                        datetime(2026, 10, 1, tzinfo=timezone.utc), "trend", "TREND",
                        source=TREND)
    state = MultiPaperState()
    state.portfolio.positions[symbol] = position

    payload = state.to_dict()
    assert payload["positions"][symbol]["source"] == TREND
    restored = MultiPaperState.from_dict(payload)
    assert restored.portfolio.positions[symbol].source == TREND

    legacy_payload = dict(payload)
    legacy_payload["positions"] = {symbol: dict(payload["positions"][symbol])}
    legacy_payload["positions"][symbol]["strategy_source"] = legacy_payload["positions"][symbol].pop("source")
    legacy_restored = MultiPaperState.from_dict(legacy_payload)
    assert legacy_restored.portfolio.positions[symbol].source == TREND


def test_multi_paper_save_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    state_path = tmp_path / "portfolio.json"
    state_path.write_text('{"previous": true}', encoding="utf-8")
    original_replace = multi_paper_module.os.replace
    calls = []

    def replace_with_transient_lock(source, target):
        calls.append((Path(source), Path(target)))
        if len(calls) < 3:
            raise PermissionError(5, "access denied")
        return original_replace(source, target)

    monkeypatch.setattr(multi_paper_module.os, "replace", replace_with_transient_lock)
    monkeypatch.setattr(multi_paper_module.time, "sleep", lambda _: None)

    MultiPaperState().save(state_path)

    assert len(calls) == 3
    assert all(source != state_path.with_name("portfolio.json.tmp") for source, _ in calls)
    assert all(target == state_path for _, target in calls)
    assert json.loads(state_path.read_text(encoding="utf-8"))["version"] == 2


def test_multi_paper_save_preserves_previous_state_after_replace_retries_exhaust(tmp_path, monkeypatch):
    state_path = tmp_path / "portfolio.json"
    original = b'{"previous": true}'
    state_path.write_bytes(original)
    calls = []

    def deny_replace(source, target):
        calls.append((Path(source), Path(target)))
        raise PermissionError(5, "access denied")

    monkeypatch.setattr(multi_paper_module.os, "replace", deny_replace)
    monkeypatch.setattr(multi_paper_module.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        MultiPaperState().save(state_path)

    assert len(calls) == multi_paper_module._SAVE_REPLACE_RETRIES
    assert state_path.read_bytes() == original
    assert all(source != state_path.with_name("portfolio.json.tmp") for source, _ in calls)


def test_multiple_symbol_orders_share_portfolio_limits_and_survive_restart(tmp_path):
    settings = load_settings(env_file=None)
    settings.sizing.max_symbols = 2
    state_path = tmp_path / "portfolio.json"
    state = MultiPaperState()
    at = datetime(2026, 9, 29, 2, tzinfo=timezone.utc)
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
    decisions = []
    for symbol, side in zip(symbols, ("long", "short", "long")):
        view = state.for_symbol(symbol)
        view.pending = queued_open(symbol, side, at)
        PaperTrader(view, settings)._execute_pending(100.0, at, 0.0, step_size=0.001, minimum_quantity=0.001)
        decisions.append(next(event for event in reversed(state.events) if event["event"] == "order_intent"))

    assert list(state.portfolio.positions) == symbols[:2]
    assert decisions[-1]["allowed"] is False
    assert decisions[-1]["decision"] == "max_symbols_reached"
    assert state.portfolio.gross_notional <= state.current_equity * settings.sizing.max_gross_notional_pct
    assert all(math.isclose(event["intent"]["quantity"] / 0.001,
                            round(event["intent"]["quantity"] / 0.001), abs_tol=1e-7)
               for event in decisions[:2])

    state.save(state_path)
    restored = MultiPaperState.load(state_path)
    assert restored.cash == state.cash
    assert set(restored.portfolio.positions) == set(state.portfolio.positions)
    assert restored.entry_costs == state.entry_costs
    assert restored.portfolio.last_exit_category == state.portfolio.last_exit_category
    assert restored.portfolio.last_exit_source == state.portfolio.last_exit_source
    assert restored.exit_management_by_symbol == state.exit_management_by_symbol
    assert restored.pending_by_symbol == state.pending_by_symbol
    assert restored.events == state.events


def test_v1_ada_state_is_seeded_without_overwriting_legacy_file(tmp_path):
    from bot.live.paper import PaperAccount

    legacy_path = tmp_path / "state.json"
    next_path = tmp_path / "portfolio.json"
    old = PaperAccount(symbol="ADA/USDT:USDT", initial_equity=10_000.0, cash=9_999.0)
    old.last_processed_close = "2026-09-29T02:00:00+00:00"
    old.mark_prices[old.symbol] = 100.0
    old.save(legacy_path)
    before = legacy_path.read_bytes()

    seeded = MultiPaperState.load(next_path, legacy_path=legacy_path)
    seeded.save(next_path)

    assert seeded.initial_equity == 10_000.0
    assert seeded.cash == 9_999.0
    assert seeded.last_processed_by_symbol[old.symbol] == old.last_processed_close
    assert legacy_path.read_bytes() == before
    assert json.loads(next_path.read_text(encoding="utf-8"))["version"] == 2


def test_runner_executes_queued_manual_close_at_fresh_ticker_and_persists_result(tmp_path):
    from bot.live.manual_close import ManualCloseCommandStore
    from bot.live.multi_runner import MultiPaperRunner

    settings = load_settings(env_file=None)
    symbol = "AAA/USDT:USDT"
    state = MultiPaperState()
    at = datetime(2026, 9, 29, 2, tzinfo=timezone.utc)
    view = state.for_symbol(symbol)
    view.pending = queued_open(symbol, "long", at)
    trader = PaperTrader(view, settings)
    trader.margin_spec = {
        "maintenance_margin_rate": 0.025,
        "maintenance_amount": 0.0,
        "liquidation_fee_rate": 0.01,
    }
    trader._execute_pending(100.0, at, 0.0, step_size=0.001, minimum_quantity=0.001)
    store = ManualCloseCommandStore(tmp_path / "portfolio.manual-close")
    request = store.submit(symbol)

    class FakeExchange:
        def load_markets(self):
            return {}

    runner = MultiPaperRunner(FakeExchange(), settings, state, manual_close_store=store)
    hour_key = "2026-09-29T02:00:00+00:00"
    changed = runner._process_manual_close_requests(
        {symbol: {"last": 103.25}}, at + timedelta(minutes=5), hour_key,
    )

    result = store.status(request["request_id"])
    close_fill = next(event for event in reversed(state.events)
                      if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert changed is True
    assert symbol not in state.portfolio.positions
    assert close_fill["reason"] == "manual_close"
    assert close_fill["execution_price_source"] == "last_trade"
    assert result["status"] == "closed"
    assert result["price"] == close_fill["price"]
    assert result["fees"] > 0
    assert state.last_processed_by_symbol[symbol] == hour_key


def test_equity_history_records_mark_to_market_and_dedups_unchanged():
    state = MultiPaperState()
    at = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    state.record_equity(at)
    state.record_equity(at + timedelta(minutes=5))  # unchanged equity -> deduped

    assert len(state.equity_history) == 1
    assert state.equity_history[0]["at"] == at.isoformat()
    assert state.equity_history[0]["equity"] == 10_000.0
    assert state.equity_history[0]["open_positions"] == 0

    # a real change (cash moves) records a new snapshot
    state.cash = 9_990.0
    state.record_equity(at + timedelta(minutes=10))
    assert len(state.equity_history) == 2
    assert state.equity_history[-1]["equity"] == 9_990.0

    restored = MultiPaperState.from_dict(state.to_dict())
    assert restored.equity_history == state.equity_history

    # A fresh state and one loaded without the key both default to [].
    assert MultiPaperState().equity_history == []
    minimal = {"version": 2, "account_type": "multi_symbol_paper",
               "initial_equity": 10_000.0, "cash": 10_000.0}
    assert MultiPaperState.from_dict(minimal).equity_history == []

