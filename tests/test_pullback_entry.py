from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.config import load_settings
from bot.live import paper
from bot.live.dashboard import build_multi_snapshot
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import PaperTrader
from bot.regime import Regime
from bot.signals import Action, Side, Signal
from bot.time_filter import TimeFilterFlags


def _pending(side: str, signal_time: datetime, *, target: float, structure: float) -> dict:
    return {
        "execution_type": "pullback", "execute_at": signal_time.isoformat(), "signal_time": signal_time.isoformat(),
        "confirmed_at": signal_time.isoformat(), "expires_at": (signal_time + timedelta(minutes=15)).isoformat(),
        "action": "OPEN", "side": side, "reason": "synthetic_entry", "strength": 1.0,
        "atr": 1.0, "regime": Regime.LOW.value if side == "long" else Regime.HIGH.value,
        "reference_price": 100.0, "pullback_price": target,
        "entry_structure_low" if side == "long" else "entry_structure_high": structure,
        "entry_signal_close": 100.0,
    }


def test_unclosed_one_hour_bar_does_not_create_entry_permission(monkeypatch):
    settings = load_settings(env_file=None)
    settings = settings.model_copy(update={
        "execution": settings.execution.model_copy(update={"wait_pullback": True}),
    })
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")

    def bars(index):
        close = pd.Series(100.0, index=index)
        return pd.DataFrame({"open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
                             "close": close, "volume": 1.0}, index=index)

    def regimes(frame, _settings):
        return pd.DataFrame({"atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
                             "range_position": 0.2, "new_low": False, "new_high": False,
                             "regime": Regime.LOW, "range_low": 98.0, "range_high": 103.0}, index=frame.index)

    monkeypatch.setattr(paper, "classify_regimes", regimes)
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(paper, "evaluate_time_filter", lambda *args, **kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(paper, "generate_signal", lambda *args, **kwargs: Signal(Side.LONG, Action.OPEN, "synthetic", 1.0))
    frame_1h, frame_4h = bars(index_1h), bars(index_4h)
    candle_open = index_1h[-1] + pd.Timedelta(1, unit="h")

    trader.update(frame_1h, frame_4h, current_open_at=candle_open.to_pydatetime(), current_open=100.0,
                  current_price=100.0, funding_rate=0.0,
                  now=(index_1h[-1] + pd.Timedelta(30, unit="m")).to_pydatetime(), wait_pullback=True)

    assert account.pending is None
    assert account.position is None

    confirmed_at = candle_open.to_pydatetime() + timedelta(seconds=5)
    trader.update(frame_1h, frame_4h, current_open_at=candle_open.to_pydatetime(), current_open=100.0,
                  current_price=100.0, funding_rate=0.0, now=confirmed_at, wait_pullback=True)

    assert account.pending is not None
    assert account.pending["execution_type"] == "pullback"
    assert account.pending["signal_time"] == candle_open.isoformat()
    assert account.position is None
    assert any(event.get("event") == "entry_confirmed" for event in state.events)


def test_deep_range_break_signal_creates_no_pending_or_order(monkeypatch):
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("US/USDT:USDT")
    trader = PaperTrader(account, settings)
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")
    one = pd.DataFrame({"open": 0.00770, "high": 0.0079, "low": 0.0076,
                        "close": 0.00773, "volume": 1.0}, index=index_1h)
    four = pd.DataFrame({"open": 0.01, "high": 0.02, "low": 0.009,
                         "close": 0.01, "volume": 1.0}, index=index_4h)

    def regimes(frame, _settings):
        return pd.DataFrame({"atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
                             "range_position": 0.2, "new_low": False, "new_high": False,
                             "regime": Regime.LOW, "range_low": 0.008255,
                             "range_high": 0.036397}, index=frame.index)

    monkeypatch.setattr(paper, "classify_regimes", regimes)
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(0.000455, index=frame.index))
    monkeypatch.setattr(paper, "evaluate_time_filter", lambda *args, **kwargs: TimeFilterFlags(True, True, False, False))
    close_at = index_1h[-1] + pd.Timedelta(1, unit="h")

    trader.update(one, four, current_open_at=close_at.to_pydatetime(), current_open=0.00773,
                  current_price=0.00773, funding_rate=0.0,
                  now=(close_at + pd.Timedelta(5, unit="s")).to_pydatetime(), wait_pullback=True)

    assert account.position is None
    assert account.pending is None
    assert not any(event.get("event") == "order_intent" for event in state.events)
    rejected = next(event for event in state.events if event.get("event") == "signal_rejected")
    assert rejected["reason"] == "long_rejected_below_range"


def test_long_waits_for_pullback_price_before_paper_fill():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("long", signal_time, target=99.6, structure=98.5)

    assert trader._advance_pullback_pending(99.8, signal_time + timedelta(minutes=1), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is False
    assert account.position is None
    assert account.pending is not None

    assert trader._advance_pullback_pending(99.6, signal_time + timedelta(minutes=2), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is True
    assert account.position is not None and account.position.side is Side.LONG
    filled = next(event for event in account.state.events if event["event"] == "pending_entry_filled")
    assert filled["entry_delay_sec"] == 120.0


def test_pending_times_out_without_market_fallback():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("long", signal_time, target=99.6, structure=98.5)

    assert trader._advance_pullback_pending(100.0, signal_time + timedelta(minutes=16), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is True
    assert account.position is None
    assert account.pending is None
    assert any(event.get("event") == "pending_entry_cancelled" and event.get("reason") == "timeout"
               for event in state.events)


def test_expired_pending_times_out_before_later_structure_invalidation():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("short", signal_time, target=100.4, structure=101.5)

    assert trader._advance_pullback_pending(102.0, signal_time + timedelta(minutes=16), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is True
    assert account.pending is None
    cancelled = next(event for event in state.events if event.get("event") == "pending_entry_cancelled")
    assert cancelled["reason"] == "timeout"


def test_pending_is_cancelled_when_long_structure_breaks():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("long", signal_time, target=99.6, structure=98.5)

    assert trader._advance_pullback_pending(98.49, signal_time + timedelta(minutes=1), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is True
    assert account.pending is None
    cancelled = next(event for event in state.events if event.get("event") == "pending_entry_cancelled")
    assert cancelled["reason"] == "invalidated"


def test_pending_rechecks_deep_range_break_at_trigger_and_never_opens():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("long", signal_time, target=99.6, structure=98.5)
    account.pending.update({"entry_range_low": 100.0, "entry_range_high": 105.0})

    cancelled = trader._advance_pullback_pending(
        99.4, signal_time + timedelta(minutes=2), 0.0,
        step_size=0.001, minimum_quantity=0.001,
    )

    assert cancelled is True
    assert account.pending is None
    assert account.position is None
    event = next(event for event in state.events if event.get("event") == "pending_entry_cancelled")
    assert event["reason"] == "long_rejected_below_range"


def test_short_waits_for_upward_retest_instead_of_chasing():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    account = state.for_symbol("AAA/USDT:USDT")
    trader = PaperTrader(account, settings)
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    account.pending = _pending("short", signal_time, target=100.4, structure=101.5)

    assert trader._advance_pullback_pending(100.2, signal_time + timedelta(minutes=1), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is False
    assert account.position is None
    assert trader._advance_pullback_pending(100.4, signal_time + timedelta(minutes=2), 0.0,
                                             step_size=0.001, minimum_quantity=0.001) is True
    assert account.position is not None and account.position.side is Side.SHORT


def test_dashboard_snapshot_exposes_waiting_pullback_entry():
    settings = load_settings(env_file=None)
    state = MultiPaperState()
    signal_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    state.pending_by_symbol["AAA/USDT:USDT"] = _pending("long", signal_time, target=99.6, structure=98.5)

    snapshot = build_multi_snapshot(state.to_dict(), tickers={}, settings=settings)

    assert snapshot["strategy_variant"] == "immediate_baseline"
    assert snapshot["pending_entries"] == [{
        "symbol": "AAA/USDT:USDT", "side": "long",
        "signal_time": signal_time.isoformat(), "confirmed_at": signal_time.isoformat(),
        "reference_price": 100.0, "pullback_price": 99.6,
        "expires_at": (signal_time + timedelta(minutes=15)).isoformat(),
    }]

