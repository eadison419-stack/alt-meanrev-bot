from datetime import datetime, timedelta, timezone
import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.config import load_settings
from bot.exchange.binance_usdm import DEMO_REST_ROOT
from bot.portfolio import Position
from bot.live import paper
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import PaperAccount, PaperTrader
from bot.regime import Regime, RegimeSnapshot
from bot.signals import Action, Side, Signal
from bot.strategy_router import MEAN_REVERSION, TREND
from bot.time_filter import TimeFilterFlags
from bot.trend_strategy import initialize_trend_exit_state


def test_open_intent_fill_persists_and_restores(tmp_path):
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    account = PaperAccount(symbol=symbol)
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }

    trader._execute_pending(100.0, opened_at, 0.0)

    assert account.position is not None
    assert account.position.side is Side.LONG
    assert account.position.source == MEAN_REVERSION
    assert account.position.stop_price < account.position.entry_price
    assert account.cash < account.initial_equity
    assert any(event["event"] == "order_intent" and event["allowed"] for event in account.events)
    saved_state = account.to_dict()
    assert saved_state["position"]["source"] == MEAN_REVERSION
    legacy_state = dict(saved_state)
    legacy_state["position"] = dict(saved_state["position"])
    legacy_state["position"]["strategy_source"] = legacy_state["position"].pop("source")
    assert PaperAccount.from_dict(legacy_state).position.source == MEAN_REVERSION
    state_path = tmp_path / "paper" / "state.json"
    account.save(state_path)
    restored = PaperAccount.load(state_path, symbol=symbol)
    assert restored.position == account.position
    assert restored.cash == account.cash
    assert restored.pending is None
    assert restored.events == account.events

    # Replaying the same scheduled execution after restart cannot duplicate a fill.
    PaperTrader(restored, settings)._execute_pending(100.0, opened_at, 0.0)
    assert sum(event.get("action") == "OPEN" for event in restored.events if event["event"] == "paper_fill") == 1
    PaperTrader(restored, settings)._close(102.0, opened_at + pd.Timedelta(3_600_000_000_000, unit="ns"), "synthetic_close")
    close_intent = [event for event in restored.events if event["event"] == "order_intent"][-1]
    assert close_intent["allowed"] is True
    assert close_intent["intent"]["action"] == "CLOSE"
    close_fill = next(event for event in reversed(restored.events)
                      if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close_fill["source"] == MEAN_REVERSION
    assert restored.position is None
    assert restored.portfolio.last_exit_source[symbol] == MEAN_REVERSION
    restored.save(state_path)
    persisted = PaperAccount.load(state_path, symbol=symbol)
    assert persisted.portfolio.last_exit_source[symbol] == MEAN_REVERSION


def test_missing_funding_fails_closed_and_records_complete_intent():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "short",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.HIGH.value,
    }

    trader._execute_pending(100.0, opened_at, None)

    assert account.position is None
    intent_event = next(event for event in account.events if event["event"] == "order_intent")
    assert intent_event["allowed"] is False
    assert intent_event["decision"] == "funding_rate_missing"
    assert intent_event["intent"]["stop_price"] > intent_event["intent"]["price"]


def test_pending_entry_from_other_strategy_cannot_take_over_existing_position():
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    account = PaperAccount(symbol=symbol)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    position = Position(symbol, Side.SHORT, 1.0, 100.0, 105.0, opened_at,
                        "existing_trend", Regime.LOW.value, TREND)
    account.portfolio.positions[symbol] = position
    account.pending = {
        "execute_at": (opened_at + timedelta(hours=1)).isoformat(), "action": "OPEN", "side": "short",
        "reason": "trend_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.HIGH.value, "source": MEAN_REVERSION,
        "signal_time": opened_at.isoformat(), "close_first": True,
    }

    PaperTrader(account, settings)._execute_pending(100.0, opened_at + timedelta(hours=1), 0.0)

    assert account.position == position
    assert any(event.get("event") == "intent_rejected"
               and event.get("reason") == "position_owned_by_other_strategy"
               for event in account.events)


def test_local_trend_pending_entry_is_simulated_but_unverified_execution_stays_blocked():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "trend_pullback_candidate", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value, "source": TREND,
        "signal_time": (opened_at - timedelta(hours=1)).isoformat(),
        "confirmed_at": opened_at.isoformat(),
        "expires_at": (opened_at + timedelta(minutes=45)).isoformat(),
        "execution_type": "pullback", "planned_quantity": 1.0,
        "reference_price": 100.0, "pullback_price": 100.0,
        "stop_price": 98.0, "structure_break_price": 99.0,
        "confirm_bars": 3, "confirmation_pass": True,
        "confirmation_bar_time": (opened_at - timedelta(hours=1)).isoformat(),
    }

    PaperTrader(account, settings)._execute_pending(100.0, opened_at, 0.0)

    assert account.position is not None
    assert account.position.source == TREND
    assert any(event.get("event") == "paper_fill" and event.get("action") == "OPEN"
               and event.get("source") == TREND and event.get("confirmation_pass") is True
               for event in account.events)
    assert any(event.get("event") == "order_intent" and event.get("allowed") is True
               and event.get("source") == TREND and event.get("confirm_bars") == 3
               for event in account.events)

    demo_account = PaperAccount(symbol="ADA/USDT:USDT")
    demo_account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "trend_pullback_candidate", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value, "source": TREND,
        "signal_time": (opened_at - timedelta(hours=1)).isoformat(),
        "execution_type": "pullback", "stop_price": 98.0,
        "structure_break_price": 99.0, "confirm_bars": 3,
        "confirmation_pass": True,
    }
    PaperTrader(demo_account, settings, execution=object())._execute_pending(
        100.0, opened_at, 0.0,
    )
    assert demo_account.position is None
    assert any(event.get("event") == "intent_rejected"
               and event.get("reason") == "trend_execution_mode_not_allowed"
               for event in demo_account.events)


def test_local_paper_trend_routes_to_pullback_fill_and_trend_exit():
    settings = load_settings(env_file=None)
    symbol = "AAA/USDT:USDT"
    signal_close_at = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)

    def trending_bars(hours: int, count: int, delta: float) -> pd.DataFrame:
        interval = pd.Timedelta(hours, unit="h")
        index = pd.date_range(end=pd.Timestamp(signal_close_at) - interval,
                              periods=count, freq=interval, name="timestamp")
        rows = []
        for bar_index in range(count):
            close = 100 + bar_index * delta
            rows.append((close - 0.08, close + 0.4, close - 0.4, close, 100.0))
        return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)

    bars_1h = trending_bars(1, 100, 0.2)
    bars_4h = trending_bars(4, 100, 0.5)

    execution_blocked_account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    PaperTrader(execution_blocked_account, settings).update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=119.8, funding_rate=0.0, now=signal_close_at,
        allow_entries=False, symbol_in_hot_universe=False, entry_execution_allowed=False,
        symbol_regime_state="TREND", market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001,
    )
    assert execution_blocked_account.pending is None
    assert any(event.get("event") == "signal_ignored"
               and event.get("reason") == "symbol_outside_hot_universe"
               for event in execution_blocked_account.events)
    assert not any(event.get("event") == "hot_universe_exemption_granted"
                   for event in execution_blocked_account.events)

    cooldown_account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    PaperTrader(cooldown_account, settings).update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=119.8, funding_rate=0.0, now=signal_close_at,
        allow_entries=False, symbol_in_hot_universe=False, entry_execution_allowed=True,
        entry_not_before=signal_close_at + timedelta(minutes=5),
        symbol_regime_state="TREND", market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001,
    )
    cooldown_ignore = next(event for event in cooldown_account.events
                           if event.get("event") == "signal_ignored")
    assert cooldown_account.pending is None
    assert cooldown_ignore["filter"] == "entry_cooldown"
    assert not any(event.get("event") == "hot_universe_exemption_granted"
                   for event in cooldown_account.events)

    rejected_account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    PaperTrader(rejected_account, settings).update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=119.8, funding_rate=None, now=signal_close_at,
        allow_entries=False, symbol_in_hot_universe=False, entry_execution_allowed=True,
        symbol_regime_state="TREND", market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001,
    )
    assert rejected_account.pending is None
    assert not any(event.get("event") == "hot_universe_exemption_granted"
                   for event in rejected_account.events)
    assert any(event.get("event") == "signal_rejected"
               and event.get("reason") == "funding_rate_missing"
               and event.get("exemption_granted") is False
               for event in rejected_account.events)

    account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    trader = PaperTrader(account, settings)
    trader.update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=119.8, funding_rate=0.0, now=signal_close_at,
        allow_entries=False, symbol_in_hot_universe=False, entry_execution_allowed=True,
        symbol_regime_state="TREND", market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001,
    )

    pending = account.pending
    assert pending is not None
    assert pending["source"] == TREND
    assert pending["execution_type"] == "pullback"
    assert pending["confirm_bars"] == 3
    assert pending["confirmation_pass"] is True
    assert pending["hot_universe_exemption"] is True
    assert pending["exemption_reason"] == "trend_confirmed_pullback"
    assert pending["exemption_expires_at"] == pending["expires_at"]
    granted = next(event for event in account.events
                   if event.get("event") == "hot_universe_exemption_granted")
    assert granted["exemption_granted"] is True
    assert granted["source"] == TREND
    assert granted["exemption_expires_at"] == pending["expires_at"]
    route_event = next(event for event in account.events
                       if event.get("event") == "strategy_route_evaluated")
    assert route_event["hot_universe_exemption_candidate"] is True
    assert route_event["trend_confirmation"]["confirm_bars"] == 3
    assert route_event["trend_confirmation"]["confirmation_pass"] is True
    assert account.position is None

    trader.update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=float(pending["pullback_price"]), funding_rate=0.0,
        now=signal_close_at + timedelta(minutes=15), symbol_regime_state="TREND",
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
    )

    assert account.position is not None
    assert account.position.source == TREND
    open_fill = next(event for event in account.events
                     if event.get("event") == "paper_fill" and event.get("action") == "OPEN")
    assert open_fill["source"] == TREND
    assert open_fill["confirmation_pass"] is True
    assert open_fill["fee"] > 0
    filled_event = next(event for event in account.events
                        if event.get("event") == "pending_entry_filled")
    assert filled_event["exemption_status"] == "filled"

    expired_account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    PaperTrader(expired_account, settings).update(
        bars_1h, bars_4h, current_open_at=signal_close_at, current_open=119.8,
        current_price=119.8, funding_rate=0.0, now=signal_close_at + timedelta(minutes=46),
        allow_entries=False, symbol_in_hot_universe=False, entry_execution_allowed=True,
        symbol_regime_state="TREND", market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001,
    )
    expired_ignore = next(event for event in expired_account.events
                          if event.get("event") == "signal_ignored")
    assert expired_account.pending is None
    assert expired_ignore["reason"] == "symbol_outside_hot_universe"
    assert expired_ignore["exemption_status"] == "expired_before_grant"

    exit_bar_at = bars_1h.index[-1] + pd.Timedelta(2, unit="h")
    exit_bar = pd.DataFrame(
        [(119.8, 120.0, 118.2, 118.3, 100.0)],
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([exit_bar_at], name="timestamp"),
    )
    bars_1h_with_exit = pd.concat([bars_1h, exit_bar])
    close_at = signal_close_at + timedelta(hours=2)
    trader.update(
        bars_1h_with_exit, bars_4h,
        current_open_at=close_at, current_open=119.8, current_price=118.3,
        funding_rate=0.0, now=close_at, symbol_regime_state="RANGE",
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
    )

    assert account.position is None
    close_fill = next(event for event in account.events
                      if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close_fill["source"] == TREND
    assert close_fill["reason"] == "trend_structure_break"
    assert close_fill["fees"] > 0
    assert close_fill["slippage_paid"] >= 0
    assert account.portfolio.last_exit_source[symbol] == TREND


def test_adaptive_pool_rejects_new_signal_from_unassigned_strategy(monkeypatch):
    settings = load_settings(env_file=None)
    now = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    index_1h = pd.date_range(end=pd.Timestamp(now) - pd.Timedelta(1, unit="h"), periods=100,
                             freq="1h", name="timestamp")
    index_4h = pd.date_range(end=pd.Timestamp(now) - pd.Timedelta(4, unit="h"), periods=100,
                             freq="4h", name="timestamp")

    def frame(index, step):
        rows = []
        for number in range(len(index)):
            close = 100.0 + number * step
            rows.append((close - 0.05, close + 0.5, close - 0.5, close, 100.0))
        return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)

    monkeypatch.setattr(paper, "generate_signal", lambda *args, **kwargs: Signal(
        Side.LONG, Action.OPEN, "synthetic_mean_reversion_entry", 1.0,
    ))
    monkeypatch.setattr(paper, "evaluate_time_filter", lambda *args, **kwargs: TimeFilterFlags(
        True, True, False, False,
    ))
    account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)

    PaperTrader(account, settings).update(
        frame(index_1h, 0.02), frame(index_4h, 0.05),
        current_open_at=now, current_open=105.0, current_price=105.0,
        funding_rate=0.0, now=now, allow_entries=True,
        symbol_in_hot_universe=True, allowed_entry_sources={TREND},
        entry_execution_allowed=True, symbol_regime_state="RANGE",
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
    )

    assert account.pending is None
    assert account.position is None
    rejected = next(event for event in account.events
                    if event.get("event") == "signal_rejected")
    assert rejected["reason"] == "source_not_in_adaptive_candidate_pool"
    assert rejected["source"] == MEAN_REVERSION
    route = next(event for event in account.events
                 if event.get("event") == "strategy_route_evaluated")
    assert route["candidate_action"] == Action.OPEN.value
    assert route["signal_source"] == MEAN_REVERSION
    assert route["adaptive_pool_allowed_sources"] == [TREND]
    assert route["adaptive_pool_source_allowed"] is False


def test_near_range_structure_shadow_risk_check_does_not_mutate_paper_state(monkeypatch):
    settings = load_settings(env_file=None)
    settings.sizing.margin_per_symbol_usdt = 100.0
    symbol = "AAA/USDT:USDT"
    account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    trader = PaperTrader(account, settings)
    before_portfolio = copy.deepcopy(account.portfolio)
    now = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    index = pd.date_range(end=pd.Timestamp(now), periods=4, freq="1h", name="timestamp")
    bars_1h = pd.DataFrame(
        [(100.0, 101.0, 99.0, 100.5, 100.0)] * len(index),
        columns=["open", "high", "low", "close", "volume"], index=index,
    )
    bars_4h = bars_1h.copy()
    monkeypatch.setattr(paper, "generate_signal", lambda *args, **kwargs: Signal(
        Side.LONG, Action.OPEN, "shadow_test_entry", 1.0,
    ))

    result = trader._evaluate_near_range_structure_shadow(
        regime=SimpleNamespace(regime=Regime.LOW), candle_1h=bars_1h.iloc[-1],
        bars_1h=bars_1h, bars_4h=bars_4h,
        flags=TimeFilterFlags(True, True, False, False),
        reference_price=100.0, atr_value=1.0, range_low=95.0, range_high=105.0,
        structure_low=99.0, structure_high=101.0, now=now, funding_rate=0.0,
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
        allowed_entry_sources={MEAN_REVERSION}, allow_entries=True,
        entry_execution_allowed=True,
        symbol_regime_evidence={"range_structure_shadow": {"eligible": True}},
    )

    assert result["shadow_only"] is True
    assert result["signal_action"] == Action.OPEN.value
    assert result["risk_assessment"]["allowed"] is True, result["risk_assessment"]
    assert result["hypothetical_entry_allowed"] is True
    assert account.pending is None
    assert account.position is None
    assert account.events == []
    assert account.portfolio == before_portfolio


@pytest.mark.parametrize(
    ("side", "range_position", "counterfactual_regime", "expected_reason"),
    [
        (Side.LONG, 0.20, Regime.LOW, "low_zone_1h_bullish_close"),
        (Side.SHORT, 0.80, Regime.HIGH, "high_zone_1h_bearish_close"),
    ],
)
def test_near_range_shadow_skips_sideways_priority_but_keeps_signal_and_pool_gates(
    side, range_position, counterfactual_regime, expected_reason,
):
    settings = load_settings(env_file=None)
    settings.sizing.margin_per_symbol_usdt = 100.0
    symbol = "AAA/USDT:USDT"
    account = PaperAccount(symbol=symbol, initial_equity=5_000.0, cash=5_000.0)
    trader = PaperTrader(account, settings)
    before_portfolio = copy.deepcopy(account.portfolio)
    now = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    index_1h = pd.date_range(end=pd.Timestamp(now), periods=4, freq="1h", name="timestamp")
    index_4h = pd.date_range(end=pd.Timestamp(now), periods=4, freq="4h", name="timestamp")
    if side is Side.LONG:
        rows_1h = [
            (100.0, 100.2, 99.0, 100.1, 100.0),
            (100.1, 100.3, 99.2, 100.2, 100.0),
            (100.2, 100.4, 99.4, 100.3, 100.0),
            (100.3, 100.7, 99.6, 100.5, 100.0),
        ]
    else:
        rows_1h = [
            (100.5, 100.7, 100.1, 100.4, 100.0),
            (100.4, 100.6, 100.0, 100.3, 100.0),
            (100.3, 100.5, 99.9, 100.2, 100.0),
            (100.2, 100.4, 99.8, 100.0, 100.0),
        ]
    bars_1h = pd.DataFrame(
        rows_1h, columns=["open", "high", "low", "close", "volume"], index=index_1h,
    )
    bars_4h = pd.DataFrame(
        [
            (100.0, 101.0, 99.0, 100.1, 100.0),
            (100.1, 102.0, 98.8, 100.2, 100.0),
            (100.2, 101.5, 98.9, 100.3, 100.0),
            (100.3, 101.7, 99.0, 100.4, 100.0),
        ],
        columns=["open", "high", "low", "close", "volume"], index=index_4h,
    )
    regime = RegimeSnapshot(
        timestamp=pd.Timestamp(now), regime=Regime.SIDEWAYS, atr=1.0, adx=10.0,
        atr_percentile_rank=10.0, range_position=range_position,
        new_low=False, new_high=False,
    )
    result = trader._evaluate_near_range_structure_shadow(
        regime=regime, candle_1h=bars_1h.iloc[-1], bars_1h=bars_1h, bars_4h=bars_4h,
        flags=TimeFilterFlags(True, True, False, False), reference_price=100.5,
        atr_value=1.0, range_low=95.0, range_high=105.0, structure_low=99.0,
        structure_high=102.0, now=now, funding_rate=0.0, market_risk_state="NORMAL",
        step_size=0.001, minimum_quantity=0.001, allowed_entry_sources={TREND},
        allow_entries=True, entry_execution_allowed=True,
        symbol_regime_evidence={"range_structure_shadow": {"eligible": True}},
    )

    assert result["rules_version"] == "range_structure_sideways_priority_v1"
    assert result["legacy_regime"] == Regime.SIDEWAYS.value
    assert result["signal_reason"] == "entries_blocked"
    assert result["counterfactual_regime"] == counterfactual_regime.value
    assert result["sideways_priority_override"] is True
    assert result["counterfactual_signal_action"] == Action.OPEN.value
    assert result["counterfactual_signal_side"] == side.value
    assert result["counterfactual_signal_reason"] == expected_reason
    assert result["risk_assessment"]["allowed"] is True, result["risk_assessment"]
    assert result["hypothetical_entry_allowed"] is False
    assert account.pending is None
    assert account.position is None
    assert account.events == []
    assert account.portfolio == before_portfolio


def test_shadow_sideways_priority_preserves_impulse_then_low_high_order():
    settings = load_settings(env_file=None)
    now = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    index = pd.date_range(end=pd.Timestamp(now), periods=2, freq="4h", name="timestamp")
    bars_4h = pd.DataFrame(
        [(100.0, 101.0, 99.0, 100.1, 100.0)] * 2,
        columns=["open", "high", "low", "close", "volume"], index=index,
    )

    def sideways(range_position, *, new_low=False, new_high=False):
        return RegimeSnapshot(
            timestamp=pd.Timestamp(now), regime=Regime.SIDEWAYS, atr=1.0, adx=10.0,
            atr_percentile_rank=10.0, range_position=range_position,
            new_low=new_low, new_high=new_high,
        )

    low, low_reason = paper._shadow_regime_without_sideways_precedence(
        sideways(0.20), bars_4h, settings,
    )
    high, high_reason = paper._shadow_regime_without_sideways_precedence(
        sideways(0.80), bars_4h, settings,
    )
    mid, mid_reason = paper._shadow_regime_without_sideways_precedence(
        sideways(0.50), bars_4h, settings,
    )
    breakout_bars = bars_4h.copy()
    breakout_bars.loc[breakout_bars.index[-1], ["open", "close"]] = [100.0, 102.0]
    impulse, impulse_reason = paper._shadow_regime_without_sideways_precedence(
        sideways(0.20), breakout_bars, settings,
    )

    assert (low.regime, low_reason) == (Regime.LOW, "range_low_precedes_sideways")
    assert (high.regime, high_reason) == (Regime.HIGH, "range_high_precedes_sideways")
    assert (mid.regime, mid_reason) == (Regime.MID, "no_extreme_after_sideways")
    assert (impulse.regime, impulse_reason) == (
        Regime.IMPULSE_UP, "impulse_precedes_range_extreme",
    )


def test_exempt_trend_pullback_cancels_at_expiry_even_with_market_fallback():
    settings = load_settings(env_file=None)
    settings.execution.timeout_action = "market_fallback"
    symbol = "ADA/USDT:USDT"
    confirmed_at = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    expires_at = confirmed_at + timedelta(minutes=45)
    account = PaperAccount(symbol=symbol)
    account.pending = {
        "execution_type": "pullback", "source": TREND, "side": "long",
        "reason": "trend_pullback_candidate", "signal_time": confirmed_at.isoformat(),
        "confirmed_at": confirmed_at.isoformat(), "execute_at": confirmed_at.isoformat(),
        "expires_at": expires_at.isoformat(), "reference_price": 100.0,
        "pullback_price": 99.0, "stop_price": 98.0, "structure_break_price": 98.5,
        "atr": 1.0, "regime": Regime.LOW.value, "strength": 1.0,
        "hot_universe_exemption": True, "exemption_reason": "trend_confirmed_pullback",
        "exemption_expires_at": expires_at.isoformat(),
        "confirm_bars": 3, "confirmation_pass": True,
    }
    trader = PaperTrader(account, settings)

    changed = trader._advance_pullback_pending(
        100.0, expires_at, 0.0, step_size=0.001, minimum_quantity=0.001,
        quote_fresh=True,
    )

    assert changed is True
    assert account.pending is None
    assert account.position is None
    cancelled = next(event for event in account.events
                     if event.get("event") == "pending_entry_cancelled")
    assert cancelled["reason"] == "timeout"
    assert cancelled["detail"] == "hot_universe_exemption_expired"
    assert cancelled["exemption_status"] == "expired"
    assert cancelled["exemption_expires_at"] == expires_at.isoformat()


def test_demo_trend_route_reaches_real_demo_execution_path():
    settings = load_settings(env_file=None)
    now = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)
    symbol = "AAA/USDT:USDT"
    interval = pd.Timedelta(1, unit="h")
    index = pd.date_range(end=pd.Timestamp(now) - interval, periods=100, freq=interval, name="timestamp")
    bars_1h = pd.DataFrame(
        [(100 + i * 0.2 - 0.08, 100 + i * 0.2 + 0.4, 100 + i * 0.2 - 0.4,
          100 + i * 0.2, 100.0) for i in range(100)],
        columns=["open", "high", "low", "close", "volume"], index=index,
    )
    index_4h = pd.date_range(end=pd.Timestamp(now) - pd.Timedelta(4, unit="h"),
                             periods=100, freq="4h", name="timestamp")
    bars_4h = pd.DataFrame(
        [(100 + i * 0.5 - 0.08, 100 + i * 0.5 + 0.4, 100 + i * 0.5 - 0.4,
          100 + i * 0.5, 100.0) for i in range(100)],
        columns=["open", "high", "low", "close", "volume"], index=index_4h,
    )
    class FakeDemoExecution:
        host = DEMO_REST_ROOT

        def __init__(self):
            self.open_calls = []

        def open(self, intent, *, stop_price):
            self.open_calls.append((intent, stop_price))
            average = float(intent.price)
            quantity = float(intent.quantity)
            return {
                "position": {"entry_price": average, "quantity": quantity,
                             "initial_margin": quantity * average / settings.sizing.leverage},
                "stop": {"stop_price": float(stop_price), "orderId": "demo-stop-1"},
                "fill": {"id": "demo-entry-1", "fee_cost": quantity * average * settings.fees.taker_bps / 10_000},
                "host": self.host,
            }

        def sync_snapshot(self):
            return {"equity": 5_000.0, "available_usdt": 4_000.0, "positions": {}}

    state = MultiPaperState(
        initial_equity=5_000.0, cash=5_000.0, execution_mode="demo_usdm",
        execution_status="synchronized",
        exchange_snapshot={"equity": 5_000.0, "available_usdt": 5_000.0, "positions": {}},
    )
    account = state.for_symbol(symbol)
    execution = FakeDemoExecution()
    trader = PaperTrader(account, settings, execution=execution)
    trader.update(
        bars_1h, bars_4h, current_open_at=now, current_open=119.8,
        current_price=119.8, funding_rate=0.0, now=now, symbol_regime_state="TREND",
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
    )

    assert account.pending is not None
    assert account.pending["source"] == TREND
    assert account.pending["confirm_bars"] == 3
    assert account.pending["confirmation_pass"] is True
    assert account.position is None
    route_event = next(event for event in account.events
                       if event.get("event") == "strategy_route_evaluated")
    assert route_event["entry_allowed"] is True
    assert route_event["signal_source"] == TREND
    assert route_event["execution_mode"] == "demo_usdm"

    assert trader._advance_pullback_pending(
        float(account.pending["pullback_price"]), now + timedelta(minutes=15), 0.0,
        step_size=0.001, minimum_quantity=0.001, permission_valid=True,
        quote_fresh=True,
    ) is True

    assert len(execution.open_calls) == 1
    assert account.position is not None
    assert account.position.source == TREND
    demo_fill = next(event for event in account.events
                     if event.get("event") == "paper_fill" and event.get("action") == "OPEN")
    assert demo_fill["source"] == TREND
    assert demo_fill["execution_source"] == "binance_usdm_demo"
    assert demo_fill["execution_mode"] == "demo_usdm"
    assert any(event.get("event") == "order_intent" and event.get("allowed") is True
               and event.get("source") == TREND and event.get("confirm_bars") == 3
               for event in account.events)


def test_live_or_unverified_execution_cannot_enable_trend_entries():
    settings = load_settings(env_file=None)
    symbol = "AAA/USDT:USDT"
    live_state = MultiPaperState(execution_mode="live")
    live_account = live_state.for_symbol(symbol)
    demo_like_execution = type("DemoLikeExecution", (), {"host": DEMO_REST_ROOT})()
    assert not PaperTrader(live_account, settings, execution=demo_like_execution)._trend_execution_enabled

    local_account = PaperAccount(symbol=symbol)
    assert not PaperTrader(local_account, settings, execution=object())._trend_execution_enabled


def test_demo_mean_reversion_open_remains_enabled():
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    opened_at = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)

    class FakeDemoExecution:
        host = DEMO_REST_ROOT

        def __init__(self):
            self.open_calls = []

        def open(self, intent, *, stop_price):
            self.open_calls.append((intent, stop_price))
            average = float(intent.price)
            quantity = float(intent.quantity)
            return {
                "position": {"entry_price": average, "quantity": quantity,
                             "initial_margin": quantity * average / settings.sizing.leverage},
                "stop": {"stop_price": float(stop_price), "orderId": "demo-stop-mr"},
                "fill": {"id": "demo-entry-mr", "fee_cost": quantity * average * settings.fees.taker_bps / 10_000},
                "host": self.host,
            }

        def sync_snapshot(self):
            return {"equity": 10_000.0, "available_usdt": 8_000.0, "positions": {}}

    state = MultiPaperState(
        initial_equity=10_000.0, cash=10_000.0, execution_mode="demo_usdm",
        execution_status="synchronized",
        exchange_snapshot={"equity": 10_000.0, "available_usdt": 10_000.0, "positions": {}},
    )
    account = state.for_symbol(symbol)
    execution = FakeDemoExecution()
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "low_zone_1h_bullish_close", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value, "source": MEAN_REVERSION,
        "signal_time": (opened_at - timedelta(hours=1)).isoformat(),
    }

    PaperTrader(account, settings, execution=execution)._execute_pending(
        100.0, opened_at, 0.0, step_size=0.001, minimum_quantity=0.001,
    )

    assert len(execution.open_calls) == 1
    assert account.position is not None
    assert account.position.source == MEAN_REVERSION
    fill = next(event for event in account.events
                if event.get("event") == "paper_fill" and event.get("action") == "OPEN")
    assert fill["source"] == MEAN_REVERSION
    assert fill["execution_source"] == "binance_usdm_demo"
    assert fill["execution_mode"] == "demo_usdm"


def test_isolated_margin_liquidation_mark_triggers_but_last_price_fills():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    trader.margin_spec = {
        "maintenance_margin_rate": 0.025,
        "maintenance_amount": 0.0,
        "liquidation_fee_rate": 0.01,
    }
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    allocated = account.isolated_margin["initial_margin"]
    equity_after_open = trader._current_equity()
    assert allocated > 0 and account.cash < account.initial_equity
    assert equity_after_open < account.initial_equity

    trader.monitor_tick(current_price=70.0, mark_price=60.0, now=opened_at + timedelta(minutes=1))
    close = [event for event in account.events if event.get("action") == "CLOSE"][-1]
    assert close["reason"] == "liquidation_mark_trigger"
    assert close["trigger_price_source"] == "mark_price"
    assert close["execution_price_source"] == "last_trade"
    assert close["price"] > 60.0
    assert account.isolated_margin is None
    assert account.position is None
    assert account.cash >= 0


def test_scheduled_paper_entry_uses_current_last_trade_not_hour_open():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")

    def bars(index):
        close = 100 + pd.Series(range(len(index)), index=index) * 0.01
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": 1.0}, index=index)

    now = (index_1h[-1] + pd.Timedelta(1, unit="h")).to_pydatetime()
    account.pending = {
        "execute_at": now.isoformat(), "action": "OPEN", "side": "long",
        "reason": "scheduled_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    account.last_processed_close = now.isoformat()
    trader.update(bars(index_1h), bars(index_4h), current_open_at=now, current_open=100.0,
                  current_price=101.0, funding_rate=0.0, now=now)

    assert account.position is not None
    assert account.position.entry_price > 101.0


def test_profit_protection_moves_to_cost_adjusted_breakeven_then_trails_and_persists(tmp_path):
    settings = load_settings(env_file=None)
    settings.exit_proposal_v1.enabled = False
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    original_stop = account.position.stop_price
    assert account.active_exit["enabled"] is True

    trader.monitor_tick(current_price=103.5, now=opened_at + timedelta(minutes=1))
    breakeven_stop = account.active_exit["protective_stop"]
    assert original_stop < 100 < breakeven_stop < 103.5
    trader.monitor_tick(current_price=105.5, now=opened_at + timedelta(minutes=2))
    trailing_stop = account.active_exit["protective_stop"]
    assert trailing_stop > breakeven_stop
    assert trailing_stop == 103.5

    state_path = tmp_path / "state.json"
    account.save(state_path)
    restored = PaperAccount.load(state_path, symbol=account.symbol)
    assert restored.active_exit == account.active_exit
    PaperTrader(restored, settings).monitor_tick(current_price=102.0, now=opened_at + timedelta(minutes=3))
    assert restored.position is None
    close = [event for event in restored.events if event.get("action") == "CLOSE"][-1]
    assert close["reason"] == "stop_hit_public_price"
    assert close["stop_price"] == 103.5


@pytest.mark.parametrize("replacement_fails", [False, True])
def test_trend_exit_syncs_tightened_demo_stop_and_preserves_old_stop_on_failure(replacement_fails):
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    opened_at = datetime(2026, 10, 17, 0, tzinfo=timezone.utc)
    now = opened_at + timedelta(hours=2)
    position = Position(
        symbol, Side.LONG, 1.0, 100.0, 98.0, opened_at,
        "trend_pullback_candidate", Regime.LOW.value, source=TREND,
    )
    account = PaperAccount(symbol=symbol)
    account.portfolio.positions[symbol] = position
    account.active_exit = {
        "enabled": False,
        "invalidation_enabled": False,
        "initial_risk": 2.0,
        "best_price": 100.0,
        "worst_price": 100.0,
        "mae_history_complete": False,
        "observation_checkpoints": [],
        "protective_stop": 98.0,
        "trend_exit_state": initialize_trend_exit_state(100.0, 98.0, 90.0),
    }
    account.exchange_orders_by_symbol = {
        symbol: {"stop": {"id": "old-stop", "stop_price": 98.0}},
    }

    index_1h = pd.date_range(end=now - timedelta(hours=1), periods=60, freq="1h", name="timestamp")
    rows_1h = [(100.0, 100.5, 99.5, 100.0, 1.0)] * 59
    rows_1h.append((100.0, 106.0, 99.5, 105.5, 1.0))
    bars_1h = pd.DataFrame(rows_1h, columns=["open", "high", "low", "close", "volume"], index=index_1h)
    index_4h = pd.date_range(end=now - timedelta(hours=4), periods=60, freq="4h", name="timestamp")
    bars_4h = pd.DataFrame(
        [(100.0, 100.5, 99.5, 100.0, 1.0)] * len(index_4h),
        columns=["open", "high", "low", "close", "volume"], index=index_4h,
    )

    class Exchange:
        @staticmethod
        def price_to_precision(symbol, price):
            return f"{price:.2f}"

    class Execution:
        host = "https://demo-fapi.binance.com"
        exchange = Exchange()

        def __init__(self):
            self.replacements = []

        def replace_stop(self, position, *, stop, stop_price):
            self.replacements.append((stop, stop_price))
            if replacement_fails:
                raise RuntimeError("replacement rejected")
            return {"id": "new-stop", "stop_price": float(stop_price)}

    execution = Execution()
    trader = PaperTrader(account, settings, execution=execution)
    trader.update(
        bars_1h, bars_4h, current_open_at=now, current_open=105.5,
        current_price=105.5, funding_rate=0.0, now=now,
        symbol_regime_state="TREND", market_risk_state="NORMAL",
    )

    assert len(execution.replacements) == 1
    assert execution.replacements[0][0]["id"] == "old-stop"
    if replacement_fails:
        assert account.active_exit["protective_stop"] == 98.0
        assert account.active_exit["trend_exit_state"]["active_stop_price"] == 98.0
        assert account.active_exit["trend_exit_state"]["active_stop_reason"] == "structure_stop"
        assert account.execution_status == "inconsistent"
        assert any(event.get("event") == "protective_stop_update_failed" for event in account.events)
    else:
        new_stop = account.exchange_orders_by_symbol[symbol]["stop"]
        assert new_stop["id"] == "new-stop"
        assert account.active_exit["protective_stop"] == new_stop["stop_price"]
        assert account.active_exit["trend_exit_state"]["active_stop_price"] == new_stop["stop_price"]
        assert account.active_exit["trend_exit_state"]["active_stop_reason"] == "trailing_stop"
        assert any(event.get("event") == "protective_stop_updated"
                   and event.get("reason") == "trend_trailing_stop"
                   for event in account.events)


def test_management_checkpoints_persist_metrics_once_and_closed_trade_has_excursions(tmp_path):
    settings = load_settings(env_file=None)
    settings.exit_proposal_v1.enabled = False
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)

    trader.monitor_tick(current_price=99.5, now=opened_at + timedelta(hours=1))
    trader.monitor_tick(
        current_price=99.5, now=opened_at + timedelta(hours=12),
        observation_context={"hotlist_rank": 7, "market_regime": "LOW"},
    )
    checkpoint = next(event for event in account.events if event["event"] == "management_observation_checkpoint")
    assert checkpoint["checkpoint"] == "12h"
    assert checkpoint["condition_met"] is True
    assert checkpoint["mfe_r"] == 0.0
    assert checkpoint["mae_available"] is True
    assert checkpoint["mae_r"] > 0
    assert checkpoint["hotlist_rank"] == 7
    assert checkpoint["market_regime"] == "LOW"
    assert checkpoint["time_exit_enabled"] is False
    assert account.position is not None

    state_path = tmp_path / "portfolio.json"
    account.save(state_path)
    restored = PaperAccount.load(state_path, symbol=account.symbol)
    restored_trader = PaperTrader(restored, settings)
    restored_trader.monitor_tick(current_price=99.5, now=opened_at + timedelta(hours=13))
    assert sum(event.get("checkpoint") == "12h" for event in restored.events
               if event.get("event") == "management_observation_checkpoint") == 1

    restored_trader.monitor_tick(current_price=99.5, now=opened_at + timedelta(hours=24))
    checkpoints = [event for event in restored.events if event.get("event") == "management_observation_checkpoint"]
    assert [event["checkpoint"] for event in checkpoints] == ["12h", "24h"]
    assert checkpoints[-1]["condition_met"] is True
    assert restored.position is not None

    restored_trader._close(99.5, opened_at + timedelta(hours=25), "manual_close")
    close = [event for event in restored.events if event.get("action") == "CLOSE"][-1]
    assert close["entry_reason"] == "synthetic_entry"
    assert close["entry_regime"] == Regime.LOW.value
    assert close["held_hours"] == 25
    assert close["mfe_available"] is True
    assert close["mae_available"] is True
    assert close["mae_r"] > 0


def test_legacy_open_position_keeps_fixed_stop_without_excursion_history():
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    account = PaperAccount(symbol=symbol)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    position = Position(symbol, Side.LONG, 1.0, 100.0, 98.0, opened_at, "legacy", Regime.LOW.value)
    account.portfolio.positions[symbol] = position

    PaperTrader(account, settings).monitor_tick(current_price=110.0, now=opened_at + timedelta(minutes=1))

    assert account.position == position
    assert account.active_exit["enabled"] is False
    assert account.active_exit["protective_stop"] == 98.0


def test_time_stop_12h_exits_stagnant_trade_after_stop_check():
    settings = load_settings(env_file=None)
    settings.exit_proposal_v1.enabled = False
    settings.exit_management.time_exit_enabled = True
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)

    changed = trader.monitor_tick(current_price=100.0, now=opened_at + timedelta(hours=12))

    assert changed and account.position is None
    close = [event for event in account.events if event.get("action") == "CLOSE"][-1]
    assert close["reason"] == "time_stop_12h_no_progress"


def test_time_exit_24h_only_closes_before_profit_protection_threshold():
    settings = load_settings(env_file=None)
    settings.exit_proposal_v1.enabled = False
    settings.exit_management.time_exit_enabled = True
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)

    def open_long():
        account = PaperAccount(symbol="ADA/USDT:USDT")
        trader = PaperTrader(account, settings)
        account.pending = {
            "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
            "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
            "regime": Regime.LOW.value,
        }
        trader._execute_pending(100.0, opened_at, 0.0)
        return account, trader

    account, trader = open_long()
    trader.monitor_tick(current_price=102.0, now=opened_at + timedelta(hours=24))
    close = [event for event in account.events if event.get("action") == "CLOSE"][-1]
    assert close["reason"] == "time_exit_24h_unprotected"

    protected, protected_trader = open_long()
    protected_trader.monitor_tick(current_price=103.1, now=opened_at + timedelta(hours=24))
    assert protected.position is not None
    assert protected.active_exit["protective_stop"] > protected.position.entry_price


def test_existing_protective_stop_has_priority_over_time_exit():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)

    trader.monitor_tick(current_price=97.0, now=opened_at + timedelta(hours=12))

    close = [event for event in account.events if event.get("action") == "CLOSE"][-1]
    assert close["reason"] == "stop_hit_public_price"


@pytest.mark.parametrize(
    ("exit_state", "expected_close"),
    [
        ({"tp1_taken": False, "breakeven_moved": False, "protective_stop": 95.0}, True),
        ({"tp1_taken": True, "breakeven_moved": False, "protective_stop": 95.0}, True),
        ({"tp1_taken": True, "breakeven_moved": True, "protective_stop": 100.0}, True),
        ({"tp1_taken": True, "breakeven_moved": True, "protective_stop": 101.0}, True),
        ({"trailing_started": True, "trailing_trigger_r": 2.0,
          "protective_stop": 101.0}, False),
    ],
)
def test_max_hold_exempts_only_persisted_2r_trailing_positions(
    monkeypatch, exit_state, expected_close,
):
    settings = load_settings(env_file=None)
    opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    position = Position(account.symbol, Side.LONG, 1.0, 100.0, 95.0,
                        opened_at, "entry", Regime.LOW.value)
    account.portfolio.positions[account.symbol] = position
    account.active_exit = exit_state
    trader = PaperTrader(account, settings)
    close_calls = []
    monkeypatch.setattr(trader, "_close", lambda price, at, reason: close_calls.append(reason) or True)

    closed = trader.check_max_hold(102.0, opened_at + timedelta(hours=25))

    assert closed is expected_close
    assert close_calls == (["max_hold_24h"] if expected_close else [])


def test_2r_trailing_runner_has_no_runner_time_limit():
    settings = load_settings(env_file=None)
    opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    position = Position(account.symbol, Side.LONG, 0.2, 100.0, 95.0,
                        opened_at, "entry", Regime.LOW.value)
    account.portfolio.positions[account.symbol] = position
    account.active_exit = {
        "trailing_started": True, "trailing_trigger_r": 2.0,
        "protective_stop": 102.0, "no_progress_count": 4,
    }
    trader = PaperTrader(account, settings)

    assert settings.exits.runner_max_hold_bars == 0
    assert trader.check_max_hold(105.0, opened_at + timedelta(days=10)) is False
    assert account.position is position
    assert account.active_exit["no_progress_count"] == 4


def test_exit_proposal_v1_partially_closes_tp1_and_preserves_remaining_position():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value, "entry_signal_low": 97.0,
        "entry_signal_high": 101.0, "entry_signal_close": 100.0,
        "low_zone_lower": 98.1, "entry_midpoint": 100.0,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    initial_quantity = account.position.quantity

    trader.monitor_tick(current_price=101.1, now=opened_at + timedelta(minutes=1))

    assert account.position is not None
    assert account.position.quantity == pytest.approx(initial_quantity * 0.6)
    assert account.active_exit["tp1_taken"] is True
    assert account.active_exit["protective_stop"] > account.position.entry_price
    event = next(event for event in account.events if event.get("event") == "paper_partial_fill")
    assert event["reason"] == "take_profit_1"
    assert event["quantity"] == pytest.approx(initial_quantity * 0.4)


def test_wide_structural_stop_rejects_entry_instead_of_reducing_fixed_margin():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value, "entry_signal_low": 97.0,
        "entry_signal_high": 101.0, "entry_signal_close": 100.0,
        "low_zone_lower": 97.0, "entry_midpoint": 100.0, "entry_structure_low": 90.0,
    }

    trader._execute_pending(100.0, opened_at, 0.0)

    assert account.position is None
    assert account.events[-1]["reason"] == "stop_too_wide"


@pytest.mark.parametrize(
    ("side", "close", "structure_low", "structure_high", "range_low", "range_high",
     "stop", "entry_regime", "live_regime", "reason"),
    [
        (Side.LONG, 94.7, 95.0, None, None, None, 94.4, Regime.LOW, Regime.LOW, "long_structure_invalid"),
        (Side.SHORT, 105.3, None, 105.0, None, None, 105.6, Regime.HIGH, Regime.HIGH, "short_structure_invalid"),
        (Side.LONG, 94.8, None, None, 95.0, None, 94.4, Regime.LOW, Regime.LOW, "max_hold_24h"),
        (Side.SHORT, 105.2, None, None, None, 105.0, 105.6, Regime.HIGH, Regime.HIGH, "max_hold_24h"),
        (Side.LONG, 100.0, None, None, None, None, 90.0, Regime.LOW, Regime.SIDEWAYS, "long_regime_faded"),
    ],
)
def test_entry_structure_break_closes_with_side_specific_reason(
    monkeypatch, side, close, structure_low, structure_high, range_low, range_high,
    stop, entry_regime, live_regime, reason,
):
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    position = Position(account.symbol, side, 1.0, 100.0, stop, opened_at, "entry", entry_regime.value)
    account.portfolio.positions[account.symbol] = position
    account.active_exit = {
        "enabled": False, "invalidation_enabled": True, "entry_atr": 1.0,
        "entry_structure_low": structure_low, "entry_structure_high": structure_high,
        "entry_range_low": range_low, "entry_range_high": range_high,
        "no_progress_count": 0, "no_progress_last_close": 100.0,
        "best_price": 100.0, "protective_stop": stop,
    }
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
                        "volume": 1.0}, index=index_1h)
    one.iloc[-1, one.columns.get_loc("close")] = close
    one.iloc[-1, one.columns.get_loc("high")] = max(100.2, close)
    one.iloc[-1, one.columns.get_loc("low")] = min(99.8, close)
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1.0}, index=index_4h)
    regimes = pd.DataFrame({
        "atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
        "range_position": 0.2, "new_low": False, "new_high": False,
        "range_low": 90.0, "range_high": 110.0, "regime": live_regime,
    }, index=index_4h)
    monkeypatch.setattr(paper, "classify_regimes", lambda *_args, **_kwargs: regimes)
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    now = (index_1h[-1] + pd.Timedelta(1, unit="h")).to_pydatetime()
    PaperTrader(account, settings).update(
        one, four, current_open_at=now, current_open=close, current_price=close,
        funding_rate=0.0, now=now,
    )

    close_event = next(event for event in reversed(account.events)
                       if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close_event["reason"] == reason


def test_execution_rejects_deep_long_range_break_without_intent_or_position():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="US/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 10, 14, 14, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "low_zone_1h_bullish_close", "strength": 1.0, "atr": 0.000455,
        "regime": Regime.LOW.value, "entry_range_low": 0.008255,
        "entry_range_high": 0.036397, "entry_structure_low": 0.0075,
        "entry_structure_high": 0.0085,
    }

    trader._execute_pending(0.00773, opened_at, 0.0)

    assert account.position is None
    assert not any(event.get("event") == "order_intent" for event in account.events)
    assert next(event for event in account.events if event.get("event") == "intent_rejected")["reason"] == "long_rejected_below_range"


def test_range_below_price_alone_does_not_invalidate_open_position():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="US/USDT:USDT")
    opened_at = datetime(2026, 10, 14, 14, tzinfo=timezone.utc)
    position = Position(account.symbol, Side.LONG, 1.0, 0.0084, 0.0072, opened_at,
                        "entry", Regime.LOW.value)
    state = {
        "entry_atr": 0.000455, "entry_structure_low": 0.0075,
        "entry_range_low": 0.008255,
    }

    trader = PaperTrader(account, settings)
    assert settings.exits.structure_invalid_buffer_atr == 0.25
    assert trader._price_invalidation_reason(position, state, 0.00773) is None
    assert trader._price_invalidation_reason(position, state, 0.0074) is None
    assert trader._price_invalidation_reason(position, state, 0.00738) == "long_structure_invalid"


@pytest.mark.parametrize(
    ("side", "structure", "stop", "just_through", "past_buffer", "reason"),
    [
        (Side.LONG, 95.0, 94.4, 94.9, 94.74, "long_structure_invalid"),
        (Side.SHORT, 105.0, 105.6, 105.1, 105.26, "short_structure_invalid"),
    ],
)
def test_structure_exit_waits_for_quarter_atr_buffer(side, structure, stop, just_through,
                                                       past_buffer, reason):
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    position = Position(account.symbol, side, 1.0, 100.0, stop,
                       datetime(2026, 1, 1, tzinfo=timezone.utc), "entry",
                       Regime.LOW.value if side is Side.LONG else Regime.HIGH.value)
    key = "entry_structure_low" if side is Side.LONG else "entry_structure_high"
    state = {"entry_atr": 1.0, key: structure, "protective_stop": stop}
    trader = PaperTrader(account, settings)

    assert trader._price_invalidation_reason(position, state, just_through) is None
    assert trader._price_invalidation_reason(position, state, past_buffer) == reason


def test_protective_stop_caps_looser_structure_invalidation():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    position = Position(account.symbol, Side.LONG, 1.0, 100.0, 96.0,
                        datetime(2026, 1, 1, tzinfo=timezone.utc), "entry", Regime.LOW.value)
    state = {"entry_atr": 1.0, "entry_structure_low": 95.0, "protective_stop": 96.0}
    trader = PaperTrader(account, settings)

    levels = trader._structure_invalidation_levels(position, state)

    assert levels["structure_point"] == 95.0
    assert levels["buffer"] == 0.25
    assert levels["invalid_price"] == 94.75
    assert levels["trigger_price"] == 96.0
    assert levels["trigger_source"] == "protective_stop"
    assert levels["structure_invalid_is_looser"] is True
    assert trader._price_invalidation_reason(position, state, 94.7) is None


def test_six_nonprogress_hourly_closes_emit_no_progress_exit(monkeypatch):
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    opened_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    position = Position(account.symbol, Side.LONG, 1.0, 100.0, 90.0, opened_at, "entry", Regime.LOW.value)
    account.portfolio.positions[account.symbol] = position
    account.active_exit = {
        "enabled": False, "invalidation_enabled": True, "entry_atr": 1.0,
        "entry_structure_low": None, "entry_structure_high": None,
        "entry_range_low": None, "entry_range_high": None,
        "no_progress_count": 5, "no_progress_last_close": 100.0,
        "best_price": 100.0, "protective_stop": 100.0,
        "tp1_taken": True, "breakeven_moved": True,
    }
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
                        "volume": 1.0}, index=index_1h)
    one.iloc[-1, one.columns.get_loc("close")] = 99.5
    one.iloc[-1, one.columns.get_loc("low")] = 99.5
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1.0}, index=index_4h)
    regimes = pd.DataFrame({
        "atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
        "range_position": 0.2, "new_low": False, "new_high": False,
        "range_low": 90.0, "range_high": 110.0, "regime": Regime.LOW,
    }, index=index_4h)
    monkeypatch.setattr(paper, "classify_regimes", lambda *_args, **_kwargs: regimes)
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    now = (index_1h[-1] + pd.Timedelta(1, unit="h")).to_pydatetime()
    PaperTrader(account, settings).update(
        one, four, current_open_at=now, current_open=100.1, current_price=100.1,
        funding_rate=0.0, now=now,
    )
    close_event = next(event for event in reversed(account.events)
                       if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close_event["reason"] == "no_progress"


@pytest.mark.parametrize(
    ("side", "best_price", "current_price"),
    [("long", 102.0, 100.5), ("short", 98.0, 100.5)],
)
def test_exit_proposal_v1_tp1_requires_current_price_to_remain_at_target(side, best_price, current_price):
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": side,
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value if side == "long" else Regime.HIGH.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    quantity = account.position.quantity
    account.active_exit["best_price"] = best_price

    trader._manage_proposal_tick(current_price, opened_at + timedelta(minutes=1))

    assert account.position.quantity == quantity
    assert account.active_exit["tp1_taken"] is False
    assert not any(event.get("reason") == "take_profit_1" for event in account.events)


def test_exit_proposal_v1_tp2_requires_current_price_and_skips_triggered_side_trail():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    quantity = account.position.quantity
    old_stop = account.active_exit["protective_stop"]
    account.active_exit.update(tp1_taken=True, best_price=103.0)

    class Exchange:
        @staticmethod
        def price_to_precision(symbol, price):
            return str(price)

    class Execution:
        exchange = Exchange()

        @staticmethod
        def replace_stop(*args, **kwargs):
            raise AssertionError("must not replace a stop already crossed by current price")

    trader.execution = Execution()
    account.exchange_orders_by_symbol = {}
    account.exchange_orders_by_symbol[account.symbol] = {
        "stop": {"id": "existing-stop", "stop_price": old_stop},
    }

    trader._manage_proposal_tick(101.0, opened_at + timedelta(minutes=1))

    assert account.position.quantity == quantity
    assert account.active_exit["tp2_taken"] is False
    assert account.active_exit["protective_stop"] == old_stop
    skipped = [event for event in account.events if event.get("event") == "protective_stop_update_skipped"]
    assert skipped[-1]["stop_price"] >= 101.0
    assert skipped[-1]["current_price"] == 101.0


def test_closed_bar_is_processed_once():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    index_4h = pd.date_range("2025-12-15T00:00:00Z", periods=100, freq="4h")

    def bars(index):
        close = 100 + pd.Series(range(len(index)), index=index) * 0.01
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=index)

    one, four = bars(index_1h), bars(index_4h)
    now = (index_1h[-1] + pd.Timedelta(3_600_000_000_000, unit="ns")).to_pydatetime()
    kwargs = dict(current_open_at=now, current_open=101.0, current_price=101.0, funding_rate=0.0, now=now)
    trader.update(one, four, **kwargs)
    processed_at = account.last_processed_close
    event_count = len(account.events)
    trader.update(one, four, **kwargs)
    assert processed_at is not None
    assert account.last_processed_close == processed_at
    assert len(account.events) == event_count


def test_manual_close_uses_paper_fees_slippage_and_cancels_pending_signal():
    settings = load_settings(env_file=None)
    account = PaperAccount(symbol="ADA/USDT:USDT")
    trader = PaperTrader(account, settings)
    trader.margin_spec = {
        "maintenance_margin_rate": 0.025,
        "maintenance_amount": 0.0,
        "liquidation_fee_rate": 0.01,
    }
    opened_at = datetime(2026, 9, 29, 1, tzinfo=timezone.utc)
    account.pending = {
        "execute_at": opened_at.isoformat(), "action": "OPEN", "side": "long",
        "reason": "synthetic_entry", "strength": 1.0, "atr": 1.0,
        "regime": Regime.LOW.value,
    }
    trader._execute_pending(100.0, opened_at, 0.0)
    account.pending = {"action": "OPEN", "reason": "stale_signal"}

    assert trader.manual_close(105.0, opened_at + timedelta(minutes=10), request_id="request-1") is True

    close_fill = next(event for event in reversed(account.events)
                      if event.get("event") == "paper_fill" and event.get("action") == "CLOSE")
    assert close_fill["reason"] == "manual_close"
    assert close_fill["execution_price_source"] == "last_trade"
    assert close_fill["fees"] > 0
    assert close_fill["slippage_paid"] > 0
    assert account.position is None
    assert account.isolated_margin is None
    assert account.pending is None
    assert any(event.get("event") == "signal_cancelled" for event in account.events)
    assert account.events[-1]["event"] == "manual_close_request_processed"
    assert trader.manual_close(105.0, opened_at + timedelta(minutes=11), request_id="request-2") is False


def test_paper_runner_audits_and_blocks_candidate_when_time_filter_data_is_unavailable(monkeypatch):
    settings = load_settings(env_file=None)
    signal_close_at = datetime(2026, 10, 26, 2, tzinfo=timezone.utc)
    index_1h = pd.date_range(end=pd.Timestamp(signal_close_at) - pd.Timedelta(1, unit="h"),
                             periods=60, freq="1h", name="timestamp")
    index_4h = pd.date_range(end=pd.Timestamp(signal_close_at) - pd.Timedelta(4, unit="h"),
                             periods=60, freq="4h", name="timestamp")

    def bars(index):
        close = pd.Series(100.0, index=index)
        return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": 1.0}, index=index)

    regime_index = pd.DatetimeIndex([pd.Timestamp("2026-10-25T20:00:00Z")])
    regimes = pd.DataFrame({"regime": [Regime.HIGH], "range_low": [90.0],
                            "range_high": [110.0]}, index=regime_index)
    monkeypatch.setattr(paper, "classify_regimes", lambda *_args: regimes)
    monkeypatch.setattr(paper, "_snapshot", lambda _row, timestamp: RegimeSnapshot(
        timestamp, Regime.HIGH, 1.0, 25.0, 50.0, 0.8, False, False,
    ))
    monkeypatch.setattr(paper, "atr", lambda frame, _period: pd.Series(1.0, index=frame.index))
    monkeypatch.setattr(paper, "evaluate_time_filter", lambda *_args: TimeFilterFlags(
        True, True, False, False, data_ready=False, data_reason="hourly_history_gap",
    ))

    def synthetic_signal(_regime, _candle, flags, _settings, **_kwargs):
        if flags.data_ready and flags.allow_short_entry:
            return Signal(Side.SHORT, Action.OPEN, "synthetic_mr_candidate", 1.0)
        return Signal(None, Action.NONE, "time_filter_data_unavailable", 0.0)

    monkeypatch.setattr(paper, "generate_signal", synthetic_signal)
    account = PaperAccount(symbol="REZ/USDT:USDT", initial_equity=5_000.0, cash=5_000.0)

    PaperTrader(account, settings).update(
        bars(index_1h), bars(index_4h), current_open_at=signal_close_at,
        current_open=100.0, current_price=100.0, funding_rate=0.0,
        now=signal_close_at, allow_entries=True, symbol_in_hot_universe=True,
        entry_execution_allowed=True, mean_reversion_only=True,
        market_risk_state="NORMAL", step_size=0.001, minimum_quantity=0.001,
    )

    assert account.position is None
    assert account.pending is None
    decision = next(event for event in account.events
                    if event.get("event") == "entry_time_filter_decision")
    assert decision["decision"] == "BLOCK"
    assert decision["reason"] == "hourly_history_gap"
    assert decision["side"] == "short"
    assert decision["bar_close_at"] == signal_close_at.isoformat()

