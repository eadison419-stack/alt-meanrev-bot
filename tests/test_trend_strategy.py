from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.config import load_settings
from bot.portfolio import Position
from bot.signals import Action, Side
from bot.strategy_router import MEAN_REVERSION, ROUTER, TREND
from bot.trend_strategy import (
    calculate_trend_quantity,
    calculate_trend_stop,
    evaluate_trend_entry,
    evaluate_trend_exit,
    initialize_trend_exit_state,
)


NOW = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)


def make_bars(timeframe: str, kind: str = "up", count: int = 100) -> pd.DataFrame:
    delta = pd.Timedelta(4 if timeframe == "4h" else 1, unit="h")
    index = pd.date_range(end=pd.Timestamp(NOW) - delta, periods=count, freq=delta, name="timestamp")
    rows = []
    for i in range(count):
        if kind == "up":
            close = 100 + i * (0.5 if timeframe == "4h" else 0.2)
            open_price, high, low = close - 0.08, close + 0.4, close - 0.4
        elif kind == "down":
            close = 150 - i * (0.5 if timeframe == "4h" else 0.2)
            open_price, high, low = close + 0.08, close + 0.4, close - 0.4
        else:
            close = 100 + i * 0.2
            open_price, high, low = close - 0.08, close + 0.4, close - 0.4
        rows.append((open_price, high, low, close, 100.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)


def entry_decision(kind: str = "up", **kwargs):
    return evaluate_trend_entry(
        "AAA/USDT:USDT", make_bars("1h", kind), make_bars("4h", kind),
        load_settings(env_file=None), now=NOW, equity=10_000,
        step_size=0.1, **kwargs,
    )


@pytest.mark.parametrize(("kind", "side"), [("up", Side.LONG), ("down", Side.SHORT)])
def test_entry_plan_uses_trend_source_closed_signal_and_trend_ttl(kind, side):
    decision = entry_decision(kind)

    assert decision.signal.action is Action.OPEN
    assert decision.signal.source == TREND
    assert decision.signal.symbol == "AAA/USDT:USDT"
    assert decision.signal.side is side
    assert decision.signal.timestamp == NOW
    assert decision.signal_bar_time == NOW
    assert decision.expires_at == datetime(2026, 10, 16, 8, 45, tzinfo=timezone.utc)
    assert decision.quantity > 0
    assert decision.stop_price < decision.pullback_price if side is Side.LONG else decision.stop_price > decision.pullback_price
    assert decision.evidence["pullback_ttl_minutes"] == 45
    assert decision.route["entry_allowed"] is True


def test_entry_ignores_unclosed_one_hour_candle():
    settings = load_settings(env_file=None)
    bars_1h = make_bars("1h")
    open_bar = pd.DataFrame(
        [[1.0, 200.0, 0.5, 1.0, 1.0]],
        columns=bars_1h.columns,
        index=[pd.Timestamp(NOW)],
    )
    baseline = evaluate_trend_entry(
        "AAA/USDT:USDT", bars_1h, make_bars("4h"), settings,
        now=NOW, equity=10_000, step_size=0.1,
    )
    with_open_bar = evaluate_trend_entry(
        "AAA/USDT:USDT", pd.concat([bars_1h, open_bar]), make_bars("4h"), settings,
        now=NOW, equity=10_000, step_size=0.1,
    )

    assert with_open_bar.signal == baseline.signal
    assert with_open_bar.pullback_price == baseline.pullback_price
    assert with_open_bar.stop_price == baseline.stop_price
    assert with_open_bar.quantity == baseline.quantity


def test_entry_ttl_is_trend_specific_and_old_candidate_expires_at_next_hour():
    settings = load_settings(env_file=None)
    settings = settings.model_copy(update={
        "execution": settings.execution.model_copy(update={"confirm_delay_sec_max": 1})
    })
    delayed = evaluate_trend_entry(
        "AAA/USDT:USDT", make_bars("1h"), make_bars("4h"), settings,
        now=NOW.replace(minute=30), equity=10_000, step_size=0.1,
    )
    expired = evaluate_trend_entry(
        "AAA/USDT:USDT", make_bars("1h"), make_bars("4h"), settings,
        now=NOW.replace(hour=9), equity=10_000, step_size=0.1,
    )

    assert delayed.signal.action is Action.OPEN
    assert delayed.expires_at == datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)
    assert expired.signal.action is Action.NONE
    assert expired.signal.reason == "trend_pullback_expired"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"market_risk_state": "PAUSE_NEW"}, "market_risk_paused"),
        ({"position_source": MEAN_REVERSION}, "position_owned_by_other_strategy"),
        ({"has_open_position": True}, "position_already_open"),
        ({"pending_source": MEAN_REVERSION}, "pending_entry_exists"),
    ],
)
def test_entry_respects_pause_and_symbol_conflict_gates(overrides, reason):
    decision = entry_decision(**overrides)

    assert decision.signal.action is Action.NONE
    assert decision.signal.source == ROUTER
    assert decision.signal.reason == reason


@pytest.mark.parametrize("symbol", ["BTC/USDT:USDT", "ETHUSDT"])
def test_entry_respects_market_universe_exclusions(symbol):
    settings = load_settings(env_file=None)
    decision = evaluate_trend_entry(
        symbol, make_bars("1h"), make_bars("4h"), settings,
        now=NOW, equity=10_000, step_size=0.1,
    )

    assert decision.signal.action is Action.NONE
    assert decision.signal.reason == "trend_symbol_excluded"


def test_entry_respects_configured_symbol_exclusions():
    settings = load_settings(env_file=None)
    settings = settings.model_copy(update={
        "universe": settings.universe.model_copy(update={"exclude_symbols": ["EXCLUDEDUSDT"]})
    })
    decision = evaluate_trend_entry(
        "EXCLUDED/USDT:USDT", make_bars("1h"), make_bars("4h"), settings,
        now=NOW, equity=10_000, step_size=0.1,
    )

    assert decision.signal.reason == "trend_symbol_excluded"


def test_entry_does_not_generate_from_uncertain_or_misaligned_data():
    settings = load_settings(env_file=None)
    bars_4h = make_bars("4h")
    bars_4h.iloc[-1, bars_4h.columns.get_loc("low")] = 1.0
    uncertain = evaluate_trend_entry(
        "AAA/USDT:USDT", make_bars("1h"), bars_4h, settings,
        now=NOW, equity=10_000, step_size=0.1,
    )
    assert uncertain.signal.action is Action.NONE

    bars_1h = make_bars("1h")
    last = bars_1h.iloc[-1].copy()
    last["open"] = last["close"] + 0.1
    last["low"] = last["close"] - 0.4
    last["high"] = last["open"] + 0.2
    bars_1h.iloc[-1] = last
    misaligned = evaluate_trend_entry(
        "AAA/USDT:USDT", bars_1h, make_bars("4h"), settings,
        now=NOW, equity=10_000, step_size=0.1,
    )
    assert misaligned.signal.action is Action.NONE
    assert misaligned.signal.reason == "trend_1h_not_aligned_up"


def test_stop_distance_and_risk_sizing_include_costs_caps_and_lot_rounding():
    settings = load_settings(env_file=None)
    bars = make_bars("1h").tail(6).copy()
    bars["low"] = [97.0, 97.1, 97.2, 97.3, 97.4, 97.5]
    stop, structure = calculate_trend_stop(Side.LONG, 100.0, 2.0, bars, settings)
    assert structure == 97.0
    assert stop == pytest.approx(96.2)

    atr_settings = settings.model_copy(update={
        "trend_strategy": settings.trend_strategy.model_copy(update={"stop_mode": "atr"})
    })
    atr_stop, _ = calculate_trend_stop(Side.SHORT, 100.0, 2.0, bars, atr_settings)
    assert atr_stop == pytest.approx(104.0)

    quantity, budget, risk_per_unit = calculate_trend_quantity(
        10_000, 100, 98, settings, step_size=0.1,
    )
    assert budget == pytest.approx(50)
    assert risk_per_unit == pytest.approx(2 + 198 * 0.0008)
    assert quantity == pytest.approx(23.1)

    capped, _, _ = calculate_trend_quantity(
        10_000, 100, 98, settings, step_size=0.1, gross_notional_cap_pct=0.01,
    )
    below_minimum, _, _ = calculate_trend_quantity(
        10_000, 100, 98, settings, step_size=0.1, minimum_quantity=30,
    )
    assert capped == pytest.approx(1.0)
    assert below_minimum == 0


def test_stop_rejects_structure_risk_outside_configured_atr_bounds():
    settings = load_settings(env_file=None)
    bars = make_bars("1h").tail(6).copy()
    bars["low"] = 90
    with pytest.raises(ValueError, match="trend_stop_distance_out_of_range"):
        calculate_trend_stop(Side.LONG, 100, 1, bars, settings)
    assert initialize_trend_exit_state(100, 98, 95, stop_mode="atr")["active_stop_reason"] == "atr_stop"


def test_exit_uses_only_full_bars_after_entry_and_closes_on_ema50_break():
    settings = load_settings(env_file=None)
    bars = make_bars("1h", count=60)
    last = bars.iloc[-1].copy()
    last["open"], last["high"], last["low"], last["close"] = 108.0, 108.2, 104.0, 105.0
    bars.iloc[-1] = last
    partial = bars.iloc[-2].copy()
    partial["low"] = 97.0
    bars.iloc[-2] = partial
    opened_at = bars.index[-2].to_pydatetime()
    position = Position(
        "AAA/USDT:USDT", Side.LONG, 1, 110, 100, opened_at,
        "trend_test", "TREND", strategy_source=TREND,
    )
    state = initialize_trend_exit_state(110, 100, 90)

    decision = evaluate_trend_exit(position, bars, settings, now=NOW, exit_state=state)

    assert decision.signal.action is Action.CLOSE
    assert decision.signal.reason == "trend_key_ema_break"
    assert decision.signal.source == TREND
    assert decision.signal.timestamp == NOW
    assert decision.exit_state["last_processed_close"] == NOW.isoformat()


def test_short_exit_mirrors_ema50_close_break():
    settings = load_settings(env_file=None)
    bars = make_bars("1h", kind="down", count=60)
    last = bars.iloc[-1].copy()
    last["open"], last["high"], last["low"], last["close"] = 139.0, 146.0, 138.5, 145.0
    bars.iloc[-1] = last
    opened_at = bars.index[50].to_pydatetime()
    position = Position(
        "AAA/USDT:USDT", Side.SHORT, 1, 140, 150, opened_at,
        "trend_test", "TREND", strategy_source=TREND,
    )
    state = initialize_trend_exit_state(140, 150, 160)

    decision = evaluate_trend_exit(position, bars, settings, now=NOW, exit_state=state)

    assert decision.signal.action is Action.CLOSE
    assert decision.signal.side is Side.SHORT
    assert decision.signal.reason == "trend_key_ema_break"


def test_exit_raises_breakeven_then_trailing_stop_without_widening():
    settings = load_settings(env_file=None)
    bars = make_bars("1h")
    position = Position(
        "AAA/USDT:USDT", Side.LONG, 1, 100, 98, bars.index[0].to_pydatetime(),
        "trend_test", "TREND", strategy_source=TREND,
    )
    state = initialize_trend_exit_state(100, 98, 90)

    decision = evaluate_trend_exit(position, bars, settings, now=NOW, exit_state=state)

    assert decision.signal.action is Action.NONE
    assert decision.exit_state["active_stop_reason"] == "trailing_stop"
    assert decision.active_stop_price > 100
    triggered = evaluate_trend_exit(
        position, bars, settings, now=NOW, exit_state=decision.exit_state,
        current_price=decision.active_stop_price - 0.01,
    )
    assert triggered.signal.action is Action.CLOSE
    assert triggered.signal.reason == "trend_trailing_stop_hit"


def test_exit_live_stop_is_trend_owned_and_other_strategy_is_untouched():
    settings = load_settings(env_file=None)
    position = Position(
        "AAA/USDT:USDT", Side.LONG, 1, 100, 98, NOW,
        "test", "TREND", strategy_source=TREND,
    )
    state = initialize_trend_exit_state(100, 98, 95)
    stopped = evaluate_trend_exit(
        position, None, settings, now=NOW, exit_state=state, current_price=97,
    )
    assert stopped.signal.action is Action.CLOSE
    assert stopped.signal.reason == "trend_structure_stop_hit"

    mean_reversion_position = Position(
        "BBB/USDT:USDT", Side.LONG, 1, 100, 98, NOW,
        "test", "LOW", strategy_source=MEAN_REVERSION,
    )
    untouched = evaluate_trend_exit(
        mean_reversion_position, None, settings, now=NOW,
        exit_state=initialize_trend_exit_state(100, 98, 95), current_price=97,
    )
    assert untouched.signal.action is Action.NONE
    assert untouched.signal.source == ROUTER
    assert untouched.signal.reason == "position_not_owned_by_trend"
