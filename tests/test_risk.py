from datetime import datetime, timedelta
from dataclasses import asdict
from zoneinfo import ZoneInfo

import pytest

from bot.config import load_settings
from bot.portfolio import PortfolioState, Position, classify_exit_category
from bot.regime import Regime
from bot.risk import (
    OrderIntent,
    RiskDecision,
    assess_entry,
    calculate_order_quantity,
    calculate_stop_price,
    calculate_adaptive_stop,
    guard_signal,
)
from bot.signals import Action, Side, Signal

SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 9, 29, 10, tzinfo=SHANGHAI)


def position(symbol="ADA/USDT:USDT", side=Side.LONG, *, quantity=10, entry=100, stop=80, opened=NOW):
    if side is Side.SHORT and stop == 80:
        stop = 120
    return Position(symbol, side, quantity, entry, stop, opened, "test_entry", Regime.LOW.value)


def intent(symbol="ADA/USDT:USDT", side=Side.LONG, *, quantity=0.1, price=100, stop=80, source=None):
    if side is Side.SHORT and stop == 80:
        stop = 120
    return OrderIntent(symbol, side, Action.OPEN, quantity, price, "test_entry", Regime.LOW, stop, source)


def approval_for(position_, portfolio):
    order = OrderIntent(
        position_.symbol,
        position_.side,
        Action.OPEN,
        position_.quantity,
        position_.entry_price,
        position_.reason,
        Regime(position_.regime),
        position_.stop_price,
        position_.strategy_source,
    )
    return assess_entry(order, portfolio, 100000, 0, NOW, load_settings(env_file=None))


def test_position_source_is_canonical_with_strategy_source_compatibility():
    current = Position("ADA/USDT:USDT", Side.LONG, 1, 100, 95, NOW, "trend", "TREND", source="TREND")
    legacy = Position("ADA/USDT:USDT", Side.LONG, 1, 100, 95, NOW, "trend", "TREND",
                      strategy_source="TREND")

    assert current.source == current.strategy_source == "TREND"
    assert legacy.source == legacy.strategy_source == "TREND"
    assert asdict(current)["source"] == "TREND"
    assert "strategy_source" not in asdict(current)
    with pytest.raises(ValueError, match="source must be"):
        Position("ADA/USDT:USDT", Side.LONG, 1, 100, 95, NOW, "bad", "TREND", source="ROUTER")


def test_atr_stops_are_directional_and_validate_inputs():
    settings = load_settings(env_file=None)
    assert calculate_stop_price(Side.LONG, 100, 2, settings) == 96
    assert calculate_stop_price(Side.SHORT, 100, 2, settings) == 104
    with pytest.raises(ValueError, match="ATR"):
        calculate_stop_price(Side.LONG, 100, 0, settings)
    with pytest.raises(ValueError, match="positive"):
        calculate_stop_price(Side.LONG, 1, 1, settings)


def test_adaptive_stop_uses_structure_and_rejects_overwide_stop_by_default():
    settings = load_settings(env_file=None)
    stop, multiplier = calculate_adaptive_stop(Side.LONG, 100, 1, 98, settings)
    assert stop == pytest.approx(98.0)  # The 2 ATR disaster cap is tighter than the structure stop.
    assert multiplier == 1.0
    with pytest.raises(ValueError, match="stop_too_wide"):
        calculate_adaptive_stop(Side.LONG, 100, 1, 95, settings)
    short_stop, short_multiplier = calculate_adaptive_stop(Side.SHORT, 100, 1, 102, settings)
    assert short_stop == pytest.approx(102.0)
    assert short_multiplier == 1.0
    settings.stops.wide_stop_action = "shrink"
    shrunk_stop, multiplier = calculate_adaptive_stop(Side.LONG, 100, 1, 95, settings)
    assert shrunk_stop == pytest.approx(98.0)
    assert multiplier == pytest.approx(settings.stops.max_stop_atr / 5.4)


def test_position_or_open_order_without_a_valid_stop_cannot_be_constructed():
    with pytest.raises(TypeError):
        Position("ADA", Side.LONG, 1, 100, opened_at=NOW, reason="x", regime="LOW")
    with pytest.raises(ValueError, match="requires"):
        position(stop=None)
    with pytest.raises(ValueError, match="requires"):
        OrderIntent("ADA", Side.LONG, Action.OPEN, 1, 100, "x", Regime.LOW)
    with pytest.raises(ValueError, match="below"):
        intent(side=Side.LONG, stop=100)
    with pytest.raises(ValueError, match="above"):
        intent(side=Side.SHORT, stop=100)


def test_quantity_targets_fixed_margin_and_rounds_down():
    settings = load_settings(env_file=None)
    quantity = calculate_order_quantity(3000, 10, PortfolioState(), settings, step_size=0.3)
    assert quantity == pytest.approx(300.0)
    assert quantity * 10 / settings.sizing.leverage == pytest.approx(settings.sizing.margin_per_symbol_usdt)
    half_quantity = calculate_order_quantity(
        3000, 10, PortfolioState(), settings, step_size=0.3, size_multiplier=0.5,
    )
    assert half_quantity * 10 / settings.sizing.leverage == pytest.approx(500)
    assert calculate_order_quantity(3000, 10, PortfolioState(), settings, step_size=0.3, minimum_quantity=301) == 0


def test_quantity_can_use_equity_scaled_lead_margin_without_changing_default_target():
    settings = load_settings(env_file=None)
    fixed = calculate_order_quantity(3000, 10, PortfolioState(), settings, step_size=0.1)
    scaled = calculate_order_quantity(1000, 10, PortfolioState(), settings, step_size=0.1,
                                      target_margin_usdt=134.2)

    assert fixed == pytest.approx(300.0)
    assert scaled == pytest.approx(40.2)


def test_quantity_shrinks_to_configured_stop_risk_budget():
    settings = load_settings(env_file=None)
    quantity = calculate_order_quantity(
        1000, 10, PortfolioState(), settings, step_size=0.1, stop_price=9,
    )
    cost_rate = settings.fees.taker_bps / 10_000 + settings.risk.max_slippage_pct
    estimated_risk = quantity * (abs(10 - 9) + (10 + 9) * cost_rate)

    assert settings.risk.max_single_trade_risk_pct == pytest.approx(0.02)
    assert quantity < settings.sizing.margin_per_symbol_usdt * settings.sizing.leverage / 10
    assert estimated_risk <= 1000 * settings.risk.max_single_trade_risk_pct
    assert calculate_order_quantity(
        1000, 10, PortfolioState(), settings, step_size=0.1,
        minimum_quantity=quantity + 0.1, stop_price=9,
    ) == 0


def test_quantity_respects_remaining_gross_cap():
    settings = load_settings(env_file=None)
    existing = position("X/USDT:USDT", quantity=4, entry=100, stop=80)
    portfolio = PortfolioState(positions={existing.symbol: existing})
    assert calculate_order_quantity(1000, 10, portfolio, settings, step_size=0.3) == 0
    quantity = calculate_order_quantity(3000, 10, portfolio, settings, step_size=0.3)
    assert quantity == pytest.approx(300.0)
    assert portfolio.gross_notional + quantity * 10 <= 3000 * settings.sizing.max_gross_notional_pct


def test_valid_entry_passes_and_open_position_blocks_duplicate_symbol():
    settings = load_settings(env_file=None)
    order = intent(source="MEAN_REVERSION")
    portfolio = PortfolioState()
    assert assess_entry(order, portfolio, 1000, 0.0005, NOW, settings).reason == "approved"
    existing = position()
    with pytest.raises(TypeError):
        portfolio.register_open(existing)
    with pytest.raises(ValueError, match="approved risk decision"):
        portfolio.register_open(existing, RiskDecision(False, "daily_drawdown_halt"))
    approval = approval_for(existing, portfolio)
    portfolio.register_open(existing, approval)
    other = position("OTHER/USDT:USDT")
    with pytest.raises(ValueError, match="does not match"):
        portfolio.register_open(other, approval)
    assert assess_entry(order, portfolio, 1000, 0.0005, NOW, settings).reason == "symbol_already_open"
    with pytest.raises(ValueError, match="already open"):
        portfolio.register_open(existing, approval)


@pytest.mark.parametrize(
    "funding_rate",
    [None, float("nan")],
)
def test_funding_filter_still_fails_closed_when_rate_is_missing(funding_rate):
    settings = load_settings(env_file=None)
    decision = assess_entry(intent(), PortfolioState(), 1000, funding_rate, NOW, settings)
    assert not decision.allowed
    assert decision.reason == "funding_rate_missing"


@pytest.mark.parametrize(
    ("side", "funding_rate", "allowed"),
    [
        (Side.LONG, 0.0011, False),
        (Side.SHORT, -0.0011, False),
        (Side.LONG, -0.0011, True),
        (Side.SHORT, 0.0011, True),
        (Side.LONG, 0.001, True),
        (Side.SHORT, -0.001, True),
    ],
)
def test_funding_filter_only_blocks_adverse_rates_strictly_over_limit(side, funding_rate, allowed):
    settings = load_settings(env_file=None)
    decision = assess_entry(
        intent(side=side), PortfolioState(), 1000, funding_rate, NOW, settings,
    )
    assert decision.allowed is allowed
    if not allowed:
        assert decision.reason == "funding_rate_blocked"


def test_cooldown_blocks_reentry_until_configured_hours_have_elapsed():
    settings = load_settings(env_file=None)
    exit_at = NOW - timedelta(hours=17, minutes=59)
    portfolio = PortfolioState(last_exit_at={"ADA/USDT:USDT": exit_at})
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).reason == "cooldown_active"
    portfolio.last_exit_at["ADA/USDT:USDT"] = NOW - timedelta(hours=18)
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).allowed


def test_reason_specific_cooldowns_for_stop_profit_strategy_and_liquidation():
    settings = load_settings(env_file=None)
    symbol = "ADA/USDT:USDT"
    portfolio = PortfolioState(last_exit_at={symbol: NOW - timedelta(hours=5, minutes=59)},
                               last_exit_category={symbol: "stop_loss"})
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).reason == "cooldown_active"
    portfolio.last_exit_at[symbol] = NOW - timedelta(hours=6)
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).allowed

    portfolio.last_exit_at[symbol] = NOW - timedelta(hours=1, minutes=59)
    portfolio.last_exit_category[symbol] = "profit_exit"
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).reason == "cooldown_active"
    portfolio.last_exit_at[symbol] = NOW - timedelta(hours=2)
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).allowed

    portfolio.last_exit_at[symbol] = NOW - timedelta(minutes=1)
    portfolio.last_exit_category[symbol] = "strategy_exit"
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).allowed

    portfolio.last_exit_at[symbol] = NOW - timedelta(hours=17, minutes=59)
    portfolio.last_exit_category[symbol] = "liquidation"
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).reason == "cooldown_active"
    portfolio.last_exit_at[symbol] = NOW - timedelta(hours=18)
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).allowed


def test_explicit_same_bar_reversal_can_bypass_symbol_cooldown():
    settings = load_settings(env_file=None)
    portfolio = PortfolioState(last_exit_at={"ADA/USDT:USDT": NOW - timedelta(hours=1)})
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings).reason == "cooldown_active"
    assert assess_entry(intent(), portfolio, 1000, 0, NOW, settings, bypass_cooldown=True).allowed


def test_exit_reason_classification_uses_realized_result_for_protective_stop():
    assert classify_exit_category("stop_hit_public_price", -1.0) == "stop_loss"
    assert classify_exit_category("stop_hit", 1.0) == "profit_exit"
    assert classify_exit_category("liquidation_mark_trigger", -100.0) == "liquidation"
    assert classify_exit_category("4h_impulse_down", 2.0) == "strategy_exit"


def test_close_records_cooldown_and_requires_monotonic_aware_time():
    portfolio = PortfolioState()
    opened = position()
    portfolio.register_open(opened, approval_for(opened, portfolio))
    closed = NOW + timedelta(hours=1)
    assert portfolio.register_close(opened.symbol, closed) == opened
    assert opened.symbol not in portfolio.positions
    assert portfolio.last_exit_at[opened.symbol] == closed
    assert portfolio.last_exit_source[opened.symbol] == opened.strategy_source
    with pytest.raises(ValueError, match="timezone-aware"):
        portfolio.register_close(opened.symbol, datetime(2026, 9, 29, 11))


def test_per_symbol_gross_and_symbol_count_caps_reject_excess_orders():
    settings = load_settings(env_file=None)
    assert assess_entry(intent(quantity=31), PortfolioState(), 1000, 0, NOW, settings).reason == "per_symbol_notional_cap"
    large = position("X/USDT:USDT", quantity=16, entry=100, stop=80)
    another = position("Y/USDT:USDT", quantity=1.1, entry=100, stop=80)
    portfolio = PortfolioState(positions={large.symbol: large, another.symbol: another})
    assert assess_entry(intent("Z/USDT:USDT", quantity=1, stop=99), portfolio, 1000, 0, NOW, settings).reason == "gross_notional_cap"
    capped = PortfolioState()
    for index in range(settings.sizing.max_symbols):
        held = position(f"Q{index}/USDT:USDT", quantity=0.01, entry=100, stop=99)
        capped.positions[held.symbol] = held
    small = intent("Z/USDT:USDT", quantity=0.1, price=100, stop=80)
    assert assess_entry(small, capped, 1000, 0, NOW, settings).reason == "max_symbols_reached"


def test_daily_drawdown_halts_at_threshold_and_stays_halted_until_shanghai_day_rollover():
    settings = load_settings(env_file=None)
    portfolio = PortfolioState()
    assert not portfolio.observe_equity(1000, NOW, settings)
    assert portfolio.observe_equity(960, NOW + timedelta(hours=1), settings)
    assert assess_entry(intent(), portfolio, 970, 0, NOW + timedelta(hours=2), settings).reason == "daily_drawdown_halt"
    next_day = datetime(2026, 9, 30, 0, 0, tzinfo=SHANGHAI)
    assert not portfolio.observe_equity(970, next_day, settings)
    assert portfolio.day_start_equity == 970


def test_daily_drawdown_uses_shanghai_date_and_rejects_clock_reversal():
    settings = load_settings(env_file=None)
    portfolio = PortfolioState()
    before_midnight = datetime(2026, 9, 28, 16, 0, tzinfo=ZoneInfo("UTC"))
    portfolio.observe_equity(1000, before_midnight, settings)
    # UTC 16:00 is Shanghai 00:00 the next local date.
    assert portfolio.trading_date == datetime(2026, 9, 29, tzinfo=SHANGHAI).date()
    with pytest.raises(ValueError, match="backwards"):
        portfolio.observe_equity(1000, before_midnight - timedelta(minutes=1), settings)


def test_signal_guard_requires_an_open_intent_and_matching_direction():
    settings = load_settings(env_file=None)
    portfolio = PortfolioState()
    open_signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    assert guard_signal(open_signal, None, portfolio, 1000, 0, NOW, settings).reason == "opening_intent_missing"
    assert guard_signal(open_signal, intent(side=Side.SHORT), portfolio, 1000, 0, NOW, settings).reason == "signal_order_side_mismatch"
    close_intent = OrderIntent("ADA/USDT:USDT", Side.LONG, Action.CLOSE, 1, 100, "test", Regime.LOW)
    assert guard_signal(open_signal, close_intent, portfolio, 1000, 0, NOW, settings).reason == "opening_intent_action_mismatch"
    close_signal = Signal(Side.LONG, Action.CLOSE, "test", 1)
    assert guard_signal(close_signal, None, portfolio, 1000, None, NOW, settings).reason == "signal_does_not_open"


def test_guard_signal_applies_all_entry_risk_checks():
    settings = load_settings(env_file=None)
    signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    assert guard_signal(signal, intent(), PortfolioState(), 1000, 0, NOW, settings).allowed


def test_dynamic_market_budget_limits_gross_exposure_and_order_sizing():
    settings = load_settings(env_file=None)
    large = position("X/USDT:USDT", quantity=5, entry=100, stop=80)
    portfolio = PortfolioState(positions={large.symbol: large})
    signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    order = intent("Y/USDT:USDT", quantity=5, price=100, stop=99)

    decision = guard_signal(signal, order, portfolio, 1000, 0, NOW, settings,
                            gross_notional_cap_pct=0.9)
    quantity = calculate_order_quantity(1000, 10, PortfolioState(), settings,
                                        step_size=0.1, gross_notional_cap_pct=0.9)

    assert decision.reason == "market_risk_budget_cap"
    assert quantity == 0


def test_zero_market_budget_fails_closed_and_close_signals_remain_allowed():
    settings = load_settings(env_file=None)
    signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    close_signal = Signal(Side.LONG, Action.CLOSE, "test", 1)

    assert guard_signal(signal, intent(), PortfolioState(), 1000, 0, NOW, settings,
                        gross_notional_cap_pct=0).reason == "market_risk_paused"
    assert guard_signal(close_signal, None, PortfolioState(), 1000, None, NOW, settings,
                        gross_notional_cap_pct=0).allowed


def test_dynamic_market_budget_cannot_raise_configured_gross_cap():
    settings = load_settings(env_file=None)
    configured = calculate_order_quantity(3000, 10, PortfolioState(), settings, step_size=0.1)
    oversized_budget = calculate_order_quantity(3000, 10, PortfolioState(), settings,
                                                step_size=0.1, gross_notional_cap_pct=99.0)

    assert oversized_budget == configured


def test_single_trade_stop_risk_includes_round_trip_fees_and_slippage():
    settings = load_settings(env_file=None)
    signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    over = intent(quantity=1.0, price=100, stop=98)
    decision = guard_signal(signal, over, PortfolioState(), 100, 0, NOW, settings)
    assert decision.reason == "single_trade_risk_cap"
    assert decision.details["estimated_round_trip_fees_and_slippage"] > 0
    assert guard_signal(signal, intent(quantity=0.8, price=100, stop=98), PortfolioState(),
                        100, 0, NOW, settings).allowed


def test_strategy_owner_conflict_and_switch_cooldown_are_enforced_even_on_reversal_bypass():
    settings = load_settings(env_file=None)
    held = position()
    portfolio = PortfolioState(positions={held.symbol: held})
    trend_signal = Signal(Side.LONG, Action.OPEN, "trend_entry", 1,
                          symbol=held.symbol, source="TREND")
    trend_intent = intent(source="TREND")
    assert guard_signal(trend_signal, trend_intent, portfolio, 10000, 0, NOW, settings).reason == (
        "position_owned_by_other_strategy"
    )

    portfolio.positions.clear()
    portfolio.last_exit_at[held.symbol] = NOW - timedelta(minutes=30)
    portfolio.last_exit_source[held.symbol] = "MEAN_REVERSION"
    portfolio.last_exit_category[held.symbol] = "stop_loss"
    portfolio.last_exit_at[held.symbol] = NOW - timedelta(hours=2)
    assert guard_signal(trend_signal, trend_intent, portfolio, 10000, 0, NOW, settings,
                        bypass_cooldown=True).reason == "cooldown_active"
    portfolio.last_exit_at[held.symbol] = NOW - timedelta(minutes=30)
    portfolio.last_exit_category[held.symbol] = "strategy_exit"
    assert guard_signal(trend_signal, trend_intent, portfolio, 10000, 0, NOW, settings,
                        bypass_cooldown=True).reason == "strategy_switch_cooldown_active"
    portfolio.last_exit_at[held.symbol] = NOW - timedelta(hours=1)
    assert guard_signal(trend_signal, trend_intent, portfolio, 10000, 0, NOW, settings,
                        bypass_cooldown=True).allowed


def test_strategy_switch_cooldown_uses_configured_closed_bar_count():
    settings = load_settings(env_file=None)
    settings.cooldowns.strategy_switch_bars = 2
    held = position()
    portfolio = PortfolioState(
        last_exit_at={held.symbol: NOW - timedelta(hours=1)},
        last_exit_category={held.symbol: "strategy_exit"},
        last_exit_source={held.symbol: "MEAN_REVERSION"},
    )
    signal = Signal(Side.LONG, Action.OPEN, "trend_entry", 1,
                    symbol=held.symbol, source="TREND")
    order = intent(source="TREND")

    one_bar = guard_signal(signal, order, portfolio, 10000, 0, NOW, settings,
                           bars_since_exit=1, bypass_cooldown=True)
    two_bars = guard_signal(signal, order, portfolio, 10000, 0, NOW, settings,
                            bars_since_exit=2)

    assert one_bar.reason == "strategy_switch_cooldown_active"
    assert one_bar.details["bars_since_exit"] == 1
    assert one_bar.details["required_bars"] == 2
    assert two_bars.allowed


def test_directional_exposure_counts_open_positions_and_reserved_orders():
    settings = load_settings(env_file=None)
    first = position("X/USDT:USDT", quantity=20, entry=100, stop=99)
    second = position("Y/USDT:USDT", quantity=20, entry=100, stop=99)
    portfolio = PortfolioState(positions={first.symbol: first, second.symbol: second})
    order = intent("Z/USDT:USDT", quantity=10, price=100, stop=99)
    assert assess_entry(order, portfolio, 5000, 0, NOW, settings).reason == "directional_exposure_cap"
    assert assess_entry(intent(quantity=1, price=100, stop=99), PortfolioState(), 10000, 0, NOW,
                        settings, reserved_long_notional=9000).reason == "directional_exposure_cap"
    assert assess_entry(intent("W/USDT:USDT"), PortfolioState(), 10000, 0, NOW,
                        settings, reserved_symbols=settings.sizing.max_symbols).reason == "max_symbols_reached"


def test_elevated_market_risk_applies_size_multiplier_and_caps():
    settings = load_settings(env_file=None)
    signal = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    # multiplier is 1.0, so RISK_ELEVATED keeps the full $3000 per-symbol cap.
    assert guard_signal(signal, intent(quantity=31, price=100, stop=99), PortfolioState(),
                        10000, 0, NOW, settings, market_risk_state="RISK_ELEVATED").reason == (
        "per_symbol_notional_cap"
    )
    assert guard_signal(signal, intent(quantity=20, price=100, stop=99), PortfolioState(),
                        10000, 0, NOW, settings, market_risk_state="RISK_ELEVATED").allowed
    normal = calculate_order_quantity(10000, 100, PortfolioState(), settings, step_size=0.1)
    elevated = calculate_order_quantity(10000, 100, PortfolioState(), settings, step_size=0.1,
                                        size_multiplier=settings.market_risk.elevated_size_multiplier)
    assert normal * 100 == pytest.approx(3000)
    # elevated_size_multiplier is 1.0: RISK_ELEVATED no longer halves new-position size.
    assert elevated * 100 == pytest.approx(3000)


def test_open_signal_without_source_is_rejected_by_the_unified_guard():
    settings = load_settings(env_file=None)
    signal = Signal(Side.LONG, Action.OPEN, "test", 1)
    assert guard_signal(signal, intent(), PortfolioState(), 10000, 0, NOW, settings).reason == (
        "strategy_source_missing"
    )
    mismatched = Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION")
    assert guard_signal(mismatched, intent(source="TREND"), PortfolioState(),
                        10000, 0, NOW, settings).reason == "signal_order_source_mismatch"


def test_open_rejections_emit_structured_risk_log():
    from loguru import logger

    settings = load_settings(env_file=None)
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        decision = guard_signal(
            Signal(Side.LONG, Action.OPEN, "test", 1, source="MEAN_REVERSION"),
        intent(quantity=1.5, price=100, stop=98), PortfolioState(), 100, 0, NOW, settings,
        )
    finally:
        logger.remove(sink_id)

    assert decision.reason == "single_trade_risk_cap"
    assert any('"reason": "single_trade_risk_cap"' in message
               and '"market_risk_state": "NORMAL"' in message for message in messages)
