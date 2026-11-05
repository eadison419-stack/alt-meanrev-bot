from datetime import datetime, timezone

from bot.signals import Action, Side, Signal, with_signal_context
from bot.strategy_router import MEAN_REVERSION, ROUTER, TREND, apply_entry_route, resolve_strategy_route


NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


def open_signal(symbol: str, source: str) -> Signal:
    return with_signal_context(
        Signal(Side.LONG, Action.OPEN, "test_entry", 1.0),
        symbol=symbol, source=source, timestamp=NOW,
    )


def test_range_routes_to_mean_reversion_and_preserves_existing_signal():
    route = resolve_strategy_route(
        "AAA/USDT:USDT", "RANGE", "NORMAL", position_source=None, at=NOW,
    )
    signal = open_signal(route.symbol, MEAN_REVERSION)

    result = apply_entry_route(signal, route)

    assert route.source == MEAN_REVERSION
    assert route.entry_allowed is True
    assert result == signal
    assert (result.symbol, result.side, result.source, result.timestamp) == (
        "AAA/USDT:USDT", Side.LONG, MEAN_REVERSION, NOW,
    )


def test_trend_is_explicit_placeholder_and_never_generates_an_open():
    route = resolve_strategy_route(
        "AAA/USDT:USDT", "TREND", "NORMAL", position_source=None, at=NOW,
    )

    result = apply_entry_route(open_signal(route.symbol, TREND), route)

    assert route.source == TREND
    assert route.route_reason == "trend_strategy_placeholder"
    assert route.entry_allowed is False
    assert result.action is Action.NONE
    assert result.reason == "trend_strategy_not_implemented"
    assert result.source == ROUTER
    assert result.symbol == route.symbol
    assert result.timestamp == NOW


def test_trend_route_is_available_only_when_caller_supplies_evaluator():
    route = resolve_strategy_route(
        "AAA/USDT:USDT", "TREND", "NORMAL", position_source=None, at=NOW,
        trend_strategy_ready=True,
    )

    assert route.source == TREND
    assert route.route_reason == "trend_strategy_selected"
    assert route.entry_allowed is True
    assert route.entry_block_reason is None


def test_uncertain_and_pause_new_routes_reject_opens():
    uncertain = resolve_strategy_route(
        "AAA/USDT:USDT", "UNCERTAIN", "NORMAL", position_source=None, at=NOW,
    )
    paused = resolve_strategy_route(
        "BBB/USDT:USDT", "RANGE", "PAUSE_NEW", position_source=None, at=NOW,
    )

    uncertain_signal = apply_entry_route(open_signal(uncertain.symbol, MEAN_REVERSION), uncertain)
    paused_signal = apply_entry_route(open_signal(paused.symbol, MEAN_REVERSION), paused)

    assert uncertain_signal.action is Action.NONE
    assert uncertain_signal.reason == "symbol_regime_uncertain"
    assert uncertain_signal.source == ROUTER
    assert paused_signal.action is Action.NONE
    assert paused_signal.reason == "market_risk_paused"


def test_position_owned_by_another_strategy_blocks_open_but_keeps_close():
    route = resolve_strategy_route(
        "AAA/USDT:USDT", "TREND", "NORMAL",
        position_source=MEAN_REVERSION, at=NOW,
    )
    candidate = open_signal(route.symbol, TREND)
    close_signal = with_signal_context(
        Signal(Side.LONG, Action.CLOSE, "existing_owner_exit", 1.0),
        symbol=route.symbol, source=MEAN_REVERSION, timestamp=NOW,
    )

    rejected = apply_entry_route(candidate, route)
    preserved_close = apply_entry_route(close_signal, route)

    assert route.entry_allowed is False
    assert route.entry_block_reason == "position_owned_by_other_strategy"
    assert rejected.action is Action.NONE
    assert rejected.reason == "position_owned_by_other_strategy"
    assert preserved_close == close_signal


def test_route_fails_closed_for_missing_or_mismatched_signal_context():
    route = resolve_strategy_route(
        "AAA/USDT:USDT", "RANGE", "NORMAL", position_source=None, at=NOW,
    )
    missing = apply_entry_route(Signal(Side.LONG, Action.OPEN, "unannotated", 1.0), route)
    wrong_symbol = apply_entry_route(open_signal("BBB/USDT:USDT", MEAN_REVERSION), route)

    assert missing.action is Action.NONE
    assert missing.reason == "signal_symbol_mismatch"
    assert wrong_symbol.action is Action.NONE
    assert wrong_symbol.reason == "signal_symbol_mismatch"
