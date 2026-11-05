import pandas as pd
import pytest

from bot.config import load_settings
from bot.regime import Regime, RegimeSnapshot
from bot.signals import Action, Side, generate_signal
from bot.time_filter import TimeFilterFlags


def snapshot(regime, *, new_low=False, new_high=False):
    return RegimeSnapshot(pd.Timestamp("2026-01-01", tz="UTC"), regime, 1.0, 25.0, 50.0, 0.5, new_low, new_high)


def candle(open_, close_):
    return pd.Series({"open": open_, "close": close_, "low": min(open_, close_) - 0.2})


def entry_history_1h(lows=(98, 98, 98)):
    return pd.DataFrame({"low": lows, "open": [99, 99, 99], "close": [99, 99, 100]})


def entry_history_4h(lows=(95, 96, 97, 98)):
    return pd.DataFrame({"low": lows})


def flags(**values):
    defaults = dict(allow_long_entry=True, allow_short_entry=True, flatten_only=False, watch_next_morning_long=False)
    defaults.update(values)
    return TimeFilterFlags(**defaults)


def test_low_zone_and_bullish_1h_opens_long():
    signal = generate_signal(snapshot(Regime.LOW), candle(100, 99), flags(), load_settings(env_file=None))
    assert signal.action is Action.NONE
    signal = generate_signal(snapshot(Regime.LOW), candle(99, 100), flags(), load_settings(env_file=None),
                             bars_1h=entry_history_1h(), bars_4h=entry_history_4h())
    assert (signal.side, signal.action, signal.reason) == (Side.LONG, Action.OPEN, "low_zone_1h_bullish_close")


def test_high_zone_and_bearish_1h_opens_short():
    signal = generate_signal(snapshot(Regime.HIGH), candle(99, 101), flags(), load_settings(env_file=None))
    assert signal.action is Action.NONE
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), load_settings(env_file=None))
    assert (signal.side, signal.action, signal.reason) == (Side.SHORT, Action.OPEN, "high_zone_1h_bearish_close")


def test_unavailable_time_filter_data_blocks_entries_even_if_direction_flags_are_true():
    unavailable = flags(data_ready=False, data_reason="hourly_history_gap")

    long_signal = generate_signal(
        snapshot(Regime.LOW), candle(99, 100), unavailable, load_settings(env_file=None),
        bars_1h=entry_history_1h(), bars_4h=entry_history_4h(),
    )
    short_signal = generate_signal(
        snapshot(Regime.HIGH), candle(100, 99), unavailable, load_settings(env_file=None),
    )

    assert long_signal.action is Action.NONE
    assert short_signal.action is Action.NONE
    assert long_signal.reason == short_signal.reason == "time_filter_data_unavailable"


def test_four_strictly_rising_4h_highs_block_short_with_reason():
    bars_4h = pd.DataFrame({"high": [101, 102, 103, 104]})
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(),
                             load_settings(env_file=None), bars_4h=bars_4h)
    assert (signal.side, signal.action, signal.reason) == (None, Action.NONE, "short_blocked_higher_highs")


def test_non_rising_4h_highs_keep_existing_short_entry_path():
    bars_4h = pd.DataFrame({"high": [101, 102, 102, 104]})
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(),
                             load_settings(env_file=None), bars_4h=bars_4h)
    assert (signal.side, signal.action, signal.reason) == (Side.SHORT, Action.OPEN, "high_zone_1h_bearish_close")


def test_rising_4h_high_filter_does_not_affect_long_entries():
    settings = load_settings(env_file=None)
    bars_4h = entry_history_4h((95, 96, 97, 98)).assign(high=[101, 102, 103, 104])
    signal = generate_signal(snapshot(Regime.LOW), candle(99, 100), flags(), settings,
                             bars_1h=entry_history_1h(), bars_4h=bars_4h)
    assert (signal.side, signal.action, signal.reason) == (Side.LONG, Action.OPEN, "low_zone_1h_bullish_close")


def test_rising_4h_high_filter_can_be_disabled():
    settings = load_settings(env_file=None)
    settings.short_entry.block_if_4h_higher_highs = False
    bars_4h = pd.DataFrame({"high": [101, 102, 103, 104]})
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), settings, bars_4h=bars_4h)
    assert (signal.side, signal.action, signal.reason) == (Side.SHORT, Action.OPEN, "high_zone_1h_bearish_close")


def test_rising_4h_highs_block_same_bar_short_reverse_but_keep_long_close():
    settings = load_settings(env_file=None)
    settings.signals.reverse_same_bar = True
    bars_4h = pd.DataFrame({"high": [101, 102, 103, 104]})
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), settings,
                             position=Side.LONG, bars_4h=bars_4h)
    assert (signal.side, signal.action, signal.next_side) == (Side.LONG, Action.CLOSE, Side.SHORT)
    assert signal.reason == "reverse_signal_close_first"


def test_new_extremes_outside_low_high_do_not_open():
    settings = load_settings(env_file=None)
    assert generate_signal(snapshot(Regime.IMPULSE_DOWN, new_low=True), candle(100, 99), flags(), settings).action is Action.NONE
    assert generate_signal(snapshot(Regime.IMPULSE_UP, new_high=True), candle(99, 101), flags(), settings).action is Action.NONE


def test_sideways_never_opens_even_at_a_range_extreme():
    signal = generate_signal(snapshot(Regime.SIDEWAYS, new_low=True), candle(100, 99), flags(), load_settings(env_file=None))
    assert signal.action is Action.NONE


def test_opposite_signal_closes_first_without_same_bar_reverse_by_default():
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), load_settings(env_file=None), position=Side.LONG)
    assert (signal.side, signal.action, signal.next_side) == (Side.LONG, Action.CLOSE, Side.SHORT)
    assert signal.reason == "reverse_signal_close_first"


def test_enabled_same_bar_reverse_is_half_size_and_close_first():
    settings = load_settings(env_file=None)
    settings.signals.reverse_same_bar = True
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), settings, position=Side.LONG)
    assert (signal.side, signal.action, signal.close_first) == (Side.SHORT, Action.OPEN, True)
    assert signal.size_multiplier == 0.5


def test_opposite_setup_closes_even_when_new_opposite_entry_is_blocked():
    settings = load_settings(env_file=None)
    settings.signals.reverse_same_bar = True
    blocked = flags(allow_short_entry=False, flatten_only=True)
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), blocked, settings, position=Side.LONG)
    assert (signal.side, signal.action, signal.next_side) == (Side.LONG, Action.CLOSE, Side.SHORT)
    assert signal.reason == "reverse_signal_close_first"


def test_unavailable_time_filter_data_cannot_open_or_queue_same_bar_reverse():
    settings = load_settings(env_file=None)
    settings.signals.reverse_same_bar = True
    unavailable = flags(data_ready=False, data_reason="hourly_history_gap")

    signal = generate_signal(
        snapshot(Regime.HIGH), candle(100, 99), unavailable, settings,
        position=Side.LONG,
    )

    assert signal.action is Action.CLOSE
    assert signal.next_side is None


def test_legacy_opposite_candle_direction_no_longer_triggers_reverse_close():
    settings = load_settings(env_file=None)
    signal = generate_signal(snapshot(Regime.HIGH), candle(99, 101), flags(), settings, position=Side.LONG)
    assert (signal.action, signal.reason) == (Action.NONE, "position_held_no_add")


def test_afternoon_filter_and_flatten_only_block_entries_but_stop_closes_position():
    settings = load_settings(env_file=None)
    blocked = flags(allow_long_entry=False)
    signal = generate_signal(snapshot(Regime.LOW), candle(99, 100), blocked, settings,
                             bars_1h=entry_history_1h(), bars_4h=entry_history_4h())
    assert signal.action is Action.NONE and "time_filter" in signal.reason
    night = flags(allow_long_entry=False, allow_short_entry=False, flatten_only=True)
    assert generate_signal(snapshot(Regime.HIGH), candle(99, 101), night, settings).action is Action.NONE
    stopped = generate_signal(snapshot(Regime.MID), candle(100, 100), night, settings, position=Side.LONG, stop_hit=True)
    assert (stopped.action, stopped.reason) == (Action.CLOSE, "stop_hit")


def test_no_adds_to_existing_same_direction_position():
    signal = generate_signal(snapshot(Regime.LOW), candle(100, 99), flags(), load_settings(env_file=None), position=Side.LONG)
    assert signal.action is Action.NONE


@pytest.mark.parametrize(
    ("position", "regime", "new_extreme", "morning_move", "reason"),
    [
        (Side.LONG, Regime.IMPULSE_UP, False, None, "4h_impulse_up"),
        (Side.SHORT, Regime.IMPULSE_DOWN, False, None, "4h_impulse_down"),
        (Side.LONG, Regime.HIGH, True, None, "high_zone_new_high"),
        (Side.SHORT, Regime.LOW, True, None, "low_zone_new_low"),
        (Side.LONG, Regime.MID, False, "UP", "morning_bullish_impulse"),
        (Side.SHORT, Regime.MID, False, "DOWN", "morning_bearish_impulse"),
    ],
)
def test_exit_conditions_close_existing_position(position, regime, new_extreme, morning_move, reason):
    side = Side.LONG if position is Side.LONG else Side.SHORT
    data = {"new_high": new_extreme} if side is Side.LONG else {"new_low": new_extreme}
    flag_data = {"morning_move": morning_move} if morning_move else {}
    signal = generate_signal(
        snapshot(regime, **data), candle(100, 100), flags(**flag_data),
        load_settings(env_file=None), position=position,
    )
    assert signal.action is Action.CLOSE
    assert signal.reason == reason


def test_entry_bar_requires_open_and_close():
    with pytest.raises(ValueError, match="open and close"):
        generate_signal(snapshot(Regime.LOW), pd.Series({"close": 99}), flags(), load_settings(env_file=None))


def test_four_descending_4h_lows_block_long_with_explicit_reason():
    signal = generate_signal(snapshot(Regime.LOW), candle(99, 100), flags(), load_settings(env_file=None),
                             bars_1h=entry_history_1h(), bars_4h=entry_history_4h((99, 98, 97, 96)))
    assert (signal.action, signal.reason) == (Action.NONE, "still_slow_down")


def test_missing_reversal_evidence_has_explicit_rejection_reason():
    signal = generate_signal(snapshot(Regime.LOW), pd.Series({"open": 100, "close": 99.5, "low": 99.4}),
                             flags(), load_settings(env_file=None),
                             bars_1h=entry_history_1h((99, 98, 97)), bars_4h=entry_history_4h())
    assert (signal.action, signal.reason) == (Action.NONE, "no_reversal_sign")


def test_reversal_sign_allows_long_to_reach_existing_entry_filters():
    settings = load_settings(env_file=None)
    signal = generate_signal(snapshot(Regime.LOW), candle(99, 100), flags(), settings,
                             bars_1h=entry_history_1h((97, 98, 98)), bars_4h=entry_history_4h())
    assert signal.action is Action.OPEN
    assert signal.side is Side.LONG


def test_bullish_close_alone_does_not_satisfy_reversal_confirmation():
    signal = generate_signal(
        snapshot(Regime.LOW), pd.Series({"open": 100, "close": 101, "low": 100}),
        flags(), load_settings(env_file=None),
        bars_1h=entry_history_1h((99, 98, 97)), bars_4h=entry_history_4h(),
    )
    assert (signal.action, signal.reason) == (Action.NONE, "no_reversal_sign")


def test_long_lower_wick_independently_satisfies_reversal_confirmation():
    signal = generate_signal(
        snapshot(Regime.LOW), pd.Series({"open": 100, "close": 101, "low": 99}),
        flags(), load_settings(env_file=None),
        bars_1h=entry_history_1h((99, 98, 97)), bars_4h=entry_history_4h(),
    )
    assert (signal.action, signal.side) == (Action.OPEN, Side.LONG)


def test_long_filters_do_not_change_short_signal():
    signal = generate_signal(snapshot(Regime.HIGH), candle(100, 99), flags(), load_settings(env_file=None),
                             bars_1h=entry_history_1h((99, 98, 97)),
                             bars_4h=entry_history_4h((99, 98, 97, 96)))
    assert (signal.action, signal.side, signal.reason) == (Action.OPEN, Side.SHORT, "high_zone_1h_bearish_close")


def test_long_rejects_range_break_of_half_atr_and_allows_shallow_break():
    settings = load_settings(env_file=None)
    entry = dict(entry_atr=0.000455, range_low=0.008255, range_high=0.036397,
                 structure_low=0.0075, structure_high=0.0085)
    deep = generate_signal(
        snapshot(Regime.LOW), candle(0.0076, 0.00773), flags(), settings,
        bars_1h=entry_history_1h(), bars_4h=entry_history_4h(),
        entry_reference_price=0.00773, **entry,
    )
    assert (deep.action, deep.side, deep.reason) == (Action.NONE, None, "long_rejected_below_range")

    shallow = generate_signal(
        snapshot(Regime.LOW), candle(0.0080, 0.0081), flags(), settings,
        bars_1h=entry_history_1h(), bars_4h=entry_history_4h(),
        entry_reference_price=0.0081, **entry,
    )
    assert (shallow.action, shallow.side) == (Action.OPEN, Side.LONG)


def test_long_rejects_if_structure_invalidation_is_not_below_entry():
    signal = generate_signal(
        snapshot(Regime.LOW), candle(0.0077, 0.00775), flags(), load_settings(env_file=None),
        bars_1h=entry_history_1h(), bars_4h=entry_history_4h(),
        entry_reference_price=0.00775, entry_atr=0.000455,
        range_low=0.007, range_high=0.02,
        structure_low=0.0078, structure_high=0.0085,
    )
    assert (signal.action, signal.side, signal.reason) == (Action.NONE, None, "invalid_on_entry")


@pytest.mark.parametrize("reference", [0.00773, 0.00767])
def test_previous_us_entry_shape_is_rejected_before_order(reference):
    settings = load_settings(env_file=None)
    signal = generate_signal(
        snapshot(Regime.LOW), candle(reference - 0.00001, reference), flags(), settings,
        bars_1h=entry_history_1h(), bars_4h=entry_history_4h(),
        entry_reference_price=reference, entry_atr=0.000455,
        range_low=0.008255, range_high=0.036397,
        structure_low=0.0075, structure_high=0.0085,
    )
    assert (signal.action, signal.reason) == (Action.NONE, "long_rejected_below_range")


def test_short_far_above_range_is_rejected_with_symmetric_reason():
    settings = load_settings(env_file=None)
    signal = generate_signal(
        snapshot(Regime.HIGH), candle(0.0101, 0.0100), flags(), settings,
        entry_reference_price=0.0100, entry_atr=0.0005,
        range_low=0.005, range_high=0.0097,
        structure_low=0.009, structure_high=0.0105,
    )
    assert (signal.action, signal.side, signal.reason) == (Action.NONE, None, "short_rejected_above_range")
