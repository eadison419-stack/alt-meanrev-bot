from dataclasses import replace
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from bot.backtest import engine
from bot.config import load_settings
from bot.regime import Regime
from bot.signals import Action, Side, Signal
from bot.time_filter import TimeFilterFlags


def synthetic_data():
    start = pd.Timestamp("2026-01-01T04:00:00Z")
    index_1h = pd.date_range(start, periods=32, freq="1h")
    opens = np.full(len(index_1h), 100.0)
    closes = np.full(len(index_1h), 100.0)
    opens[16], closes[16] = 101.0, 101.2
    opens[25], closes[25] = 102.0, 102.0
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    bars_1h = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0},
        index=index_1h,
    )
    index_4h = pd.date_range(start - pd.Timedelta(4, unit="h"), periods=10, freq="4h")
    base = np.full(len(index_4h), 100.0)
    bars_4h = pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1, "close": base, "volume": 1.0},
        index=index_4h,
    )
    funding_index = pd.DatetimeIndex([
        start - pd.Timedelta(8, unit="h"),
        start + pd.Timedelta(12, unit="h") + pd.Timedelta(2, unit="ms"),
        start + pd.Timedelta(16, unit="h") + pd.Timedelta(2, unit="ms"),
        start + pd.Timedelta(20, unit="h") + pd.Timedelta(2, unit="ms"),
    ])
    funding = pd.DataFrame(
        {"funding_rate": [0.0, 0.0005, 0.0003, 0.0004], "mark_price": [100.0] * 4},
        index=funding_index,
    )
    return bars_1h, bars_4h, funding


def patch_strategy(monkeypatch, *, close_index=24, stop_low=None, gap_index=None):
    def fake_classify(frame, settings):
        return pd.DataFrame(
            {
                "atr": 1.0,
                "adx": 25.0,
                "atr_percentile_rank": 50.0,
                "range_position": 0.2,
                "new_low": False,
                "new_high": False,
                "regime": Regime.LOW,
            },
            index=frame.index,
        )

    def fake_flags(*args, **kwargs):
        return TimeFilterFlags(True, True, False, False)

    def fake_signal(regime, candle, flags, settings, *, position=None, stop_hit=False, **_history):
        if position is None and candle.name == bars_index[15]:
            return Signal(Side.LONG, Action.OPEN, "synthetic_open", 1.0)
        if position is Side.LONG and close_index is not None and candle.name == bars_index[close_index]:
            return Signal(Side.LONG, Action.CLOSE, "synthetic_close", 1.0)
        return Signal(None, Action.NONE, "none", 0.0)

    bars_1h, _, _ = synthetic_data()
    bars_index = bars_1h.index
    if stop_low is not None:
        bars_1h.iloc[17, bars_1h.columns.get_loc("low")] = stop_low
    if gap_index is not None:
        bars_1h.iloc[gap_index, bars_1h.columns.get_loc("open")] = 90.0
        bars_1h.iloc[gap_index, bars_1h.columns.get_loc("high")] = 91.0
        bars_1h.iloc[gap_index, bars_1h.columns.get_loc("low")] = 89.0
        bars_1h.iloc[gap_index, bars_1h.columns.get_loc("close")] = 90.0
    monkeypatch.setattr(engine, "classify_regimes", fake_classify)
    monkeypatch.setattr(engine, "evaluate_time_filter", fake_flags)
    monkeypatch.setattr(engine, "generate_signal", fake_signal)
    return bars_1h


def test_signal_fills_at_next_open_and_charges_fees_slippage_and_funding(monkeypatch):
    bars_1h = patch_strategy(monkeypatch)
    _, bars_4h, funding = synthetic_data()
    settings = load_settings(env_file=None)

    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, settings)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.opened_at == bars_1h.index[16]
    assert trade.closed_at == bars_1h.index[25]
    assert trade.entry_price == pytest.approx(101.0 * (1 + settings.fees.slippage_bps / 10_000))
    assert trade.exit_price == pytest.approx(102.0 * (1 - settings.fees.slippage_bps / 10_000))
    assert trade.fees > 0 and trade.slippage_paid > 0
    assert trade.funding_paid == pytest.approx(trade.quantity * 100.0 * 0.0004)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees - trade.funding_paid)
    assert trade.source == "MEAN_REVERSION"
    assert trade.strategy_source == trade.source
    assert result.metrics["trade_count"] == 1


def test_signal_source_is_carried_through_next_open_fill_to_trade(monkeypatch):
    bars_1h = patch_strategy(monkeypatch)
    original_signal = engine.generate_signal

    def trend_candidate(*args, **kwargs):
        signal = original_signal(*args, **kwargs)
        if signal.action is Action.OPEN:
            return replace(signal, source="TREND")
        return signal

    opened = []
    original_register = engine.PortfolioState.register_open

    def register_with_source(self, position, approval):
        opened.append((position.source, approval.intent.source))
        original_register(self, position, approval)

    monkeypatch.setattr(engine, "generate_signal", trend_candidate)
    monkeypatch.setattr(engine.PortfolioState, "register_open", register_with_source)
    _, bars_4h, funding = synthetic_data()

    result = engine.run_backtest(
        "ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None),
    )

    assert opened == [("TREND", "TREND")]
    assert result.trades[0].opened_at == bars_1h.index[16]
    assert result.trades[0].source == "TREND"


def test_single_symbol_backtest_rejects_cross_source_open_before_close_first(monkeypatch):
    bars_1h, bars_4h, funding = synthetic_data()
    index = bars_1h.index

    monkeypatch.setattr(engine, "classify_regimes", lambda frame, _settings: pd.DataFrame(
        {"atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
         "range_position": 0.2, "new_low": False, "new_high": False,
         "regime": Regime.LOW}, index=frame.index,
    ))
    monkeypatch.setattr(engine, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[15]:
            return Signal(Side.LONG, Action.OPEN, "mean_reversion_open", 1.0,
                          source="MEAN_REVERSION")
        if position is Side.LONG and candle.name == index[17]:
            return Signal(Side.SHORT, Action.OPEN, "trend_reverse", 1.0,
                          source="TREND", close_first=True)
        if position is Side.LONG and candle.name == index[20]:
            return Signal(Side.LONG, Action.CLOSE, "mean_reversion_close", 1.0,
                          source="MEAN_REVERSION")
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(engine, "generate_signal", signal)
    log_stream = StringIO()
    sink_id = logger.add(log_stream, format="{message}")
    try:
        result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding,
                                     load_settings(env_file=None))
    finally:
        logger.remove(sink_id)

    assert result.skipped_entries["POSITION_OWNERSHIP_CONFLICT"] == 1
    assert len(result.trades) == 1
    assert result.trades[0].source == "MEAN_REVERSION"
    assert result.trades[0].exit_reason == "mean_reversion_close"
    assert '"reason": "POSITION_OWNERSHIP_CONFLICT"' in log_stream.getvalue()


def test_single_symbol_backtest_switch_cooldown_counts_bars(monkeypatch):
    bars_1h, bars_4h, funding = synthetic_data()
    index = bars_1h.index
    settings = load_settings(env_file=None)
    settings.cooldowns.strategy_switch_bars = 2

    monkeypatch.setattr(engine, "classify_regimes", lambda frame, _settings: pd.DataFrame(
        {"atr": 1.0, "adx": 25.0, "atr_percentile_rank": 50.0,
         "range_position": 0.2, "new_low": False, "new_high": False,
         "regime": Regime.LOW}, index=frame.index,
    ))
    monkeypatch.setattr(engine, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[15]:
            return Signal(Side.LONG, Action.OPEN, "mean_reversion_open", 1.0,
                          source="MEAN_REVERSION")
        if position is Side.LONG and candle.name == index[17]:
            return Signal(Side.LONG, Action.CLOSE, "mean_reversion_close", 1.0,
                          source="MEAN_REVERSION")
        if position is None and candle.name in {index[18], index[19]}:
            return Signal(Side.LONG, Action.OPEN, "trend_open", 1.0, source="TREND")
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(engine, "generate_signal", signal)
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, settings)

    assert result.skipped_entries["strategy_switch_cooldown_active"] == 1
    assert [trade.source for trade in result.trades] == ["MEAN_REVERSION", "TREND"]
    assert result.trades[0].closed_at == index[18]
    assert result.trades[1].opened_at == index[20]


def test_long_mode_blocks_short_entries(monkeypatch):
    bars_1h = patch_strategy(monkeypatch)
    _, bars_4h, funding = synthetic_data()
    result = engine.run_backtest(
        "ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None), mode="short"
    )
    assert result.trades == ()


def test_stop_takes_priority_and_fills_at_stop_price(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None, stop_low=97.0)
    _, bars_4h, funding = synthetic_data()
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None))

    assert result.trades[0].exit_reason == "stop_hit"
    assert result.trades[0].closed_at == bars_1h.index[17] + pd.Timedelta(1, unit="h")
    assert result.trades[0].exit_price < result.trades[0].entry_price


def test_gap_through_existing_stop_fills_at_bar_open(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None, gap_index=17)
    _, bars_4h, funding = synthetic_data()
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None))

    trade = result.trades[0]
    assert trade.exit_reason == "stop_gap"
    assert trade.exit_price == pytest.approx(90.0 * (1 - load_settings(env_file=None).fees.slippage_bps / 10_000))


def test_breakeven_trail_tightens_after_favorable_bar_without_same_bar_lookahead(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None)
    _, bars_4h, funding = synthetic_data()
    # A large favorable excursion activates both thresholds. The low also
    # crosses the newly calculated trail, but that stop is only active next bar.
    bars_1h.iloc[17, bars_1h.columns.get_loc("open")] = 106.0
    bars_1h.iloc[17, bars_1h.columns.get_loc("close")] = 108.0
    bars_1h.iloc[17, bars_1h.columns.get_loc("high")] = 110.0
    bars_1h.iloc[17, bars_1h.columns.get_loc("low")] = 105.0
    bars_1h.iloc[18, bars_1h.columns.get_loc("open")] = 108.0
    bars_1h.iloc[18, bars_1h.columns.get_loc("high")] = 109.0
    bars_1h.iloc[18, bars_1h.columns.get_loc("close")] = 106.0
    bars_1h.iloc[18, bars_1h.columns.get_loc("low")] = 105.0

    result = engine.run_backtest(
        "ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None),
        exit_policy="breakeven_trail",
    )

    trade = result.trades[0]
    assert trade.exit_reason == "stop_gap"
    assert trade.closed_at == bars_1h.index[18] + pd.Timedelta(1, unit="h")
    assert trade.exit_price > trade.entry_price
    assert trade.net_pnl > 0


def test_fixed_exit_policy_remains_default_and_invalid_policy_is_rejected(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None)
    _, bars_4h, funding = synthetic_data()
    settings = load_settings(env_file=None)
    default = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, settings)
    explicit = engine.run_backtest(
        "ADA/USDT:USDT", bars_1h, bars_4h, funding, settings, exit_policy="fixed"
    )
    assert default.trades == explicit.trades
    with pytest.raises(ValueError, match="exit_policy"):
        engine.run_backtest(
            "ADA/USDT:USDT", bars_1h, bars_4h, funding, settings, exit_policy="experimental"
        )


def test_time_managed_policy_uses_12h_stagnation_exit(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None)
    bars_1h.loc[bars_1h.index[25], ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.0]
    _, bars_4h, funding = synthetic_data()
    result = engine.run_backtest(
        "ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None),
        exit_policy="time_managed",
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "time_stop_12h_no_progress"
    assert result.trades[0].closed_at == bars_1h.index[28]


def test_settlement_at_final_bar_close_is_charged_before_forced_liquidation(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None)
    _, bars_4h, funding = synthetic_data()
    final_settlement = bars_1h.index[-1] + pd.Timedelta(1, unit="h") + pd.Timedelta(2, unit="ms")
    funding.loc[final_settlement] = {"funding_rate": 0.0001, "mark_price": 100.0}
    settings = load_settings(env_file=None)
    settings.stops.use_structure_stop = False
    settings.stops.move_to_breakeven_atr = 100.0
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding.sort_index(), settings)

    trade = result.trades[0]
    assert trade.exit_reason == "backtest_end"
    assert trade.funding_paid == pytest.approx(trade.quantity * 100.0 * (0.0004 + 0.0001))


def test_missing_funding_fails_closed_and_timestamp_naive_bounds_are_rejected(monkeypatch):
    bars_1h = patch_strategy(monkeypatch)
    _, bars_4h, funding = synthetic_data()
    empty_funding = funding.iloc[0:0]
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, empty_funding, load_settings(env_file=None))
    assert result.trades == ()
    assert result.skipped_entries.get("funding_rate_missing") == 1
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.run_backtest(
            "ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None),
            start=pd.Timestamp(datetime(2026, 1, 1)),
        )


def test_explicit_reverse_closes_then_opens_at_next_bar_open_with_reduced_size(monkeypatch):
    bars_1h = patch_strategy(monkeypatch, close_index=None)
    _, bars_4h, funding = synthetic_data()
    index = bars_1h.index

    def reverse_signal(regime, candle, flags, settings, *, position=None, stop_hit=False, **_history):
        if position is None and candle.name == index[15]:
            return Signal(Side.LONG, Action.OPEN, "synthetic_open", 1.0)
        if position is Side.LONG and candle.name == index[17]:
            return Signal(Side.SHORT, Action.OPEN, "reverse_after_close", 1.0, close_first=True, size_multiplier=0.5)
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(engine, "generate_signal", reverse_signal)
    result = engine.run_backtest("ADA/USDT:USDT", bars_1h, bars_4h, funding, load_settings(env_file=None))

    assert len(result.trades) == 2
    assert result.trades[0].exit_reason == "reverse_close_first"
    assert result.trades[0].closed_at == index[18]
    assert result.trades[1].side is Side.SHORT
    assert result.trades[1].opened_at == index[18]
    assert result.trades[1].quantity < result.trades[0].quantity
