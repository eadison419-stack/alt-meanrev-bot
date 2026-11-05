from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from loguru import logger

from bot.backtest import portfolio
from bot.backtest.metrics import summarize_by_source
from bot.config import load_settings
from bot.regime import Regime
from bot.signals import Action, Side, Signal
from bot.strategy_router import TREND
from bot.symbol_regime import SymbolRegimeState
from bot.trend_strategy import TrendEntryDecision
from bot.time_filter import TimeFilterFlags


def test_historical_rank_uses_only_elapsed_bars_and_minimum_listing_age():
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 0.1
    index = pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="1h")
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 100.0}, index=index)
    burst = flat.copy()
    burst.iloc[29, burst.columns.get_loc("volume")] = 10_000.0
    frames = {"AAA": flat, "BBB": burst}
    turnovers = {symbol: portfolio._quote_turnover(frame) for symbol, frame in frames.items()}
    close_time = index[-1] + pd.Timedelta(1, unit="h")
    ranked = portfolio._historical_rank(
        close_time, frames, turnovers,
        {"AAA": int((close_time - pd.Timedelta(10, unit="D")).timestamp() * 1000),
         "BBB": int((close_time - pd.Timedelta(20, unit="D")).timestamp() * 1000)},
        settings, limit=2,
    )

    assert ranked == ["BBB"]


def test_portfolio_time_exit_closes_stagnant_position_after_12_hours(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    start_index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                        "close": 100.0, "volume": 1_000_000.0}, index=start_index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == start_index[25]:
            return Signal(Side.LONG, Action.OPEN, "synthetic_open", 1.0)
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding_index = pd.DatetimeIndex([start_index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)
    from_time = start_index[25] + pd.Timedelta(1, unit="h")
    result = portfolio.run_portfolio_backtest(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four},
        {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": int((start_index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=from_time, end=start_index[-1] + pd.Timedelta(1, unit="h"),
        top_n=20, exit_policy="time_managed",
    )
    shadow_result = portfolio.run_portfolio_backtest(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four},
        {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": int((start_index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=from_time, end=start_index[-1] + pd.Timedelta(1, unit="h"),
        top_n=20, exit_policy="time_managed", record_long_shadows=True,
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "time_stop_12h_no_progress"
    assert result.trades[0].closed_at == start_index[38]
    assert result.trades == shadow_result.trades
    assert result.metrics == shadow_result.metrics
    assert shadow_result.long_trade_excursions


def test_portfolio_pending_fill_copies_signal_source_to_position_and_trade(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "source_transport", 1.0, source="TREND")
        if position is Side.LONG and candle.name == index[28]:
            return Signal(Side.LONG, Action.CLOSE, "source_transport_close", 1.0)
        return Signal(None, Action.NONE, "none", 0.0)

    opened = []
    original_register = portfolio.PortfolioState.register_open

    def register_with_source(self, position, approval):
        opened.append((position.source, approval.intent.source))
        original_register(self, position, approval)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(portfolio.PortfolioState, "register_open", register_with_source)
    symbol = "AAA/USDT:USDT"
    funding_index = pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)

    result = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding},
        {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=index[25] + pd.Timedelta(1, unit="h"),
        end=index[40] + pd.Timedelta(1, unit="h"), top_n=20,
    )

    assert opened == [("TREND", "TREND")]
    assert result.trades[0].opened_at == index[26]
    assert result.trades[0].source == "TREND"


def test_opt_in_trend_route_fills_next_open_and_uses_trend_stop_and_costs(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    one.loc[index[28], "low"] = 97.5
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    signal_at = index[26]
    entry_calls = []

    def trend_entry(symbol, _one, _four, _settings, *, now, **_kwargs):
        entry_calls.append(pd.Timestamp(now))
        if pd.Timestamp(now) == signal_at:
            signal = Signal(Side.LONG, Action.OPEN, "trend_test_entry", 1.0,
                            symbol=symbol, source=TREND, timestamp=now)
        else:
            signal = Signal(None, Action.NONE, "no_trend_entry", 0.0,
                            symbol=symbol, source=TREND, timestamp=now)
        return TrendEntryDecision(signal)

    opened_positions = []
    original_register_open = portfolio.PortfolioState.register_open

    def capture_open(self, position, approval):
        opened_positions.append((position, approval))
        original_register_open(self, position, approval)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "classify_symbol_regime", lambda *_args, **_kwargs: SimpleNamespace(
        state=SymbolRegimeState.TREND, reason="trend_confirmed_up"))
    monkeypatch.setattr(portfolio, "classify_market_risk", lambda *_args, **_kwargs: SimpleNamespace(
        state=SimpleNamespace(value="NORMAL")))
    monkeypatch.setattr(portfolio, "evaluate_trend_entry", trend_entry)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(portfolio.PortfolioState, "register_open", capture_open)
    symbol = "AAA/USDT:USDT"
    funding = pd.DataFrame(
        {"funding_rate": [0.0, 0.0004], "mark_price": [100.0, 100.0]},
        index=pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h"), index[27]]),
    )
    market_bars = four.copy()
    result = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding},
        {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=signal_at, end=index[40] + pd.Timedelta(1, unit="h"), top_n=20,
        initial_equity=1_000.0, enable_trend_routing=True, market_bars_4h=market_bars,
    )

    assert entry_calls[0] == signal_at
    assert len(result.trades) == 1
    trade = result.trades[0]
    position, approval = opened_positions[0]
    assert trade.source == TREND
    assert position.source == approval.intent.source == TREND
    assert trade.opened_at == index[26]
    assert trade.closed_at == index[29]
    assert trade.exit_reason == "trend_structure_stop_hit"
    assert trade.quantity > 0
    assert position.quantity == pytest.approx(trade.quantity)
    assert position.entry_price == pytest.approx(trade.entry_price)
    risk_cost_rate = (settings.fees.taker_bps + settings.fees.slippage_bps) / 10_000
    estimated_trend_risk = position.quantity * (
        abs(position.entry_price - position.stop_price)
        + (position.entry_price + position.stop_price) * risk_cost_rate
    )
    assert estimated_trend_risk <= 1_000.0 * settings.trend_strategy.risk_per_trade_pct
    assert position.notional <= 1_000.0 * min(3.0, settings.sizing.leverage)
    assert trade.fees > 0
    assert trade.slippage_paid > 0
    assert trade.funding_paid > 0
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees - trade.funding_paid)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_000.0 + trade.net_pnl)
    assert result.metrics["net_pnl"] == pytest.approx(result.equity_curve.iloc[-1] - 1_000.0)
    assert result.integrity_checks["all_passed"] is True
    assert result.integrity_checks["all_signal_inputs_use_closed_bars"] is True
    assert result.integrity_checks["no_dual_strategy_position_overlap"] is True
    source_summary = summarize_by_source(result.trades)
    assert source_summary[TREND]["trade_count"] == 1
    assert source_summary[TREND]["win_rate_pct"] == 0.0
    assert source_summary[TREND]["net_pnl"] == pytest.approx(trade.net_pnl)
    assert source_summary[TREND]["fees"] == pytest.approx(trade.fees)
    assert source_summary[TREND]["funding_paid"] == pytest.approx(trade.funding_paid)
    assert source_summary[TREND]["slippage_paid"] == pytest.approx(trade.slippage_paid)
    assert source_summary["MEAN_REVERSION"]["trade_count"] == 0
    assert sum(row["net_pnl"] for row in source_summary.values()) == pytest.approx(result.metrics["net_pnl"])


def test_trend_routing_disabled_matches_explicit_mean_reversion_baseline(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "baseline_open", 1.0)
        if position is Side.LONG and candle.name == index[28]:
            return Signal(Side.LONG, Action.CLOSE, "baseline_close", 1.0)
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    symbol = "AAA/USDT:USDT"
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]},
                           index=pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")]))
    common = ({symbol: one}, {symbol: four}, {symbol: funding},
              {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)})
    kwargs = {"start": index[25] + pd.Timedelta(1, unit="h"),
              "end": index[40] + pd.Timedelta(1, unit="h"), "top_n": 20}
    default = portfolio.run_portfolio_backtest(*common, settings, **kwargs)
    explicit = portfolio.run_portfolio_backtest(*common, settings, enable_trend_routing=False, **kwargs)

    assert default.trades == explicit.trades
    pd.testing.assert_series_equal(default.equity_curve, explicit.equity_curve)
    assert default.metrics == explicit.metrics
    assert default.rejected_signals_by_source == explicit.rejected_signals_by_source
    assert default.integrity_checks == explicit.integrity_checks
    assert default.result_fingerprint == explicit.result_fingerprint
    assert default.integrity_checks["all_passed"] is True
    assert default.integrity_checks["closed_bar_input_checks"] > 0
    assert default.integrity_checks["actionable_signals_checked"] > 0
    assert default.integrity_checks["closed_bar_input_violations"] == 0
    assert default.integrity_checks["future_signal_timestamp_violations"] == 0
    assert default.trades[0].opened_at == index[26]
    assert default.trades[0].closed_at == index[29]
    assert default.rejected_signals_by_source["TREND"] == {}
    assert default.rejected_signals_by_source["MEAN_REVERSION"] == {}
    source_summary = summarize_by_source(default.trades)
    assert source_summary[TREND]["trade_count"] == 0
    assert source_summary[TREND]["net_pnl"] == 0.0
    assert source_summary["MEAN_REVERSION"]["trade_count"] == 1


def test_portfolio_rejects_cross_strategy_open_without_closing_held_position(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "mean_reversion_open", 1.0,
                          source="MEAN_REVERSION")
        if position is Side.LONG and candle.name == index[27]:
            return Signal(Side.SHORT, Action.OPEN, "trend_reverse", 1.0,
                          source="TREND", close_first=True)
        if position is Side.LONG and candle.name == index[31]:
            return Signal(Side.LONG, Action.CLOSE, "mean_reversion_close", 1.0,
                          source="MEAN_REVERSION")
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding_index = pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)
    log_stream = StringIO()
    sink_id = logger.add(log_stream, format="{message}")
    symbol = "AAA/USDT:USDT"
    try:
        result = portfolio.run_portfolio_backtest(
            {symbol: one}, {symbol: four}, {symbol: funding},
            {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
            settings, start=index[25], end=index[35] + pd.Timedelta(1, unit="h"), top_n=20,
        )
    finally:
        logger.remove(sink_id)

    assert result.skipped_entries["POSITION_OWNERSHIP_CONFLICT"] == 1
    assert result.rejected_signals_by_source["TREND"]["POSITION_OWNERSHIP_CONFLICT"] == 1
    assert len(result.trades) == 1
    assert result.trades[0].source == "MEAN_REVERSION"
    assert result.trades[0].exit_reason == "mean_reversion_close"
    assert '"reason": "POSITION_OWNERSHIP_CONFLICT"' in log_stream.getvalue()
    assert '"position_source": "MEAN_REVERSION"' in log_stream.getvalue()


def test_portfolio_switch_cooldown_counts_actual_one_hour_bars(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    settings.cooldowns.strategy_switch_bars = 2
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "mean_reversion_open", 1.0,
                          source="MEAN_REVERSION")
        if position is Side.LONG and candle.name == index[28]:
            return Signal(Side.LONG, Action.CLOSE, "mean_reversion_close", 1.0,
                          source="MEAN_REVERSION")
        if position is None and candle.name in {index[29], index[30]}:
            return Signal(Side.LONG, Action.OPEN, "trend_open", 1.0, source="TREND")
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding_index = pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)
    symbol = "AAA/USDT:USDT"

    result = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding},
        {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=index[25], end=index[33] + pd.Timedelta(1, unit="h"), top_n=20,
    )

    assert result.skipped_entries["strategy_switch_cooldown_active"] == 1
    assert result.rejected_signals_by_source["TREND"]["strategy_switch_cooldown_active"] == 1
    assert [trade.source for trade in result.trades] == ["MEAN_REVERSION", "TREND"]
    assert result.trades[0].closed_at == index[29]
    assert result.trades[1].opened_at == index[31]


@pytest.mark.parametrize(("symbol_state", "source"), [
    (SymbolRegimeState.TREND, TREND),
    (SymbolRegimeState.RANGE, "MEAN_REVERSION"),
])
def test_pause_new_counts_only_valid_candidates_blocked_by_route(monkeypatch, symbol_state, source):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    calls = 0

    def trend_entry(symbol, _one, _four, _settings, *, now, market_risk_state, **_kwargs):
        nonlocal calls
        calls += 1
        assert market_risk_state == "NORMAL"
        signal = (Signal(Side.LONG, Action.OPEN, "trend_candidate", 1.0,
                         symbol=symbol, source=TREND, timestamp=now)
                  if calls == 1 else
                  Signal(None, Action.NONE, "no_candidate", 0.0,
                         symbol=symbol, source=TREND, timestamp=now))
        return TrendEntryDecision(signal)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "classify_symbol_regime", lambda *_args, **_kwargs: SimpleNamespace(
        state=symbol_state,
        reason="trend_confirmed_up" if symbol_state is SymbolRegimeState.TREND else "range_confirmed"))
    monkeypatch.setattr(portfolio, "classify_market_risk", lambda *_args, **_kwargs: SimpleNamespace(
        state=SimpleNamespace(value="PAUSE_NEW")))
    monkeypatch.setattr(portfolio, "evaluate_trend_entry", trend_entry)
    monkeypatch.setattr(portfolio, "generate_signal", lambda _regime, candle, *_args, **_kwargs:
                        Signal(Side.LONG, Action.OPEN, "mean_reversion_candidate", 1.0)
                        if candle.name == index[25] else Signal(None, Action.NONE, "no_candidate", 0.0))
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    symbol = "AAA/USDT:USDT"
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]},
                           index=pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")]))
    market_bars = four.copy()

    result = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding},
        {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=index[25], end=index[28] + pd.Timedelta(1, unit="h"), top_n=20,
        enable_trend_routing=True, market_bars_4h=market_bars,
    )

    assert result.trades == ()
    assert result.skipped_entries["market_risk_paused"] == 1
    assert result.rejected_signals_by_source[source]["market_risk_paused"] == 1
    assert result.market_risk_execution_summary["PAUSE_NEW"]["opened_count"] == 0


def test_risk_elevated_reports_execution_without_scaling_mean_reversion(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "mean_reversion_open", 1.0)
        if position is Side.LONG and candle.name == index[28]:
            return Signal(Side.LONG, Action.CLOSE, "mean_reversion_close", 1.0)
        return Signal(None, Action.NONE, "none", 0.0)

    market_state = ["NORMAL"]
    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "classify_symbol_regime", lambda *_args, **_kwargs: SimpleNamespace(
        state=SymbolRegimeState.RANGE, reason="range_confirmed"))
    monkeypatch.setattr(portfolio, "classify_market_risk", lambda *_args, **_kwargs: SimpleNamespace(
        state=SimpleNamespace(value=market_state[0])))
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    symbol = "AAA/USDT:USDT"
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]},
                           index=pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")]))
    onboard_ms = {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)}
    kwargs = {
        "start": index[25], "end": index[35] + pd.Timedelta(1, unit="h"),
        "top_n": 20, "enable_trend_routing": True, "market_bars_4h": four,
    }
    normal = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding}, onboard_ms, settings, **kwargs,
    )
    market_state[0] = "RISK_ELEVATED"
    elevated = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding}, onboard_ms, settings, **kwargs,
    )

    assert len(normal.trades) == len(elevated.trades) == 1
    assert normal.trades[0].source == elevated.trades[0].source == "MEAN_REVERSION"
    assert normal.trades[0].quantity == pytest.approx(elevated.trades[0].quantity)  # multiplier 1.0: no scaling
    elevated_summary = elevated.market_risk_execution_summary["RISK_ELEVATED"]
    assert elevated_summary["opened_count"] == 1
    assert elevated_summary["configured_size_multiplier"] == settings.market_risk.elevated_size_multiplier
    assert elevated_summary["opened_notional"] == pytest.approx(elevated.trades[0].quantity
                                                                   * elevated.trades[0].entry_price)


def test_risk_elevated_reports_execution_without_scaling_trend(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    calls = [0]

    def trend_entry(symbol, _one, _four, _settings, *, now, **_kwargs):
        calls[0] += 1
        signal = (Signal(Side.LONG, Action.OPEN, "trend_candidate", 1.0,
                         symbol=symbol, source=TREND, timestamp=now)
                  if calls[0] == 1 else
                  Signal(None, Action.NONE, "no_candidate", 0.0,
                         symbol=symbol, source=TREND, timestamp=now))
        return TrendEntryDecision(signal)

    market_state = ["NORMAL"]
    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "classify_symbol_regime", lambda *_args, **_kwargs: SimpleNamespace(
        state=SymbolRegimeState.TREND, reason="trend_confirmed_up"))
    monkeypatch.setattr(portfolio, "classify_market_risk", lambda *_args, **_kwargs: SimpleNamespace(
        state=SimpleNamespace(value=market_state[0])))
    monkeypatch.setattr(portfolio, "evaluate_trend_entry", trend_entry)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    symbol = "AAA/USDT:USDT"
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]},
                           index=pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")]))
    onboard_ms = {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)}
    kwargs = {
        "start": index[25], "end": index[35] + pd.Timedelta(1, unit="h"),
        "top_n": 20, "enable_trend_routing": True, "market_bars_4h": four,
    }
    normal = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding}, onboard_ms, settings, **kwargs,
    )
    market_state[0] = "RISK_ELEVATED"
    calls[0] = 0
    elevated = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding}, onboard_ms, settings, **kwargs,
    )

    assert len(normal.trades) == len(elevated.trades) == 1
    assert normal.trades[0].source == elevated.trades[0].source == TREND
    assert normal.trades[0].quantity == pytest.approx(elevated.trades[0].quantity)  # multiplier 1.0: no scaling
    elevated_summary = elevated.market_risk_execution_summary["RISK_ELEVATED"]
    assert elevated_summary["opened_count"] == 1
    assert elevated_summary["configured_size_multiplier"] == settings.market_risk.elevated_size_multiplier


def test_reverse_open_uses_configured_half_size(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "synthetic_open", 1.0)
        if position is Side.LONG and candle.name == index[30]:
            return Signal(Side.SHORT, Action.OPEN, "synthetic_reverse", 1.0,
                          close_first=True, size_multiplier=0.5)
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding_index = pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)
    result = portfolio.run_portfolio_backtest(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four}, {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=index[0], end=index[-1] + pd.Timedelta(1, unit="h"), top_n=20,
    )

    first_trade, reverse_trade = result.trades[:2]
    assert first_trade.exit_reason == "reverse_close_first"
    assert reverse_trade.side is Side.SHORT
    assert reverse_trade.quantity * reverse_trade.entry_price == pytest.approx(
        settings.sizing.margin_per_symbol_usdt * settings.sizing.leverage * 0.5, rel=1e-7,
    )


def test_reverse_open_executes_when_symbol_left_hot_universe(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 1.0
    symbol = "AAA/USDT:USDT"
    index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index)
    four_index = pd.date_range("2025-12-30T00:00:00Z", periods=30, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=four_index)

    def regimes(frame, _settings):
        return pd.DataFrame({"regime": Regime.LOW, "atr": 1.0, "adx": 25.0,
                             "atr_percentile_rank": 50.0, "range_position": 0.2,
                             "new_low": False, "new_high": False}, index=frame.index)

    def signal(_regime, candle, _flags, _settings, *, position=None, stop_hit=False, **_history):
        if position is None and candle.name == index[25]:
            return Signal(Side.LONG, Action.OPEN, "synthetic_open", 1.0)
        if position is Side.LONG and candle.name == index[31]:
            return Signal(Side.SHORT, Action.OPEN, "synthetic_reverse", 1.0,
                          close_first=True, size_multiplier=0.5)
        return Signal(None, Action.NONE, "none", 0.0)

    monkeypatch.setattr(portfolio, "classify_regimes", regimes)
    monkeypatch.setattr(portfolio, "generate_signal", signal)
    monkeypatch.setattr(portfolio, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(portfolio, "_historical_rank",
                        lambda at, *_args, **_kwargs: [symbol] if at < index[32] else [])
    funding_index = pd.DatetimeIndex([index[0] - pd.Timedelta(2, unit="h")])
    funding = pd.DataFrame({"funding_rate": [0.0], "mark_price": [100.0]}, index=funding_index)
    result = portfolio.run_portfolio_backtest(
        {symbol: one}, {symbol: four}, {symbol: funding},
        {symbol: int((index[0] - pd.Timedelta(30, unit="D")).timestamp() * 1000)},
        settings, start=index[0], end=index[-1] + pd.Timedelta(1, unit="h"), top_n=20,
    )

    assert len(result.trades) == 2
    assert result.trades[0].side is Side.LONG
    assert result.trades[0].exit_reason == "reverse_close_first"
    assert result.trades[1].side is Side.SHORT
    assert result.trades[1].quantity * result.trades[1].entry_price == pytest.approx(
        settings.sizing.margin_per_symbol_usdt * settings.sizing.leverage * 0.5, rel=1e-7,
    )
