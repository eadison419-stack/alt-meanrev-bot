from types import SimpleNamespace

import pandas as pd

from bot.backtest import validation
from bot.config import load_settings
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.regime import Regime
from bot.signals import Action, Side, Signal
from bot.symbol_regime import SymbolRegimeAssessment, SymbolRegimeState
from bot.time_filter import TimeFilterFlags


def _assessment(state, reason, at):
    return SymbolRegimeAssessment(
        "AAA/USDT:USDT", state, reason, at.to_pydatetime(), at.to_pydatetime(),
        {"close": 100.0, "adx": 10.0},
    )


def _market(state, at):
    return MarketRiskAssessment(
        state, "test", at.to_pydatetime(), "BTC/USDT:USDT", "4h", {"fixture": True},
    )


def test_routing_audit_uses_closed_bars_and_pause_rejects_entry(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 0.1
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index_1h)
    index_4h = pd.date_range("2025-12-25T00:00:00Z", periods=60, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=index_4h)
    start = index_1h[70] + pd.Timedelta(1, unit="h")
    end = start + pd.Timedelta(30, unit="min")
    observed = {}

    def regimes(_frame, _settings):
        return pd.DataFrame({
            "regime": [Regime.LOW] * len(index_4h), "atr": 1.0, "adx": 20.0,
            "atr_percentile_rank": 50.0, "range_position": 0.2,
            "new_low": False, "new_high": False,
            "range_low": 90.0, "range_high": 110.0,
        }, index=index_4h)

    def signal(_regime, _candle, _flags, _settings, **history):
        observed["one_hour_last"] = history["bars_1h"].index[-1]
        observed["four_hour_last_close"] = history["bars_4h"].index[-1] + pd.Timedelta(4, unit="h")
        return Signal(Side.LONG, Action.OPEN, "fixture_long", 1.0)

    monkeypatch.setattr(validation, "classify_regimes", regimes)
    monkeypatch.setattr(validation, "_historical_rank", lambda *_args, **_kwargs: ["AAA/USDT:USDT"])
    monkeypatch.setattr(validation, "classify_market_risk",
                        lambda _bars, _settings, *, now: _market(MarketRiskState.PAUSE_NEW, pd.Timestamp(now)))
    monkeypatch.setattr(validation, "classify_symbol_regime",
                        lambda symbol, _bars, _settings, *, now: _assessment(
                            SymbolRegimeState.RANGE, "range_confirmed", pd.Timestamp(now)))
    monkeypatch.setattr(validation, "generate_signal", signal)
    monkeypatch.setattr(validation, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding = pd.DataFrame(
        {"funding_rate": [0.0, 0.5], "mark_price": [100.0, 100.0]},
        index=pd.DatetimeIndex([index_1h[0], start]),
    )

    result = validation.run_routing_audit(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four}, four,
        {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": 0}, settings, start=start, end=end, top_n=1,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["raw_signal_action"] == "OPEN"
    assert row["signal_action"] == "NONE"
    assert row["reject_reason"] == "market_risk_paused"
    assert row["market_risk"]["state"] == "PAUSE_NEW"
    assert pd.Timestamp(row["signal_time"]) <= pd.Timestamp(row["closed_1h_bar_time"])
    assert observed["one_hour_last"] == index_1h[70]
    assert observed["four_hour_last_close"] <= pd.Timestamp(row["decision_time"])
    assert result.summary["invariants"]["no_open_candidate_during_pause_new"] is True


def test_routing_audit_tracks_strategy_source_and_excludes_btc_eth(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 0.1
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=90, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index_1h)
    index_4h = pd.date_range("2025-12-25T00:00:00Z", periods=60, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=index_4h)
    start = index_1h[70] + pd.Timedelta(1, unit="h")
    end = start + pd.Timedelta(30, unit="min")

    monkeypatch.setattr(validation, "classify_regimes", lambda _frame, _settings: pd.DataFrame({
        "regime": [Regime.LOW] * len(index_4h), "atr": 1.0, "adx": 20.0,
        "atr_percentile_rank": 50.0, "range_position": 0.2,
        "new_low": False, "new_high": False,
        "range_low": 90.0, "range_high": 110.0,
    }, index=index_4h))
    monkeypatch.setattr(validation, "_historical_rank", lambda *_args, **_kwargs: [
        "AAA/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT:USDT",
    ])
    monkeypatch.setattr(validation, "classify_market_risk",
                        lambda _bars, _settings, *, now: _market(MarketRiskState.NORMAL, pd.Timestamp(now)))
    monkeypatch.setattr(validation, "classify_symbol_regime",
                        lambda symbol, _bars, _settings, *, now: _assessment(
                            SymbolRegimeState.RANGE, "range_confirmed", pd.Timestamp(now)))
    monkeypatch.setattr(validation, "generate_signal",
                        lambda *_args, **_kwargs: Signal(Side.LONG, Action.OPEN, "fixture_long", 1.0))
    monkeypatch.setattr(validation, "evaluate_time_filter",
                        lambda *_args, **_kwargs: TimeFilterFlags(True, True, False, False))
    funding = pd.DataFrame(
        {"funding_rate": [0.0, 0.5], "mark_price": [100.0, 100.0]},
        index=pd.DatetimeIndex([index_1h[0], start]),
    )

    result = validation.run_routing_audit(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four}, four,
        {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": 0}, settings, start=start, end=end, top_n=3,
    )

    assert [row["symbol"] for row in result.rows] == ["AAA/USDT:USDT"]
    assert result.rows[0]["signal_source"] == "MEAN_REVERSION"
    assert result.rows[0]["source"] == "MEAN_REVERSION"
    assert result.rows[0]["signal_action"] == "OPEN"
    assert result.rows[0]["execution_state"] == "baseline_fill_engine_required"
    assert result.rows[0]["funding_rate_asof"] == 0.0
    assert result.rows[0]["funding_observed_at"] == index_1h[0].isoformat()
    assert result.summary["accepted_signal_candidates_by_source"] == {"MEAN_REVERSION": 1}


def test_trend_audit_passes_only_closed_prefix_and_never_counts_plan_as_fill(monkeypatch):
    settings = load_settings(env_file=None)
    settings.universe.min_quote_volume_24h = 0.1
    index_1h = pd.date_range("2026-01-01T00:00:00Z", periods=90, freq="1h")
    one = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                        "close": 100.0, "volume": 1_000_000.0}, index=index_1h)
    index_4h = pd.date_range("2025-12-25T00:00:00Z", periods=60, freq="4h")
    four = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1_000_000.0}, index=index_4h)
    start = index_1h[70] + pd.Timedelta(1, unit="h")
    end = start + pd.Timedelta(30, unit="min")
    observed = {}

    monkeypatch.setattr(validation, "classify_regimes", lambda _frame, _settings: pd.DataFrame({
        "regime": [Regime.LOW] * len(index_4h), "atr": 1.0, "adx": 20.0,
        "atr_percentile_rank": 50.0, "range_position": 0.2,
        "new_low": False, "new_high": False,
        "range_low": 90.0, "range_high": 110.0,
    }, index=index_4h))
    monkeypatch.setattr(validation, "_historical_rank", lambda *_args, **_kwargs: ["AAA/USDT:USDT"])
    monkeypatch.setattr(validation, "classify_market_risk",
                        lambda _bars, _settings, *, now: _market(MarketRiskState.NORMAL, pd.Timestamp(now)))
    monkeypatch.setattr(validation, "classify_symbol_regime",
                        lambda symbol, _bars, _settings, *, now: _assessment(
                            SymbolRegimeState.TREND, "trend_confirmed_up", pd.Timestamp(now)))

    def trend_entry(symbol, one_closed, four_closed, _settings, *, now, **_kwargs):
        at = pd.Timestamp(now)
        assert (one_closed.index + pd.Timedelta(1, unit="h") <= at).all()
        assert (four_closed.index + pd.Timedelta(4, unit="h") <= at).all()
        observed["one_closed_last"] = one_closed.index[-1]
        signal = Signal(Side.LONG, Action.OPEN, "trend_pullback_candidate", 1.0,
                        symbol=symbol, source="TREND", timestamp=at.to_pydatetime())
        return SimpleNamespace(signal=signal, to_dict=lambda: {"reason": "trend_pullback_candidate"})

    monkeypatch.setattr(validation, "evaluate_trend_entry", trend_entry)
    funding = pd.DataFrame(
        {"funding_rate": [0.0], "mark_price": [100.0]},
        index=pd.DatetimeIndex([index_1h[0]]),
    )

    result = validation.run_routing_audit(
        {"AAA/USDT:USDT": one}, {"AAA/USDT:USDT": four}, four,
        {"AAA/USDT:USDT": funding},
        {"AAA/USDT:USDT": 0}, settings, start=start, end=end, top_n=1,
    )

    row = result.rows[0]
    assert observed["one_closed_last"] == index_1h[70]
    assert row["signal_source"] == "TREND"
    assert row["source"] == "TREND"
    assert row["signal_action"] == "OPEN"
    assert row["execution_state"] == "not_filled_by_decision_audit"
    assert row["accepted_candidate"] is True
