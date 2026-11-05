from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd

from bot.config import load_settings
from bot import market_risk
from bot.market_risk import MarketRiskState, classify_market_risk, fetch_market_risk_status

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
BTC = "BTC/USDT:USDT"


def bars(direction: int = 0, *, now: datetime = NOW) -> pd.DataFrame:
    end = pd.Timestamp(now).floor("4h") - pd.Timedelta(4, unit="h")
    index = pd.date_range(end=end, periods=100, freq="4h", tz="UTC")
    closes = [100.0]
    for _ in range(99):
        closes.append(closes[-1] * (1 + direction * 0.005))
    close = np.asarray(closes)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1_000.0}, index=index)


def test_market_risk_emits_normal_and_elevated_observations():
    settings = load_settings(env_file=None)

    normal = classify_market_risk(bars(), settings, now=NOW)
    elevated = classify_market_risk(bars(-1), settings, now=NOW)

    assert normal.state is MarketRiskState.NORMAL
    assert normal.reason == "btc_conditions_normal"
    assert elevated.state is MarketRiskState.RISK_ELEVATED
    assert elevated.evidence["trend_extreme"] is True
    assert elevated.checked_at.tzinfo is not None
    assert elevated.evidence["last_closed_bar_at"]


def test_confirmed_extreme_downside_emits_pause_new(monkeypatch):
    import bot.market_risk as market_risk

    settings = load_settings(env_file=None)
    frame = bars(-1)
    prior = float(frame["close"].iloc[-2])
    frame.iloc[-1, frame.columns.get_loc("open")] = prior
    frame.iloc[-1, frame.columns.get_loc("close")] = prior - 2.0
    frame.iloc[-1, frame.columns.get_loc("high")] = prior + 0.1
    frame.iloc[-1, frame.columns.get_loc("low")] = prior - 2.1
    monkeypatch.setattr(
        market_risk, "indicator_frame",
        lambda data, *_args: pd.DataFrame({"atr": 1.0, "adx": 25.0}, index=data.index),
    )

    result = classify_market_risk(frame, settings, now=NOW)

    assert result.state is MarketRiskState.PAUSE_NEW
    assert result.reason == "btc_extreme_downside_confirmed"
    assert result.evidence["downside_impulse"] is True
    assert result.evidence["downside_trend"] is True


def test_missing_stale_and_gapped_market_data_are_observational_pause():
    settings = load_settings(env_file=None)
    missing = classify_market_risk(None, settings, now=NOW)
    stale = classify_market_risk(bars(), settings, now=NOW + timedelta(hours=8))
    gapped = classify_market_risk(bars().drop(index=bars().index[-2]), settings, now=NOW)

    assert missing.state is MarketRiskState.PAUSE_NEW
    assert missing.reason == "btc_bars_missing"
    assert stale.state is MarketRiskState.PAUSE_NEW
    assert stale.reason == "stale_closed_bar"
    assert gapped.state is MarketRiskState.PAUSE_NEW
    assert gapped.reason == "closed_bar_gap"


def test_unclosed_extreme_four_hour_bar_is_ignored():
    settings = load_settings(env_file=None)
    closed = bars()
    open_bar = pd.DataFrame(
        {"open": [100.0], "high": [1_000.0], "low": [1.0], "close": [2.0], "volume": [1_000.0]},
        index=pd.DatetimeIndex([closed.index[-1] + pd.Timedelta(4, unit="h")]),
    )
    with_open = pd.concat([closed, open_bar])

    baseline = classify_market_risk(closed, settings, now=NOW)
    including_open = classify_market_risk(with_open, settings, now=NOW)

    assert baseline.state is MarketRiskState.NORMAL
    assert including_open.state is baseline.state
    assert including_open.evidence == baseline.evidence


def test_fetch_uses_only_configured_btc_four_hour_public_candles():
    settings = load_settings(env_file=None)

    class Exchange:
        calls = []

        def fetch_ohlcv(self, symbol, *, timeframe, limit):
            self.calls.append((symbol, timeframe, limit))
            frame = bars()
            return [[int(index.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
                    for index, row in frame.iterrows()]

    exchange = Exchange()
    result = fetch_market_risk_status(exchange, settings, now=NOW)

    assert exchange.calls == [(BTC, "4h", 100)]
    assert result.state is MarketRiskState.NORMAL
    assert result.symbol == BTC


def test_fetch_failure_is_recorded_without_throwing():
    settings = load_settings(env_file=None)

    class Exchange:
        def fetch_ohlcv(self, *_args, **_kwargs):
            raise RuntimeError("public endpoint unavailable")

    result = fetch_market_risk_status(Exchange(), settings, now=NOW)

    assert result.state is MarketRiskState.PAUSE_NEW
    assert result.evidence["fetch_error"].startswith("RuntimeError:")


def test_standalone_cli_prints_one_status_and_closes_public_exchange(monkeypatch, capsys):
    settings = load_settings(env_file=None)
    assessment = classify_market_risk(bars(), settings, now=NOW)

    class Exchange:
        closed = False

        def close(self):
            self.closed = True

    exchange = Exchange()
    monkeypatch.setattr(market_risk, "load_settings", lambda: settings)
    monkeypatch.setattr(market_risk.ccxt, "binanceusdm", lambda _options: exchange)
    monkeypatch.setattr(market_risk, "configure_system_proxy", lambda _exchange: None)
    monkeypatch.setattr(market_risk, "fetch_market_risk_status",
                        lambda _exchange, _settings, *, now: assessment)

    assert market_risk.main([]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["state"] in {state.value for state in MarketRiskState}
    assert payload["state"] == "NORMAL"
    assert payload["mode"] == "entry_gate"
    assert exchange.closed is True


def test_naive_evaluation_time_is_rejected():
    try:
        classify_market_risk(None, load_settings(env_file=None), now=datetime(2026, 10, 15, 12))
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("naive evaluation time should be rejected")
