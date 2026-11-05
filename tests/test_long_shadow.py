import pandas as pd

from bot.backtest import long_shadow
from bot.config import load_settings
from bot.regime import Regime


def test_long_shadow_confirmation_and_failure_opportunity_are_sidecar_only(monkeypatch):
    index = pd.date_range("2026-10-01T00:00:00Z", periods=8, freq="1h")
    bars = pd.DataFrame({
        "open": [100, 100, 100, 100, 100, 100, 100, 100],
        "high": [101, 103, 101, 101, 104, 100, 100, 100],
        "low": [99, 100, 99, 98, 99, 99, 99, 99],
        "close": [100, 102, 100, 99, 101, 100, 100, 100],
        "volume": [1000] * 8,
    }, index=index)
    signals = (
        {"symbol": "AAA", "signal_time": index[1], "signal_open": 99.5, "signal_low": 99,
         "signal_close": 100, "atr": 1.0, "structure_low": 98, "regime": Regime.LOW.value},
        {"symbol": "AAA", "signal_time": index[3], "signal_open": 100, "signal_low": 99,
         "signal_close": 100, "atr": 1.0, "structure_low": 98, "regime": Regime.LOW.value},
    )
    monkeypatch.setattr(long_shadow, "_risk_distance", lambda *_args: 1.0)
    monkeypatch.setattr(long_shadow, "_simulate_one", lambda *_args: {
        "entry_price": 102, "exit_price": 103, "exit_time": index[5], "exit_reason": "stop_hit",
        "net_pnl": 1.0, "r_multiple": 0.5, "mfe_r": 1.0, "mae_r": -0.25,
        "held_hours": 4.0,
    })
    result = long_shadow.analyze_long_shadows(
        signals, {"AAA": bars}, {"AAA": pd.DataFrame()},
        {"AAA": pd.DataFrame()}, {"AAA": pd.DataFrame()},
        load_settings(env_file=None), initial_equity=5_000, exit_policy="breakeven_trail",
    )

    assert result[0]["confirm_pass"] is True
    assert result[0]["entry_price"] == 102
    assert result[1]["confirm_pass"] is False
    assert result[1]["fail_reason"] == "new_low"
    assert result[1]["missed_max_up_r_6h"] == 4.0
    assert len(result) == len(signals)
