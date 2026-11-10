import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import bot
from bot.config import PROJECT_ROOT, load_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("CONFIG_PATH", "BOT_MODE", "BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


def test_smoke_and_settings_import():
    from bot.config import settings
    loaded = load_settings(env_file=None)
    assert settings.timezone == "Asia/Shanghai"
    assert loaded.runtime.bot_mode == "paper"
    assert settings.runtime.binance_testnet is True
    assert settings.pyramiding.enabled is False
    assert loaded.exits.no_progress_bars == 6
    assert loaded.exits.max_hold_bars == 24
    assert loaded.exits.runner_max_hold_bars == 0
    assert loaded.lead_risk.max_portfolio_stop_risk_pct == 0.04
    assert loaded.lead_risk.giveback_risk_weight == 0.15
    assert not hasattr(loaded.exit_proposal_v1, "no_progress_bars")
    assert loaded.trend_strategy.risk_per_trade_pct == 0.005
    assert loaded.trend_strategy.pullback_ttl_minutes == 45
    assert loaded.symbol_regime.trend_adx_min == 25
    assert loaded.symbol_regime.confirm_bars == 3
    assert loaded.symbol_regime.range_confirm_bars == 2
    assert loaded.mean_reversion_match.weights.route == 25
    assert loaded.mean_reversion_match.weights.legacy_regime == 15
    assert loaded.mean_reversion_match.penalties.impulse == 30
    assert loaded.trend_match.weights.range_position == 40
    assert loaded.trend_match.range_lookback_bars == 48
    assert loaded.trend_match.recent_return_lookback_bars == 12
    assert loaded.trend_match.score_aggregation == "max"
    assert loaded.trend_match.common_deductions.one_sided_move == 15
    assert loaded.low_position_watch.enabled is True
    assert loaded.low_position_watch.range_position_max_pct == 45
    assert loaded.low_position_watch.ema50_distance_min_pct == -10
    assert loaded.low_position_watch.recent_return_max_pct == 15
    assert loaded.bollinger_observation.enabled is True
    assert loaded.bollinger_observation.period == 20
    assert loaded.bollinger_observation.standard_deviations == 2
    assert loaded.bollinger_observation.width_lookback_bars == 50
    assert loaded.bollinger_observation.width_low_quantile == 0.30
    assert loaded.bollinger_observation.volume_filter_enabled is False
    assert loaded.cooldowns.strategy_switch_hours == 1
    assert loaded.cooldowns.strategy_switch_bars == 1
    assert loaded.risk.max_single_trade_risk_pct == 0.02
    assert loaded.sizing.max_directional_notional_pct == 0.9
    assert loaded.market_risk.elevated_size_multiplier == 1.0
    assert loaded.sizing.max_symbols == 5


def test_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_settings(env_file=None).timeframes.regime == "4h"


def test_environment_over_dotenv_and_secrets_hidden(monkeypatch, tmp_path):
    fixture = tmp_path / "runtime.fixture"
    fixture.write_text("BOT_MODE=live\nBINANCE_API_SECRET=synthetic-test-value\n", encoding="utf-8")
    monkeypatch.setenv("BOT_MODE", "paper")
    settings = load_settings(env_file=fixture)
    assert settings.runtime.bot_mode == "paper"
    assert settings.runtime.binance_api_secret.get_secret_value() == "synthetic-test-value"
    assert "synthetic-test-value" not in repr(settings.runtime)
    assert "synthetic-test-value" not in settings.model_dump_json()


def write_config(tmp_path, update):
    data = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml").read_text())
    update(data)
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_config_path_sources(monkeypatch, tmp_path):
    path = write_config(tmp_path, lambda data: data["signals"].update(cooldown_hours=24))
    fixture = tmp_path / "runtime.fixture"
    fixture.write_text(f"CONFIG_PATH={path.as_posix()}\n", encoding="utf-8")
    assert load_settings(env_file=fixture).signals.cooldown_hours == 24
    monkeypatch.setenv("CONFIG_PATH", "configs/default.yaml")
    assert load_settings(env_file=fixture).signals.cooldown_hours == 18
    assert load_settings(path, env_file=fixture).signals.cooldown_hours == 24


@pytest.mark.parametrize("update", [
    lambda data: data["regime"].update(low_pct=0.9),
    lambda data: data["signals"].update(stop_atr_mult=0),
    lambda data: data["short_entry"].update(higher_highs_lookback_4h=1),
    lambda data: data["risk"].update(daily_drawdown_halt_pct=-1),
    lambda data: data["sizing"].update(max_symbols=0),
    lambda data: data["sizing"].update(leverage=4),
    lambda data: data["sizing"].update(max_directional_notional_pct=2.0),
    lambda data: data["market_risk"].update(elevated_size_multiplier=1.5),
    lambda data: data["lead_risk"].update(giveback_risk_weight=1.01),
    lambda data: data["low_position_watch"].update(range_position_min_pct=50),
    lambda data: data["symbol_regime"].update(trend_adx_min=15, range_adx_max=20),
    lambda data: data["symbol_regime"].update(ema_fast_bars=50, ema_slow_bars=20),
    lambda data: data["mean_reversion_match"]["weights"].update(
        route=0, adx=0, ema50_slope=0, ema50_distance=0, structure=0, legacy_regime=0,
    ),
    lambda data: data["trend_match"]["weights"].update(
        range_position=0, ema50_distance=0, recent_return=0, adx_rising=0, ema50_slope=0, structure=0,
    ),
    lambda data: data["trend_match"].update(
        preferred_position_min_pct=70, preferred_position_max_pct=60,
    ),
    lambda data: data["session"].update(morning_end="07:00"),
    lambda data: data["exit_management"].update(breakeven_trigger_r=3, trailing_trigger_r=2),
    lambda data: data["trend_strategy"].update(min_stop_atr=2.5, max_stop_atr=2.5),
    lambda data: data["trend_strategy"].update(breakeven_trigger_r=3, trailing_trigger_r=2),
    lambda data: data["trend_strategy"].update(pullback_ttl_minutes=60),
    lambda data: data.update(unknown=True),
])
def test_invalid_configuration_rejected(tmp_path, update):
    path = write_config(tmp_path, update)
    with pytest.raises(ValidationError):
        load_settings(path, env_file=None)


def test_missing_or_empty_configuration_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml", env_file=None)
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(path, env_file=None)


def test_modules_import_without_network():
    for module in pkgutil.walk_packages(bot.__path__, bot.__name__ + "."):
        importlib.import_module(module.name)


@pytest.mark.parametrize("args", [[], ["--live"]])
def test_live_disabled_even_with_environment_and_flag(monkeypatch, args):
    monkeypatch.setenv("BOT_MODE", "live")
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/run_live.py"), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Phase 1" in result.stderr


def test_paper_help_does_not_connect_to_exchange():
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/run_paper.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--once" in result.stdout
    assert "public-data cycle" in result.stdout


def test_dashboard_help_does_not_connect_to_exchange():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_dashboard.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--no-open" in result.stdout
    assert "localhost" in result.stdout


def test_multi_paper_help_does_not_connect_to_exchange():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_multi_paper.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--top" in result.stdout
    assert "hottest crypto perpetuals" in result.stdout


def test_backtest_help_does_not_run_simulation():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_backtest.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "{long,short,both,all}" in result.stdout


def test_download_help_does_not_connect_to_exchange():
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/download_klines.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "comma-separated USDT-M swaps" in result.stdout


def test_logging_to_file(tmp_path):
    from loguru import logger
    from bot.utils.logging import configure_logging
    path = tmp_path / "bot.log"
    configure_logging(log_path=path)
    logger.info("phase-one-smoke")
    logger.remove()
    assert "phase-one-smoke" in path.read_text(encoding="utf-8")
