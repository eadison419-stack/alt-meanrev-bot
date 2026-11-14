"""Validated strategy YAML and private runtime settings; no exchange side effects."""

from datetime import time
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(gt=0)]
NonNegativeCount = Annotated[int, Field(ge=0)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Timeframes(Section):
    regime: Literal["4h"]
    exec: Literal["1h"]


class Lookback(Section):
    range_bars_4h: Count
    atr_len: Count
    adx_len: Count
    min_regime_bars_4h: Count
    afternoon_extreme_lookback_1h: Count


class Regime(Section):
    low_pct: Fraction
    high_pct: Fraction
    adx_sideways: Annotated[float, Field(gt=0, le=100)]
    atrpct_percentile: Annotated[float, Field(ge=0, le=100)]
    atrpct_lookback: Count
    slow_lookback_4h: Count
    slow_move_atr_mult: Positive

    @model_validator(mode="after")
    def ordered_zones(self):
        if self.low_pct >= self.high_pct:
            raise ValueError("low_pct must be less than high_pct")
        return self


class Signals(Section):
    impulse_atr_mult: Positive
    stop_atr_mult: Positive
    allow_short: bool
    allow_long: bool
    reverse_same_bar: bool
    reverse_size_mult: Fraction
    cooldown_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class LongEntry(Section):
    require_reversal_sign: bool
    reversal_lookback_1h: Count
    block_if_4h_lower_lows: bool
    lower_lows_lookback_4h: Count
    reject_if_below_range: bool
    max_below_range_atr: NonNegative


class ShortEntry(Section):
    block_if_4h_higher_highs: bool
    higher_highs_lookback_4h: Annotated[int, Field(gt=1)]
    reject_if_above_range: bool
    max_above_range_atr: NonNegative


class Execution(Section):
    confirm_on_1h_close: bool
    confirm_delay_sec_max: Count
    entry_quote_max_age_sec: Count
    entry_adverse_atr_observe: Annotated[list[Positive], Field(min_length=1, max_length=10)]
    wait_pullback: bool
    wait_bars_15m: Count
    pullback_atr: Positive
    timeout_action: Literal["cancel", "market_fallback"]
    pullback_timeframe: Literal["15m"]


class ExitManagement(Section):
    enabled: bool
    breakeven_trigger_r: Positive
    trailing_trigger_r: Positive
    trailing_atr_mult: Positive
    time_exit_enabled: bool
    time_check_hours: Positive
    time_exit_hours: Positive
    time_min_progress_r: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def ordered_triggers(self):
        if self.breakeven_trigger_r >= self.trailing_trigger_r:
            raise ValueError("breakeven trigger must be below trailing trigger")
        if self.time_check_hours >= self.time_exit_hours:
            raise ValueError("time check must be earlier than final time exit")
        return self


class MrCandidateExit(Section):
    first_lock_r: Positive
    second_lock_trigger_r: Positive
    second_lock_r: Positive
    trailing_trigger_r: Positive

    @model_validator(mode="after")
    def ordered_profit_locks(self):
        if self.first_lock_r >= self.second_lock_r:
            raise ValueError("MR candidate profit locks must increase")
        if self.second_lock_trigger_r >= self.trailing_trigger_r:
            raise ValueError("MR candidate second lock trigger must be below trailing trigger")
        return self


class ExitProposalV1(Section):
    enabled: bool
    structure_enabled: bool
    tp1_atr: Positive
    tp1_frac: Fraction
    tp2_atr: Positive
    tp2_frac: Fraction
    trail_atr: Positive
    trail_tight_atr: Positive

    @model_validator(mode="after")
    def valid_targets(self):
        if self.tp1_atr >= self.tp2_atr:
            raise ValueError("tp1_atr must be below tp2_atr")
        if self.tp1_frac + self.tp2_frac >= 1:
            raise ValueError("tp1_frac + tp2_frac must leave a runner position")
        return self


class Exits(Section):
    structure_invalid_enabled: bool
    structure_invalid_buffer_atr: NonNegative
    regime_fade_enabled: bool
    regime_fade_min_atr: Positive
    no_progress_bars: Count
    no_progress_min_atr: Positive
    no_progress_action: Literal["close"]
    max_hold_bars: Count
    runner_max_hold_bars: NonNegativeCount
    startup_stale_exit: bool


class Stops(Section):
    use_structure_stop: bool
    swing_lookback_1h: Count
    structure_buffer_atr: Positive
    min_stop_atr: Positive
    min_stop_action: Literal["reject", "expand"]
    max_stop_atr: Positive
    wide_stop_action: Literal["reject", "shrink"]
    hard_cap_atr: Positive
    move_to_breakeven_atr: Positive


class Cooldowns(Section):
    stop_loss_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    profit_exit_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    strategy_exit_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    strategy_switch_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    strategy_switch_bars: NonNegativeCount
    liquidation_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Session(Section):
    morning_start: time
    morning_end: time
    afternoon_start: time
    afternoon_end: time
    night_new_entries: bool
    afternoon_extreme_pct: Annotated[float, Field(gt=0, lt=0.5)]

    @model_validator(mode="after")
    def ordered_sessions(self):
        times = (self.morning_start, self.morning_end, self.afternoon_start, self.afternoon_end)
        if any(value.tzinfo is not None for value in times):
            raise ValueError("session times are local Shanghai wall-clock times")
        if not self.morning_start < self.morning_end <= self.afternoon_start < self.afternoon_end:
            raise ValueError("sessions must be ordered and non-overlapping")
        return self


class Sizing(Section):
    leverage: Annotated[int, Field(gt=0, le=3)]
    margin_per_symbol_usdt: Positive
    live_margin_equity_pct: Fraction
    max_symbols: Count
    max_gross_notional_pct: Positive
    max_directional_notional_pct: Positive
    splits: Count

    @model_validator(mode="after")
    def directional_cap_within_gross_cap(self):
        if self.max_directional_notional_pct > self.max_gross_notional_pct:
            raise ValueError("directional exposure cap cannot exceed total gross exposure cap")
        return self


class Pyramiding(Section):
    enabled: bool
    max_adds: Annotated[int, Field(ge=0)]


class Risk(Section):
    daily_drawdown_halt_pct: Fraction
    max_slippage_pct: Fraction
    funding_abs_block: Fraction
    max_single_trade_risk_pct: Fraction


class LeadRisk(Section):
    min_leverage: Annotated[int, Field(ge=3, le=10)]
    max_leverage: Annotated[int, Field(ge=3, le=10)]
    max_symbol_notional_pct: Fraction
    max_portfolio_stop_risk_pct: Fraction
    giveback_risk_weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    daily_drawdown_halt_pct: Fraction
    liquidation_stop_distance_multiple: Annotated[float, Field(ge=2, allow_inf_nan=False)]
    min_stop_liquidation_gap_pct: Annotated[float, Field(ge=0.01, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def valid_leverage_range(self):
        if self.min_leverage > self.max_leverage:
            raise ValueError("Lead minimum leverage cannot exceed maximum leverage")
        return self


class MarketRisk(Section):
    enabled: bool
    symbol: str
    elevated_size_multiplier: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def fixed_market_symbol(self):
        if self.symbol != "BTC/USDT:USDT":
            raise ValueError("market risk observation must use the BTC USDT perpetual")
        return self


class SymbolRegime(Section):
    enabled: bool
    min_closed_bars: Count
    adx_period: Count
    trend_adx_min: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)]
    range_adx_max: Annotated[float, Field(ge=0, lt=100, allow_inf_nan=False)]
    ema_fast_bars: Count
    ema_slow_bars: Count
    slope_lookback_bars: Count
    trend_slope_min_pct: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    range_slope_max_pct: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    range_max_distance_pct: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    structure_lookback_bars: Annotated[int, Field(gt=1)]
    confirm_bars: Count
    range_confirm_bars: Count = 2
    range_structure_mode: Literal["neutral", "not_conflict", "any"] = "neutral"

    @model_validator(mode="after")
    def consistent_thresholds(self):
        if self.ema_fast_bars >= self.ema_slow_bars:
            raise ValueError("symbol regime fast EMA must be shorter than slow EMA")
        if self.trend_adx_min <= self.range_adx_max:
            raise ValueError("symbol regime trend ADX must be above range ADX")
        if self.trend_slope_min_pct <= self.range_slope_max_pct:
            raise ValueError("symbol regime trend slope must be above range slope")
        return self


class MeanReversionMatchWeights(Section):
    route: NonNegative
    adx: NonNegative
    ema50_slope: NonNegative
    ema50_distance: NonNegative
    structure: NonNegative
    legacy_regime: NonNegative

    @model_validator(mode="after")
    def positive_total(self):
        if sum((self.route, self.adx, self.ema50_slope, self.ema50_distance,
                self.structure, self.legacy_regime)) <= 0:
            raise ValueError("mean-reversion match weights must have a positive total")
        return self


class MeanReversionMatchPenalties(Section):
    trend_tendency: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    impulse: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    structure_conflict: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    low_high_edge: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]


class MeanReversionMatch(Section):
    weights: MeanReversionMatchWeights
    penalties: MeanReversionMatchPenalties
    ema50_distance_zero_multiple: Annotated[float, Field(gt=1, allow_inf_nan=False)]


class TrendMatchWeights(Section):
    range_position: NonNegative
    ema50_distance: NonNegative
    recent_return: NonNegative
    adx_rising: NonNegative
    ema50_slope: NonNegative
    structure: NonNegative

    @model_validator(mode="after")
    def positive_total(self):
        values = (self.range_position, self.ema50_distance, self.recent_return,
                  self.adx_rising, self.ema50_slope, self.structure)
        if sum(values) <= 0:
            raise ValueError("trend match weights must have a positive total")
        return self


class TrendMatchCommonDeductions(Section):
    lower_extreme_position_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    upper_extreme_position_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    one_sided_move_min_pct: Positive
    one_sided_move: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    abnormal_return_min_pct: Positive
    abnormal_return: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    structure_conflict: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.lower_extreme_position_pct >= self.upper_extreme_position_pct:
            raise ValueError("trend match extreme position thresholds must be increasing")
        return self


class TrendMatch(Section):
    weights: TrendMatchWeights
    score_aggregation: Literal["max", "mean"]
    direction_bias_min_score: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    direction_bias_min_delta: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    common_deductions: TrendMatchCommonDeductions
    range_lookback_bars: Count
    preferred_position_min_pct: Annotated[float, Field(ge=0, lt=100, allow_inf_nan=False)]
    preferred_position_max_pct: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)]
    upper_position_penalty_start_pct: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)]
    upper_position_zero_score_pct: Annotated[float, Field(gt=0, le=200, allow_inf_nan=False)]
    below_min_position_score: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    upper_position_score_at_penalty_start: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    ema50_distance_full_score_pct: NonNegative
    ema50_distance_zero_score_pct: Positive
    recent_return_lookback_bars: Count
    recent_return_no_penalty_pct: NonNegative
    recent_return_zero_score_pct: Positive
    adx_rising_lookback_bars: Count
    adx_min_rise: Positive
    adx_preferred_max: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)]
    adx_extreme_max: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)]
    ema50_slope_full_score_min_pct: NonNegative
    ema50_slope_full_score_max_pct: Positive
    ema50_slope_zero_score_pct: Positive

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if not (self.preferred_position_min_pct < self.preferred_position_max_pct
                <= self.upper_position_penalty_start_pct < self.upper_position_zero_score_pct):
            raise ValueError("trend match range-position thresholds must be increasing")
        if self.ema50_distance_full_score_pct >= self.ema50_distance_zero_score_pct:
            raise ValueError("trend match EMA50 distance zero threshold must exceed full-score threshold")
        if self.recent_return_no_penalty_pct >= self.recent_return_zero_score_pct:
            raise ValueError("trend match recent-return zero threshold must exceed no-penalty threshold")
        if self.adx_preferred_max >= self.adx_extreme_max:
            raise ValueError("trend match extreme ADX threshold must exceed preferred threshold")
        if not (self.ema50_slope_full_score_min_pct < self.ema50_slope_full_score_max_pct
                < self.ema50_slope_zero_score_pct):
            raise ValueError("trend match EMA50 slope thresholds must be increasing")
        return self


class TrendStrategy(Section):
    risk_per_trade_pct: Fraction
    entry_ema_bars: Count
    pullback_ttl_minutes: Annotated[int, Field(gt=0, lt=60)]
    pullback_atr: Positive
    stop_mode: Literal["structure", "atr"]
    stop_swing_lookback_1h: Count
    structure_buffer_atr: Positive
    min_stop_atr: Positive
    max_stop_atr: Positive
    atr_stop_multiple: Positive
    exit_ema_bars: Count
    breakeven_trigger_r: Positive
    trailing_trigger_r: Positive
    trailing_atr_multiple: Positive

    @model_validator(mode="after")
    def valid_trend_policy(self):
        if self.min_stop_atr >= self.max_stop_atr:
            raise ValueError("trend minimum stop ATR must be below maximum stop ATR")
        if self.breakeven_trigger_r >= self.trailing_trigger_r:
            raise ValueError("trend breakeven trigger must be below trailing trigger")
        return self


class LowPositionWatch(Section):
    enabled: bool
    range_position_min_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    range_position_max_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    ema50_distance_min_pct: Annotated[float, Field(allow_inf_nan=False)]
    ema50_distance_max_pct: Annotated[float, Field(allow_inf_nan=False)]
    recent_return_min_pct: Annotated[float, Field(allow_inf_nan=False)]
    recent_return_max_pct: Annotated[float, Field(allow_inf_nan=False)]
    ema50_slope_min_pct: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.range_position_min_pct >= self.range_position_max_pct:
            raise ValueError("low-position watch range-position thresholds must be increasing")
        if self.ema50_distance_min_pct >= self.ema50_distance_max_pct:
            raise ValueError("low-position watch EMA50-distance thresholds must be increasing")
        if self.recent_return_min_pct >= self.recent_return_max_pct:
            raise ValueError("low-position watch recent-return thresholds must be increasing")
        return self


class BollingerObservation(Section):
    enabled: bool
    period: Annotated[int, Field(ge=2, le=500)]
    standard_deviations: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    width_lookback_bars: Annotated[int, Field(ge=10, le=500)]
    width_low_quantile: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    middle_slope_lookback_bars: Annotated[int, Field(ge=1, le=100)]
    middle_flat_tolerance_pct: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    expansion_lookback_bars: Annotated[int, Field(ge=1, le=50)]
    volume_filter_enabled: bool


class Universe(Section):
    min_quote_volume_24h: Positive
    min_list_days: Annotated[int, Field(ge=0)]
    exclude_symbols: list[str]


class Fees(Section):
    taker_bps: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    slippage_bps: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", hide_input_in_errors=True)
    binance_api_key: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    binance_api_secret: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    binance_testnet: bool = True
    bot_mode: Literal["paper", "live"] = "paper"
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    config_path: Path = Path("configs/default.yaml")
    vsb_execution_mode: Literal["DISABLED", "SHADOW", "LIVE"] | None = None


class Settings(Section):
    vsb_execution_mode: Literal["DISABLED", "SHADOW", "LIVE"] = "SHADOW"
    exchange: Literal["binanceusdm"]
    market_type: Literal["swap"]
    settle: Literal["USDT"]
    timezone: Literal["Asia/Shanghai"]
    timeframes: Timeframes
    lookback: Lookback
    regime: Regime
    signals: Signals
    long_entry: LongEntry
    short_entry: ShortEntry
    execution: Execution
    exit_management: ExitManagement
    mr_candidate_exit: MrCandidateExit
    exit_proposal_v1: ExitProposalV1
    exits: Exits
    stops: Stops
    cooldowns: Cooldowns
    session: Session
    sizing: Sizing
    pyramiding: Pyramiding
    risk: Risk
    lead_risk: LeadRisk
    market_risk: MarketRisk
    symbol_regime: SymbolRegime
    mean_reversion_match: MeanReversionMatch
    trend_match: TrendMatch
    trend_strategy: TrendStrategy
    low_position_watch: LowPositionWatch
    bollinger_observation: BollingerObservation
    universe: Universe
    fees: Fees
    runtime: RuntimeSettings = Field(repr=False, exclude=True)

    @field_validator("timezone")
    @classmethod
    def available_timezone(cls, value):
        ZoneInfo(value)
        return value


def load_settings(config_path: str | Path | None = None, *, env_file: str | Path | None = PROJECT_ROOT / ".env") -> Settings:
    """Explicit path > CONFIG_PATH from environment/.env > project default.

    Relative YAML paths resolve against the project, independent of working directory.
    Environment variables override .env. YAML cannot provide private runtime fields.
    """
    runtime = RuntimeSettings(_env_file=env_file, _env_file_encoding="utf-8")
    path = Path(config_path) if config_path is not None else runtime.config_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a YAML mapping")
    if "runtime" in payload:
        raise ValueError("runtime settings must come from environment, not YAML")
    if runtime.vsb_execution_mode is not None:
        payload["vsb_execution_mode"] = runtime.vsb_execution_mode
    return Settings(**payload, runtime=runtime)


def __getattr__(name: str):
    # Supports the requested `from bot.config import settings` without eager I/O.
    if name == "settings":
        return load_settings()
    raise AttributeError(name)
