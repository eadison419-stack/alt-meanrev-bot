"""Log-only per-symbol trend/range classification from completed 4H candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Mapping

import pandas as pd

from bot.data.klines import TIMEFRAME_MS
from bot.indicators import indicator_frame, validate_ohlcv

FOUR_HOURS = pd.Timedelta(TIMEFRAME_MS["4h"], unit="ms")
STALE_GRACE = pd.Timedelta(5, unit="min")


class SymbolRegimeState(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class SymbolRegimeAssessment:
    symbol: str
    state: SymbolRegimeState
    reason: str
    checked_at: datetime
    last_closed_bar_at: datetime | None
    evidence: dict

    def to_dict(self) -> dict:
        return {
            "mode": "strategy_routing",
            "symbol": self.symbol,
            "timeframe": "4h",
            "state": self.state.value,
            "reason": self.reason,
            "checked_at": self.checked_at.isoformat(),
            "last_closed_bar_at": self.last_closed_bar_at.isoformat() if self.last_closed_bar_at else None,
            "evidence": self.evidence,
        }


def _thresholds(policy) -> dict[str, float | int | bool]:
    return {
        "enabled": policy.enabled,
        "min_closed_bars": policy.min_closed_bars,
        "adx_period": policy.adx_period,
        "trend_adx_min": policy.trend_adx_min,
        "range_adx_max": policy.range_adx_max,
        "ema_fast_bars": policy.ema_fast_bars,
        "ema_slow_bars": policy.ema_slow_bars,
        "slope_lookback_bars": policy.slope_lookback_bars,
        "trend_slope_min_pct": policy.trend_slope_min_pct,
        "range_slope_max_pct": policy.range_slope_max_pct,
        "range_max_distance_pct": policy.range_max_distance_pct,
        "structure_lookback_bars": policy.structure_lookback_bars,
        "confirm_bars": policy.confirm_bars,
        "range_confirm_bars": policy.range_confirm_bars,
        "stale_after_hours": 4 + STALE_GRACE.total_seconds() / 3600,
    }


def _structure(frame: pd.DataFrame, index: int, lookback: int) -> tuple[str, dict[str, bool]]:
    prior = frame.iloc[index - 2 * lookback + 1:index - lookback + 1]
    recent = frame.iloc[index - lookback + 1:index + 1]
    higher_high = float(recent["high"].max()) > float(prior["high"].max())
    lower_high = float(recent["high"].max()) < float(prior["high"].max())
    higher_low = float(recent["low"].min()) > float(prior["low"].min())
    lower_low = float(recent["low"].min()) < float(prior["low"].min())
    if higher_high and higher_low:
        state = "UP"
    elif lower_high and lower_low:
        state = "DOWN"
    elif (higher_high and lower_low) or (lower_high and higher_low):
        state = "CONFLICT"
    else:
        state = "NEUTRAL"
    return state, {
        "higher_high": higher_high,
        "lower_high": lower_high,
        "higher_low": higher_low,
        "lower_low": lower_low,
    }


def _range_structure_ok(structure_state: str, mode: str) -> bool:
    """Whether a structure state satisfies the RANGE gate under a given mode.

    "neutral" (production default) requires a flat NEUTRAL structure;
    "not_conflict" accepts NEUTRAL/UP/DOWN (only excludes CONFLICT);
    "any" drops the structure gate entirely.
    """
    if mode == "any":
        return True
    if mode == "not_conflict":
        return structure_state in {"NEUTRAL", "UP", "DOWN"}
    return structure_state == "NEUTRAL"


def _structure_states(frame: pd.DataFrame, lookback: int) -> pd.Series:
    """Vectorized per-bar structure states, equivalent to _structure per bar.

    recent = the last ``lookback`` bars; prior = the ``lookback`` bars before them.
    UP = higher high and higher low; DOWN = lower high and lower low;
    CONFLICT = expanding/contracting both ways; else NEUTRAL.
    """
    high, low = frame["high"], frame["low"]
    recent_high = high.rolling(lookback, min_periods=lookback).max()
    prior_high = high.shift(lookback).rolling(lookback, min_periods=lookback).max()
    recent_low = low.rolling(lookback, min_periods=lookback).min()
    prior_low = low.shift(lookback).rolling(lookback, min_periods=lookback).min()
    higher_high = recent_high > prior_high
    lower_high = recent_high < prior_high
    higher_low = recent_low > prior_low
    lower_low = recent_low < prior_low
    states = pd.Series("NEUTRAL", index=frame.index, dtype=object)
    states[higher_high & higher_low] = "UP"
    states[lower_high & lower_low] = "DOWN"
    states[(higher_high & lower_low) | (lower_high & higher_low)] = "CONFLICT"
    return states


def _raw_candidates(closed: pd.DataFrame, policy) -> tuple[list[str], dict]:
    adx = indicator_frame(closed, policy.adx_period, policy.adx_period)["adx"]
    ema_fast = closed["close"].ewm(span=policy.ema_fast_bars, adjust=False,
                                   min_periods=policy.ema_fast_bars).mean()
    ema_slow = closed["close"].ewm(span=policy.ema_slow_bars, adjust=False,
                                   min_periods=policy.ema_slow_bars).mean()
    structure_states = _structure_states(closed, policy.structure_lookback_bars)
    adx_arr = adx.to_numpy(dtype=float)
    ema_fast_arr = ema_fast.to_numpy(dtype=float)
    ema_slow_arr = ema_slow.to_numpy(dtype=float)
    close_arr = closed["close"].to_numpy(dtype=float)
    structure_states_arr = structure_states.to_numpy(dtype=object)
    candidates: list[str] = []
    reasons: list[str] = []
    range_structure_shadow_bars: list[dict | None] = []
    metrics: dict = {}
    for index in range(len(closed)):
        if index < max(policy.ema_slow_bars + policy.slope_lookback_bars - 1,
                       2 * policy.adx_period - 1,
                       2 * policy.structure_lookback_bars - 1):
            candidates.append("UNCERTAIN")
            reasons.append("indicator_warmup")
            range_structure_shadow_bars.append(None)
            continue
        adx_value = float(adx_arr[index])
        fast_value = float(ema_fast_arr[index])
        slow_value = float(ema_slow_arr[index])
        close = float(close_arr[index])
        slope_reference = float(ema_slow_arr[index - policy.slope_lookback_bars])
        if (not all(math.isfinite(value) for value in
                    (adx_value, fast_value, slow_value, close, slope_reference))
                or slow_value <= 0 or slope_reference <= 0):
            candidates.append("UNCERTAIN")
            reasons.append("indicators_unavailable")
            range_structure_shadow_bars.append(None)
            continue

        slope_pct = slow_value / slope_reference - 1
        distance_pct = close / slow_value - 1
        structure_state = structure_states_arr[index]
        structure_flags: dict[str, bool] = {}
        if index == len(closed) - 1:
            _, structure_flags = _structure(closed, index, policy.structure_lookback_bars)
        ma_up_ordered = close > fast_value > slow_value
        ma_down_ordered = close < fast_value < slow_value
        trend_slope_up = slope_pct >= policy.trend_slope_min_pct
        trend_slope_down = slope_pct <= -policy.trend_slope_min_pct
        range_slope_flat = abs(slope_pct) <= policy.range_slope_max_pct
        range_distance_ok = abs(distance_pct) <= policy.range_max_distance_pct
        trend_adx_ok = adx_value >= policy.trend_adx_min
        range_adx_ok = adx_value <= policy.range_adx_max
        range_structure_shadow_bars.append({
            "bar_at": closed.index[index].isoformat(),
            "adx": adx_value,
            "ema50_slope_pct": slope_pct,
            "close_to_ema50_pct": distance_pct,
            "numeric_conditions_pass": bool(
                range_adx_ok and range_slope_flat and range_distance_ok
            ),
            "structure": structure_state,
        })
        ma_up = ma_up_ordered and trend_slope_up
        ma_down = ma_down_ordered and trend_slope_down
        trend_up = trend_adx_ok and ma_up and structure_state == "UP"
        trend_down = trend_adx_ok and ma_down and structure_state == "DOWN"
        in_range = (
            range_adx_ok
            and range_slope_flat
            and range_distance_ok
            and _range_structure_ok(structure_state, policy.range_structure_mode)
        )
        if trend_up:
            candidate, reason = "TREND_UP", "trend_components_aligned_up"
        elif trend_down:
            candidate, reason = "TREND_DOWN", "trend_components_aligned_down"
        elif in_range:
            candidate, reason = "RANGE", "range_components_aligned"
        elif adx_value >= policy.trend_adx_min and structure_state == "CONFLICT":
            candidate, reason = "UNCERTAIN", "trend_structure_conflict"
        elif adx_value >= policy.trend_adx_min and (
                structure_state == "UP" and not ma_up or structure_state == "DOWN" and not ma_down):
            candidate, reason = "UNCERTAIN", "trend_components_conflict"
        elif structure_state == "CONFLICT":
            candidate, reason = "UNCERTAIN", "structure_conflict"
        else:
            candidate, reason = "UNCERTAIN", "no_clear_regime"
        candidates.append(candidate)
        reasons.append(reason)
        if index == len(closed) - 1:
            metrics = {
                "close": close,
                "adx": adx_value,
                "ema_fast": fast_value,
                "ema_slow": slow_value,
                "slow_ema_slope_pct": slope_pct,
                "close_to_slow_ema_pct": distance_pct,
                "structure": structure_state,
                "structure_flags": structure_flags,
                "ma_up_aligned": ma_up,
                "ma_down_aligned": ma_down,
                "condition_checks": {
                    "TREND_UP": {
                        "adx_at_or_above_min": trend_adx_ok,
                        "ema20_ema50_order_up": ma_up_ordered,
                        "ema50_slope_at_or_above_min": trend_slope_up,
                        "structure_up": structure_state == "UP",
                    },
                    "TREND_DOWN": {
                        "adx_at_or_above_min": trend_adx_ok,
                        "ema20_ema50_order_down": ma_down_ordered,
                        "ema50_slope_at_or_below_negative_min": trend_slope_down,
                        "structure_down": structure_state == "DOWN",
                    },
                    "RANGE": {
                        "adx_at_or_below_max": range_adx_ok,
                        "ema50_slope_flat": range_slope_flat,
                        "close_within_ema50_distance": range_distance_ok,
                        "structure_neutral": structure_state == "NEUTRAL",
                    },
                },
                "condition_failures": {
                    "TREND_UP": [
                        name for name, passed in {
                            "adx_below_trend_min": trend_adx_ok,
                            "ema_order_not_up": ma_up_ordered,
                            "ema50_slope_below_trend_min": trend_slope_up,
                            "structure_not_up": structure_state == "UP",
                        }.items() if not passed
                    ],
                    "TREND_DOWN": [
                        name for name, passed in {
                            "adx_below_trend_min": trend_adx_ok,
                            "ema_order_not_down": ma_down_ordered,
                            "ema50_slope_above_negative_trend_min": trend_slope_down,
                            "structure_not_down": structure_state == "DOWN",
                        }.items() if not passed
                    ],
                    "RANGE": [
                        name for name, passed in {
                            "adx_above_range_max": range_adx_ok,
                            "ema50_slope_not_flat": range_slope_flat,
                            "close_too_far_from_ema50": range_distance_ok,
                            "structure_not_neutral": structure_state == "NEUTRAL",
                        }.items() if not passed
                    ],
                },
                "raw_candidate": candidate,
                "raw_reason": reason,
            }
    return candidates, {
        "metrics": metrics,
        "raw_reasons": reasons,
        "range_structure_shadow_bars": range_structure_shadow_bars,
    }


def _near_miss_classification(state: SymbolRegimeState, raw_candidate: str | None,
                              condition_failures: dict, *, trend_confirmation_pass: bool,
                              range_confirmation_pass: bool) -> dict:
    """Describe uncertain assessments that miss one primary gate, without changing state."""
    if state is not SymbolRegimeState.UNCERTAIN:
        return {"near_trend": False, "near_range": False, "near_miss_conditions": {}}

    misses: dict[str, list[str]] = {}
    if raw_candidate in {"TREND_UP", "TREND_DOWN"}:
        if not trend_confirmation_pass:
            misses[raw_candidate] = ["trend_confirmation_not_met"]
    elif raw_candidate == "RANGE":
        if not range_confirmation_pass:
            misses["RANGE"] = ["range_confirmation_not_met"]
    else:
        for strategy in ("TREND_UP", "TREND_DOWN", "RANGE"):
            failures = list(condition_failures.get(strategy) or [])
            if len(failures) == 1:
                misses[strategy] = failures

    return {
        "near_trend": any(strategy.startswith("TREND_") for strategy in misses),
        "near_range": "RANGE" in misses,
        "near_miss_conditions": misses,
    }


def classify_symbol_regime(symbol: str, bars: pd.DataFrame | None, settings, *, now: datetime,
                           fetch_error: str | None = None) -> SymbolRegimeAssessment:
    """Classify one symbol from 4H OHLCV, independently excluding unclosed bars."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("symbol regime evaluation time must be timezone-aware")
    checked_at = now.astimezone(timezone.utc)
    policy = settings.symbol_regime
    thresholds = _thresholds(policy)

    def result(state: SymbolRegimeState, reason: str, evidence: dict | None = None,
               last_closed_bar_at: datetime | None = None):
        details = {
            "thresholds": thresholds,
            "near_trend": False,
            "near_range": False,
            "near_miss_conditions": {},
            **(evidence or {}),
        }
        if fetch_error:
            details["fetch_error"] = fetch_error[:300]
        return SymbolRegimeAssessment(symbol, state, reason, checked_at, last_closed_bar_at, details)

    if not policy.enabled:
        return result(SymbolRegimeState.UNCERTAIN, "observer_disabled")
    if bars is None or bars.empty:
        return result(SymbolRegimeState.UNCERTAIN, "bars_missing")

    try:
        validate_ohlcv(bars)
        now_ts = pd.Timestamp(checked_at)
        closed = bars.loc[bars.index + FOUR_HOURS <= now_ts]
        required = max(
            policy.min_closed_bars,
            policy.ema_slow_bars + policy.slope_lookback_bars,
            2 * policy.adx_period,
            2 * policy.structure_lookback_bars,
        )
        last_closed_at = closed.index[-1] + FOUR_HOURS if not closed.empty else None
        last_closed = last_closed_at.to_pydatetime() if last_closed_at is not None else None
        if len(closed) < required:
            return result(SymbolRegimeState.UNCERTAIN, "insufficient_closed_bars", {
                "closed_bar_count": int(len(closed)),
                "required_closed_bar_count": int(required),
                "last_closed_bar_at": last_closed.isoformat() if last_closed else None,
            }, last_closed)
        recent = closed.index[-required:]
        if len(recent) > 1 and not (recent[1:] - recent[:-1] == FOUR_HOURS).all():
            return result(SymbolRegimeState.UNCERTAIN, "closed_bar_gap", {
                "closed_bar_count": int(len(closed)),
                "required_closed_bar_count": int(required),
                "last_closed_bar_at": last_closed.isoformat() if last_closed else None,
            }, last_closed)

        closed_age = now_ts - last_closed_at
        if closed_age > FOUR_HOURS + STALE_GRACE:
            return result(SymbolRegimeState.UNCERTAIN, "stale_closed_bar", {
                "closed_bar_count": int(len(closed)),
                "closed_bar_age_seconds": float(closed_age.total_seconds()),
            }, last_closed)

        candidates, raw = _raw_candidates(closed, policy)
        trend_confirmation = candidates[-policy.confirm_bars:]
        trend_reasons = raw["raw_reasons"][-policy.confirm_bars:]
        stable_trend = (len(trend_confirmation) == policy.confirm_bars
                        and len(set(trend_confirmation)) == 1)
        range_confirmation = candidates[-policy.range_confirm_bars:]
        range_reasons = raw["raw_reasons"][-policy.range_confirm_bars:]
        stable_range = (len(range_confirmation) == policy.range_confirm_bars
                        and len(set(range_confirmation)) == 1)
        raw_candidate = raw["metrics"].get("raw_candidate")
        if raw_candidate in {"TREND_UP", "TREND_DOWN"}:
            confirmation, reasons = trend_confirmation, trend_reasons
            stable_candidate = stable_trend
        elif raw_candidate == "RANGE":
            confirmation, reasons = range_confirmation, range_reasons
            stable_candidate = stable_range
        else:
            confirmation, reasons = trend_confirmation, trend_reasons
            stable_candidate = False
        trend_current_streak = 0
        if raw_candidate in {"TREND_UP", "TREND_DOWN"}:
            for candidate in reversed(candidates):
                if candidate != raw_candidate:
                    break
                trend_current_streak += 1
        range_current_streak = 0
        if raw_candidate == "RANGE":
            for candidate in reversed(candidates):
                if candidate != "RANGE":
                    break
                range_current_streak += 1
        if stable_trend and trend_confirmation[0] in {"TREND_UP", "TREND_DOWN"}:
            state = SymbolRegimeState.TREND
            reason = ("trend_confirmed_up" if trend_confirmation[0] == "TREND_UP"
                      else "trend_confirmed_down")
        elif stable_range and range_confirmation[0] == "RANGE":
            state, reason = SymbolRegimeState.RANGE, "range_confirmed"
        elif raw_candidate == "UNCERTAIN":
            state, reason = SymbolRegimeState.UNCERTAIN, raw["metrics"].get("raw_reason", "no_clear_regime")
        else:
            state, reason = SymbolRegimeState.UNCERTAIN, "confirmation_not_met"

        near_miss = _near_miss_classification(
            state, raw_candidate, raw["metrics"].get("condition_failures", {}),
            trend_confirmation_pass=(stable_trend
                                     and trend_confirmation[0] in {"TREND_UP", "TREND_DOWN"}),
            range_confirmation_pass=stable_range and range_confirmation[0] == "RANGE",
        )
        shadow_bars = raw.get("range_structure_shadow_bars", [])[-policy.range_confirm_bars:]
        structure_only_pass = (
            state is SymbolRegimeState.UNCERTAIN
            and len(shadow_bars) == policy.range_confirm_bars
            and all(
                item is not None
                and item["numeric_conditions_pass"]
                and item["structure"] in {"UP", "DOWN"}
                for item in shadow_bars
            )
        )
        shadow_structures = [item["structure"] for item in shadow_bars if item is not None]

        evidence = {
            "closed_bar_count": int(len(closed)),
            "last_closed_bar_at": last_closed.isoformat(),
            "closed_bar_age_seconds": float(closed_age.total_seconds()),
            **raw["metrics"],
            "confirmation_candidates": confirmation,
            "confirmation_reasons": reasons,
            "confirmation_pass": bool(stable_candidate and confirmation[0] != "UNCERTAIN"),
            "trend_confirmation_candidates": trend_confirmation,
            "trend_confirmation_reasons": trend_reasons,
            "trend_confirmation_required_bars": int(policy.confirm_bars),
            "trend_confirmation_current_streak": int(trend_current_streak),
            "trend_confirmation_pass": bool(stable_trend and trend_confirmation[0] in {"TREND_UP", "TREND_DOWN"}),
            "range_confirmation_candidates": range_confirmation,
            "range_confirmation_reasons": range_reasons,
            "range_confirmation_required_bars": int(policy.range_confirm_bars),
            "range_confirmation_current_streak": int(range_current_streak),
            "range_confirmation_pass": bool(stable_range and range_confirmation[0] == "RANGE"),
            "near_range_structure_only": bool(structure_only_pass),
            "range_structure_shadow": {
                "eligible": bool(structure_only_pass),
                "required_bars": int(policy.range_confirm_bars),
                "direction_consistent": bool(
                    len(shadow_structures) == policy.range_confirm_bars
                    and len(set(shadow_structures)) == 1
                ),
                "bars": shadow_bars,
            },
            **near_miss,
        }
        return result(state, reason, evidence, last_closed)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return result(SymbolRegimeState.UNCERTAIN, "invalid_bars_or_indicators", {
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        })


def classify_symbol_regimes(bars_by_symbol: Mapping[str, pd.DataFrame | None], settings, *, now: datetime,
                            fetch_errors: Mapping[str, str] | None = None) -> dict[str, SymbolRegimeAssessment]:
    """Return independent observational classifications for a hot-symbol batch."""
    fetch_errors = fetch_errors or {}
    return {
        symbol: classify_symbol_regime(symbol, bars, settings, now=now,
                                       fetch_error=fetch_errors.get(symbol))
        for symbol, bars in bars_by_symbol.items()
    }
