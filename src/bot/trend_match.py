"""Observation-only trend suitability scoring from completed 4H candles."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd

from bot.data.klines import TIMEFRAME_MS
from bot.indicators import indicator_frame, validate_ohlcv

FOUR_HOURS = pd.Timedelta(TIMEFRAME_MS["4h"], unit="ms")
STALE_GRACE = pd.Timedelta(5, unit="min")


def _position_score(position_pct: float, policy) -> float:
    if position_pct < policy.preferred_position_min_pct:
        return float(policy.below_min_position_score)
    if position_pct <= policy.preferred_position_max_pct:
        return 100.0
    if position_pct <= policy.upper_position_penalty_start_pct:
        width = policy.upper_position_penalty_start_pct - policy.preferred_position_max_pct
        progress = (position_pct - policy.preferred_position_max_pct) / width
        return 100.0 + progress * (policy.upper_position_score_at_penalty_start - 100.0)
    if position_pct < policy.upper_position_zero_score_pct:
        width = policy.upper_position_zero_score_pct - policy.upper_position_penalty_start_pct
        progress = (position_pct - policy.upper_position_penalty_start_pct) / width
        return policy.upper_position_score_at_penalty_start * (1.0 - progress)
    return 0.0


def _distance_score(distance_pct: float, full_score_pct: float, zero_score_pct: float) -> float:
    distance = abs(distance_pct)
    if distance <= full_score_pct:
        return 100.0
    if distance >= zero_score_pct:
        return 0.0
    return 100.0 * (zero_score_pct - distance) / (zero_score_pct - full_score_pct)


def _recent_return_score(return_pct: float, no_penalty_pct: float, zero_score_pct: float) -> float:
    if return_pct <= no_penalty_pct:
        return 100.0
    if return_pct >= zero_score_pct:
        return 0.0
    return 100.0 * (zero_score_pct - return_pct) / (zero_score_pct - no_penalty_pct)


def _adx_rising_score(adx: float, change: float, policy) -> float:
    if adx >= policy.adx_extreme_max:
        return 0.0
    if change >= policy.adx_min_rise:
        score = 100.0
    elif change > 0:
        score = 75.0
    elif change >= -policy.adx_min_rise:
        score = 45.0
    else:
        score = 20.0
    if adx > policy.adx_preferred_max:
        score *= ((policy.adx_extreme_max - adx)
                  / (policy.adx_extreme_max - policy.adx_preferred_max))
    return score


def _ema_slope_score(slope_pct: float, policy) -> float:
    if slope_pct <= 0:
        return 0.0
    if slope_pct < policy.ema50_slope_full_score_min_pct:
        return 100.0 * slope_pct / policy.ema50_slope_full_score_min_pct
    if slope_pct <= policy.ema50_slope_full_score_max_pct:
        return 100.0
    if slope_pct >= policy.ema50_slope_zero_score_pct:
        return 0.0
    return 100.0 * (
        (policy.ema50_slope_zero_score_pct - slope_pct)
        / (policy.ema50_slope_zero_score_pct - policy.ema50_slope_full_score_max_pct)
    )


def _structure(frame: pd.DataFrame, lookback: int) -> str:
    prior = frame.iloc[-2 * lookback:-lookback]
    recent = frame.iloc[-lookback:]
    higher_high = float(recent["high"].max()) > float(prior["high"].max())
    lower_high = float(recent["high"].max()) < float(prior["high"].max())
    higher_low = float(recent["low"].min()) > float(prior["low"].min())
    lower_low = float(recent["low"].min()) < float(prior["low"].min())
    if higher_high and higher_low:
        return "UP"
    if lower_high and lower_low:
        return "DOWN"
    if (higher_high and lower_low) or (lower_high and higher_low):
        return "CONFLICT"
    return "NEUTRAL"


def _common_deductions(position_pct: float, recent_return_pct: float, structure: str, policy) -> dict:
    rules = policy.common_deductions
    deductions = {}
    at_extreme_high_after_rally = (
        position_pct >= rules.upper_extreme_position_pct
        and recent_return_pct >= rules.one_sided_move_min_pct
    )
    at_extreme_low_after_drop = (
        position_pct <= rules.lower_extreme_position_pct
        and recent_return_pct <= -rules.one_sided_move_min_pct
    )
    if (at_extreme_high_after_rally or at_extreme_low_after_drop) and rules.one_sided_move:
        deductions["extreme_one_sided_move"] = float(rules.one_sided_move)
    if abs(recent_return_pct) >= rules.abnormal_return_min_pct and rules.abnormal_return:
        deductions["abnormal_recent_move"] = float(rules.abnormal_return)
    if structure == "CONFLICT" and rules.structure_conflict:
        deductions["structure_conflict"] = float(rules.structure_conflict)
    return deductions


def _directional_score(component_scores: dict, weights: dict, common_deductions: dict,
                       reason_labels: dict) -> dict:
    total_weight = sum(float(value) for value in weights.values())
    weighted_contributions = {
        key: component_scores[key] * float(weight) / total_weight
        for key, weight in weights.items()
    }
    component_deductions = {
        key: (100.0 - component_scores[key]) * float(weight) / total_weight
        for key, weight in weights.items()
    }
    raw_score = sum(weighted_contributions.values())
    score = round(min(100.0, max(0.0, raw_score - sum(common_deductions.values()))), 2)
    positive = sorted(
        (key for key, value in component_scores.items() if value >= 65),
        key=lambda key: weighted_contributions[key], reverse=True,
    )[:3]
    negative = sorted(
        (key for key, value in component_scores.items() if value < 65),
        key=lambda key: component_deductions[key], reverse=True,
    )
    return {
        "score": score,
        "component_scores": {key: round(value, 2) for key, value in component_scores.items()},
        "weighted_contributions": {
            key: round(value, 2) for key, value in weighted_contributions.items()
        },
        "deductions": {key: round(value, 2) for key, value in component_deductions.items()},
        "common_deductions": common_deductions,
        "reasons": [reason_labels[key] for key in positive],
        "deduction_reasons": [reason_labels[key] for key in negative]
        + list(common_deductions),
    }


def evaluate_trend_match(symbol: str, bars_4h: pd.DataFrame | None, settings, *, now: datetime) -> dict:
    """Return a 0–100 trend-suitability diagnostic without affecting trading decisions."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("trend match evaluation time must be timezone-aware")
    checked_at = now.astimezone(timezone.utc)
    policy = settings.trend_match
    regime = settings.symbol_regime

    def unavailable(reason: str, *, last_closed_bar_at: str | None = None) -> dict:
        empty_side = {
            "score": None,
            "component_scores": {},
            "weighted_contributions": {},
            "deductions": {},
            "common_deductions": {},
            "reasons": [],
            "deduction_reasons": [],
        }
        return {
            "symbol": symbol,
            "score": None,
            "trend_long_match_score": None,
            "trend_short_match_score": None,
            "trend_match_score": None,
            "direction_bias": "neutral",
            "score_aggregation": policy.score_aggregation,
            "status": "unavailable",
            "unavailable_reason": reason,
            "last_closed_bar_at": last_closed_bar_at,
            "metrics": {},
            "component_scores": {},
            "weighted_contributions": {},
            "deductions": {},
            "reasons": [reason],
            "deduction_reasons": [],
            "long": empty_side.copy(),
            "short": empty_side.copy(),
            "common_deductions": {},
        }

    if bars_4h is None or bars_4h.empty:
        return unavailable("bars_missing")
    try:
        validate_ohlcv(bars_4h)
        closed = bars_4h.loc[bars_4h.index + FOUR_HOURS <= pd.Timestamp(checked_at)]
        last_closed_at = closed.index[-1] + FOUR_HOURS if not closed.empty else None
        last_closed = last_closed_at.isoformat() if last_closed_at is not None else None
        required = max(
            policy.range_lookback_bars + 1,
            policy.recent_return_lookback_bars + 1,
            regime.ema_slow_bars + regime.slope_lookback_bars,
            2 * regime.adx_period + policy.adx_rising_lookback_bars,
            2 * regime.structure_lookback_bars,
        )
        if len(closed) < required:
            return unavailable("insufficient_closed_bars", last_closed_bar_at=last_closed)
        if len(closed) > 1 and not (closed.index[1:] - closed.index[:-1] == FOUR_HOURS).all():
            return unavailable("closed_bar_gap", last_closed_bar_at=last_closed)
        closed_age = pd.Timestamp(checked_at) - last_closed_at
        if closed_age > FOUR_HOURS + STALE_GRACE:
            return unavailable("stale_closed_bar", last_closed_bar_at=last_closed)

        close = float(closed["close"].iloc[-1])
        prior_range = closed.iloc[-(policy.range_lookback_bars + 1):-1]
        range_high = float(prior_range["high"].max())
        range_low = float(prior_range["low"].min())
        range_width = range_high - range_low
        if range_width <= 0 or close <= 0:
            return unavailable("invalid_prior_range", last_closed_bar_at=last_closed)
        range_position_pct = 100.0 * (close - range_low) / range_width

        ema50 = closed["close"].ewm(
            span=regime.ema_slow_bars, adjust=False, min_periods=regime.ema_slow_bars,
        ).mean()
        current_ema50 = float(ema50.iloc[-1])
        slope_lookback = regime.slope_lookback_bars
        reference_ema50 = float(ema50.iloc[-1 - slope_lookback])
        if current_ema50 <= 0 or reference_ema50 <= 0:
            return unavailable("ema50_unavailable", last_closed_bar_at=last_closed)
        ema50_distance_pct = 100.0 * (close / current_ema50 - 1.0)
        ema50_slope_pct = 100.0 * (current_ema50 / reference_ema50 - 1.0)

        return_lookback = policy.recent_return_lookback_bars
        previous_close = float(closed["close"].iloc[-1 - return_lookback])
        if previous_close <= 0:
            return unavailable("invalid_return_reference", last_closed_bar_at=last_closed)
        recent_return_pct = 100.0 * (close / previous_close - 1.0)

        adx_frame = indicator_frame(closed, regime.adx_period, regime.adx_period)
        adx_values = adx_frame["adx"].dropna()
        adx_lookback = policy.adx_rising_lookback_bars
        if len(adx_values) <= adx_lookback:
            return unavailable("adx_history_unavailable", last_closed_bar_at=last_closed)
        adx = float(adx_values.iloc[-1])
        adx_change = adx - float(adx_values.iloc[-1 - adx_lookback])
        structure = _structure(closed, regime.structure_lookback_bars)
        values = (range_position_pct, ema50_distance_pct, ema50_slope_pct,
                  recent_return_pct, adx, adx_change)
        if not all(math.isfinite(value) for value in values):
            return unavailable("non_finite_metrics", last_closed_bar_at=last_closed)
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
        return unavailable(f"classification_unavailable:{type(exc).__name__}")

    ema50_distance_score = _distance_score(
        ema50_distance_pct, policy.ema50_distance_full_score_pct,
        policy.ema50_distance_zero_score_pct,
    )
    adx_score = _adx_rising_score(adx, adx_change, policy)
    long_component_scores = {
        "range_position": _position_score(range_position_pct, policy),
        "ema50_distance": ema50_distance_score,
        "recent_return": _recent_return_score(
            recent_return_pct, policy.recent_return_no_penalty_pct,
            policy.recent_return_zero_score_pct,
        ),
        "adx_rising": adx_score,
        "ema50_slope": _ema_slope_score(ema50_slope_pct, policy),
        "structure": {"UP": 100.0, "NEUTRAL": 65.0, "CONFLICT": 20.0, "DOWN": 0.0}[structure],
    }
    short_oriented_position_pct = 100.0 - range_position_pct
    short_component_scores = {
        "range_position": _position_score(short_oriented_position_pct, policy),
        "ema50_distance": ema50_distance_score,
        "recent_return": _recent_return_score(
            -recent_return_pct, policy.recent_return_no_penalty_pct,
            policy.recent_return_zero_score_pct,
        ),
        "adx_rising": adx_score,
        "ema50_slope": _ema_slope_score(-ema50_slope_pct, policy),
        "structure": {"DOWN": 100.0, "NEUTRAL": 65.0, "CONFLICT": 20.0, "UP": 0.0}[structure],
    }
    weights = policy.weights.model_dump()
    common_deductions = _common_deductions(range_position_pct, recent_return_pct, structure, policy)
    direction_context = {
        "range_position": (
            "range_position_preferred_band"
            if policy.preferred_position_min_pct <= range_position_pct <= policy.preferred_position_max_pct else
            "range_position_low_side_moderate" if range_position_pct < policy.preferred_position_min_pct else
            "range_position_upper_extension_deduction"
            if range_position_pct > policy.upper_position_penalty_start_pct else
            "range_position_upper_mid"
        ),
        "ema50_distance": (
            "price_near_ema50" if ema50_distance_score >= 80 else
            "price_far_above_ema50" if ema50_distance_pct > 0 else "price_far_below_ema50"
        ),
        "adx_rising": (
            "adx_rising_below_extreme" if adx_change >= policy.adx_min_rise and adx < policy.adx_extreme_max else
            "adx_not_rising_or_extreme"
        ),
    }
    long_labels = {
        **direction_context,
        "recent_return": (
            "recent_return_not_extended" if long_component_scores["recent_return"] >= 80 else
            "recent_rally_penalty"
        ),
        "ema50_slope": "ema50_slope_improving" if ema50_slope_pct > 0 else "ema50_slope_not_positive",
        "structure": {
            "UP": "structure_higher_high_and_low",
            "NEUTRAL": "structure_neutral",
            "CONFLICT": "structure_conflict",
            "DOWN": "structure_lower_high_and_low",
        }[structure],
        "extreme_one_sided_move": "extreme_one_sided_move",
        "abnormal_recent_move": "abnormal_recent_move",
        "structure_conflict": "structure_conflict_common_deduction",
    }
    short_labels = {
        **direction_context,
        "range_position": (
            "range_position_preferred_band"
            if 100.0 - policy.preferred_position_max_pct <= range_position_pct
            <= 100.0 - policy.preferred_position_min_pct else
            "range_position_upper_side_moderate"
            if range_position_pct > 100.0 - policy.preferred_position_min_pct else
            "range_position_lower_extension_deduction"
            if range_position_pct < 100.0 - policy.upper_position_penalty_start_pct else
            "range_position_lower_mid"
        ),
        "recent_return": (
            "recent_decline_not_extended" if short_component_scores["recent_return"] >= 80 else
            "recent_decline_penalty"
        ),
        "ema50_slope": "ema50_slope_turning_negative" if ema50_slope_pct < 0 else "ema50_slope_not_negative",
        "structure": {
            "DOWN": "structure_lower_high_and_low",
            "NEUTRAL": "structure_neutral",
            "CONFLICT": "structure_conflict",
            "UP": "structure_higher_high_and_low",
        }[structure],
        "extreme_one_sided_move": "extreme_one_sided_move",
        "abnormal_recent_move": "abnormal_recent_move",
        "structure_conflict": "structure_conflict_common_deduction",
    }
    long_score = _directional_score(long_component_scores, weights, common_deductions, long_labels)
    short_score = _directional_score(short_component_scores, weights, common_deductions, short_labels)
    if policy.score_aggregation == "max":
        composite_score = max(long_score["score"], short_score["score"])
    else:
        composite_score = round((long_score["score"] + short_score["score"]) / 2.0, 2)
    score_delta = long_score["score"] - short_score["score"]
    if (long_score["score"] >= policy.direction_bias_min_score
            and score_delta > 0 and score_delta >= policy.direction_bias_min_delta):
        direction_bias = "long"
    elif (short_score["score"] >= policy.direction_bias_min_score
            and score_delta < 0 and score_delta <= -policy.direction_bias_min_delta):
        direction_bias = "short"
    else:
        direction_bias = "neutral"
    return {
        "symbol": symbol,
        # Keep `score` as the original long-side value for callers that predate dual scoring.
        "score": long_score["score"],
        "trend_long_match_score": long_score["score"],
        "trend_short_match_score": short_score["score"],
        "trend_match_score": composite_score,
        "direction_bias": direction_bias,
        "score_aggregation": policy.score_aggregation,
        "status": "scored",
        "unavailable_reason": None,
        "last_closed_bar_at": last_closed,
        "metrics": {
            "range_lookback_bars": policy.range_lookback_bars,
            "prior_range_high": range_high,
            "prior_range_low": range_low,
            "range_position_pct": range_position_pct,
            "close": close,
            "ema50": current_ema50,
            "ema50_distance_pct": ema50_distance_pct,
            "recent_return_lookback_bars": return_lookback,
            "recent_return_pct": recent_return_pct,
            "adx": adx,
            "adx_rising_lookback_bars": adx_lookback,
            "adx_change": adx_change,
            "ema50_slope_pct": ema50_slope_pct,
            "structure": structure,
        },
        "component_scores": long_score["component_scores"],
        "component_weights": weights,
        "weighted_contributions": long_score["weighted_contributions"],
        "deductions": long_score["deductions"],
        "reasons": long_score["reasons"],
        "deduction_reasons": long_score["deduction_reasons"],
        "long": long_score,
        "short": short_score,
        "common_deductions": common_deductions,
    }
