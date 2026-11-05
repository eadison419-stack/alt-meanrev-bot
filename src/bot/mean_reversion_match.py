"""Observation-only scoring for mean-reversion suitability from completed 4H bars."""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd

from bot.data.klines import TIMEFRAME_MS
from bot.indicators import validate_ohlcv
from bot.regime import classify_regime
from bot.symbol_regime import classify_symbol_regime

FOUR_HOURS = pd.Timedelta(TIMEFRAME_MS["4h"], unit="ms")


def _bounded_score(value: float, ideal_max: float, zero_at: float) -> float:
    if value <= ideal_max:
        return 100.0
    if value >= zero_at:
        return 0.0
    return 100.0 * (zero_at - value) / (zero_at - ideal_max)


def evaluate_mean_reversion_match(symbol: str, bars_4h: pd.DataFrame | None, settings, *,
                                  now: datetime) -> dict:
    """Return a diagnostic 0–100 score without affecting routing or hot ranking.

    Missing, stale, gapped, or insufficient 4H history has no score rather than a
    fabricated zero. Both legacy and per-symbol classifiers receive closed bars only.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("mean-reversion match evaluation time must be timezone-aware")
    checked_at = now.astimezone(timezone.utc)

    def unavailable(reason: str, *, route_state: str | None = None,
                    route_reason: str | None = None, last_closed_bar_at: str | None = None) -> dict:
        return {
            "symbol": symbol,
            "score": None,
            "status": "unavailable",
            "unavailable_reason": reason,
            "route_state": route_state,
            "route_reason": route_reason,
            "legacy_regime": None,
            "last_closed_bar_at": last_closed_bar_at,
            "metrics": {},
            "component_scores": {},
            "penalties": {},
            "reasons": [reason],
        }

    if bars_4h is None or bars_4h.empty:
        return unavailable("bars_missing")
    try:
        validate_ohlcv(bars_4h)
        closed = bars_4h.loc[bars_4h.index + FOUR_HOURS <= pd.Timestamp(checked_at)]
        route = classify_symbol_regime(symbol, closed, settings, now=checked_at)
        last_closed = (closed.index[-1] + FOUR_HOURS).isoformat() if not closed.empty else None
        metrics = route.evidence
        if "adx" not in metrics:
            return unavailable(
                route.reason, route_state=route.state.value, route_reason=route.reason,
                last_closed_bar_at=last_closed,
            )
        legacy = classify_regime(closed, settings)
        required = ("adx", "slow_ema_slope_pct", "close_to_slow_ema_pct", "structure")
        if any(key not in metrics or metrics[key] is None for key in required):
            return unavailable(
                "classification_metrics_unavailable", route_state=route.state.value,
                route_reason=route.reason, last_closed_bar_at=last_closed,
            )
        adx = float(metrics["adx"])
        slope = abs(float(metrics["slow_ema_slope_pct"]))
        distance = abs(float(metrics["close_to_slow_ema_pct"]))
        if not all(math.isfinite(value) for value in (adx, slope, distance, legacy.range_position)):
            return unavailable(
                "classification_metrics_non_finite", route_state=route.state.value,
                route_reason=route.reason, last_closed_bar_at=last_closed,
            )
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
        return unavailable(f"classification_unavailable:{type(exc).__name__}")

    route_state = route.state.value
    raw_candidate = metrics.get("raw_candidate")
    if route_state == "RANGE":
        route_score = 100.0
    elif route_state == "TREND":
        route_score = 0.0
    elif raw_candidate == "RANGE":
        route_score = 60.0
    elif raw_candidate in {"TREND_UP", "TREND_DOWN"}:
        route_score = 20.0
    else:
        route_score = 35.0

    symbol_policy = settings.symbol_regime
    match_policy = settings.mean_reversion_match
    component_scores = {
        "route": route_score,
        "adx": _bounded_score(adx, symbol_policy.range_adx_max, symbol_policy.trend_adx_min),
        "ema50_slope": _bounded_score(
            slope, symbol_policy.range_slope_max_pct, symbol_policy.trend_slope_min_pct,
        ),
        "ema50_distance": _bounded_score(
            distance, symbol_policy.range_max_distance_pct,
            symbol_policy.range_max_distance_pct * match_policy.ema50_distance_zero_multiple,
        ),
        "structure": {"NEUTRAL": 100.0, "UP": 30.0, "DOWN": 30.0, "CONFLICT": 0.0}.get(
            str(metrics["structure"]), 0.0,
        ),
    }
    regime_name = legacy.regime.value
    position = float(legacy.range_position)
    if regime_name == "SIDEWAYS":
        legacy_score = 100.0
    elif regime_name == "MID":
        legacy_score = max(0.0, 100.0 * (1.0 - abs(position - 0.5) / 0.5))
    elif regime_name in {"LOW", "HIGH"}:
        legacy_score = 55.0
    elif regime_name in {"SLOW_UP", "SLOW_DOWN"}:
        legacy_score = 25.0
    else:
        legacy_score = 0.0
    component_scores["legacy_regime"] = legacy_score

    weights = match_policy.weights.model_dump()
    weight_total = sum(float(value) for value in weights.values())
    weighted_score = sum(component_scores[key] * float(weight)
                         for key, weight in weights.items()) / weight_total
    penalty_policy = match_policy.penalties
    penalties = {}
    if route_state == "TREND" or raw_candidate in {"TREND_UP", "TREND_DOWN"}:
        penalties["trend_tendency"] = float(penalty_policy.trend_tendency)
    if regime_name in {"IMPULSE_UP", "IMPULSE_DOWN"}:
        penalties["impulse"] = float(penalty_policy.impulse)
    if str(metrics["structure"]) == "CONFLICT":
        penalties["structure_conflict"] = float(penalty_policy.structure_conflict)
    if regime_name in {"LOW", "HIGH"}:
        penalties["low_high_edge"] = float(penalty_policy.low_high_edge)
    score = round(min(100.0, max(0.0, weighted_score - sum(penalties.values()))), 2)

    labels = {
        "route": ("range_route_confirmed" if route_state == "RANGE" else
                  "range_candidate_unconfirmed" if raw_candidate == "RANGE" else
                  "trend_route" if route_state == "TREND" else "route_uncertain"),
        "adx": "adx_in_range_band" if component_scores["adx"] >= 80 else "adx_above_range_band",
        "ema50_slope": "ema50_slope_flat" if component_scores["ema50_slope"] >= 80 else "ema50_slope_elevated",
        "ema50_distance": "close_near_ema50" if component_scores["ema50_distance"] >= 80 else "close_far_from_ema50",
        "structure": ("structure_neutral" if metrics["structure"] == "NEUTRAL" else
                       "structure_conflict" if metrics["structure"] == "CONFLICT" else
                       "structure_directional"),
        "legacy_regime": {
            "SIDEWAYS": "legacy_sideways",
            "MID": "legacy_mid_range" if legacy_score >= 60 else "legacy_off_center",
            "LOW": "legacy_low_zone",
            "HIGH": "legacy_high_zone",
            "SLOW_UP": "legacy_slow_up",
            "SLOW_DOWN": "legacy_slow_down",
            "IMPULSE_UP": "legacy_impulse_up",
            "IMPULSE_DOWN": "legacy_impulse_down",
        }.get(regime_name, "legacy_regime_unavailable"),
    }
    ordered_components = sorted(
        component_scores,
        key=lambda key: component_scores[key] * float(weights[key]),
        reverse=True,
    )
    reasons = [labels[key] for key in ordered_components[:3]]
    reasons.extend(f"{key}_penalty" for key, value in penalties.items() if value > 0)
    return {
        "symbol": symbol,
        "score": score,
        "status": "scored",
        "unavailable_reason": None,
        "route_state": route_state,
        "route_reason": route.reason,
        "legacy_regime": regime_name,
        "last_closed_bar_at": last_closed,
        "metrics": {
            "adx": adx,
            "ema50_slope_pct": float(metrics["slow_ema_slope_pct"]),
            "close_to_ema50_pct": float(metrics["close_to_slow_ema_pct"]),
            "structure": str(metrics["structure"]),
            "raw_candidate": raw_candidate,
            "legacy_range_position": position,
        },
        "component_scores": {key: round(value, 2) for key, value in component_scores.items()},
        "penalties": penalties,
        "reasons": reasons,
    }
