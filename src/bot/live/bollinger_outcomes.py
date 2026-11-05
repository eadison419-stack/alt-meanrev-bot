"""Isolated, no-order forward outcome tracking for Bollinger observer hits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd


RULES_VERSION = "bollinger_observer_v1"
HORIZONS_HOURS = (4, 12, 24, 48)
MAX_ENTRY_DELAY_SECONDS = 15 * 60
_STATE_VERSION = 1
_HOUR = timedelta(hours=1)
_FOUR_HOURS = timedelta(hours=4)


def _aware(value: str | datetime | pd.Timestamp) -> datetime:
    parsed = pd.Timestamp(value).to_pydatetime()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Bollinger outcome timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _price(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _bar_time(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").to_pydatetime()


def _direction(signal_type: str) -> int:
    if signal_type.startswith("long_"):
        return 1
    if signal_type.startswith("short_"):
        return -1
    raise ValueError(f"unsupported Bollinger signal type: {signal_type}")


def _sample_key(symbol: str, bar_at: str, signal_type: str) -> str:
    return f"{RULES_VERSION}|{symbol}|{bar_at}|{signal_type}"


def _ceil_hour(value: datetime) -> datetime:
    floored = value.replace(minute=0, second=0, microsecond=0)
    return floored if floored == value else floored + _HOUR


def entry_timing_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Classify how long after the closed 4H signal its first quote was captured."""
    available_at = _aware(sample["last_closed_bar_at"]) + _FOUR_HOURS
    entry_at = _aware(sample["entry_at"])
    delay = round((entry_at - available_at).total_seconds(), 3)
    if delay < 0:
        status, reason = "invalid_timing", "entry_before_signal_bar_close"
    elif delay <= MAX_ENTRY_DELAY_SECONDS:
        status, reason = "on_time", "captured_within_15m_of_signal_bar_close"
    else:
        status, reason = "delayed", "captured_more_than_15m_after_signal_bar_close"
    return {
        "signal_available_at": available_at.isoformat(),
        "entry_delay_seconds": delay,
        "entry_timing_status": status,
        "entry_timing_reason": reason,
        "entry_timing_threshold_seconds": MAX_ENTRY_DELAY_SECONDS,
    }


def _return_metrics(sample: dict[str, Any], bars: list[dict[str, Any]], end_bar: dict[str, Any],
                    horizon_hours: int) -> dict[str, Any]:
    entry = float(sample["entry_price"])
    side = sample["side"]
    direction = 1 if side == "long" else -1
    highs = [float(bar["high"]) for bar in bars]
    lows = [float(bar["low"]) for bar in bars]
    close = float(end_bar["close"])
    if side == "long":
        mfe_pct = (max(highs) / entry - 1) * 100
        mae_pct = (min(lows) / entry - 1) * 100
        slippage_rate = max(0.0, float(sample["cost_parameters"]["slippage_bps"])) / 10_000
        entry_fill = entry * (1 + slippage_rate)
        exit_fill = close * (1 - slippage_rate)
        gross_per_unit = exit_fill - entry_fill
    else:
        mfe_pct = (entry / min(lows) - 1) * 100
        mae_pct = (entry / max(highs) - 1) * 100
        slippage_rate = max(0.0, float(sample["cost_parameters"]["slippage_bps"])) / 10_000
        entry_fill = entry * (1 - slippage_rate)
        exit_fill = close * (1 + slippage_rate)
        gross_per_unit = entry_fill - exit_fill

    gross_return_pct = direction * (close - entry) / entry * 100
    fee_rate = max(0.0, float(sample["cost_parameters"]["taker_bps"])) / 10_000
    estimated_fees = (entry_fill + exit_fill) * fee_rate
    estimated_net_pct = (gross_per_unit - estimated_fees) / entry * 100
    first_open = _aware(sample["excursion_coverage_starts_at"])
    last_open = _aware(bars[-1]["open_at"])
    expected_bars = int((last_open - first_open).total_seconds() // 3600) + 1
    observed_opens = [_aware(bar["open_at"]) for bar in bars]
    coverage_complete = (
        bool(observed_opens) and observed_opens[0] == first_open and len(bars) == expected_bars
        and all((right - left) == _HOUR for left, right in zip(observed_opens, observed_opens[1:]))
    )
    observed_at = _aware(end_bar["close_at"])
    entry_at = _aware(sample["entry_at"])
    return {
        "target_horizon_hours": horizon_hours,
        "target_at": (entry_at + timedelta(hours=horizon_hours)).isoformat(),
        "observed_at": observed_at.isoformat(),
        "actual_horizon_hours": round((observed_at - entry_at).total_seconds() / 3600, 4),
        "exit_close": close,
        "gross_return_pct": round(gross_return_pct, 6),
        "estimated_net_return_pct": round(estimated_net_pct, 6),
        "estimated_cost_impact_pct": round(gross_return_pct - estimated_net_pct, 6),
        "return_basis": "unleveraged price return per entry notional",
        "cost_parameters": sample["cost_parameters"],
        "cost_parameters_source": sample.get("cost_parameters_source", "unknown"),
        "mfe_pct": round(mfe_pct, 6),
        "mae_pct": round(mae_pct, 6),
        "bars_observed": len(bars),
        "coverage_complete": coverage_complete,
        "excursion_coverage_starts_at": sample["excursion_coverage_starts_at"],
        "funding_included": False,
    }


class BollingerOutcomeTracker:
    """Persist hypothetical forward outcomes without touching trading state."""

    def __init__(self, path: str | Path, settings):
        self.path = Path(path)
        self.settings = settings
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict)
                    or data.get("version") != _STATE_VERSION
                    or data.get("rules_version") != RULES_VERSION
                    or not isinstance(data.get("samples"), dict)):
                raise ValueError("unsupported Bollinger outcome state version")
            self.state = data
            self._backfill_sample_parameters()
        else:
            self.state = {
                "version": _STATE_VERSION,
                "rules_version": RULES_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entry_reference": "ticker_last_at_first_tracker_observation",
                "entry_timing_policy": "on_time if entry quote is within 15 minutes of closed 4H signal availability",
                "horizons_hours": list(HORIZONS_HOURS),
                "bar_timeframe": "1h_closed_bars",
                "cost_model": "configured taker fee and slippage on entry and exit; funding excluded",
                "samples": {},
            }
        self.state["entry_reference"] = "ticker_last_at_first_tracker_observation"
        self.state.setdefault(
            "entry_timing_policy", "on_time if entry quote is within 15 minutes of closed 4H signal availability",
        )
        self.state.setdefault("horizons_hours", list(HORIZONS_HOURS))
        self.state.setdefault("bar_timeframe", "1h_closed_bars")
        self.state.setdefault(
            "cost_model", "configured taker fee and slippage on entry and exit; funding excluded",
        )
        self.state["last_loaded_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    @property
    def active_symbols(self) -> set[str]:
        return {
            sample["symbol"]
            for sample in self.state["samples"].values()
            if len(sample.get("outcomes", {})) < len(HORIZONS_HOURS)
        }

    def process_refresh(self, candidates: list[dict[str, Any]], bars_by_symbol: dict[str, pd.DataFrame | None],
                        *, now: datetime) -> dict[str, Any]:
        """Register new hits, advance completed-bar outcomes, then persist once if changed."""
        now = _aware(now)
        changed = False
        added_samples = 0
        updated_outcomes = 0
        invalid_candidates = 0

        for candidate in candidates:
            symbol = candidate.get("symbol")
            bar_at = candidate.get("last_closed_bar_at")
            entry_price = _price(candidate.get("current_price"))
            if not symbol or not bar_at or entry_price is None:
                invalid_candidates += 1
                continue
            try:
                _aware(bar_at)
                signal_close = _price(candidate.get("signal_close"))
            except (TypeError, ValueError, OverflowError):
                invalid_candidates += 1
                continue
            for signal_type in candidate.get("signal_types", []):
                try:
                    side = "long" if _direction(signal_type) > 0 else "short"
                except (TypeError, ValueError):
                    invalid_candidates += 1
                    continue
                key = _sample_key(str(symbol), str(bar_at), signal_type)
                if key in self.state["samples"]:
                    continue
                self.state["samples"][key] = {
                    "sample_id": key,
                    "rules_version": RULES_VERSION,
                    "rule_parameters": self.settings.bollinger_observation.model_dump(exclude={"enabled"}),
                    "rule_parameters_source": "recorded_at_signal",
                    "symbol": str(symbol),
                    "signal_type": signal_type,
                    "side": side,
                    "last_closed_bar_at": str(bar_at),
                    "signal_close": signal_close,
                    "entry_at": now.isoformat(),
                    **entry_timing_for_sample({"last_closed_bar_at": str(bar_at), "entry_at": now}),
                    "entry_timing_source": "recorded_at_signal",
                    "entry_price": entry_price,
                    "entry_reference": "first_tracker_observation_ticker_last",
                    "cost_parameters": {
                        "taker_bps": max(0.0, float(self.settings.fees.taker_bps)),
                        "slippage_bps": max(0.0, float(self.settings.fees.slippage_bps)),
                    },
                    "cost_parameters_source": "recorded_at_signal",
                    "excursion_coverage_starts_at": _ceil_hour(now).isoformat(),
                    "bars": {},
                    "outcomes": {},
                }
                added_samples += 1
                changed = True

        for sample in self.state["samples"].values():
            if len(sample.get("outcomes", {})) >= len(HORIZONS_HOURS):
                continue
            frame = bars_by_symbol.get(sample["symbol"])
            if frame is None or frame.empty or not {"high", "low", "close"}.issubset(frame.columns):
                continue
            first_full_bar = _aware(sample["excursion_coverage_starts_at"])
            entry_at = _aware(sample["entry_at"])
            latest_needed_bar_close = entry_at + timedelta(hours=max(HORIZONS_HOURS) + 1)
            for index, row in frame.iterrows():
                try:
                    open_at = _bar_time(index)
                    close_at = open_at + _HOUR
                    high, low, close = (_price(row.get(name)) for name in ("high", "low", "close"))
                except (TypeError, ValueError, OverflowError):
                    continue
                if (open_at < first_full_bar or close_at > now or close_at > latest_needed_bar_close
                        or high is None or low is None or close is None or low > high):
                    continue
                bar_key = open_at.isoformat()
                if bar_key not in sample["bars"]:
                    sample["bars"][bar_key] = {
                        "open_at": bar_key,
                        "close_at": close_at.isoformat(),
                        "high": high,
                        "low": low,
                        "close": close,
                    }
                    changed = True

            bars = sorted(sample["bars"].values(), key=lambda bar: bar["open_at"])
            for horizon in HORIZONS_HOURS:
                horizon_key = f"{horizon}h"
                if horizon_key in sample["outcomes"]:
                    continue
                target_at = entry_at + timedelta(hours=horizon)
                end_bar = next((bar for bar in bars if _aware(bar["close_at"]) >= target_at), None)
                if end_bar is None:
                    continue
                eligible_bars = [bar for bar in bars
                                 if _aware(bar["close_at"]) <= _aware(end_bar["close_at"])]
                sample["outcomes"][horizon_key] = _return_metrics(
                    sample, eligible_bars, end_bar, horizon,
                )
                updated_outcomes += 1
                changed = True

        if changed:
            self.state["last_updated_at"] = now.isoformat()
            self.save()
        return {
            "enabled": True,
            "rules_version": RULES_VERSION,
            "entry_reference": self.state["entry_reference"],
            "entry_timing_threshold_seconds": MAX_ENTRY_DELAY_SECONDS,
            "horizons_hours": list(HORIZONS_HOURS),
            "new_samples": added_samples,
            "newly_completed_horizons": updated_outcomes,
            "invalid_candidates": invalid_candidates,
            "summary": self.summarize(),
        }

    def summarize(self) -> dict[str, Any]:
        return summarize_outcome_state(self.state)

    def _backfill_sample_parameters(self) -> None:
        current_costs = {
            "taker_bps": max(0.0, float(self.settings.fees.taker_bps)),
            "slippage_bps": max(0.0, float(self.settings.fees.slippage_bps)),
        }
        current_rules = self.settings.bollinger_observation.model_dump(exclude={"enabled"})
        for sample in self.state["samples"].values():
            if not isinstance(sample, dict):
                raise ValueError("invalid Bollinger outcome sample")
            if not isinstance(sample.get("cost_parameters"), dict):
                sample["cost_parameters"] = dict(current_costs)
                sample["cost_parameters_source"] = "backfilled_from_config_at_load"
            if not isinstance(sample.get("rule_parameters"), dict):
                sample["rule_parameters"] = dict(current_rules)
                sample["rule_parameters_source"] = "backfilled_from_config_at_load"
            if sample.get("entry_timing_status") is None:
                sample.update(entry_timing_for_sample(sample))
                sample["entry_timing_source"] = "backfilled_from_signal_timestamp"
            for outcome in sample.get("outcomes", {}).values():
                if not isinstance(outcome, dict):
                    raise ValueError("invalid Bollinger outcome result")
                outcome.setdefault("cost_parameters", dict(sample["cost_parameters"]))
                outcome.setdefault("cost_parameters_source", sample["cost_parameters_source"])
                outcome.setdefault("return_basis", "unleveraged price return per entry notional")
                outcome.setdefault("funding_included", False)
                if ("gross_return_pct" in outcome and "estimated_net_return_pct" in outcome):
                    outcome.setdefault(
                        "estimated_cost_impact_pct",
                        round(float(outcome["gross_return_pct"])
                              - float(outcome["estimated_net_return_pct"]), 6),
                    )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, self.path)


def summarize_outcome_state(data: dict[str, Any]) -> dict[str, Any]:
    samples = list(data.get("samples", {}).values())
    timing = {}
    for sample in samples:
        try:
            timing[sample["sample_id"]] = sample.get("entry_timing_status") or entry_timing_for_sample(sample)[
                "entry_timing_status"
            ]
        except (KeyError, TypeError, ValueError, OverflowError):
            timing[sample.get("sample_id", str(id(sample)))] = "invalid_timing"
    cost_parameter_sets = sorted({
        (float(sample["cost_parameters"]["taker_bps"]),
         float(sample["cost_parameters"]["slippage_bps"]))
        for sample in samples if isinstance(sample.get("cost_parameters"), dict)
    })
    by_signal_type = {}
    for signal_type in ("long_start", "short_start", "long_trend", "short_trend"):
        matching = [sample for sample in samples if sample.get("signal_type") == signal_type]
        timely = [sample for sample in matching if timing.get(sample.get("sample_id")) == "on_time"]
        delayed = [sample for sample in matching if timing.get(sample.get("sample_id")) == "delayed"]
        invalid_timing = [sample for sample in matching
                          if timing.get(sample.get("sample_id")) == "invalid_timing"]
        horizons = {}
        for horizon in HORIZONS_HOURS:
            outcomes = [sample["outcomes"][f"{horizon}h"] for sample in timely
                        if f"{horizon}h" in sample.get("outcomes", {})]
            delayed_outcomes = [sample["outcomes"][f"{horizon}h"] for sample in delayed
                                if f"{horizon}h" in sample.get("outcomes", {})]
            net_values = [float(outcome["estimated_net_return_pct"]) for outcome in outcomes]
            gross_values = [float(outcome["gross_return_pct"]) for outcome in outcomes]
            full_coverage = [outcome for outcome in outcomes if outcome.get("coverage_complete")]
            horizons[f"{horizon}h"] = {
                "completed_count": len(outcomes),
                "pending_count": len(timely) - len(outcomes),
                "delayed_completed_count": len(delayed_outcomes),
                "delayed_pending_count": len(delayed) - len(delayed_outcomes),
                "coverage_complete_count": len(full_coverage),
                "mean_gross_return_pct": _mean(gross_values),
                "mean_estimated_net_return_pct": _mean(net_values),
                "estimated_net_positive_rate": (
                    sum(value > 0 for value in net_values) / len(net_values) if net_values else None
                ),
                "mean_mfe_pct_full_coverage_only": _mean(
                    [float(outcome["mfe_pct"]) for outcome in full_coverage]
                ),
                "mean_mae_pct_full_coverage_only": _mean(
                    [float(outcome["mae_pct"]) for outcome in full_coverage]
                ),
            }
        by_signal_type[signal_type] = {
            "sample_count": len(matching),
            "eligible_sample_count": len(timely),
            "delayed_sample_count": len(delayed),
            "invalid_timing_sample_count": len(invalid_timing),
            "pending_sample_count": sum(
                len(sample.get("outcomes", {})) < len(HORIZONS_HOURS) for sample in timely
            ),
            "delayed_pending_sample_count": sum(
                len(sample.get("outcomes", {})) < len(HORIZONS_HOURS) for sample in delayed
            ),
            "horizons": horizons,
        }
    all_timing = list(timing.values())
    return {
        "rules_version": data.get("rules_version"),
        "sample_count": len(samples),
        "entry_reference": "ticker_last_at_first_tracker_observation",
        "entry_timing_policy": data.get(
            "entry_timing_policy",
            "on_time if entry quote is within 15 minutes of closed 4H signal availability",
        ),
        "entry_timing_threshold_seconds": MAX_ENTRY_DELAY_SECONDS,
        "eligible_sample_count": all_timing.count("on_time"),
        "delayed_sample_count": all_timing.count("delayed"),
        "invalid_timing_sample_count": all_timing.count("invalid_timing"),
        "pending_sample_count": sum(len(sample.get("outcomes", {})) < len(HORIZONS_HOURS)
                                     for sample in samples
                                     if timing.get(sample.get("sample_id")) == "on_time"),
        "delayed_pending_sample_count": sum(len(sample.get("outcomes", {})) < len(HORIZONS_HOURS)
                                            for sample in samples
                                            if timing.get(sample.get("sample_id")) == "delayed"),
        "cost_parameter_sets_bps": [
            {"taker_bps": taker, "slippage_bps": slippage}
            for taker, slippage in cost_parameter_sets
        ],
        "by_signal_type": by_signal_type,
        "cost_model": data.get("cost_model"),
        "funding_included": False,
        "interpretation": "hypothetical forward outcomes, not executed trades or realized PnL",
    }


def _mean(values: list[float]) -> float | None:
    return round(math.fsum(values) / len(values), 6) if values else None

