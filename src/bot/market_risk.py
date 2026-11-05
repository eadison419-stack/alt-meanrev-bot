"""Observational broad-market risk status from completed BTC 4H candles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math

import ccxt
import pandas as pd

from bot.config import load_settings
from bot.data.klines import ohlcv_frame
from bot.exchange.network import configure_system_proxy
from bot.indicators import indicator_frame, validate_ohlcv

FOUR_HOURS = pd.Timedelta(4, unit="h")
STALE_GRACE = pd.Timedelta(5, unit="min")


class MarketRiskState(str, Enum):
    NORMAL = "NORMAL"
    RISK_ELEVATED = "RISK_ELEVATED"
    PAUSE_NEW = "PAUSE_NEW"


@dataclass(frozen=True)
class MarketRiskAssessment:
    state: MarketRiskState
    reason: str
    checked_at: datetime
    symbol: str
    timeframe: str
    evidence: dict[str, float | int | bool | str | None]

    def to_dict(self) -> dict:
        return {
            "mode": "entry_gate",
            "state": self.state.value,
            "reason": self.reason,
            "checked_at": self.checked_at.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "evidence": self.evidence,
        }


def classify_market_risk(bars: pd.DataFrame | None, settings, *, now: datetime,
                         fetch_error: str | None = None) -> MarketRiskAssessment:
    """Classify BTC context using completed 4H bars for the new-entry gate.

    The warning thresholds reuse the existing impulse and slow-move settings.
    A simultaneous large downside 4H body and confirmed 24H down move is the
    most severe observation. A single large move, in either direction, is
    elevated risk. Missing or stale data is reported as PAUSE_NEW for visibility.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("market risk evaluation time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    policy = settings.market_risk
    common = {
        "symbol": policy.symbol,
        "timeframe": "4h",
        "monitoring_enabled": policy.enabled,
    }

    def result(state: MarketRiskState, reason: str, evidence: dict | None = None):
        details = {**common, **(evidence or {})}
        if fetch_error:
            details["fetch_error"] = fetch_error
        return MarketRiskAssessment(state, reason, now, policy.symbol, "4h", details)

    if not policy.enabled:
        return result(MarketRiskState.NORMAL, "market_risk_monitor_disabled")
    if bars is None or bars.empty:
        return result(MarketRiskState.PAUSE_NEW, "btc_bars_missing")

    try:
        validate_ohlcv(bars)
        closed = bars.loc[bars.index + FOUR_HOURS <= pd.Timestamp(now)]
        required = settings.lookback.min_regime_bars_4h
        if len(closed) < required:
            return result(MarketRiskState.PAUSE_NEW, "insufficient_closed_bars", {
                "closed_bar_count": int(len(closed)),
                "required_closed_bar_count": int(required),
            })
        recent = closed.index[-required:]
        if len(recent) > 1 and not (recent[1:] - recent[:-1] == FOUR_HOURS).all():
            return result(MarketRiskState.PAUSE_NEW, "closed_bar_gap", {
                "closed_bar_count": int(len(closed)),
                "required_closed_bar_count": int(required),
            })

        close_at = closed.index[-1] + FOUR_HOURS
        if pd.Timestamp(now) - close_at > FOUR_HOURS + STALE_GRACE:
            return result(MarketRiskState.PAUSE_NEW, "stale_closed_bar", {
                "last_closed_bar_at": close_at.isoformat(),
                "closed_bar_age_seconds": float((pd.Timestamp(now) - close_at).total_seconds()),
            })

        features = indicator_frame(closed, settings.lookback.atr_len, settings.lookback.adx_len)
        atr_value = float(features["atr"].iloc[-1])
        adx_value = float(features["adx"].iloc[-1])
        if not math.isfinite(atr_value) or atr_value <= 0 or not math.isfinite(adx_value):
            return result(MarketRiskState.PAUSE_NEW, "indicators_unavailable", {
                "last_closed_bar_at": close_at.isoformat(),
                "atr": atr_value if math.isfinite(atr_value) else None,
                "adx": adx_value if math.isfinite(adx_value) else None,
            })

        last = closed.iloc[-1]
        body = float(last["close"]) - float(last["open"])
        atr = atr_value
        lookback = settings.regime.slow_lookback_4h
        net_move = float(closed["close"].iloc[-1]) - float(closed["close"].iloc[-1 - lookback])
        body_atr = body / atr
        net_move_atr = net_move / atr
        impulse_limit = float(settings.signals.impulse_atr_mult)
        adx_limit = float(settings.regime.adx_sideways)
        trend_limit = float(settings.regime.slow_move_atr_mult)
        volatility_spike = abs(body_atr) >= impulse_limit
        trend_extreme = adx_value >= adx_limit and abs(net_move_atr) >= trend_limit
        downside_impulse = body_atr <= -impulse_limit
        downside_trend = adx_value >= adx_limit and net_move_atr <= -trend_limit
        severe_downside = downside_impulse and downside_trend
        evidence = {
            "last_closed_bar_at": close_at.isoformat(),
            "closed_bar_count": int(len(closed)),
            "open": float(last["open"]),
            "close": float(last["close"]),
            "atr": atr_value,
            "adx": adx_value,
            "body_atr": body_atr,
            "net_move_24h": net_move,
            "net_move_24h_atr": net_move_atr,
            "impulse_threshold_atr": impulse_limit,
            "adx_threshold": adx_limit,
            "trend_threshold_atr": trend_limit,
            "volatility_spike": volatility_spike,
            "trend_extreme": trend_extreme,
            "downside_impulse": downside_impulse,
            "downside_trend": downside_trend,
            "severe_downside": severe_downside,
        }
        if severe_downside:
            return result(MarketRiskState.PAUSE_NEW, "btc_extreme_downside_confirmed", evidence)
        if volatility_spike or trend_extreme:
            return result(MarketRiskState.RISK_ELEVATED, "btc_volatility_or_trend_elevated", evidence)
        return result(MarketRiskState.NORMAL, "btc_conditions_normal", evidence)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return result(MarketRiskState.PAUSE_NEW, "invalid_bars_or_indicators", {
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        })


def fetch_market_risk_status(exchange, settings, *, now: datetime) -> MarketRiskAssessment:
    """Fetch BTC public candles and return the current new-entry gate state."""
    if not settings.market_risk.enabled:
        return classify_market_risk(None, settings, now=now)
    try:
        limit = max(100, settings.lookback.min_regime_bars_4h)
        rows = exchange.fetch_ohlcv(settings.market_risk.symbol, timeframe="4h", limit=limit)
        bars = ohlcv_frame(rows) if rows else pd.DataFrame()
        return classify_market_risk(bars, settings, now=now)
    except Exception as exc:
        return classify_market_risk(None, settings, now=now,
                                    fetch_error=f"{type(exc).__name__}: {str(exc)[:300]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the BTC market risk entry-gate state.")
    parser.parse_args(argv)
    settings = load_settings()
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    configure_system_proxy(exchange)
    try:
        assessment = fetch_market_risk_status(exchange, settings, now=datetime.now(timezone.utc))
    finally:
        exchange.close()
    print(json.dumps(assessment.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
