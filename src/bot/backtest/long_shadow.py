"""Isolated next-closed-1H confirmation analysis for portfolio replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bot.backtest.engine import _apply_slippage, _snapshot
from bot.indicators import atr
from bot.regime import Regime
from bot.risk import calculate_adaptive_stop, calculate_order_quantity
from bot.signals import Action, Side, generate_signal, with_signal_context
from bot.time_filter import evaluate_time_filter
from bot.portfolio import PortfolioState

HOUR = pd.Timedelta(1, unit="h")


@dataclass(frozen=True)
class ShadowLongTrade:
    symbol: str
    signal_time: pd.Timestamp
    confirm_time: pd.Timestamp
    signal_open: float
    signal_low: float
    signal_close: float
    confirm_open: float
    confirm_low: float
    confirm_close: float
    confirm_pass: bool
    fail_reason: str | None
    missed_long_net_pnl: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str | None = None
    net_pnl: float | None = None
    r_multiple: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    held_hours: float | None = None
    missed_max_up_r_6h: float | None = None
    missed_max_up_price: float | None = None
    opportunity_cost_status: str | None = None


def _risk_distance(signal: dict, entry: float, settings) -> float | None:
    try:
        stop, _ = calculate_adaptive_stop(
            Side.LONG, entry, float(signal["atr"]), signal.get("structure_low"), settings,
        )
    except ValueError as error:
        if str(error) == "stop_too_wide":
            return None
        raise
    if stop is None or not np.isfinite(stop) or stop >= entry:
        return None
    return entry - float(stop)


def _simulate_one(signal: dict, confirm_time: pd.Timestamp, entry_raw: float,
                  bars: pd.DataFrame, bars_4h: pd.DataFrame, regimes: pd.DataFrame, funding: pd.DataFrame,
                  settings, initial_equity: float, exit_policy: str) -> dict:
    """One independent B position. It has no shared cash, slots, or signal queue."""
    fee = settings.fees.taker_bps / 10_000
    slip = settings.fees.slippage_bps / 10_000
    entry, entry_slip = _apply_slippage(entry_raw, Side.LONG, is_entry=True, slippage_rate=slip)
    risk = _risk_distance(signal, entry, settings)
    if risk is None:
        return {"fail_reason": "shadow_invalid_stop"}
    stop, stop_mult = calculate_adaptive_stop(
        Side.LONG, entry, float(signal["atr"]), signal.get("structure_low"), settings,
    )
    size_mult = float(signal.get("size_multiplier", 1.0)) * stop_mult
    quantity = calculate_order_quantity(initial_equity, entry, PortfolioState(), settings,
                                        step_size=1e-8, minimum_quantity=0.0,
                                        size_multiplier=size_mult)
    if quantity <= 0:
        return {"fail_reason": "shadow_size_zero"}
    entry_fee = quantity * entry * fee
    funding_paid = 0.0
    best = entry
    worst = entry
    protective = float(stop)
    no_progress = 0
    previous_close = entry
    held = bars.loc[bars.index >= confirm_time]
    if held.empty:
        return {"fail_reason": "shadow_no_post_confirm_bars"}
    last_time = confirm_time
    last_price = entry_raw
    exit_reason = "backtest_end"
    exit_raw = entry_raw
    pending_exit: str | None = None
    atr_1h = atr(bars, settings.lookback.atr_len)
    regime_closes = regimes.index + pd.Timedelta(hours=4)
    for open_time, candle in held.iterrows():
        close_time = open_time + HOUR
        high, low = float(candle["high"]), float(candle["low"])
        last_time, last_price = close_time, float(candle["close"])
        settled = funding.loc[(funding.index > open_time) & (funding.index <= close_time)] if not funding.empty else funding
        for _, row in settled.iterrows():
            mark = row.get("mark_price")
            if pd.isna(mark) or float(mark) <= 0:
                mark = float(candle["close"])
            funding_paid += quantity * float(mark) * float(row["funding_rate"])
        if pending_exit is not None:
            exit_raw, exit_reason = float(candle["open"]), pending_exit
            last_time = open_time
            break
        stops = ((low <= protective), float(candle["open"]) <= protective)
        if stops[0]:
            exit_raw = float(candle["open"]) if stops[1] else protective
            exit_reason = "stop_gap" if stops[1] else "stop_hit"
            break
        best, worst = max(best, high), min(worst, low)
        close = float(candle["close"])
        no_progress = 0 if close > previous_close else no_progress + 1
        previous_close = close
        reason = None
        if (settings.exits.structure_invalid_enabled and signal.get("structure_low") is not None
                and close < float(signal["structure_low"]) - settings.exits.structure_invalid_buffer_atr * float(signal["atr"])):
            reason = "long_structure_invalid"
        regime_rows = regimes.loc[regimes.index + pd.Timedelta(hours=4) <= close_time]
        if reason is None and settings.exits.regime_fade_enabled and not regime_rows.empty:
            if (signal["regime"] == Regime.LOW.value
                    and regime_rows.iloc[-1]["regime"] is Regime.SIDEWAYS
                    and close - entry < settings.exits.regime_fade_min_atr * float(signal["atr"])):
                reason = "long_regime_faded"
        if (reason is None and no_progress >= settings.exits.no_progress_bars
                and best - entry < settings.exits.no_progress_min_atr * float(signal["atr"])):
            reason = "no_progress"
        held_hours = (close_time - confirm_time).total_seconds() / 3600
        idx = bars.index.searchsorted(open_time)
        regime_idx = int(np.searchsorted(regime_closes.asi8, close_time.value, side="right") - 1)
        if reason is None and regime_idx >= 0 and idx < len(bars):
            snap = _snapshot(regimes.iloc[regime_idx], regimes.index[regime_idx])
            atr_value = atr_1h.iloc[idx]
            if snap is not None and np.isfinite(atr_value) and atr_value > 0:
                history = bars.iloc[max(0, idx - 47):idx + 1]
                flags = evaluate_time_filter(close_time.to_pydatetime(), history, settings,
                                             precomputed_atr=atr_1h.reindex(history.index))
                swing = bars.iloc[max(0, idx - settings.stops.swing_lookback_1h + 1):idx + 1]
                regime_row = regimes.iloc[regime_idx]
                sig = generate_signal(
                    snap, candle, flags, settings, position=Side.LONG,
                    bars_1h=bars.iloc[:idx + 1],
                    bars_4h=bars_4h.loc[bars_4h.index <= regimes.index[regime_idx]],
                    entry_reference_price=close, entry_atr=float(atr_value),
                    range_low=regime_row.get("range_low"), range_high=regime_row.get("range_high"),
                    structure_low=float(swing["low"].min()), structure_high=float(swing["high"].max()),
                )
                sig = with_signal_context(sig, symbol=symbol, source="MEAN_REVERSION",
                                          timestamp=close_time.to_pydatetime())
                if sig.action is Action.CLOSE:
                    reason = sig.reason
        if reason is None and held_hours >= settings.exits.max_hold_bars:
            reason = "max_hold_24h"
        if reason:
            pending_exit = reason
            continue
        if exit_policy == "time_managed":
            progress_r = max(0.0, best - entry) / risk
            close_fill, _ = _apply_slippage(close, Side.LONG, is_entry=False, slippage_rate=slip)
            net_if_closed = quantity * (close_fill - entry) - entry_fee - quantity * close_fill * fee - funding_paid
            if (held_hours >= settings.exit_management.time_check_hours
                    and progress_r < settings.exit_management.time_min_progress_r and net_if_closed <= 0):
                pending_exit = "time_stop_12h_no_progress"
                continue
            if (held_hours >= settings.exit_management.time_exit_hours
                    and progress_r < settings.exit_management.breakeven_trigger_r):
                pending_exit = "time_exit_24h_unprotected"
                continue
        favorable = best - entry
        if favorable >= settings.stops.move_to_breakeven_atr * float(signal["atr"]):
            protective = max(protective, entry * (1 + fee) / ((1 - slip) * (1 - fee)))
        progress_r = favorable / risk
        if exit_policy in ("breakeven_trail", "time_managed") and progress_r >= settings.exit_management.trailing_trigger_r:
            trail = best - settings.exit_management.trailing_atr_mult * risk / settings.signals.stop_atr_mult
            protective = max(protective, trail)
    exit, exit_slip = _apply_slippage(exit_raw, Side.LONG, is_entry=False, slippage_rate=slip)
    gross = quantity * (exit - entry)
    net = gross - entry_fee - quantity * exit * fee - funding_paid
    # Fee and slippage are included in final R; stop-distance risk is the denominator.
    risk_usdt = quantity * risk
    return {
        "entry_price": entry, "exit_price": exit, "exit_time": last_time,
        "exit_reason": exit_reason, "net_pnl": net,
        "r_multiple": net / risk_usdt if risk_usdt else None,
        "mfe_r": (best - entry) / risk,
        "mae_r": (entry - worst) / risk,
        "held_hours": (last_time - confirm_time).total_seconds() / 3600,
        "fees": entry_fee + quantity * exit * fee,
        "funding_paid": funding_paid, "slippage_paid": quantity * (entry_slip + exit_slip),
    }


def analyze_long_shadows(signals: tuple[dict, ...], bars_1h: dict[str, pd.DataFrame],
                         bars_4h: dict[str, pd.DataFrame],
                         regimes: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame],
                         settings, *, initial_equity: float, exit_policy: str) -> list[dict]:
    results = []
    for signal in signals:
        frame = bars_1h[signal["symbol"]]
        confirm_start = pd.Timestamp(signal["signal_time"])
        idx = frame.index.get_indexer([confirm_start])[0]
        if idx < 0:
            results.append(asdict(ShadowLongTrade(
                signal["symbol"], confirm_start, confirm_start, signal["signal_open"], signal["signal_low"],
                signal["signal_close"], np.nan, np.nan, np.nan, False, "confirm_bar_missing",
            )))
            continue
        confirm = frame.iloc[idx]
        confirm_time = confirm_start + HOUR
        passed = float(confirm["low"]) >= float(signal["signal_low"]) and (
            float(confirm["close"]) >= float(signal["signal_close"])
            or float(confirm["close"]) > float(confirm["open"])
        )
        fail_reason = None
        if float(confirm["low"]) < float(signal["signal_low"]):
            fail_reason = "new_low"
        elif not (float(confirm["close"]) >= float(signal["signal_close"])
                  or float(confirm["close"]) > float(confirm["open"])):
            fail_reason = "close_weaker"
        row = ShadowLongTrade(
            signal["symbol"], confirm_start, confirm_time, float(signal["signal_open"]),
            float(signal["signal_low"]), float(signal["signal_close"]), float(confirm["open"]),
            float(confirm["low"]), float(confirm["close"]), passed, fail_reason,
        )
        data = asdict(row)
        if passed:
            sim = _simulate_one(signal, confirm_time, float(confirm["close"]), frame,
                                bars_4h[signal["symbol"]],
                                regimes[signal["symbol"]], funding.get(signal["symbol"], pd.DataFrame()),
                                settings, initial_equity, exit_policy)
            data.update(sim)
        else:
            risk = _risk_distance(signal, float(signal["signal_close"]), settings)
            subsequent = frame.iloc[idx + 1:idx + 7]
            if len(subsequent):
                rise = float(max(0.0, subsequent["high"].max() - float(signal["signal_close"])))
                data["missed_max_up_price"] = rise
                data["missed_max_up_r_6h"] = rise / risk if risk else None
                data["opportunity_cost_status"] = "complete_6_bars" if len(subsequent) == 6 else "truncated_at_window_end"
                if risk is None:
                    data["opportunity_cost_status"] += ";initial_stop_risk_unavailable"
            else:
                data["opportunity_cost_status"] = "no_forward_bars"
        results.append(data)
    return results
