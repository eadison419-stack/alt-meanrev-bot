"""Causal next-open OHLCV backtest engine shared with live strategy and risk rules."""

from dataclasses import dataclass
import json
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from bot.indicators import atr, validate_ohlcv
from bot.portfolio import PortfolioState, Position
from bot.regime import Regime, RegimeSnapshot, classify_regimes
from bot.risk import (
    OrderIntent,
    calculate_adaptive_stop,
    calculate_order_quantity,
    guard_signal,
)
from bot.signals import (Action, Side, Signal, entry_rejection_reason, generate_signal,
                         with_signal_context)
from bot.strategy_router import MEAN_REVERSION, TREND
from bot.time_filter import TimeFilterFlags, evaluate_time_filter


BacktestMode = Literal["long", "short", "both"]
ExitPolicy = Literal["fixed", "breakeven_trail", "time_managed"]
HOUR = pd.Timedelta(1, unit="h")
HOUR_NS = 3_600_000_000_000
FOUR_HOURS_NS = 4 * HOUR_NS
FUNDING_BPS_TIME_TOLERANCE_MS = 60_000


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: Side
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    fees: float
    slippage_paid: float
    funding_paid: float
    entry_reason: str
    exit_reason: str
    strategy_source: str = "MEAN_REVERSION"

    @property
    def source(self) -> str:
        """Strategy source, with ``strategy_source`` retained for compatibility."""
        return self.strategy_source


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    mode: BacktestMode
    initial_equity: float
    trades: tuple[Trade, ...]
    equity_curve: pd.Series
    metrics: dict[str, float | int | None]
    skipped_entries: dict[str, int]


@dataclass
class _OpenTrade:
    position: Position
    entry_fee: float
    entry_slippage: float
    funding_paid: float
    protective_stop: float
    initial_risk: float
    best_price: float
    entry_atr: float
    structure_low: float | None
    structure_high: float | None
    range_low: float | None
    range_high: float | None
    last_close: float
    no_progress_count: int = 0


def _validate_bars(frame: pd.DataFrame, name: str, step_ns: int) -> None:
    validate_ohlcv(frame)
    if len(frame) < 2:
        raise ValueError(f"{name} data must contain at least two bars")
    differences = np.diff(frame.index.asi8)
    if not np.all(differences == step_ns):
        raise ValueError(f"{name} timestamps must be continuous at the expected interval")
    if "volume" in frame and (not np.isfinite(frame["volume"].to_numpy(dtype=float)).all() or (frame["volume"] < 0).any()):
        raise ValueError(f"{name} volume must be finite and non-negative")


def _validate_funding(funding: pd.DataFrame) -> None:
    if not isinstance(funding.index, pd.DatetimeIndex) or funding.index.tz is None:
        raise ValueError("funding timestamps must be timezone-aware")
    missing = {"funding_rate", "mark_price"}.difference(funding.columns)
    if missing:
        raise ValueError(f"funding data is missing columns: {sorted(missing)}")
    if not funding.index.is_monotonic_increasing or funding.index.has_duplicates:
        raise ValueError("funding timestamps must be strictly increasing and unique")
    rates = funding["funding_rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("funding rates must be finite")
    marks = funding["mark_price"].dropna().to_numpy(dtype=float)
    if len(marks) and (not np.isfinite(marks).all() or (marks <= 0).any()):
        raise ValueError("funding mark prices must be finite and positive")


def _snapshot(row: pd.Series, timestamp: pd.Timestamp) -> RegimeSnapshot | None:
    if row.get("regime") is None or pd.isna(row.get("regime")):
        return None
    required = ("atr", "adx", "atr_percentile_rank", "range_position")
    if any(pd.isna(row.get(key)) for key in required):
        return None
    return RegimeSnapshot(
        timestamp=timestamp,
        regime=Regime(row["regime"]),
        atr=float(row["atr"]),
        adx=float(row["adx"]),
        atr_percentile_rank=float(row["atr_percentile_rank"]),
        range_position=float(row["range_position"]),
        new_low=bool(row["new_low"]),
        new_high=bool(row["new_high"]),
    )


def _apply_slippage(price: float, side: Side, *, is_entry: bool, slippage_rate: float) -> tuple[float, float]:
    # A long entry and short exit buy; a short entry and long exit sell.
    buys = (side is Side.LONG and is_entry) or (side is Side.SHORT and not is_entry)
    fill = price * (1 + slippage_rate) if buys else price * (1 - slippage_rate)
    return fill, abs(fill - price)


def _close_trade(
    symbol: str,
    active: _OpenTrade,
    raw_exit_price: float,
    closed_at: pd.Timestamp,
    exit_reason: str,
    *,
    fee_rate: float,
    slippage_rate: float,
    cash: float,
) -> tuple[Trade, float]:
    side = active.position.side
    fill_price, exit_slip_per_unit = _apply_slippage(raw_exit_price, side, is_entry=False, slippage_rate=slippage_rate)
    gross = active.position.quantity * (fill_price - active.position.entry_price) * (1 if side is Side.LONG else -1)
    exit_fee = active.position.quantity * fill_price * fee_rate
    cash += gross - exit_fee
    total_fees = active.entry_fee + exit_fee
    total_slippage = active.entry_slippage + active.position.quantity * exit_slip_per_unit
    trade = Trade(
        symbol=symbol,
        side=side,
        opened_at=pd.Timestamp(active.position.opened_at),
        closed_at=closed_at,
        entry_price=active.position.entry_price,
        exit_price=fill_price,
        quantity=active.position.quantity,
        gross_pnl=gross,
        net_pnl=gross - total_fees - active.funding_paid,
        fees=total_fees,
        slippage_paid=total_slippage,
        funding_paid=active.funding_paid,
        entry_reason=active.position.reason,
        exit_reason=exit_reason,
        strategy_source=active.position.source,
    )
    return trade, cash


def _funding_mark_price(
    funding_row: pd.Series,
    funding_time: pd.Timestamp,
    bars: pd.DataFrame,
) -> float:
    mark = funding_row["mark_price"]
    if pd.notna(mark) and float(mark) > 0:
        return float(mark)
    position = bars.index.searchsorted(funding_time, side="left")
    candidates = [index for index in (position - 1, position) if 0 <= index < len(bars)]
    if not candidates:
        raise ValueError("cannot estimate mark price for a funding settlement")
    nearest = min(candidates, key=lambda index: abs(bars.index[index].value - funding_time.value))
    if abs(bars.index[nearest].value - funding_time.value) > FUNDING_BPS_TIME_TOLERANCE_MS * 1_000_000:
        raise ValueError("missing funding mark price and no nearby 1H open is available")
    return float(bars["open"].iloc[nearest])


def run_backtest(
    symbol: str,
    ohlcv_1h: pd.DataFrame,
    ohlcv_4h: pd.DataFrame,
    funding: pd.DataFrame,
    settings,
    *,
    mode: BacktestMode = "both",
    initial_equity: float = 10_000.0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    time_flags_by_close: dict[pd.Timestamp, TimeFilterFlags] | None = None,
    exit_policy: ExitPolicy = "fixed",
) -> BacktestResult:
    """Run one-symbol historical simulation with bar-close signals and next-open fills.

    Funding used by the entry risk filter is the latest settlement strictly before
    the signal close, so a final historical funding value is never anticipated.
    A protective stop is checked against each hourly bar's high/low. Gaps through
    an existing stop exit at that bar's open, then pay exit slippage and fees.
    """
    if mode not in ("long", "short", "both"):
        raise ValueError("mode must be long, short, or both")
    if exit_policy not in ("fixed", "breakeven_trail", "time_managed"):
        raise ValueError("exit_policy must be fixed or breakeven_trail")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial equity must be finite and positive")
    _validate_bars(ohlcv_1h, "1H", HOUR_NS)
    _validate_bars(ohlcv_4h, "4H", FOUR_HOURS_NS)
    _validate_funding(funding)

    start = pd.Timestamp(start) if start is not None else ohlcv_1h.index[0]
    end = pd.Timestamp(end) if end is not None else ohlcv_1h.index[-1] + HOUR
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("backtest start and end must be timezone-aware")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if end <= start:
        raise ValueError("backtest end must be after start")
    bars = ohlcv_1h.loc[ohlcv_1h.index + HOUR <= end]
    if bars.empty:
        raise ValueError("no complete 1H bars fall within the requested backtest end")
    regimes_4h = classify_regimes(ohlcv_4h, settings)
    regimes_4h_closes_ns = ohlcv_4h.index.asi8 + FOUR_HOURS_NS
    atr_1h = atr(bars, settings.lookback.atr_len)
    bar_close_times = bars.index + HOUR
    regimes_by_bar = np.searchsorted(regimes_4h_closes_ns, bar_close_times.asi8, side="right") - 1

    funding_times_ns = funding.index.asi8
    funding_position = 0
    cash = float(initial_equity)
    portfolio = PortfolioState()
    active: _OpenTrade | None = None
    pending: tuple[Signal, float | None, float, dict | None] | None = None
    trades: list[Trade] = []
    equity_values: list[float] = [float(initial_equity)]
    equity_times: list[pd.Timestamp] = [bars.index[0]]
    skipped_entries: dict[str, int] = {}
    last_exit_bar_index: int | None = None
    fee_rate = settings.fees.taker_bps / 10_000
    slippage_rate = settings.fees.slippage_bps / 10_000
    # Kline history does not contain historical lot filters; use fine-grained
    # continuous sizing and explicitly report this execution approximation.
    quantity_step = 1e-8

    def reject_entry(reason: str, signal: Signal, at: pd.Timestamp, bar_index: int,
                     *, position_source: str | None = None,
                     details: dict | None = None) -> None:
        skipped_entries[reason] = skipped_entries.get(reason, 0) + 1
        logger.opt(lazy=True).warning("Backtest entry rejected {}", lambda: json.dumps({
            "time": at.isoformat(),
            "bar_index": bar_index,
            "symbol": signal.symbol or symbol,
            "side": signal.direction,
            "source": signal.source,
            "signal_reason": signal.reason,
            "position_source": position_source,
            "reason": reason,
            "details": details or {},
        }, ensure_ascii=False, sort_keys=True))

    for index, (open_time, candle) in enumerate(bars.iterrows()):
        open_time = pd.Timestamp(open_time)
        close_time = open_time + HOUR
        raw_open = float(candle["open"])

        # Funding at an exact hour may be timestamped a few milliseconds after
        # the OHLCV open. Include it here and charge positions held before the event.
        if index > 0:
            prior_open_ns = bars.index[index - 1].value
            current_open_ns = open_time.value + FUNDING_BPS_TIME_TOLERANCE_MS * 1_000_000
            while funding_position < len(funding) and funding_times_ns[funding_position] <= current_open_ns:
                funding_time = funding.index[funding_position]
                if funding_times_ns[funding_position] > prior_open_ns and active is not None:
                    settlement = funding.iloc[funding_position]
                    mark_price = _funding_mark_price(settlement, funding_time, bars)
                    direction = 1 if active.position.side is Side.LONG else -1
                    amount = active.position.quantity * mark_price * float(settlement["funding_rate"]) * direction
                    cash -= amount
                    active.funding_paid += amount
                funding_position += 1

        if pending is not None:
            signal, funding_rate, signal_atr, entry_levels = pending
            if signal.action is Action.CLOSE:
                if active is not None:
                    trade, cash = _close_trade(
                        symbol, active, raw_open, open_time, signal.reason,
                        fee_rate=fee_rate, slippage_rate=slippage_rate, cash=cash,
                    )
                    trades.append(trade)
                    portfolio.register_close(symbol, open_time.to_pydatetime(), reason=trade.exit_reason, net_pnl=trade.net_pnl)
                    last_exit_bar_index = index
                    active = None
            elif signal.action is Action.OPEN:
                held = portfolio.positions.get(symbol)
                if held is not None and signal.source != held.source:
                    reject_entry("POSITION_OWNERSHIP_CONFLICT", signal, open_time, index,
                                 position_source=held.source,
                                 details={"signal_source": signal.source,
                                          "position_source": held.source})
                    pending = None
                    continue
                if active is not None and signal.close_first:
                    trade, cash = _close_trade(
                        symbol, active, raw_open, open_time, "reverse_close_first",
                        fee_rate=fee_rate, slippage_rate=slippage_rate, cash=cash,
                    )
                    trades.append(trade)
                    portfolio.register_close(symbol, open_time.to_pydatetime(), reason=trade.exit_reason, net_pnl=trade.net_pnl)
                    last_exit_bar_index = index
                    active = None
                if active is not None:
                    reject_entry("position_still_open", signal, open_time, index,
                                 position_source=active.position.source)
                elif (mode == "long" and signal.side is not Side.LONG) or (mode == "short" and signal.side is not Side.SHORT):
                    reject_entry("mode_filter", signal, open_time, index)
                else:
                    fill_price, entry_slip_per_unit = _apply_slippage(
                        raw_open, signal.side, is_entry=True, slippage_rate=slippage_rate
                    )
                    entry_rejection = entry_rejection_reason(
                        signal.side, fill_price, signal_atr,
                        (entry_levels or {}).get("range_low"), (entry_levels or {}).get("range_high"),
                        (entry_levels or {}).get("structure_low"), (entry_levels or {}).get("structure_high"), settings,
                        require_structure=True,
                    )
                    if entry_rejection is not None:
                        reject_entry(entry_rejection, signal, open_time, index,
                                     details={"fill_price": fill_price})
                        pending = None
                        continue
                    equity_at_open = cash
                    try:
                        regime_index = regimes_by_bar[index - 1] if index > 0 else -1
                        if regime_index < 0:
                            raise ValueError("no completed 4H regime at entry signal")
                        regime_row = regimes_4h.iloc[regime_index]
                        structure_price = ((entry_levels or {}).get("structure_low") if signal.side is Side.LONG
                                           else (entry_levels or {}).get("structure_high"))
                        stop_price, stop_multiplier = calculate_adaptive_stop(
                            signal.side, fill_price, signal_atr, structure_price, settings,
                        )
                        quantity = calculate_order_quantity(
                            equity_at_open, fill_price, portfolio, settings,
                            step_size=quantity_step, minimum_quantity=0.0,
                            size_multiplier=signal.size_multiplier * stop_multiplier,
                        )
                        if quantity <= 0:
                            raise ValueError("stop_too_wide" if stop_multiplier < 1 else "stop_distance_below_minimum")
                        order = OrderIntent(
                            symbol, signal.side, Action.OPEN, quantity, fill_price, signal.reason,
                            Regime(regime_row["regime"]), stop_price, signal.source,
                        )
                        decision = guard_signal(
                            signal, order, portfolio, equity_at_open, funding_rate,
                            open_time.to_pydatetime(), settings, bypass_cooldown=signal.close_first,
                            bars_since_exit=(index - last_exit_bar_index
                                             if last_exit_bar_index is not None else None),
                        )
                    except ValueError as error:
                        reject_entry(str(error), signal, open_time, index,
                                     details={"fill_price": fill_price})
                    else:
                        if not decision.allowed:
                            held = portfolio.positions.get(symbol)
                            reject_entry(decision.reason, signal, open_time, index,
                                         position_source=held.source if held is not None else None,
                                         details=decision.details)
                        else:
                            position = Position(
                                symbol=symbol, side=signal.side, quantity=quantity,
                                entry_price=fill_price, stop_price=stop_price,
                                opened_at=open_time.to_pydatetime(), reason=signal.reason,
                                regime=Regime(regime_row["regime"]).value,
                                source=signal.source,
                            )
                            portfolio.register_open(position, decision)
                            entry_fee = quantity * fill_price * fee_rate
                            cash -= entry_fee
                            active = _OpenTrade(
                                position=position, entry_fee=entry_fee,
                                entry_slippage=quantity * entry_slip_per_unit,
                                funding_paid=0.0,
                                protective_stop=stop_price,
                                initial_risk=abs(fill_price - stop_price),
                                best_price=fill_price,
                                entry_atr=float(signal_atr),
                                structure_low=(entry_levels or {}).get("structure_low"),
                                structure_high=(entry_levels or {}).get("structure_high"),
                                range_low=(entry_levels or {}).get("range_low"),
                                range_high=(entry_levels or {}).get("range_high"),
                                last_close=float(bars.iloc[index - 1]["close"]) if index > 0 else fill_price,
                            )
            pending = None

        # A stop order is active as soon as a next-open entry fills. Hourly OHLC
        # cannot reveal intrabar ordering, so stop wins if both stop and other
        # close conditions would be present in the same bar.
        if active is not None:
            stop = active.protective_stop
            hit = (active.position.side is Side.LONG and float(candle["low"]) <= stop) or (
                active.position.side is Side.SHORT and float(candle["high"]) >= stop
            )
            if hit:
                gap_through = (active.position.side is Side.LONG and raw_open <= stop) or (
                    active.position.side is Side.SHORT and raw_open >= stop
                )
                raw_exit = raw_open if gap_through else stop
                trade, cash = _close_trade(
                    symbol, active, raw_exit, close_time, "stop_gap" if gap_through else "stop_hit",
                    fee_rate=fee_rate, slippage_rate=slippage_rate, cash=cash,
                )
                trades.append(trade)
                portfolio.register_close(symbol, open_time.to_pydatetime(), reason=trade.exit_reason, net_pnl=trade.net_pnl)
                last_exit_bar_index = index + 1
                active = None

        # Update the experimental profit-protection stop only after checking the
        # stop carried into this candle. OHLC cannot establish event order, so a
        # newly tightened stop becomes active on the next candle (no lookahead).
        time_exit_reason = None
        invalidation_reason = None
        if active is not None:
            side = active.position.side
            current_close = float(candle["close"])
            active.best_price = (max(active.best_price, float(candle["high"])) if side is Side.LONG
                                 else min(active.best_price, float(candle["low"])))
            exit_cfg = settings.exits
            if side is Side.LONG:
                if (exit_cfg.structure_invalid_enabled and active.structure_low is not None
                        and current_close < active.structure_low - exit_cfg.structure_invalid_buffer_atr * active.entry_atr):
                    invalidation_reason = "long_structure_invalid"
            else:
                if (exit_cfg.structure_invalid_enabled and active.structure_high is not None
                        and current_close > active.structure_high + exit_cfg.structure_invalid_buffer_atr * active.entry_atr):
                    invalidation_reason = "short_structure_invalid"
            regime_index = regimes_by_bar[index]
            if invalidation_reason is None and exit_cfg.regime_fade_enabled and regime_index >= 0:
                regime_label = regimes_4h.iloc[regime_index]["regime"]
                progress = ((current_close - active.position.entry_price) if side is Side.LONG
                            else (active.position.entry_price - current_close))
                entry_regime = Regime.LOW.value if side is Side.LONG else Regime.HIGH.value
                if (active.position.regime == entry_regime and regime_label is Regime.SIDEWAYS
                        and progress < exit_cfg.regime_fade_min_atr * active.entry_atr):
                    invalidation_reason = "long_regime_faded" if side is Side.LONG else "short_regime_faded"
            moved_favorably = current_close > active.last_close if side is Side.LONG else current_close < active.last_close
            active.no_progress_count = 0 if moved_favorably else active.no_progress_count + 1
            active.last_close = current_close
            favorable = ((active.best_price - active.position.entry_price) if side is Side.LONG
                         else (active.position.entry_price - active.best_price))
            if (invalidation_reason is None and active.no_progress_count >= exit_cfg.no_progress_bars
                    and favorable < exit_cfg.no_progress_min_atr * active.entry_atr):
                invalidation_reason = "no_progress"
        if active is not None and exit_policy == "time_managed":
            side = active.position.side
            high, low = float(candle["high"]), float(candle["low"])
            active.best_price = max(active.best_price, high) if side is Side.LONG else min(active.best_price, low)
            favorable = (active.best_price - active.position.entry_price if side is Side.LONG
                         else active.position.entry_price - active.best_price)
            progress_r = favorable / active.initial_risk
            held_hours = (close_time - pd.Timestamp(active.position.opened_at)).total_seconds() / 3600
            policy = settings.exit_management
            if exit_policy == "time_managed":
                estimated_exit, _ = _apply_slippage(
                    float(candle["close"]), side, is_entry=False, slippage_rate=slippage_rate,
                )
                direction = 1 if side is Side.LONG else -1
                net_if_closed = (
                    active.position.quantity * (estimated_exit - active.position.entry_price) * direction
                    - active.entry_fee - active.position.quantity * estimated_exit * fee_rate
                    - active.funding_paid
                )
                if (held_hours >= policy.time_check_hours
                        and progress_r < policy.time_min_progress_r and net_if_closed <= 0):
                    time_exit_reason = "time_stop_12h_no_progress"
                elif held_hours >= policy.time_exit_hours and progress_r < policy.breakeven_trigger_r:
                    time_exit_reason = "time_exit_24h_unprotected"

        if active is not None:
            side = active.position.side
            entry = active.position.entry_price
            high, low = float(candle["high"]), float(candle["low"])
            active.best_price = max(active.best_price, high) if side is Side.LONG else min(active.best_price, low)
            favorable = (active.best_price - entry) if side is Side.LONG else (entry - active.best_price)
            management = settings.exit_management
            if favorable >= settings.stops.move_to_breakeven_atr * active.entry_atr:
                # Raw trigger level that approximately covers round-trip fees and
                # slippage. Funding remains an actual cash flow and can still make
                # final net PnL slightly negative.
                fee = fee_rate
                slip = slippage_rate
                if side is Side.LONG:
                    breakeven = entry * (1 + fee) / ((1 - slip) * (1 - fee))
                    active.protective_stop = max(active.protective_stop, breakeven)
                else:
                    breakeven = entry * (1 - fee) / ((1 + slip) * (1 + fee))
                    active.protective_stop = min(active.protective_stop, breakeven)
            if (exit_policy in ("breakeven_trail", "time_managed")
                    and favorable >= management.trailing_trigger_r * active.initial_risk):
                trail_distance = management.trailing_atr_mult * active.initial_risk / settings.signals.stop_atr_mult
                trailing = active.best_price - trail_distance if side is Side.LONG else active.best_price + trail_distance
                if side is Side.LONG:
                    active.protective_stop = max(active.protective_stop, trailing)
                else:
                    active.protective_stop = min(active.protective_stop, trailing)

        close_equity = cash
        if active is not None:
            direction = 1 if active.position.side is Side.LONG else -1
            close_equity += active.position.quantity * (float(candle["close"]) - active.position.entry_price) * direction
        portfolio.observe_equity(max(close_equity, 1e-12), close_time.to_pydatetime(), settings)
        equity_times.append(close_time)
        equity_values.append(close_equity)

        if not (start <= close_time < end):
            continue
        regime_index = regimes_by_bar[index]
        if regime_index < 0:
            continue
        regime = _snapshot(regimes_4h.iloc[regime_index], ohlcv_4h.index[regime_index])
        if regime is None:
            continue
        if time_flags_by_close is not None and close_time in time_flags_by_close:
            flags = time_flags_by_close[close_time]
        else:
            history_start = max(0, index - 47)
            recent_bars = bars.iloc[history_start : index + 1]
            recent_atr = atr_1h.reindex(recent_bars.index)
            flags = evaluate_time_filter(
                close_time.to_pydatetime(), recent_bars, settings, precomputed_atr=recent_atr
            )
        current_position = active.position.side if active is not None else None
        signal_range_row = regimes_4h.iloc[regime_index]
        signal_swing = bars.iloc[max(0, index - settings.stops.swing_lookback_1h + 1):index + 1]
        signal_range_low, signal_range_high = signal_range_row.get("range_low"), signal_range_row.get("range_high")
        signal = generate_signal(
            regime, candle, flags, settings, position=current_position,
            bars_1h=bars.iloc[:index + 1], bars_4h=ohlcv_4h.iloc[:regime_index + 1],
            entry_reference_price=float(candle["close"]), entry_atr=float(atr_1h.iloc[index]),
            range_low=float(signal_range_low) if signal_range_low is not None and np.isfinite(signal_range_low) else None,
            range_high=float(signal_range_high) if signal_range_high is not None and np.isfinite(signal_range_high) else None,
            structure_low=float(signal_swing["low"].min()) if not signal_swing.empty else None,
            structure_high=float(signal_swing["high"].max()) if not signal_swing.empty else None,
        )
        if time_exit_reason is not None and active is not None:
            if invalidation_reason is None:
                signal = Signal(None, Action.CLOSE, time_exit_reason, 1.0)
        if invalidation_reason is not None and active is not None:
            signal = Signal(active.position.side, Action.CLOSE, invalidation_reason, 1.0)
        if (active is not None
                and close_time - pd.Timestamp(active.position.opened_at) >= pd.Timedelta(settings.exits.max_hold_bars, unit="h")
                and signal.action is not Action.CLOSE):
            signal = Signal(active.position.side, Action.CLOSE, "max_hold_24h", 1.0)
        if signal.action is Action.OPEN:
            source = signal.source if signal.source is not None else MEAN_REVERSION
        elif active is not None:
            source = active.position.source
        else:
            source = signal.source if signal.source in {MEAN_REVERSION, TREND} else MEAN_REVERSION
        signal = with_signal_context(signal, symbol=symbol, source=source,
                                     timestamp=close_time.to_pydatetime())
        if signal.action is Action.OPEN and (
            (mode == "long" and signal.side is not Side.LONG)
            or (mode == "short" and signal.side is not Side.SHORT)
        ):
            reject_entry("mode_filter", signal, close_time, index)
            continue
        if signal.action in (Action.OPEN, Action.CLOSE):
            latest_rate_index = np.searchsorted(funding_times_ns, close_time.value, side="left") - 1
            latest_rate = float(funding["funding_rate"].iloc[latest_rate_index]) if latest_rate_index >= 0 else None
            entry_levels = None
            if signal.action is Action.OPEN:
                lookback = settings.stops.swing_lookback_1h
                swing = bars.iloc[max(0, index - lookback + 1):index + 1]
                regime_row = regimes_4h.iloc[regime_index]
                range_low, range_high = regime_row.get("range_low"), regime_row.get("range_high")
                entry_levels = {
                    "structure_low": float(swing["low"].min()) if not swing.empty else None,
                    "structure_high": float(swing["high"].max()) if not swing.empty else None,
                    "range_low": float(range_low) if range_low is not None and np.isfinite(range_low) else None,
                    "range_high": float(range_high) if range_high is not None and np.isfinite(range_high) else None,
                }
            pending = (signal, latest_rate, float(atr_1h.iloc[index]), entry_levels)

    final_time = bars.index[-1] + HOUR
    final_open_ns = bars.index[-1].value
    final_time_ns = final_time.value
    while funding_position < len(funding) and funding_times_ns[funding_position] <= final_time_ns + FUNDING_BPS_TIME_TOLERANCE_MS * 1_000_000:
        funding_time = funding.index[funding_position]
        if funding_times_ns[funding_position] > final_open_ns and active is not None:
            settlement = funding.iloc[funding_position]
            if pd.isna(settlement["mark_price"]) and abs(funding_time.value - final_time_ns) <= FUNDING_BPS_TIME_TOLERANCE_MS * 1_000_000:
                mark_price = float(bars["close"].iloc[-1])
            else:
                mark_price = _funding_mark_price(settlement, funding_time, bars)
            direction = 1 if active.position.side is Side.LONG else -1
            amount = active.position.quantity * mark_price * float(settlement["funding_rate"]) * direction
            cash -= amount
            active.funding_paid += amount
        funding_position += 1

    if active is not None:
        raw_close = float(bars["close"].iloc[-1])
        trade, cash = _close_trade(
            symbol, active, raw_close, final_time, "backtest_end",
            fee_rate=fee_rate, slippage_rate=slippage_rate, cash=cash,
        )
        trades.append(trade)
        portfolio.register_close(symbol, final_time.to_pydatetime(), reason=trade.exit_reason, net_pnl=trade.net_pnl)
        last_exit_bar_index = len(bars)
        equity_values[-1] = cash
    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_times), name="equity")
    from bot.backtest.metrics import summarize_backtest

    metrics = summarize_backtest(trades, equity_curve, initial_equity)
    return BacktestResult(
        symbol=symbol,
        mode=mode,
        initial_equity=float(initial_equity),
        trades=tuple(trades),
        equity_curve=equity_curve,
        metrics=metrics,
        skipped_entries=skipped_entries,
    )
