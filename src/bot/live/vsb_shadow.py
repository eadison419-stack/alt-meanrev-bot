"""Independent virtual VSB account. This module has no exchange/executor API."""

from datetime import datetime, timedelta
import math

from bot.lead_risk import lead_entry_exposure
from bot.indicators import validate_ohlcv
from bot.live.multi_paper import MultiPaperState
from bot.live.paper import effective_max_hold_hours
from bot.mr_candidate import (staged_profit_lock_stop, trailing_stop, no_progress_update,
                              structure_invalid_triggered, structure_invalid_quote_exit)
from bot.portfolio import Position, classify_exit_category
from bot.regime import Regime
from bot.risk import OrderIntent, assess_entry, calculate_adaptive_stop, calculate_order_quantity
from bot.signals import Action, Side
from bot.strategy_router import MEAN_REVERSION
from bot.vsb import strategy_metadata


class VsbShadowLedger:
    def __init__(self, path, settings):
        self.path, self.settings = path, settings
        self.state = MultiPaperState.load(path)
        if self.state.execution_mode not in {"local_paper", "vsb_shadow"}:
            raise ValueError("shadow ledger must be a virtual account")
        self.state.execution_mode = "vsb_shadow"
        self.state.strategy_variant = "vsb_shadow"
        self.state.execution_status = "synchronized"

    @property
    def symbols(self):
        return set(self.state.portfolio.positions) | set(self.state.pending_by_symbol)

    def save(self):
        self.state.save(self.path)

    def event(self, kind, at, **fields):
        self.state.events.append({"event": kind, "at": at.isoformat(), "execution_mode": "vsb_shadow",
                                  "virtual": True, "strategy_tag": "VSB", **fields})

    def _exposure(self, now):
        """Build evidence only from this virtual account, never from live state."""
        state = self.state
        for meta in state.exit_management_by_symbol.values():
            at = meta.get("last_quote_at")
            if at is None or not 0 <= (now - datetime.fromisoformat(at)).total_seconds() <= 15:
                raise ValueError("shadow risk quote unavailable")
        state.exchange_snapshot = {"at": now.isoformat(), "virtual": True, "positions": {
            s: dict(side=p.side.value, quantity=p.quantity, mark_price=state.mark_prices[s])
            for s, p in state.portfolio.positions.items()}}
        state.exchange_orders_by_symbol = {s: {"stop": dict(
            id="virtual_stop", amount=p.quantity, stop_price=state.exit_management_by_symbol[s]["protective_stop"],
            virtual=True)} for s, p in state.portfolio.positions.items()}
        state.protective_stop_audit = dict(status="verified", mismatch_count=0,
                                          position_count=len(state.portfolio.positions), virtual=True)
        return lead_entry_exposure(state, self.settings, now)

    def offer(self, candidate, quote, at, funding_rate, market_risk):
        state, cfg = self.state, self.settings
        (symbol, _, side, close, stop, entry_atr, low, high, regime, reason,
         _, context, _, plan) = candidate
        bar = context.bar_close_at
        if at > bar + timedelta(seconds=cfg.execution.confirm_delay_sec_max):
            return False
        key = bar.isoformat()
        if state.last_observed_by_symbol.get(symbol) == key:
            return False
        state.last_observed_by_symbol[symbol] = key
        if symbol in self.symbols or state.pending_by_symbol:
            return False
        direction = 1 if side is Side.LONG else -1
        adverse = -direction * (quote - context.closed_ema) / context.closed_ema
        if (context.breakout_level is None or direction * (quote - context.breakout_level) <= 0
                or not -1.2 * context.deviation_pct <= adverse < context.deviation_pct):
            self.event("shadow_entry_rejected", at, symbol=symbol, reason="breakout_quote_invalid")
            self.save()
            return False
        is_limit = plan["order_type"] == "limit"
        price = float(plan["limit_price"]) if is_limit else quote
        if is_limit and direction * (quote - price) <= 0:
            self.event("shadow_entry_rejected", at, symbol=symbol, reason="limit_already_touched")
            self.save()
            return False
        stop, _ = calculate_adaptive_stop(side, price, entry_atr, low if side is Side.LONG else high, cfg)
        multiplier = cfg.market_risk.elevated_size_multiplier if market_risk == "RISK_ELEVATED" else 1.
        exposure = self._exposure(at)
        equity = state.current_equity
        qty = calculate_order_quantity(equity, price, state.portfolio, cfg, step_size=1e-8,
            stop_price=stop, target_notional_usdt=equity * cfg.lead_risk.max_symbol_notional_pct,
            exposure=exposure, side=side, portfolio_risk_cap_pct=cfg.lead_risk.max_portfolio_stop_risk_pct,
            size_multiplier=multiplier)
        if qty <= 0:
            self.event("shadow_entry_rejected", at, symbol=symbol, reason="shadow_risk_budget_exhausted")
            self.save()
            return False
        intent = OrderIntent(symbol, side, Action.OPEN, qty, price, reason, Regime(regime), stop, MEAN_REVERSION)
        approval = assess_entry(intent, state.portfolio, equity, funding_rate, at, cfg,
            exposure=exposure, portfolio_risk_cap_pct=cfg.lead_risk.max_portfolio_stop_risk_pct,
            per_symbol_notional_usdt=equity * cfg.lead_risk.max_symbol_notional_pct,
            market_risk_state=market_risk)
        if qty <= 0 or not approval.allowed:
            self.event("shadow_entry_rejected", at, symbol=symbol, reason=approval.reason)
            self.save()
            return False
        fill_price = price if is_limit else price * (1 + direction * cfg.fees.slippage_bps / 10000)
        order = dict(symbol=symbol, side=side.value, price=fill_price, stop=stop, quantity=qty,
            entry_atr=entry_atr, structure_low=low, structure_high=high, reason=reason, regime=regime,
            signal_at=key, expires_at=plan["limit_expires_at"], **strategy_metadata(plan))
        if is_limit:
            state.pending_by_symbol[symbol] = order
            self.event("shadow_limit_pending", at, **order)
        else:
            self._fill(order, at)
        self.save()
        return True

    def _fill(self, order, at):
        s, cfg = self.state, self.settings
        symbol, side = order["symbol"], Side(order["side"])
        price, qty, stop = order["price"], order["quantity"], order["stop"]
        fee = price * qty * cfg.fees.taker_bps / 10000
        s.cash -= fee
        s.portfolio.positions[symbol] = Position(symbol, side, qty, price, stop, at,
                                               order["reason"], order["regime"])
        s.mark_prices[symbol] = price
        s.entry_costs[symbol] = dict(entry_fee=fee)
        s.exit_management_by_symbol[symbol] = dict(**strategy_metadata(order), entry_atr=order["entry_atr"],
            initial_risk=abs(price - stop), protective_stop=stop, best_price=price, worst_price=price,
            structure_low=order["structure_low"], structure_high=order["structure_high"],
            last_close=price, no_progress_count=0, trailing_started=False, last_quote_at=at.isoformat())
        self.event("paper_fill", at, action="OPEN", symbol=symbol, price=price, quantity=qty,
                   stop_price=stop, entry_atr=order["entry_atr"], fee=fee)

    def close(self, symbol, price, reason, at):
        s, cfg = self.state, self.settings
        p = s.portfolio.positions[symbol]
        meta = s.exit_management_by_symbol[symbol]
        direction = 1 if p.side is Side.LONG else -1
        fill = price * (1 - direction * cfg.fees.slippage_bps / 10000)
        gross = p.quantity * (fill - p.entry_price) * direction
        exit_fee = fill * p.quantity * cfg.fees.taker_bps / 10000
        fees = exit_fee + s.entry_costs[symbol]["entry_fee"]
        risk = meta["initial_risk"]
        favorable = direction * (meta["best_price"] - p.entry_price)
        adverse = -direction * (meta["worst_price"] - p.entry_price)
        s.cash += gross - exit_fee
        self.event("paper_fill", at, action="CLOSE", symbol=symbol, side=p.side.value, price=fill,
            entry_price=p.entry_price, opened_at=p.opened_at.isoformat(), quantity=p.quantity,
            reason=reason, gross_pnl=gross, fees=fees, net_pnl=gross - fees,
            realized_r=(gross - fees) / (risk * p.quantity), gross_r=gross / (risk * p.quantity),
            mfe_r=max(0., favorable / risk), mae_r=max(0., adverse / risk),
            mfe=max(0., favorable), mae=max(0., adverse), initial_risk=risk,
            observation_gaps=meta.get("observation_gaps", 0))
        s.portfolio.last_exit_at[symbol] = at
        s.portfolio.last_exit_source[symbol] = MEAN_REVERSION
        s.portfolio.last_exit_category[symbol] = ("stop_loss" if "structure_invalid" in reason and gross - fees <= 0
                                                 else classify_exit_category(reason, gross - fees))
        for mapping in (s.portfolio.positions, s.exit_management_by_symbol, s.entry_costs, s.mark_prices):
            mapping.pop(symbol, None)

    def tick(self, quotes, now, *, allow_pending=True):
        """Timestamped contract prices; stale observations never fill or manage."""
        s, cfg = self.state, self.settings
        for symbol, order in list(s.pending_by_symbol.items()):
            if not allow_pending or now >= datetime.fromisoformat(order["expires_at"]):
                s.pending_by_symbol.pop(symbol)
                self.event("shadow_limit_cancelled", now, symbol=symbol,
                           reason="expired" if allow_pending else "mode_changed")
        for symbol in list(self.symbols):
            quote = quotes.get(symbol)
            if quote is None:
                continue
            price, quoted_at = quote
            if not math.isfinite(price) or price <= 0 or not 0 <= (now - quoted_at).total_seconds() <= 15:
                continue
            pending = s.pending_by_symbol.get(symbol)
            if pending and (price <= pending["price"] if pending["side"] == "long" else price >= pending["price"]):
                s.pending_by_symbol.pop(symbol)
                self._fill(pending, now)
            p = s.portfolio.positions.get(symbol)
            if p is None or quoted_at < p.opened_at:
                continue
            meta = s.exit_management_by_symbol[symbol]
            last = meta.get("last_quote_at")
            if last and quoted_at <= datetime.fromisoformat(last):
                continue
            if last and (quoted_at - datetime.fromisoformat(last)).total_seconds() > 15:
                meta["observation_gaps"] = meta.get("observation_gaps", 0) + 1
            meta["last_quote_at"] = quoted_at.isoformat()
            direction = 1 if p.side is Side.LONG else -1
            s.mark_prices[symbol] = price
            meta["best_price"] = max(meta["best_price"], price) if direction == 1 else min(meta["best_price"], price)
            meta["worst_price"] = min(meta["worst_price"], price) if direction == 1 else max(meta["worst_price"], price)
            stop = meta["protective_stop"]
            if direction * (price - stop) <= 0:
                self.close(symbol, price, "stop_hit", now)
                continue
            invalid = structure_invalid_quote_exit(p.side, price, meta["structure_low"], meta["structure_high"],
                meta["entry_atr"], stop, buffer_atr=cfg.exits.structure_invalid_buffer_atr)
            if invalid:
                self.close(symbol, price, invalid[0], now)
                continue
            favorable = direction * (meta["best_price"] - p.entry_price)
            r = favorable / meta["initial_risk"]
            locked = staged_profit_lock_stop(p.side, p.entry_price, meta["entry_atr"], meta["initial_risk"],
                favorable, stop, first_trigger_atr=cfg.stops.move_to_breakeven_atr,
                first_lock_r=cfg.mr_candidate_exit.first_lock_r,
                second_trigger_r=cfg.mr_candidate_exit.second_lock_trigger_r,
                second_lock_r=cfg.mr_candidate_exit.second_lock_r,
                fee_rate=cfg.fees.taker_bps / 10000, slip_rate=cfg.fees.slippage_bps / 10000)
            target = trailing_stop(p.side, meta["best_price"], meta["initial_risk"], r, locked,
                trigger_r=cfg.mr_candidate_exit.trailing_trigger_r,
                atr_mult=cfg.exit_management.trailing_atr_mult, stop_atr_mult=cfg.signals.stop_atr_mult)
            updated_at = meta.get("stop_updated_at")
            can_update = (direction * (locked - stop) > 1e-12 or updated_at is None
                          or (now - datetime.fromisoformat(updated_at)).total_seconds() >= 15)
            if can_update and direction * (target - stop) > 1e-12:
                if direction * (price - target) <= 0:
                    self.close(symbol, price, "profit_protection_crossed", now)
                    continue
                meta.update(protective_stop=target, stop_updated_at=now.isoformat())
                self.event("shadow_stop_updated", now, symbol=symbol, stop_price=target, progress_r=r)
            if (r >= cfg.mr_candidate_exit.trailing_trigger_r
                    and direction * (meta["protective_stop"] - target) >= 0
                    and direction * (target - p.entry_price) > 0):
                meta["trailing_started"] = True
            limit = effective_max_hold_hours(p.side, p.entry_price, meta["protective_stop"], meta, cfg.exits)
            if limit is not None and (now - p.opened_at).total_seconds() >= limit * 3600:
                self.close(symbol, price, "max_hold_24h", now)
        self.save()

    def hourly(self, frames, now):
        s, cfg = self.state, self.settings
        for symbol, p in list(s.portfolio.positions.items()):
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                continue
            try:
                validate_ohlcv(frame)
            except (TypeError, ValueError):
                self.event("shadow_hourly_unavailable", now, symbol=symbol, reason="invalid_ohlcv")
                continue
            closed_at = frame.index[-1].to_pydatetime() + timedelta(hours=1)
            if closed_at != now.replace(minute=0, second=0, microsecond=0):
                continue
            meta = s.exit_management_by_symbol[symbol]
            if not p.opened_at < closed_at <= now or meta.get("last_hourly_bar") == closed_at.isoformat():
                continue
            meta["last_hourly_bar"] = closed_at.isoformat()
            close = float(frame.iloc[-1]["close"])
            # OHLC extremes are admissible only for complete post-fill bars.
            if frame.index[-1].to_pydatetime() >= p.opened_at:
                high, low = float(frame.iloc[-1]["high"]), float(frame.iloc[-1]["low"])
                meta["best_price"] = max(meta["best_price"], high) if p.side is Side.LONG else min(meta["best_price"], low)
                meta["worst_price"] = min(meta["worst_price"], low) if p.side is Side.LONG else max(meta["worst_price"], high)
            direction = 1 if p.side is Side.LONG else -1
            reason = None
            if structure_invalid_triggered(p.side, close, meta["structure_low"], meta["structure_high"],
                                            meta["entry_atr"], buffer_atr=cfg.exits.structure_invalid_buffer_atr):
                reason = "long_structure_invalid" if direction == 1 else "short_structure_invalid"
            holding_hour = (closed_at - p.opened_at.replace(minute=0, second=0, microsecond=0)).total_seconds() / 3600
            if reason is None and 1 <= holding_hour <= 4 and direction * (close - meta["box_mid"]) < 0:
                reason = "vsb_false_breakout"
            favorable = direction * (meta["best_price"] - p.entry_price)
            trigger, count = no_progress_update(p.side, close, meta["last_close"], meta["no_progress_count"],
                favorable, meta["entry_atr"], bars=cfg.exits.no_progress_bars, min_atr=cfg.exits.no_progress_min_atr)
            meta.update(last_close=close, no_progress_count=count)
            if reason is None and trigger:
                reason = "no_progress"
            if reason:
                self.close(symbol, close, reason, now)
        self.save()
