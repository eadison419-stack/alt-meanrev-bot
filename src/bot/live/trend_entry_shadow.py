"""No-order comparison of TREND pullback entries and capped market entries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from bot.portfolio import Position
from bot.signals import Action, Side
from bot.strategy_router import TREND
from bot.trend_strategy import (
    calculate_trend_stop,
    evaluate_trend_exit,
    initialize_trend_exit_state,
)


CHASE_GUARDS_ATR = (0.1, 0.2, 0.3)
_STATE_VERSION = 1


def quote_execution_price(side: Side | str, *, last: float | None,
                          bid: float | None, ask: float | None) -> tuple[float | None, str]:
    """Use the executable side of the quote, falling back to last when unavailable."""
    side = Side(side)
    preferred = ask if side is Side.LONG else bid
    if preferred is not None and math.isfinite(float(preferred)) and float(preferred) > 0:
        return float(preferred), "ask" if side is Side.LONG else "bid"
    if last is not None and math.isfinite(float(last)) and float(last) > 0:
        return float(last), "last_proxy"
    return None, "unavailable"


def adverse_chase_atr(side: Side | str, *, market_price: float,
                      reference_price: float, atr_value: float) -> float:
    """Return only the adverse distance from the confirmed signal price, in ATR units."""
    side = Side(side)
    if any(not math.isfinite(value) for value in (market_price, reference_price, atr_value)):
        raise ValueError("shadow entry prices and ATR must be finite")
    if market_price <= 0 or reference_price <= 0 or atr_value <= 0:
        raise ValueError("shadow entry prices and ATR must be positive")
    distance = (market_price - reference_price if side is Side.LONG
                else reference_price - market_price)
    return max(0.0, distance) / atr_value


def _aware(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("shadow timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _valid_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _apply_entry_slippage(price: float, side: Side, rate: float) -> float:
    return price * (1 + rate) if side is Side.LONG else price * (1 - rate)


def _apply_exit_slippage(price: float, side: Side, rate: float) -> float:
    return price * (1 - rate) if side is Side.LONG else price * (1 + rate)


def _candidate_key(pending: dict[str, Any]) -> str:
    return f"{pending.get('symbol')}|{pending.get('signal_time')}"


def _new_candidate(pending: dict[str, Any]) -> dict[str, Any]:
    side = Side(pending["side"])
    reference_price = pending.get("signal_reference_price") or pending.get("reference_price")
    return {
        "candidate_id": _candidate_key(pending),
        "symbol": pending["symbol"],
        "source": TREND,
        "side": side.value,
        "signal_time": pending["signal_time"],
        "confirmed_at": pending.get("confirmed_at"),
        "expires_at": pending.get("expires_at"),
        "reference_price": float(reference_price),
        "planned_pullback_price": float(pending["pullback_price"]),
        "atr_1h": float(pending["atr"]),
        "structure_break_price": float(pending["structure_break_price"]),
        "regime": str(pending.get("regime") or "MID"),
        "initial_quote": None,
        "scenarios": {
            "pullback": {"status": "waiting_pullback"},
            **{f"hybrid_{guard:.1f}atr": {"status": "waiting_first_quote", "guard_atr": guard}
               for guard in CHASE_GUARDS_ATR},
        },
    }


class TrendEntryShadowTracker:
    """Persist isolated no-order entry scenarios and manage their virtual exits."""

    def __init__(self, path: str | Path, settings):
        self.path = Path(path)
        self.settings = settings
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict)
                    or data.get("version") != _STATE_VERSION
                    or not isinstance(data.get("candidates"), dict)):
                raise ValueError("unsupported trend entry shadow state version")
            self.state = data
        else:
            self.state = {
                "version": _STATE_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "guards_atr": list(CHASE_GUARDS_ATR),
                "cost_model": "configured taker fee and slippage; funding excluded",
                "candidates": {},
            }
        self.state["last_loaded_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    @property
    def active_symbols(self) -> set[str]:
        active = set()
        for candidate in self.state["candidates"].values():
            if any(scenario.get("status") in {"waiting_first_quote", "waiting_pullback", "open"}
                   for scenario in candidate["scenarios"].values()):
                active.add(candidate["symbol"])
        return active

    def needs_quote(self, symbol: str) -> bool:
        return any(
            candidate["symbol"] == symbol
            and any(scenario.get("status") in {"waiting_first_quote", "waiting_pullback"}
                    for scenario in candidate["scenarios"].values())
            for candidate in self.state["candidates"].values()
        )

    def register_pending(self, pending: dict[str, Any] | None) -> bool:
        if not pending or pending.get("source") != TREND or pending.get("execution_type") != "pullback":
            return False
        key = _candidate_key(pending)
        if key in self.state["candidates"]:
            return False
        required = ("symbol", "side", "signal_time", "reference_price", "pullback_price",
                    "atr", "stop_price", "structure_break_price", "expires_at")
        if any(pending.get(field) is None for field in required):
            return False
        self.state["candidates"][key] = _new_candidate(pending)
        return True

    def observe_symbol(self, symbol: str, *, now: datetime, current_price: float,
                       quote_fresh: bool, quote_metadata: dict[str, Any] | None,
                       bars_1h: pd.DataFrame | None, market_risk_state: str) -> bool:
        """Update virtual entry and exit states. This method never submits orders."""
        now = _aware(now)
        last = _valid_price(current_price)
        if last is None:
            return False
        metadata = quote_metadata or {}
        changed = False
        for candidate in self.state["candidates"].values():
            if candidate["symbol"] != symbol:
                continue
            side = Side(candidate["side"])
            market_price, price_basis = quote_execution_price(
                side, last=metadata.get("quote_price") if quote_fresh else last,
                bid=metadata.get("quote_bid") if quote_fresh else None,
                ask=metadata.get("quote_ask") if quote_fresh else None,
            )
            if market_price is None:
                continue
            invalidated = self._structure_invalid(candidate, last)
            for name, scenario in candidate["scenarios"].items():
                status = scenario.get("status")
                if status == "open":
                    if not quote_fresh:
                        continue
                    changed = self._advance_open_scenario(
                        candidate, scenario, now=now, current_price=last,
                        market_price=market_price, price_basis=price_basis,
                        quote_metadata=metadata if quote_fresh else {}, bars_1h=bars_1h,
                    ) or changed
                    continue
                if status not in {"waiting_first_quote", "waiting_pullback"}:
                    continue
                if market_risk_state == "PAUSE_NEW":
                    scenario.update(status="cancelled", reason="market_risk_paused", ended_at=now.isoformat())
                    changed = True
                    continue
                if invalidated and quote_fresh:
                    scenario.update(status="cancelled", reason="structure_invalidated", ended_at=now.isoformat())
                    changed = True
                    continue
                expires_at = _aware(candidate["expires_at"])
                if now >= expires_at:
                    scenario.update(status="expired", reason="timeout", ended_at=now.isoformat())
                    changed = True
                    continue
                if not quote_fresh:
                    continue

                pullback = float(candidate["planned_pullback_price"])
                touched = (last <= pullback if side is Side.LONG else last >= pullback)
                if scenario.get("status") == "waiting_first_quote":
                    if candidate.get("initial_quote") is None:
                        candidate["initial_quote"] = {
                            "at": now.isoformat(), "last_price": last,
                            "bid": _valid_price(metadata.get("quote_bid")),
                            "ask": _valid_price(metadata.get("quote_ask")),
                            "market_price": market_price, "price_basis": price_basis,
                            "quote_age_sec": metadata.get("quote_age_sec"),
                        }
                        changed = True
                    if touched:
                        changed = self._open_scenario(
                            candidate, scenario, entry_price=market_price, opened_at=now,
                            entry_mode="pullback_touch", price_basis=price_basis, bars_1h=bars_1h,
                        ) or changed
                    elif name == "pullback":
                        scenario["status"] = "waiting_pullback"
                        changed = True
                    else:
                        guard = float(scenario["guard_atr"])
                        deviation = adverse_chase_atr(
                            side, market_price=market_price,
                            reference_price=float(candidate["reference_price"]),
                            atr_value=float(candidate["atr_1h"]),
                        )
                        scenario["initial_adverse_deviation_atr_from_signal"] = deviation
                        if deviation <= guard + 1e-12:
                            changed = self._open_scenario(
                                candidate, scenario, entry_price=market_price, opened_at=now,
                                entry_mode="immediate_within_guard", price_basis=price_basis,
                                bars_1h=bars_1h,
                            ) or changed
                        else:
                            scenario["status"] = "waiting_pullback"
                            changed = True
                elif touched:
                    changed = self._open_scenario(
                        candidate, scenario, entry_price=market_price, opened_at=now,
                        entry_mode="pullback_touch", price_basis=price_basis, bars_1h=bars_1h,
                    ) or changed
        return changed

    @staticmethod
    def _structure_invalid(candidate: dict[str, Any], current_price: float) -> bool:
        structure = float(candidate["structure_break_price"])
        return (current_price < structure if candidate["side"] == Side.LONG.value
                else current_price > structure)

    def _open_scenario(self, candidate: dict[str, Any], scenario: dict[str, Any], *,
                       entry_price: float, opened_at: datetime, entry_mode: str,
                       price_basis: str, bars_1h: pd.DataFrame | None) -> bool:
        side = Side(candidate["side"])
        slip_rate = max(0.0, float(self.settings.fees.slippage_bps)) / 10_000
        fill_price = _apply_entry_slippage(entry_price, side, slip_rate)
        try:
            stop, structure = calculate_trend_stop(
                side, fill_price, float(candidate["atr_1h"]), bars_1h, self.settings,
            )
            state = initialize_trend_exit_state(fill_price, stop, structure,
                                                stop_mode=self.settings.trend_strategy.stop_mode)
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            scenario.update(status="risk_rejected", reason=str(exc)[:200], ended_at=opened_at.isoformat(),
                            entry_mode=entry_mode, entry_price=fill_price)
            return True
        scenario.update(
            status="open", entry_mode=entry_mode, opened_at=opened_at.isoformat(),
            entry_price=fill_price, reference_entry_price=entry_price,
            price_basis=price_basis, stop_price=stop, structure_break_price=structure,
            initial_risk=abs(fill_price - stop), exit_state=state,
            last_mark_price=entry_price,
        )
        return True

    def _advance_open_scenario(self, candidate: dict[str, Any], scenario: dict[str, Any], *,
                               now: datetime, current_price: float, market_price: float,
                               price_basis: str, quote_metadata: dict[str, Any],
                               bars_1h: pd.DataFrame | None) -> bool:
        side = Side(candidate["side"])
        opened_at = _aware(scenario["opened_at"])
        position = Position(
            symbol=candidate["symbol"], side=side, quantity=1.0,
            entry_price=float(scenario["entry_price"]), stop_price=float(scenario["stop_price"]),
            opened_at=opened_at, reason="trend_pullback_candidate", regime=candidate["regime"],
            source=TREND,
        )
        decision = evaluate_trend_exit(
            position, bars_1h, self.settings, now=now,
            exit_state=scenario["exit_state"], current_price=current_price, log_decision=False,
        )
        changed = scenario.get("exit_state") != decision.exit_state
        scenario["exit_state"] = decision.exit_state
        scenario["last_mark_price"] = current_price
        if decision.signal.action is not Action.CLOSE:
            return changed

        reason = decision.signal.reason
        if reason.endswith("_hit"):
            raw_exit = float(decision.evidence.get("trigger_stop", decision.active_stop_price))
            exit_basis = "simulated_stop_trigger"
        else:
            raw_exit, exit_basis = quote_execution_price(
                Side.SHORT if side is Side.LONG else Side.LONG,
                last=current_price,
                bid=_valid_price(quote_metadata.get("quote_bid")),
                ask=_valid_price(quote_metadata.get("quote_ask")),
            )
        if raw_exit is None:
            return changed
        slip_rate = max(0.0, float(self.settings.fees.slippage_bps)) / 10_000
        fill_exit = _apply_exit_slippage(raw_exit, side, slip_rate)
        fee_rate = max(0.0, float(self.settings.fees.taker_bps)) / 10_000
        direction = 1.0 if side is Side.LONG else -1.0
        gross = direction * (fill_exit - float(scenario["entry_price"]))
        costs = (float(scenario["entry_price"]) + fill_exit) * fee_rate
        net = gross - costs
        risk = float(scenario["initial_risk"])
        scenario.update(
            status="closed", reason=reason, closed_at=now.isoformat(),
            exit_price=fill_exit, exit_reference_price=raw_exit, exit_price_basis=exit_basis,
            gross_pnl_per_unit=gross, estimated_costs_per_unit=costs,
            estimated_net_pnl_per_unit=net, net_r=net / risk,
            funding_included=False,
        )
        return True

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


def summarize_shadow_state(data: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    names = ["pullback", *(f"hybrid_{guard:.1f}atr" for guard in CHASE_GUARDS_ATR)]
    candidates = data.get("candidates", {})
    for name in names:
        scenarios = [candidate.get("scenarios", {}).get(name, {}) for candidate in candidates.values()]
        statuses = {}
        for scenario in scenarios:
            status = scenario.get("status", "missing")
            statuses[status] = statuses.get(status, 0) + 1
        closed = [float(scenario["net_r"]) for scenario in scenarios
                  if scenario.get("status") == "closed" and scenario.get("net_r") is not None]
        wins = [value for value in closed if value > 0]
        losses = [value for value in closed if value < 0]
        summaries[name] = {
            "candidate_count": len(scenarios),
            "statuses": statuses,
            "closed_count": len(closed),
            "win_rate": len(wins) / len(closed) if closed else None,
            "mean_net_r": sum(closed) / len(closed) if closed else None,
            "profit_factor": (sum(wins) / abs(sum(losses)) if losses else None),
            "net_r_sequence": closed,
        }
    return {
        "version": data.get("version"),
        "created_at": data.get("created_at"),
        "candidate_count": len(candidates),
        "cost_model": data.get("cost_model"),
        "scenarios": summaries,
    }
