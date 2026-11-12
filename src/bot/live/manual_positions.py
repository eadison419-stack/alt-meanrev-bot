"""Verified one-way fill replay for changes to an already managed position."""

from datetime import datetime, timezone
import math


def replay_position_changes(position, trades, expected, *, cursor=None, entry_fee=None,
                            quantity_tolerance=1e-10):
    """Return a change plan without modifying the position or submitting orders.

    A persisted fill cursor makes retries idempotent, including a reduce/add
    round trip that leaves the same quantity and average entry price.
    """
    sign = 1 if position.side.value == "long" else -1
    quantity = float(position.quantity)
    average = float(position.entry_price)
    opened_at = position.opened_at
    fee = entry_fee
    start = None
    if cursor:
        key = (int(cursor["time"]), int(cursor["id"]))
        start = next((i + 1 for i, row in enumerate(trades) if row[:2] == key), None)
        if start is None or fee is None:
            raise RuntimeError("manual_sync_cursor_or_entry_fee_unavailable")
    else:
        # Upgrade an existing ledger only from a verified prefix matching its
        # original entry. Never treat the current exchange balance as the anchor.
        signed = avg = remaining_fee = 0.0
        cycle_at = None
        for i, (timestamp, trade_id, side, amount, price, commission) in enumerate(trades):
            delta = amount if side == "BUY" else -amount
            if signed == 0 or signed * delta > 0:
                avg = (abs(signed) * avg + amount * price) / (abs(signed) + amount)
                remaining_fee += commission
                if signed == 0:
                    cycle_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                signed += delta
            else:
                previous = abs(signed)
                remaining_fee *= max(0.0, previous - amount) / previous
                signed += delta
                if abs(signed) < quantity_tolerance:
                    signed = avg = remaining_fee = 0.0
                    cycle_at = None
                elif amount > previous:
                    avg, remaining_fee = price, commission * (amount - previous) / amount
                    cycle_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
            if (cycle_at is not None and abs((cycle_at - opened_at).total_seconds()) <= 30
                    and signed * sign > 0 and abs(abs(signed) - quantity) <= quantity_tolerance
                    and math.isclose(avg, average, rel_tol=1e-6, abs_tol=1e-10)):
                start = i + 1
                fee = remaining_fee if fee is None else float(fee)
                key = (timestamp, trade_id)
                break
        if start is None:
            raise RuntimeError("manual_sync_original_position_not_reconstructable")
    if not math.isfinite(fee) or fee < 0:
        raise RuntimeError("manual_sync_entry_fee_invalid")

    changes, closed_cycles = [], []
    totals = {"gross_pnl": 0.0, "fees": 0.0, "quantity": 0.0, "notional": 0.0}
    for timestamp, trade_id, side, amount, price, commission in trades[start:]:
        key = (timestamp, trade_id)
        at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        fill_sign = 1 if side == "BUY" else -1
        if quantity and fill_sign != sign:
            reduced = min(quantity, amount)
            entry_cost = fee * reduced / quantity
            close_cost = commission * reduced / amount
            gross = reduced * (price - average) * sign
            totals["gross_pnl"] += gross
            totals["fees"] += entry_cost + close_cost
            totals["quantity"] += reduced
            totals["notional"] += reduced * price
            changes.append({"action": "reduce", "at": at.isoformat(), "trade_id": trade_id,
                            "side": "long" if sign > 0 else "short", "quantity": reduced,
                            "price": price, "gross_pnl": gross,
                            "entry_fee": entry_cost, "exit_fee": close_cost})
            quantity -= reduced
            fee = max(0.0, fee - entry_cost)
            if quantity <= quantity_tolerance:
                closed_cycles.append({**totals, "side": "long" if sign > 0 else "short",
                                      "opened_at": opened_at.isoformat(), "closed_at": at.isoformat()})
                quantity = average = fee = 0.0
                totals = {"gross_pnl": 0.0, "fees": 0.0, "quantity": 0.0, "notional": 0.0}
            amount -= reduced
            commission -= close_cost
        if amount > quantity_tolerance:
            if quantity == 0:
                sign, opened_at = fill_sign, at
            average = (quantity * average + amount * price) / (quantity + amount)
            quantity += amount
            fee += commission
            changes.append({"action": "add", "at": at.isoformat(), "trade_id": trade_id,
                            "side": "long" if sign > 0 else "short", "quantity": amount,
                            "price": price, "entry_fee": commission})

    remaining = None
    if quantity > quantity_tolerance:
        remaining = {"side": "long" if sign > 0 else "short", "quantity": quantity,
                     "entry_price": average, "entry_fee": fee, "opened_at": opened_at}
    matches = expected is None and remaining is None
    if expected is not None and remaining is not None:
        matches = (remaining["side"] == expected["side"]
                   and abs(quantity - float(expected["quantity"])) <= quantity_tolerance
                   and math.isclose(average, float(expected["entry_price"]), rel_tol=1e-6, abs_tol=1e-10))
    if not matches:
        raise RuntimeError("manual_sync_fills_do_not_match_exchange_snapshot")
    return {"remaining": remaining, "changes": changes, "closed_cycles": closed_cycles,
            "realized": totals, "cursor": {"time": key[0], "id": key[1]}}
