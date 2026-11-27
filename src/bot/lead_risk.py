"""Lead capital limits and isolated USD-M liquidation checks, independent of signals.

Binance formula and bracket definitions:
https://www.binance.com/en/support/faq/detail/b3c689c1f50a44cabb3a84e663b81d93
https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets
"""

from datetime import datetime
import math

from bot.risk import EntryExposure
from bot.signals import Side


def settings_for_execution(settings, mode):
    if mode != "lead_usdm":
        return settings
    return settings.model_copy(update={"risk": settings.risk.model_copy(update={
        "daily_drawdown_halt_pct": settings.lead_risk.daily_drawdown_halt_pct,
    })})


def _positive(value):
    if isinstance(value, bool):
        raise ValueError("invalid positive risk input")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid positive risk input")
    return value


def lead_entry_exposure(state, settings, now):
    """Never use original entry stops or stale closed candles as live risk inputs."""
    snapshot = state.exchange_snapshot
    at = datetime.fromisoformat(snapshot["at"])
    if (state.execution_status != "synchronized" or at.tzinfo is None
            or not 0 <= (now - at).total_seconds() <= 90):
        raise ValueError("lead_risk_snapshot_unavailable")
    live = snapshot["positions"]
    if set(live) != set(state.portfolio.positions):
        raise ValueError("lead_risk_positions_mismatch")
    if any(row.get("status") in {"submitting", "uncertain", "protected", "limit_pending"}
           for row in state.mr_entry_attempts.values()):
        raise ValueError("lead_risk_pending_entry")
    audit = state.protective_stop_audit
    if (audit.get("status") != "verified" or audit.get("mismatch_count") != 0
            or audit.get("position_count") != len(live)):
        raise ValueError("lead_risk_stops_unverified")
    notionals = {Side.LONG: 0.0, Side.SHORT: 0.0}
    capital_risk = 0.0
    giveback_risk = 0.0
    cost = settings.fees.taker_bps / 10_000 + settings.risk.max_slippage_pct
    for symbol, position in state.portfolio.positions.items():
        row = live[symbol]
        mark = _positive(row.get("mark_price") or row.get("info", {}).get("markPrice"))
        quantity = _positive(row["quantity"])
        stop = state.exchange_orders_by_symbol[symbol]["stop"]
        stop_price = _positive(stop["stop_price"])
        if (row["side"] != position.side.value
                or not math.isclose(quantity, position.quantity, rel_tol=1e-8, abs_tol=1e-10)
                or not stop.get("id")
                or not math.isclose(_positive(stop["amount"]), quantity, rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError("lead_risk_protection_mismatch")
        direction = 1 if position.side is Side.LONG else -1
        distance = direction * (mark - stop_price)
        if distance <= 0:
            raise ValueError("lead_risk_stop_crossed")
        notionals[position.side] += quantity * mark
        entry = _positive(position.entry_price)
        principal_distance = max(0.0, direction * (entry - stop_price))
        # Principal risk stays anchored to entry; floating profit only contributes
        # the amount that could be given back before reaching entry or the stop.
        capital_risk += quantity * (principal_distance + (entry + stop_price) * cost)
        giveback_per_unit = (max(0.0, mark - max(entry, stop_price))
                             if position.side is Side.LONG
                             else max(0.0, min(entry, stop_price) - mark))
        giveback_risk += quantity * giveback_per_unit
    weighted_giveback_risk = giveback_risk * settings.lead_risk.giveback_risk_weight
    stop_risk = capital_risk + weighted_giveback_risk
    return EntryExposure(notionals[Side.LONG], notionals[Side.SHORT], stop_risk,
                         capital_risk=capital_risk, giveback_risk=giveback_risk,
                         weighted_giveback_risk=weighted_giveback_risk)


def parse_brackets(response, market_id):
    rows = response if isinstance(response, list) else [response]
    matches = [row for row in rows if isinstance(row, dict) and row.get("symbol") == market_id]
    if len(matches) != 1 or not isinstance(matches[0].get("brackets"), list):
        raise ValueError("lead_leverage_brackets_unavailable")
    # User-specific limits are read from the signed response, never a public/static table.
    result = []
    for row in matches[0]["brackets"]:
        floor, cap = float(row["notionalFloor"]), _positive(row["notionalCap"])
        mmr, cum = _positive(row["maintMarginRatio"]), float(row["cum"])
        leverage = _positive(row["initialLeverage"])
        if (not math.isfinite(floor) or floor < 0 or cap <= floor or mmr >= 1
                or not math.isfinite(cum) or cum < 0 or leverage != int(leverage)
                or (result and (not math.isclose(result[-1]["cap"], floor)
                    or mmr < result[-1]["mmr"]))
                or (not result and floor != 0)):
            raise ValueError("lead_leverage_brackets_invalid")
        if result:
            expected_cum = result[-1]["cum"] + floor * (mmr - result[-1]["mmr"])
            if not math.isclose(cum, expected_cum, rel_tol=1e-7, abs_tol=1e-6):
                raise ValueError("lead_leverage_brackets_discontinuous")
        elif cum != 0:
            raise ValueError("lead_leverage_brackets_invalid")
        result.append({"floor": floor, "cap": cap, "mmr": mmr, "cum": cum,
                       "max_leverage": int(leverage)})
    if not result:
        raise ValueError("lead_leverage_brackets_unavailable")
    return result


def liquidation_price(side, entry, quantity, wallet, brackets):
    """Solve wallet + signed unrealized PnL = mark notional * MMR - cum.

    The resulting liquidation notional must belong to the assumed maintenance tier.
    """
    entry, quantity, wallet = map(_positive, (entry, quantity, wallet))
    direction = 1 if Side(side) is Side.LONG else -1
    for tier in brackets:
        price = (wallet + tier["cum"] - direction * quantity * entry) / (
            quantity * (tier["mmr"] - direction))
        if price > 0 and tier["floor"] <= price * quantity <= tier["cap"]:
            return price
    raise ValueError("lead_liquidation_tier_unresolved")


def liquidation_is_safe(side, entry, stop, liquidation, policy):
    try:
        entry, stop, liquidation = map(_positive, (entry, stop, liquidation))
    except (TypeError, ValueError, OverflowError):
        return False
    direction = 1 if Side(side) is Side.LONG else -1
    # A tightened stop may already lock in profit. Its remaining principal risk is zero.
    stop_distance = max(0.0, direction * (entry - stop))
    liquidation_distance = direction * (entry - liquidation)
    return (liquidation_distance > 0
            and direction * (stop - liquidation) > 0
            and liquidation_distance >= policy.liquidation_stop_distance_multiple * stop_distance
            and abs(stop - liquidation) / liquidation >= policy.min_stop_liquidation_gap_pct)


def select_leverage(side, entry, quantity, stop, brackets, settings, available):
    """Choose leverage only after quantity; never use leverage to expand notional."""
    entry, quantity, available = map(_positive, (entry, quantity, available))
    notional = entry * quantity
    tier = next((t for t in brackets if t["floor"] <= notional <= t["cap"]), None)
    if tier is None:
        raise ValueError("lead_notional_outside_brackets")
    policy = settings.lead_risk
    costs = settings.fees.taker_bps / 10_000 + settings.risk.max_slippage_pct
    for leverage in range(min(policy.max_leverage, tier["max_leverage"]), policy.min_leverage - 1, -1):
        margin = notional / leverage
        # Reserve full-notional entry/exit costs; conservatively deduct entry costs from wallet.
        if available < margin + 2 * notional * costs:
            continue
        wallet = margin - notional * costs
        if wallet < notional * policy.min_isolated_margin_pct:
            continue
        try:
            liquidation = liquidation_price(side, entry, quantity, wallet, brackets)
        except ValueError:
            continue
        if liquidation_is_safe(side, entry, stop, liquidation, policy):
            return {"leverage": leverage, "estimated_liquidation_price": liquidation,
                    "notional": notional, "initial_margin": margin,
                    "maintenance_margin_rate": tier["mmr"]}
    raise ValueError("lead_no_safe_leverage_or_margin")
