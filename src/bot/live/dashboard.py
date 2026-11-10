"""Loopback-only paper dashboard data service."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time as datetime_time, timedelta, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from bot.config import load_settings
from bot.live.manual_close import ManualCloseCommandStore, command_directory_for_state
from bot.live.paper import effective_max_hold_hours, runner_tail_is_exempt


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _exchange_execution_source(mode: str) -> str | None:
    return {"demo_usdm": "binance_usdm_demo",
            "lead_usdm": "binance_usdm_lead"}.get(mode)


def _display_close_reason(reason: str | None) -> str | None:
    return {
        "exchange_flat_unverified": "外部平仓（成交价／盈亏待核验）",
        "external_close_exchange_trades_verified": "手动／外部平仓（交易所成交已核验）",
        "protective_stop_exchange_algo": "机器人保护止损",
    }.get(reason, reason)


def build_order_snapshot(state: dict[str, Any], *, start_date: str | None = None,
                         end_date: str | None = None,
                         current_positions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pair paper fills into one round-trip trade and filter by Shanghai entry date."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    if start and end and start > end:
        raise ValueError("start date must not be after end date")
    lower = datetime.combine(start, datetime_time.min, SHANGHAI) if start else None
    upper = datetime.combine(end, datetime_time.max, SHANGHAI) if end else None
    fills = []
    source_events = state.get("events", [])
    exchange_source = _exchange_execution_source(state.get("execution_mode", ""))
    exchange_mode = exchange_source is not None
    if exchange_mode:
        source_events = [event for event in source_events
                         if event.get("execution_source") == exchange_source
                         or event.get("event") in {"demo_partial_fill_reconciled",
                                                    "external_position_reconciled_flat"}]
    for event in source_events:
        if event.get("event") not in {"paper_fill", "demo_partial_fill_reconciled",
                                      "external_position_reconciled_flat",
                                      "manual_position_adopted"}:
            continue
        try:
            at = datetime.fromisoformat(event["at"].replace("Z", "+00:00"))
            if at.tzinfo is None:
                continue
            local_at = at.astimezone(SHANGHAI)
        except (KeyError, TypeError, ValueError):
            continue
        fills.append((at, event))
    fills.sort(key=lambda item: item[0])

    # Some historical Demo partial-fill OPEN records omitted ``symbol``. Resolve
    # those only when their order identifiers point to exactly one known symbol.
    symbols_by_order_id: dict[str, set[str]] = defaultdict(set)
    symbols_by_client_order_id: dict[str, set[str]] = defaultdict(set)
    for _, event in fills:
        symbol = event.get("symbol")
        if not symbol:
            continue
        order_id = event.get("orderId")
        client_order_id = event.get("clientOrderId")
        if order_id is not None:
            symbols_by_order_id[str(order_id)].add(symbol)
        if client_order_id:
            symbols_by_client_order_id[str(client_order_id)].add(symbol)

    active: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    for at, event in fills:
        symbol = event.get("symbol") or ""
        if not symbol:
            candidates = set()
            order_id = event.get("orderId")
            client_order_id = event.get("clientOrderId")
            if order_id is not None:
                candidates.update(symbols_by_order_id.get(str(order_id), ()))
            if client_order_id:
                candidates.update(symbols_by_client_order_id.get(str(client_order_id), ()))
            if len(candidates) == 1:
                symbol = next(iter(candidates))
        if (event.get("action") == "OPEN"
                or event.get("event") in {"demo_partial_fill_reconciled", "manual_position_adopted"}):
            active[symbol] = {
                "symbol": symbol, "side": event.get("side"),
                "quantity": event.get("filled_quantity", event.get("quantity")),
                "entry_price": (event.get("average") if event.get("event") == "demo_partial_fill_reconciled"
                                else event.get("price")),
                "opened_at": event.get("opened_at") or at.isoformat(),
                "entry_fee": (float(event["fee"]) if event.get("fee") is not None
                              else None if exchange_mode else 0.0),
                "entry_reason": event.get("reason", event.get("entry_reason")),
                "entry_regime": event.get("regime", event.get("entry_regime")),
                "strategy_sample_eligible": event.get("strategy_sample_eligible", True),
                "sample_exclusion_reason": event.get("sample_exclusion_reason"),
            }
        elif event.get("action") == "CLOSE":
            trade = active.pop(symbol, {
                "symbol": symbol, "side": event.get("side"), "quantity": event.get("quantity"),
                "entry_price": None, "opened_at": None, "entry_fee": 0.0,
            })
            trade.update({"closed_at": (event.get("closed_at", at.isoformat())
                                          if exchange_mode else at.isoformat()),
                          "exit_price": event.get("price"),
                          "realized_pnl": float(event["net_pnl"]) if event.get("net_pnl") is not None else None,
                          "fees": (float(event["fees"]) if event.get("fees") is not None
                                   else None if exchange_mode else 0.0), "exit_reason": event.get("reason"),
                          "entry_reason": event.get("entry_reason", trade.get("entry_reason")),
                          "entry_regime": event.get("entry_regime", trade.get("entry_regime")),
                          "exit_reason": _display_close_reason(event.get("reason")),
                          "held_hours": event.get("held_hours"), "mfe_r": event.get("mfe_r"),
                          "mfe_available": event.get("mfe_available"), "mae_r": event.get("mae_r"),
                          "mae_available": event.get("mae_available"),
                          "slippage_paid": event.get("slippage_paid"),
                          "status": "closed", "unrealized_pnl": None,
                          "pnl_recorded": event.get("net_pnl") is not None,
                          "strategy_sample_eligible": event.get(
                              "strategy_sample_eligible", trade.get("strategy_sample_eligible", True)),
                          "sample_exclusion_reason": event.get(
                              "sample_exclusion_reason", trade.get("sample_exclusion_reason"))})
            if exchange_mode:
                trade.update({
                    "exit_time_verified": event.get("exit_time_verified"),
                    "reconciled_at": event.get("reconciled_at"),
                    "elapsed_to_reconciliation_hours": event.get("elapsed_to_reconciliation_hours"),
                    "initial_stop_price": event.get("initial_stop_price"),
                    "initial_risk": event.get("initial_risk"),
                    "mfe_price": event.get("mfe_price"), "mae_price": event.get("mae_price"),
                    "observed_mfe_r": event.get("observed_mfe_r"),
                    "observed_mae_r": event.get("observed_mae_r"),
                    "mfe_to_reconciliation_r": event.get("mfe_to_reconciliation_r"),
                    "mae_to_reconciliation_r": event.get("mae_to_reconciliation_r"),
                    "mfe_to_reconciliation_price": event.get("mfe_to_reconciliation_price"),
                    "mae_to_reconciliation_price": event.get("mae_to_reconciliation_price"),
                    "excursion_source": event.get("excursion_source"),
                    "excursion_through": event.get("excursion_through"),
                    "excursion_history_complete": event.get("excursion_history_complete"),
                    "excursion_bar_count": event.get("excursion_bar_count"),
                    "excursion_sample_count": event.get("excursion_sample_count"),
                })
            trades.append(trade)
        elif event.get("event") == "external_position_reconciled_flat":
            trade = active.pop(event.get("symbol") or "", {
                "symbol": event.get("symbol"), "side": event.get("side"),
                "quantity": event.get("quantity"), "entry_price": event.get("entry_price"),
                "opened_at": event.get("opened_at"), "entry_fee": event.get("entry_fee"),
                "entry_reason": event.get("reason"),
            })
            recorded = bool(event.get("pnl_recorded")) and event.get("net_pnl") is not None
            stop_exit = event.get("reason") == "protective_stop_exchange_algo"
            trade.update({"closed_at": event.get("closed_at"),
                          "exit_price": event.get("close_price"),
                          "realized_pnl": float(event["net_pnl"]) if recorded else None,
                          "fees": event.get("fees"),
                          "exit_reason": ("机器人保护止损" if stop_exit else
                                          "外部平仓（成交价／盈亏未记账）"),
                          "status": "closed", "unrealized_pnl": None,
                          "pnl_recorded": recorded,
                          "external_order_ids": event.get("external_order_ids", []),
                          "strategy_sample_eligible": event.get(
                              "strategy_sample_eligible", trade.get("strategy_sample_eligible", True)),
                          "sample_exclusion_reason": event.get(
                              "sample_exclusion_reason", trade.get("sample_exclusion_reason"))})
            if not trade["strategy_sample_eligible"]:
                trade["exit_reason"] = f"{trade['exit_reason']}；已排除策略样本"
            trades.append(trade)
    live_by_symbol = {item.get("symbol"): item for item in (current_positions or [])}
    for symbol, trade in active.items():
        position = live_by_symbol.get(symbol, {})
        trade.update({"opened_at": position.get("opened_at", trade["opened_at"]),
                      "side": position.get("side", trade["side"]),
                      "quantity": position.get("quantity", trade["quantity"]),
                      "entry_price": position.get("entry_price", trade["entry_price"]),
                      "current_price": position.get("price"),
                      "unrealized_pnl": position.get("unrealized_pnl"),
                      "fees": trade["entry_fee"], "realized_pnl": None,
                      "exit_price": None, "closed_at": None, "exit_reason": None, "status": "open"})
        trades.append(trade)
    for symbol, position in live_by_symbol.items():
        if symbol not in active:
            opened_at = position.get("opened_at")
            trades.append({"symbol": symbol, "side": position.get("side"), "quantity": position.get("quantity"),
                           "entry_price": position.get("entry_price"), "opened_at": opened_at,
                           "current_price": position.get("price"), "unrealized_pnl": position.get("unrealized_pnl"),
                           "realized_pnl": None, "fees": None, "exit_price": None, "closed_at": None,
                           "exit_reason": None, "status": "open"})
    def trade_date(trade):
        raw = trade.get("opened_at") or trade.get("closed_at")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(SHANGHAI)
        except (AttributeError, TypeError, ValueError):
            return None
    filtered = []
    for trade in trades:
        local_at = trade_date(trade)
        if local_at is None or (lower and local_at < lower) or (upper and local_at > upper):
            continue
        filtered.append(trade)
    filtered.sort(key=lambda row: row.get("opened_at") or row.get("closed_at") or "", reverse=True)
    sample_trades = [row for row in filtered if row.get("strategy_sample_eligible", True)]
    closed = [row for row in sample_trades if row["status"] == "closed"]
    opened = [row for row in sample_trades if row["status"] == "open"]
    recorded_closed = [row for row in closed if row.get("realized_pnl") is not None]
    wins = sum(float(row["realized_pnl"]) > 0 for row in recorded_closed)
    losses = sum(float(row["realized_pnl"]) < 0 for row in recorded_closed)
    exits_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in recorded_closed:
        exits_by_reason[str(trade.get("exit_reason") or "unknown")].append(trade)
    exit_reason_stats = []
    for reason, reason_trades in exits_by_reason.items():
        reason_wins = sum(float(trade["realized_pnl"]) > 0 for trade in reason_trades)
        mfe_values = [float(trade["mfe_r"]) for trade in reason_trades if trade.get("mfe_r") is not None]
        mae_values = [float(trade["mae_r"]) for trade in reason_trades if trade.get("mae_r") is not None]
        held_values = [float(trade["held_hours"]) for trade in reason_trades
                       if trade.get("held_hours") is not None]
        net_values = [float(trade["realized_pnl"]) for trade in reason_trades]
        exit_reason_stats.append({
            "reason": reason, "trade_count": len(reason_trades), "wins": reason_wins,
            "losses": len(reason_trades) - reason_wins, "win_rate": reason_wins / len(reason_trades),
            "net_pnl": sum(net_values), "avg_net_pnl": sum(net_values) / len(net_values),
            "avg_mfe_r": sum(mfe_values) / len(mfe_values) if mfe_values else None,
            "avg_mae_r": sum(mae_values) / len(mae_values) if mae_values else None,
            "avg_held_hours": sum(held_values) / len(held_values) if held_values else None,
        })
    exit_reason_stats.sort(key=lambda row: (-row["trade_count"], row["reason"]))
    return {
        "start_date": start_date, "end_date": end_date, "timezone": "Asia/Shanghai",
        "trade_count": len(sample_trades), "open_trade_count": len(opened), "closed_trade_count": len(closed),
        "realized_pnl": sum(float(row["realized_pnl"]) for row in recorded_closed),
        "unrecorded_close_count": len(closed) - len(recorded_closed),
        "unrealized_pnl": sum(float(row.get("unrealized_pnl") or 0.0) for row in opened),
        "wins": wins, "losses": losses,
        "win_rate": wins / len(recorded_closed) if recorded_closed else None,
        "trades": filtered, "exit_reason_stats": exit_reason_stats,
    }


def build_snapshot(state: dict[str, Any], *, price: float | None, market_error: str | None = None,
                   state_updated_at: float | None = None) -> dict[str, Any]:
    if state.get("version") != 1:
        raise ValueError("unsupported paper state version")
    initial = float(state["initial_equity"])
    cash = float(state["cash"])
    position = state.get("position")
    if price is not None and (not math.isfinite(price) or price <= 0):
        raise ValueError("market price must be finite and positive")
    unrealized = 0.0
    if position and price is not None:
        direction = 1 if position["side"] == "long" else -1
        unrealized = float(position["quantity"]) * (price - float(position["entry_price"])) * direction
    equity = cash + unrealized + float((state.get("isolated_margin") or {}).get("initial_margin", 0.0))

    events = state.get("events", [])
    closes = [event for event in events if event.get("event") == "paper_fill" and event.get("action") == "CLOSE"]
    realized = sum(float(event.get("net_pnl", 0.0)) for event in closes)
    wins = sum(float(event.get("net_pnl", 0.0)) > 0 for event in closes)
    losses = sum(float(event.get("net_pnl", 0.0)) < 0 for event in closes)
    gross_wins = sum(max(0.0, float(event.get("net_pnl", 0.0))) for event in closes)
    gross_losses = -sum(min(0.0, float(event.get("net_pnl", 0.0))) for event in closes)
    equity_curve = [{"at": None, "equity": initial}]
    curve_equity = initial
    for event in closes:
        curve_equity += float(event.get("net_pnl", 0.0))
        equity_curve.append({"at": event.get("at"), "equity": round(curve_equity, 8)})

    start_equity = state.get("day_start_equity")
    daily_drawdown = None
    if start_equity is not None and float(start_equity) > 0:
        daily_drawdown = (float(start_equity) - equity) / float(start_equity)
    active_trade = {
        "symbol": state["symbol"],
        "side": position["side"],
        "quantity": float(position["quantity"]),
        "entry_price": float(position["entry_price"]),
        "stop_price": float(position["stop_price"]),
        "reason": position["reason"],
        "opened_at": position["opened_at"],
        "unrealized_pnl": unrealized,
        "stop_distance_pct": abs(price - float(position["stop_price"])) / price * 100 if price else None,
    } if position else None

    visible_events = []
    for event in reversed(events[-30:]):
        visible_events.append({key: event.get(key) for key in (
            "event", "at", "action", "side", "reason", "decision", "allowed", "quantity",
            "price", "stop_price", "net_pnl", "funding_paid", "state", "checked_at",
            "symbol", "timeframe", "evidence", "gross_notional_cap_pct", "mode",
        ) if key in event})
    file_updated = datetime.fromtimestamp(state_updated_at, timezone.utc).isoformat() if state_updated_at else None
    market_risk_status = next(({
        key: event.get(key) for key in ("mode", "state", "reason", "checked_at", "symbol", "timeframe", "evidence")
        if key in event
    } for event in reversed(events) if event.get("event") == "market_risk_observed"), {})
    return {
        "symbol": state["symbol"],
        "server_time": datetime.now(timezone.utc).isoformat(),
        "market_time": datetime.now(timezone.utc).isoformat() if price is not None else None,
        "state_updated_at": file_updated,
        "market_error": market_error,
        "price": price,
        "equity": equity,
        "initial_equity": initial,
        "cash": cash,
        "unrealized_pnl": unrealized,
        "realized_pnl": realized,
        "total_pnl": equity - initial,
        "daily_drawdown": daily_drawdown,
        "daily_halted": bool(state.get("daily_halted", False)),
        "position": active_trade,
        "trade_count": len(closes),
        "win_rate": wins / len(closes) if closes else None,
        "profit_factor": gross_wins / gross_losses if gross_losses else (None if not gross_wins else "∞"),
        "wins": wins,
        "losses": losses,
        "equity_curve": equity_curve,
        "events": visible_events,
        "positions": [active_trade] if active_trade else [],
        "watchlist": [],
        "market_risk_status": market_risk_status,
        "last_processed_close": state.get("last_processed_close"),
        "last_observed_open": state.get("last_observed_open"),
    }


def _position_time_observation(raw: dict[str, Any], exit_state: dict[str, Any], entry_cost: dict[str, Any],
                               price: float | None, now: datetime, settings) -> dict[str, Any]:
    opened_at = datetime.fromisoformat(raw["opened_at"])
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    held_hours = max(0.0, (now - opened_at.astimezone(timezone.utc)).total_seconds() / 3600)
    side_sign = 1 if raw["side"] == "long" else -1
    entry = float(raw["entry_price"])
    initial_risk = float(exit_state.get("initial_risk") or abs(entry - float(raw["stop_price"])))
    best_price = exit_state.get("best_price")
    managed = bool(exit_state.get("enabled")) and best_price is not None and initial_risk > 0
    current_r = (side_sign * (price - entry) / initial_risk
                 if price is not None and initial_risk > 0 else None)
    # The dashboard ticker refreshes faster than the runner persists extrema.
    # Include its latest price so the displayed MFE cannot trail current R.
    if managed and price is not None:
        best_price = max(float(best_price), price) if side_sign > 0 else min(float(best_price), price)
    mfe_r = (side_sign * (float(best_price) - entry) / initial_risk if managed else None)
    net_if_closed = None
    if price is not None:
        slip_rate = settings.fees.slippage_bps / 10_000
        fee_rate = settings.fees.taker_bps / 10_000
        exit_fill = float(price) * (1 + slip_rate if raw["side"] == "short" else 1 - slip_rate)
        net_if_closed = (
            float(raw["quantity"]) * (exit_fill - entry) * side_sign
            - float(entry_cost.get("entry_fee", 0.0))
            - float(raw["quantity"]) * exit_fill * fee_rate
            - float(entry_cost.get("funding_paid", 0.0))
        )

    policy = settings.exit_management

    def checkpoint(hours: float, progress_limit: float, *, requires_nonpositive_net: bool) -> dict[str, Any]:
        due = held_hours >= hours
        if not managed:
            status, met = "持仓极值历史不足", None
        elif not due:
            remaining = max(0, int((hours - held_hours) * 60))
            status, met = f"未到时点，约还需 {remaining // 60} 小时 {remaining % 60} 分", False
        else:
            progress_ok = mfe_r < progress_limit
            net_ok = net_if_closed is not None and net_if_closed <= 0
            met = progress_ok and (net_ok if requires_nonpositive_net else True)
            if met:
                status = "满足观察条件" + ("，估算净收益不大于零" if requires_nonpositive_net else "")
            elif requires_nonpositive_net and progress_ok and not net_ok:
                status = "已到时点，但估算平仓净收益为正"
            else:
                status = "已到时点，但 MFE 已超过阈值"
        return {"due": due, "condition_met": met, "status": status}

    return {
        "held_hours": held_hours,
        "current_r": current_r,
        "initial_risk": initial_risk if initial_risk > 0 else None,
        "mfe_r": mfe_r,
        "mfe_available": managed,
        "estimated_net_if_closed": net_if_closed,
        "time_exit_enabled": bool(policy.time_exit_enabled),
        "check_12h": checkpoint(policy.time_check_hours, policy.time_min_progress_r,
                                 requires_nonpositive_net=True),
        "check_24h": checkpoint(policy.time_exit_hours, policy.breakeven_trigger_r,
                                 requires_nonpositive_net=False),
    }


def _position_management_observation(position: dict[str, Any], watch: dict[str, Any] | None,
                                     settings) -> dict[str, Any]:
    """Read-only management cues; this function never changes paper orders or stops."""
    if position.get("stop_price") is None:
        return {
            "stop_net_if_closed": None, "protection_status": "未关联机器人保护止损",
            "hotlist_rank": (watch or {}).get("rank"), "hotlist_score": (watch or {}).get("hot_score"),
            "outside_hotlist": bool((watch or {}).get("held_outside_top")),
            "regime": (watch or {}).get("regime", "--"), "close_signal": False,
            "advice": ["交易所存在未关联本地订单记录的持仓，请先复核并同步或平仓"],
            "read_only": True,
        }
    side_sign = 1 if position["side"] == "long" else -1
    entry, stop = float(position["entry_price"]), float(position["stop_price"])
    observation = position["time_observation"]
    risk = observation.get("initial_risk")
    stop_cost = position.get("stop_net_if_closed")
    stop_status = "历史极值不足，保护状态无法评估"
    if position.get("profit_protection"):
        if stop_cost is None:
            stop_status = "已有动态保护止损，净保护效果待估"
        elif stop_cost > 0:
            stop_status = "保护止损按估算可锁定净利润"
        elif stop_cost == 0:
            stop_status = "保护止损约在估算盈亏平衡位"
        else:
            stop_status = "已有动态止损，但尚未覆盖估算成本"
    elif stop_cost is not None and stop_cost >= 0:
        stop_status = "固定止损按估算已覆盖成本"
    elif stop_cost is not None:
        stop_status = "固定止损仍可能产生净亏损"

    advice = []
    mfe = observation.get("mfe_r")
    if (observation.get("mfe_available") and observation["held_hours"] >= settings.exit_management.time_check_hours
            and mfe is not None and mfe >= settings.exit_management.breakeven_trigger_r
            and stop_cost is not None and stop_cost < 0):
        advice.append("已到时间检查且曾达到保本触发位，当前止损未覆盖估算成本，建议复核是否收紧")
    if (observation.get("mfe_available") and mfe is not None
            and mfe >= settings.exit_management.trailing_trigger_r and risk):
        trail_distance = (settings.exit_management.trailing_atr_mult * risk
                          / settings.signals.stop_atr_mult)
        best = entry + side_sign * mfe * risk
        suggested = best - side_sign * trail_distance
        if side_sign * (suggested - stop) > max(1e-12, risk * 1e-6):
            advice.append("已达到追踪触发位，账面保护止损落后于按当前参数估算的追踪位，请复核")

    regime = (watch or {}).get("regime", "--")
    status = (watch or {})
    close_signal = status.get("action") == "CLOSE"
    if close_signal:
        advice.append(f"策略已给出平仓信号：{status.get('signal_reason') or '原因未知'}")
    elif ((position["side"] == "long" and regime == "IMPULSE_UP")
          or (position["side"] == "short" and regime == "IMPULSE_DOWN")):
        advice.append(f"4 小时状态 {regime} 与当前仓位方向的退出规则相关，建议重点复核")

    outside_hotlist = bool((watch or {}).get("held_outside_top"))
    if outside_hotlist:
        advice.append("标的已离开当前热点前 20；这是当前排名提示，不代表已确认热度持续退潮")

    return {
        "stop_net_if_closed": stop_cost,
        "protection_status": stop_status,
        "hotlist_rank": (watch or {}).get("rank"),
        "hotlist_score": (watch or {}).get("hot_score"),
        "outside_hotlist": outside_hotlist,
        "regime": regime,
        "close_signal": close_signal,
        "advice": advice or ["目前未发现上述人工复核提示"],
        "read_only": True,
    }


def build_multi_snapshot(state: dict[str, Any], *, tickers: dict[str, dict], market_error: str | None = None,
                         state_updated_at: float | None = None, settings=None,
                         now: datetime | None = None) -> dict[str, Any]:
    if state.get("version") != 2 or state.get("account_type") != "multi_symbol_paper":
        raise ValueError("unsupported multi-symbol paper state")
    settings = settings or load_settings()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    initial, cash = float(state["initial_equity"]), float(state["cash"])
    events = state.get("events", [])
    mark_prices = state.get("mark_prices", {})
    exit_states = state.get("exit_management_by_symbol", {})
    margin_states = state.get("isolated_margin_by_symbol", {})
    exchange_source = _exchange_execution_source(state.get("execution_mode", ""))
    exchange_mode = exchange_source is not None
    lead_mode = state.get("execution_mode") == "lead_usdm"
    if exchange_mode:
        events = [event for event in events if event.get("execution_source") == exchange_source
                  or event.get("event") == "external_position_reconciled_flat"]
    exchange_snapshot = state.get("exchange_snapshot", {}) if exchange_mode else {}
    exchange_positions = exchange_snapshot.get("positions", {})
    if exchange_mode:
        display_positions = {}
        for symbol, live in exchange_positions.items():
            local = state.get("positions", {}).get(symbol, {})
            display_positions[symbol] = {
                **local, "symbol": symbol, "side": live["side"], "quantity": live["quantity"],
                "entry_price": live["entry_price"], "exchange_unrealized_pnl": live["unrealized_pnl"],
                "margin_mode": live.get("margin_mode"), "exchange_leverage": live.get("leverage"),
                "exchange_initial_margin": live.get("initial_margin"),
                "reason": local.get("reason", "external_exchange_position"),
                "regime": local.get("regime", "EXTERNAL"),
                "opened_at": local.get("opened_at"),
            }
    else:
        display_positions = state.get("positions", {})
    positions = []
    unrealized = 0.0
    for symbol, raw in display_positions.items():
        exit_state = exit_states.get(symbol) or {}
        raw_stop = exit_state.get("protective_stop", raw.get("stop_price"))
        stop_price = float(raw_stop) if raw_stop is not None else None
        ticker = tickers.get(symbol) or {}
        raw_price = ticker.get("last") or ticker.get("close")
        price = float(raw_price) if raw_price is not None else mark_prices.get(symbol)
        pnl = float(raw["exchange_unrealized_pnl"]) if "exchange_unrealized_pnl" in raw else 0.0
        if "exchange_unrealized_pnl" not in raw and price is not None:
            direction = 1 if raw["side"] == "long" else -1
            pnl = float(raw["quantity"]) * (float(price) - float(raw["entry_price"])) * direction
        unrealized += pnl
        margin = margin_states.get(symbol, {})
        qty, entry = float(raw["quantity"]), float(raw["entry_price"])
        mmr = float(margin.get("maintenance_margin_rate", 0.0))
        margin_value = (raw.get("exchange_initial_margin") if exchange_mode
                        else margin.get("initial_margin"))
        allocated = float(margin_value) if margin_value is not None else 0.0
        maintenance_amount = float(margin.get("maintenance_amount", 0.0))
        liq = None
        denominator = qty * (1 - mmr) if raw["side"] == "long" else qty * (1 + mmr)
        if allocated and denominator > 0:
            liq = (qty * entry - allocated - maintenance_amount) / denominator if raw["side"] == "long" else (qty * entry + allocated + maintenance_amount) / denominator
        maintenance = max(0.0, qty * float(price) * mmr - maintenance_amount) if price is not None else None
        tail_exempt = runner_tail_is_exempt(
            raw["side"], float(raw["entry_price"]), raw.get("stop_price"), exit_state, settings.exits,
        )
        max_hold_hours = effective_max_hold_hours(
            raw["side"], float(raw["entry_price"]), raw.get("stop_price"), exit_state, settings.exits,
        )
        positions.append({
            "symbol": symbol, "side": raw["side"], "quantity": float(raw["quantity"]),
            "entry_price": float(raw["entry_price"]), "stop_price": stop_price,
            "profit_protection": bool(exit_state.get("enabled", False)),
            "reason": raw["reason"], "opened_at": raw["opened_at"], "price": price,
            "unrealized_pnl": pnl,
            "price_source": "Binance 最新成交价" if raw_price is not None else "最近可用行情",
            "pnl_source": ("Lead Portfolio 账户快照" if lead_mode else "Demo 账户快照")
                          if exchange_mode else "本机行情估算",
            "pnl_updated_at": exchange_snapshot.get("at") if exchange_mode else now.isoformat(),
            "margin_mode": raw.get("margin_mode") or ("isolated" if margin_value is not None else "legacy"),
            "isolated_margin": allocated if margin_value is not None else None,
            "margin_source": ("Binance Lead Portfolio 快照" if lead_mode else "Binance Demo 交易所快照")
                             if exchange_mode else "本地 Paper 账本",
            "margin_updated_at": exchange_snapshot.get("at") if exchange_mode else now.isoformat(),
            "leverage": raw.get("exchange_leverage", margin.get("leverage")), "maintenance_margin": maintenance,
            "liquidation_price_estimate": liq,
            "stop_distance_pct": abs(float(price) - stop_price) / float(price) * 100
            if price is not None and float(price) > 0 and stop_price is not None else None,
            "time_observation": (_position_time_observation(
                raw, exit_state, state.get("entry_costs", {}).get(symbol, {}), price, now, settings
            ) if raw.get("opened_at") and (stop_price is not None or exit_state.get("initial_risk")) else {
                "held_hours": None, "current_r": None, "initial_risk": None, "mfe_r": None,
                "mfe_available": False, "estimated_net_if_closed": None, "time_exit_enabled": False,
                "check_12h": {"due": False, "condition_met": None, "status": "未关联本地持仓记录"},
                "check_24h": {"due": False, "condition_met": None, "status": "未关联本地持仓记录"},
            }),
            "exit_management": {
                "policy": exit_state.get("policy") or ("legacy" if exit_state else "untracked"),
                "tp1_taken": bool(exit_state.get("tp1_taken", False)),
                "tp2_taken": bool(exit_state.get("tp2_taken", False)),
                "breakeven_moved": bool(exit_state.get("breakeven_moved", False)),
                "no_progress_count": exit_state.get("no_progress_count"),
                "no_progress_bars": settings.exits.no_progress_bars,
                "max_hold_hours": max_hold_hours,
                "max_hold_exempt": tail_exempt,
            },
        })
        position = positions[-1]
        observation = position["time_observation"]
        held_hours = observation.get("held_hours")
        position["exit_management"]["max_hold_due"] = (
            held_hours is not None and max_hold_hours is not None
            and max_hold_hours > 0 and held_hours >= max_hold_hours
        )
        if position["price"] is not None and stop_price is not None:
            stop_slip = settings.fees.slippage_bps / 10_000
            stop_fee = settings.fees.taker_bps / 10_000
            stop_fill = stop_price * (1 + stop_slip if raw["side"] == "short" else 1 - stop_slip)
            entry_cost = state.get("entry_costs", {}).get(symbol, {})
            position["stop_net_if_closed"] = (
                qty * (stop_fill - entry) * (1 if raw["side"] == "long" else -1)
                - float(entry_cost.get("entry_fee", 0.0))
                - qty * stop_fill * stop_fee
                - float(entry_cost.get("funding_paid", 0.0))
            )
        else:
            position["stop_net_if_closed"] = None
    equity = (float(exchange_snapshot.get("equity")) if exchange_mode and exchange_snapshot.get("equity") is not None
              else cash + unrealized + sum(float(item.get("initial_margin", 0.0)) for item in margin_states.values()))
    closes = [event for event in events if event.get("event") == "paper_fill" and event.get("action") == "CLOSE"
              and event.get("strategy_sample_eligible", True)
              and event.get("net_pnl") is not None]
    external_closes = [event for event in events if exchange_mode
                       and event.get("event") == "external_position_reconciled_flat"
                       and event.get("strategy_sample_eligible", True)]
    recorded_external_closes = [event for event in external_closes
                                if event.get("pnl_recorded") and event.get("net_pnl") is not None]
    recorded_closes = closes + recorded_external_closes
    realized = sum(float(event["net_pnl"]) for event in recorded_closes)
    wins = sum(float(event["net_pnl"]) > 0 for event in recorded_closes)
    losses = sum(float(event["net_pnl"]) < 0 for event in recorded_closes)
    positive = sum(max(0.0, float(event["net_pnl"])) for event in recorded_closes)
    negative = -sum(min(0.0, float(event["net_pnl"])) for event in recorded_closes)
    curve = [{"at": None, "equity": initial}]
    curve_value = initial
    for event in sorted(recorded_closes, key=lambda item: item.get("closed_at") or item.get("at") or ""):
        curve_value += float(event.get("net_pnl", 0.0))
        curve.append({"at": event.get("closed_at") or event.get("at"), "equity": round(curve_value, 8)})
    day_start = state.get("day_start_equity")
    drawdown = (float(day_start) - equity) / float(day_start) if day_start and float(day_start) > 0 else None

    watchlist = []
    for row in state.get("watchlist", []):
        symbol = row["symbol"]
        ticker = tickers.get(symbol) or {}
        price = ticker.get("last") or ticker.get("close") or mark_prices.get(symbol)
        status = state.get("symbol_status", {}).get(symbol, {})
        watchlist.append({
            **row, "price": float(price) if price is not None else None,
            "quote_volume_24h": float(ticker.get("quoteVolume") or row.get("quote_volume_24h") or 0),
            "regime": status.get("regime", "--"), "action": status.get("action", "NONE"),
            "side": status.get("side"), "signal_reason": status.get("reason", "--"),
            "signal_at": status.get("at"),
            "position": next((item for item in positions if item["symbol"] == symbol), None),
        })
    watch_by_symbol = {row["symbol"]: row for row in watchlist}
    for position in positions:
        position["management_observation"] = _position_management_observation(
            position, watch_by_symbol.get(position["symbol"]), settings
        )
    visible_events = []
    for event in reversed(events[-30:]):
        visible_events.append({key: event.get(key) for key in (
            "event", "at", "symbol", "action", "side", "reason", "decision", "allowed", "quantity",
            "direction", "source", "signal_source", "candidate_source", "signal_timestamp",
            "candidate_action", "candidate_side", "candidate_reason", "result_action", "result_side",
            "regime_state", "market_risk_state", "route_reason", "entry_allowed", "entry_block_reason",
            "price", "average", "amount", "execution_price", "stop_price", "net_pnl", "funding_paid", "orderId", "host",
            "execution_source", "warning", "external_order_ids", "pnl_recorded", "signal_time",
            "confirmed_at", "filled_at", "entry_delay_sec", "signal_to_fill_delay_sec",
            "quote_at", "quote_received_at", "quote_age_sec", "quote_price", "quote_source",
            "reference_price", "pullback_price", "state", "gross_notional_cap_pct", "checked_at", "anchors",
            "expires_at", "detail",
        ) if key in event})
    pending_entries = []
    for symbol, pending in state.get("pending_by_symbol", {}).items():
        if pending.get("execution_type") == "pullback":
            pending_entries.append({
                "symbol": symbol, "side": pending.get("side"),
                "signal_time": pending.get("signal_time"), "confirmed_at": pending.get("confirmed_at"),
                "reference_price": pending.get("reference_price"),
                "pullback_price": pending.get("pullback_price"), "expires_at": pending.get("expires_at"),
            })
    updated = datetime.fromtimestamp(state_updated_at, timezone.utc).isoformat() if state_updated_at else None
    return {
        "symbol": "Binance USDT-M · 热点币", "server_time": now.isoformat(),
        "market_time": now.isoformat() if not market_error else None,
        "state_updated_at": updated, "market_error": market_error, "price": None,
        "time_exit_policy": {
            "enabled": bool(settings.exit_management.time_exit_enabled),
            "check_hours": settings.exit_management.time_check_hours,
            "min_progress_r": settings.exit_management.time_min_progress_r,
            "exit_hours": settings.exit_management.time_exit_hours,
            "protected_progress_r": settings.exit_management.breakeven_trigger_r,
            "max_hold_bars": settings.exits.max_hold_bars,
        },
        "execution_mode": state.get("execution_mode", "local_paper"),
        "execution_status": state.get("execution_status", "local"),
        "execution_host": exchange_snapshot.get("host"),
        "exchange_sync_at": exchange_snapshot.get("at"),
        "protective_stop_audit": state.get("protective_stop_audit", {}),
        "inconsistencies": state.get("inconsistencies", []),
        "foreign_order_ids": state.get("foreign_order_ids", []),
        "equity": equity, "initial_equity": initial, "cash": cash,
        "unrealized_pnl": unrealized, "realized_pnl": realized, "total_pnl": equity - initial,
        "daily_drawdown": drawdown, "daily_halted": bool(state.get("daily_halted", False)),
        "position": positions[0] if len(positions) == 1 else None, "positions": positions,
        "watchlist": watchlist, "pending_entries": pending_entries,
        "market_risk_status": state.get("market_risk_status", {}),
        "strategy_variant": state.get("strategy_variant", "immediate_baseline"),
        "trade_count": len(closes) + len(external_closes),
        "unrecorded_close_count": len(external_closes) - len(recorded_external_closes),
        "win_rate": wins / len(recorded_closes) if recorded_closes else None,
        "profit_factor": positive / negative if negative else (None if not positive else "∞"),
        "wins": wins, "losses": losses, "equity_curve": curve, "events": visible_events,
        "last_processed_close": max(state.get("last_processed_by_symbol", {}).values(), default=None),
        "last_observed_open": max(state.get("last_observed_by_symbol", {}).values(), default=None),
    }


class DashboardService:
    def __init__(self, state_path: str | Path, symbol: str | None, exchange, settings=None):
        self.state_path = Path(state_path)
        self.symbol = symbol
        self.exchange = exchange
        self.settings = settings or load_settings()
        self.manual_close_store = ManualCloseCommandStore(command_directory_for_state(self.state_path))
        self._cache_lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None
        self._display_mfe_prices: dict[str, tuple[str, str, float]] = {}
        self._equity_history: list[dict[str, Any]] = []
        self._last_equity_history_key: str | None = None
        self._hotlist_signature: tuple | None = None
        self._hotlist_observed_at: str | None = None

    def _merge_display_mfe(self, state: dict[str, Any], tickers: dict[str, dict]) -> None:
        """Keep dashboard-observed extrema between slower runner ledger updates."""
        exit_states = state.get("exit_management_by_symbol", {})
        for symbol, position in state.get("positions", {}).items():
            exit_state = exit_states.get(symbol) or {}
            best = exit_state.get("best_price")
            if not exit_state.get("enabled") or best is None:
                continue
            side = position["side"]
            identity = (position.get("opened_at", ""), side)
            best = float(best)
            remembered = self._display_mfe_prices.get(symbol)
            if remembered and remembered[:2] == identity:
                best = max(best, remembered[2]) if side == "long" else min(best, remembered[2])
            ticker = tickers.get(symbol) or {}
            raw_price = ticker.get("last") or ticker.get("close")
            if raw_price is not None:
                price = float(raw_price)
                if math.isfinite(price) and price > 0:
                    best = max(best, price) if side == "long" else min(best, price)
            exit_state["best_price"] = best
            exit_states[symbol] = exit_state
            self._display_mfe_prices[symbol] = (*identity, best)

    def snapshot(self) -> dict[str, Any]:
        with self._cache_lock:
            if self._cached is not None and time.monotonic() - self._cached_at < 3:
                return self._cached
            if not self.state_path.exists():
                raise FileNotFoundError(f"paper state not found: {self.state_path}")
            state_stat = self.state_path.stat()
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") == 2:
                try:
                    tickers = self.exchange.fetch_tickers()
                    error = None
                except Exception as exc:  # keep ledger visible during transient market-data outages
                    tickers = {}
                    error = f"public tickers unavailable: {type(exc).__name__}"
                self._merge_display_mfe(state, tickers)
                self._cached = build_multi_snapshot(state, tickers=tickers, market_error=error,
                                                    state_updated_at=state_stat.st_mtime, settings=self.settings)
                history_key = self._cached.get("exchange_sync_at") or self._cached.get("state_updated_at")
                if history_key and history_key != self._last_equity_history_key:
                    self._equity_history.append({"at": history_key, "equity": float(self._cached["equity"])})
                    self._equity_history = self._equity_history[-1440:]
                    self._last_equity_history_key = history_key
                signature = tuple((row.get("symbol"), row.get("rank"), row.get("hot_score"))
                                  for row in state.get("watchlist", []))
                if signature != self._hotlist_signature:
                    self._hotlist_signature = signature
                    self._hotlist_observed_at = datetime.now(timezone.utc).isoformat()
                self._cached["equity_history"] = list(self._equity_history)
                self._cached["watchlist_updated_at"] = self._hotlist_observed_at
            else:
                if state.get("symbol") != self.symbol:
                    raise ValueError("dashboard symbol does not match the paper state")
                price = None
                error = None
                try:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    raw_price = ticker.get("last") or ticker.get("close")
                    price = float(raw_price) if raw_price is not None else None
                    if price is None:
                        error = "public ticker returned no last price"
                except Exception as exc:  # keep ledger visible during transient market-data outages
                    error = f"public ticker unavailable: {type(exc).__name__}"
                self._cached = build_snapshot(state, price=price, market_error=error,
                                              state_updated_at=state_stat.st_mtime)
            self._cached_at = time.monotonic()
            self._cached["manual_close_enabled"] = (
                state.get("version") == 2
                and state.get("account_type") == "multi_symbol_paper"
                and state.get("execution_mode") != "lead_usdm"
            )
            return self._cached

    def orders(self, *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(f"paper state not found: {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        live_snapshot = self.snapshot()
        return build_order_snapshot(state, start_date=start_date, end_date=end_date,
                                    current_positions=live_snapshot.get("positions", []))

    def request_manual_close(self, symbol: str) -> dict[str, str]:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("version") != 2 or state.get("account_type") != "multi_symbol_paper":
            raise ValueError("manual close is available only for multi-symbol paper mode")
        if state.get("execution_mode") == "lead_usdm":
            raise ValueError("manual close through the Paper dashboard is unavailable for Lead Portfolio mode")
        exchange_positions = (state.get("exchange_snapshot", {}).get("positions", {})
                              if _exchange_execution_source(state.get("execution_mode", "")) else {})
        if symbol not in state.get("positions", {}) and symbol not in exchange_positions:
            raise LookupError("the selected symbol has no open paper position")
        request = self.manual_close_store.submit_unique(symbol)
        if request is None:
            raise FileExistsError("a manual close request is already pending for this symbol")
        return {"request_id": request["request_id"], "status": "queued", "symbol": symbol}

    def manual_close_status(self, request_id: str) -> dict[str, Any] | None:
        return self.manual_close_store.status(request_id)


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service: DashboardService, html_path: Path):
        self.service = service
        self.html_path = html_path
        super().__init__(address, _DashboardHandler)


class _DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer

    def do_GET(self):
        host_header = self.headers.get("Host", "").split(":", maxsplit=1)[0].strip("[]").lower()
        if host_header not in {"127.0.0.1", "localhost"}:
            self.send_error(403)
            return
        route = urlparse(self.path).path
        if route == "/api/orders":
            query = parse_qs(urlparse(self.path).query, strict_parsing=False)
            if set(query) - {"start", "end"} or any(len(values) != 1 for values in query.values()):
                self.send_error(400, "use optional start and end dates in YYYY-MM-DD format")
                return
            start_date = query.get("start", [None])[0]
            end_date = query.get("end", [None])[0]
            try:
                payload = json.dumps(self.server.service.orders(start_date=start_date, end_date=end_date),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400 if isinstance(exc, ValueError) else 503, str(exc))
                return
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/api/state":
            try:
                payload = json.dumps(self.server.service.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(503, str(exc))
                return
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route.startswith("/api/manual-close/"):
            request_id = route.removeprefix("/api/manual-close/")
            if not request_id or "/" in request_id:
                self.send_error(404)
                return
            try:
                result = self.server.service.manual_close_status(request_id)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            if result is None:
                self.send_error(404, "manual close request not found")
                return
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route in ("/", "/index.html"):
            try:
                payload = self.server.html_path.read_bytes()
            except OSError:
                self.send_error(500, "dashboard asset is unavailable")
                return
            self._send(200, "text/html; charset=utf-8", payload)
            return
        self.send_error(404)

    def do_POST(self):
        host_header = self.headers.get("Host", "").lower()
        host_name = host_header.split(":", maxsplit=1)[0].strip("[]")
        if host_name not in {"127.0.0.1", "localhost"}:
            self.send_error(403)
            return
        origin = self.headers.get("Origin", "")
        if origin != f"http://{host_header}" or self.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            self.send_error(403, "same-origin request required")
            return
        if urlparse(self.path).path != "/api/manual-close":
            self.send_error(404)
            return
        if self.headers.get("Content-Type", "").split(";", maxsplit=1)[0].strip().lower() != "application/json":
            self.send_error(415, "application/json required")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        if not 1 <= content_length <= 1024:
            self.send_error(413 if content_length > 1024 else 400, "invalid request size")
            return
        try:
            body = json.loads(self.rfile.read(content_length))
            if not isinstance(body, dict) or set(body) != {"symbol"} or not isinstance(body["symbol"], str):
                raise ValueError("body must contain only a string symbol")
            result = self.server.service.request_manual_close(body["symbol"])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.send_error(400, str(exc))
            return
        except LookupError as exc:
            self.send_error(409, str(exc))
            return
        except FileExistsError as exc:
            self.send_error(409, str(exc))
            return
        except (OSError, TypeError) as exc:
            self.send_error(503, str(exc))
            return
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(202, "application/json; charset=utf-8", payload)

    def _send(self, status: int, content_type: str, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def serve_dashboard(state_path: str | Path, symbol: str | None, exchange, *, host: str = "127.0.0.1",
                    port: int = 8765, settings=None):
    if host != "127.0.0.1":
        raise ValueError("dashboard must bind to loopback only")
    html_path = Path(__file__).with_name("dashboard.html")
    return _DashboardHTTPServer((host, port), DashboardService(state_path, symbol, exchange, settings), html_path)
