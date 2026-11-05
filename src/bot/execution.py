"""Execution adapter boundary: strategy intents become exchange orders here."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any, Callable
from uuid import uuid4

from loguru import logger

from bot.exchange.binance_usdm import DEMO_REST_ROOT, parse_live_position, verify_demo_urls
from bot.signals import Action, Side


BOT_ORDER_PREFIX = "mrb"


class EntryPreflightRejected(RuntimeError):
    """A candidate was rejected before an opening order was submitted."""


def _utc_now():
    return datetime.now(timezone.utc)


def _number(value, default=0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


class DemoExecutionService:
    """Signed USDⓈ-M Demo orders and account snapshots; never selects production URLs."""

    def __init__(self, exchange, settings):
        verify_demo_urls(exchange)
        self.exchange = exchange
        self.settings = settings
        self.host = DEMO_REST_ROOT
        self._last_fill_audit_at = 0.0

    def initialize(self) -> dict[str, Any]:
        self._ensure_one_way()
        snapshot = self.sync_snapshot()
        logger.info("Binance USDⓈ-M Demo account check: TESTNET=true BOT_MODE=paper host={} USDT={} position_mode=one-way",
                    self.host, snapshot["usdt_balance"])
        if not math.isfinite(float(snapshot["usdt_balance"])) or float(snapshot["usdt_balance"]) <= 0:
            raise RuntimeError("USDⓈ-M Demo account has no positive USDT wallet balance; refusing execution")
        return snapshot

    def _ensure_one_way(self) -> None:
        verify_demo_urls(self.exchange)
        mode = self.exchange.fetch_position_mode()
        if bool(mode.get("hedged")):
            self.exchange.set_position_mode(False)
            mode = self.exchange.fetch_position_mode()
        if bool(mode.get("hedged")):
            raise RuntimeError("Binance account is not in one-way position mode")

    def sync_snapshot(self) -> dict[str, Any]:
        verify_demo_urls(self.exchange)
        balance = self.exchange.fetch_balance()
        positions = {}
        for row in self.exchange.fetch_positions():
            normalized = parse_live_position(row)
            if normalized is not None:
                normalized["symbol"] = normalized["symbol"] or row.get("symbol")
                positions[normalized["symbol"]] = normalized
        usdt = balance.get("USDT", {})
        info = balance.get("info", {})
        total = _number(usdt.get("total"), _number(info.get("totalWalletBalance")))
        free = _number(usdt.get("free"), _number(info.get("availableBalance")))
        unrealized = sum(item["unrealized_pnl"] for item in positions.values())
        equity = _number(info.get("totalMarginBalance"), total + unrealized)
        return {
            "at": datetime.now(timezone.utc).isoformat(),
            "host": self.host,
            "usdt_balance": total,
            "available_usdt": free,
            "unrealized_pnl": unrealized,
            "equity": equity,
            "position_mode": "one-way",
            "positions": positions,
        }

    def _market_rules(self, symbol: str) -> tuple[float, float, float]:
        market = self.exchange.market(symbol)
        info = market.get("info", {})
        step, market_step = 0.0, 0.0
        minimum = _number((market.get("limits") or {}).get("amount", {}).get("min"))
        min_cost = _number((market.get("limits") or {}).get("cost", {}).get("min"))
        for rule in info.get("filters", []):
            if rule.get("filterType") == "LOT_SIZE":
                step = _number(rule.get("stepSize"), step)
                minimum = max(minimum, _number(rule.get("minQty")))
            elif rule.get("filterType") == "MARKET_LOT_SIZE":
                market_step = _number(rule.get("stepSize"), market_step)
                minimum = max(minimum, _number(rule.get("minQty")))
            elif rule.get("filterType") in {"MIN_NOTIONAL", "NOTIONAL"}:
                min_cost = max(min_cost, _number(rule.get("notional", rule.get("minNotional"))))
        return market_step or step, minimum, min_cost

    def _prepare_symbol(self, symbol: str) -> None:
        verify_demo_urls(self.exchange)
        try:
            self.exchange.set_margin_mode("isolated", symbol)
        except Exception as exc:
            if "No need to change margin type" not in str(exc) and "already isolated" not in str(exc).lower():
                raise
        leverage = int(self.settings.sizing.leverage)
        result = self.exchange.set_leverage(leverage, symbol)
        actual = (result or {}).get("leverage")
        if actual is not None and int(actual) != leverage:
            raise RuntimeError("Binance did not confirm the configured leverage")

    def _market_fill(self, symbol: str, side: str, quantity: float, *, reduce_only: bool,
                     client_order_id: str | None = None) -> dict[str, Any]:
        verify_demo_urls(self.exchange)
        if quantity <= 0 or not math.isfinite(quantity):
            raise ValueError("exchange order quantity must be finite and positive")
        client_id = client_order_id or f"{BOT_ORDER_PREFIX}_{uuid4().hex[:28]}"
        params = {"clientOrderId": client_id, "newOrderRespType": "RESULT"}
        if reduce_only:
            params["reduceOnly"] = True
        order = self.exchange.create_order(symbol, "market", side, quantity, None, params)
        order_id = order.get("id") or (order.get("info") or {}).get("orderId")
        if not order_id:
            raise RuntimeError("Demo exchange returned no orderId; reconciliation required")
        filled = _number(order.get("filled"), _number((order.get("info") or {}).get("executedQty")))
        average = _number(order.get("average"), _number((order.get("info") or {}).get("avgPrice")))
        if not filled or not average:
            order = self.exchange.fetch_order(order_id, symbol)
            filled = _number(order.get("filled"), _number((order.get("info") or {}).get("executedQty")))
            average = _number(order.get("average"), _number((order.get("info") or {}).get("avgPrice")))
        status = order.get("status")
        status = status.lower() if isinstance(status, str) else status
        terminal_statuses = {None, "closed", "canceled", "cancelled", "expired"}
        if filled <= 0 or average <= 0 or status not in terminal_statuses:
            raise RuntimeError(f"Demo market order {order_id} was not confirmed filled")
        fill_data = {}
        try:
            market_id = self.exchange.market(symbol)["id"]
            trades = self.exchange.fapiPrivateGetUserTrades({
                "symbol": market_id, "orderId": str(order_id), "limit": 1000,
            }) or []
            matched = [row for row in trades if str(row.get("orderId")) == str(order_id)]
            filled_quantity = sum(_number(row.get("qty")) for row in matched)
            step, _, _ = self._market_rules(symbol)
            if (matched and abs(filled_quantity - filled) <= max(step, filled * 1e-8)):
                fill_data["trade_data_verified"] = True
                if all(row.get("commission") is not None and row.get("commissionAsset") == "USDT"
                       for row in matched):
                    fill_data["fee_cost"] = sum(_number(row["commission"]) for row in matched)
                if all(row.get("realizedPnl") is not None for row in matched):
                    fill_data["realized_pnl"] = sum(_number(row["realizedPnl"]) for row in matched)
                timestamps = [int(row.get("time") or 0) for row in matched]
                if timestamps and all(value > 0 for value in timestamps):
                    fill_data["closed_at"] = datetime.fromtimestamp(
                        max(timestamps) / 1000, timezone.utc,
                    ).isoformat()
        except Exception:
            pass
        return {"id": str(order_id), "clientOrderId": client_id, "filled": filled,
                "status": status, "average": average, "side": side,
                **fill_data, "info": order.get("info", {})}

    def funding_fee(self, symbol: str, opened_at: datetime, closed_at: datetime) -> float | None:
        """Return verified USDT funding income for one completed Demo position."""
        if opened_at.tzinfo is None or closed_at.tzinfo is None or closed_at < opened_at:
            return None
        try:
            verify_demo_urls(self.exchange)
            market_id = self.exchange.market(symbol)["id"]
            start_ms = int(opened_at.timestamp() * 1000)
            end_ms = int(closed_at.timestamp() * 1000)
            rows = self.exchange.fapiPrivateGetIncome({
                "symbol": market_id,
                "incomeType": "FUNDING_FEE",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            }) or []
            if len(rows) >= 1000:
                return None
            if any(row.get("asset") != "USDT" or row.get("income") is None
                   or row.get("incomeType") != "FUNDING_FEE"
                   or row.get("time") is None
                   or not start_ms <= int(row["time"]) <= end_ms for row in rows):
                return None
            return sum(_number(row["income"]) for row in rows)
        except Exception:
            return None

    def funding_history(self, started_at: datetime, ended_at: datetime) -> list[dict] | None:
        """Read a bounded USDT funding interval for delayed settlement reconciliation."""
        if (started_at.tzinfo is None or ended_at.tzinfo is None
                or ended_at < started_at):
            return None
        try:
            verify_demo_urls(self.exchange)
            start_ms, end_ms = int(started_at.timestamp() * 1000), int(ended_at.timestamp() * 1000)
            rows = self.exchange.fapiPrivateGetIncome({
                "incomeType": "FUNDING_FEE", "startTime": start_ms, "endTime": end_ms,
                "limit": 1000,
            }) or []
            if len(rows) >= 1000:
                return None
            result = []
            for row in rows:
                amount, at = float(row["income"]), int(row["time"])
                if (row.get("asset") != "USDT" or row.get("incomeType") != "FUNDING_FEE"
                        or not row.get("symbol") or not math.isfinite(amount)
                        or not start_ms <= at <= end_ms):
                    return None
                result.append({"symbol": row["symbol"], "time": at, "income": amount})
            return result
        except Exception:
            return None

    def resolve_entry_fee(self, symbol: str, side: Side, quantity: float,
                          entry_price: float, opened_at: datetime) -> float | None:
        """Recover an older Demo entry fee only when one matching order is unambiguous."""
        if opened_at.tzinfo is None:
            return None
        try:
            verify_demo_urls(self.exchange)
            market_id = self.exchange.market(symbol)["id"]
            start_ms = int((opened_at.timestamp() - 10) * 1000)
            end_ms = int((opened_at.timestamp() + 180) * 1000)
            rows = self.exchange.fapiPrivateGetUserTrades({
                "symbol": market_id, "startTime": start_ms, "endTime": end_ms, "limit": 1000,
            }) or []
            step, _, _ = self._market_rules(symbol)
            expected_side = "BUY" if side is Side.LONG else "SELL"
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                order_id = row.get("orderId")
                if (not order_id or str(row.get("side", "")).upper() != expected_side
                        or row.get("time") is None
                        or not start_ms <= int(row["time"]) <= end_ms):
                    continue
                grouped.setdefault(str(order_id), []).append(row)
            matches = []
            for trades in grouped.values():
                filled = sum(_number(row.get("qty")) for row in trades)
                notional = sum(_number(row.get("qty")) * _number(row.get("price")) for row in trades)
                average = notional / filled if filled > 0 else 0.0
                if (abs(filled - quantity) > max(step, quantity * 1e-8)
                        or abs(average - entry_price) > max(entry_price * 1e-6, 1e-12)):
                    continue
                if not all(row.get("commission") is not None and row.get("commissionAsset") == "USDT"
                           for row in trades):
                    continue
                matches.append(sum(_number(row["commission"]) for row in trades))
            return matches[0] if len(matches) == 1 else None
        except Exception:
            return None

    def _position(self, symbol: str) -> dict[str, Any] | None:
        verify_demo_urls(self.exchange)
        rows = self.exchange.fetch_positions([symbol])
        return next((parsed for row in rows if (parsed := parse_live_position(row)) is not None), None)

    def _place_stop(self, symbol: str, side: str, amount: float, stop_price: float) -> dict[str, Any]:
        if not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError("a valid protective stop is required before order submission")
        verify_demo_urls(self.exchange)
        close_side = "sell" if side == "long" else "buy"
        amount = float(self.exchange.amount_to_precision(symbol, amount))
        stop_price = float(self.exchange.price_to_precision(symbol, stop_price))
        client_id = f"{BOT_ORDER_PREFIX}_s_{uuid4().hex[:24]}"
        order = self.exchange.create_order(
            symbol, "STOP_MARKET", close_side, amount, None,
            {"stopPrice": stop_price, "reduceOnly": True, "workingType": "CONTRACT_PRICE",
             "clientOrderId": client_id},
        )
        order_id = order.get("id") or (order.get("info") or {}).get("algoId")
        if not order_id:
            raise RuntimeError("Demo protective stop returned no orderId")
        return {"id": str(order_id), "clientOrderId": client_id, "stop_price": stop_price,
                "amount": amount, "info": order.get("info", {})}

    def _cancel_stop(self, symbol: str, stop: dict[str, Any] | None) -> None:
        if not stop or not stop.get("id"):
            return
        verify_demo_urls(self.exchange)
        self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": stop["id"]})

    @staticmethod
    def _is_bot_reduce_stop(row: dict[str, Any]) -> bool:
        return (str(row.get("clientAlgoId") or "").startswith(f"{BOT_ORDER_PREFIX}_s_")
                and str(row.get("orderType") or "").upper() == "STOP_MARKET"
                and str(row.get("reduceOnly")).strip().lower() in {"true", "1"})

    def _retire_superseded_stops(self, symbol: str, side: str, active_stop: dict[str, Any] | None,
                                 expected_quantity: float) -> list[dict[str, Any]]:
        """Keep the verified remaining-position stop and remove older bot stops."""
        verify_demo_urls(self.exchange)
        if active_stop is None:
            return []
        market_id = self.exchange.market(symbol)["id"]

        def open_orders() -> list[dict[str, Any]]:
            return self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []

        rows = open_orders()
        active_id = str((active_stop or {}).get("id") or "")
        active = next((row for row in rows if str(row.get("algoId")) == active_id), None)
        close_side = "SELL" if side == "long" else "BUY"
        step, _, _ = self._market_rules(symbol)
        if (active_stop is not None
                and (active is None or not self._is_bot_reduce_stop(active)
                     or str(active.get("side") or "").upper() != close_side
                     or str(active.get("algoStatus") or "").upper() != "NEW"
                     or abs(_number(active.get("quantity")) - expected_quantity) > max(step, 1e-10))):
            raise RuntimeError("remaining position protective stop is not confirmed at the actual quantity")

        for row in rows:
            order_id = row.get("algoId")
            if (order_id is None or str(order_id) == active_id
                    or str(row.get("algoStatus") or "").upper() != "NEW"
                    or str(row.get("side") or "").upper() != close_side
                    or not self._is_bot_reduce_stop(row)):
                continue
            try:
                self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": str(order_id)})
            except Exception:
                # Re-read below: some API errors mean the stop was already canceled.
                continue

        return [row for row in open_orders()
                if str(row.get("algoId")) != active_id
                and str(row.get("algoStatus") or "").upper() == "NEW"
                and self._is_bot_reduce_stop(row)]

    def reconcile_reduced_position_stops(self, symbol: str, side: str,
                                         active_stop: dict[str, Any] | None,
                                         expected_quantity: float) -> list[dict[str, Any]]:
        """Retry old bot-stop cleanup without risking the confirmed active stop."""
        return self._retire_superseded_stops(symbol, side, active_stop, expected_quantity)

    def audit_active_protective_stops(self, positions: dict[str, Any],
                                      tracked_orders: dict[str, dict[str, Any]]) -> list[str]:
        """Read-only verify each saved Demo position has exactly one matching bot stop."""
        verify_demo_urls(self.exchange)
        mismatches = []
        for symbol, position in positions.items():
            order_data = tracked_orders.get(symbol, {})
            stop = order_data.get("stop") or {}
            stop_id = str(stop.get("id") or "")
            if not stop_id:
                mismatches.append(f"{symbol}: active protective stop id missing")
                continue
            rows = self.exchange.fapiPrivateGetOpenAlgoOrders(
                {"symbol": self.exchange.market(symbol)["id"]}
            ) or []
            active = next((row for row in rows if str(row.get("algoId")) == stop_id), None)
            expected_side = "SELL" if position.side.value == "long" else "BUY"
            step, _, _ = self._market_rules(symbol)
            expected_stop_price = float(self.exchange.price_to_precision(symbol, float(stop["stop_price"])))
            valid = (
                active is not None
                and self._is_bot_reduce_stop(active)
                and str(active.get("algoStatus") or "").upper() == "NEW"
                and str(active.get("side") or "").upper() == expected_side
                and abs(_number(active.get("quantity")) - float(position.quantity)) <= max(step, 1e-10)
                and math.isclose(_number(active.get("triggerPrice")), expected_stop_price,
                                 rel_tol=1e-9, abs_tol=max(expected_stop_price * 1e-9, 1e-12))
            )
            if not valid:
                mismatches.append(f"{symbol}: saved protective stop is not active at the saved side/quantity/price")
                continue
            extras = [row for row in rows
                      if str(row.get("algoId")) != stop_id
                      and str(row.get("algoStatus") or "").upper() == "NEW"
                      and self._is_bot_reduce_stop(row)]
            if extras:
                mismatches.append(f"{symbol}: extra active bot protective stop(s) found")
        return mismatches

    def _cancel_untracked_bot_stops(self, symbol: str) -> None:
        """Cancel only this adapter's conditional orders when recovering an unknown position."""
        verify_demo_urls(self.exchange)
        market_id = self.exchange.market(symbol)["id"]
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        for row in rows:
            client_id = str(row.get("clientAlgoId") or "")
            algo_id = row.get("algoId")
            if client_id.startswith(f"{BOT_ORDER_PREFIX}_s_") and algo_id is not None:
                self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": algo_id})

    def cancel_saved_stop_if_open(self, symbol: str, stop: dict[str, Any] | None) -> None:
        """Cancel and verify one saved bot stop after its exchange position is flat."""
        if not stop or not stop.get("id"):
            return
        verify_demo_urls(self.exchange)
        market_id = self.exchange.market(symbol)["id"]
        algo_id = str(stop["id"])
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        match = next((row for row in rows if str(row.get("algoId")) == algo_id), None)
        if match is None:
            return
        client_id = str(match.get("clientAlgoId") or "")
        if not client_id.startswith(f"{BOT_ORDER_PREFIX}_s_") or not match.get("reduceOnly"):
            raise RuntimeError("saved stop id is not a confirmed bot-owned reduceOnly order")
        self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": algo_id})
        remaining = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        if any(str(row.get("algoId")) == algo_id for row in remaining):
            raise RuntimeError("bot protective stop cancellation was not confirmed")

    def cancel_orphaned_bot_stops(self, live_symbols: set[str]) -> list[dict[str, str]]:
        """Cancel bot-owned reduceOnly stops only when the exchange confirms the symbol is flat."""
        verify_demo_urls(self.exchange)
        market_symbols = {
            str(market.get("id")): market.get("symbol")
            for market in (getattr(self.exchange, "markets", {}) or {}).values()
            if market.get("id") and market.get("symbol")
        }

        def orphan_orders():
            rows = self.exchange.fapiPrivateGetOpenAlgoOrders({}) or []
            result = []
            for row in rows:
                if (str(row.get("algoStatus") or "").upper() != "NEW"
                        or not self._is_bot_reduce_stop(row)):
                    continue
                symbol = market_symbols.get(str(row.get("symbol") or ""))
                if symbol is None:
                    raise RuntimeError("cannot map an open bot stop to a known Demo market")
                if symbol not in live_symbols:
                    result.append((symbol, row))
            return result

        orphans = orphan_orders()
        cancelled = []
        for symbol, row in orphans:
            algo_id = row.get("algoId")
            if algo_id is None:
                raise RuntimeError("open bot protective stop has no algoId")
            self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": str(algo_id)})
            cancelled.append({"symbol": symbol, "algo_id": str(algo_id)})
        remaining = orphan_orders()
        if remaining:
            ids = ",".join(str(row.get("algoId")) for _, row in remaining)
            raise RuntimeError(f"orphaned bot protective stops remain open: {ids}")
        return cancelled

    def resolve_protective_stop_fill(self, symbol: str, order_id: str,
                                     stop: dict[str, Any] | None,
                                     expected_quantity: float) -> dict[str, Any] | None:
        """Return a fully verified fill for this bot's triggered reduceOnly stop."""
        if not stop or not stop.get("id"):
            return None
        verify_demo_urls(self.exchange)
        market_id = self.exchange.market(symbol)["id"]
        algo = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])})
        reduce_only = str(algo.get("reduceOnly")).strip().lower() in {"true", "1"}
        if (str(algo.get("actualOrderId") or "") != str(order_id)
                or not str(algo.get("clientAlgoId") or "").startswith(f"{BOT_ORDER_PREFIX}_s_")
                or not reduce_only
                or str(algo.get("orderType") or "").upper() != "STOP_MARKET"):
            return None
        trades = self.exchange.fapiPrivateGetUserTrades({
            "symbol": market_id, "orderId": str(order_id), "limit": 1000,
        }) or []
        matched = [row for row in trades if str(row.get("orderId")) == str(order_id)]
        quantity = sum(_number(row.get("qty")) for row in matched)
        notional = sum(_number(row.get("qty")) * _number(row.get("price")) for row in matched)
        if quantity <= 0 or notional <= 0:
            return None
        step, _, _ = self._market_rules(symbol)
        if abs(quantity - float(expected_quantity)) > max(step, float(expected_quantity) * 1e-8):
            return None
        fees = [_number(row.get("commission")) for row in matched
                if row.get("commissionAsset") == "USDT"]
        if len(fees) != len(matched):
            return None
        if any(row.get("realizedPnl") is None for row in matched):
            return None
        realized = [_number(row.get("realizedPnl")) for row in matched]
        timestamps = [int(row.get("time") or 0) for row in matched]
        if any(value <= 0 for value in timestamps):
            return None
        return {
            "order_id": str(order_id), "algo_id": str(stop["id"]),
            "quantity": quantity, "average": notional / quantity,
            "gross_pnl": sum(realized), "exit_fee": sum(fees),
            "closed_at": datetime.fromtimestamp(max(timestamps) / 1000, timezone.utc).isoformat(),
            "client_algo_id": str(algo["clientAlgoId"]),
        }

    def resolve_protective_stop_algo_fill(self, symbol: str, stop: dict[str, Any] | None,
                                          expected_quantity: float) -> dict[str, Any] | None:
        """Resolve the triggered exchange order id for a tracked bot stop, then verify its fills."""
        if not stop or not stop.get("id"):
            return None
        verify_demo_urls(self.exchange)
        algo = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])})
        order_id = algo.get("actualOrderId")
        if not order_id:
            return None
        return self.resolve_protective_stop_fill(
            symbol, str(order_id), stop, expected_quantity,
        )

    def inspect_entry_attempt(self, attempt: dict[str, Any]) -> dict[str, Any]:
        """Read-only recovery by the exact saved order identity; never resend."""
        verify_demo_urls(self.exchange)
        symbol = attempt["symbol"]
        market_id = self.exchange.market(symbol)["id"]
        order = self.exchange.fapiPrivateGetOrder({
            "symbol": market_id, "origClientOrderId": attempt["clientOrderId"],
        })
        if (str(order.get("clientOrderId")) != attempt["clientOrderId"]
                or order.get("symbol") != market_id
                or str(order.get("side")).upper() != ("BUY" if attempt["side"] == "long" else "SELL")
                or str(order.get("reduceOnly")).lower() not in {"false", "0"}):
            return {"status": "identity_mismatch"}
        live = self._position(symbol)
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        filled = _number(order.get("executedQty"))
        terminal = str(order.get("status")).upper()
        if filled == 0 and terminal in {"CANCELED", "EXPIRED", "REJECTED"} and live is None:
            return {"status": "not_filled", "orderId": str(order["orderId"])}
        if terminal not in {"FILLED", "CANCELED", "EXPIRED"} or filled <= 0:
            return {"status": "order_not_terminal"}
        trades = self.exchange.fapiPrivateGetUserTrades({
            "symbol": market_id, "orderId": str(order["orderId"]), "limit": 1000,
        }) or []
        trades = [row for row in trades if str(row.get("orderId")) == str(order["orderId"])]
        tolerance = max(filled * 1e-8, 1e-12)
        if not trades or abs(sum(_number(row.get("qty")) for row in trades) - filled) > tolerance:
            return {"status": "trades_unconfirmed"}
        if live is None:
            # Flat is not evidence that no entry happened. Its close and PnL must
            # be reconciled before cooldown / repeated-loss gates can be trusted.
            return {"status": "filled_then_flat", "orderId": str(order["orderId"])}
        if live["side"] != attempt["side"] or abs(live["quantity"] - filled) > tolerance:
            return {"status": "position_mismatch"}
        stops = [row for row in rows if self._is_bot_reduce_stop(row)
                 and str(row.get("algoStatus")).upper() == "NEW"]
        expected_side = "SELL" if attempt["side"] == "long" else "BUY"
        if (len(stops) != 1 or str(stops[0].get("side")).upper() != expected_side
                or abs(_number(stops[0].get("quantity")) - filled) > tolerance
                or not math.isclose(_number(stops[0].get("triggerPrice")), attempt["stop_price"],
                                    rel_tol=1e-9, abs_tol=1e-12)):
            return {"status": "protection_unconfirmed"}
        average = _number(order.get("avgPrice"))
        if average <= 0 or not math.isclose(live["entry_price"], average, rel_tol=1e-7, abs_tol=1e-12):
            return {"status": "entry_price_mismatch"}
        stop = stops[0]
        fill = {"id": str(order["orderId"]), "clientOrderId": attempt["clientOrderId"],
                "filled": filled, "average": average, "trade_data_verified": True}
        if all(row.get("commission") is not None and row.get("commissionAsset") == "USDT"
               for row in trades):
            fill["fee_cost"] = sum(_number(row["commission"]) for row in trades)
        times = [int(row.get("time") or 0) for row in trades]
        if all(value > 0 for value in times):
            fill["closed_at"] = datetime.fromtimestamp(max(times) / 1000, timezone.utc).isoformat()
        return {"status": "protected", "result": {
            "fill": fill, "position": live,
            "stop": {"id": str(stop["algoId"]), "clientOrderId": stop.get("clientAlgoId"),
                     "stop_price": _number(stop["triggerPrice"]), "amount": filled},
            "order_submitted_at": attempt["submitted_at"],
        }}

    def open(self, intent, *, stop_price: float,
             before_submit: Callable | None = None,
             submission_deadline: datetime | None = None,
             client_order_id: str | None = None,
             on_submit: Callable | None = None) -> dict[str, Any]:
        if intent.action is not Action.OPEN:
            raise ValueError("open executor accepts only OPEN intents")
        if stop_price is None or not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError("opening intent has no valid protective stop")
        if submission_deadline is not None and _utc_now() > submission_deadline:
            raise EntryPreflightRejected("entry_signal_expired")
        verify_demo_urls(self.exchange)
        self._ensure_one_way()
        if self._position(intent.symbol) is not None:
            raise RuntimeError("exchange position already exists for symbol")
        self._prepare_symbol(intent.symbol)
        if before_submit is not None:
            original = (intent.symbol, intent.side, intent.action, intent.source)
            intent, stop_price = before_submit(intent, stop_price)
            if (intent.symbol, intent.side, intent.action, intent.source) != original:
                raise ValueError("entry preflight cannot change the order identity")
            if stop_price is None or not math.isfinite(stop_price) or stop_price <= 0:
                raise ValueError("entry preflight returned no valid protective stop")
        step, minimum, minimum_notional = self._market_rules(intent.symbol)
        amount = float(intent.quantity)
        if step > 0:
            amount = math.floor((amount + step * 1e-10) / step) * step
        amount = float(self.exchange.amount_to_precision(intent.symbol, amount))
        stop_price = float(self.exchange.price_to_precision(intent.symbol, stop_price))
        validation_error = EntryPreflightRejected if before_submit is not None else ValueError
        if amount <= 0 or amount < minimum:
            raise validation_error("rounded quantity is below the exchange minimum")
        if intent.price * amount < minimum_notional:
            raise validation_error("order notional is below Binance minimum notional")
        side = "buy" if intent.side is Side.LONG else "sell"
        if intent.side is Side.LONG and stop_price >= intent.price:
            raise validation_error("protective stop must be below planned long entry")
        if intent.side is Side.SHORT and stop_price <= intent.price:
            raise validation_error("protective stop must be above planned short entry")
        submitted_at = _utc_now()
        if submission_deadline is not None and submitted_at > submission_deadline:
            raise EntryPreflightRejected("entry_signal_expired")
        if on_submit is not None:
            if not client_order_id:
                raise ValueError("durable entry submission requires a client order id")
            on_submit(amount, stop_price, submitted_at)
            # Saving the intent can itself consume the remaining signal window.
            if submission_deadline is not None and _utc_now() > submission_deadline:
                raise EntryPreflightRejected("entry_signal_expired")
        fill_kwargs = {"client_order_id": client_order_id} if client_order_id else {}
        fill = self._market_fill(intent.symbol, side, amount, reduce_only=False, **fill_kwargs)
        live_position = self._position(intent.symbol)
        if live_position is None or live_position["side"] != intent.side.value:
            raise RuntimeError("market fill did not produce the expected exchange position")
        if step > 0 and abs(live_position["quantity"] - fill["filled"]) > step + 1e-12:
            raise RuntimeError("exchange position quantity differs from the confirmed fill")
        logger.info("Demo OPEN_FILL symbol={} side={} amount={} orderId={} host={} source={} reason={} average={} stop_status=pending",
                    intent.symbol, intent.side.value, live_position["quantity"], fill["id"],
                    self.host, intent.source, intent.reason, live_position["entry_price"])
        if ((intent.side is Side.LONG and stop_price >= live_position["entry_price"])
                or (intent.side is Side.SHORT and stop_price <= live_position["entry_price"])):
            try:
                self._market_fill(intent.symbol, "sell" if intent.side is Side.LONG else "buy",
                                  live_position["quantity"], reduce_only=True)
            except Exception as exc:
                raise RuntimeError("fill moved beyond planned stop and emergency close failed; account is inconsistent") from exc
            raise RuntimeError("fill moved beyond planned stop; Demo position was emergency-closed")
        try:
            stop = self._place_stop(intent.symbol, intent.side.value, live_position["quantity"], stop_price)
        except Exception as stop_error:
            try:
                self._market_fill(intent.symbol, "sell" if intent.side is Side.LONG else "buy",
                                  live_position["quantity"], reduce_only=True)
            except Exception as close_error:
                raise RuntimeError("protective stop placement failed and emergency reduceOnly close failed; account is inconsistent") from close_error
            raise RuntimeError("protective stop placement failed; Demo position was emergency-closed") from stop_error
        return {"fill": fill, "position": live_position, "stop": stop,
                "order_submitted_at": submitted_at.isoformat(),
                "host": self.host, "reason": intent.reason, "regime": intent.regime.value}

    def close(self, position, *, stop: dict[str, Any] | None = None,
              reason: str | None = None) -> dict[str, Any]:
        symbol = position.symbol
        live_position = self._position(symbol)
        if live_position is None:
            return {"already_flat": True, "position": None, "host": self.host}
        if live_position["side"] != position.side.value:
            raise RuntimeError("exchange side differs from local position; account is inconsistent")
        if stop:
            self._cancel_stop(symbol, stop)
        else:
            self._cancel_untracked_bot_stops(symbol)
        try:
            fill = self._market_fill(symbol, "sell" if position.side is Side.LONG else "buy",
                                      live_position["quantity"], reduce_only=True)
        except Exception:
            if stop and stop.get("stop_price"):
                restored = self._place_stop(symbol, position.side.value, live_position["quantity"], float(stop["stop_price"]))
                stop.update(restored)
            raise
        remaining = self._position(symbol)
        if remaining is not None:
            raise RuntimeError("reduceOnly close was sent but exchange position is still open")
        logger.info("Demo CLOSE symbol={} side={} amount={} orderId={} host={} source={} reason={} average={}",
                    symbol, position.side.value, fill["filled"], fill["id"], self.host, position.source,
                    reason or "strategy_close", fill["average"])
        return {"fill": fill, "position": None, "host": self.host}

    def reduce(self, position, quantity: float, *, stop: dict[str, Any] | None,
               reason: str | None = None) -> dict[str, Any]:
        """Reduce a Demo position while keeping its remaining quantity protected."""
        symbol = position.symbol
        verify_demo_urls(self.exchange)
        live = self._position(symbol)
        if live is None or live["side"] != position.side.value:
            raise RuntimeError("cannot partially reduce while exchange/local positions disagree")
        step, minimum, _ = self._market_rules(symbol)
        quantity = min(float(quantity), float(live["quantity"]))
        if step > 0:
            quantity = math.floor((quantity + step * 1e-10) / step) * step
        quantity = float(self.exchange.amount_to_precision(symbol, quantity))
        remaining = float(live["quantity"]) - quantity
        if quantity <= 0 or quantity < minimum or remaining <= 0:
            raise ValueError("partial close quantity or remaining position is below the exchange minimum")

        # Install protection for the expected remainder before reducing. For the
        # short interval both reduceOnly stops can coexist; the old stop is removed
        # only after the exchange confirms the reduced position size.
        replacement = self._place_stop(symbol, position.side.value, remaining,
                                       float(stop["stop_price"])) if stop else None
        try:
            fill = self._market_fill(symbol, "sell" if position.side is Side.LONG else "buy",
                                     quantity, reduce_only=True)
            updated = self._position(symbol)
            if updated is None or updated["side"] != position.side.value:
                raise RuntimeError("Demo partial reduction unexpectedly closed or reversed the position")
            if abs(float(updated["quantity"]) - remaining) > max(step, 1e-10):
                raise RuntimeError("Demo remaining position quantity differs from expected partial reduction")
        except Exception:
            if replacement:
                try:
                    self._cancel_stop(symbol, replacement)
                except Exception:
                    pass
            raise
        cleanup_error = None
        stale_stops = []
        try:
            stale_stops = self._retire_superseded_stops(
                symbol, position.side.value, replacement, float(updated["quantity"]),
            )
        except Exception as exc:
            cleanup_error = type(exc).__name__
            # The market fill and remaining position are already confirmed. Preserve them
            # for the caller even when stop cleanup needs a later reconciliation cycle.
            stale_stops = [stop] if stop else [{"id": None, "reason": "stop_cleanup_unverified"}]
            logger.error("Demo partial fill confirmed for {} but superseded stop cleanup failed: {}",
                         symbol, cleanup_error)
        return {"fill": fill, "position": updated, "stop": replacement, "host": self.host,
                "reason": reason, "stale_stops": stale_stops,
                "stop_cleanup_error": cleanup_error}

    def replace_stop(self, position, *, stop: dict[str, Any] | None,
                     stop_price: float) -> dict[str, Any]:
        verify_demo_urls(self.exchange)
        live_position = self._position(position.symbol)
        if live_position is None or live_position["side"] != position.side.value:
            raise RuntimeError("cannot update stop while exchange/local positions disagree")

        symbol = position.symbol
        candidate = float(self.exchange.price_to_precision(symbol, stop_price))
        if not math.isfinite(candidate) or candidate <= 0:
            raise ValueError("protective stop price must be finite and positive")

        current_stop_price = None
        if stop and stop.get("id"):
            current = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])}) or {}
            expected_side = "SELL" if position.side is Side.LONG else "BUY"
            step, _, _ = self._market_rules(symbol)
            actual_price = _number(current.get("triggerPrice"), float("nan"))
            saved_price = _number(stop.get("stop_price"), float("nan"))
            saved_price = float(self.exchange.price_to_precision(symbol, saved_price))
            valid = (
                str(current.get("algoId")) == str(stop["id"])
                and str(current.get("symbol") or "").upper() == self.exchange.market(symbol)["id"].upper()
                and str(current.get("clientAlgoId") or "").startswith(f"{BOT_ORDER_PREFIX}_s_")
                and str(current.get("orderType") or "").upper() == "STOP_MARKET"
                and str(current.get("algoStatus") or "").upper() == "NEW"
                and str(current.get("side") or "").upper() == expected_side
                and str(current.get("reduceOnly")).strip().lower() in {"true", "1"}
                and abs(_number(current.get("quantity")) - float(live_position["quantity"]))
                <= max(step, 1e-10)
                and math.isfinite(actual_price) and actual_price > 0
                and math.isfinite(saved_price)
                and math.isclose(actual_price, saved_price, rel_tol=1e-9,
                                 abs_tol=max(actual_price * 1e-9, 1e-12))
            )
            if not valid:
                raise RuntimeError("cannot replace an unverified or stale Demo protective stop")
            current_stop_price = actual_price
        else:
            current_stop_price = _number(position.stop_price, float("nan"))
            current_stop_price = float(self.exchange.price_to_precision(symbol, current_stop_price))
            if not math.isfinite(current_stop_price) or current_stop_price <= 0:
                raise RuntimeError("cannot replace a Demo stop without a valid existing risk stop")

        if position.side is Side.LONG:
            if candidate < current_stop_price:
                raise ValueError(
                    f"refusing to loosen long protective stop: current={current_stop_price} requested={candidate}"
                )
            if candidate == current_stop_price and stop:
                return stop
        else:
            if candidate > current_stop_price:
                raise ValueError(
                    f"refusing to loosen short protective stop: current={current_stop_price} requested={candidate}"
                )
            if candidate == current_stop_price and stop:
                return stop

        new_stop = self._place_stop(symbol, position.side.value, live_position["quantity"], candidate)
        try:
            self._cancel_stop(position.symbol, stop)
        except Exception:
            try:
                self._cancel_stop(position.symbol, new_stop)
            finally:
                raise
        return new_stop

    def audit_recent_fills(self, symbols: set[str], known_order_ids: set[str],
                           seen_foreign_order_ids: set[str]) -> list[dict[str, str]]:
        """Report unknown fills on symbols the bot has traded or currently holds.

        Binance's USDⓈ-M user-trades endpoint is symbol-scoped. This audit is
        intentionally restricted to tracked symbols to keep request weight bounded.
        """
        warnings = []
        if time.monotonic() - self._last_fill_audit_at < 300:
            return warnings
        self._last_fill_audit_at = time.monotonic()
        verify_demo_urls(self.exchange)
        for symbol in sorted(symbols):
            try:
                trades = self.exchange.fetch_my_trades(symbol, since=None, limit=1000)
            except Exception as exc:
                logger.warning("Demo fill audit unavailable for {}: {}", symbol, type(exc).__name__)
                continue
            for trade in trades:
                info = trade.get("info") or {}
                order_id = str(trade.get("order") or info.get("orderId") or "")
                if order_id and order_id not in known_order_ids and order_id not in seen_foreign_order_ids:
                    warnings.append({"symbol": symbol, "orderId": order_id})
        return warnings
