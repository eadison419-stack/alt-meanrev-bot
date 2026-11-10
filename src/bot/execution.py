"""Execution adapter boundary: strategy intents become exchange orders here."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import math
import time
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from ccxt import OrderNotFound, RequestTimeout
from loguru import logger

from bot.exchange.binance_usdm import (DEMO_REST_ROOT, LIVE_REST_ROOT, parse_live_position,
                                       verify_demo_urls, verify_live_urls)
from bot.exchange.clock import active_clock, ExchangeClockError
from bot.signals import Action, Side
from bot.lead_risk import liquidation_is_safe, parse_brackets, select_leverage


BOT_ORDER_PREFIX = "mrb"


class EntryPreflightRejected(RuntimeError):
    """A candidate was rejected before an opening order was submitted."""


def _utc_now():
    if active_clock() is not None:
        return active_clock().now()
    return datetime.now(timezone.utc)


def _require_entry_clock():
    if active_clock() is not None:
        try:
            active_clock().require_fresh()
        except ExchangeClockError as exc:
            raise EntryPreflightRejected("exchange_clock_stale_or_unavailable") from exc


def _number(value, default=0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


class DemoExecutionService:
    """Signed USDⓈ-M orders and account snapshots behind a pinned host verifier."""

    def __init__(self, exchange, settings, *, mode="demo", auto_one_way=True):
        self.mode = mode
        self.label = "Lead Portfolio" if mode == "lead" else "Demo"
        self._url_verifier = verify_live_urls if mode == "lead" else verify_demo_urls
        self._url_verifier(exchange)
        self.exchange = exchange
        self.settings = settings
        self.host = LIVE_REST_ROOT if mode == "lead" else DEMO_REST_ROOT
        self.source = "binance_usdm_lead" if mode == "lead" else "binance_usdm_demo"
        self.lead_market_ids: set[str] = set()
        self.auto_one_way = auto_one_way
        self._last_fill_audit_at = 0.0
        self._close_orders = {}
        self._close_checkpoint = None

    def configure_close_recovery(self, orders_by_symbol, checkpoint):
        """Use the runner's existing durable order ledger for uncertain closes."""
        self._close_orders = orders_by_symbol
        self._close_checkpoint = checkpoint

    def _save_close_attempt(self):
        if self._close_checkpoint is not None:
            self._close_checkpoint()

    def _verify_urls(self):
        self._url_verifier(self.exchange)

    def initialize(self) -> dict[str, Any]:
        if self.mode == "lead":
            self._verify_lead_api_access()
        self._ensure_one_way()
        snapshot = self.sync_snapshot()
        logger.info("Binance USDⓈ-M {} account check: host={} USDT={} position_mode=one-way",
                    self.label,
                    self.host, snapshot["usdt_balance"])
        if not math.isfinite(float(snapshot["usdt_balance"])) or float(snapshot["usdt_balance"]) <= 0:
            raise RuntimeError("USDⓈ-M account has no positive USDT wallet balance; refusing execution")
        return snapshot

    def _verify_lead_api_access(self) -> None:
        self._verify_urls()
        status = self.exchange.sapiGetCopyTradingFuturesUserStatus({})
        data = status.get("data") if isinstance(status, dict) else None
        if (not isinstance(data, dict) or status.get("success") is not True
                or status.get("code") != "000000" or data.get("isLeadTrader") is not True):
            raise RuntimeError("Binance API key is not verified as an active Futures Lead Portfolio key")
        whitelist = self.exchange.sapiGetCopyTradingFuturesLeadSymbol({})
        rows = whitelist.get("data") if isinstance(whitelist, dict) else None
        if (not isinstance(rows, list) or whitelist.get("success") is not True
                or whitelist.get("code") != "000000"):
            raise RuntimeError("Binance Lead Portfolio symbol whitelist could not be verified")
        symbols = {
            str(row.get("symbol", "")).strip().upper()
            for row in rows
            if isinstance(row, dict) and str(row.get("quoteAsset", "")).upper() == "USDT"
        }
        if not symbols or any(not symbol.endswith("USDT") for symbol in symbols):
            raise RuntimeError("Binance Lead Portfolio has no valid USDT futures symbols in its whitelist")
        self.lead_market_ids = symbols
        logger.info("Binance Lead Portfolio verified; USDT contract whitelist size={}", len(symbols))

    def _ensure_one_way(self) -> None:
        self._verify_urls()
        mode = self.exchange.fetch_position_mode()
        if bool(mode.get("hedged")):
            if not self.auto_one_way:
                raise RuntimeError("Binance Lead account must already be in one-way position mode")
            self.exchange.set_position_mode(False)
            mode = self.exchange.fetch_position_mode()
        if bool(mode.get("hedged")):
            raise RuntimeError("Binance account is not in one-way position mode")

    def sync_snapshot(self) -> dict[str, Any]:
        self._verify_urls()
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
            "at": _utc_now().isoformat(),
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
        self._prepare_margin(symbol)
        leverage = int(self.settings.sizing.leverage)
        result = self.exchange.set_leverage(leverage, symbol)
        actual = (result or {}).get("leverage")
        if actual is not None and int(actual) != leverage:
            raise RuntimeError("Binance did not confirm the configured leverage")

    def _prepare_margin(self, symbol: str) -> None:
        self._verify_urls()
        try:
            self.exchange.set_margin_mode("isolated", symbol)
        except Exception as exc:
            if "No need to change margin type" not in str(exc) and "already isolated" not in str(exc).lower():
                raise

    def _prepare_entry(self, intent):
        self._prepare_symbol(intent.symbol)
        return None

    def _finalize_entry(self, intent, preparation):
        return None

    def _verify_new_position(self, intent, live_position, stop, safety):
        return None

    def _check_or_exit_new_position(self, intent, stop, safety):
        try:
            live = self._position(intent.symbol)
            self._verify_new_position(intent, live, stop, safety)
            return live
        except Exception as safety_error:
            try:
                current = self._position(intent.symbol)
                if current is not None:
                    if current["side"] != intent.side.value:
                        raise RuntimeError("new position side changed during safety verification")
                    self._market_fill(intent.symbol, "sell" if intent.side is Side.LONG else "buy",
                                      current["quantity"], reduce_only=True)
                if self._position(intent.symbol) is not None:
                    raise RuntimeError("new position safety exit is not flat")
                self.cancel_saved_stop_if_open(intent.symbol, stop)
            except Exception as close_error:
                raise RuntimeError("new position safety verification and emergency close failed; reconciliation required") from close_error
            raise RuntimeError("new position failed liquidation/protection verification and was reduceOnly closed; reconcile fills before further entries") from safety_error

    def _market_fill(self, symbol: str, side: str, quantity: float, *, reduce_only: bool,
                     client_order_id: str | None = None, lookup_only=False,
                     allow_unfilled=False) -> dict[str, Any]:
        self._verify_urls()
        if quantity <= 0 or not math.isfinite(quantity):
            raise ValueError("exchange order quantity must be finite and positive")
        client_id = client_order_id or f"{BOT_ORDER_PREFIX}_{uuid4().hex[:28]}"
        params = {"clientOrderId": client_id, "newOrderRespType": "RESULT"}
        if reduce_only:
            params["reduceOnly"] = True
        if lookup_only:
            order = self._lookup_submitted_market_order(
                symbol, side, quantity, reduce_only, client_id, allow_unfilled=allow_unfilled,
            )
        else:
            try:
                order = self.exchange.create_order(symbol, "market", side, quantity, None, params)
            except (OrderNotFound, RequestTimeout):
                order = self._lookup_submitted_market_order(symbol, side, quantity, reduce_only, client_id)
        order_id = order.get("id") or (order.get("info") or {}).get("orderId")
        if not order_id:
            raise RuntimeError("USDⓈ-M exchange returned no orderId; reconciliation required")
        filled = _number(order.get("filled"), _number((order.get("info") or {}).get("executedQty")))
        average = _number(order.get("average"), _number((order.get("info") or {}).get("avgPrice")))
        if allow_unfilled and filled == 0 and order.get("status") == "canceled":
            return {"id": str(order_id), "clientOrderId": client_id,
                    "filled": 0, "average": 0, "status": "canceled", "side": side}
        if not filled or not average:
            try:
                order = self.exchange.fetch_order(order_id, symbol)
            except (OrderNotFound, RequestTimeout):
                order = self._lookup_submitted_market_order(symbol, side, quantity, reduce_only,
                                                             client_id, expected_order_id=order_id)
            filled = _number(order.get("filled"), _number((order.get("info") or {}).get("executedQty")))
            average = _number(order.get("average"), _number((order.get("info") or {}).get("avgPrice")))
        status = order.get("status")
        status = status.lower() if isinstance(status, str) else status
        terminal_statuses = {None, "closed", "canceled", "cancelled", "expired"}
        if filled <= 0 or average <= 0 or status not in terminal_statuses:
            raise RuntimeError(f"USDⓈ-M market order {order_id} was not confirmed filled")
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

    def _lookup_submitted_market_order(self, symbol, side, quantity, reduce_only, client_id,
                                       *, expected_order_id=None, allow_unfilled=False):
        """Allow short read propagation after submit; never send the market order again."""
        market_id = self.exchange.market(symbol)["id"]
        params = {"symbol": market_id, "origClientOrderId": client_id}
        for delay in (0, .2, .5, 1):
            if delay:
                time.sleep(delay)
            try:
                row = self.exchange.fapiPrivateGetOrder(params)
            except (OrderNotFound, RequestTimeout):
                continue
            order_id = str(row.get("orderId") or "")
            if (not order_id or row.get("symbol") != market_id or row.get("clientOrderId") != client_id
                    or (expected_order_id is not None and order_id != str(expected_order_id))
                    or str(row.get("type")).upper() != "MARKET" or row.get("side") != side.upper()
                    or str(row.get("positionSide")).upper() != "BOTH"
                    or str(row.get("reduceOnly")).lower() != str(reduce_only).lower()
                    or not math.isclose(_number(row.get("origQty")), quantity, rel_tol=1e-8, abs_tol=1e-12)):
                raise RuntimeError("submitted market order identity mismatch; reconciliation required")
            status = str(row.get("status")).upper()
            filled, average = _number(row.get("executedQty")), _number(row.get("avgPrice"))
            if status in {"FILLED", "CANCELED", "EXPIRED"} and 0 < filled <= quantity and average > 0:
                return {"id": order_id, "filled": filled, "average": average, "status": "closed", "info": row}
            if status in {"CANCELED", "EXPIRED", "REJECTED"} and filled == 0:
                if allow_unfilled:
                    return {"id": order_id, "filled": 0, "average": 0,
                            "status": "canceled", "info": row}
                break
        raise OrderNotFound("submitted market fill not confirmed by client order id; reconciliation required")

    def funding_fee(self, symbol: str, opened_at: datetime, closed_at: datetime) -> float | None:
        """Return verified USDT funding income for one completed Demo position."""
        if opened_at.tzinfo is None or closed_at.tzinfo is None or closed_at < opened_at:
            return None
        try:
            self._verify_urls()
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
            self._verify_urls()
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
            self._verify_urls()
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

    def reconstruct_open_position(self, symbol: str, expected: dict[str, Any], *,
                                  at: datetime | None = None) -> dict[str, Any]:
        """Rebuild the current one-way position from bounded, verified USD-M fills."""
        self._verify_urls()
        at = at or _utc_now()
        if at.tzinfo is None:
            raise ValueError("position reconstruction time must be timezone-aware")
        market_id = self.exchange.market(symbol)["id"]
        start_ms = int((at.timestamp() - 90 * 24 * 60 * 60) * 1000)
        end_ms = int(at.timestamp() * 1000)
        rows_by_id = {}
        window_start = start_ms
        while window_start <= end_ms:
            window_end = min(end_ms, window_start + 7 * 24 * 60 * 60 * 1000 - 1)
            batch = self.exchange.fapiPrivateGetUserTrades({
                "symbol": market_id, "startTime": window_start,
                "endTime": window_end, "limit": 1000,
            })
            if not isinstance(batch, list) or len(batch) >= 1000:
                raise RuntimeError("manual position trade history is incomplete or reached the API limit")
            for row in batch:
                try:
                    trade_id, timestamp = int(row["id"]), int(row["time"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError("manual position trade history contains an invalid row") from exc
                if not window_start <= timestamp <= window_end:
                    raise RuntimeError("manual position trade history is outside the requested window")
                previous = rows_by_id.get(trade_id)
                if previous is not None:
                    if previous != row:
                        raise RuntimeError("manual position trade history contains conflicting duplicates")
                    continue
                rows_by_id[trade_id] = row
                if len(rows_by_id) >= 1000:
                    raise RuntimeError("manual position trade history reached the total safety limit")
            window_start = window_end + 1
        rows = list(rows_by_id.values())
        trades = []
        seen_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("manual position trade history contains an invalid row")
            try:
                trade_id = int(row["id"])
                timestamp = int(row["time"])
                quantity = float(row["qty"])
                price = float(row["price"])
                commission = float(row["commission"])
                side = str(row["side"]).upper()
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("manual position trade history is incomplete") from exc
            if (trade_id in seen_ids or timestamp < start_ms
                    or timestamp > int(at.timestamp() * 1000) + 30_000
                    or (row.get("symbol") is not None
                        and str(row.get("symbol")).upper() != str(market_id).upper())
                    or side not in {"BUY", "SELL"}
                    or str(row.get("positionSide") or "BOTH").upper() != "BOTH"
                    or not all(math.isfinite(value) for value in (quantity, price, commission))
                    or quantity <= 0 or price <= 0 or commission < 0
                    or row.get("commissionAsset") != "USDT"):
                raise RuntimeError("manual position trade history has unsupported or invalid fields")
            seen_ids.add(trade_id)
            trades.append((timestamp, trade_id, side, quantity, price, commission))
        trades.sort(key=lambda item: (item[0], item[1]))
        if not trades:
            raise RuntimeError("manual position trade history is empty")

        signed_quantity = 0.0
        average_entry = 0.0
        entry_fee = 0.0
        opened_at = None
        for timestamp, _trade_id, side, quantity, price, commission in trades:
            fill_signed = quantity if side == "BUY" else -quantity
            if signed_quantity == 0 or signed_quantity * fill_signed > 0:
                old_quantity = abs(signed_quantity)
                signed_quantity += fill_signed
                average_entry = ((average_entry * old_quantity + price * quantity)
                                 / abs(signed_quantity))
                entry_fee += commission
                if opened_at is None:
                    opened_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                continue

            old_quantity = abs(signed_quantity)
            closed_quantity = min(old_quantity, quantity)
            remaining_old = old_quantity - closed_quantity
            entry_fee *= remaining_old / old_quantity
            signed_quantity += fill_signed
            if remaining_old <= 1e-12:
                average_entry = 0.0
                entry_fee = 0.0
                opened_at = None
            if quantity > old_quantity:
                opened_quantity = quantity - old_quantity
                signed_quantity = math.copysign(opened_quantity, fill_signed)
                average_entry = price
                entry_fee = commission * opened_quantity / quantity
                opened_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)

        try:
            expected_side = str(expected["side"]).lower()
            expected_quantity = float(expected["quantity"])
            expected_entry = float(expected["entry_price"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("exchange position snapshot is incomplete") from exc
        step, _, _ = self._market_rules(symbol)
        if (expected_side not in {"long", "short"} or expected_quantity <= 0
                or expected_entry <= 0 or not math.isfinite(expected_quantity)
                or not math.isfinite(expected_entry) or opened_at is None
                or abs(abs(signed_quantity) - expected_quantity) > max(step, expected_quantity * 1e-8)
                or ("long" if signed_quantity > 0 else "short") != expected_side
                or abs(average_entry - expected_entry) > max(expected_entry * 1e-6, 1e-10)):
            raise RuntimeError("replayed fills do not match the current exchange position")
        return {"side": expected_side, "quantity": expected_quantity,
                "entry_price": average_entry, "opened_at": opened_at,
                "entry_fee": entry_fee}

    def _position(self, symbol: str) -> dict[str, Any] | None:
        self._verify_urls()
        rows = self.exchange.fetch_positions([symbol])
        return next((parsed for row in rows if (parsed := parse_live_position(row)) is not None), None)

    def _place_stop(self, symbol: str, side: str, amount: float, stop_price: float) -> dict[str, Any]:
        if not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError("a valid protective stop is required before order submission")
        self._verify_urls()
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
            raise RuntimeError("USDⓈ-M protective stop returned no orderId")
        return {"id": str(order_id), "clientOrderId": client_id, "stop_price": stop_price,
                "amount": amount, "info": order.get("info", {})}

    def _cancel_stop(self, symbol: str, stop: dict[str, Any] | None) -> None:
        if not stop or not stop.get("id"):
            return
        self._verify_urls()
        if stop.get("external"):
            current = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])}) or {}
            step, _, _ = self._market_rules(symbol)
            expected_quantity = _number(stop.get("amount"), float("nan"))
            expected_price = _number(stop.get("stop_price"), float("nan"))
            try:
                expected_price = float(self.exchange.price_to_precision(symbol, expected_price))
            except (TypeError, ValueError, OverflowError):
                expected_price = float("nan")
            if (str(current.get("algoId")) != str(stop["id"])
                    or str(current.get("symbol") or "").upper()
                    != str(self.exchange.market(symbol)["id"]).upper()
                    or str(current.get("clientAlgoId") or "")
                    != str(stop.get("clientOrderId") or "")
                    or str(current.get("orderType") or "").upper() != "STOP_MARKET"
                    or str(current.get("reduceOnly")).strip().lower() not in {"true", "1"}
                    or (stop.get("order_side") and str(current.get("side") or "").upper()
                        != str(stop["order_side"]).upper())
                    or abs(_number(current.get("quantity")) - expected_quantity) > max(step, 1e-10)
                    or not math.isfinite(expected_price)
                    or not math.isclose(_number(current.get("triggerPrice")), expected_price,
                                        rel_tol=1e-9, abs_tol=max(expected_price * 1e-9, 1e-12))):
                raise RuntimeError("refusing to cancel an unverified adopted protective stop")
        self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": stop["id"]})

    def _saved_stop_matches(self, symbol: str, position, stop: dict[str, Any],
                            row: dict[str, Any], quantity: float) -> bool:
        market_id = str(self.exchange.market(symbol)["id"]).upper()
        expected_side = "SELL" if position.side is Side.LONG else "BUY"
        step, _, _ = self._market_rules(symbol)
        client_id = str(row.get("clientAlgoId") or "")
        is_owned = self._is_bot_reduce_stop(row)
        is_adopted = bool(stop.get("external")) and client_id == str(stop.get("clientOrderId") or "")
        try:
            expected_price = float(self.exchange.price_to_precision(symbol, float(stop["stop_price"])))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return (
            str(row.get("algoId")) == str(stop.get("id"))
            and str(row.get("symbol") or "").upper() == market_id
            and (is_owned or is_adopted)
            and str(row.get("orderType") or "").upper() == "STOP_MARKET"
            and str(row.get("algoStatus") or "").upper() == "NEW"
            and str(row.get("side") or "").upper() == expected_side
            and str(row.get("reduceOnly")).strip().lower() in {"true", "1"}
            and abs(_number(row.get("quantity")) - quantity) <= max(step, 1e-10)
            and math.isclose(_number(row.get("triggerPrice")), expected_price,
                             rel_tol=1e-9, abs_tol=max(expected_price * 1e-9, 1e-12))
        )

    def adopt_manual_position_stop(self, position, *, strategy_stop: float,
                                   market_price: float) -> dict[str, Any]:
        """Verify or install one strategy stop before the caller records an adopted position."""
        symbol = position.symbol
        self._verify_urls()
        live = self._position(symbol)
        step, _, _ = self._market_rules(symbol)
        if (live is None or live["side"] != position.side.value
                or abs(float(live["quantity"]) - float(position.quantity)) > max(step, 1e-10)
                or abs(float(live["entry_price"]) - float(position.entry_price))
                > max(float(position.entry_price) * 1e-6, 1e-10)):
            raise RuntimeError("exchange position changed during manual adoption")
        candidate = float(self.exchange.price_to_precision(symbol, strategy_stop))
        market_price = float(market_price)
        if (not math.isfinite(candidate) or candidate <= 0
                or not math.isfinite(market_price) or market_price <= 0
                or (position.side is Side.LONG and candidate >= market_price)
                or (position.side is Side.SHORT and candidate <= market_price)):
            raise RuntimeError("strategy stop is invalid or already crossed by the market")

        market_id = self.exchange.market(symbol)["id"]
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        active_stops = [row for row in rows
                        if str(row.get("algoStatus") or "").upper() == "NEW"
                        and str(row.get("orderType") or "").upper() == "STOP_MARKET"]
        if len(active_stops) > 1:
            raise RuntimeError("multiple active STOP_MARKET orders make manual adoption ambiguous")

        old_stop = None
        if active_stops:
            row = active_stops[0]
            if (not self._is_bot_reduce_stop(row)
                    and str(row.get("reduceOnly")).strip().lower() not in {"true", "1"}):
                raise RuntimeError("existing STOP_MARKET is not a confirmed reduceOnly protective order")
            client_id = str(row.get("clientAlgoId") or "")
            stop_price = _number(row.get("triggerPrice"), float("nan"))
            quantity = _number(row.get("quantity"), float("nan"))
            expected_side = "SELL" if position.side is Side.LONG else "BUY"
            step, _, _ = self._market_rules(symbol)
            valid = (
                row.get("algoId") is not None
                and str(row.get("symbol") or "").upper() == str(market_id).upper()
                and str(row.get("side") or "").upper() == expected_side
                and math.isfinite(quantity)
                and abs(quantity - float(position.quantity)) <= max(step, 1e-10)
                and math.isfinite(stop_price) and stop_price > 0
                and ((position.side is Side.LONG and stop_price < market_price
                      and stop_price < position.entry_price)
                     or (position.side is Side.SHORT and stop_price > market_price
                         and stop_price > position.entry_price))
            )
            if not valid:
                raise RuntimeError("existing STOP_MARKET does not safely protect the current position")
            old_stop = {"id": str(row["algoId"]), "clientOrderId": client_id,
                        "stop_price": stop_price, "amount": quantity,
                        "external": not self._is_bot_reduce_stop(row),
                        "order_side": expected_side, "info": row}
            tighter = (stop_price >= candidate if position.side is Side.LONG
                       else stop_price <= candidate)
            if tighter:
                latest = self._position(symbol)
                if (latest is None or latest["side"] != position.side.value
                        or abs(float(latest["quantity"]) - float(position.quantity)) > max(step, 1e-10)
                        or abs(float(latest["entry_price"]) - float(position.entry_price))
                        > max(float(position.entry_price) * 1e-6, 1e-10)):
                    raise RuntimeError("exchange position changed during manual adoption")
                return old_stop

        new_stop = self._place_stop(symbol, position.side.value, live["quantity"], candidate)
        refreshed = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        verified = next((row for row in refreshed
                         if str(row.get("algoId")) == str(new_stop["id"])), None)
        saved = {**new_stop, "external": False,
                 "order_side": "SELL" if position.side is Side.LONG else "BUY"}
        if verified is None or not self._saved_stop_matches(
                symbol, position, saved, verified, float(live["quantity"])):
            try:
                self._cancel_stop(symbol, new_stop)
            finally:
                raise RuntimeError("new strategy protective stop could not be verified")
        latest = self._position(symbol)
        if (latest is None or latest["side"] != position.side.value
                or abs(float(latest["quantity"]) - float(position.quantity)) > max(step, 1e-10)
                or abs(float(latest["entry_price"]) - float(position.entry_price))
                > max(float(position.entry_price) * 1e-6, 1e-10)):
            raise RuntimeError("exchange position changed during manual adoption")
        if old_stop:
            old_row = next((row for row in refreshed
                            if str(row.get("algoId")) == str(old_stop["id"])), None)
            if old_row is not None and not self._saved_stop_matches(
                    symbol, position, old_stop, old_row, float(live["quantity"])):
                raise RuntimeError("existing protective stop changed during replacement")
            if old_row is not None:
                self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": old_stop["id"]})
                remaining = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
                if any(str(row.get("algoId")) == str(old_stop["id"]) for row in remaining):
                    raise RuntimeError("existing protective stop cancellation was not confirmed")
        return saved

    @staticmethod
    def _is_bot_reduce_stop(row: dict[str, Any]) -> bool:
        return (str(row.get("clientAlgoId") or "").startswith(f"{BOT_ORDER_PREFIX}_s_")
                and str(row.get("orderType") or "").upper() == "STOP_MARKET"
                and str(row.get("reduceOnly")).strip().lower() in {"true", "1"})

    def _retire_superseded_stops(self, symbol: str, side: str, active_stop: dict[str, Any] | None,
                                 expected_quantity: float) -> list[dict[str, Any]]:
        """Keep the verified remaining-position stop and remove older bot stops."""
        self._verify_urls()
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
        """Read-only verify each saved exchange position has its tracked protective stop."""
        self._verify_urls()
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
            valid = (
                active is not None
                and self._saved_stop_matches(symbol, position, stop, active,
                                             float(position.quantity))
            )
            if not valid:
                mismatches.append(f"{symbol}: saved protective stop is not active at the saved side/quantity/price")
                continue
            extras = [row for row in rows
                      if str(row.get("algoId")) != stop_id
                      and str(row.get("algoStatus") or "").upper() == "NEW"
                      and (self._is_bot_reduce_stop(row)
                           or (stop.get("external")
                               and str(row.get("orderType") or "").upper() == "STOP_MARKET"))]
            if extras:
                mismatches.append(f"{symbol}: extra active bot protective stop(s) found")
        return mismatches

    def _cancel_untracked_bot_stops(self, symbol: str) -> None:
        """Cancel only this adapter's conditional orders when recovering an unknown position."""
        self._verify_urls()
        market_id = self.exchange.market(symbol)["id"]
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        for row in rows:
            client_id = str(row.get("clientAlgoId") or "")
            algo_id = row.get("algoId")
            if client_id.startswith(f"{BOT_ORDER_PREFIX}_s_") and algo_id is not None:
                self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": algo_id})

    def cancel_saved_stop_if_open(self, symbol: str, stop: dict[str, Any] | None) -> None:
        """Cancel and verify one tracked protective stop after its position is flat."""
        if not stop or not stop.get("id"):
            return
        self._verify_urls()
        market_id = self.exchange.market(symbol)["id"]
        algo_id = str(stop["id"])
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        match = next((row for row in rows if str(row.get("algoId")) == algo_id), None)
        if match is None:
            return
        client_id = str(match.get("clientAlgoId") or "")
        is_bot = self._is_bot_reduce_stop(match)
        is_adopted = bool(stop.get("external")) and client_id == str(stop.get("clientOrderId") or "")
        expected_symbol = str(market_id).upper()
        expected_quantity = _number(stop.get("amount"), float("nan"))
        step, _, _ = self._market_rules(symbol)
        expected_price = _number(stop.get("stop_price"), float("nan"))
        try:
            expected_price = float(self.exchange.price_to_precision(symbol, expected_price))
        except (TypeError, ValueError, OverflowError):
            expected_price = float("nan")
        if (not (is_bot or is_adopted)
                or str(match.get("symbol") or "").upper() != expected_symbol
                or str(match.get("orderType") or "").upper() != "STOP_MARKET"
                or str(match.get("algoStatus") or "").upper() != "NEW"
                or str(match.get("reduceOnly")).strip().lower() not in {"true", "1"}
                or (stop.get("order_side") and str(match.get("side") or "").upper()
                    != str(stop["order_side"]).upper())
                or (is_adopted and (
                    abs(_number(match.get("quantity")) - expected_quantity) > max(step, 1e-10)
                    or not math.isfinite(expected_price)
                    or not math.isclose(_number(match.get("triggerPrice")), expected_price,
                                        rel_tol=1e-9, abs_tol=max(expected_price * 1e-9, 1e-12))
                ))):
            raise RuntimeError("saved stop id is not a confirmed tracked reduceOnly order")
        self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": algo_id})
        remaining = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
        if any(str(row.get("algoId")) == algo_id for row in remaining):
            raise RuntimeError("bot protective stop cancellation was not confirmed")

    def cancel_orphaned_bot_stops(self, live_symbols: set[str]) -> list[dict[str, str]]:
        """Cancel bot-owned reduceOnly stops only when the exchange confirms the symbol is flat."""
        self._verify_urls()
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
                    raise RuntimeError("cannot map an open bot stop to a known USDⓈ-M market")
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
        """Return a fully verified fill for a tracked reduceOnly protective stop."""
        if not stop or not stop.get("id"):
            return None
        self._verify_urls()
        market_id = self.exchange.market(symbol)["id"]
        algo = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])})
        reduce_only = str(algo.get("reduceOnly")).strip().lower() in {"true", "1"}
        client_id = str(algo.get("clientAlgoId") or "")
        tracked_client = (client_id.startswith(f"{BOT_ORDER_PREFIX}_s_")
                          or (bool(stop.get("external"))
                              and client_id == str(stop.get("clientOrderId") or "")))
        algo_symbol = algo.get("symbol")
        if (str(algo.get("actualOrderId") or "") != str(order_id)
                or str(algo.get("algoId") or stop["id"]) != str(stop["id"])
                or (algo_symbol is not None and str(algo_symbol).upper() != str(market_id).upper())
                or (stop.get("external") and algo_symbol is None)
                or not tracked_client
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
        """Resolve the triggered exchange order id for a tracked stop, then verify its fills."""
        if not stop or not stop.get("id"):
            return None
        self._verify_urls()
        algo = self.exchange.fapiPrivateGetAlgoOrder({"algoId": str(stop["id"])})
        order_id = algo.get("actualOrderId")
        if not order_id:
            return None
        return self.resolve_protective_stop_fill(
            symbol, str(order_id), stop, expected_quantity,
        )

    def resolve_exchange_flat_close_fill(self, symbol: str, side: Side,
                                         opened_at: datetime, observed_flat_at: datetime,
                                         expected_quantity: float) -> dict[str, Any] | None:
        """Verify a full manual/external close from bounded USD-M user-trade history."""
        if (opened_at.tzinfo is None or observed_flat_at.tzinfo is None
                or observed_flat_at < opened_at or not math.isfinite(expected_quantity)
                or expected_quantity <= 0):
            return None
        if (observed_flat_at - opened_at > timedelta(days=7)
                or opened_at < _utc_now() - timedelta(days=90)):
            return None
        try:
            self._verify_urls()
            market_id = self.exchange.market(symbol)["id"]
            start_ms = int(opened_at.timestamp() * 1000)
            end_ms = int(observed_flat_at.timestamp() * 1000)
            rows = self.exchange.fapiPrivateGetUserTrades({
                "symbol": market_id, "startTime": start_ms, "endTime": end_ms,
                "limit": 1000,
            })
            if not isinstance(rows, list) or len(rows) >= 1000:
                return None
            expected_side = "BUY" if side is Side.SHORT else "SELL"
            closing_trades = []
            trade_ids = set()
            for row in rows:
                if not isinstance(row, dict):
                    return None
                if (row.get("symbol") is not None
                        and str(row["symbol"]).upper() != str(market_id).upper()):
                    return None
                timestamp = int(row.get("time") or 0)
                if timestamp < start_ms or timestamp > end_ms:
                    return None
                if str(row.get("positionSide") or "BOTH").upper() != "BOTH":
                    return None
                trade_side = str(row.get("side") or "").upper()
                if trade_side not in {"BUY", "SELL"}:
                    buyer = row.get("buyer")
                    if isinstance(buyer, bool):
                        trade_side = "BUY" if buyer else "SELL"
                    elif str(buyer).strip().lower() in {"true", "1"}:
                        trade_side = "BUY"
                    elif str(buyer).strip().lower() in {"false", "0"}:
                        trade_side = "SELL"
                    else:
                        return None
                if trade_side != expected_side:
                    continue
                try:
                    trade_id = str(row["id"])
                    order_id = str(row["orderId"])
                    quantity = float(row["qty"])
                    price = float(row["price"])
                    commission = float(row["commission"])
                    realized_pnl = float(row["realizedPnl"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return None
                if (not trade_id or not order_id or trade_id in trade_ids
                        or row.get("commissionAsset") != "USDT"
                        or not all(math.isfinite(value) for value in
                                   (quantity, price, commission, realized_pnl))
                        or quantity <= 0 or price <= 0):
                    return None
                trade_ids.add(trade_id)
                closing_trades.append({
                    "trade_id": trade_id, "order_id": order_id, "quantity": quantity,
                    "price": price, "commission": commission,
                    "realized_pnl": realized_pnl, "timestamp": timestamp,
                })
            if not closing_trades:
                return None
            quantity = math.fsum(row["quantity"] for row in closing_trades)
            step, _, _ = self._market_rules(symbol)
            tolerance = max(step * 1e-6, expected_quantity * 1e-8, 1e-12)
            if abs(quantity - expected_quantity) > tolerance:
                return None
            notional = math.fsum(row["quantity"] * row["price"] for row in closing_trades)
            if not math.isfinite(notional) or notional <= 0:
                return None
            timestamps = [row["timestamp"] for row in closing_trades]
            order_ids = sorted({row["order_id"] for row in closing_trades})
            return {
                "order_ids": order_ids, "quantity": quantity, "average": notional / quantity,
                "gross_pnl": math.fsum(row["realized_pnl"] for row in closing_trades),
                "exit_fee": math.fsum(row["commission"] for row in closing_trades),
                "closed_at": datetime.fromtimestamp(max(timestamps) / 1000, timezone.utc).isoformat(),
            }
        except Exception:
            return None

    def _inspect_missing_entry_attempt(self, attempt: dict[str, Any],
                                        market_id: str) -> dict[str, Any]:
        """Confirm absence only while Binance still retains unfilled order history."""
        def blocked(reason):
            return {"status": "missing_order_unconfirmed", "reason": reason}

        try:
            submitted = datetime.fromisoformat(str(attempt["submitted_at"]).replace("Z", "+00:00"))
            quantity = float(attempt["quantity"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return blocked("invalid_saved_submission")
        if (submitted.tzinfo is None or not math.isfinite(quantity) or quantity <= 0
                or attempt.get("side") not in {"long", "short"}
                or not str(attempt.get("clientOrderId", "")).startswith(f"{BOT_ORDER_PREFIX}_")
                or attempt.get("status") not in {"submitting", "uncertain"}
                or attempt.get("result")):
            return blocked("invalid_saved_submission")
        now = _utc_now()
        start = submitted - timedelta(minutes=1)
        # Allow REST propagation, and stay inside the three-day retention of
        # canceled/expired orders with no fills. Never infer absence after pruning.
        if now - submitted < timedelta(minutes=5) or now - start >= timedelta(days=3):
            return blocked("submission_outside_verifiable_window")
        symbol = attempt["symbol"]
        if self._position(symbol) is not None:
            return blocked("symbol_position_exists")
        window = {"symbol": market_id, "startTime": int(start.timestamp() * 1000),
                  "endTime": int(now.timestamp() * 1000), "limit": 1000}
        queries = (
            ("order_history", self.exchange.fapiPrivateGetAllOrders, window),
            ("trade_history", self.exchange.fapiPrivateGetUserTrades, window),
            ("open_orders", self.exchange.fapiPrivateGetOpenOrders, {"symbol": market_id}),
            ("open_algo_orders", self.exchange.fapiPrivateGetOpenAlgoOrders, {"symbol": market_id}),
        )
        for name, query, params in queries:
            rows = query(dict(params))
            if not isinstance(rows, list):
                return blocked(f"{name}_invalid_response")
            # Any same-symbol activity is ambiguous, including a filled-then-flat
            # position or another client order. A full page must never mean empty.
            if rows:
                return blocked(f"{name}_present")
        if self._position(symbol) is not None:
            return blocked("symbol_position_changed")
        return {"status": "not_filled", "evidence": {
            "kind": "missing_order_verified_empty", "clientOrderId": attempt["clientOrderId"],
            "history_start_at": start.isoformat(), "history_end_at": now.isoformat(),
            "checked_at": _utc_now().isoformat(), "order_count": 0, "trade_count": 0,
            "open_order_count": 0, "open_algo_order_count": 0, "flat_position_checks": 2,
        }}

    def inspect_entry_attempt(self, attempt: dict[str, Any]) -> dict[str, Any]:
        """Read-only recovery by the exact saved order identity; never resend."""
        self._verify_urls()
        symbol = attempt["symbol"]
        market_id = self.exchange.market(symbol)["id"]
        try:
            order = self.exchange.fapiPrivateGetOrder({
                "symbol": market_id, "origClientOrderId": attempt["clientOrderId"],
            })
        except OrderNotFound:
            return self._inspect_missing_entry_attempt(attempt, market_id)
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
        saved_stop = {"id": str(stop["algoId"]), "clientOrderId": stop.get("clientAlgoId"),
                      "stop_price": _number(stop["triggerPrice"]), "amount": filled}
        safety = (attempt.get("audit") or {}).get("capital_safety")
        if self.mode == "lead" and safety is not None:
            # A restart between fill and readback must not bypass the new-entry guard.
            identity = SimpleNamespace(symbol=symbol, side=Side(attempt["side"]))
            safety = dict(safety)
            live = self._check_or_exit_new_position(identity, saved_stop, safety)
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
            "stop": saved_stop,
            "capital_safety": safety,
            "order_submitted_at": attempt["submitted_at"],
        }}

    def open(self, intent, *, stop_price: float,
             before_submit: Callable | None = None,
             submission_deadline: datetime | None = None,
             client_order_id: str | None = None,
             on_submit: Callable | None = None,
             on_safety_plan: Callable | None = None) -> dict[str, Any]:
        _require_entry_clock()
        if intent.action is not Action.OPEN:
            raise ValueError("open executor accepts only OPEN intents")
        if stop_price is None or not math.isfinite(stop_price) or stop_price <= 0:
            raise ValueError("opening intent has no valid protective stop")
        if submission_deadline is not None and _utc_now() > submission_deadline:
            raise EntryPreflightRejected("entry_signal_expired")
        self._verify_urls()
        self._ensure_one_way()
        if self._position(intent.symbol) is not None:
            raise RuntimeError("exchange position already exists for symbol")
        preparation = self._prepare_entry(intent)
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
        safety = self._finalize_entry(replace(intent, quantity=amount, stop_price=stop_price), preparation)
        if safety is not None and on_safety_plan is not None:
            on_safety_plan(safety)
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
        _require_entry_clock()
        fill = self._market_fill(intent.symbol, side, amount, reduce_only=False, **fill_kwargs)
        live_position = self._position(intent.symbol)
        for delay in (.2, .5):
            if live_position is not None:
                break
            time.sleep(delay)
            live_position = self._position(intent.symbol)
        if live_position is None or live_position["side"] != intent.side.value:
            raise RuntimeError("market fill did not produce the expected exchange position")
        if step > 0 and abs(live_position["quantity"] - fill["filled"]) > step + 1e-12:
            raise RuntimeError("exchange position quantity differs from the confirmed fill")
        logger.info("{} OPEN_FILL symbol={} side={} amount={} orderId={} host={} source={} reason={} average={} stop_status=pending",
                    self.label, intent.symbol, intent.side.value, live_position["quantity"], fill["id"],
                    self.host, intent.source, intent.reason, live_position["entry_price"])
        if ((intent.side is Side.LONG and stop_price >= live_position["entry_price"])
                or (intent.side is Side.SHORT and stop_price <= live_position["entry_price"])):
            try:
                self._market_fill(intent.symbol, "sell" if intent.side is Side.LONG else "buy",
                                  live_position["quantity"], reduce_only=True)
            except Exception as exc:
                raise RuntimeError("fill moved beyond planned stop and emergency close failed; account is inconsistent") from exc
            raise RuntimeError("fill moved beyond planned stop; position was emergency-closed")
        try:
            stop = self._place_stop(intent.symbol, intent.side.value, live_position["quantity"], stop_price)
        except Exception as stop_error:
            try:
                self._market_fill(intent.symbol, "sell" if intent.side is Side.LONG else "buy",
                                  live_position["quantity"], reduce_only=True)
            except Exception as close_error:
                raise RuntimeError("protective stop placement failed and emergency reduceOnly close failed; account is inconsistent") from close_error
            raise RuntimeError("protective stop placement failed; position was emergency-closed") from stop_error
        if safety is not None:
            # Keep the strategy stop active throughout readback and any reduceOnly safety exit.
            live_position = self._check_or_exit_new_position(intent, stop, safety)
        return {"fill": fill, "position": live_position, "stop": stop,
                "capital_safety": safety,
                "order_submitted_at": submitted_at.isoformat(),
                "host": self.host, "reason": intent.reason, "regime": intent.regime.value}

    def close(self, position, *, stop: dict[str, Any] | None = None,
              reason: str | None = None) -> dict[str, Any]:
        symbol = position.symbol
        orders = self._close_orders.setdefault(symbol, {})
        attempt = orders.get("close_attempt")
        if attempt and (attempt["side"] != position.side.value
                        or attempt["opened_at"] != position.opened_at.isoformat()):
            raise RuntimeError("close attempt belongs to a different position")
        live_position = self._position(symbol)
        if live_position is None:
            if attempt and attempt["status"] != "prepared" and not attempt.get("fill"):
                attempt["fill"] = self._market_fill(
                    symbol, "sell" if position.side is Side.LONG else "buy",
                    attempt["quantity"], reduce_only=True,
                    client_order_id=attempt["clientOrderId"], lookup_only=True,
                    allow_unfilled=True,
                )
                self._save_close_attempt()
            self.cancel_saved_stop_if_open(symbol, stop)
            if self._close_checkpoint is None:
                self._close_orders.pop(symbol, None)
            if (attempt and attempt.get("fill", {}).get("filled", 0) > 0
                    and math.isclose(attempt["fill"]["filled"], position.quantity, rel_tol=1e-8)):
                return {"fill": attempt["fill"], "position": None, "host": self.host}
            return {"already_flat": True, "position": None, "host": self.host}
        if live_position["side"] != position.side.value:
            raise RuntimeError("exchange side differs from local position; account is inconsistent")
        if attempt is None:
            attempt = {"clientOrderId": f"{BOT_ORDER_PREFIX}_{uuid4().hex[:28]}",
                       "quantity": live_position["quantity"], "side": position.side.value,
                       "opened_at": position.opened_at.isoformat(),
                       "reason": reason, "status": "prepared"}
            orders["close_attempt"] = attempt
        lookup_only = attempt["status"] != "prepared"
        if not attempt.get("fill"):
            attempt["status"] = "submitted"
            # Persist the identity before sending. A failed save sends no order;
            # a restart after a successful save only queries this same identity.
            try:
                self._save_close_attempt()
            except Exception:
                if not lookup_only:
                    attempt["status"] = "prepared"
                raise
            try:
                attempt["fill"] = self._market_fill(
                    symbol, "sell" if position.side is Side.LONG else "buy",
                    attempt["quantity"], reduce_only=True,
                    client_order_id=attempt["clientOrderId"], lookup_only=lookup_only,
                )
            except Exception:
                if self._position(symbol) is None:
                    return self.close(position, stop=stop, reason=reason)
                raise
            self._save_close_attempt()
        fill = attempt["fill"]
        remaining = self._position(symbol)
        if remaining is not None:
            raise RuntimeError("reduceOnly close was sent but exchange position is still open")
        self.cancel_saved_stop_if_open(symbol, stop)
        if self._close_checkpoint is None:
            self._close_orders.pop(symbol, None)
        if not math.isclose(fill["filled"], position.quantity, rel_tol=1e-8):
            return {"already_flat": True, "position": None, "host": self.host}
        logger.info("{} CLOSE symbol={} side={} amount={} orderId={} host={} source={} reason={} average={}",
                    self.label, symbol, position.side.value, fill["filled"], fill["id"], self.host, position.source,
                    reason or "strategy_close", fill["average"])
        return {"fill": fill, "position": None, "host": self.host}

    def reduce(self, position, quantity: float, *, stop: dict[str, Any] | None,
               reason: str | None = None) -> dict[str, Any]:
        """Reduce a Demo position while keeping its remaining quantity protected."""
        symbol = position.symbol
        self._verify_urls()
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
                raise RuntimeError("USDⓈ-M partial reduction unexpectedly closed or reversed the position")
            if abs(float(updated["quantity"]) - remaining) > max(step, 1e-10):
                raise RuntimeError("USDⓈ-M remaining position quantity differs from expected partial reduction")
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
            logger.error("{} partial fill confirmed for {} but superseded stop cleanup failed: {}",
                         self.label, symbol, cleanup_error)
        return {"fill": fill, "position": updated, "stop": replacement, "host": self.host,
                "reason": reason, "stale_stops": stale_stops,
                "stop_cleanup_error": cleanup_error}

    def replace_stop(self, position, *, stop: dict[str, Any] | None,
                     stop_price: float) -> dict[str, Any]:
        self._verify_urls()
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
                and (self._is_bot_reduce_stop(current)
                     or (bool(stop.get("external")
                         and str(current.get("clientAlgoId") or "")
                         == str(stop.get("clientOrderId") or ""))))
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
                raise RuntimeError(f"cannot replace an unverified or stale {self.label} protective stop")
            current_stop_price = actual_price
        else:
            current_stop_price = _number(position.stop_price, float("nan"))
            current_stop_price = float(self.exchange.price_to_precision(symbol, current_stop_price))
            if not math.isfinite(current_stop_price) or current_stop_price <= 0:
                raise RuntimeError(f"cannot replace a {self.label} stop without a valid existing risk stop")

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
        if stop and stop.get("external"):
            market_id = self.exchange.market(symbol)["id"]
            rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id}) or []
            saved = {**new_stop, "external": False,
                     "order_side": "SELL" if position.side is Side.LONG else "BUY"}
            active = next((row for row in rows
                           if str(row.get("algoId")) == str(new_stop["id"])), None)
            if active is None or not self._saved_stop_matches(
                    symbol, position, saved, active, float(live_position["quantity"])):
                try:
                    self._cancel_stop(symbol, new_stop)
                finally:
                    raise RuntimeError("replacement strategy stop could not be verified")
            # Keep the verified new protection if the user's old order cannot be
            # conclusively retired; the caller will pause management for review.
            self._cancel_stop(symbol, stop)
            return saved
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
        self._verify_urls()
        for symbol in sorted(symbols):
            try:
                trades = self.exchange.fetch_my_trades(symbol, since=None, limit=1000)
            except Exception as exc:
                logger.warning("{} fill audit unavailable for {}: {}", self.label, symbol, type(exc).__name__)
                continue
            for trade in trades:
                info = trade.get("info") or {}
                order_id = str(trade.get("order") or info.get("orderId") or "")
                if order_id and order_id not in known_order_ids and order_id not in seen_foreign_order_ids:
                    warnings.append({"symbol": symbol, "orderId": order_id})
        return warnings


class LeadPortfolioExecutionService(DemoExecutionService):
    """Production execution for a dedicated Binance Lead Portfolio account."""

    def __init__(self, exchange, settings):
        super().__init__(exchange, settings, mode="lead", auto_one_way=False)

    def _prepare_entry(self, intent):
        self._verify_urls()
        market_id = self.exchange.market(intent.symbol)["id"]
        try:
            brackets = parse_brackets(self.exchange.fapiPrivateGetLeverageBracket({
                "symbol": market_id,
            }), market_id)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EntryPreflightRejected("lead_leverage_brackets_invalid") from exc
        config = self.exchange.fapiPrivateGetSymbolConfig({"symbol": market_id})
        if (not isinstance(config, list) or len(config) != 1
                or config[0].get("symbol") != market_id
                or str(config[0].get("isAutoAddMargin")).lower() not in {"false", "0"}):
            raise EntryPreflightRejected("lead_auto_add_margin_unverified")
        self._prepare_margin(intent.symbol)
        return brackets

    def _finalize_entry(self, intent, preparation):
        self._verify_urls()
        balance = self.exchange.fetch_balance()
        available = (balance.get("info") or {}).get("availableBalance")
        if available is None:
            available = (balance.get("USDT") or {}).get("free")
        try:
            safety = select_leverage(intent.side, intent.price, intent.quantity, intent.stop_price,
                                     preparation, self.settings, available)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EntryPreflightRejected("lead_no_safe_leverage_or_margin") from exc
        if self._position(intent.symbol) is not None:
            raise EntryPreflightRejected("lead_position_appeared_before_leverage_change")
        result = self.exchange.set_leverage(safety["leverage"], intent.symbol)
        try:
            actual = float(result["leverage"])
            limit = float(result["maxNotionalValue"])
            if (not math.isfinite(actual) or actual != safety["leverage"]
                    or not math.isfinite(limit) or limit < intent.notional):
                raise ValueError("unconfirmed leverage/notional limit")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EntryPreflightRejected("lead_leverage_confirmation_failed") from exc
        return safety

    def _verify_new_position(self, intent, live_position, stop, safety):
        if (live_position is None or live_position["side"] != intent.side.value
                or live_position.get("margin_mode") != "isolated"
                or not math.isclose(float(live_position["quantity"]), float(stop["amount"]), rel_tol=1e-8)
                or not liquidation_is_safe(intent.side, live_position["entry_price"], stop["stop_price"],
                                           live_position.get("liquidation_price"), self.settings.lead_risk)):
            raise RuntimeError("new Lead position liquidation buffer is unverified or insufficient")
        market_id = self.exchange.market(intent.symbol)["id"]
        config = self.exchange.fapiPrivateGetSymbolConfig({"symbol": market_id})
        if (not isinstance(config, list) or len(config) != 1
                or config[0].get("symbol") != market_id
                or float(config[0].get("leverage", 0)) != safety["leverage"]
                or str(config[0].get("isAutoAddMargin")).lower() not in {"false", "0"}):
            raise RuntimeError("new Lead position leverage or auto margin readback failed")
        position = SimpleNamespace(side=intent.side, quantity=live_position["quantity"])
        rows = self.exchange.fapiPrivateGetOpenAlgoOrders({"symbol": market_id})
        active = [row for row in rows if str(row.get("algoStatus")).upper() == "NEW"
                  and str(row.get("orderType")).upper() == "STOP_MARKET"]
        if (len(active) != 1 or active[0].get("workingType") != "CONTRACT_PRICE"
                or not self._saved_stop_matches(intent.symbol, position, stop, active[0], position.quantity)):
            raise RuntimeError("new Lead position protective stop readback failed")
        safety["actual_liquidation_price"] = float(live_position["liquidation_price"])
        safety["verified"] = True
