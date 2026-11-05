"""USDⓈ-M Demo Trading client with an explicit host lock."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import ccxt

from bot.exchange.network import configure_system_proxy


DEMO_REST_ROOT = "https://demo-fapi.binance.com"
DEMO_WS_ROOT = "wss://demo-fstream.binance.com"
FAPI_URLS = {
    "fapiPublic": f"{DEMO_REST_ROOT}/fapi/v1",
    "fapiPublicV2": f"{DEMO_REST_ROOT}/fapi/v2",
    "fapiPublicV3": f"{DEMO_REST_ROOT}/fapi/v3",
    "fapiPrivate": f"{DEMO_REST_ROOT}/fapi/v1",
    "fapiPrivateV2": f"{DEMO_REST_ROOT}/fapi/v2",
    "fapiPrivateV3": f"{DEMO_REST_ROOT}/fapi/v3",
    "fapiData": f"{DEMO_REST_ROOT}/futures/data",
}


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    testnet: bool
    bot_mode: str


def resolve_execution_mode(*, bot_mode: str, testnet: bool, live_flag: bool = False) -> ExecutionMode:
    """Enforce the allowed local-Paper and USDⓈ-M Demo mode matrix."""
    normalized = str(bot_mode).strip().lower()
    if live_flag or normalized == "live":
        raise RuntimeError("live mode and --live are disabled; no production trading is available")
    if normalized != "paper":
        raise RuntimeError("BOT_MODE must be paper")
    if not isinstance(testnet, bool):
        raise RuntimeError("BINANCE_TESTNET must be true or false")
    return ExecutionMode("demo_usdm" if testnet else "local_paper", testnet, normalized)


def lock_demo_urls(exchange) -> None:
    """Enable CCXT demo mode, then pin and verify every USDⓈ-M URL explicitly."""
    enable = getattr(exchange, "enable_demo_trading", None)
    if not callable(enable):
        raise RuntimeError("installed CCXT lacks Binance demo-trading support")
    enable(True)
    api = exchange.urls.get("api")
    if not isinstance(api, dict):
        raise RuntimeError("CCXT Binance USDⓈ-M API URL map is unavailable")
    api.update(FAPI_URLS)
    verify_demo_urls(exchange)


def verify_demo_urls(exchange) -> None:
    api = exchange.urls.get("api", {})
    for key, expected in FAPI_URLS.items():
        value = api.get(key)
        if value != expected or "demo-fapi.binance.com" not in str(value).lower():
            raise RuntimeError(f"refusing private request: {key} is not locked to demo-fapi.binance.com")
    if any("fapi.binance.com" in str(api.get(key, "")).lower()
           and "demo-fapi.binance.com" not in str(api.get(key, "")).lower()
           for key in FAPI_URLS):
        raise RuntimeError("refusing private request to production Binance Futures host")


def create_demo_usdm(runtime, *, exchange_factory=None):
    """Build a private USDⓈ-M Demo client without exposing credentials in logs."""
    key = runtime.binance_api_key.get_secret_value().strip()
    secret = runtime.binance_api_secret.get_secret_value().strip()
    if not runtime.binance_testnet or runtime.bot_mode != "paper":
        raise RuntimeError("Demo client requires BINANCE_TESTNET=true and BOT_MODE=paper")
    if not key or not secret:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required for Demo execution")
    factory = exchange_factory or ccxt.binanceusdm
    exchange = factory({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSubType": "linear"},
    })
    try:
        lock_demo_urls(exchange)
        configure_system_proxy(exchange)
        return exchange
    except Exception:
        exchange.close()
        raise


def parse_live_position(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one non-zero CCXT USDⓈ-M position snapshot."""
    contracts = row.get("contracts")
    if contracts is None:
        contracts = row.get("info", {}).get("positionAmt")
    try:
        quantity = abs(float(contracts))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    side = row.get("side")
    if side not in ("long", "short"):
        amount = float(row.get("info", {}).get("positionAmt", 0.0))
        side = "long" if amount > 0 else "short"
    entry = row.get("entryPrice")
    if entry is None:
        entry = row.get("info", {}).get("entryPrice")
    unrealized = row.get("unrealizedPnl")
    if unrealized is None:
        unrealized = row.get("info", {}).get("unRealizedProfit", 0.0)
    try:
        entry, unrealized = float(entry), float(unrealized or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(entry) or entry <= 0 or not math.isfinite(unrealized):
        return None
    return {
        "symbol": row.get("symbol") or row.get("info", {}).get("symbol"), "side": side, "quantity": quantity,
        "entry_price": entry, "unrealized_pnl": unrealized,
        "margin_mode": row.get("marginMode") or row.get("info", {}).get("marginType"),
        "leverage": row.get("leverage") or row.get("info", {}).get("leverage"),
        "initial_margin": row.get("initialMargin"), "info": row.get("info", {}),
    }
