"""USDⓈ-M exchange clients with explicit Demo and production host locks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import ccxt

from bot.exchange.network import configure_system_proxy


DEMO_REST_ROOT = "https://demo-fapi.binance.com"
DEMO_WS_ROOT = "wss://demo-fstream.binance.com"
LIVE_REST_ROOT = "https://fapi.binance.com"
LIVE_SAPI_URL = "https://api.binance.com/sapi/v1"
DEMO_FAPI_URLS = {
    "fapiPublic": f"{DEMO_REST_ROOT}/fapi/v1",
    "fapiPublicV2": f"{DEMO_REST_ROOT}/fapi/v2",
    "fapiPublicV3": f"{DEMO_REST_ROOT}/fapi/v3",
    "fapiPrivate": f"{DEMO_REST_ROOT}/fapi/v1",
    "fapiPrivateV2": f"{DEMO_REST_ROOT}/fapi/v2",
    "fapiPrivateV3": f"{DEMO_REST_ROOT}/fapi/v3",
    "fapiData": f"{DEMO_REST_ROOT}/futures/data",
}
LIVE_FAPI_URLS = {key: value.replace(DEMO_REST_ROOT, LIVE_REST_ROOT)
                  for key, value in DEMO_FAPI_URLS.items()}
# Kept as a compatibility alias for existing Demo tests and scripts.
FAPI_URLS = DEMO_FAPI_URLS


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    testnet: bool
    bot_mode: str


def resolve_execution_mode(*, bot_mode: str, testnet: bool, live_flag: bool = False,
                           live_confirmed: bool = False) -> ExecutionMode:
    """Enforce local Paper, USDⓈ-M Demo, and explicitly confirmed Lead modes."""
    normalized = str(bot_mode).strip().lower()
    if live_flag or normalized == "live":
        if not live_flag:
            raise RuntimeError("live mode requires --live")
        if not live_confirmed:
            raise RuntimeError("live mode requires interactive human confirmation")
        if normalized != "live":
            raise RuntimeError("--live requires BOT_MODE=live")
        if testnet:
            raise RuntimeError("Lead Portfolio mode requires BINANCE_TESTNET=false")
        return ExecutionMode("lead_usdm", False, normalized)
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


def verify_live_urls(exchange) -> None:
    """Refuse signed USDⓈ-M requests unless every Futures URL is production-pinned."""
    api = exchange.urls.get("api", {})
    for key, expected in LIVE_FAPI_URLS.items():
        value = api.get(key)
        if value != expected or "fapi.binance.com" not in str(value).lower():
            raise RuntimeError(f"refusing private request: {key} is not locked to fapi.binance.com")
    if api.get("sapi") != LIVE_SAPI_URL:
        raise RuntimeError("refusing Lead status request: sapi is not locked to api.binance.com")
    if any("demo-fapi.binance.com" in str(api.get(key, "")).lower()
           for key in LIVE_FAPI_URLS):
        raise RuntimeError("refusing private request to Binance Demo Futures host")


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


def create_lead_usdm(runtime, *, exchange_factory=None):
    """Build a production USDⓈ-M client for a dedicated Lead Portfolio API key."""
    key = runtime.binance_api_key.get_secret_value().strip()
    secret = runtime.binance_api_secret.get_secret_value().strip()
    if runtime.binance_testnet or runtime.bot_mode != "live":
        raise RuntimeError("Lead client requires BINANCE_TESTNET=false and BOT_MODE=live")
    if not key or not secret:
        raise RuntimeError("Lead Portfolio API credentials are required for live execution")
    factory = exchange_factory or ccxt.binanceusdm
    exchange = factory({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSubType": "linear"},
    })
    try:
        api = exchange.urls.get("api")
        if not isinstance(api, dict):
            raise RuntimeError("CCXT Binance USDⓈ-M API URL map is unavailable")
        api.update(LIVE_FAPI_URLS)
        api["sapi"] = LIVE_SAPI_URL
        verify_live_urls(exchange)
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
