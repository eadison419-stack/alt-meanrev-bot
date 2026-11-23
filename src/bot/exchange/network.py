"""Shared proxy and IP-wide Binance request cooldown helpers."""

from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

import ccxt


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RATE_LIMIT_STATE_PATH = _PROJECT_ROOT / "data" / "binance_rate_limit_cooldown.json"
_BAN_UNTIL = re.compile(r"banned\s+until\s+(\d{13})", re.IGNORECASE)
_REQUEST_LOCK = threading.Lock()


class BinanceRequestPacer:
    """Coordinate CCXT request weights across clients in the same process."""

    def __init__(self, *, monotonic=time.monotonic, sleep=time.sleep):
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_at = 0.0

    def wait(self, cost: float, rate_limit_ms: float) -> None:
        delay = max(0.0, self._next_at - self._monotonic())
        if delay:
            self._sleep(delay)
        self._next_at = self._monotonic() + cost * rate_limit_ms / 1000


_REQUEST_PACER = BinanceRequestPacer()


class BinanceRateLimitCooldown(ccxt.RateLimitExceeded):
    """Raised locally while Binance's IP-based rate-limit cooldown is active."""

    def __init__(self, blocked_until_ms: int, reason: str = "rate_limit_cooldown"):
        self.blocked_until_ms = int(blocked_until_ms)
        self.reason = reason
        super().__init__(f"Binance requests suppressed until {self.blocked_until_ms}: {reason}")


class BinanceRateLimitGate:
    """Share 418/429 backoff across CCXT clients and local processes on this host."""

    def __init__(self, state_path: str | Path = RATE_LIMIT_STATE_PATH, *, wall_time=time.time):
        self.state_path = Path(state_path)
        self.wall_time = wall_time
        self._lock = threading.RLock()
        self._blocked_until_ms = 0

    def _read(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("binance_rate_limit_state_unavailable") from exc
        if not isinstance(state, dict):
            raise RuntimeError("binance_rate_limit_state_invalid")
        return state

    def blocked_until_ms(self) -> int:
        with self._lock:
            raw = self._read().get("blocked_until_ms", 0) or 0
            try:
                persisted = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("binance_rate_limit_state_invalid") from exc
            return max(self._blocked_until_ms, persisted)

    def remaining_seconds(self) -> float:
        try:
            return max(0.0, self.blocked_until_ms() / 1000 - self.wall_time())
        except Exception:
            # A damaged or unreadable cooldown marker must not trigger another request.
            return 300.0

    def ensure_allowed(self) -> None:
        now_ms = int(self.wall_time() * 1000)
        try:
            blocked_until = self.blocked_until_ms()
        except Exception:
            blocked_until = now_ms + 300_000
        if blocked_until > now_ms:
            raise BinanceRateLimitCooldown(blocked_until)

    def _retry_after_seconds(self, headers) -> float | None:
        if not hasattr(headers, "items"):
            return None
        raw = next((value for key, value in headers.items()
                    if str(key).lower() == "retry-after"), None)
        if raw is None:
            return None
        try:
            seconds = float(raw)
            return seconds if math.isfinite(seconds) and seconds > 0 else None
        except (TypeError, ValueError, OverflowError):
            try:
                retry_at = parsedate_to_datetime(str(raw))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, retry_at.timestamp() - self.wall_time())
            except (TypeError, ValueError, OverflowError):
                return None

    def trip(self, error: Exception, *, response_headers=None, endpoint=None) -> int:
        """Persist a conservative cooldown; return its absolute Unix timestamp in ms."""
        now_ms = int(self.wall_time() * 1000)
        with self._lock:
            try:
                previous = self._read()
            except RuntimeError:
                previous = {}
            try:
                retry_count = max(0, int(previous.get("retry_count", 0) or 0)) + 1
            except (TypeError, ValueError, OverflowError):
                retry_count = 1
            try:
                persisted_until = int(previous.get("blocked_until_ms", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                persisted_until = 0
            previous_until = max(self._blocked_until_ms, persisted_until)
            error_type = type(error).__name__
            retry_after = self._retry_after_seconds(response_headers)
            if isinstance(error, ccxt.DDoSProtection):
                match = _BAN_UNTIL.search(str(error))
                until_ms = int(match.group(1)) if match else now_ms + 300_000
            else:
                fallback = min(1800, 60 * (2 ** min(retry_count - 1, 5)))
                until_ms = now_ms + int(max(fallback, retry_after or 0.0) * 1000)
            until_ms = max(until_ms, now_ms + int((retry_after or 0.0) * 1000))
            until_ms = max(previous_until, until_ms, now_ms + 1_000)
            self._blocked_until_ms = until_ms
            state = {
                "version": 1,
                "blocked_until_ms": until_ms,
                "retry_count": retry_count,
                "error_type": error_type,
                "updated_at_ms": now_ms,
            }
            if endpoint:
                state["endpoint"] = str(endpoint).split("?", 1)[0][:128]
            status = re.search(r"\b(403|418|429)\b", str(error))
            if status:
                state["http_status"] = int(status.group(1))
            if hasattr(response_headers, "items"):
                weight = next((value for key, value in response_headers.items()
                               if str(key).lower() == "x-mbx-used-weight-1m"), None)
                if str(weight).isdigit():
                    state["used_weight_1m"] = int(weight)
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = self.state_path.with_name(
                    f"{self.state_path.name}.{os.getpid()}.tmp"
                )
                temporary_path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
                temporary_path.replace(self.state_path)
            except OSError:
                # The in-process gate remains active even if the shared marker cannot be written.
                pass
            return until_ms


_RATE_LIMIT_GATE = BinanceRateLimitGate()


def rate_limit_cooldown_remaining() -> float:
    """Return the remaining host-wide Binance cooldown, without making a request."""
    return _RATE_LIMIT_GATE.remaining_seconds()


def configure_system_proxy(exchange) -> None:
    """Apply the OS proxy and install a shared Binance 418/429 circuit breaker."""
    proxies = urllib.request.getproxies()
    exchange.session.proxies.update({key: value for key, value in proxies.items() if key in ("http", "https")})
    if getattr(exchange, "id", None) != "binanceusdm" or getattr(exchange, "_bot_rate_limit_guarded", False):
        return
    original_request = getattr(exchange, "request", None)
    if not callable(original_request):
        return
    exchange.enableLastResponseHeaders = True

    def guarded_request(path, api="public", method="GET", params=None,
                        headers=None, body=None, config=None):
        # Serialize this process's independently rate-limited CCXT clients and
        # re-read the shared marker so Dashboard and Runner processes coordinate.
        with _REQUEST_LOCK:
            _RATE_LIMIT_GATE.ensure_allowed()
            cost_function = getattr(exchange, "calculate_rate_limiter_cost", None)
            if method == "GET" and str(api).startswith("fapi") and callable(cost_function):
                cost = float(cost_function(api, method, path, params or {}, config or {}))
                rate_limit_ms = float(exchange.rateLimit)
                if not math.isfinite(cost) or cost < 0 or not math.isfinite(rate_limit_ms) or rate_limit_ms < 0:
                    raise RuntimeError("binance_request_weight_invalid")
                _REQUEST_PACER.wait(cost, rate_limit_ms)
                _RATE_LIMIT_GATE.ensure_allowed()
            try:
                return original_request(path, api, method, params or {}, headers, body, config or {})
            except BinanceRateLimitCooldown:
                raise
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
                _RATE_LIMIT_GATE.trip(
                    exc, response_headers=getattr(exchange, "last_response_headers", None),
                    endpoint=f"{api}:{method}:{path}",
                )
                raise

    exchange.request = guarded_request
    exchange._bot_rate_limit_guarded = True
