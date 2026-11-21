"""Bounded Binance time calibration, advanced by monotonic time.

No OS clock changes. Trading guards use the conservative upper time bound so
calibration uncertainty can shorten, but never extend, an entry/quote window.
https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Check-Server-Time
"""

from datetime import datetime, timezone
import math
import time


class ExchangeClockError(RuntimeError):
    pass


class ExchangeClock:
    REFRESH_SECONDS = 60
    MAX_AGE_SECONDS = 300
    MAX_RTT_SECONDS = 1
    MAX_OFFSET_SECONDS = 30

    def __init__(self, *, monotonic=time.monotonic, wall_time=time.time):
        self._monotonic, self._wall_time = monotonic, wall_time
        self._anchor = None
        self._at = None
        self._rtt = None
        self._offset = None
        self._signature_anchor = None
        self._lower_anchor = None

    def synchronize(self, exchange):
        samples = []
        original_timeout = getattr(exchange, "timeout", None)
        if original_timeout is not None:
            exchange.timeout = min(original_timeout, 1000)
        try:
            return self._sample(exchange, samples)
        finally:
            if original_timeout is not None:
                exchange.timeout = original_timeout

    def _sample(self, exchange, samples):
        for _ in range(3):
            before = self._monotonic()
            try:
                response = exchange.fapiPublicGetTime({})
            except Exception:
                continue
            after = self._monotonic()
            server = response.get("serverTime") if isinstance(response, dict) else None
            try:
                server = float(server) / 1000
            except (TypeError, ValueError, OverflowError):
                continue
            rtt = after - before
            if (isinstance(response.get("serverTime"), bool) or not math.isfinite(server)
                    or server <= 0 or not 0 <= rtt <= self.MAX_RTT_SECONDS
                    or abs(server + rtt / 2 - self._wall_time()) > self.MAX_OFFSET_SECONDS):
                continue
            samples.append((rtt, server, after))
        if not samples:
            raise ExchangeClockError("exchange_clock_no_bounded_sample")
        rtt, server, at = min(samples)
        upper = server + rtt
        # Fresh clocks reject sudden changes. Once stale, require corroborating
        # bounded samples before replacing the signing clock after a long outage.
        if self._anchor is not None:
            expected = self._signature_anchor + at - self._at
            if abs(server + rtt / 2 - expected) > 1:
                projected = [value + delay / 2 + at - sampled_at
                             for delay, value, sampled_at in samples]
                if (self.age() <= self.MAX_AGE_SECONDS or len(samples) < 2
                        or max(projected) - min(projected) > self.MAX_RTT_SECONDS):
                    raise ExchangeClockError("exchange_clock_changed_abruptly")
        if self._anchor is not None:
            # Keep the entry deadline clock conservative and nondecreasing even
            # when a recovered signing clock needs to move backwards.
            upper = max(upper, self._anchor + at - self._at)
        self._anchor, self._at, self._rtt = upper, at, rtt
        self._signature_anchor = server + rtt / 2
        self._lower_anchor = server
        self._offset = server + rtt / 2 - self._wall_time() + (self._monotonic() - at)
        exchange.nonce = self.timestamp_ms
        return self.status()

    def age(self):
        return math.inf if self._at is None else self._monotonic() - self._at

    def require_fresh(self):
        if not 0 <= self.age() <= self.MAX_AGE_SECONDS:
            raise ExchangeClockError("exchange_clock_stale_or_unavailable")

    def now(self):
        if self._anchor is None:
            raise ExchangeClockError("exchange_clock_unavailable")
        return datetime.fromtimestamp(self._anchor + self._monotonic() - self._at, timezone.utc)

    def timestamp_ms(self):
        # Signed requests use the midpoint, safely inside Binance's 1s future limit.
        if self._signature_anchor is None:
            raise ExchangeClockError("exchange_clock_unavailable")
        return int((self._signature_anchor + self._monotonic() - self._at) * 1000)

    def lower_now(self):
        if self._lower_anchor is None:
            raise ExchangeClockError("exchange_clock_unavailable")
        return datetime.fromtimestamp(self._lower_anchor + self._monotonic() - self._at, timezone.utc)

    def status(self):
        return {"offset_ms": round(self._offset * 1000, 3),
                "rtt_ms": round(self._rtt * 1000, 3),
                "uncertainty_ms": round((self._anchor - self._lower_anchor) * 1000, 3),
                "age_sec": round(self.age(), 3), "max_age_sec": self.MAX_AGE_SECONDS}


_active_clock = None


def active_clock():
    return _active_clock


def activate_clock(clock):
    global _active_clock
    _active_clock = clock
