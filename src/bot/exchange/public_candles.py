"""Bounded parallel public candle reads with one unauthenticated client per thread."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import ccxt

from bot.exchange.network import configure_system_proxy


class CandleBatchReader:
    def __init__(self, exchange, now, *, factory=None, workers=4):
        self.exchange, self.now = exchange, now
        self.factory = factory or self._client
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="public-candles")
        self.local = threading.local()
        self.clients = []
        self.lock = threading.Lock()

    def _client(self):
        client = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 5000})
        configure_system_proxy(client)
        client.set_markets(self.exchange.markets)
        return client

    def _read(self, request, deadline, abort):
        if abort.is_set():
            raise ccxt.RateLimitExceeded("public_candle_batch_paused")
        remaining = (deadline - self.now()).total_seconds()
        if remaining <= 0:
            raise ccxt.RequestTimeout("public_candle_read_deadline")
        if not hasattr(self.local, "client"):
            self.local.client = self.factory()
            with self.lock:
                self.clients.append(self.local.client)
        client = self.local.client
        client.timeout = max(1, min(5000, int(remaining * 1000)))
        symbol, timeframe, limit = request
        try:
            return client.fetch_ohlcv(symbol, timeframe, limit=limit)
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.AuthenticationError):
            abort.set()
            raise

    def read_many(self, requests, deadline, *, monitor=None):
        abort = threading.Event()
        pending = {self.pool.submit(self._read, request, deadline, abort): request for request in dict.fromkeys(requests)}
        result = {}
        fatal = None
        # Results remain keyed by identity; strategy evaluation retains original order.
        for future in as_completed(pending):
            request = pending[future]
            try:
                result[request] = future.result()
            except Exception as exc:
                result[request] = exc
                if isinstance(exc, (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.AuthenticationError)):
                    fatal = fatal or exc
            if monitor is not None:
                monitor()
        if fatal is not None:
            fatal.mr_phase, fatal.mr_endpoint = "public_candles", "public"
            raise fatal
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for client in self.clients:
            client.close()


def prefetched_read(results, request, fallback):
    if request not in results:
        return fallback()
    value = results.pop(request)
    if isinstance(value, Exception):
        raise value
    return value
