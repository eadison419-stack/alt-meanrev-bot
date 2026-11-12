import json
from types import SimpleNamespace

import ccxt
import pytest

from bot.exchange import network


def test_binance_ban_until_is_shared_and_prevents_later_requests(tmp_path, monkeypatch):
    now = [1_790_000_000.0]
    gate = network.BinanceRateLimitGate(tmp_path / "cooldown.json", wall_time=lambda: now[0])
    monkeypatch.setattr(network, "_RATE_LIMIT_GATE", gate)
    calls = []
    ban_until_ms = int((now[0] + 900) * 1000)

    def blocked_request(*args, **kwargs):
        calls.append((args, kwargs))
        raise ccxt.DDoSProtection(f"HTTP 418 IP banned until {ban_until_ms}")

    def exchange(request):
        return SimpleNamespace(
            id="binanceusdm", session=SimpleNamespace(proxies={}), request=request,
        )

    first = exchange(blocked_request)
    network.configure_system_proxy(first)
    with pytest.raises(ccxt.DDoSProtection):
        first.request("ticker/24hr")

    saved = json.loads((tmp_path / "cooldown.json").read_text(encoding="utf-8"))
    assert saved["blocked_until_ms"] == ban_until_ms
    assert saved["error_type"] == "DDoSProtection"
    assert "IP" not in json.dumps(saved)

    second_calls = []
    second = exchange(lambda *args, **kwargs: second_calls.append((args, kwargs)))
    network.configure_system_proxy(second)
    with pytest.raises(network.BinanceRateLimitCooldown):
        second.request("ticker/24hr")
    assert len(calls) == 1
    assert second_calls == []

    now[0] += 901
    second.request("ticker/24hr")
    assert len(second_calls) == 1


def test_rate_limit_uses_retry_after_or_bounded_backoff(tmp_path):
    now = [1_000.0]
    gate = network.BinanceRateLimitGate(tmp_path / "cooldown.json", wall_time=lambda: now[0])

    server_delay = gate.trip(ccxt.RateLimitExceeded("429"), response_headers={"Retry-After": "120"})
    assert server_delay == 1_120_000

    now[0] = 2_000.0
    fallback = gate.trip(ccxt.RateLimitExceeded("429"))
    assert fallback == 2_120_000


def test_invalid_shared_marker_fails_closed_for_five_minutes(tmp_path):
    state_path = tmp_path / "cooldown.json"
    state_path.write_text("not-json", encoding="utf-8")
    gate = network.BinanceRateLimitGate(state_path, wall_time=lambda: 1_000.0)

    assert gate.remaining_seconds() == 300
    with pytest.raises(network.BinanceRateLimitCooldown):
        gate.ensure_allowed()
