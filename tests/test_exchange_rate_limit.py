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


def test_public_and_private_clients_share_weighted_request_pacing(tmp_path, monkeypatch):
    clock = [1000.0]
    gate = network.BinanceRateLimitGate(tmp_path / "cooldown.json", wall_time=lambda: clock[0])
    pacer = network.BinanceRequestPacer(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(network, "_RATE_LIMIT_GATE", gate)
    monkeypatch.setattr(network, "_REQUEST_PACER", pacer)
    requests = []

    def client():
        return SimpleNamespace(
            id="binanceusdm", session=SimpleNamespace(proxies={}), rateLimit=50,
            calculate_rate_limiter_cost=lambda api, method, path, params, config: config["cost"],
            request=lambda *args: requests.append((args[0], clock[0])),
        )

    public, private = client(), client()
    network.configure_system_proxy(public)
    network.configure_system_proxy(private)
    public.request("ticker/24hr", "fapiPublic", config={"cost": 40})
    private.request("order", "fapiPrivate", "POST", config={"cost": 0})
    private.request("copyTrading/futures/userStatus", "sapi", config={"cost": 3000})
    private.request("userTrades", "fapiPrivate", config={"cost": 5})
    public.request("time", "fapiPublic", config={"cost": 1})
    assert requests == [("ticker/24hr", 1000.0), ("order", 1000.0),
                        ("copyTrading/futures/userStatus", 1000.0),
                        ("userTrades", 1002.0), ("time", 1002.25)]


def test_pacing_rechecks_a_cooldown_started_while_waiting(tmp_path, monkeypatch):
    clock = [1000.0]
    gate = network.BinanceRateLimitGate(tmp_path / "cooldown.json", wall_time=lambda: clock[0])
    monkeypatch.setattr(network, "_RATE_LIMIT_GATE", gate)
    calls = []

    def sleep(seconds):
        clock[0] += seconds
        gate.trip(ccxt.DDoSProtection("418"))

    monkeypatch.setattr(network, "_REQUEST_PACER", network.BinanceRequestPacer(
        monotonic=lambda: clock[0], sleep=sleep))
    client = SimpleNamespace(
        id="binanceusdm", session=SimpleNamespace(proxies={}), rateLimit=50,
        calculate_rate_limiter_cost=lambda *args: 40,
        request=lambda *args: calls.append(args),
    )
    network.configure_system_proxy(client)
    client.request("ticker/24hr", "fapiPublic")
    with pytest.raises(network.BinanceRateLimitCooldown):
        client.request("account", "fapiPrivate")
    assert len(calls) == 1


def test_ban_retry_after_and_safe_endpoint_metadata_are_preserved(tmp_path):
    gate = network.BinanceRateLimitGate(tmp_path / "cooldown.json", wall_time=lambda: 1000.0)
    until = gate.trip(
        ccxt.DDoSProtection("HTTP 418 signature=private_value"),
        response_headers={"Retry-After": "900", "X-MBX-USED-WEIGHT-1M": "2395"},
        endpoint="fapiPrivate:GET:account?signature=private_value",
    )
    assert until == 1_900_000
    saved = json.loads((tmp_path / "cooldown.json").read_text(encoding="utf-8"))
    assert saved["endpoint"] == "fapiPrivate:GET:account"
    assert saved["http_status"] == 418
    assert saved["used_weight_1m"] == 2395
    assert "private_value" not in json.dumps(saved)
