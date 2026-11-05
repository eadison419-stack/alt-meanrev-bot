from types import SimpleNamespace

from bot.exchange.network import configure_system_proxy


def test_configure_system_proxy_applies_http_and_https_only(monkeypatch):
    monkeypatch.setattr(
        "bot.exchange.network.urllib.request.getproxies",
        lambda: {
            "http": "http://127.0.0.1:7892",
            "https": "http://127.0.0.1:7892",
            "ftp": "http://127.0.0.1:7892",
        },
    )
    exchange = SimpleNamespace(session=SimpleNamespace(proxies={}))

    configure_system_proxy(exchange)

    assert exchange.session.proxies == {
        "http": "http://127.0.0.1:7892",
        "https": "http://127.0.0.1:7892",
    }


def test_configure_system_proxy_is_noop_without_configured_proxy(monkeypatch):
    monkeypatch.setattr("bot.exchange.network.urllib.request.getproxies", lambda: {})
    exchange = SimpleNamespace(session=SimpleNamespace(proxies={"https": "https://explicit-proxy"}))

    configure_system_proxy(exchange)

    assert exchange.session.proxies == {"https": "https://explicit-proxy"}
