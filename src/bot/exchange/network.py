"""Network helpers shared by public Binance data downloaders."""

import urllib.request


def configure_system_proxy(exchange) -> None:
    """Pass OS-configured HTTP(S) proxies to CCXT's requests session when present."""
    proxies = urllib.request.getproxies()
    exchange.session.proxies.update({key: value for key, value in proxies.items() if key in ("http", "https")})
