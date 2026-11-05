"""Connectivity and safety check for Binance USDⓈ-M Demo Trading."""

from bot.config import load_settings
from bot.exchange.binance_usdm import DEMO_REST_ROOT, DEMO_WS_ROOT, create_demo_usdm, resolve_execution_mode
from bot.execution import DemoExecutionService


def main() -> int:
    settings = load_settings()
    try:
        mode = resolve_execution_mode(bot_mode=settings.runtime.bot_mode,
                                      testnet=settings.runtime.binance_testnet)
        print(f"TESTNET={str(mode.testnet).lower()} BOT_MODE={mode.bot_mode} host={DEMO_REST_ROOT if mode.testnet else 'local-only'}")
        if mode.name != "demo_usdm":
            raise RuntimeError("ping_exchange requires BINANCE_TESTNET=true; local-only mode makes no private exchange calls")
        print(f"ws={DEMO_WS_ROOT}")
        exchange = create_demo_usdm(settings.runtime)
        try:
            markets = exchange.load_markets()
            service = DemoExecutionService(exchange, settings)
            snapshot = service.initialize()
            position_mode = exchange.fetch_position_mode()
            print(f"USDT={snapshot['usdt_balance']:.8f} available={snapshot['available_usdt']:.8f}")
            print(f"markets={len(markets)} position_mode={'hedge' if position_mode.get('hedged') else 'one-way'}")
        finally:
            exchange.close()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
