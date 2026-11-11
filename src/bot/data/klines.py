"""Public OHLCV ingestion and configured USDT-M symbol selection."""

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
import math
import statistics
from typing import Protocol

import pandas as pd


class OHLCVExchange(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None): ...

    def load_markets(self): ...

    def fetch_tickers(self): ...


TIMEFRAME_MS = {"1h": 3_600_000, "4h": 14_400_000}


def normalize_symbol(symbol: str) -> str:
    base_quote = symbol.upper().split(":", maxsplit=1)[0]
    return "".join(character for character in base_quote if character.isalnum())


def select_universe(exchange: OHLCVExchange, settings, *, now: datetime | None = None) -> list[str]:
    """Select currently active USDT-settled swaps using configured volume and age filters."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    excluded = {"BTCUSDT", "ETHUSDT"}
    excluded.update(normalize_symbol(value) for value in settings.universe.exclude_symbols)
    min_onboard = int((now - timedelta(days=settings.universe.min_list_days)).timestamp() * 1000)
    symbols = []
    for symbol, market in markets.items():
        if not (market.get("active") is True and market.get("swap") is True):
            continue
        if market.get("quote") != "USDT" or market.get("settle") != "USDT":
            continue
        if normalize_symbol(symbol) in excluded:
            continue
        info = market.get("info") or {}
        onboard = info.get("onboardDate")
        if onboard is None or int(onboard) <= 0 or int(onboard) > min_onboard:
            continue
        ticker = tickers.get(symbol) or {}
        quote_volume = ticker.get("quoteVolume")
        if quote_volume is None or float(quote_volume) < settings.universe.min_quote_volume_24h:
            continue
        symbols.append(symbol)
    return sorted(symbols)


def filter_tradable_crypto_universe(markets: dict, tickers: dict, settings, *,
                                    now: datetime | None = None) -> list[tuple[str, float]]:
    """Return every contract that passes basic tradability gates, without a hot-rank cap."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    excluded = {"BTCUSDT", "ETHUSDT"}
    excluded.update(normalize_symbol(value) for value in settings.universe.exclude_symbols)
    min_onboard = int((now - timedelta(days=settings.universe.min_list_days)).timestamp() * 1000)
    eligible: list[tuple[str, float]] = []
    for symbol, market in markets.items():
        info = market.get("info") or {}
        if not (market.get("active") is True and market.get("swap") is True):
            continue
        if market.get("quote") != "USDT" or market.get("settle") != "USDT":
            continue
        if info.get("contractType") != "PERPETUAL" or info.get("underlyingType") != "COIN":
            continue
        if "Crypto" not in (info.get("underlyingSubType") or []):
            continue
        if normalize_symbol(symbol) in excluded:
            continue
        onboard = info.get("onboardDate")
        if onboard is None or int(onboard) <= 0 or int(onboard) > min_onboard:
            continue
        quote_volume = (tickers.get(symbol) or {}).get("quoteVolume")
        try:
            quote_volume = float(quote_volume)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(quote_volume) or quote_volume < settings.universe.min_quote_volume_24h:
            continue
        eligible.append((symbol, quote_volume))
    return sorted(eligible, key=lambda item: (-item[1], item[0]))


def hot_activity_metrics(rows: list[list[float]], *, now_ms: int) -> dict[str, float | None]:
    """Derive closed-bar volume expansion and symmetric 4h/24h moves.

    Volume expansion needs the latest closed 1H bar plus 24 prior closed bars;
    each bar's quote turnover is base volume times its OHLC average price.
    """
    by_open_time = {}
    for row in rows:
        if len(row) >= 6 and int(row[0]) + TIMEFRAME_MS["1h"] <= now_ms:
            by_open_time[int(row[0])] = row
    closed = [by_open_time[timestamp] for timestamp in sorted(by_open_time)]
    result: dict[str, float | None] = {
        "volume_burst_ratio": None, "return_4h_pct": None, "return_24h_pct": None,
    }
    if len(closed) < 25:
        return result
    try:
        closes = [float(row[4]) for row in closed[-25:]]
        quote_turnover = [float(row[5]) * sum(float(row[i]) for i in range(1, 5)) / 4
                          for row in closed[-25:]]
    except (TypeError, ValueError, OverflowError):
        return result
    if any(not math.isfinite(value) or value <= 0 for value in closes + quote_turnover):
        return result
    baseline = statistics.median(quote_turnover[:-1])
    if baseline > 0:
        result["volume_burst_ratio"] = quote_turnover[-1] / baseline
    result["return_4h_pct"] = (closes[-1] / closes[-5] - 1.0) * 100 if closes[-5] else None
    result["return_24h_pct"] = (closes[-1] / closes[0] - 1.0) * 100 if closes[0] else None
    return result


def _percentile_scores(values: dict[str, float | None]) -> dict[str, float]:
    valid = sorted(value for value in values.values() if value is not None and math.isfinite(value))
    if not valid:
        return {symbol: 0.0 for symbol in values}
    scores = {}
    for symbol, value in values.items():
        if value is None or not math.isfinite(value):
            scores[symbol] = 0.0
            continue
        lower = bisect_left(valid, value)
        equal = bisect_right(valid, value) - lower
        scores[symbol] = (lower + equal / 2) / len(valid)
    return scores


def score_hot_candidates(candidates: list[tuple[str, float]], tickers: dict,
                         activity: dict[str, dict[str, float | None]]) -> list[dict]:
    """Score and sort every basically tradable symbol without applying Top-N."""
    symbols = [symbol for symbol, _ in candidates]
    burst = {symbol: activity.get(symbol, {}).get("volume_burst_ratio") for symbol in symbols}
    move4h = {}
    move24h = {}
    for symbol in symbols:
        return_4h = activity.get(symbol, {}).get("return_4h_pct")
        return_24h = activity.get(symbol, {}).get("return_24h_pct")
        move4h[symbol] = abs(return_4h) if return_4h is not None else None
        move24h[symbol] = abs(return_24h) if return_24h is not None else None
    turnover = {symbol: next(volume for candidate, volume in candidates if candidate == symbol) for symbol in symbols}
    trades = {}
    for symbol in symbols:
        ticker = tickers.get(symbol) or {}
        info = ticker.get("info") or {}
        raw_count = info.get("count", ticker.get("count"))
        try:
            count = float(raw_count) if raw_count is not None else None
            trades[symbol] = count if count is not None and math.isfinite(count) and count >= 0 else None
        except (TypeError, ValueError, OverflowError):
            trades[symbol] = None

    burst_rank = _percentile_scores(burst)
    move4h_rank, move24h_rank = _percentile_scores(move4h), _percentile_scores(move24h)
    turnover_rank, trades_rank = _percentile_scores(turnover), _percentile_scores(trades)
    ranked = []
    for symbol, quote_volume in candidates:
        price_score = (move4h_rank[symbol] + move24h_rank[symbol]) / 2
        liquidity_score = (turnover_rank[symbol] + trades_rank[symbol]) / 2
        score = 100 * (0.40 * burst_rank[symbol] + 0.30 * price_score + 0.30 * liquidity_score)
        metrics = activity.get(symbol, {})
        ranked.append({
            "symbol": symbol, "quote_volume_24h": quote_volume,
            "hot_score": round(score, 2), "volume_burst_ratio": metrics.get("volume_burst_ratio"),
            "return_4h_pct": metrics.get("return_4h_pct"),
            "return_24h_pct": metrics.get("return_24h_pct"),
            "trade_count_24h": trades[symbol],
        })
    ranked.sort(key=lambda row: (-row["hot_score"], -row["quote_volume_24h"], row["symbol"]))
    return ranked


def rank_hot_candidates(candidates: list[tuple[str, float]], tickers: dict,
                        activity: dict[str, dict[str, float | None]], *, limit: int) -> list[dict]:
    """Score eligible symbols from cross-sectional percentile ranks.

    Weights are 40% volume burst, 30% absolute price activity, and 30%
    liquidity/trade count. Missing components contribute no percentile score.
    """
    if not 1 <= limit <= 20:
        raise ValueError("hot candidate limit must be between 1 and 20")
    return score_hot_candidates(candidates, tickers, activity)[:limit]


def rank_mean_reversion_candidates(candidates_metrics: list[dict]) -> list[dict]:
    """Rank eligible symbols by mean-reversion volatility, activity, and liquidity.

    Inputs must already pass tradability, account whitelist, and closed-daily-history
    gates. Scores use bounded component values and cross-sectional midranks so no
    single extreme raw metric can dominate the final 30/35/35 weighting.
    """
    if not candidates_metrics:
        return []

    symbols = [str(item["symbol"]) for item in candidates_metrics]
    if len(symbols) != len(set(symbols)):
        raise ValueError("candidate symbols must be unique")

    def finite_metric(item, key, *, positive=False):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or (positive and value <= 0):
            return None
        return value

    volume_health = {}
    elasticity = {}
    quote_volume = {}
    trade_count = {}
    spread = {}
    depth = {}
    efficiency = {}
    for item in candidates_metrics:
        symbol = str(item["symbol"])
        ratio = finite_metric(item, "volume_ratio_24h_7d", positive=True)
        if ratio is None:
            volume_health[symbol] = None
        elif ratio < 1.2:
            volume_health[symbol] = max(0.1, ratio / 1.2)
        elif ratio <= 2.5:
            volume_health[symbol] = 1.0
        elif ratio <= 3.5:
            volume_health[symbol] = 1.0 - (ratio - 2.5) * 0.3
        else:
            volume_health[symbol] = max(0.2, 0.7 - (ratio - 3.5) * 0.5)

        natr = finite_metric(item, "natr_4h", positive=True)
        range_24h = finite_metric(item, "range_24h", positive=True)
        net_move_24h = finite_metric(item, "net_move_24h")
        if natr is None or range_24h is None or net_move_24h is None:
            elasticity[symbol] = None
            efficiency[symbol] = None
        else:
            two_way = range_24h / max(abs(net_move_24h), range_24h * 0.2)
            efficiency[symbol] = min(two_way, 3.0)
            elasticity[symbol] = natr * efficiency[symbol]

        quote_volume[symbol] = finite_metric(item, "quote_volume_24h", positive=True)
        trade_count[symbol] = finite_metric(item, "trade_count_24h", positive=True)
        spread[symbol] = finite_metric(item, "spread_pct", positive=True)
        depth[symbol] = finite_metric(item, "top_book_depth_usdt", positive=True)

    volume_scores = _percentile_scores(volume_health)
    elasticity_scores = _percentile_scores(elasticity)
    turnover_scores = _percentile_scores(quote_volume)
    trade_scores = _percentile_scores(trade_count)
    spread_scores = _percentile_scores({
        symbol: -value if value is not None else None for symbol, value in spread.items()
    })
    depth_scores = _percentile_scores(depth)

    ranked = []
    for item in candidates_metrics:
        symbol = str(item["symbol"])
        liquidity_score = (
            0.40 * turnover_scores[symbol]
            + 0.20 * trade_scores[symbol]
            + 0.20 * spread_scores[symbol]
            + 0.20 * depth_scores[symbol]
        )
        components = {
            "volume_health": volume_scores[symbol],
            "elasticity": elasticity_scores[symbol],
            "liquidity": liquidity_score,
        }
        score = 100 * (
            0.30 * components["volume_health"]
            + 0.35 * components["elasticity"]
            + 0.35 * components["liquidity"]
        )
        ranked.append({
            **item,
            "hot_score": round(score, 2),
            "two_way_efficiency": efficiency[symbol],
            "hot_score_components": components,
        })
    ranked.sort(key=lambda row: (
        -row["hot_score"],
        -float(row.get("quote_volume_24h") or 0),
        str(row["symbol"]),
    ))
    return ranked


def select_tradable_crypto_universe(exchange: OHLCVExchange, settings, *,
                                    now: datetime | None = None) -> list[tuple[str, float]]:
    """Fetch current exchange metadata and return all basically tradable crypto contracts."""
    return filter_tradable_crypto_universe(exchange.load_markets(), exchange.fetch_tickers(), settings,
                                           now=now)


def resolve_symbols(exchange: OHLCVExchange, symbols: str | list[str], settings, *, now: datetime | None = None) -> list[str]:
    if isinstance(symbols, str) and symbols.strip().upper() == "AUTO":
        return select_universe(exchange, settings, now=now)
    markets = exchange.load_markets()
    requested = [item.strip() for item in (symbols.split(",") if isinstance(symbols, str) else symbols) if item.strip()]
    by_normalized = {normalize_symbol(symbol): symbol for symbol, market in markets.items() if market.get("swap") and market.get("quote") == "USDT" and market.get("settle") == "USDT"}
    excluded = {normalize_symbol(value) for value in settings.universe.exclude_symbols}
    resolved = []
    for item in requested:
        normalized = normalize_symbol(item)
        if normalized in excluded:
            raise ValueError(f"configured excluded symbol cannot be downloaded: {item}")
        if normalized not in by_normalized:
            raise ValueError(f"unknown Binance USDT-M swap symbol: {item}")
        resolved.append(by_normalized[normalized])
    if not resolved:
        raise ValueError("at least one symbol is required")
    return sorted(set(resolved))


def fetch_closed_ohlcv(
    exchange: OHLCVExchange,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
) -> list[list[float]]:
    """Fetch bars forward by open time, dropping partial bars and guarding pagination stalls."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if limit <= 0:
        raise ValueError("limit must be positive")
    step = TIMEFRAME_MS[timeframe]
    end = until_ms if until_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = int(since_ms)
    bars: dict[int, list[float]] = {}
    while cursor < end:
        page = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not page:
            break
        timestamps = [int(row[0]) for row in page]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("exchange returned unordered or duplicate OHLCV timestamps")
        advanced = False
        for row in page:
            timestamp = int(row[0])
            if timestamp >= end:
                continue
            if timestamp + step > end:
                continue
            if timestamp >= cursor:
                bars[timestamp] = [timestamp, *map(float, row[1:6])]
                advanced = True
        next_cursor = timestamps[-1] + step
        if next_cursor <= cursor or not advanced:
            break
        cursor = next_cursor
    return [bars[key] for key in sorted(bars)]


def ohlcv_frame(rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    if frame.empty:
        frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return frame
    frame["timestamp"] = pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True)
    return frame.set_index("timestamp").sort_index()
