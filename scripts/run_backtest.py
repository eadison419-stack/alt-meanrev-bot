"""Run the Mode A backtest against locally cached public Binance data."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from bot.backtest.engine import BacktestMode, run_backtest
from bot.config import PROJECT_ROOT, load_settings
from bot.data.cache import KlineCache
from bot.data.funding import load_funding_csv
from bot.data.klines import normalize_symbol
from bot.time_filter import precompute_time_filters


TRADE_FIELD_NAMES = (
    "symbol", "side", "opened_at", "closed_at", "entry_price", "exit_price", "quantity",
    "gross_pnl", "net_pnl", "fees", "slippage_paid", "funding_paid", "entry_reason", "exit_reason",
    "strategy_source", "source",
)


def _resolve_symbol(value: str) -> str:
    cleaned = value.strip().upper()
    if "/" in cleaned and ":" in cleaned:
        return cleaned
    compact = normalize_symbol(cleaned)
    if compact.endswith("USDT") and len(compact) > 4:
        return f"{compact[:-4]}/USDT:USDT"
    raise ValueError("symbol must be a USDT-M perpetual, such as ADAUSDT")


def _parse_bound(value: str | None, timezone: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo(timezone))
    return timestamp.tz_convert("UTC")


def _write_result(result, output_dir: Path, run_id: str, assumptions: dict) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{normalize_symbol(result.symbol)}_{result.mode}_{run_id}"
    trades_path = output_dir / f"{stem}_trades.csv"
    equity_path = output_dir / f"{stem}_equity.csv"
    summary_path = output_dir / f"{stem}_summary.json"

    with trades_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=TRADE_FIELD_NAMES)
        writer.writeheader()
        for trade in result.trades:
            row = asdict(trade)
            row["side"] = trade.side.value
            row["opened_at"] = trade.opened_at.isoformat()
            row["closed_at"] = trade.closed_at.isoformat()
            row["source"] = trade.source
            row["strategy_source"] = trade.source
            writer.writerow(row)
    result.equity_curve.rename_axis("timestamp").to_csv(equity_path, encoding="utf-8-sig", header=True)
    source_summary = {
        source: sum(trade.source == source for trade in result.trades)
        for source in sorted({trade.source for trade in result.trades})
    }
    summary = {
        "symbol": result.symbol,
        "mode": result.mode,
        "period_start_utc": result.equity_curve.index[0].isoformat(),
        "period_end_utc": result.equity_curve.index[-1].isoformat(),
        "metrics": result.metrics,
        "source_summary": source_summary,
        "skipped_entries": result.skipped_entries,
        "assumptions": assumptions,
        "files": {"trades": str(trades_path), "equity": str(equity_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return trades_path, equity_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ADAUSDT", help="One manually selected USDT-M perpetual")
    parser.add_argument("--mode", choices=["long", "short", "both", "all"], default="all")
    parser.add_argument("--start", help="Inclusive start; naive values are interpreted in Asia/Shanghai")
    parser.add_argument("--end", help="Exclusive end; naive values are interpreted in Asia/Shanghai")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--db", type=Path, default=Path("data/market.sqlite"))
    parser.add_argument("--funding", type=Path)
    parser.add_argument("--exit-policy", choices=["fixed", "breakeven_trail", "time_managed"], default="fixed",
                        help="time_managed adds 12h/24h time exits on top of breakeven/trailing protection")
    parser.add_argument("--output-dir", type=Path, default=Path("data/backtests"))
    args = parser.parse_args()
    settings = load_settings()
    try:
        symbol = _resolve_symbol(args.symbol)
        start, end = _parse_bound(args.start, settings.timezone), _parse_bound(args.end, settings.timezone)
    except ValueError as error:
        parser.error(str(error))
    db_path = args.db if args.db.is_absolute() else PROJECT_ROOT / args.db
    funding_path = args.funding or Path(f"data/funding_{normalize_symbol(symbol)}.csv")
    funding_path = funding_path if funding_path.is_absolute() else PROJECT_ROOT / funding_path
    if not db_path.exists():
        parser.error(f"market cache does not exist: {db_path}; run scripts/download_klines.py first")
    if not funding_path.exists():
        parser.error(f"funding cache does not exist: {funding_path}; run scripts/download_funding.py first")
    try:
        cache = KlineCache(db_path)
        bars_1h = cache.load(symbol, settings.timeframes.exec)
        bars_4h = cache.load(symbol, settings.timeframes.regime)
        funding = load_funding_csv(funding_path)
    except (OSError, ValueError) as error:
        parser.error(f"could not read market data: {error}")
    modes: tuple[BacktestMode, ...] = ("long", "short", "both") if args.mode == "all" else (args.mode,)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    run_id = datetime.now(ZoneInfo(settings.timezone)).strftime("%Y%m%d-%H%M%S")
    flags_bars = bars_1h if end is None else bars_1h.loc[bars_1h.index + pd.Timedelta(1, unit="h") <= end]
    time_flags = precompute_time_filters(flags_bars, settings)
    assumptions = {
        "signal_timing": "completed 1H close",
        "entry_and_signal_exit_fill": "next 1H bar open",
        "stop_fill": "stop level, or bar open when an existing position gaps through the stop; additional adverse slippage applies",
        "funding_entry_filter": "latest historical settlement strictly before signal close",
        "funding_settlement": "historical Binance rate and markPrice; nearest 1H open is a proxy only when markPrice is absent",
        "quantity_filter": "continuous quantity rounded down to 1e-8; historical Binance lot-size filters are not reconstructed",
        "scope": "one manually selected symbol; current-volume AUTO universe is not used for historical selection",
        "intrabar_limit": "OHLC does not reveal event order inside an hour; protective stop takes priority",
        "daily_drawdown_sampling": "portfolio equity is checked at hourly close; transient intrabar drawdown is not reconstructed",
        "margin_model": "historical maintenance margin and liquidation prices are not reconstructed",
        "exit_policy": args.exit_policy,
    }
    for mode in modes:
        try:
            result = run_backtest(
                symbol, bars_1h, bars_4h, funding, settings,
                mode=mode, initial_equity=args.initial_equity, start=start, end=end,
                time_flags_by_close=time_flags, exit_policy=args.exit_policy,
            )
        except (ValueError, KeyError) as error:
            parser.error(f"backtest failed: {error}")
        paths = _write_result(result, output_dir, run_id, assumptions)
        print(json.dumps({"mode": mode, "metrics": result.metrics,
                          "source_summary": source_summary, "summary": str(paths[2])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
