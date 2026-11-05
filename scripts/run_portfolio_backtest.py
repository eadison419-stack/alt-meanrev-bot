"""Replay the current crypto hot-universe as a shared multi-position paper portfolio."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import ccxt
import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.backtest.long_shadow import analyze_long_shadows
from bot.backtest.metrics import summarize_by_source
from bot.backtest.validation import run_routing_audit
from bot.config import PROJECT_ROOT, load_settings
from bot.data.cache import KlineCache
from bot.data.funding import load_funding_csv
from bot.data.klines import filter_tradable_crypto_universe, normalize_symbol
from bot.exchange.network import configure_system_proxy


def _stable_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mode_comparison(policy_results: dict[str, dict]) -> dict:
    baseline = policy_results["MEAN_REVERSION_ONLY"]
    routed = policy_results["MULTI_STRATEGY_ROUTED"]
    numeric_metrics = ("net_pnl", "total_return_pct", "max_drawdown_pct", "trade_count",
                       "fees", "funding_paid")
    deltas = {}
    for key in numeric_metrics:
        left, right = baseline["overall"][key], routed["overall"][key]
        deltas[key] = float(right - left) if left is not None and right is not None else None
    source_deltas = {}
    for source in ("MEAN_REVERSION", "TREND"):
        baseline_source = baseline["by_source"][source]
        routed_source = routed["by_source"][source]
        source_deltas[source] = {
            "trade_count": int(routed_source["trade_count"] - baseline_source["trade_count"]),
            "net_pnl": float(routed_source["net_pnl"] - baseline_source["net_pnl"]),
            "win_rate_pct": (
                float(routed_source["win_rate_pct"] - baseline_source["win_rate_pct"])
                if routed_source["win_rate_pct"] is not None
                and baseline_source["win_rate_pct"] is not None else None
            ),
        }
    baseline_checks = {
        "trend_routing_disabled": baseline["trend_routing_enabled"] is False,
        "trend_trade_count_zero": baseline["by_source"]["TREND"]["trade_count"] == 0,
        "integrity_passed": bool(baseline["integrity_checks"]["all_passed"]),
        "equity_reconciliation_passed": bool(baseline["reconciliation"]["matches_within_tolerance"]),
        "cost_component_reconciliation_passed": bool(
            baseline["reconciliation"].get("components_match_within_tolerance", True)
        ),
    }
    baseline_checks["passed"] = all(baseline_checks.values())
    routed_checks = {
        "trend_routing_enabled": routed["trend_routing_enabled"] is True,
        "integrity_passed": bool(routed["integrity_checks"]["all_passed"]),
        "btc_4h_market_risk_data_available": routed["market_risk_data_available"] is True,
        "equity_reconciliation_passed": bool(routed["reconciliation"]["matches_within_tolerance"]),
        "cost_component_reconciliation_passed": bool(
            routed["reconciliation"].get("components_match_within_tolerance", True)
        ),
    }
    routed_checks["passed"] = all(routed_checks.values())
    return {
        "modes": policy_results,
        "delta_MULTI_STRATEGY_ROUTED_minus_MEAN_REVERSION_ONLY": deltas,
        "by_source_delta_MULTI_STRATEGY_ROUTED_minus_MEAN_REVERSION_ONLY": source_deltas,
        "regression_checks": {
            "pure_mean_reversion_baseline": baseline_checks,
            "multi_strategy_integrity": routed_checks,
            "passed": baseline_checks["passed"] and routed_checks["passed"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--db", type=Path, default=Path("data/market.sqlite"))
    parser.add_argument("--funding-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/backtests/portfolio"))
    parser.add_argument("--exit-policy", choices=["breakeven_trail", "time_managed", "compare"], default="compare")
    parser.add_argument("--long-confirm-shadow", action="store_true",
                        help="report-only next-closed-1H long confirmation sidecar; never changes A")
    parser.add_argument("--routing-audit", action="store_true",
                        help="write per-hour closed-bar classifications, routes, signals and rejection reasons")
    parser.add_argument("--enable-trend-routing", action="store_true",
                        help="fill TREND routes at the next 1H open; the 45-minute pullback is not simulated")
    parser.add_argument("--compare-modes", action="store_true",
                        help="run pure mean-reversion and full routed modes on the same loaded inputs")
    parser.add_argument("--universe-snapshot", type=Path,
                        help="reuse a saved eligible-symbol/onboard-date JSON snapshot without querying Binance")
    parser.add_argument("--start-utc", help="fixed ISO-8601 replay start, for example 2026-10-08T00:00:00Z")
    parser.add_argument("--end-utc", help="fixed ISO-8601 replay end close time")
    args = parser.parse_args()
    if args.days <= 0 or not 1 <= args.top <= 20 or args.initial_equity <= 0:
        parser.error("--days and --initial-equity must be positive; --top must be between 1 and 20")
    if bool(args.start_utc) != bool(args.end_utc):
        parser.error("--start-utc and --end-utc must be supplied together")
    if args.compare_modes and args.enable_trend_routing:
        parser.error("--compare-modes runs both modes; omit --enable-trend-routing")

    settings = load_settings()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    input_snapshot = None
    if args.universe_snapshot:
        snapshot_path = args.universe_snapshot if args.universe_snapshot.is_absolute() else PROJECT_ROOT / args.universe_snapshot
        if not snapshot_path.exists():
            parser.error(f"universe snapshot does not exist: {snapshot_path}")
        try:
            input_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            onboard_ms = {str(symbol): int(value)
                          for symbol, value in input_snapshot["onboard_ms"].items()}
            eligible = [(str(symbol), 0.0) for symbol in input_snapshot["eligible_symbols"]]
            if not eligible or set(symbol for symbol, _ in eligible) != set(onboard_ms):
                raise ValueError("eligible_symbols and onboard_ms must contain the same non-empty set")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"invalid universe snapshot: {error}")
    else:
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        configure_system_proxy(exchange)
        try:
            markets, tickers = exchange.load_markets(), exchange.fetch_tickers()
            now_ms = exchange.milliseconds()
        except ccxt.BaseError as error:
            parser.error(f"Binance public market metadata unavailable: {error}")
        finally:
            exchange.close()

        eligible = filter_tradable_crypto_universe(
            markets, tickers, settings, now=datetime.fromtimestamp(now_ms / 1000, timezone.utc),
        )
        onboard_ms = {symbol: int((markets[symbol].get("info") or {}).get("onboardDate", 0))
                      for symbol, _ in eligible}
    db_path = args.db if args.db.is_absolute() else PROJECT_ROOT / args.db
    funding_dir = args.funding_dir if args.funding_dir.is_absolute() else PROJECT_ROOT / args.funding_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if not db_path.exists():
        parser.error(f"market cache does not exist: {db_path}")
    cache = KlineCache(db_path)
    bars_1h, bars_4h, funding = {}, {}, {}
    market_bars_4h = cache.load(settings.market_risk.symbol, "4h")
    symbols_missing = []
    for symbol, _ in eligible:
        one, four = cache.load(symbol, "1h"), cache.load(symbol, "4h")
        if len(one) < 25 or len(four) < settings.lookback.min_regime_bars_4h:
            continue
        funding_path = funding_dir / f"funding_{normalize_symbol(symbol)}.csv"
        bars_1h[symbol], bars_4h[symbol] = one, four
        if not funding_path.exists():
            symbols_missing.append({"symbol": symbol, "reason": "funding_csv_missing"})
            funding[symbol] = pd.DataFrame(columns=["funding_rate", "mark_price"],
                                           index=pd.DatetimeIndex([], tz="UTC"))
        else:
            rates = load_funding_csv(funding_path)
            funding[symbol] = rates
            if rates.empty:
                symbols_missing.append({"symbol": symbol, "reason": "funding_history_empty"})
    if not bars_1h:
        parser.error("no eligible symbols have complete local kline and funding history")

    if args.start_utc:
        try:
            start_ts, end_ts = pd.Timestamp(args.start_utc), pd.Timestamp(args.end_utc)
            if start_ts.tzinfo is None or end_ts.tzinfo is None or end_ts <= start_ts:
                raise ValueError("window must be timezone-aware and end after start")
            start_ts, end_ts = start_ts.tz_convert("UTC"), end_ts.tz_convert("UTC")
        except (TypeError, ValueError) as error:
            parser.error(f"invalid replay window: {error}")
    else:
        end_ts = pd.Timestamp(datetime.fromtimestamp(now_ms / 1000, timezone.utc).replace(
            minute=0, second=0, microsecond=0))
        start_ts = end_ts - pd.Timedelta(days=args.days)
    start, end = start_ts.to_pydatetime(), end_ts.to_pydatetime()
    replay_data_end = max((frame.index[-1] + pd.Timedelta(hours=1) for frame in bars_1h.values()),
                          default=end_ts)
    funding_coverage = {}
    missing_symbols = {row["symbol"] for row in symbols_missing}
    for symbol in sorted(bars_1h):
        rates = funding.get(symbol)
        latest_index = (int(rates.index.searchsorted(replay_data_end, side="left") - 1)
                        if rates is not None and not rates.empty else -1)
        latest = rates.index[latest_index] if latest_index >= 0 else None
        age_hours = ((replay_data_end - latest).total_seconds() / 3600) if latest is not None else None
        reason = None
        if latest is None:
            reason = "funding_history_missing"
        elif age_hours > 12:
            reason = "funding_history_stale_at_market_data_end"
        funding_coverage[symbol] = {
            "first_settlement_in_file": rates.index[0].isoformat()
            if rates is not None and not rates.empty else None,
            "last_settlement_before_market_data_end": latest.isoformat() if latest is not None else None,
            "latest_settlement_age_hours_at_market_data_end": age_hours,
            "quality": reason or "settlement_history_reaches_market_data_end",
        }
        if reason and symbol not in missing_symbols:
            symbols_missing.append({"symbol": symbol, "reason": reason})
            missing_symbols.add(symbol)
    policies = ("breakeven_trail", "time_managed") if args.exit_policy == "compare" else (args.exit_policy,)
    run_id = f"{end.strftime('%Y%m%d-%H%M%S')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_output = output_dir / f"universe_{run_id}.json"
    if input_snapshot is None:
        snapshot_payload = {
            "generated_at_utc": datetime.fromtimestamp(now_ms / 1000, timezone.utc).isoformat(),
            "eligible_symbols": [symbol for symbol, _ in eligible],
            "onboard_ms": onboard_ms,
            "source": "Binance public market metadata and tickers",
        }
        snapshot_output.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        snapshot_output = snapshot_path

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    funding_hashes = {}
    for symbol in bars_1h:
        path = funding_dir / f"funding_{normalize_symbol(symbol)}.csv"
        if path.exists():
            funding_hashes[path.name] = sha256_file(path)
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
        "eligible_symbols": [symbol for symbol, _ in eligible],
        "symbols_with_data": sorted(bars_1h), "top_n": args.top,
        "mode": ("MODE_COMPARISON" if args.compare_modes else
                 "MULTI_STRATEGY_ROUTED" if args.enable_trend_routing else "MEAN_REVERSION_ONLY"),
        "trend_routing_enabled": (None if args.compare_modes else args.enable_trend_routing),
        "available_data_utc": {
            "market_4h_first": market_bars_4h.index[0].isoformat() if not market_bars_4h.empty else None,
            "market_4h_last_open": market_bars_4h.index[-1].isoformat() if not market_bars_4h.empty else None,
            "symbol_1h_first_open": min((frame.index[0] for frame in bars_1h.values()), default=None).isoformat()
            if bars_1h else None,
            "symbol_1h_last_open": max((frame.index[-1] for frame in bars_1h.values()), default=None).isoformat()
            if bars_1h else None,
            "replay_market_data_end": replay_data_end.isoformat(),
        },
        "funding_coverage_by_symbol": funding_coverage,
        "inputs": {
            "market_database": str(db_path),
            "market_database_sha256": sha256_file(db_path),
            "config_file": str(config_path),
            "config_sha256": sha256_file(config_path),
            "funding_csv_sha256": funding_hashes,
            "universe_snapshot": str(snapshot_output),
            "universe_snapshot_sha256": sha256_file(snapshot_output),
            "btc_4h_rows": int(len(market_bars_4h)),
        },
        "known_biases": [
            "universe uses contracts active and eligible today; delisted historical contracts are unavailable",
            "historical hot ranking is reconstructed once per 1H close, not the live 5-minute cadence",
            "OHLCV-derived quote turnover approximates traded USDT volume; historical trade-count ranking is unavailable",
            "isolated margin is accounted per position, but historical mark-price liquidation is not modeled",
        ],
    }
    if args.compare_modes:
        modes = [("MEAN_REVERSION_ONLY", False), ("MULTI_STRATEGY_ROUTED", True)]
    elif args.enable_trend_routing:
        modes = [("MULTI_STRATEGY_ROUTED", True)]
    else:
        modes = [("MEAN_REVERSION_ONLY", False)]
    comparison_runs: dict[str, dict[str, dict]] = {policy: {} for policy in policies}
    eligible_symbol_names = {symbol for symbol, _ in eligible}
    reproducibility_inputs = {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "top_n": args.top,
        "initial_equity": float(args.initial_equity),
        "record_long_shadows": bool(args.long_confirm_shadow),
        "market_database_sha256": manifest["inputs"]["market_database_sha256"],
        "config_sha256": manifest["inputs"]["config_sha256"],
        "funding_csv_sha256": funding_hashes,
        "universe": [
            {"symbol": symbol, "onboard_ms": int(onboard_ms[symbol])}
            for symbol in sorted(onboard_ms) if symbol in eligible_symbol_names
        ],
    }
    reproducibility_input_fingerprint = _stable_sha256(reproducibility_inputs)
    baseline_runs = []
    run_options = [(policy, mode_name, trend_enabled)
                   for policy in policies for mode_name, trend_enabled in modes]
    for policy, mode_name, trend_enabled in run_options:
        result = run_portfolio_backtest(
            bars_1h, bars_4h, funding, onboard_ms, settings,
            start=start_ts, end=end_ts, top_n=args.top,
            initial_equity=args.initial_equity, exit_policy=policy,
            record_long_shadows=args.long_confirm_shadow,
            enable_trend_routing=trend_enabled,
            market_bars_4h=market_bars_4h,
        )
        stem = f"hot{args.top}_{args.days}d_{policy}_{run_id}"
        if trend_enabled:
            stem = f"trend_routed_{stem}"
        elif args.compare_modes:
            stem = f"mean_reversion_only_{stem}"
        trades_path = output_dir / f"{stem}_trades.csv"
        equity_path = output_dir / f"{stem}_equity.csv"
        summary_path = output_dir / f"{stem}_summary.json"
        trade_fields = list(asdict(result.trades[0]).keys()) if result.trades else [
            "symbol", "side", "opened_at", "closed_at", "entry_price", "exit_price",
            "quantity", "gross_pnl", "net_pnl", "fees", "slippage_paid", "funding_paid",
            "entry_reason", "exit_reason", "strategy_source",
        ]
        if "source" not in trade_fields:
            source_index = trade_fields.index("strategy_source") + 1 if "strategy_source" in trade_fields else len(trade_fields)
            trade_fields.insert(source_index, "source")
        with trades_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=trade_fields)
            writer.writeheader()
            for trade in result.trades:
                row = asdict(trade)
                row["side"] = trade.side.value
                row["opened_at"], row["closed_at"] = trade.opened_at.isoformat(), trade.closed_at.isoformat()
                row["source"] = trade.source
                writer.writerow(row)
        result.equity_curve.rename_axis("timestamp").to_csv(equity_path, encoding="utf-8-sig", header=True)
        source_summary = summarize_by_source(result.trades)
        source_net_pnl = sum(row["net_pnl"] for row in source_summary.values())
        overall_net_pnl = float(result.metrics["net_pnl"])
        reconciliation_delta = float(source_net_pnl - overall_net_pnl)
        source_reconciliation = {
            source: {
                "net_pnl_from_components": row["gross_pnl"] - row["fees"] - row["funding_paid"],
                "net_pnl_delta": row["gross_pnl"] - row["fees"] - row["funding_paid"] - row["net_pnl"],
                "matches_within_tolerance": math.isclose(
                    row["gross_pnl"] - row["fees"] - row["funding_paid"], row["net_pnl"],
                    rel_tol=1e-9, abs_tol=1e-8,
                ),
                "slippage_is_embedded_in_gross_pnl": True,
            }
            for source, row in source_summary.items()
        }
        component_deltas = {
            "gross_pnl": sum(row["gross_pnl"] for row in source_summary.values())
            - float(result.metrics["gross_pnl"]),
            "fees": sum(row["fees"] for row in source_summary.values()) - float(result.metrics["fees"]),
            "funding_paid": sum(row["funding_paid"] for row in source_summary.values())
            - float(result.metrics["funding_paid"]),
            "slippage_paid": sum(row["slippage_paid"] for row in source_summary.values())
            - float(result.metrics["slippage_paid"]),
        }
        rejection_by_source = result.rejected_signals_by_source or {}
        rejected_total = sum(sum(counts.values()) for counts in rejection_by_source.values())
        rejected_signals = {
            "total": rejected_total,
            "by_reason": result.skipped_entries,
            "by_source": {
                source: {
                    "total": sum(counts.values()),
                    "by_reason": counts,
                }
                for source, counts in sorted(rejection_by_source.items())
            },
        }
        reconciliation = {
            "overall_equity_change": overall_net_pnl,
            "sum_source_net_pnl": float(source_net_pnl),
            "source_vs_equity_delta": reconciliation_delta,
            "matches_within_tolerance": math.isclose(
                float(source_net_pnl), overall_net_pnl, rel_tol=1e-9, abs_tol=1e-8,
            ),
            "cost_and_gross_component_deltas": component_deltas,
            "components_match_within_tolerance": all(
                math.isclose(delta, 0.0, rel_tol=1e-9, abs_tol=1e-8)
                for delta in component_deltas.values()
            ),
            "by_source": source_reconciliation,
            "cost_note": "gross_pnl reflects slipped execution prices; do not subtract slippage_paid again",
        }
        overall = {
            "metrics": result.metrics,
            "rejected_signals": rejected_signals,
            "market_risk_execution": result.market_risk_execution_summary or {},
        }
        run_reproducibility_input = {
            **reproducibility_inputs,
            "exit_policy": policy,
            "trend_routing_enabled": trend_enabled,
        }
        run_integrity = {
            **(result.integrity_checks or {}),
            "btc_4h_market_risk_data_available": not market_bars_4h.empty if trend_enabled else None,
        }
        if trend_enabled:
            run_integrity["all_passed"] = bool(run_integrity.get("all_passed")
                                               and not market_bars_4h.empty)
        mode_result = {
            "mode": mode_name,
            "trend_routing_enabled": trend_enabled,
            "market_risk_data_available": not market_bars_4h.empty,
            "overall": result.metrics,
            "by_source": source_summary,
            "rejected_signals": rejected_signals,
            "market_risk_execution": result.market_risk_execution_summary or {},
            "reconciliation": reconciliation,
            "integrity_checks": run_integrity,
            "input_fingerprint": _stable_sha256(run_reproducibility_input),
            "result_fingerprint": result.result_fingerprint,
        }
        comparison_runs[policy][mode_name] = mode_result
        summary = {"mode": mode_name, "exit_policy": policy, "overall": overall, "by_source": source_summary,
                   "reconciliation": reconciliation, "metrics": result.metrics,
                   "strategy_source_summary": source_summary,
                   "source_summary": source_summary,
                   "trend_routing_enabled": trend_enabled,
                   "market_risk_execution": result.market_risk_execution_summary or {},
                   "integrity_checks": run_integrity,
                   "reproducibility": {
                       "input_fingerprint": _stable_sha256(run_reproducibility_input),
                       "result_fingerprint": result.result_fingerprint,
                   },
                   "skipped_entries": result.skipped_entries, "symbols_seen": result.symbols_seen,
                   "ranking_hours": result.ranking_hours,
                   "manifest": {**manifest, "mode": mode_name,
                                "trend_routing_enabled": trend_enabled},
                   "files": {"trades": str(trades_path), "equity": str(equity_path)}}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        baseline_runs.append({"mode": mode_name, "exit_policy": policy, "metrics": result.metrics,
                              "strategy_source_summary": source_summary,
                              "summary_file": str(summary_path)})
        print(json.dumps({"mode": mode_name, "exit_policy": policy, "overall": overall,
                          "metrics": result.metrics,
                          "by_source": source_summary, "reconciliation": reconciliation,
                          "source_summary": source_summary,
                          "symbols_seen": result.symbols_seen, "summary": str(summary_path)}, ensure_ascii=False))
        if args.long_confirm_shadow:
            from bot.regime import classify_regimes
            shadow_rows = analyze_long_shadows(
                result.shadow_long_signals, bars_1h, bars_4h,
                {symbol: classify_regimes(frame, settings) for symbol, frame in bars_4h.items()},
                funding, settings, initial_equity=args.initial_equity, exit_policy=policy,
            )
            a_rows = list(result.long_trade_excursions)
            a_by_entry = {(row["symbol"], pd.Timestamp(row["opened_at"])): row for row in a_rows}
            for row in shadow_rows:
                row["a_actual_trade"] = a_by_entry.get((row["symbol"], pd.Timestamp(row["signal_time"])))

            def stats(rows):
                values = [row for row in rows if row.get("net_pnl") is not None]
                pnl = [float(row["net_pnl"]) for row in values]
                rs = [float(row["r_multiple"]) for row in values if row.get("r_multiple") is not None]
                mfes = [float(row["mfe_r"]) for row in values if row.get("mfe_r") is not None]
                equity, peak, drawdown = args.initial_equity, args.initial_equity, 0.0
                for row in sorted(values, key=lambda item: pd.Timestamp(item.get("closed_at", item.get("exit_time")))):
                    equity += float(row["net_pnl"])
                    peak = max(peak, equity)
                    drawdown = max(drawdown, (peak - equity) / peak * 100 if peak else 0.0)
                return {
                    "trades": len(values), "win_rate_pct": 100 * sum(x > 0 for x in pnl) / len(pnl) if pnl else None,
                    "mean_R": sum(rs) / len(rs) if rs else None,
                    "median_R": float(pd.Series(rs).median()) if rs else None,
                    "median_MFE_R": float(pd.Series(mfes).median()) if mfes else None,
                    "net_pnl_usdt": sum(pnl), "max_drawdown_realized_pct": drawdown,
                    "mfe_under_0_5R_pct": 100 * sum(x < 0.5 for x in mfes) / len(mfes) if mfes else None,
                    "mfe_observations": len(mfes),
                }

            fewer = [row["a_actual_trade"] for row in shadow_rows
                     if not row.get("confirm_pass") and row.get("a_actual_trade") is not None]
            a_stats = stats(a_rows)
            b_stats = stats([row for row in shadow_rows
                             if row.get("confirm_pass") and row.get("net_pnl") is not None])
            shorts = [trade for trade in result.trades if trade.side.value == "short"]
            b_unopened = {}
            for row in shadow_rows:
                if row.get("confirm_pass") and row.get("net_pnl") is None:
                    reason = row.get("fail_reason") or "shadow_simulation_incomplete"
                    b_unopened[reason] = b_unopened.get(reason, 0) + 1
            report = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
                "window_start_beijing": pd.Timestamp(start).tz_convert(settings.timezone).isoformat(),
                "window_end_beijing": pd.Timestamp(end).tz_convert(settings.timezone).isoformat(),
                "exit_policy": policy, "top_n": args.top,
                "A_long_actual": a_stats,
                "B_long_shadow_passed": b_stats,
                "B_confirm_passed": sum(bool(row.get("confirm_pass")) for row in shadow_rows),
                "B_confirm_failed": sum(not bool(row.get("confirm_pass")) for row in shadow_rows),
                "B_confirm_passed_but_not_simulated": b_unopened,
                "A_actual_performance_on_B_failed_signals": stats(fewer),
                "B_failure_opportunity_cost_rows": [row for row in shadow_rows if not row.get("confirm_pass")],
                "short_pollution_check": {"trade_count": len(shorts),
                                          "net_pnl_usdt": sum(t.net_pnl for t in shorts),
                                          "status": "clean: shadow positions never enter A portfolio"},
                "funding_missing_symbols": symbols_missing,
                "metric_note": "R uses net PnL divided by initial stop-distance risk; drawdown is realized close-to-close, not mark-to-market.",
                "pool_limitations": [
                    "historical hourly Top-N is reconstructed from OHLCV for currently active eligible contracts; delisted symbols are unavailable",
                    "OHLCV quote-turnover is an approximation; historical live ticker ranking cannot be recovered",
                    "funding-missing eligible symbols are retained in ranking but cannot pass existing funding-risk entry guard",
                    "B simulations are independent single-trade shadows sized at the configured per-symbol target; they consume no A capital, slots, or signal queue",
                    "the portfolio replay's current exit model does not include Demo Paper TP1/TP2 partial closes",
                ],
            }
            report_path = output_dir / f"{stem}_long-shadow.json"
            details_path = output_dir / f"{stem}_long-shadow.csv"
            missing_path = output_dir / f"{stem}_funding-missing.csv"
            markdown_path = output_dir / f"{stem}_long-shadow.md"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str,
                                              allow_nan=False), encoding="utf-8")
            with details_path.open("w", newline="", encoding="utf-8-sig") as stream:
                fields = list(shadow_rows[0]) if shadow_rows else ["symbol", "signal_time", "confirm_pass", "fail_reason"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in shadow_rows:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                                     if isinstance(value, (dict, list)) else value for key, value in row.items()})
            with missing_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["symbol", "reason"])
                writer.writeheader()
                writer.writerows(symbols_missing)
            def show(value, suffix=""):
                return "—" if value is None else f"{value:.2f}{suffix}"
            markdown = [
                f"# 多头下一根 1H 确认影子对照（{policy}）", "",
                f"窗口：{pd.Timestamp(start).tz_convert(settings.timezone).isoformat()} 至 {pd.Timestamp(end).tz_convert(settings.timezone).isoformat()}（北京时间；CSV 时间仍为 UTC）  ",
                f"候选池：按项目历史 1H OHLCV 逐小时重建 Top-{args.top}；多空只跑一遍 A 组合。  ",
                "B 为逐信号独立影子推演，不进入 A 的资金、仓位、冷却或 pending 队列。", "",
                "## A 实际与 B 假设统计", "",
                "| 指标 | A 多单实际 | B 确认通过且可模拟 |", "|---|---:|---:|",
                f"| 笔数 | {a_stats['trades']} | {b_stats['trades']}（确认通过 {report['B_confirm_passed']}；不可模拟 {report['B_confirm_passed'] - b_stats['trades']}） |",
                f"| 胜率 | {show(a_stats['win_rate_pct'], '%')} | {show(b_stats['win_rate_pct'], '%')} |",
                f"| 平均 R | {show(a_stats['mean_R'])} | {show(b_stats['mean_R'])} |",
                f"| 中位 R | {show(a_stats['median_R'])} | {show(b_stats['median_R'])} |",
                f"| 中位 MFE | {show(a_stats['median_MFE_R'], 'R')} | {show(b_stats['median_MFE_R'], 'R')} |",
                f"| 净盈亏 | {show(a_stats['net_pnl_usdt'], ' USDT')} | {show(b_stats['net_pnl_usdt'], ' USDT')} |",
                f"| 最大回撤（已实现逐笔） | {show(a_stats['max_drawdown_realized_pct'], '%')} | {show(b_stats['max_drawdown_realized_pct'], '%')} |",
                f"| MFE 未达 0.5R | {show(a_stats['mfe_under_0_5R_pct'], '%')} | {show(b_stats['mfe_under_0_5R_pct'], '%')} |", "",
                "## B 失败信号对应 A 实际成交", "",
                (f"确认失败信号中匹配到 A 实际多单 {len(fewer)} 笔：净盈亏 "
                 f"{show(report['A_actual_performance_on_B_failed_signals']['net_pnl_usdt'], ' USDT')}，"
                 f"胜率 {show(report['A_actual_performance_on_B_failed_signals']['win_rate_pct'], '%')}。完整逐信号／逐笔字段见同名 CSV。"), "",
                "## 空单污染核验", "",
                f"A 唯一运行的空单为 {len(shorts)} 笔，净盈亏 {sum(t.net_pnl for t in shorts):.2f} USDT。B 从未写入 A 状态，因此结构上不会改变空单；纯 A 对照值与此相同。", "",
                f"## 数据质量", "",
                f"资金费缺失标的：{len(symbols_missing)} 个，清单见 `{missing_path.name}`。报告详细数据见 `{details_path.name}` 与 `{report_path.name}`。", "",
                "限制：候选池仅含当前仍活跃合约，存在幸存者偏差；历史成交额用 OHLCV 估算。本组合回放器不模拟 Demo Paper 的 TP1／TP2 部分止盈。B 每笔使用配置的单币目标仓位，独立统计净盈亏，不与 A 的组合共享资金；回撤按已实现平仓序列计算。通过确认但止损／仓位无法建立的项目不计入 B 成交统计。", "",
            ]
            markdown_path.write_text("\n".join(markdown), encoding="utf-8")
            print(json.dumps({"long_shadow_report": str(report_path), "details": str(details_path),
                              "markdown": str(markdown_path), "funding_missing": str(missing_path),
                              "signals": len(shadow_rows)}, ensure_ascii=False))
    if args.compare_modes:
        comparison_by_policy = {
            policy: _mode_comparison(comparison_runs[policy]) for policy in policies
        }
        comparison_report = {
            "schema_version": 1,
            "comparison_type": "MEAN_REVERSION_ONLY_vs_MULTI_STRATEGY_ROUTED",
            "inputs": {
                "fingerprint": reproducibility_input_fingerprint,
                "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(),
                "top_n": args.top,
                "initial_equity": float(args.initial_equity),
                "market_database_sha256": manifest["inputs"]["market_database_sha256"],
                "config_sha256": manifest["inputs"]["config_sha256"],
                "funding_csv_sha256": funding_hashes,
                "same_loaded_data_and_parameters_for_both_modes": True,
            },
            "mode_comparison_by_exit_policy": comparison_by_policy,
            "integrity_checks": {
                "each_exit_policy_passed": {
                    policy: bool(row["regression_checks"]["passed"])
                    for policy, row in comparison_by_policy.items()
                },
                "all_passed": all(row["regression_checks"]["passed"]
                                   for row in comparison_by_policy.values()),
            },
        }
        comparison_stem = (
            f"mode_comparison_top{args.top}_{start.strftime('%Y%m%dT%H%M')}_"
            f"{end.strftime('%Y%m%dT%H%M')}_{run_id}"
        )
        comparison_path = output_dir / f"{comparison_stem}.json"
        comparison_path.write_text(json.dumps(
            comparison_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
        ), encoding="utf-8")
        print(json.dumps({"mode_comparison": str(comparison_path),
                          "input_fingerprint": reproducibility_input_fingerprint,
                          "all_integrity_checks_passed": comparison_report["integrity_checks"]["all_passed"]},
                         ensure_ascii=False))

    if args.routing_audit:
        audit = run_routing_audit(
            bars_1h, bars_4h, market_bars_4h, funding, onboard_ms, settings,
            start=start_ts, end=end_ts, top_n=args.top,
            initial_equity=args.initial_equity,
        )
        audit_stem = f"hot{args.top}_{args.days}d_routing_{run_id}"
        decisions_path = output_dir / f"{audit_stem}_decisions.jsonl"
        with decisions_path.open("w", encoding="utf-8") as stream:
            for row in audit.rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        summary_path = output_dir / f"{audit_stem}_summary.json"
        report = {
            **audit.summary,
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "window_start_beijing": pd.Timestamp(start).tz_convert(settings.timezone).isoformat(),
            "window_end_beijing": pd.Timestamp(end).tz_convert(settings.timezone).isoformat(),
            "top_n": args.top,
            "manifest": manifest,
            "mean_reversion_baseline_runs": [row for row in baseline_runs
                                               if row["mode"] == "MEAN_REVERSION_ONLY"],
            "portfolio_runs": baseline_runs,
            "paper_trend_entry_enabled": False,
            "baseline_pnl_source": "existing MEAN_REVERSION portfolio replay summaries",
            "routed_trade_pnl": "not produced by the decision-only routing audit",
            "known_limits": [
                "the routing log is a decision audit and does not simulate routed positions or executions",
                "TREND pullback plans expire within 45 minutes; the current shared portfolio replay has 1H fills and cannot establish that a pullback filled",
                "position ownership and strategy-switch cooldown are validated by the shared risk/router tests, not by this signal-only audit",
                "baseline portfolio reports contain net PnL, fees, slippage, funding and drawdown; they are not routed-strategy performance",
            ],
        }
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"routing_audit_summary": str(summary_path),
                          "routing_decisions": str(decisions_path),
                          "decision_count": len(audit.rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
