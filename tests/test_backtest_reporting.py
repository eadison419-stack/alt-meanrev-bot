import pandas as pd

from scripts.run_portfolio_backtest import _mode_comparison
from bot.backtest.engine import Trade
from bot.backtest.portfolio import _trade_overlap_counts
from bot.signals import Side


def _mode(*, routed, net_pnl, trades, mean_reversion, trend):
    return {
        "mode": "MULTI_STRATEGY_ROUTED" if routed else "MEAN_REVERSION_ONLY",
        "trend_routing_enabled": routed,
        "market_risk_data_available": True,
        "overall": {
            "net_pnl": net_pnl,
            "total_return_pct": net_pnl / 100,
            "max_drawdown_pct": 2.0,
            "trade_count": trades,
            "fees": 1.5,
            "funding_paid": 0.25,
        },
        "by_source": {
            "MEAN_REVERSION": mean_reversion,
            "TREND": trend,
        },
        "rejected_signals": {"by_reason": {"single_trade_risk_cap": 2}},
        "reconciliation": {"matches_within_tolerance": True},
        "integrity_checks": {"all_passed": True},
        "result_fingerprint": "a" if not routed else "b",
    }


def test_mode_comparison_reports_metrics_deltas_and_integrity():
    baseline = _mode(
        routed=False, net_pnl=12.0, trades=3,
        mean_reversion={"trade_count": 3, "net_pnl": 12.0, "win_rate_pct": 66.7},
        trend={"trade_count": 0, "net_pnl": 0.0, "win_rate_pct": None},
    )
    routed = _mode(
        routed=True, net_pnl=18.0, trades=4,
        mean_reversion={"trade_count": 2, "net_pnl": 8.0, "win_rate_pct": 50.0},
        trend={"trade_count": 2, "net_pnl": 10.0, "win_rate_pct": 50.0},
    )

    report = _mode_comparison({
        "MEAN_REVERSION_ONLY": baseline,
        "MULTI_STRATEGY_ROUTED": routed,
    })

    assert report["delta_MULTI_STRATEGY_ROUTED_minus_MEAN_REVERSION_ONLY"]["net_pnl"] == 6.0
    assert report["delta_MULTI_STRATEGY_ROUTED_minus_MEAN_REVERSION_ONLY"]["trade_count"] == 1.0
    assert report["by_source_delta_MULTI_STRATEGY_ROUTED_minus_MEAN_REVERSION_ONLY"]["TREND"] == {
        "trade_count": 2,
        "net_pnl": 10.0,
        "win_rate_pct": None,
    }
    assert report["regression_checks"]["pure_mean_reversion_baseline"]["passed"] is True
    assert report["regression_checks"]["multi_strategy_integrity"]["passed"] is True
    assert report["regression_checks"]["passed"] is True


def test_trade_overlap_check_detects_cross_strategy_overlap():
    def trade(source, opened_at, closed_at):
        return Trade(
            symbol="AAA/USDT:USDT", side=Side.LONG,
            opened_at=pd.Timestamp(opened_at), closed_at=pd.Timestamp(closed_at),
            entry_price=100.0, exit_price=101.0, quantity=1.0,
            gross_pnl=1.0, net_pnl=1.0, fees=0.0, slippage_paid=0.0,
            funding_paid=0.0, entry_reason="test", exit_reason="test",
            strategy_source=source,
        )

    overlaps = _trade_overlap_counts((
        trade("MEAN_REVERSION", "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"),
        trade("TREND", "2026-01-01T03:00:00Z", "2026-01-01T05:00:00Z"),
        trade("MEAN_REVERSION", "2026-01-01T05:00:00Z", "2026-01-01T06:00:00Z"),
    ))

    assert overlaps == (1, 1, 0)
