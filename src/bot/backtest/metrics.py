"""Performance metrics for the local historical simulator."""

from collections.abc import Sequence

import pandas as pd

from bot.backtest.engine import Trade


STRATEGY_SOURCES = ("MEAN_REVERSION", "TREND")


def summarize_by_source(trades: Sequence[Trade]) -> dict[str, dict[str, float | int | None]]:
    """Return realized trade attribution by strategy source, including empty sources."""
    groups: dict[str, list[Trade]] = {source: [] for source in STRATEGY_SOURCES}
    for trade in trades:
        source = getattr(trade, "source", None) or getattr(trade, "strategy_source", None)
        source = source or "MEAN_REVERSION"
        groups.setdefault(str(source), []).append(trade)

    summary = {}
    for source, group in groups.items():
        net_pnl = [float(trade.net_pnl) for trade in group]
        summary[source] = {
            "trade_count": len(group),
            "win_rate_pct": (100 * sum(value > 0 for value in net_pnl) / len(net_pnl)
                             if net_pnl else None),
            "gross_pnl": float(sum(trade.gross_pnl for trade in group)),
            "net_pnl": float(sum(net_pnl)),
            "fees": float(sum(trade.fees for trade in group)),
            "funding_paid": float(sum(trade.funding_paid for trade in group)),
            "slippage_paid": float(sum(trade.slippage_paid for trade in group)),
        }
    return summary


def _win_rate(trades: Sequence[Trade]) -> float | None:
    return sum(trade.net_pnl > 0 for trade in trades) / len(trades) if trades else None


def _profit_factor(trades: Sequence[Trade]) -> float | None:
    gains = sum(trade.net_pnl for trade in trades if trade.net_pnl > 0)
    losses = -sum(trade.net_pnl for trade in trades if trade.net_pnl < 0)
    if losses == 0:
        return None
    return gains / losses


def summarize_backtest(trades: Sequence[Trade], equity_curve: pd.Series, initial_equity: float) -> dict[str, float | int | None]:
    if initial_equity <= 0 or equity_curve.empty:
        raise ValueError("a positive initial equity and non-empty equity curve are required")
    running_peak = equity_curve.cummax().clip(lower=initial_equity)
    drawdowns = equity_curve.div(running_peak).sub(1)
    metrics: dict[str, float | int | None] = {
        "initial_equity": float(initial_equity),
        "final_equity": float(equity_curve.iloc[-1]),
        "net_pnl": float(equity_curve.iloc[-1] - initial_equity),
        "total_return_pct": float((equity_curve.iloc[-1] / initial_equity - 1) * 100),
        "max_drawdown_pct": float(abs(drawdowns.min()) * 100),
        "trade_count": len(trades),
        "win_rate_pct": None if _win_rate(trades) is None else float(_win_rate(trades) * 100),
        "profit_factor": _profit_factor(trades),
        "gross_pnl": float(sum(trade.gross_pnl for trade in trades)),
        "fees": float(sum(trade.fees for trade in trades)),
        "slippage_paid": float(sum(trade.slippage_paid for trade in trades)),
        "funding_paid": float(sum(trade.funding_paid for trade in trades)),
    }
    for side in ("long", "short"):
        group = [trade for trade in trades if trade.side.value == side]
        metrics[f"{side}_trade_count"] = len(group)
        metrics[f"{side}_net_pnl"] = float(sum(trade.net_pnl for trade in group))
        metrics[f"{side}_win_rate_pct"] = None if _win_rate(group) is None else float(_win_rate(group) * 100)
    return metrics
