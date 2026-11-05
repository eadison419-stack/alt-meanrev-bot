"""MR "new candidate" OOS validation: base vs the in-sample winning exit preset.

The candidate ("mr_candidate=True" in run_portfolio_backtest) is the unified preset
for a standalone mean-reversion strategy: hotness top-N universe (no strategy
routing, no 4H RANGE gate), EMA50-deviation entries (1%), and the EMA20 Bollinger
mid-band protection as the only added exit (no TP1/TP2, 1.0-ATR breakeven).

In-sample (5 x 30-day windows, real funding) this was the single best exit config
(mid-band protection alone: +1758 vs base -86 net over five windows). It is still
regime-dependent (wins in up/choppy, loses in down) and tuned in-sample, so run
larger / out-of-sample windows before any live landing.

Run:  python scripts/diagnose_mr_candidate_oos.py --window 2
      python scripts/diagnose_mr_candidate_oos.py --window all --window-days 60
      python scripts/diagnose_mr_candidate_oos.py --window all --variant candidate
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import pandas as pd

from bot.backtest.portfolio import run_portfolio_backtest
from bot.config import load_settings
from bot.data.cache import KlineCache

DEFAULT_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT",
    "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "APTUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TIAUSDT", "ORDIUSDT", "JUPUSDT",
    "AAVEUSDT", "UNIUSDT", "ETCUSDT", "TRXUSDT", "HBARUSDT", "STXUSDT",
]


def load_funding(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"funding cache {path} not found; generate it on a machine with network "
            f"access (data/ab_funding.pkl) or point --funding-cache at one."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _report(label: str, r, t0: float) -> None:
    m = r.metrics
    mr = [t for t in r.trades if (t.source or "") == "MEAN_REVERSION"]
    net = sum(t.net_pnl for t in mr)
    wins = sum(1 for t in mr if t.net_pnl > 0)
    win_rate = 100.0 * wins / len(mr) if mr else 0.0
    by = Counter(t.exit_reason for t in mr)
    bb = [t for t in mr if t.exit_reason == "bollinger_mid_rejection"]
    print(f"[{label}] ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  MR legs={len(mr)}  win={win_rate:.1f}%  net={net:.2f}  "
          f"return={m.get('total_return_pct')}%  maxDD={m.get('max_drawdown_pct')}%  "
          f"PF={m.get('profit_factor')}", flush=True)
    print(f"  bb_exit={len(bb)}/{sum(t.net_pnl for t in bb):.0f}  "
          f"stop_hit={by.get('stop_hit', 0)}  take_profit_1={by.get('take_profit_1', 0)}  "
          f"take_profit_2={by.get('take_profit_2', 0)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, choices=("1", "2", "3", "4", "5", "all"))
    parser.add_argument("--variant", default="all", help="base, candidate, or all")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--cache", default="data/market.sqlite")
    parser.add_argument("--funding-cache", default="data/ab_funding.pkl")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cache = KlineCache(args.cache)
    funding = load_funding(Path(args.funding_cache))
    bars_1h = {s: cache.load(s, "1h") for s in symbols}
    bars_4h = {s: cache.load(s, "4h") for s in symbols}
    settings = load_settings(env_file=None)

    from loguru import logger
    logger.remove()
    logger.add(lambda _msg: None, level="ERROR")

    end = pd.Timestamp.utcnow().floor("h")
    d = args.window_days
    windows = [int(w) for w in (("1", "2", "3", "4", "5") if args.window == "all" else (args.window,))]
    variants = ["base", "candidate"] if args.variant == "all" else [args.variant]

    for w in windows:
        if w == 1:
            start, stop = end - pd.Timedelta(days=2 * d), end - pd.Timedelta(days=d)
        elif w == 2:
            start, stop = end - pd.Timedelta(days=d), end
        elif w == 3:
            start, stop = end - pd.Timedelta(days=3 * d), end - pd.Timedelta(days=2 * d)
        elif w == 4:
            start, stop = end - pd.Timedelta(days=4 * d), end - pd.Timedelta(days=3 * d)
        else:
            start, stop = end - pd.Timedelta(days=5 * d), end - pd.Timedelta(days=4 * d)
        print(f"=== W{w} {start.date()}..{stop.date()} ===", flush=True)
        for variant in variants:
            t0 = time.perf_counter()
            result = run_portfolio_backtest(
                bars_1h, bars_4h, funding, {s: 0 for s in symbols}, settings,
                start=start, end=stop, top_n=args.top_n, initial_equity=10_000.0,
                enable_trend_routing=False, market_bars_4h=None,
                mr_candidate=(variant == "candidate"),
                mr_entry_mode="ema_deviation", mr_deviation_pct=0.01,
            )
            _report(variant, result, t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
