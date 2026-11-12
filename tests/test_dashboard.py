import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from urllib.request import Request, urlopen
import pytest

from bot.config import load_settings
from bot.live.dashboard import DashboardService, build_multi_snapshot, build_order_snapshot, build_snapshot, serve_dashboard


def test_dashboard_frontend_treats_lead_portfolio_as_an_exchange_account():
    html_path = Path(__file__).resolve().parents[1] / "src" / "bot" / "live" / "dashboard.html"
    html = html_path.read_text(encoding="utf-8")
    assert "function isExchangeMode(mode){return mode==='demo_usdm'||mode==='lead_usdm'}" in html
    assert "function accountModeLabel(mode){return mode==='lead_usdm'?'Binance Lead Portfolio'" in html
    assert "${tp.trailing_exemption_r??2}R" in html
    assert "不限时（2R追踪止损）" in html
    assert "x.trailing_started?'2R追踪已启动'" in html


def sample_state():
    opened = datetime(2026, 9, 29, 1, tzinfo=timezone.utc).isoformat()
    return {
        "version": 1, "symbol": "ADA/USDT:USDT", "initial_equity": 10_000.0, "cash": 10_029.0,
        "position": {"side": "long", "quantity": 1.0, "entry_price": 100.0, "stop_price": 98.0,
                     "reason": "synthetic_entry", "opened_at": opened},
        "day_start_equity": 10_000.0, "daily_halted": False,
        "last_processed_close": opened, "events": [
            {"event": "paper_fill", "action": "CLOSE", "at": opened, "side": "short", "net_pnl": 30.0,
             "reason": "synthetic_exit", "price": 130.0},
        ],
    }


def test_dashboard_snapshot_computes_paper_metrics_from_synthetic_state():
    data = build_snapshot(sample_state(), price=105.0)
    assert data["equity"] == 10_034.0
    assert data["unrealized_pnl"] == 5.0
    assert data["realized_pnl"] == 30.0
    assert data["total_pnl"] == 34.0
    assert data["position"]["stop_distance_pct"] > 0
    assert data["trade_count"] == 1
    json.dumps(data, allow_nan=False)


def test_order_statistics_filter_by_shanghai_dates_and_include_fills_only():
    state = {"events": [
        {"event": "paper_fill", "action": "OPEN", "at": "2026-09-28T16:30:00+00:00",
         "symbol": "AAA/USDT:USDT", "side": "long", "quantity": 2, "price": 10, "fee": 0.1,
         "reason": "low_zone_1h_bearish_close", "regime": "LOW"},
        {"event": "paper_fill", "action": "CLOSE", "at": "2026-09-29T03:00:00+00:00",
         "symbol": "AAA/USDT:USDT", "side": "long", "net_pnl": 3.5, "fees": 0.2,
         "reason": "no_progress",
         "entry_reason": "low_zone_1h_bearish_close", "entry_regime": "LOW", "held_hours": 10.5,
         "mfe_r": 1.2, "mfe_available": True, "mae_r": 0.4, "mae_available": True,
         "slippage_paid": 0.1},
        {"event": "paper_fill", "action": "OPEN", "at": "2026-09-29T10:00:00+00:00",
         "symbol": "BBB/USDT:USDT", "side": "short", "quantity": 1, "price": 20, "fee": 0.1},
        {"event": "order_intent", "action": "OPEN", "at": "2026-09-29T04:00:00+00:00"},
        {"event": "paper_fill", "action": "OPEN", "at": "2026-09-29T16:30:00+00:00",
         "symbol": "CCC/USDT:USDT", "side": "long", "quantity": 1, "price": 5, "fee": 0.1},
        {"event": "paper_fill", "action": "CLOSE", "at": "2026-09-30T16:30:00+00:00",
         "symbol": "CCC/USDT:USDT", "side": "long", "net_pnl": -1.0},
    ]}
    data = build_order_snapshot(state, start_date="2026-09-29", end_date="2026-09-29", current_positions=[{
        "symbol": "BBB/USDT:USDT", "side": "short", "quantity": 1, "entry_price": 20,
        "opened_at": "2026-09-29T18:00:00+08:00", "price": 19.5, "unrealized_pnl": 0.5,
    }])
    assert data["timezone"] == "Asia/Shanghai"
    assert (data["trade_count"], data["open_trade_count"], data["closed_trade_count"]) == (2, 1, 1)
    assert data["realized_pnl"] == 3.5
    assert data["unrealized_pnl"] == 0.5
    assert data["win_rate"] == 1.0
    assert len(data["trades"]) == 2
    assert data["trades"][0]["symbol"] == "BBB/USDT:USDT"
    assert data["trades"][0]["status"] == "open"
    assert data["trades"][1]["symbol"] == "AAA/USDT:USDT"
    assert data["trades"][1]["status"] == "closed"
    assert data["trades"][1]["entry_reason"] == "low_zone_1h_bearish_close"
    assert data["trades"][1]["entry_regime"] == "LOW"
    assert data["trades"][1]["mfe_r"] == 1.2
    assert data["trades"][1]["mae_r"] == 0.4
    assert data["exit_reason_stats"] == [{
        "reason": "no_progress", "trade_count": 1, "wins": 1, "losses": 0,
        "win_rate": 1.0, "net_pnl": 3.5, "avg_net_pnl": 3.5,
        "avg_mfe_r": 1.2, "avg_mae_r": 0.4, "avg_held_hours": 10.5,
    }]
    with pytest.raises(ValueError, match="start date"):
        build_order_snapshot(state, start_date="2026-09-30", end_date="2026-09-29")
    with pytest.raises(ValueError):
        build_order_snapshot(state, start_date="20260929")


def test_demo_external_flat_is_closed_without_fabricating_pnl():
    symbol = "ASTER/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 5000.0, "cash": 4249.0,
        "positions": {}, "exchange_snapshot": {"positions": {}, "equity": 5000.0,
            "at": "2026-10-02T01:06:00+00:00", "host": "https://demo-fapi.binance.com"},
        "events": [
            {"event": "paper_fill", "execution_source": "binance_usdm_demo", "action": "OPEN",
             "at": "2026-10-01T11:00:29+00:00", "symbol": symbol, "side": "short",
             "quantity": 1000.0, "price": 0.7456, "fee": 0.37},
            {"event": "external_position_reconciled_flat", "at": "2026-10-02T01:05:39+00:00",
             "symbol": symbol, "side": "short", "quantity": 1000.0,
             "external_order_ids": ["743942067"], "close_price": None, "net_pnl": None,
             "pnl_recorded": False},
        ], "watchlist": [], "inconsistencies": [], "foreign_order_ids": [],
    }

    orders = build_order_snapshot(state, start_date="2026-09-26", end_date="2026-10-02")
    assert (orders["trade_count"], orders["open_trade_count"], orders["closed_trade_count"]) == (1, 0, 1)
    assert orders["unrecorded_close_count"] == 1
    assert orders["realized_pnl"] == 0
    assert orders["win_rate"] is None
    row = orders["trades"][0]
    assert row["status"] == "closed"
    assert row["closed_at"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["pnl_recorded"] is False
    assert row["external_order_ids"] == ["743942067"]

    snapshot = build_multi_snapshot(
        state, tickers={}, settings=load_settings(env_file=None),
        now=datetime(2026, 10, 2, 1, 6, tzinfo=timezone.utc),
    )
    assert snapshot["trade_count"] == 1
    assert snapshot["unrecorded_close_count"] == 1
    assert snapshot["realized_pnl"] == 0


def test_manual_position_adoption_is_visible_but_excluded_from_strategy_samples():
    symbol = "KOMA/USDT:USDT"
    opened = "2026-11-08T02:00:00+00:00"
    state = {
        "execution_mode": "lead_usdm",
        "events": [
            {"event": "manual_position_adopted", "execution_source": "binance_usdm_lead",
             "at": "2026-11-08T03:00:00+00:00", "opened_at": opened,
             "symbol": symbol, "side": "short", "quantity": 10_000.0,
             "price": 0.02003, "entry_price": 0.02003, "fee": 0.1,
             "reason": "manual_position_adopted", "regime": "MID",
             "strategy_sample_eligible": False,
             "sample_exclusion_reason": "manual_position_adopted"},
            {"event": "paper_fill", "execution_source": "binance_usdm_lead",
             "action": "CLOSE", "at": "2026-11-08T04:00:00+00:00",
             "symbol": symbol, "side": "short", "quantity": 10_000.0,
             "price": 0.019, "net_pnl": 9.8, "fees": 0.3, "reason": "no_progress"},
        ],
    }

    snapshot = build_order_snapshot(state, start_date="2026-11-08", end_date="2026-11-08")
    assert snapshot["trade_count"] == 0
    assert len(snapshot["trades"]) == 1
    trade = snapshot["trades"][0]
    assert trade["entry_price"] == 0.02003
    assert trade["entry_reason"] == "manual_position_adopted"
    assert trade["entry_regime"] == "MID"
    assert trade["strategy_sample_eligible"] is False
    assert trade["sample_exclusion_reason"] == "manual_position_adopted"
    assert snapshot["exit_reason_stats"] == []


def test_demo_paper_fill_with_missing_costs_keeps_fees_unknown():
    symbol = "ADA/USDT:USDT"
    state = {
        "execution_mode": "demo_usdm",
        "events": [
            {"event": "paper_fill", "execution_source": "binance_usdm_demo", "action": "OPEN",
             "at": "2026-10-02T01:00:00+00:00", "symbol": symbol, "side": "long",
             "quantity": 2.0, "price": 100.0},
            {"event": "paper_fill", "execution_source": "binance_usdm_demo", "action": "CLOSE",
             "at": "2026-10-02T01:30:00+00:00", "symbol": symbol, "side": "long",
             "quantity": 2.0, "reason": "exchange_flat_unverified", "price": None,
             "net_pnl": None, "pnl_recorded": False},
        ],
    }

    orders = build_order_snapshot(state, start_date="2026-10-02", end_date="2026-10-02")
    assert orders["unrecorded_close_count"] == 1
    assert orders["realized_pnl"] == 0
    assert orders["trades"][0]["fees"] is None
    assert orders["trades"][0]["realized_pnl"] is None
    assert orders["trades"][0]["exit_reason"] == "外部平仓（成交价／盈亏待核验）"


def test_verified_manual_exchange_close_has_localized_dashboard_reason():
    symbol = "ADA/USDT:USDT"
    state = {
        "execution_mode": "lead_usdm",
        "events": [
            {"event": "paper_fill", "execution_source": "binance_usdm_lead", "action": "OPEN",
             "at": "2026-10-02T01:00:00+00:00", "symbol": symbol, "side": "long",
             "quantity": 2.0, "price": 100.0, "fee": 0.2},
            {"event": "paper_fill", "execution_source": "binance_usdm_lead", "action": "CLOSE",
             "at": "2026-10-02T01:30:00+00:00", "closed_at": "2026-10-02T01:29:59+00:00",
             "symbol": symbol, "side": "long", "quantity": 2.0, "price": 105.0,
             "reason": "external_close_exchange_trades_verified", "fees": 0.4,
             "net_pnl": 9.6, "pnl_recorded": True},
        ],
    }

    orders = build_order_snapshot(state, start_date="2026-10-02", end_date="2026-10-02")

    assert orders["trades"][0]["exit_reason"] == "手动／外部平仓（交易所成交已核验）"
    assert orders["trades"][0]["exit_price"] == 105.0
    assert orders["trades"][0]["realized_pnl"] == 9.6


def test_verified_bot_stop_reconciliation_is_included_in_demo_trade_statistics():
    symbol = "ADA/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 5000.0, "cash": 5000.0,
        "positions": {}, "exchange_snapshot": {"positions": {}, "equity": 5000.0,
            "at": "2026-10-02T02:00:00+00:00", "host": "https://demo-fapi.binance.com"},
        "events": [
            {"event": "paper_fill", "execution_source": "binance_usdm_demo", "action": "OPEN",
             "at": "2026-10-02T01:00:00+00:00", "symbol": symbol, "side": "long",
             "quantity": 2.0, "price": 100.0, "fee": 0.2},
            {"event": "external_position_reconciled_flat", "at": "2026-10-02T01:30:00+00:00",
             "closed_at": "2026-10-02T01:30:00+00:00", "symbol": symbol, "side": "long",
             "quantity": 2.0, "external_order_ids": ["1001"], "close_price": 95.0,
             "fees": 0.4, "net_pnl": -10.4, "pnl_recorded": True,
             "reason": "protective_stop_exchange_algo"},
        ], "watchlist": [], "inconsistencies": [], "foreign_order_ids": [],
    }

    orders = build_order_snapshot(state, start_date="2026-10-02", end_date="2026-10-02")
    assert orders["realized_pnl"] == -10.4
    assert orders["unrecorded_close_count"] == 0
    assert orders["trades"][0]["exit_reason"] == "机器人保护止损"
    assert orders["trades"][0]["exit_price"] == 95.0

    snapshot = build_multi_snapshot(
        state, tickers={}, settings=load_settings(env_file=None),
        now=datetime(2026, 10, 2, 2, 0, tzinfo=timezone.utc),
    )
    assert snapshot["trade_count"] == 1
    assert snapshot["unrecorded_close_count"] == 0
    assert snapshot["realized_pnl"] == -10.4


def test_multi_dashboard_snapshot_shows_ranked_signals_and_shared_equity():
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "initial_equity": 1_000.0, "cash": 999.0,
        "positions": {"AAA/USDT:USDT": {"side": "short", "quantity": 2.0, "entry_price": 10.0,
                                           "stop_price": 11.0, "reason": "held", "opened_at": "2026-09-29T01:00:00+00:00"}},
        "mark_prices": {"AAA/USDT:USDT": 9.0}, "watchlist": [
            {"symbol": "AAA/USDT:USDT", "rank": 1, "quote_volume_24h": 50_000_000,
             "hot_score": 82.5, "volume_burst_ratio": 2.3, "return_4h_pct": -3.0,
             "return_24h_pct": 8.0, "trade_count_24h": 12_000},
            {"symbol": "BBB/USDT:USDT", "rank": 2, "quote_volume_24h": 40_000_000},
        ],
        "exit_management_by_symbol": {"AAA/USDT:USDT": {
            "enabled": True, "protective_stop": 9.5, "best_price": 8.8, "initial_risk": 1.0,
            "policy": "proposal_v1", "tp1_taken": True, "tp2_taken": False,
            "breakeven_moved": True, "no_progress_count": 2,
        }},
        "protective_stop_audit": {"at": "2026-09-29T14:00:00+00:00", "status": "verified",
                                   "position_count": 1, "mismatch_count": 0, "mismatches": []},
        "symbol_status": {"BBB/USDT:USDT": {"regime": "HIGH", "action": "OPEN", "side": "short",
                                               "reason": "high_zone_signal", "at": "2026-09-29T01:00:00+00:00"}},
        "market_risk_status": {"mode": "entry_gate", "state": "RISK_ELEVATED",
                                "reason": "btc_volatility_or_trend_elevated",
                                "symbol": "BTC/USDT:USDT", "checked_at": "2026-09-29T14:00:00+00:00"},
        "events": [],
    }
    tickers = {"AAA/USDT:USDT": {"last": 9.0, "quoteVolume": 51_000_000},
               "BBB/USDT:USDT": {"last": 20.0, "quoteVolume": 41_000_000}}

    data = build_multi_snapshot(
        state, tickers=tickers, settings=load_settings(env_file=None),
        now=datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc),
    )

    assert data["equity"] == 1_001.0
    assert data["positions"][0]["unrealized_pnl"] == 2.0
    assert data["positions"][0]["stop_price"] == 9.5
    assert data["positions"][0]["profit_protection"] is True
    assert data["positions"][0]["time_observation"]["held_hours"] == 13
    assert data["positions"][0]["time_observation"]["current_r"] == 1.0
    assert round(data["positions"][0]["time_observation"]["mfe_r"], 2) == 1.2
    assert data["positions"][0]["time_observation"]["check_12h"]["due"] is True
    assert data["positions"][0]["time_observation"]["check_12h"]["condition_met"] is False
    assert data["positions"][0]["time_observation"]["check_24h"]["due"] is False
    assert data["positions"][0]["exit_management"]["policy"] == "proposal_v1"
    assert data["positions"][0]["exit_management"]["no_progress_count"] == 2
    assert data["positions"][0]["exit_management"]["max_hold_due"] is False
    assert data["protective_stop_audit"]["status"] == "verified"
    assert data["time_exit_policy"]["enabled"] is False
    assert data["watchlist"][1]["action"] == "OPEN"
    assert data["watchlist"][1]["side"] == "short"
    assert data["watchlist"][0]["quote_volume_24h"] == 51_000_000
    assert data["market_risk_status"]["state"] == "RISK_ELEVATED"
    assert data["watchlist"][0]["hot_score"] == 82.5
    assert data["watchlist"][0]["volume_burst_ratio"] == 2.3


def test_multi_dashboard_marks_time_observation_without_enabling_auto_exit():
    now = datetime(2026, 9, 29, 13, 0, tzinfo=timezone.utc)
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {"AAA/USDT:USDT": {"side": "long", "quantity": 1.0, "entry_price": 100.0,
                                           "stop_price": 90.0, "reason": "test", "opened_at": "2026-09-29T00:00:00+00:00"}},
        "entry_costs": {"AAA/USDT:USDT": {"entry_fee": 0.05, "funding_paid": 0.0}},
        "exit_management_by_symbol": {"AAA/USDT:USDT": {
            "enabled": True, "initial_risk": 10.0, "best_price": 101.0, "protective_stop": 90.0,
        }},
        "isolated_margin_by_symbol": {}, "watchlist": [], "events": [],
    }

    data = build_multi_snapshot(
        state, tickers={"AAA/USDT:USDT": {"last": 99.0}},
        settings=load_settings(env_file=None), now=now,
    )
    observation = data["positions"][0]["time_observation"]

    assert observation["held_hours"] == 13
    assert observation["mfe_r"] == 0.1
    assert observation["estimated_net_if_closed"] < 0
    assert observation["check_12h"]["condition_met"] is True
    assert observation["check_24h"]["due"] is False
    assert observation["time_exit_enabled"] is False
    assert data["time_exit_policy"]["enabled"] is False

    data = build_multi_snapshot(
        state, tickers={"AAA/USDT:USDT": {"last": 99.0}},
        settings=load_settings(env_file=None), now=now + timedelta(hours=12),
    )
    assert data["positions"][0]["time_observation"]["check_24h"]["condition_met"] is True


def test_demo_dashboard_positions_and_unrealized_pnl_come_from_exchange_snapshot():
    symbol = "AAA/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 1200.0, "cash": 800.0,
        "positions": {}, "exchange_snapshot": {
            "at": "2026-09-29T12:00:00+00:00", "host": "https://demo-fapi.binance.com",
            "equity": 1015.0, "positions": {symbol: {
                "side": "long", "quantity": 2.5, "entry_price": 10.25,
                "unrealized_pnl": 4.75, "initial_margin": 8.54,
                "margin_mode": "isolated", "leverage": 3,
            }},
        },
        "watchlist": [], "events": [], "inconsistencies": [],
    }
    data = build_multi_snapshot(
        state, tickers={symbol: {"last": 12.0}}, settings=load_settings(env_file=None),
        now=datetime(2026, 9, 29, 12, 5, tzinfo=timezone.utc),
    )

    assert data["equity"] == 1015.0
    assert len(data["positions"]) == 1
    assert data["positions"][0]["quantity"] == 2.5
    assert data["positions"][0]["entry_price"] == 10.25
    assert data["positions"][0]["unrealized_pnl"] == 4.75
    assert data["positions"][0]["price_source"] == "Binance 最新成交价"
    assert data["positions"][0]["pnl_source"] == "Demo 账户快照"
    assert data["positions"][0]["pnl_updated_at"] == "2026-09-29T12:00:00+00:00"
    assert data["positions"][0]["price_source"] == "Binance 最新成交价"
    assert data["positions"][0]["pnl_source"] == "Demo 账户快照"
    assert data["positions"][0]["pnl_updated_at"] == "2026-09-29T12:00:00+00:00"
    assert data["positions"][0]["opened_at"] is None
    assert data["positions"][0]["margin_mode"] == "isolated"


def test_lead_dashboard_uses_live_snapshot_and_disables_paper_manual_close(tmp_path):
    symbol = "AAA/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "lead_usdm",
        "execution_status": "synchronized", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {}, "exchange_snapshot": {
            "at": "2026-09-29T12:00:00+00:00", "host": "https://fapi.binance.com",
            "equity": 995.0, "available_usdt": 800.0, "positions": {symbol: {
                "side": "long", "quantity": 2.5, "entry_price": 10.25,
                "unrealized_pnl": 4.75, "initial_margin": 8.54,
                "margin_mode": "isolated", "leverage": 3,
            }},
        },
        "watchlist": [], "events": [], "inconsistencies": [],
    }
    snapshot = build_multi_snapshot(
        state, tickers={symbol: {"last": 12.0}}, settings=load_settings(env_file=None),
        now=datetime(2026, 9, 29, 12, 5, tzinfo=timezone.utc),
    )
    assert snapshot["equity"] == 995.0
    assert snapshot["execution_host"] == "https://fapi.binance.com"
    assert snapshot["positions"][0]["pnl_source"] == "Lead Portfolio 账户快照"

    state_path = tmp_path / "lead.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class PublicExchange:
        @staticmethod
        def fetch_tickers():
            return {}

    service = DashboardService(state_path, None, PublicExchange(), settings=load_settings(env_file=None))
    assert service.snapshot()["manual_close_enabled"] is False
    with pytest.raises(ValueError, match="unavailable for Lead Portfolio"):
        service.request_manual_close(symbol)


def test_dashboard_service_records_one_equity_point_per_new_account_snapshot(tmp_path):
    state_path = tmp_path / "portfolio.json"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {}, "exchange_snapshot": {
            "at": "2026-09-29T12:00:00+00:00", "host": "https://demo-fapi.binance.com",
            "equity": 1000.0, "positions": {},
        }, "watchlist": [{"symbol": "AAA/USDT:USDT", "rank": 1, "hot_score": 80.0}], "events": [],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class FakeExchange:
        def fetch_tickers(self):
            return {}

    service = DashboardService(state_path, None, FakeExchange())
    first = service.snapshot()
    assert len(first["equity_history"]) == 1
    assert first["equity_history"][0]["equity"] == 1000.0
    assert first["watchlist_updated_at"] is not None

    state["exchange_snapshot"]["at"] = "2026-09-29T12:01:00+00:00"
    state["exchange_snapshot"]["equity"] = 1002.5
    state_path.write_text(json.dumps(state), encoding="utf-8")
    service._cached_at = 0
    second = service.snapshot()
    assert len(second["equity_history"]) == 2
    assert second["equity_history"][-1] == {
        "at": "2026-09-29T12:01:00+00:00", "equity": 1002.5,
    }


def test_multi_dashboard_caches_tickers_longer_than_portfolio_snapshots(tmp_path, monkeypatch):
    state_path = tmp_path / "portfolio.json"
    state_path.write_text(json.dumps({
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 1000.0, "cash": 1000.0,
        "positions": {}, "exchange_snapshot": {
            "at": "2026-09-29T12:00:00+00:00", "host": "https://demo-fapi.binance.com",
            "equity": 1000.0, "positions": {},
        }, "watchlist": [], "events": [],
    }), encoding="utf-8")

    class CountingExchange:
        calls = 0

        def fetch_tickers(self):
            self.calls += 1
            return {}

    clock = [100.0]
    monkeypatch.setattr("bot.live.dashboard.time.monotonic", lambda: clock[0])
    exchange = CountingExchange()
    service = DashboardService(state_path, None, exchange)

    service.snapshot()
    clock[0] += 4
    service._cached_at = 0
    service.snapshot()
    assert exchange.calls == 1

    clock[0] += 27
    service._cached_at = 0
    service.snapshot()
    assert exchange.calls == 2


def test_dashboard_service_records_one_equity_point_per_new_account_snapshot(tmp_path):
    state_path = tmp_path / "portfolio.json"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "execution_mode": "demo_usdm",
        "execution_status": "synchronized", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {}, "exchange_snapshot": {
            "at": "2026-09-29T12:00:00+00:00", "host": "https://demo-fapi.binance.com",
            "equity": 1000.0, "positions": {},
        }, "watchlist": [{"symbol": "AAA/USDT:USDT", "rank": 1, "hot_score": 80.0}], "events": [],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class FakeExchange:
        def fetch_tickers(self):
            return {}

    service = DashboardService(state_path, None, FakeExchange())
    first = service.snapshot()
    assert len(first["equity_history"]) == 1
    assert first["equity_history"][0]["equity"] == 1000.0
    assert first["watchlist_updated_at"] is not None

    state["exchange_snapshot"]["at"] = "2026-09-29T12:01:00+00:00"
    state["exchange_snapshot"]["equity"] = 1002.5
    state_path.write_text(json.dumps(state), encoding="utf-8")
    service._cached_at = 0
    second = service.snapshot()
    assert len(second["equity_history"]) == 2
    assert second["equity_history"][-1] == {
        "at": "2026-09-29T12:01:00+00:00", "equity": 1002.5,
    }


@pytest.mark.parametrize(
    ("side", "entry", "stop", "stored_best", "last"),
    [("long", 100.0, 90.0, 101.0, 103.0), ("short", 100.0, 110.0, 99.0, 97.0)],
)
def test_dashboard_mfe_includes_newer_ticker_price(side, entry, stop, stored_best, last):
    symbol = "AAA/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {symbol: {"side": side, "quantity": 1.0, "entry_price": entry,
                               "stop_price": stop, "reason": "test",
                               "opened_at": "2026-09-29T00:00:00+00:00"}},
        "exit_management_by_symbol": {symbol: {
            "enabled": True, "initial_risk": 10.0, "best_price": stored_best,
            "protective_stop": stop,
        }},
        "isolated_margin_by_symbol": {}, "watchlist": [], "events": [],
    }
    data = build_multi_snapshot(
        state, tickers={symbol: {"last": last}}, settings=load_settings(env_file=None),
        now=datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc),
    )
    observation = data["positions"][0]["time_observation"]

    assert observation["mfe_r"] == observation["current_r"]


def test_dashboard_preserves_mfe_peak_after_ticker_retraces(tmp_path, monkeypatch):
    symbol = "AAA/USDT:USDT"
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {symbol: {"side": "long", "quantity": 1.0, "entry_price": 100.0,
                               "stop_price": 90.0, "reason": "test",
                               "opened_at": "2026-09-29T00:00:00+00:00"}},
        "exit_management_by_symbol": {symbol: {
            "enabled": True, "initial_risk": 10.0, "best_price": 101.0, "protective_stop": 90.0,
        }},
        "isolated_margin_by_symbol": {}, "watchlist": [], "events": [],
    }
    state_path = tmp_path / "portfolio.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    class ChangingExchange:
        price = 103.0

        def fetch_tickers(self):
            return {symbol: {"last": self.price}}

    exchange = ChangingExchange()
    clock = [100.0]
    monkeypatch.setattr("bot.live.dashboard.time.monotonic", lambda: clock[0])
    service = DashboardService(state_path, None, exchange, load_settings(env_file=None))
    first = service.snapshot()["positions"][0]["time_observation"]
    assert first["mfe_r"] == 0.3

    exchange.price = 102.0
    clock[0] += 31
    service._cached_at = 0
    after_retrace = service.snapshot()["positions"][0]["time_observation"]

    assert after_retrace["current_r"] == 0.2
    assert after_retrace["mfe_r"] == 0.3


def test_multi_dashboard_adds_read_only_position_management_cues():
    now = datetime(2026, 9, 30, 7, 0, tzinfo=timezone.utc)
    state = {
        "version": 2, "account_type": "multi_symbol_paper", "initial_equity": 1000.0, "cash": 900.0,
        "positions": {"AAA/USDT:USDT": {"side": "long", "quantity": 1.0, "entry_price": 100.0,
                                           "stop_price": 100.0, "reason": "test",
                                           "opened_at": "2026-09-29T00:00:00+00:00"}},
        "entry_costs": {"AAA/USDT:USDT": {"entry_fee": 0.05, "funding_paid": 0.0}},
        "exit_management_by_symbol": {"AAA/USDT:USDT": {
            "enabled": True, "initial_risk": 10.0, "best_price": 130.0, "protective_stop": 100.0,
        }},
        "isolated_margin_by_symbol": {},
        "watchlist": [{"symbol": "AAA/USDT:USDT", "rank": None, "held_outside_top": True}],
        "symbol_status": {"AAA/USDT:USDT": {"regime": "IMPULSE_UP", "action": "CLOSE",
                                                  "reason": "reverse_signal_close_first"}},
        "events": [],
    }
    data = build_multi_snapshot(
        state, tickers={"AAA/USDT:USDT": {"last": 125.0}},
        settings=load_settings(env_file=None), now=now,
    )
    cues = data["positions"][0]["management_observation"]

    assert cues["read_only"] is True
    assert cues["outside_hotlist"] is True
    assert cues["close_signal"] is True
    assert cues["regime"] == "IMPULSE_UP"
    assert any("建议复核是否收紧" in item for item in cues["advice"])
    assert any("已离开当前热点前 20" in item for item in cues["advice"])
    assert any("reverse_signal_close_first" in item for item in cues["advice"])


def test_dashboard_reads_state_and_serves_local_api(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(sample_state()), encoding="utf-8")

    class FakeExchange:
        def fetch_ticker(self, symbol):
            assert symbol == "ADA/USDT:USDT"
            return {"last": 105.0}

    service = DashboardService(state_path, "ADA/USDT:USDT", FakeExchange())
    assert service.snapshot()["price"] == 105.0
    server = serve_dashboard(state_path, "ADA/USDT:USDT", FakeExchange(), port=0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(base + "/api/state", timeout=3) as response:
            payload = json.loads(response.read())
            assert response.headers["Cache-Control"].startswith("no-store")
        assert payload["equity"] == 10_034.0
        with urlopen(base + "/api/orders?start=2026-09-29&end=2026-09-29", timeout=3) as response:
            order_payload = json.loads(response.read())
        assert order_payload["closed_trade_count"] == 1
        assert order_payload["realized_pnl"] == 30.0
        with urlopen(base + "/", timeout=3) as response:
            assert b"\xe6\x89\xa7\xe8\xa1\x8c\xe7\x9b\x91\xe6\x8e\xa7" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_manual_close_endpoint_queues_once_for_an_open_paper_position(tmp_path):
    from bot.live.multi_paper import MultiPaperState
    from bot.portfolio import Position
    from bot.signals import Side

    state_path = tmp_path / "portfolio.json"
    state = MultiPaperState()
    symbol = "AAA/USDT:USDT"
    state.portfolio.positions[symbol] = Position(
        symbol=symbol, side=Side.LONG, quantity=2.0, entry_price=100.0,
        stop_price=95.0, opened_at=datetime(2026, 9, 29, 1, tzinfo=timezone.utc),
        reason="synthetic_entry", regime="LOW",
    )
    state.isolated_margin_by_symbol[symbol] = {"initial_margin": 50.0}
    state.save(state_path)

    class FakeExchange:
        pass

    server = serve_dashboard(state_path, None, FakeExchange(), port=0)
    assert server.service.snapshot()["manual_close_enabled"] is True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = json.dumps({"symbol": symbol}).encode()
        request = Request(base + "/api/manual-close", data=body, method="POST", headers={
            "Content-Type": "application/json", "Origin": base, "Sec-Fetch-Site": "same-origin",
        })
        with urlopen(request, timeout=3) as response:
            assert response.status == 202
            accepted = json.loads(response.read())
        assert accepted["status"] == "queued"
        with urlopen(base + "/api/manual-close/" + accepted["request_id"], timeout=3) as response:
            assert json.loads(response.read())["status"] == "queued"

        duplicate = Request(base + "/api/manual-close", data=body, method="POST", headers={
            "Content-Type": "application/json", "Origin": base, "Sec-Fetch-Site": "same-origin",
        })
        with pytest.raises(Exception) as error:
            urlopen(duplicate, timeout=3)
        assert getattr(error.value, "code", None) == 409
        assert len(list((tmp_path / "portfolio.manual-close").glob("*.json"))) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
