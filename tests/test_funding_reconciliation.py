from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bot.config import load_settings
from bot.exchange.binance_usdm import FAPI_URLS
from bot.execution import DemoExecutionService
from bot.live.multi_paper import MultiPaperState
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 5, 8, 10, tzinfo=timezone.utc)
SYMBOL = "TEST/USDT:USDT"


def _closed_state():
    start, end = NOW - timedelta(hours=2), NOW - timedelta(minutes=6)
    event = {
        "event": "paper_fill", "action": "CLOSE", "symbol": SYMBOL,
        "execution_source": "binance_usdm_demo", "at": end.isoformat(),
        "opened_at": start.isoformat(), "gross_pnl": 10., "fees": .5,
        "funding_paid": 0., "net_pnl": 9.5, "pnl_recorded": True,
    }
    state = MultiPaperState(events=[event])
    return state, event, start, end


@pytest.mark.parametrize("funding", [1.25, -1.25])
def test_delayed_funding_adjustment_is_audited_idempotent_and_periodic(funding):
    state, event, start, end = _closed_state()
    calls = []

    def history(first, last):
        calls.append((first, last))
        return [{"symbol": "TESTUSDT", "time": int((end-timedelta(seconds=20)).timestamp()*1000),
                 "income": funding}]

    execution = SimpleNamespace(exchange=SimpleNamespace(market=lambda _: {"id": "TESTUSDT"}),
                                funding_history=history)
    assert runner._reconcile_demo_funding(execution, state, NOW)
    assert event["funding_paid"] == funding and event["net_pnl"] == 9.5 + funding
    adjustment = next(e for e in state.events if e.get("event") == "funding_reconciliation_adjusted")
    assert adjustment["previous_net_pnl"] == 9.5 and adjustment["net_pnl"] == 9.5 + funding
    assert not state.portfolio.positions and state.initial_equity == 10000
    assert not runner._reconcile_demo_funding(execution, state, NOW+timedelta(minutes=1))
    assert len(calls) == 1
    runner._reconcile_demo_funding(execution, state, NOW+timedelta(minutes=5))
    assert len(calls) == 2
    assert sum(e.get("event") == "funding_reconciliation_adjusted" for e in state.events) == 1
    assert not runner._reconcile_demo_funding(execution, state, NOW+timedelta(days=2))


def test_unavailable_funding_history_preserves_recorded_pnl_and_retries_later():
    state, event, *_ = _closed_state()
    execution = SimpleNamespace(exchange=SimpleNamespace(market=lambda _: {"id": "TESTUSDT"}),
                                funding_history=lambda *a: None)
    assert runner._reconcile_demo_funding(execution, state, NOW)
    assert event["net_pnl"] == 9.5 and "funding_rechecked_at" not in event
    assert state.events[-1]["status"] == "unavailable"


def test_ambiguous_position_intervals_do_not_double_count_funding():
    state, event, start, end = _closed_state()
    duplicate = dict(event)
    state.events.append(duplicate)
    execution = SimpleNamespace(
        exchange=SimpleNamespace(market=lambda _: {"id": "TESTUSDT"}),
        funding_history=lambda *a: [{"symbol": "TESTUSDT", "time": int(end.timestamp()*1000),
                                    "income": 1.25}],
    )
    runner._reconcile_demo_funding(execution, state, NOW)
    assert event["net_pnl"] == duplicate["net_pnl"] == 9.5
    assert "funding_rechecked_at" not in event and "funding_rechecked_at" not in duplicate
    assert state.events[-1]["status"] == "ambiguous"


def test_funding_window_is_bounded_and_never_queries_newly_closed_positions():
    state, event, start, end = _closed_state()
    event["opened_at"] = (NOW-timedelta(days=12)).isoformat()
    calls = []
    execution = SimpleNamespace(exchange=SimpleNamespace(market=lambda _: {"id": "TESTUSDT"}),
                                funding_history=lambda a, b: calls.append((a, b)) or [])
    runner._reconcile_demo_funding(execution, state, NOW)
    assert len(calls) == 3
    assert all(b-a < timedelta(days=5) for a, b in calls)
    assert all(calls[i][1]+timedelta(milliseconds=1) == calls[i+1][0] for i in range(2))
    event.pop("funding_rechecked_at")
    event["at"] = (NOW-timedelta(seconds=10)).isoformat()
    state.events = [event]
    calls.clear()
    assert not runner._reconcile_demo_funding(execution, state, NOW)
    assert not calls


@pytest.mark.parametrize("bad_row", [
    {"asset": "BNB"}, {"income": "NaN"}, {"incomeType": "COMMISSION"},
    {"time": 0}, {"symbol": ""},
])
def test_funding_history_rejects_invalid_or_out_of_interval_rows(bad_row):
    row = {"asset": "USDT", "incomeType": "FUNDING_FEE", "symbol": "TESTUSDT",
           "time": int(NOW.timestamp()*1000), "income": "1.25", **bad_row}
    client = SimpleNamespace(urls={"api": dict(FAPI_URLS)},
                             fapiPrivateGetIncome=lambda params: [row])
    execution = DemoExecutionService(client, load_settings(env_file=None))
    assert execution.funding_history(NOW-timedelta(seconds=1), NOW+timedelta(seconds=1)) is None


def test_funding_history_fails_closed_when_response_reaches_limit():
    client = SimpleNamespace(urls={"api": dict(FAPI_URLS)},
                             fapiPrivateGetIncome=lambda params: [{}]*1000)
    execution = DemoExecutionService(client, load_settings(env_file=None))
    assert execution.funding_history(NOW-timedelta(hours=1), NOW) is None
