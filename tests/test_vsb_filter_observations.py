from datetime import timedelta

import pandas as pd
import pytest

from bot.config import load_settings
from bot.live.multi_paper import MultiPaperState
from bot.live.vsb_shadow import VsbShadowLedger
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.signals import Side
from bot.time_filter import TimeFilterFlags
from scripts import report_vsb_filter_observations as report
from scripts import run_mr_candidate_paper as runner
from test_vsb import AT, SYMBOL, frames


class PublicOnly:
    def fetch_ticker(self, symbol):
        pytest.fail("counterfactual observations must not need 5-second ticker reads")

    def fetch_funding_rate(self, symbol):
        pytest.fail("time-filtered VSB observations must not reach order preparation")

    def fetch_order_book(self, symbol, limit):
        pytest.fail("time-filtered VSB observations must not reach order preparation")

    def fetch_ohlcv(self, symbol, timeframe, limit):
        pytest.fail("the signal symbol already has hourly bars in the candidate batch")

    def create_order(self, *args, **kwargs):
        pytest.fail("counterfactual observations never submit orders")


def _denied_vsb_shadow(monkeypatch, tmp_path):
    one, four = frames(Side.SHORT)
    settings = load_settings(env_file=None)
    state = MultiPaperState(execution_mode="vsb_shadow", strategy_variant="vsb_shadow")
    state.mr_candidate_pool[SYMBOL] = {
        "strategy_tag": "VSB", "side": "short", "status": "active",
    }
    monkeypatch.setattr(runner, "_utc_now", lambda: AT + timedelta(seconds=1))
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(
        False, False, False, False, afternoon_dump_today=True, at_lower_extreme=True,
        recent_move_atr=1.61,
    ))
    monkeypatch.setattr(runner, "_cached_daily_direction_gate",
                        lambda *a, **kw: {"ready": True, "direction": "short"})
    shadow = VsbShadowLedger(tmp_path / "vsb-shadow.json", settings)
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", AT, "BTC/USDT:USDT", "4h", {})
    return one, four, state, shadow, risk, settings


def test_time_filter_blocked_vsb_records_counterfactual_without_orders_or_positions(
        monkeypatch, tmp_path):
    one, four, live, shadow, risk, _settings = _denied_vsb_shadow(monkeypatch, tmp_path)
    live.mr_candidate_pool[SYMBOL]["status"] = "active"
    runner._run_vsb_shadow_cycle(PublicOnly(), shadow, live, {SYMBOL: one}, {SYMBOL: four}, {},
                                 shadow.settings, AT, risk, None)

    records = [event for event in shadow.state.events
               if event.get("event") == "vsb_time_filter_observation"]
    assert len(records) == 1
    record = records[0]
    assert record["side"] == "short"
    assert record["filter_reasons"] == ["afternoon_dump_lower10_blocks_short"]
    assert record["status"] == "tracking"
    assert record["entry_reference"] == "signal_close"
    decision = next(event for event in shadow.state.events
                    if event.get("event") == "entry_time_filter_decision")
    assert decision["strategy_tag"] == decision["source"] == "VSB"
    assert not shadow.state.portfolio.positions
    assert not shadow.state.pending_by_symbol
    assert not [event for event in shadow.state.events
                if event.get("event") in {"paper_fill", "shadow_limit_pending"}]


def test_filter_observation_tracks_forward_r_mfe_mae_and_survives_reload(monkeypatch, tmp_path):
    one, four, _live, shadow, _risk, settings = _denied_vsb_shadow(monkeypatch, tmp_path)
    state = shadow.state
    state.mr_candidate_pool[SYMBOL] = {
        "strategy_tag": "VSB", "side": "short", "status": "active",
    }
    candidate = runner._entry_candidate(None, state, True, SYMBOL, {SYMBOL: one}, {SYMBOL: four},
        {}, settings, AT, {})
    assert candidate is not None
    blocked = candidate[13]["vsb_time_filter_observation"]
    assert shadow.observe_time_filter_blocked(candidate, blocked, AT + timedelta(seconds=1))
    assert not shadow.observe_time_filter_blocked(candidate, blocked, AT + timedelta(seconds=2))

    entry, risk = candidate[3], abs(candidate[3] - candidate[4])
    future = []
    moves = [(-.2, .1, -.4), (.3, 1.1, -.1), (.1, .2, -.2)]
    for offset, (close_r, high_r, low_r) in enumerate(moves):
        future.append({"open": entry, "high": entry + high_r * risk,
                       "low": entry + low_r * risk, "close": entry + close_r * risk,
                       "volume": 100.})
    forward = pd.DataFrame(future, index=pd.date_range(start=AT, periods=3, freq="h"))
    frame = pd.concat([one, forward])
    shadow.hourly({SYMBOL: frame}, AT + timedelta(hours=3))
    shadow = VsbShadowLedger(shadow.path, settings)

    record = next(event for event in shadow.state.events
                  if event.get("event") == "vsb_time_filter_observation")
    one_hour = record["horizons"]["1h"]
    three_hours = record["horizons"]["3h"]
    assert one_hour["status"] == "complete"
    assert one_hour["forward_close_r"] == pytest.approx(.2)
    assert one_hour["mfe_r"] == pytest.approx(.4)
    assert one_hour["mae_r"] == pytest.approx(.1)
    assert three_hours["status"] == "complete"
    assert three_hours["initial_stop_touched"] is True
    assert record["initial_stop_touched"] is True
    assert record["status"] == "tracking"
    assert record["observation_gaps"] == 0


def test_filter_observation_marks_missing_hour_as_partial(monkeypatch, tmp_path):
    one, four, _live, shadow, _risk, settings = _denied_vsb_shadow(monkeypatch, tmp_path)
    shadow.state.mr_candidate_pool[SYMBOL] = {
        "strategy_tag": "VSB", "side": "short", "status": "active",
    }
    candidate = runner._entry_candidate(None, shadow.state, True, SYMBOL, {SYMBOL: one},
        {SYMBOL: four}, {}, settings, AT, {})
    assert candidate is not None
    shadow.observe_time_filter_blocked(
        candidate, candidate[13]["vsb_time_filter_observation"], AT + timedelta(seconds=1))
    entry, risk = candidate[3], abs(candidate[3] - candidate[4])
    future = pd.DataFrame([
        {"open": entry, "high": entry + .1 * risk, "low": entry - .2 * risk,
         "close": entry - .1 * risk, "volume": 100.},
        {"open": entry, "high": entry + .1 * risk, "low": entry - .2 * risk,
         "close": entry - .1 * risk, "volume": 100.},
    ], index=[AT, AT + timedelta(hours=2)])
    shadow.hourly({SYMBOL: pd.concat([one, future])}, AT + timedelta(hours=3))
    record = next(event for event in shadow.state.events
                  if event.get("event") == "vsb_time_filter_observation")
    assert record["observation_gaps"] == 1
    assert record["horizons"]["1h"]["status"] == "complete"
    assert record["horizons"]["3h"]["status"] == "partial"


def test_filter_observation_completes_all_horizons_after_24_hours(monkeypatch, tmp_path):
    one, four, _live, shadow, _risk, settings = _denied_vsb_shadow(monkeypatch, tmp_path)
    shadow.state.mr_candidate_pool[SYMBOL] = {
        "strategy_tag": "VSB", "side": "short", "status": "active",
    }
    candidate = runner._entry_candidate(None, shadow.state, True, SYMBOL, {SYMBOL: one},
        {SYMBOL: four}, {}, settings, AT, {})
    assert candidate is not None
    shadow.observe_time_filter_blocked(
        candidate, candidate[13]["vsb_time_filter_observation"], AT + timedelta(seconds=1))
    entry, risk = candidate[3], abs(candidate[3] - candidate[4])
    future = pd.DataFrame([{
        "open": entry, "high": entry + .1 * risk, "low": entry - .1 * risk,
        "close": entry, "volume": 100.,
    } for _ in range(24)], index=pd.date_range(start=AT, periods=24, freq="h"))
    shadow.hourly({SYMBOL: pd.concat([one, future])}, AT + timedelta(hours=24))
    record = next(event for event in shadow.state.events
                  if event.get("event") == "vsb_time_filter_observation")
    assert record["status"] == "complete"
    assert record["observation_gaps"] == 0
    assert all(record["horizons"][f"{hour}h"]["status"] == "complete"
               for hour in (1, 3, 6, 12, 24))


def test_filter_report_uses_only_complete_gap_free_horizons():
    events = [{
        "event": "vsb_time_filter_observation",
        "filter_reasons": ["morning_down_blocks_short"],
        "status": "complete",
        "horizons": {
            "1h": {"status": "complete", "observation_gaps": 0,
                   "forward_close_r": .5, "mfe_r": .8, "mae_r": .2,
                   "initial_stop_touched": False},
            "3h": {"status": "partial", "observation_gaps": 1,
                   "forward_close_r": -1., "mfe_r": 1., "mae_r": 1.2,
                   "initial_stop_touched": True},
        },
    }]
    observations, grouped = report.summarize(events)
    assert len(observations) == 1
    assert len(grouped["morning_down_blocks_short"][1]) == 1
    assert grouped["morning_down_blocks_short"][3] == []
    rendered = report.render_report(events)
    assert "n=1" in rendered and "1h" in rendered
    assert "3h：完整样本 0" in rendered
