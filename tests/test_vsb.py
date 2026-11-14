"""Offline VSB acceptance, including frozen execution and legacy MR boundaries."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bot.config import load_settings
from bot.execution import EntryPreflightRejected
from bot.indicators import atr
from bot.live.multi_paper import MultiPaperState
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.portfolio import Position
from bot.regime import Regime
from bot.risk import OrderIntent
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from bot.vsb import breakout_signal, effective_entry_atr, squeeze_context
from scripts import run_mr_candidate_paper as runner


AT = datetime(2026, 11, 14, 5, tzinfo=timezone.utc)
SYMBOL = "COIN/USDT:USDT"


def frames(side=Side.LONG, *, jump=1.5, early_width=10.):
    values = np.array(list(100 + 3 * np.sin(np.arange(84) / 3)) + [100.] * 35 + [100 + jump])
    width = np.array([early_width] * 60 + [.05] * 60)
    one = pd.DataFrame(dict(open=values, high=values + width, low=values - width,
                            close=values, volume=100.),
                       index=pd.date_range(end=AT - timedelta(hours=1), periods=120, freq="h"))
    one.iloc[-1, one.columns.get_loc("open")] = 100
    one.iloc[-1, one.columns.get_loc("low")] = 99.95
    one.iloc[-1, one.columns.get_loc("volume")] = 200
    if side is Side.SHORT:
        opening, high, low, close = (one[c].copy() for c in ("open", "high", "low", "close"))
        one["open"], one["close"] = 200 - opening, 200 - close
        one["high"], one["low"] = 200 - low, 200 - high
    four = pd.DataFrame(dict(open=100., high=102., low=98., close=100., volume=100.),
                        index=pd.date_range(end=pd.Timestamp(AT).floor("4h") - timedelta(hours=4),
                                            periods=100, freq="4h"))
    return one, four


@pytest.mark.parametrize("side", list(Side))
def test_real_channels_release_once_and_exclude_breakout_from_box(side):
    one, _ = frames(side)
    before = squeeze_context(one.iloc[:-1])
    box = squeeze_context(one)
    assert before["squeeze_on"] and box["squeeze_released"]
    assert box["sqz_bars"] == before["sqz_bars"] == 18
    assert box["box_high"] == pytest.approx(100.05)
    assert box["box_low"] == pytest.approx(99.95)
    assert box["box_mid"] == 100
    assert box["squeeze_volume_mean"] == 100
    raw = float(atr(one, 14).iloc[-1])
    assert breakout_signal(one, side, box, raw).action is Action.OPEN
    next_bar = one.iloc[[-1]].copy()
    next_bar.index += timedelta(hours=1)
    assert not squeeze_context(pd.concat([one, next_bar]))["squeeze_released"]
    assert effective_entry_atr(one, raw) > raw * 5
    with pytest.raises(ValueError):
        effective_entry_atr(one.tail(59), raw)


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("case,reason", [
    ("volume", "vsb_volume_unconfirmed"), ("clv", "vsb_clv_unconfirmed"),
    ("direction", "vsb_ema9_direction_unconfirmed"), ("buffer", "vsb_breakout_buffer_not_met"),
    ("short_run", "vsb_release_unconfirmed"), ("long_run", "vsb_release_unconfirmed"),
    ("unknown", "vsb_release_unconfirmed"),
])
def test_breakout_gates_fail_closed(side, case, reason):
    one, _ = frames(side)
    box = squeeze_context(one)
    if case == "volume":
        one.iloc[-1, one.columns.get_loc("volume")] = 179.999
    elif case == "clv":
        col = "high" if side is Side.LONG else "low"
        one.iloc[-1, one.columns.get_loc(col)] += 2 if side is Side.LONG else -2
    elif case == "direction":
        one.iloc[-1, one.columns.get_loc("open")] = one["close"].iloc[-1]
    elif case == "buffer":
        box["box_high" if side is Side.LONG else "box_low"] = float(one["close"].iloc[-1])
    elif case == "short_run":
        box["sqz_bars"] = 5
    elif case == "long_run":
        box["sqz_bars"] = 37
    else:
        box["squeeze_complete"] = False
    assert breakout_signal(one, side, box, float(atr(one, 14).iloc[-1])).reason == reason


@pytest.mark.parametrize("adverse,sqz,expected", [
    (.03, 5, "MR"), (.03, 6, None), (.016001, 0, "MR"),
    (.015, 6, "VSB"), (-.019199, 36, "VSB"), (-.01921, 12, None),
    (.015, 5, None), (.015, 37, None),
])
def test_exclusive_structure_boundaries(monkeypatch, adverse, sqz, expected):
    one, four = frames()
    close = 100 * (1 - adverse)
    one.iloc[-1, :4] = [close, close + .1, close - .1, close]
    monkeypatch.setattr(runner, "squeeze_context", lambda _: dict(
        squeeze_on=True, squeeze_released=False, squeeze_complete=True, sqz_bars=sqz,
        box_high=101., box_low=99., box_mid=100., squeeze_volume_mean=100.))
    # Fix the threshold to isolate exact band endpoints from floating arithmetic.
    monkeypatch.setattr(runner, "_atr_deviation_threshold", lambda *a: (2., .02, .016))
    row = runner._lead_structure(one, four, {"ready": True, "direction": "long"},
                                 load_settings(env_file=None), AT)
    assert row.get("strategy_tag") == expected
    assert row["status"] == ("active" if expected else "removed")


def candidate(monkeypatch, side=Side.LONG, **frame_args):
    one, four = frames(side, **frame_args)
    settings = load_settings(env_file=None)
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    state.mr_candidate_pool[SYMBOL] = dict(strategy_tag="VSB", side=side.value)
    monkeypatch.setattr(runner, "_utc_now", lambda: AT + timedelta(seconds=1))
    monkeypatch.setattr(runner, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(runner, "_cached_daily_direction_gate", lambda *a, **kw: dict(ready=True, direction=side.value))
    result = runner._entry_candidate(None, state, True, SYMBOL, {SYMBOL: one}, {SYMBOL: four}, {},
                                     settings, AT, {})
    return result, state, settings, one, four


@pytest.mark.parametrize("side", list(Side))
def test_real_candidate_uses_atr_floor_and_shared_execution_plan(monkeypatch, side):
    result, _, settings, one, _ = candidate(monkeypatch, side)
    assert result is not None
    assert result[2] is side
    assert result[5] == effective_entry_atr(one, float(atr(one, 14).iloc[-1]))
    expected_stop, _ = runner.calculate_adaptive_stop(side, result[3], result[5],
                                                     result[6] if side is Side.LONG else result[7], settings)
    assert result[4] == expected_stop
    assert result[13]["strategy_tag"] == "VSB"
    assert result[13]["box_mid"] == 100
    assert result[13]["order_type"] == "market"
    assert result[1]["is_market_order"] == 1
    assert result[11].strategy_tag == "VSB"


@pytest.mark.parametrize("side", list(Side))
def test_vsb_large_bar_uses_original_441_percent_limit(monkeypatch, side):
    result, _, _, one, _ = candidate(monkeypatch, side, jump=3., early_width=2.5)
    assert result is not None
    plan = result[13]
    assert plan["order_type"] == "limit"
    bar = one.iloc[-1]
    expected = float(bar["close"]) + (float(bar["open"]) - float(bar["close"])) * .441
    assert plan["limit_price"] == pytest.approx(expected)
    assert plan["limit_expires_at"] == (AT + timedelta(hours=3)).isoformat()
    assert result[1]["is_market_order"] == 0


def test_four_level_priority_has_no_space_cap():
    def row(name, *values):
        return name, dict(zip(("tier_bucket", "is_market_order", "bar_quality", "space_r"), values))
    rows = [row("extreme", 0, 1, 1, 99), row("limit", 2, 0, 1, 99),
            row("quality", 2, 1, .9, 1), row("space4", 2, 1, .8, 4),
            row("space5", 2, 1, .8, 5), row("tier1", 1, 1, 1, 99)]
    assert [r[0] for r in sorted(rows, key=runner._entry_priority_sort_key)] == [
        "quality", "space5", "space4", "limit", "tier1", "extreme"]


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("check,closes", [(1, True), (4, True), (5, False)])
def test_false_breakout_first_four_checks_survives_restart(monkeypatch, side, check, closes):
    one, _ = frames(side)
    entry = 101.5 if side is Side.LONG else 98.5
    stop = 90 if side is Side.LONG else 110
    state = MultiPaperState()
    state.portfolio.positions[SYMBOL] = Position(SYMBOL, side, 1., entry, stop,
        AT - timedelta(minutes=30), "vsb", Regime.MID.value)
    state.exit_management_by_symbol[SYMBOL] = dict(strategy_tag="VSB", box_mid=100.,
        entry_atr=5., initial_risk=5., structure_low=90., structure_high=110.,
        vsb_hourly_checks=check - 1, last_close=entry, mid_reached=True)
    state = MultiPaperState.from_dict(state.to_dict())
    one.iloc[-1, :4] = [100., 100.2, 99.8, 99.9 if side is Side.LONG else 100.1]
    # The last candle closes after the actual fill.
    one.index += timedelta(hours=1)
    seen = []
    monkeypatch.setattr(runner, "_close_position", lambda *a: seen.append(a[5]))
    runner._evaluate_exits(state, None, True, {SYMBOL: one}, {}, load_settings(env_file=None), 0, 0,
                           AT + timedelta(hours=1))
    assert ("vsb_false_breakout" in seen) is closes
    if not closes:
        assert state.exit_management_by_symbol[SYMBOL]["vsb_hourly_checks"] == check
        runner._evaluate_exits(state, None, True, {SYMBOL: one}, {}, load_settings(env_file=None), 0, 0,
                               AT + timedelta(hours=1))
        assert state.exit_management_by_symbol[SYMBOL]["vsb_hourly_checks"] == check


def test_route_change_resets_pool_identity_and_pending_preserves_box():
    state = MultiPaperState()
    runner._update_lead_pool(state, {SYMBOL: dict(status="active", side="long")}, AT)
    assert state.mr_candidate_pool[SYMBOL]["strategy_tag"] == "MR"
    runner._update_lead_pool(state, {SYMBOL: dict(status="active", side="long", strategy_tag="VSB",
                                               box_mid=100.)}, AT + timedelta(hours=1))
    assert state.mr_candidate_pool[SYMBOL]["first_seen_at"] == (AT + timedelta(hours=1)).isoformat()
    runner._update_lead_pool(state, {SYMBOL: dict(status="pending", reason="missing")}, AT + timedelta(hours=2))
    assert state.mr_candidate_pool[SYMBOL]["strategy_tag"] == "VSB"
    assert state.mr_candidate_pool[SYMBOL]["box_mid"] == 100


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("case", ["valid", "inside_box", "overextended", "stale", "expired"])
def test_vsb_preflight_reuses_quote_and_expiry_guards(monkeypatch, side, case):
    result, state, settings, _, _ = candidate(monkeypatch, side)
    # Exercise the guard with an offline account; the Lead account checks have
    # their existing independent regression suite.
    state.execution_mode = "demo_usdm"
    context = result[11]
    at = AT + timedelta(seconds=61 if case == "expired" else 1)
    monkeypatch.setattr(runner, "_utc_now", lambda: at)
    direction = 1 if side is Side.LONG else -1
    price = 100. if case == "inside_box" else 100. + direction * 1.6
    if case == "overextended":
        price = 100. + direction * 8.
    quoted_at = at - timedelta(seconds=16 if case == "stale" else 0)
    client = SimpleNamespace(fetch_order_book=lambda *a, **kw: dict(
        timestamp=int(quoted_at.timestamp() * 1000), bids=[[price - .01, 1000]], asks=[[price + .01, 1000]]),
        price_to_precision=lambda s, p: str(p))
    audit = {}
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", AT, "BTC", "4h", {})
    guard, deadline = runner._entry_preflight(client, SimpleNamespace(exchange=client), state, settings,
        {"precision": {"amount": .01}, "limits": {}}, context, result[12], 0., risk, audit)
    intent = OrderIntent(SYMBOL, side, Action.OPEN, 1., result[3], "vsb", Regime.MID, result[4], "MEAN_REVERSION")
    assert deadline == AT + timedelta(seconds=60)
    if case == "valid":
        refreshed, stop = guard(intent, result[4])
        assert audit["decision"] == "ALLOW"
        assert refreshed.stop_price == stop
    else:
        with pytest.raises(EntryPreflightRejected):
            guard(intent, result[4])
        assert audit["decision"] == "BLOCK"


@pytest.mark.parametrize("side", list(Side))
def test_uncertain_or_limit_fill_recovers_frozen_strategy_after_restart(monkeypatch, side):
    result, state, _, _, _ = candidate(monkeypatch, side)
    position = Position(SYMBOL, side, 1., result[3], result[4], AT, "vsb", Regime.MID.value)
    state.portfolio.positions[SYMBOL] = position
    state.mr_entry_attempts["order"] = dict(symbol=SYMBOL, side=side.value, status="limit_pending",
        bar=(AT - timedelta(hours=1)).isoformat(), clientOrderId="order", stop_price=result[4],
        meta=dict(entry_atr=result[5], structure_low=result[6], structure_high=result[7],
                  last_close=result[3], **runner.strategy_metadata(result[13])))
    state = MultiPaperState.from_dict(state.to_dict())
    execution = SimpleNamespace(inspect_entry_attempt=lambda attempt: dict(status="protected", result=dict(
        position=dict(quantity=1.), stop=dict(id="original-protection", stop_price=result[4]))))
    assert runner._reconcile_pending_entries(execution, state, lambda: None)
    meta = state.exit_management_by_symbol[SYMBOL]
    assert meta["strategy_tag"] == "VSB" and meta["box_mid"] == 100
    assert meta["entry_atr"] == result[5]
    assert state.mr_entry_attempts["order"]["status"] == "completed"
    assert runner._reconcile_pending_entries(execution, state, lambda: None)
    assert state.exchange_orders_by_symbol[SYMBOL]["stop"]["id"] == "original-protection"


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("tag", ["MR", "VSB"])
def test_manual_reconstruction_classifies_without_changing_stop_service(monkeypatch, side, tag):
    one, four = frames(side)
    entry_price = (95. if side is Side.LONG else 105.) if tag == "MR" else float(one["close"].iloc[-1])
    monkeypatch.setattr(runner, "_utc_now", lambda: AT)
    monkeypatch.setattr(runner, "_adoption_closed_frame", lambda *a: one if a[3] == "1h" else four)
    monkeypatch.setattr(runner, "classify_regimes", lambda *a: pd.DataFrame({"regime": [Regime.MID]}))
    # Keep the legacy MR test's price near its original local swing structure.
    if tag == "MR":
        one = one + 0  # independent data; volume is irrelevant to MR adoption
        one.loc[:, ["open", "high", "low", "close"]] = [entry_price, entry_price + 1, entry_price - 1, entry_price]
    stopped = []
    exchange = SimpleNamespace(markets={SYMBOL: {}}, market=lambda s: {"id": "COINUSDT"},
        fetch_ticker=lambda s: {"last": entry_price, "timestamp": int(AT.timestamp() * 1000)})
    def protect(position, **kwargs):
        stopped.append(kwargs["strategy_stop"])
        return dict(id="verified", stop_price=kwargs["strategy_stop"], amount=1.)
    execution = SimpleNamespace(settings=load_settings(env_file=None), exchange=exchange,
        lead_market_ids={"COINUSDT"}, adopt_manual_position_stop=protect)
    recovered = dict(side=side.value, quantity=1., entry_price=entry_price, opened_at=AT, entry_fee=.1)
    prepared = runner._prepare_lead_manual_position(execution, SYMBOL, {}, AT, recovered=recovered)
    state = MultiPaperState(execution_mode="lead_usdm")
    runner._install_lead_manual_position(execution, state, SYMBOL, AT, prepared)
    meta = state.exit_management_by_symbol[SYMBOL]
    assert meta["strategy_tag"] == tag
    assert meta["entry_atr"] == (effective_entry_atr(one, float(atr(one, 14).iloc[-1])) if tag == "VSB"
                                  else float(atr(one.tail(60), 14).iloc[-1]))
    if tag == "VSB":
        assert meta["box_mid"] == 100.
    assert len(stopped) == 1


def test_real_pool_release_survives_restart_and_uses_single_hourly_read(monkeypatch):
    from test_lead_candidate_pool import PublicMarket
    market = PublicMarket(count=1)
    symbol = market.symbols[0]
    original = market.fetch_ohlcv
    one, _ = frames()
    unfinished = one.iloc[[-1]].copy()
    unfinished.index += timedelta(hours=1)
    full = pd.concat([one, unfinished])
    def read(s, timeframe, *, limit):
        if timeframe != "1h":
            return original(s, timeframe, limit=limit)
        market.calls.append((s, timeframe, limit))
        return [[int(stamp.timestamp() * 1000), *map(float, row)]
                for stamp, row in full.tail(limit).iterrows()]
    market.fetch_ohlcv = read
    monkeypatch.setattr(runner, "_utc_now", lambda: AT)
    state = MultiPaperState(execution_mode="lead_usdm")
    for _ in range(2):
        cache = runner.ClosedMarketCache()
        result = runner._build_lead_universe(market, state, load_settings(env_file=None), AT,
            lead_market_ids={market.markets[symbol]["id"]}, cache=cache)
        assert state.mr_candidate_pool[symbol]["strategy_tag"] == "VSB"
        assert state.mr_candidate_pool[symbol]["squeeze_released"]
        assert len(result[2][symbol]) == 119
        assert len(cache.lead_hourly[symbol]) == 59
        state = MultiPaperState.from_dict(state.to_dict())
    assert len([call for call in market.calls if call[1] == "1h"]) == 2


@pytest.mark.parametrize("exit_kind", ["mid", "fade"])
@pytest.mark.parametrize("tag", ["MR", "VSB"])
def test_only_vsb_skips_mid_rejection_and_regime_fade(monkeypatch, exit_kind, tag):
    from test_mr_candidate_runner import _short_exit_fixture
    settings, state, symbol, one, now = _short_exit_fixture(is_demo=True)
    position = state.portfolio.positions[symbol]
    state.portfolio.positions[symbol] = replace(position, regime=Regime.HIGH.value)
    one.iloc[-1, :4] = [100., 101.1, 99.9, 101.]
    meta = state.exit_management_by_symbol[symbol]
    meta.update(strategy_tag=tag, box_mid=110., mid_reached=exit_kind == "mid")
    regimes = {symbol: pd.DataFrame({"regime": [Regime.SIDEWAYS]}, index=[one.index[-1]])}
    closed = []
    monkeypatch.setattr(runner, "_close_position", lambda *a: closed.append(a[5]))
    runner._evaluate_exits(state, None, True, {symbol: one}, regimes, settings, 0, 0, now)
    assert closed == ([] if tag == "VSB" else [
        "bollinger_mid_rejection" if exit_kind == "mid" else "short_regime_faded"])


@pytest.mark.parametrize("side", list(Side))
def test_vsb_still_obeys_structure_invalidation(monkeypatch, side):
    one, _ = frames(side)
    state = MultiPaperState()
    state.portfolio.positions[SYMBOL] = Position(SYMBOL, side, 1., 100., 90. if side is Side.LONG else 110.,
        AT - timedelta(hours=2), "vsb", Regime.MID.value)
    state.exit_management_by_symbol[SYMBOL] = dict(strategy_tag="VSB", box_mid=100.,
        entry_atr=1., initial_risk=2., structure_low=99., structure_high=101.)
    close = 98. if side is Side.LONG else 102.
    one.iloc[-1, :4] = [close, close + .1, close - .1, close]
    seen = []
    monkeypatch.setattr(runner, "_close_position", lambda *a: seen.append(a[5]))
    runner._evaluate_exits(state, None, True, {SYMBOL: one}, {}, load_settings(env_file=None), 0, 0, AT)
    assert seen == [f"{side.value}_structure_invalid"]


def test_manual_vsb_missing_box_blocks_before_installing_any_stop(monkeypatch):
    one, four = frames()
    one.loc[:, ["open", "high", "low", "close"]] = [100., 100.1, 99.9, 100.]
    monkeypatch.setattr(runner, "_adoption_closed_frame", lambda *a: one if a[3] == "1h" else four)
    execution = SimpleNamespace(settings=load_settings(env_file=None), lead_market_ids={"COINUSDT"},
        exchange=SimpleNamespace(markets={SYMBOL: {}}, market=lambda s: {"id": "COINUSDT"}),
        adopt_manual_position_stop=lambda *a, **kw: pytest.fail("unverified context cannot install a stop"))
    with pytest.raises(RuntimeError, match="VSB squeeze box is unavailable"):
        runner._prepare_lead_manual_position(execution, SYMBOL, {}, AT, recovered=dict(
            side="long", quantity=1., entry_price=100., opened_at=AT, entry_fee=.1))


def test_late_restart_cannot_extend_early_false_breakout_window(monkeypatch):
    one, _ = frames()
    one.index += timedelta(hours=5)
    one.iloc[-1, :4] = [100., 100.1, 99.8, 99.9]
    state = MultiPaperState()
    state.portfolio.positions[SYMBOL] = Position(SYMBOL, Side.LONG, 1., 101.5, 90.,
        AT - timedelta(minutes=30), "vsb", Regime.MID.value)
    state.exit_management_by_symbol[SYMBOL] = dict(strategy_tag="VSB", box_mid=100.,
        entry_atr=5., initial_risk=5., structure_low=90., structure_high=110., last_close=100.)
    seen = []
    monkeypatch.setattr(runner, "_close_position", lambda *a: seen.append(a[5]))
    runner._evaluate_exits(state, None, True, {SYMBOL: one}, {}, load_settings(env_file=None),
                           0, 0, AT + timedelta(hours=5))
    assert not seen
