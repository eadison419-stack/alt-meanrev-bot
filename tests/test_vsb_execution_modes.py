"""Execution isolation and a frozen MR oracle captured from commit 91f8ca6."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from bot.config import load_settings
from bot.live.multi_paper import MultiPaperState
from bot.live.vsb_shadow import VsbShadowLedger
from bot.market_risk import MarketRiskAssessment, MarketRiskState
from bot.portfolio import PortfolioState, Position
from bot.regime import Regime
from bot.risk import EntryExposure, OrderIntent
from bot.signals import Action, Side
from bot.time_filter import TimeFilterFlags
from scripts import run_mr_candidate_paper as runner
from test_vsb import AT, SYMBOL, candidate, frames


def mr_snapshot(module, monkeypatch, mode):
    """Same input cases run against the frozen runner and the current runner."""
    cfg = load_settings(env_file=None).model_copy(update={"vsb_execution_mode": mode})
    monkeypatch.setattr(module, "_utc_now", lambda: AT + timedelta(seconds=1))
    monkeypatch.setattr(module, "evaluate_time_filter", lambda *a, **kw: TimeFilterFlags(True, True, False, False))
    monkeypatch.setattr(module, "squeeze_context", lambda f: dict(squeeze_on=False, squeeze_released=False,
                                                               squeeze_complete=True, sqz_bars=0))
    entries, candidates = [], []
    for side in Side:
        monkeypatch.setattr(module, "_cached_daily_direction_gate", lambda *a, **kw: dict(ready=True, direction=side.value))
        for deviation in (.02, .04, .06):
            for width in (.4, 1.):
                one, four = frames(side)
                one = one.tail(59).copy()
                close = 100. - 100 * deviation if side is Side.LONG else 100. + 100 * deviation
                opening = close - .2 if side is Side.LONG else close + .2
                one.loc[:, ["open", "high", "low", "close"]] = [opening, max(opening, close) + width,
                                                                 min(opening, close) - width, close]
                one.iloc[-2, one.columns.get_loc("close")] = opening
                state = MultiPaperState(execution_mode="lead_usdm")
                pool = module._lead_structure(one, four, {"ready": True, "direction": side.value}, cfg, AT)
                result = module._entry_candidate(None, state, True, SYMBOL, {SYMBOL: one}, {SYMBOL: four}, {}, cfg, AT, {})
                record = dict(side=side.value, deviation=deviation, width=width,
                              pool_status=pool["status"], pool_tag=pool.get("strategy_tag"))
                if result:
                    record.update(price=result[3], stop=result[4], atr=result[5], priority=result[1], plan=result[13])
                    candidates.append((str(len(entries)), result[1]))
                    risks = []
                    for used in (0., 20., 39.9, 40.):
                        exposure = EntryExposure(0, 0, used)
                        audit = {}
                        qty = module.calculate_order_quantity(1000., result[3], PortfolioState(), cfg,
                            step_size=.01, stop_price=result[4], side=side, exposure=exposure,
                            portfolio_risk_cap_pct=.04, target_notional_usdt=600, audit=audit)
                        if qty <= 0:
                            risks.append(dict(used=used, quantity=qty, allowed=False, reason="quantity_rejected", audit=audit))
                            continue
                        decision = module.assess_entry(OrderIntent(SYMBOL, side, Action.OPEN, qty, result[3],
                            "MR", Regime.MID, result[4], "MEAN_REVERSION"), PortfolioState(), 1000., 0., AT, cfg,
                            exposure=exposure, portfolio_risk_cap_pct=.04, per_symbol_notional_usdt=600)
                        risks.append(dict(used=used, quantity=qty, allowed=decision.allowed,
                                          reason=decision.reason, audit=audit, approval=decision.details))
                    record["risks"] = risks
                entries.append(record)
    exits = []
    from test_mr_candidate_runner import _short_exit_fixture
    for kind in ("mid", "fade", "structure", "no_progress", "hold"):
        cfg0, state, symbol, one, now = _short_exit_fixture(is_demo=True)
        cfg0 = cfg0.model_copy(update={"vsb_execution_mode": mode})
        meta = state.exit_management_by_symbol[symbol]
        position = state.portfolio.positions[symbol]
        state.portfolio.positions[symbol] = replace(position, regime=Regime.HIGH.value)
        one.iloc[-1, :4] = [100., 101.1, 99.9, 101.]
        regimes = {}
        if kind == "mid":
            meta["mid_reached"] = True
        elif kind == "fade":
            regimes[symbol] = pd.DataFrame({"regime": [Regime.SIDEWAYS]}, index=[one.index[-1]])
        elif kind == "structure":
            meta["structure_high"] = 100.
        elif kind == "no_progress":
            meta["no_progress_count"] = cfg0.exits.no_progress_bars - 1
        closed = []
        monkeypatch.setattr(module, "_close_position", lambda *a: closed.append(a[5]))
        if kind == "hold":
            module._apply_max_hold_exit(True, None, state, symbol, replace(position,
                opened_at=now - timedelta(hours=25)), 101., now, cfg0, 0, 0)
        else:
            module._evaluate_exits(state, None, True, {symbol: one}, regimes, cfg0, 0, 0, now)
        exits.append(dict(kind=kind, closed=closed, metadata=state.exit_management_by_symbol[symbol]))
    return dict(entries=entries, ordering=[c[0] for c in sorted(candidates, key=module._entry_priority_sort_key)], exits=exits)


@pytest.mark.parametrize("mode", ["DISABLED", "SHADOW"])
def test_mr_exactly_matches_pre_switch_oracle(monkeypatch, mode):
    oracle = json.loads(Path(__file__).with_name("mr_vsb_mode_baseline.json").read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(mr_snapshot(runner, monkeypatch, mode)))
    assert actual == oracle["snapshot"]
    assert sum("risks" in r for r in actual["entries"]) >= 4


def test_config_defaults_validation_and_environment_priority(monkeypatch):
    monkeypatch.delenv("VSB_EXECUTION_MODE", raising=False)
    cfg = load_settings(env_file=None)
    assert cfg.vsb_execution_mode == "SHADOW"
    for mode in ("DISABLED", "SHADOW", "LIVE"):
        assert type(cfg).model_validate({**cfg.model_dump(), "runtime": cfg.runtime, "vsb_execution_mode": mode}).vsb_execution_mode == mode
    for mode in ("live", "", "UNKNOWN", None):
        with pytest.raises(ValidationError):
            type(cfg).model_validate({**cfg.model_dump(), "runtime": cfg.runtime, "vsb_execution_mode": mode})
    monkeypatch.setenv("VSB_EXECUTION_MODE", "DISABLED")
    assert load_settings(env_file=None).vsb_execution_mode == "DISABLED"


@pytest.mark.parametrize("mode", ["DISABLED", "SHADOW"])
def test_non_live_vsb_cannot_reach_any_risk_or_execution_api(monkeypatch, mode):
    result, state, cfg, _, _ = candidate(monkeypatch)
    cfg = cfg.model_copy(update={"vsb_execution_mode": mode})
    def forbidden(*a, **kw):
        pytest.fail("VSB must not touch live sizing, approval or executor")
    monkeypatch.setattr(runner, "calculate_order_quantity", forbidden)
    monkeypatch.setattr(runner, "assess_entry", forbidden)
    monkeypatch.setattr(runner, "_open_position", forbidden)
    before = state.to_dict()
    assert runner._submit_entry_candidate(None, state, SimpleNamespace(create_order=forbidden), True, {},
        (*result, None), cfg, 5, 0, 0, AT, SimpleNamespace()) is None
    assert state.to_dict() == before
    assert runner._live_entry_symbols(state, [SYMBOL, "MR"], cfg) == ["MR"]


class PublicOnly:
    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.}

    def fetch_order_book(self, symbol, limit):
        return {"timestamp": int((AT + timedelta(seconds=1)).timestamp() * 1000),
                "bids": [[101.5, 1000]], "asks": [[101.51, 1000]]}

    def fetch_ticker(self, symbol):
        return {"timestamp": int((AT + timedelta(seconds=1)).timestamp() * 1000), "last": 101.5}

    def create_order(self, *a, **kw):
        pytest.fail("shadow must never submit Binance orders")


def test_shadow_still_enters_when_live_account_is_full_or_blocked(monkeypatch, tmp_path):
    result, live, cfg, one, four = candidate(monkeypatch)
    live.mr_candidate_pool[SYMBOL].update(status="active")
    live.execution_status = "inconsistent"
    for i in range(5):
        live.portfolio.positions[str(i)] = Position(str(i), Side.LONG, 1, 100, 90, AT, "MR", "MID")
    live.portfolio.daily_halted = True
    before = json.dumps(live.to_dict(), sort_keys=True)
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", AT, "BTC", "4h", {})
    runner._run_vsb_shadow_cycle(PublicOnly(), shadow, live, {SYMBOL: one}, {SYMBOL: four}, {}, cfg, AT, risk, None)
    assert SYMBOL in shadow.state.portfolio.positions
    assert json.dumps(live.to_dict(), sort_keys=True) == before
    assert all(e["execution_mode"] == "vsb_shadow" for e in shadow.state.events if e.get("virtual"))


@pytest.mark.parametrize("side", list(Side))
def test_virtual_market_trade_records_r_mfe_mae_and_restart(monkeypatch, tmp_path, side):
    result, _, cfg, _, _ = candidate(monkeypatch, side)
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    assert shadow.offer(result, result[3], AT + timedelta(seconds=1), 0., "NORMAL")
    position = shadow.state.portfolio.positions[SYMBOL]
    initial_cash = shadow.state.initial_equity
    meta = shadow.state.exit_management_by_symbol[SYMBOL]
    direction = 1 if side is Side.LONG else -1
    quote = position.entry_price + direction * meta["entry_atr"] * 1.01
    at = AT + timedelta(seconds=6)
    shadow.tick({SYMBOL: (quote, at)}, at)
    assert direction * (shadow.state.exit_management_by_symbol[SYMBOL]["protective_stop"] - position.entry_price) > 0
    shadow = VsbShadowLedger(shadow.path, cfg)
    shadow.close(SYMBOL, quote, "test_exit", at + timedelta(seconds=5))
    shadow.save()
    event = shadow.state.events[-1]
    assert event["action"] == "CLOSE" and event["realized_r"] > 0
    assert event["mfe_r"] > 0 and event["mae_r"] == 0
    assert event["net_pnl"] == pytest.approx(shadow.state.cash - initial_cash)
    assert event["realized_r"] == pytest.approx(event["net_pnl"] / (event["initial_risk"] * event["quantity"]))


@pytest.mark.parametrize("outcome", ["fill", "expire", "stale", "disabled"])
def test_virtual_441_limit_waits_without_orders(monkeypatch, tmp_path, outcome):
    result, _, cfg, _, _ = candidate(monkeypatch, jump=3., early_width=2.5)
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    assert shadow.offer(result, result[3], AT + timedelta(seconds=1), 0., "NORMAL")
    assert not shadow.state.portfolio.positions
    assert SYMBOL in shadow.state.pending_by_symbol
    shadow = VsbShadowLedger(shadow.path, cfg)
    at = AT + (timedelta(hours=3) if outcome == "expire" else timedelta(seconds=6))
    quote_at = at - timedelta(seconds=16) if outcome == "stale" else at
    price = result[13]["limit_price"]
    shadow.tick({SYMBOL: (price, quote_at)}, at, allow_pending=outcome != "disabled")
    assert (SYMBOL in shadow.state.portfolio.positions) is (outcome == "fill")
    assert (SYMBOL in shadow.state.pending_by_symbol) is (outcome == "stale")


@pytest.mark.parametrize("side", list(Side))
def test_shadow_2r_trailing_waives_24h_and_stale_quotes_do_nothing(monkeypatch, tmp_path, side):
    result, _, cfg, _, _ = candidate(monkeypatch, side)
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    shadow.offer(result, result[3], AT + timedelta(seconds=1), 0., "NORMAL")
    p = shadow.state.portfolio.positions[SYMBOL]
    direction = 1 if side is Side.LONG else -1
    risk = shadow.state.exit_management_by_symbol[SYMBOL]["initial_risk"]
    at = AT + timedelta(seconds=6)
    shadow.tick({SYMBOL: (p.entry_price + direction * 2.1 * risk, at)}, at)
    assert shadow.state.exit_management_by_symbol[SYMBOL]["trailing_started"]
    before = json.dumps(shadow.state.to_dict(), sort_keys=True)
    shadow.tick({SYMBOL: (1., at - timedelta(seconds=16))}, at)
    assert json.dumps(shadow.state.to_dict(), sort_keys=True) == before
    late = AT + timedelta(hours=25)
    shadow.tick({SYMBOL: (p.entry_price + direction * 2.1 * risk, late)}, late)
    assert SYMBOL in shadow.state.portfolio.positions


@pytest.mark.parametrize("side", list(Side))
def test_live_vsb_exit_skips_mid_and_keeps_2r_hold_exemption(monkeypatch, side):
    one, _ = frames(side)
    cfg = load_settings(env_file=None)
    state = MultiPaperState()
    entry = float(one.iloc[-1]["close"])
    one.iloc[-1, :4] = [100.1, 100.2, 99.8, 100.]
    from bot.mr_candidate import MidExitState, mid_band, mid_exit_check
    assert mid_exit_check(side, entry, float(mid_band(one["close"], 20, True).iloc[-1]), 100.,
                          state=MidExitState(reached_mid=True))[0]
    stop = 90 if side is Side.LONG else 110
    state.portfolio.positions[SYMBOL] = Position(SYMBOL, side, 1, entry, stop, AT - timedelta(hours=25), "VSB", "MID")
    state.exit_management_by_symbol[SYMBOL] = dict(strategy_tag="VSB", box_mid=100., entry_atr=1., initial_risk=2.,
        protective_stop=stop, structure_low=90., structure_high=110., mid_reached=True,
        best_price=entry + (5 if side is Side.LONG else -5), worst_price=entry, last_close=entry, trailing_started=True)
    closed = []
    monkeypatch.setattr(runner, "_close_position", lambda *a: closed.append(a[5]))
    runner._evaluate_exits(state, None, True, {SYMBOL: one}, {}, cfg, 0, 0, AT)
    runner._apply_max_hold_exit(True, None, state, SYMBOL, state.portfolio.positions[SYMBOL], entry, AT, cfg, 0, 0)
    assert closed == []


def test_shadow_failures_do_not_change_live_account(monkeypatch, tmp_path):
    result, live, cfg, one, four = candidate(monkeypatch)
    live.mr_candidate_pool[SYMBOL]["status"] = "active"
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    monkeypatch.setattr(shadow, "save", lambda: (_ for _ in ()).throw(OSError("offline write failure")))
    before = live.to_dict()
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", AT, "BTC", "4h", {})
    runner._run_vsb_shadow_cycle(PublicOnly(), shadow, live, {SYMBOL: one}, {SYMBOL: four}, {}, cfg, AT, risk, None)
    assert live.to_dict() == before


@pytest.mark.parametrize("mode", ["DISABLED", "SHADOW", "LIVE"])
def test_live_hourly_scan_splits_before_candidate_and_risk_budget(monkeypatch, mode):
    cfg = load_settings(env_file=None).model_copy(update={"vsb_execution_mode": mode})
    state = MultiPaperState(execution_mode="lead_usdm", execution_status="synchronized")
    state.mr_candidate_pool = {"MR": dict(strategy_tag="MR"), "VSB": dict(strategy_tag="VSB")}
    monkeypatch.setattr(runner, "_utc_now", lambda: AT + timedelta(seconds=1))
    seen = []
    monkeypatch.setattr(runner, "_entry_candidate", lambda *a, **kw: seen.append(a[3]))
    monkeypatch.setattr(runner, "_submit_entry_candidate", lambda *a, **kw: pytest.fail("no actual candidate"))
    risk = MarketRiskAssessment(MarketRiskState.NORMAL, "normal", AT, "BTC", "4h", {})
    runner._evaluate_entries(None, state, None, True, {}, ["MR", "VSB"], {}, {}, {}, cfg,
                             5, 0, 0, AT, risk)
    assert seen == (["MR", "VSB"] if mode == "LIVE" else ["MR"])


@pytest.mark.parametrize("side", list(Side))
def test_effective_atr_rescues_a_compressed_stop_without_changing_limits(monkeypatch, side):
    from bot.indicators import atr
    result, _, cfg, one, _ = candidate(monkeypatch, side)
    raw = float(atr(one, 14).iloc[-1])
    structure = result[6] if side is Side.LONG else result[7]
    with pytest.raises(ValueError, match="stop_too_wide"):
        runner.calculate_adaptive_stop(side, result[3], raw, structure, cfg)
    stop, _ = runner.calculate_adaptive_stop(side, result[3], result[5], structure, cfg)
    assert .3 <= abs(result[3] - stop) / result[5] <= 2.0
    assert result[5] == max(raw, .75 * float(atr(one, 60).iloc[-1]))


@pytest.mark.parametrize("mode", ["DISABLED", "SHADOW", "LIVE"])
@pytest.mark.parametrize("adverse", [-.2, -.03, -.01, 0., .01, .02, .2])
@pytest.mark.parametrize("sqz", [0, 5, 6, 10, 24, 36, 37])
def test_pool_is_always_exclusive_and_disabled_never_routes_vsb(monkeypatch, mode, adverse, sqz):
    one, four = frames()
    close = 100 * (1 - adverse)
    one.iloc[-1, :4] = [close, close + .1, close - .1, close]
    monkeypatch.setattr(runner, "squeeze_context", lambda f: dict(squeeze_on=True, squeeze_released=False,
        squeeze_complete=True, sqz_bars=sqz, box_high=101., box_low=99., box_mid=100., squeeze_volume_mean=100.))
    cfg = load_settings(env_file=None).model_copy(update={"vsb_execution_mode": mode})
    state = MultiPaperState()
    row = runner._lead_structure(one, four, {"ready": True, "direction": "long"}, cfg, AT)
    runner._update_lead_pool(state, {SYMBOL: row}, AT)
    assert len(state.mr_candidate_pool) <= 1
    if SYMBOL in state.mr_candidate_pool:
        tag = state.mr_candidate_pool[SYMBOL]["strategy_tag"]
        assert tag in {"MR", "VSB"}
        assert not (mode == "DISABLED" and tag == "VSB")


@pytest.mark.parametrize("outcome", ["false_breakout", "no_progress", "hold"])
def test_shadow_hourly_exits_and_counters_are_durable(monkeypatch, tmp_path, outcome):
    result, _, cfg, one, _ = candidate(monkeypatch)
    shadow = VsbShadowLedger(tmp_path / "shadow.json", cfg)
    shadow.offer(result, result[3], AT + timedelta(seconds=1), 0., "NORMAL")
    if outcome == "false_breakout":
        one.iloc[-1, :4] = [100.1, 100.2, 99.8, 99.9]
        one.index += timedelta(hours=1)
        shadow.hourly({SYMBOL: one}, AT + timedelta(hours=1))
        assert not shadow.state.portfolio.positions
        assert shadow.state.events[-1]["reason"] == "vsb_false_breakout"
    elif outcome == "no_progress":
        meta = shadow.state.exit_management_by_symbol[SYMBOL]
        meta.update(no_progress_count=cfg.exits.no_progress_bars - 1, last_close=102.)
        # Past the falsification window, low MFE and a non-advancing close exit.
        one.index += timedelta(hours=7)
        one.iloc[-1, :4] = [101.5, 101.6, 101.4, 101.5]
        shadow.hourly({SYMBOL: one}, AT + timedelta(hours=7))
        assert shadow.state.events[-1]["reason"] == "no_progress"
    else:
        p = shadow.state.portfolio.positions[SYMBOL]
        at = AT + timedelta(hours=25)
        shadow.tick({SYMBOL: (p.entry_price, at)}, at)
        assert shadow.state.events[-1]["reason"] == "max_hold_24h"
    restored = VsbShadowLedger(shadow.path, cfg)
    assert not restored.state.portfolio.positions
