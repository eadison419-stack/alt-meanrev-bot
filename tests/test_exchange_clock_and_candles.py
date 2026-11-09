from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading
import pandas as pd

import ccxt
import pytest

from bot.config import load_settings
from bot.entry_quality import execution_quote
from bot.exchange import clock as clock_module
from bot.exchange.clock import ExchangeClock, ExchangeClockError
from bot.exchange.public_candles import CandleBatchReader, prefetched_read
from bot.execution import DemoExecutionService, EntryPreflightRejected
from bot.exchange.binance_usdm import FAPI_URLS
from bot.live.multi_paper import MultiPaperState
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 9, 0, tzinfo=timezone.utc)


def _clock(offset=1.6, delays=(.1, .08, .12)):
    mono, wall = [0.0], [NOW.timestamp()]
    clock = ExchangeClock(monotonic=lambda: mono[0], wall_time=lambda: wall[0] + mono[0])
    remaining = list(delays)

    def server(_):
        delay = remaining.pop(0) if remaining else .08
        mono[0] += delay
        return {"serverTime": int((wall[0] + mono[0] + offset - delay / 2) * 1000)}

    client = SimpleNamespace(fapiPublicGetTime=server)
    clock.synchronize(client)
    return clock, client, mono, wall


def _book(at):
    return {"timestamp": at.timestamp() * 1000, "bids": [[99, 10]], "asks": [[100, 10]]}


def test_measured_clock_offset_accepts_current_quote_without_future_tolerance(monkeypatch):
    clock, _, mono, wall = _clock()
    true_now = datetime.fromtimestamp(wall[0] + mono[0] + 1.6, timezone.utc)
    local_now = true_now - timedelta(seconds=1.6)
    assert execution_quote(_book(true_now), "long", local_now, max_age_sec=15)[1] == "entry_quote_from_future"
    quote, error = execution_quote(_book(true_now), "long", clock.now(), max_age_sec=15)
    assert error is None and 0 <= quote["age_sec"] < .1
    assert execution_quote(_book(clock.now() + timedelta(milliseconds=1)), "long", clock.now(), max_age_sec=15)[1] == "entry_quote_from_future"
    assert execution_quote(_book(clock.now() - timedelta(seconds=15.001)), "long", clock.now(), max_age_sec=15)[1] == "entry_quote_stale"
    monkeypatch.setattr(clock_module, "_active_clock", clock)
    assert runner._utc_now() == clock.now()


def test_os_clock_jumps_do_not_move_trading_clock_or_signed_timestamp():
    clock, client, mono, wall = _clock()
    before, signed = clock.now(), client.nonce()
    wall[0] += 3600
    mono[0] += 2
    assert (clock.now() - before).total_seconds() == pytest.approx(2)
    assert client.nonce() - signed == 2000
    wall[0] -= 7200
    mono[0] += 2
    assert (clock.now() - before).total_seconds() == pytest.approx(4)


def test_clock_refresh_does_not_move_backwards_when_network_improves():
    clock, client, mono, _ = _clock(delays=(.9, .9, .9))
    before = clock.now()
    mono[0] += 1
    clock.synchronize(client)
    assert clock.now() >= before + timedelta(seconds=1)


@pytest.mark.parametrize("response", [{}, {"serverTime": "bad"}, {"serverTime": float('nan')},
    {"serverTime": True}, {"serverTime": NOW.timestamp() * 1000 + 60000}])
def test_unreliable_time_response_never_installs_clock(response):
    clock = ExchangeClock(monotonic=lambda: 1, wall_time=lambda: NOW.timestamp())
    with pytest.raises(ExchangeClockError):
        clock.synchronize(SimpleNamespace(fapiPublicGetTime=lambda _: response))


def test_high_latency_calibration_fails_closed():
    with pytest.raises(ExchangeClockError):
        _clock(delays=(1.1, 1.2, 1.3))


def test_stale_clock_blocks_open_but_keeps_management_time(monkeypatch):
    clock, _, mono, _ = _clock()
    monkeypatch.setattr(clock_module, "_active_clock", clock)
    mono[0] += 301
    assert runner._utc_now().tzinfo is not None
    with pytest.raises(ExchangeClockError):
        runner._require_entry_clock()
    service = DemoExecutionService(SimpleNamespace(urls={"api": FAPI_URLS}), object())
    with pytest.raises(EntryPreflightRejected, match="clock_stale"):
        service.open(None, stop_price=98)


def test_refresh_failure_keeps_last_good_clock_and_records_pause(monkeypatch):
    clock, client, mono, _ = _clock()
    monkeypatch.setattr(clock_module, "_active_clock", clock)
    mono[0] += 301
    client.fapiPublicGetTime = lambda _: {}
    state = MultiPaperState()
    runner._refresh_exchange_clock(clock, client, state)
    assert state.events[-1]["new_entries_paused"] is True
    assert clock.now().tzinfo is not None


def test_background_funding_does_not_start_in_last_minute():
    assert runner._background_reads_allowed(NOW - timedelta(seconds=61))
    assert not runner._background_reads_allowed(NOW - timedelta(seconds=60))
    assert not runner._background_reads_allowed(NOW - timedelta(seconds=5))
    assert runner._background_reads_allowed(NOW)


def test_wait_returns_at_poll_deadline_before_slow_monitor(monkeypatch):
    tick = [0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    calls = []
    runner._wait_with_monitor(5, lambda: calls.append(1))
    assert calls == [] and tick[0] == 5


def test_public_read_budget_does_not_extend_entry_window():
    settings = load_settings(env_file=None)
    assert runner._public_read_deadline(NOW + timedelta(seconds=30), settings) == NOW + timedelta(seconds=60)
    assert runner._public_read_deadline(NOW + timedelta(minutes=10), settings) == NOW + timedelta(minutes=59, seconds=55)


def test_parallel_candles_use_distinct_clients_and_preserve_request_identity():
    clients, active, peak = [], [0], [0]
    lock = threading.Lock()
    gate = threading.Barrier(4)

    class Client:
        def __init__(self):
            self.owner = threading.get_ident()
            self.closed = False
            clients.append(self)

        def fetch_ohlcv(self, symbol, timeframe, *, limit):
            assert self.owner == threading.get_ident()
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            gate.wait(timeout=3)
            with lock:
                active[0] -= 1
            return [symbol, timeframe, limit]

        def close(self):
            self.closed = True

    reader = CandleBatchReader(None, lambda: NOW, factory=Client)
    requests = [(symbol, '1h', 60) for symbol in ['D', 'B', 'A', 'C']]
    try:
        result = reader.read_many(requests, NOW + timedelta(seconds=5))
        assert [result[key] for key in requests] == [list(key) for key in requests]
        assert peak[0] == 4 and len(clients) == 4
    finally:
        reader.close()
    assert all(c.closed for c in clients)


def test_expired_read_budget_starts_no_request():
    reader = CandleBatchReader(None, lambda: NOW, factory=lambda: pytest.fail('expired request started'))
    try:
        result = reader.read_many([('A', '1h', 60)], NOW - timedelta(seconds=1))
        assert isinstance(result[('A', '1h', 60)], ccxt.RequestTimeout)
    finally:
        reader.close()


def test_prefetched_failure_retains_bounded_retry_and_never_uses_wrong_symbol():
    key = ('A', '1h', 60)
    data = {key: ccxt.NetworkError('temporary'), ('B', '1h', 60): [2]}
    with pytest.raises(ccxt.NetworkError):
        prefetched_read(data, key, lambda: [1])
    assert prefetched_read(data, key, lambda: [1]) == [1]
    assert prefetched_read(data, ('B', '1h', 60), lambda: pytest.fail('wrong fallback')) == [2]


def test_time_calibration_survives_one_timeout_and_restores_client_timeout():
    clock = ExchangeClock(monotonic=lambda: 1, wall_time=lambda: NOW.timestamp())
    calls = []
    def response(_):
        calls.append(1)
        if len(calls) == 1:
            raise ccxt.RequestTimeout('temporary')
        return {'serverTime': NOW.timestamp() * 1000 + 1600}
    client = SimpleNamespace(fapiPublicGetTime=response, timeout=10000)
    clock.synchronize(client)
    assert client.timeout == 10000
    assert clock.now() == NOW + timedelta(seconds=1.6)


def test_rate_limit_stops_queued_public_reads():
    calls = []
    class Client:
        def fetch_ohlcv(self, *args, **kwargs):
            calls.append(1)
            raise ccxt.RateLimitExceeded('simulated 429')
        def close(self):pass
    reader = CandleBatchReader(None, lambda: NOW, factory=Client, workers=1)
    try:
        with pytest.raises(ccxt.RateLimitExceeded):
            reader.read_many([(str(i), '1h', 60) for i in range(20)], NOW + timedelta(seconds=5))
        assert len(calls) == 1
    finally:reader.close()


def _rows(period_ms, count=100):
    end = int(NOW.timestamp() * 1000)
    return [[end-(count-i)*period_ms, 100+i, 102+i, 99+i, 101+i, 10+i]
            for i in range(count+1)]


def test_prefetch_preserves_hotlist_ranking_and_closed_frames(monkeypatch):
    settings=load_settings(env_file=None)
    symbols=['A','B','C']
    data={s:_rows(3600000) for s in symbols}
    data['B'][-2][5]=1000
    class Client:
        def load_markets(self):return {}
        def fetch_tickers(self):return {s:{'quoteVolume':1000+i} for i,s in enumerate(symbols)}
        def fetch_ohlcv(self,symbol,timeframe,*,limit):return data[symbol]
    class ReorderedBatch:
        def read_many(self,requests,deadline,**kwargs):
            return {key:data[key[0]] for key in reversed(requests)}
    monkeypatch.setattr(runner,'filter_tradable_crypto_universe', lambda *a,**kw:[(s,1000) for s in symbols])
    old=runner._build_universe(Client(),settings,NOW,int(NOW.timestamp()*1000))
    new=runner._build_universe(Client(),settings,NOW,int(NOW.timestamp()*1000),candle_reader=ReorderedBatch())
    assert old[1]==new[1]
    for symbol in symbols:pd.testing.assert_frame_equal(old[2][symbol],new[2][symbol])


def test_prefetch_preserves_daily_gate_and_four_hour_stop_inputs():
    class Client:
        def fetch_ohlcv(self,symbol,timeframe,*,limit):return _rows({'4h':14400000,'1d':86400000}[timeframe])
    client=Client()
    old=runner._daily_direction_gate(client,'A',NOW,50)
    new=runner._daily_direction_gate(client,'A',NOW,50,prefetched={('A','1d',100):_rows(86400000)})
    assert old==new and old['direction']=='long'
    before=runner._cached_four_hour(client,'A',int(NOW.timestamp()*1000),runner.ClosedMarketCache())
    after=runner._cached_four_hour(client,'A',int(NOW.timestamp()*1000),runner.ClosedMarketCache(),
        prefetched={('A','4h',100):_rows(14400000)})
    pd.testing.assert_frame_equal(before,after)


def test_uncertainty_cannot_admit_unclosed_candle_at_hour_boundary(monkeypatch):
    clock, _, mono, _ = _clock(offset=0, delays=(.1,.1,.1))
    monkeypatch.setattr(clock_module,'_active_clock',clock)
    # Lower cutoff remains just before the hour while upper guard is just after it.
    boundary=NOW+timedelta(hours=1)
    mono[0] += (boundary-clock.lower_now()).total_seconds()-.025
    assert clock.lower_now()<boundary<clock.now()
    row=[int((boundary-timedelta(hours=1)).timestamp()*1000),100,101,99,100,1]
    assert runner._closed([row],3600000,int(clock.now().timestamp()*1000))==[]
    monkeypatch.setattr(runner.time,'sleep',lambda delay:mono.__setitem__(0,mono[0]+delay))
    runner._wait_for_closed_hour()
    assert runner._closed([row],3600000,int(clock.now().timestamp()*1000))==[row]


@pytest.mark.parametrize("failure_phase", ["submit", "first_lookup"])
def test_transient_order_not_found_reads_exact_fill_and_places_stop_once(monkeypatch, failure_phase):
    from bot.regime import Regime
    from bot.risk import OrderIntent
    from bot.signals import Action, Side

    monkeypatch.setattr("bot.execution.time.sleep", lambda _: None)
    calls = []
    position = {"side": "long", "quantity": 6, "entry_price": 100}

    def submit(*args):
        calls.append("market_submit")
        if failure_phase == "submit":
            raise ccxt.OrderNotFound("simulated propagation delay")
        return {"id": "123", "filled": 0}

    def lookup(params):
        assert params == {"symbol": "TESTUSDT", "origClientOrderId": "mrb_saved"}
        calls.append("exact_read")
        if calls.count("exact_read") == 1:
            raise ccxt.OrderNotFound("simulated propagation delay")
        return {"symbol": "TESTUSDT", "clientOrderId": "mrb_saved", "orderId": 123,
                "type": "MARKET", "positionSide": "BOTH", "side": "BUY", "reduceOnly": False,
                "origQty": "6", "executedQty": "6", "avgPrice": "100", "status": "FILLED"}

    def missing(*args):
        raise ccxt.OrderNotFound("simulated propagation delay")

    client = SimpleNamespace(urls={"api": FAPI_URLS}, create_order=submit,
                             fetch_order=missing, fapiPrivateGetOrder=lookup,
                             market=lambda _: {"id": "TESTUSDT"},
                             amount_to_precision=lambda _, value: str(value),
                             price_to_precision=lambda _, value: str(value))
    service = DemoExecutionService(client, load_settings(env_file=None))
    live = iter([None, None, position])
    monkeypatch.setattr(service, "_ensure_one_way", lambda: None)
    monkeypatch.setattr(service, "_prepare_symbol", lambda _: None)
    monkeypatch.setattr(service, "_market_rules", lambda _: (1, 1, 5))
    monkeypatch.setattr(service, "_position", lambda _: next(live))

    def protect(symbol, side, quantity, price):
        assert (symbol, side, quantity, price) == ("TEST/USDT:USDT", "long", 6, 98)
        calls.append("stop_submit")
        return {"id": "stop", "amount": quantity, "stop_price": price}

    monkeypatch.setattr(service, "_place_stop", protect)
    intent = OrderIntent("TEST/USDT:USDT", Side.LONG, Action.OPEN, 6, 100, "test", Regime.MID, 98)
    result = service.open(intent, stop_price=98, client_order_id="mrb_saved")
    assert result["fill"]["id"] == "123"
    assert calls == ["market_submit", "exact_read", "exact_read", "stop_submit"]


@pytest.mark.parametrize("case", ["missing", "wrong_side", "wrong_id"])
def test_ambiguous_fill_recovery_never_resends_or_uses_another_order(monkeypatch, case):
    monkeypatch.setattr("bot.execution.time.sleep", lambda _: None)
    submits = []

    def submit(*args):
        submits.append(args)
        raise ccxt.RequestTimeout("simulated lost response")

    def lookup(_):
        if case == "missing":
            raise ccxt.OrderNotFound("simulated missing")
        return {"symbol": "TESTUSDT", "clientOrderId": "other" if case == "wrong_id" else "mrb_saved",
                "orderId": 123, "side": "SELL" if case == "wrong_side" else "BUY", "type": "MARKET",
                "positionSide": "BOTH", "reduceOnly": False, "origQty": "6", "status": "FILLED",
                "executedQty": "6", "avgPrice": "100"}

    service = DemoExecutionService(SimpleNamespace(urls={"api": FAPI_URLS}, create_order=submit,
        fapiPrivateGetOrder=lookup, market=lambda _: {"id": "TESTUSDT"}), object())
    with pytest.raises((ccxt.OrderNotFound, RuntimeError)):
        service._market_fill("TEST/USDT:USDT", "buy", 6, reduce_only=False, client_order_id="mrb_saved")
    assert len(submits) == 1
