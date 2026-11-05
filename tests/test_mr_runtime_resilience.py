import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import ccxt
import pytest

from bot.config import load_settings
from bot.live import mr_scan
from bot.live.multi_paper import MultiPaperState
from bot.portfolio import Position
from bot.signals import Side
from scripts import run_mr_candidate_paper as runner


NOW = datetime(2026, 11, 5, 5, 0, 20, tzinfo=timezone.utc)


def test_checkpoint_preserves_old_file_then_persists_failure_and_recovery(tmp_path, monkeypatch):
    state = MultiPaperState()
    path = tmp_path / "state.json"
    state.save(path)
    original = path.read_bytes()
    save = state.save
    locked = [True]

    def persist(target):
        if locked[0]:
            raise PermissionError("locked")
        save(target)

    monkeypatch.setattr(state, "save", persist)
    checkpoint = runner._RunnerCheckpoint(state, path)
    assert not checkpoint.try_save()
    assert not checkpoint.try_save()
    assert checkpoint.failed and path.read_bytes() == original
    assert len(state.events) == 1
    locked[0] = False
    assert checkpoint.try_save() and not checkpoint.failed
    assert [e["event"] for e in json.loads(path.read_text())["events"]] == [
        "state_persistence_unavailable", "state_persistence_recovered",
    ]


def test_startup_retries_only_bounded_network_reads(monkeypatch):
    calls, sleeps = [], []

    def initialize():
        calls.append(1)
        if len(calls) < 3:
            raise ccxt.NetworkError("offline")
        return {"equity": 5000}

    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert runner._initialize_demo(SimpleNamespace(initialize=initialize)) == {"equity": 5000}
    assert len(calls) == 3 and sleeps == [1, 2]
    calls.clear()

    def denied():
        calls.append(1)
        raise ccxt.AuthenticationError("denied")

    with pytest.raises(ccxt.AuthenticationError):
        runner._initialize_demo(SimpleNamespace(initialize=denied))
    assert len(calls) == 1


def test_startup_network_failure_stops_after_three_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    def unavailable():
        calls.append(1)
        raise ccxt.NetworkError("offline")

    with pytest.raises(ccxt.NetworkError):
        runner._initialize_demo(SimpleNamespace(initialize=unavailable))
    assert len(calls) == 3


@pytest.mark.parametrize("expired", [False, True])
def test_public_read_retry_respects_deadline_and_records_endpoint(monkeypatch, expired):
    calls, monitored = [], []
    monkeypatch.setattr(runner, "_utc_now", lambda: NOW)

    def read():
        calls.append(1)
        raise ccxt.NetworkError("GET ?signature=example-signature")

    with pytest.raises(ccxt.NetworkError) as caught:
        runner._read_entry_data(
            read, phase="universe_tickers", retry=True,
            deadline=NOW + timedelta(seconds=-1 if expired else 20),
            monitor=lambda: monitored.append(1),
        )
    assert len(calls) == (1 if expired else 2)
    assert len(monitored) == 1
    assert caught.value.mr_phase == "universe_tickers"
    assert caught.value.mr_endpoint == "public"


def _mock_main(monkeypatch, tmp_path, state):
    settings = load_settings(env_file=None)
    monkeypatch.setattr(sys, "argv", ["runner", "--demo", "--once", "--state",
                                      str(tmp_path / "state.json")])
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_settings", lambda: settings)
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(runner.ccxt, "binanceusdm", lambda *a: client)
    monkeypatch.setattr(runner, "configure_system_proxy", lambda *a: None)
    monkeypatch.setattr(runner, "create_demo_usdm", lambda *a: client)
    monkeypatch.setattr(runner, "DemoExecutionService", lambda *a: client)
    monkeypatch.setattr(runner.MultiPaperState, "load", lambda *a, **kw: state)
    monkeypatch.setattr(runner, "_initialize_demo", lambda *a: {"equity": 5000})
    monkeypatch.setattr(runner, "_reconcile_pending_entries", lambda *a: True)
    monkeypatch.setattr(runner, "_demo_sync_stopped", lambda *a: True)
    monkeypatch.setattr(runner, "logger", SimpleNamespace(**{
        name: lambda *a, **kw: None for name in ("remove", "add", "info", "warning", "error", "exception")
    }))


def test_runner_keeps_managing_known_positions_but_never_scans_with_failed_storage(monkeypatch, tmp_path):
    state = MultiPaperState()
    state.portfolio.positions["KNOWN"] = Position("KNOWN", Side.LONG, 1, 100, 95, NOW, "test", "MID")
    _mock_main(monkeypatch, tmp_path, state)
    managed = []
    monkeypatch.setattr(runner, "_manage_demo_positions", lambda *a: managed.append(1) or True)
    monkeypatch.setattr(runner, "_cycle", lambda *a, **kw: pytest.fail("entries evaluated with unavailable storage"))

    def locked(*a):
        raise PermissionError("locked")

    monkeypatch.setattr(state, "save", locked)
    assert runner.main() == 1
    assert managed and state.portfolio.positions


def test_failed_first_account_initialization_does_not_create_a_fake_equity_baseline(monkeypatch, tmp_path):
    state = MultiPaperState()
    _mock_main(monkeypatch, tmp_path, state)

    def unavailable(*a):
        raise ccxt.NetworkError("GET ?signature=example-signature")

    monkeypatch.setattr(runner, "_initialize_demo", unavailable)
    monkeypatch.setattr(state, "save", lambda *a: pytest.fail("uninitialized baseline persisted"))
    monkeypatch.setattr(runner, "_cycle", lambda *a, **kw: pytest.fail("startup failure reached entries"))
    assert runner.main() == 1
    assert not (tmp_path / "state.json").exists()


def test_runner_recovers_storage_before_resuming_entry_processing(monkeypatch, tmp_path):
    state = MultiPaperState()
    _mock_main(monkeypatch, tmp_path, state)
    attempts, cycles = [], []

    def persist(*a):
        attempts.append(1)
        if len(attempts) < 2:
            raise PermissionError("locked")

    monkeypatch.setattr(state, "save", persist)
    monkeypatch.setattr(runner, "_cycle", lambda *a, **kw: cycles.append(len(attempts)))
    assert runner.main() == 0
    assert len(cycles) == 1 and cycles[0] >= 2


def test_long_running_storage_outage_keeps_monitoring_until_recovery(monkeypatch, tmp_path):
    state = MultiPaperState()
    state.portfolio.positions["KNOWN"] = Position("KNOWN", Side.LONG, 1, 100, 95, NOW, "test", "MID")
    _mock_main(monkeypatch, tmp_path, state)
    monkeypatch.setattr(sys, "argv", sys.argv[:2] + sys.argv[3:])  # continuous mode
    locked, managed, cycles = [True], [], []

    def persist(*a):
        if locked[0]:
            raise PermissionError("locked")

    def wait(seconds, monitor):
        assert not cycles
        monitor(force=True)
        locked[0] = False

    def cycle(*a, **kw):
        assert not locked[0]
        cycles.append(1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(state, "save", persist)
    monkeypatch.setattr(runner, "_manage_demo_positions", lambda *a: managed.append(1) or True)
    monkeypatch.setattr(runner, "_wait_with_monitor", wait)
    monkeypatch.setattr(runner, "_cycle", cycle)
    with pytest.raises(KeyboardInterrupt):
        runner.main()
    assert len(managed) >= 3 and cycles == [1]


def test_failed_pre_submit_storage_never_becomes_an_ambiguous_order(monkeypatch):
    state = MultiPaperState()
    row = {"bar": (NOW - timedelta(hours=1)).isoformat()}
    saves, sent = [], []

    def persist():
        saves.append(1)
        if len(saves) > 1:
            raise PermissionError("locked")

    class Execution:
        def open(self, intent, *, stop_price, client_order_id, on_submit):
            on_submit(intent.quantity, stop_price, NOW)
            sent.append(1)

    with pytest.raises(mr_scan.ScanCheckpointError):
        runner._open_position(True, Execution(), state, "KNOWN", Side.LONG, 100, 95, 1,
                              3, 0, 0, "test", "MID", NOW, scan_row=row, checkpoint=persist)
    assert not sent and not mr_scan.unresolved_attempts(state)
    assert row["order_may_have_been_sent"] is False
    assert state.events[-1]["status"] == "not_submitted"
