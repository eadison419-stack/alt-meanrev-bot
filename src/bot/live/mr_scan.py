"""Durable MR entry progress; no strategy conditions or exchange writes here."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256

import ccxt


class ScanCheckpointError(RuntimeError):
    """Persistence failed: the runner must stop before any further entry."""


def checkpoint(save):
    if save is not None:
        try:
            save()
        except Exception as exc:
            raise ScanCheckpointError("MR scan checkpoint failed") from exc


def unresolved_attempts(state):
    return [row for row in state.mr_entry_attempts.values()
            if row.get("status") in {"submitting", "uncertain", "protected"}]


def scan_for(state, bar, symbols, delay_seconds, save=None):
    key = bar.isoformat()
    scan = state.mr_entry_scan
    if scan.get("bar") != key:
        if scan and scan.get("status") not in {"completed", "expired"}:
            state.events.append({"event": "entry_scan_expired", "bar": scan["bar"],
                                 "deadline": scan["deadline"], "superseded_by_bar": key,
                                 "unfinished_symbols": [symbol for symbol in scan["symbols"]
                                                        if scan["outcomes"].get(symbol, {}).get("status")
                                                        in {None, "retry", "evaluating", "candidate"}]})
        scan = {"bar": key, "deadline": (bar + timedelta(hours=1, seconds=delay_seconds)).isoformat(),
                "symbols": list(dict.fromkeys(symbols)), "outcomes": {}, "status": "pending"}
        # On upgrade an old ledger has no scan journal. Its confirmed OPENs
        # must still consume the candle, including entries already closed.
        close_at = bar + timedelta(hours=1)
        for event in state.events:
            symbol = event.get("symbol")
            if (symbol not in scan["symbols"] or event.get("event") != "paper_fill"
                    or event.get("action") != "OPEN" or event.get("source") != "MEAN_REVERSION"):
                continue
            try:
                filled_at = datetime.fromisoformat(event["at"])
                if filled_at.tzinfo is None:
                    filled_at = filled_at.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if close_at <= filled_at < close_at + timedelta(hours=1):
                scan["outcomes"][symbol] = {"bar": key, "side": event.get("side"),
                                            "attempts": 0, "status": "opened",
                                            "restored_from": "paper_fill"}
        for attempt in state.mr_entry_attempts.values():
            if (attempt.get("bar") == key and attempt.get("symbol") in scan["symbols"]
                    and attempt.get("status") in {"completed", "not_filled", "not_submitted"}):
                scan["outcomes"][attempt["symbol"]] = {
                    "bar": key, "side": attempt["side"], "attempts": 0,
                    "status": "reconciled", "restored_from": "entry_attempt",
                }
        state.mr_entry_scan = scan
        checkpoint(save)
    return scan


def client_order_id(symbol, side, bar):
    identity = f"{symbol}|{side}|{bar}"
    return "mrb_" + sha256(identity.encode()).hexdigest()[:28]


def retryable_read(exc):
    return (isinstance(exc, (ccxt.NetworkError, ccxt.ExchangeNotAvailable))
            and not isinstance(exc, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)))


def symbol_error(exc):
    return isinstance(exc, (ccxt.BadSymbol, ccxt.InvalidOrder))


def outcome(state, scan, symbol, status, at, save=None, **details):
    row = scan["outcomes"].setdefault(symbol, {"bar": scan["bar"], "attempts": 0})
    row.update(status=status, **details)
    state.events.append({"event": "entry_scan_outcome", "at": at.isoformat(),
                         "symbol": symbol, **row})
    checkpoint(save)
    return row


def attempt_outcome(state, attempt, status, at, **details):
    state.events.append({
        "event": "entry_attempt_outcome", "at": at.isoformat(),
        **{key: attempt[key] for key in ("symbol", "side", "bar", "clientOrderId")},
        "status": status, "endpoint": "demo", **details,
    })


def expired(scan, at):
    return at > datetime.fromisoformat(scan["deadline"])
