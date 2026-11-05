"""Persistent file queue for explicit paper-position close requests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4


_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


def command_directory_for_state(state_path: str | Path) -> Path:
    target = Path(state_path)
    return target.parent / f"{target.stem}.manual-close"


class ManualCloseCommandStore:
    """Coordinate dashboard requests with the single paper-runner writer."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def _request_path(self, request_id: str) -> Path:
        if not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("invalid manual close request id")
        return self.directory / f"{request_id}.json"

    def _result_path(self, request_id: str) -> Path:
        self._request_path(request_id)
        return self.directory / f"{request_id}.result.json"

    def submit(self, symbol: str) -> dict[str, str]:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol is required")
        self.directory.mkdir(parents=True, exist_ok=True)
        request_id = uuid4().hex
        request = {"request_id": request_id, "symbol": symbol,
                   "created_at": datetime.now(timezone.utc).isoformat()}
        target = self._request_path(request_id)
        temporary = self.directory / f"{request_id}.tmp"
        temporary.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        return request

    def submit_unique(self, symbol: str) -> dict[str, str] | None:
        with self._lock:
            if self.pending_for_symbol(symbol):
                return None
            return self.submit(symbol)

    def pending(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        requests = []
        for path in self.directory.glob("*.json"):
            if path.name.endswith(".result.json"):
                continue
            request_id = path.stem
            if not _REQUEST_ID.fullmatch(request_id) or self._result_path(request_id).exists():
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if item.get("request_id") == request_id and isinstance(item.get("symbol"), str):
                requests.append(item)
        return sorted(requests, key=lambda row: (row.get("created_at", ""), row["request_id"]))

    def pending_for_symbol(self, symbol: str) -> bool:
        return any(item["symbol"] == symbol for item in self.pending())

    def complete(self, request_id: str, result: dict[str, Any]) -> None:
        target = self._result_path(request_id)
        if target.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f"{request_id}.result.tmp"
        payload = {"request_id": request_id, "completed_at": datetime.now(timezone.utc).isoformat(), **result}
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    def status(self, request_id: str) -> dict[str, Any] | None:
        result_path = self._result_path(request_id)
        request_path = self._request_path(request_id)
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        if request_path.exists():
            return {"request_id": request_id, "status": "queued"}
        return None
