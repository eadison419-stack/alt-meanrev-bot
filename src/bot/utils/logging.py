"""Explicit logging setup. Callers must never log credentials or full settings."""

import sys
import re
import traceback
from pathlib import Path

from loguru import logger


_SECRET_KEY = (r"(?:[a-z0-9]+[_-])*(?:signature|api[_-]?key|api[_-]?secret|"
               r"secret|access[_-]?token|token|password)")
_QUOTED_SENSITIVE_VALUE = re.compile(
    rf"""(?i)(["']{_SECRET_KEY}["']\s*:\s*)(["'])(.*?)\2"""
)
_SENSITIVE_VALUE = re.compile(
    rf"""(?i)(\b{_SECRET_KEY}["']?\s*[:=]\s*["']?)(?!\[REDACTED\])([^&\s"',}}\]]+)"""
)


def redact_sensitive_text(value: str) -> str:
    value = _QUOTED_SENSITIVE_VALUE.sub(
        lambda match: match[1] + match[2] + "[REDACTED]" + match[2], value,
    )
    return _SENSITIVE_VALUE.sub(r"\1[REDACTED]", value)


def redact_log_record(record) -> bool:
    message = record["message"]
    exception = record["exception"]
    if exception is not None:
        message += "\n" + "".join(traceback.format_exception(*exception))
        # Format the chain ourselves without captured locals, then redact it.
        record["exception"] = None
    record["message"] = redact_sensitive_text(message)
    return True


def redacted_excepthook(exc_type, value, tb) -> None:
    sys.stderr.write(redact_sensitive_text(
        "".join(traceback.format_exception(exc_type, value, tb))
    ))


def configure_logging(level: str = "INFO", log_path: Path | None = None) -> None:
    logger.remove()
    options = dict(level=level, backtrace=False, diagnose=False, filter=redact_log_record)
    logger.add(sys.stderr, **options)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, encoding="utf-8", **options)
