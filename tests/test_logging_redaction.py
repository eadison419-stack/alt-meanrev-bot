import io
import sys

import ccxt
from loguru import logger

from bot.utils.logging import redact_log_record, redact_sensitive_text, redacted_excepthook


def test_credentials_are_redacted_in_urls_json_and_environment_fields():
    raw = ('https://demo.invalid/path?timestamp=1&signature=example-signature&token=example-token '
           '{"apiKey": "example key with spaces", "password": "example-password"} '
           'BINANCE_DEMO_API_SECRET=example-secret')
    result = redact_sensitive_text(raw)
    assert redact_sensitive_text(result) == result
    for value in ("example-signature", "example-token", "example key with spaces",
                  "example-password", "example-secret"):
        assert value not in result
    assert "timestamp=1" in result and "[REDACTED]" in result


def test_loguru_exception_chain_and_unhandled_stderr_are_redacted(capsys):
    stream = io.StringIO()
    logger.remove()
    sink = logger.add(stream, format="{message}", filter=redact_log_record,
                      backtrace=False, diagnose=False)
    try:
        try:
            raise ccxt.NetworkError("GET https://demo.invalid/path?signature=example-signature")
        except ccxt.NetworkError as cause:
            try:
                raise RuntimeError("account initialization failed") from cause
            except RuntimeError:
                logger.exception("startup failed")
                redacted_excepthook(*sys.exc_info())
    finally:
        logger.remove(sink)
    for rendered in (stream.getvalue(), capsys.readouterr().err):
        assert "example-signature" not in rendered
        assert "[REDACTED]" in rendered
        assert "NetworkError" in rendered and "RuntimeError" in rendered
