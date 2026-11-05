"""Background launcher for the CTA dashboard (pythonw-friendly).

Runs scripts/run_dashboard.py with stdout/stderr redirected to a log file and
the process id recorded, so the dashboard can run under pythonw.exe (no console)
and be stopped by pid. The browser tab still opens via webbrowser.open.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stream = open(log_dir / "cta_dashboard.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = stream
    sys.stderr = stream
    pid_path = _ROOT / "data" / "paper" / "cta_dashboard.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    target = Path(__file__).with_name("run_dashboard.py")
    sys.argv = [str(target)] + sys.argv[1:]
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
