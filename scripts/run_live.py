"""Live trading (Phase 7, disabled)."""
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description='Live trading (Phase 7, disabled)')
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("Live mode requires --live; trading is disabled in Phase 1.")
    parser.error("Not implemented in Phase 1. No exchange connection or order was made.")


if __name__ == "__main__":
    raise SystemExit(main())
