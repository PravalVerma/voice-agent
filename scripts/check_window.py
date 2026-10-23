"""Check the compliance calling window for a borrower timezone and time.

Usage: uv run python scripts/check_window.py --tz Asia/Kolkata [--at 2026-10-23T19:30]

--at is read as the borrower's OWN local wall-clock time, which is how the
gate is specified: the window is never evaluated against server time.
"""

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from _bootstrap import load_config_or_die

from app.compliance import check_calling_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tz", default="Asia/Kolkata", help="borrower IANA timezone (default: Asia/Kolkata)")
    parser.add_argument("--at", help="borrower-local time as YYYY-MM-DDTHH:MM (default: now)")
    args = parser.parse_args()

    cfg = load_config_or_die()

    try:
        zone = ZoneInfo(args.tz)
    except ZoneInfoNotFoundError:
        # Windows ships no IANA database, so this is the likely failure here and
        # it is worth naming the fix rather than printing a bare lookup error.
        sys.exit(
            f"error: no time zone data for {args.tz!r} - if this is a valid name, "
            f"the tzdata package is missing: uv sync"
        )

    if args.at:
        try:
            when = datetime.fromisoformat(args.at).replace(tzinfo=zone)
        except ValueError:
            sys.exit(f"error: --at must be YYYY-MM-DDTHH:MM, got {args.at!r}")
    else:
        when = datetime.now(zone)

    c = cfg.compliance
    print(f"window: {c.call_window_start}-{c.call_window_end} on weekdays {c.allowed_weekdays} (Mon=0)")
    print(f"borrower-local: {when:%Y-%m-%d %H:%M %Z} ({when:%A})")

    decision = check_calling_window(args.tz, when, cfg=cfg)
    if decision.ok:
        print("PASS - a call may be placed now")
    else:
        print(f"BLOCK {decision.code} - {decision.detail}")


if __name__ == "__main__":
    main()
