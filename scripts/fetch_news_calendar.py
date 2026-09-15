#!/usr/bin/env python3
"""Build data/news_calendar.csv from FRED and federalreserve.gov.

    python scripts/fetch_news_calendar.py                      # 2015-01-01 .. yesterday
    python scripts/fetch_news_calendar.py --start 2020-01-01

Needs a free FRED API key (https://fredaccount.stlouisfed.org/apikeys) in the
FRED_API_KEY environment variable or on a FRED_API_KEY= line in .env. The key
is never printed. Coverage ends the day before the fetch, so re-run this before
each evaluation block; run_pipeline.py warns about sessions it does not cover.
"""

import argparse
import os
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mesproto.config import (  # noqa: E402
    FOMC_EVENT, NEWS_CALENDAR_LAG_DAYS, NEWS_CALENDAR_START, NEWS_FRED_RELEASES,
)
from mesproto.news import (  # noqa: E402
    fetch_news_rows, load_news_calendar, missing_years, write_news_calendar,
)


def read_api_key(env_file: Path) -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key or not env_file.is_file():
        return key
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip().removeprefix("export ").strip() == "FRED_API_KEY":
            return value.strip().strip("'\"")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat, default=NEWS_CALENDAR_START)
    ap.add_argument("--end", type=date.fromisoformat, default=None,
                    help="last covered day (default and maximum: yesterday)")
    ap.add_argument("--out", default="data/news_calendar.csv")
    ap.add_argument("--env-file", default=".env",
                    help="read FRED_API_KEY from here if it is not in the environment")
    args = ap.parse_args()

    latest = date.today() - timedelta(days=NEWS_CALENDAR_LAG_DAYS)
    end = args.end or latest
    if end > latest:
        sys.exit(f"--end cannot be after {latest}: FRED lists a release only once its "
                 f"data loads, so a later day would look quiet when it is unknown")
    if args.start > end:
        sys.exit("--start is after --end")

    key = read_api_key(Path(args.env_file))
    if not key:
        sys.exit("no FRED API key: set FRED_API_KEY, or add FRED_API_KEY=... to "
                 f"{args.env_file}. Free at https://fredaccount.stlouisfed.org/apikeys")

    events = tuple(NEWS_FRED_RELEASES) + (FOMC_EVENT,)
    rows = fetch_news_rows(key, args.start, end)
    gaps = missing_years(rows, args.start, end, events)
    if gaps:
        sys.exit(f"refusing to write {args.out}: no dates for {', '.join(gaps)}. "
                 f"A source format probably changed.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sources = ("FRED release/dates (" + " ".join(f"{e}={r}" for e, r in
               NEWS_FRED_RELEASES.items()) + "); federalreserve.gov FOMC calendars, "
               f"scheduled meetings only; fetched {date.today().isoformat()}")
    write_news_calendar(args.out, rows, args.start, end, events, sources)
    cal = load_news_calendar(args.out)            # round-trip: the file must load

    counts = Counter(event for _, event in rows)
    print(f"wrote {args.out}: {cal.start} .. {cal.end}, {len(cal.days)} release days")
    for event in events:
        print(f"  {event:<13} {counts[event]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
