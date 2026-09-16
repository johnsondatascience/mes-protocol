"""Scheduled-release calendar for the news stand-downs.

S1 stands down before S1_NEWS_STAND_DOWN_UNTIL on days carrying any release in
S1_NEWS_EVENTS. Those days come from a local CSV that
scripts/fetch_news_calendar.py builds from two public sources:

  - FRED release dates (St. Louis Fed; free API key) for the releases in
    NEWS_FRED_RELEASES. These are the dates each release actually came out —
    the 2025 shutdown delays included — not the schedule as first published.
    FRED also lists a few annual-revision days (about one a year per release).
    They count as release days: a handful of extra stand-down mornings a year,
    and never a missed release.
  - federalreserve.gov FOMC calendars, for the statement day (the last day)
    of each scheduled meeting. Unscheduled meetings, notation votes and
    cancelled meetings are skipped: none could be planned around that morning.

A calendar states the dates and event types it is complete for. Anything
outside that is unknown — None — never "no news" (invariant 2): a calendar
fetched in March says nothing about June.

Parsing is pure and tested offline. Only the fetch_* functions touch the
network (through mesproto.sources), and nothing runs on import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Collection, Iterable, Mapping, Optional

from .config import FOMC_CALENDAR_URL, FOMC_EVENT, FOMC_HISTORICAL_URL, NEWS_FRED_RELEASES
from .sources import fetch_fred_release_dates, http_get

CSV_HEADER = "date,event"


@dataclass(frozen=True)
class NewsCalendar:
    """Release days, plus the date range and event types they are complete for."""
    start: date
    end: date
    events: frozenset[str]
    days: Mapping[date, frozenset[str]] = field(default_factory=dict)

    def is_news_day(self, d: date, events: Collection[str]) -> Optional[bool]:
        """True if any of `events` falls on `d`; None if the calendar cannot say."""
        wanted = set(events)
        if not self.start <= d <= self.end or not wanted <= self.events:
            return None
        return bool(self.days.get(d, frozenset()) & wanted)


# ---------------------------------------------------------------------------
# the local file
# ---------------------------------------------------------------------------

def write_news_calendar(path: str, rows: Iterable[tuple[date, str]], start: date,
                        end: date, events: Collection[str], sources: str) -> None:
    """Write the calendar CSV with its coverage and event types declared up top."""
    rows = sorted(set(rows))
    for d, event in rows:
        if event not in events or not start <= d <= end:
            raise ValueError(f"row {d},{event} is outside the declared calendar")
    lines = [
        "# mesproto news calendar: scheduled releases for the news stand-downs",
        f"# coverage: {start.isoformat()} {end.isoformat()}",
        f"# events: {','.join(sorted(events))}",
        f"# sources: {sources}",
        CSV_HEADER,
    ] + [f"{d.isoformat()},{event}" for d, event in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def load_news_calendar(path: str) -> NewsCalendar:
    """Read a calendar CSV. The '# coverage:' and '# events:' lines are required:
    a list of dates alone cannot tell a quiet day from a day it never covered."""
    start = end = None
    events: Optional[frozenset[str]] = None
    days: dict[date, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.strip()
            where = f"{path}:{n}"
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                key = key.strip().lower()
                if key == "coverage":
                    try:
                        first, last = value.split()
                        start, end = date.fromisoformat(first), date.fromisoformat(last)
                    except ValueError:
                        raise ValueError(f"{where}: coverage must be 'START END' "
                                         f"as YYYY-MM-DD") from None
                    if start > end:
                        raise ValueError(f"{where}: coverage starts after it ends")
                elif key == "events":
                    events = frozenset(e.strip() for e in value.split(",") if e.strip())
                continue
            if line.replace(" ", "") == CSV_HEADER:
                continue
            if start is None or end is None or events is None:
                raise ValueError(f"{where}: the '# coverage:' and '# events:' lines "
                                 f"must come before the rows")
            day_field, _, event = line.partition(",")
            try:
                d = date.fromisoformat(day_field.strip())
            except ValueError:
                raise ValueError(f"{where}: not a YYYY-MM-DD date: "
                                 f"{day_field.strip()!r}") from None
            event = event.strip()
            if event not in events:
                raise ValueError(f"{where}: event {event!r} is not declared in "
                                 f"'# events:' {sorted(events)}")
            if not start <= d <= end:
                raise ValueError(f"{where}: {d} is outside the declared coverage "
                                 f"{start}..{end}")
            days.setdefault(d, set()).add(event)
    if start is None or end is None or events is None:
        raise ValueError(f"{path}: missing the '# coverage: START END' or "
                         f"'# events: A,B' line")
    return NewsCalendar(start, end, events, {d: frozenset(v) for d, v in days.items()})


def missing_years(rows: Iterable[tuple[date, str]], start: date, end: date,
                  events: Collection[str]) -> list[str]:
    """'EVENT YEAR' for each full covered year in which an event never occurs.

    Every release here happens every year, so a gap means the source format
    changed and a parser skipped it — the calendar must not be written.
    """
    seen = {(event, d.year) for d, event in rows}
    first = start.year if start == date(start.year, 1, 1) else start.year + 1
    last = end.year if end == date(end.year, 12, 31) else end.year - 1
    return [f"{event} {year}" for event in sorted(events)
            for year in range(first, last + 1) if (event, year) not in seen]


# ---------------------------------------------------------------------------
# federalreserve.gov FOMC calendars
# ---------------------------------------------------------------------------

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
# current page: one panel per year, a month cell then a days cell per meeting
_YEAR_PANEL = re.compile(r">(\d{4}) FOMC Meetings<")
_MEETING_CELL = re.compile(
    r'class="[^"]*\bfomc-meeting__(month|date)\b[^"]*"[^>]*>\s*(?:<strong>)?([^<]*)')
# historical pages: "January 28-29 Meeting - 2020", "March 19 (notation vote) - 2020"
_HISTORICAL_HEADING = re.compile(r'<h5 class="panel-heading[^"]*">\s*([^<]*?)\s*</h5>')
_HISTORICAL_TEXT = re.compile(r"^(\S+)\s+(.+?)\s+-\s+(\d{4})$")
_DAYS = re.compile(r"^(\d{1,2})(?:-(\d{1,2}))?\*?$")


def _month_number(text: str) -> int:
    key = text.strip().lower().rstrip(".")
    for i, name in enumerate(_MONTHS, start=1):
        if key in (name, name[:3]) or (i == 9 and key == "sept"):
            return i
    raise ValueError(f"unrecognised month {text!r} in an FOMC calendar")


def _statement_day(year: int, months: str, days: str) -> Optional[date]:
    """Last day of a scheduled meeting; None for anything that is not one."""
    if "(" in days:
        return None                       # (unscheduled), (cancelled), (notation vote)
    m = _DAYS.match(days.strip())
    if m is None:
        raise ValueError(f"unrecognised FOMC meeting days {months!r} {days!r} ({year})")
    parts = [_month_number(p) for p in months.split("/")]
    return date(year, parts[-1], int(m.group(2) or m.group(1)))


def parse_fomc_calendar(html: str) -> list[date]:
    """Statement days of scheduled FOMC meetings, from either page layout."""
    out: list[date] = []

    panels = list(_YEAR_PANEL.finditer(html))
    for k, panel in enumerate(panels):
        year = int(panel.group(1))
        stop = panels[k + 1].start() if k + 1 < len(panels) else len(html)
        month: Optional[str] = None
        for kind, text in _MEETING_CELL.findall(html[panel.end():stop]):
            text = text.strip()
            if kind == "month":
                if month is not None:
                    raise ValueError(f"FOMC {year}: month {month!r} has no meeting days")
                month = text
                continue
            if month is None:
                raise ValueError(f"FOMC {year}: meeting days {text!r} with no month")
            d = _statement_day(year, month, text)
            if d is not None:
                out.append(d)
            month = None
        if month is not None:
            raise ValueError(f"FOMC {year}: month {month!r} has no meeting days")

    for heading in _HISTORICAL_HEADING.findall(html):
        m = _HISTORICAL_TEXT.match(heading)
        if m is None:
            continue                          # not a meeting heading
        months, days, year = m.groups()
        days = re.sub(r"\s+Meeting$", "", days)
        d = _statement_day(int(year), months, days)
        if d is not None:
            out.append(d)

    return sorted(set(out))


# ---------------------------------------------------------------------------
# network — called only by scripts/fetch_news_calendar.py
# ---------------------------------------------------------------------------

def fetch_fomc_statement_days(first_year: int) -> list[date]:
    """Scheduled FOMC statement days from `first_year` on. The main calendar
    page holds recent years; older years each have a historical page."""
    days = parse_fomc_calendar(http_get(FOMC_CALENDAR_URL).decode("utf-8", "replace"))
    if not days:
        raise ValueError("no FOMC meetings found on the calendar page — layout changed?")
    for year in range(first_year, min(d.year for d in days)):
        page = http_get(FOMC_HISTORICAL_URL.format(year=year)).decode("utf-8", "replace")
        days += parse_fomc_calendar(page)
    return sorted(set(days))


def fetch_news_rows(api_key: str, start: date, end: date) -> list[tuple[date, str]]:
    """(date, event) for every FRED release in NEWS_FRED_RELEASES and every
    scheduled FOMC statement day in [start, end]."""
    rows = [(d, event) for event, release_id in NEWS_FRED_RELEASES.items()
            for d in fetch_fred_release_dates(release_id, api_key)]
    rows += [(d, FOMC_EVENT) for d in fetch_fomc_statement_days(start.year)]
    return sorted((d, event) for d, event in rows if start <= d <= end)
