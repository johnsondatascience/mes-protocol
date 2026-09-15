#!/usr/bin/env python3
"""Tests for mesproto.news — the scheduled-release calendar.

The fixtures below are trimmed copies of the real FRED and federalreserve.gov
responses, so the parsers are pinned to the formats the fetch script meets
without the suite touching the network.
"""
import os
import tempfile
from datetime import date

from mesproto.news import (
    NewsCalendar, load_news_calendar, missing_years, parse_fomc_calendar,
    parse_fred_release_dates, write_news_calendar,
)

# federalreserve.gov/monetarypolicy/fomccalendars.htm — one panel per year
FOMC_CURRENT = """
<div class="panel panel-default"><div class="panel-heading"><h5 class="panel-title text-capitalize">FOMC Search</h5></div></div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="42828">2026 FOMC Meetings</a></h4></div>
        <div class="row fomc-meeting" ">
            <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>January</strong></div>
            <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">27-28</div>
            <div class="fomc-meeting__minutes col-xs-12">Minutes</div>
        <div class="fomc-meeting--shaded row fomc-meeting" ">
            <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>March</strong></div>
            <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">17-18*</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="42827">2025 FOMC Meetings</a></h4></div>
            <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>July</strong></div>
            <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">29-30</div>
    <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>August</strong></div>
    <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-2">22 (notation vote)</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="39116">2024 FOMC Meetings</a></h4></div>
            <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2"><strong>Apr/May</strong></div>
            <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">30-1</div>
</div>
"""

# federalreserve.gov/monetarypolicy/fomchistorical{2018,2019,2020}.htm
FOMC_HISTORICAL = """
<h5 class="panel-heading panel-heading--shaded">Jul/Aug 31-1 Meeting - 2018</h5>
<h5 class="panel-heading panel-heading--shaded">April/May 30-1 Meeting - 2019</h5>
<h5 class="panel-heading panel-heading--shaded">January 28-29 Meeting - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">March 2 (unscheduled) Meeting - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">March 15 (unscheduled) Meeting - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">March 17-18 (cancelled) Meeting - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">March 19 (notation vote) - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">April 28-29 Meeting - 2020</h5>
"""


def test_fomc_current_page_gives_scheduled_statement_days():
    """The statement comes on a meeting's last day. A meeting spanning two
    months ends in the second. Notation votes are not meetings."""
    got = parse_fomc_calendar(FOMC_CURRENT)
    assert got == [date(2024, 5, 1), date(2025, 7, 30),
                   date(2026, 1, 28), date(2026, 3, 18)], got
    print(f"  {[d.isoformat() for d in got]}")


def test_fomc_historical_page_skips_unscheduled_and_cancelled():
    """An emergency meeting could not have been known that morning, so it
    cannot be a pre-planned stand-down; a cancelled one did not happen."""
    got = parse_fomc_calendar(FOMC_HISTORICAL)
    assert got == [date(2018, 8, 1), date(2019, 5, 1),
                   date(2020, 1, 29), date(2020, 4, 29)], got
    print(f"  {[d.isoformat() for d in got]}")


def test_fomc_unrecognised_date_raises():
    """A layout change must fail loudly, not quietly drop meetings."""
    for html in (FOMC_CURRENT.replace("29-30", "29–30th"),
                 FOMC_HISTORICAL.replace("April 28-29 Meeting", "Aprl 28-29 Meeting")):
        try:
            parse_fomc_calendar(html)
        except ValueError:
            continue
        raise AssertionError("an unparseable meeting must raise")
    print("  malformed day and month both raise")


def test_fred_release_dates():
    payload = {"count": 3, "limit": 10000, "release_dates": [
        {"release_id": 10, "date": "2025-09-11"},
        {"release_id": 10, "date": "2025-10-24"},
        {"release_id": 10, "date": "2025-12-18"}]}
    assert parse_fred_release_dates(payload) == \
        [date(2025, 9, 11), date(2025, 10, 24), date(2025, 12, 18)]

    for bad in ({"error_code": 400, "error_message": "Bad Request."},
                {**payload, "count": 20000}):            # truncated response
        try:
            parse_fred_release_dates(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must raise")
    print("  dates parsed; API error and truncated response raise")


def test_missing_years_flags_a_full_year_with_no_releases():
    """Every release in the calendar happens every year. A full covered year
    with none means the source format changed and the parser skipped it."""
    rows = [(date(2024, 3, 12), "CPI"), (date(2025, 3, 12), "CPI"),
            (date(2024, 3, 20), "FOMC")]
    gaps = missing_years(rows, start=date(2023, 6, 1), end=date(2026, 2, 1),
                         events=("CPI", "FOMC"))
    assert gaps == ["FOMC 2025"], gaps          # 2023 and 2026 are partial years
    print(f"  {gaps}")


def _calendar_file(text):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_calendar_round_trip_and_coverage():
    """Invariant 2: a day the calendar does not cover, or an event type it does
    not carry, is unknown (None) — never 'no news'."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_news_calendar(path, [(date(2026, 3, 6), "NFP"), (date(2026, 3, 11), "CPI")],
                            start=date(2026, 1, 1), end=date(2026, 3, 31),
                            events=("CPI", "NFP"), sources="test")
        cal = load_news_calendar(path)
    finally:
        os.remove(path)
    assert isinstance(cal, NewsCalendar)
    assert cal.is_news_day(date(2026, 3, 6), ("CPI", "NFP")) is True
    assert cal.is_news_day(date(2026, 3, 9), ("CPI", "NFP")) is False
    assert cal.is_news_day(date(2026, 3, 6), ("CPI",)) is False, "NFP is not asked for"
    assert cal.is_news_day(date(2026, 4, 1), ("CPI", "NFP")) is None, "outside coverage"
    assert cal.is_news_day(date(2026, 3, 9), ("CPI", "FOMC")) is None, "FOMC not carried"
    print("  event day True; quiet covered day False; uncovered day or event None")


def test_calendar_file_must_declare_what_it_covers():
    good = "# coverage: 2026-01-01 2026-03-31\n# events: CPI,NFP\ndate,event\n2026-03-06,NFP\n"
    path = _calendar_file(good)
    try:
        assert load_news_calendar(path).is_news_day(date(2026, 3, 6), ("NFP",)) is True
    finally:
        os.remove(path)

    broken = {
        "no coverage line": good.replace("# coverage: 2026-01-01 2026-03-31\n", ""),
        "no events line": good.replace("# events: CPI,NFP\n", ""),
        "event outside coverage": good + "2026-04-10,CPI\n",
        "undeclared event": good + "2026-03-18,FOMC\n",
        "malformed date": good + "March 11,CPI\n",
    }
    for why, text in broken.items():
        path = _calendar_file(text)
        try:
            load_news_calendar(path)
        except ValueError:
            continue
        finally:
            os.remove(path)
        raise AssertionError(f"{why}: must raise")
    print(f"  rejected: {', '.join(broken)}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
