#!/usr/bin/env python3
"""Tests for mesproto.sources — parsers for FRED responses, pinned offline."""
from datetime import date

from mesproto.sources import parse_fred_observations, parse_fred_release_dates


def test_fred_observations_skip_missing_values():
    """FRED marks market holidays with '.'; those days have no close at all."""
    payload = {"count": 3, "observations": [
        {"date": "2026-01-16", "value": "6940.01"},
        {"date": "2026-01-19", "value": "."},
        {"date": "2026-01-20", "value": "6796.86"}]}
    assert parse_fred_observations(payload) == \
        {date(2026, 1, 16): 6940.01, date(2026, 1, 20): 6796.86}
    for bad in ({"error_code": 400, "error_message": "Bad Request."},
                {**payload, "count": 5000}):
        try:
            parse_fred_observations(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must raise")
    print("  '.' skipped; API error and truncated response raise")


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
