"""Network access to the public reference sources: FRED and federalreserve.gov.

Only the scripts in scripts/ call the fetch_* functions. Parsers are pure and
tested offline, and nothing here runs on import. Errors never echo a request
URL, because FRED's carries the API key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Mapping, Optional

from .config import (
    FRED_MAX_LIMIT, FRED_OBSERVATIONS_MAX_LIMIT, FRED_RELEASE_DATES_URL,
    FRED_SERIES_OBSERVATIONS_URL, HTTP_TIMEOUT_S, HTTP_USER_AGENT,
)


def http_get(url: str, params: Optional[dict] = None) -> bytes:
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(full, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return response.read()


def read_api_key(env_file: Path, name: str = "FRED_API_KEY") -> str:
    """The key from the environment, else from a NAME= line in `env_file`."""
    key = os.environ.get(name, "").strip()
    if key or not env_file.is_file():
        return key
    for line in env_file.read_text(encoding="utf-8").splitlines():
        field, sep, value = line.partition("=")
        if sep and field.strip().removeprefix("export ").strip() == name:
            return value.strip().strip("'\"")
    return ""


def _fred_json(url: str, params: dict, what: str) -> Mapping:
    try:
        body = http_get(url, params)
    except urllib.error.HTTPError as e:
        body = e.read()                        # FRED explains errors in the body
    except urllib.error.URLError as e:
        raise RuntimeError(f"FRED {what}: {e.reason}") from None
    try:
        return json.loads(body)
    except ValueError:
        raise RuntimeError(f"FRED {what}: response was not JSON") from None


def parse_fred_release_dates(payload: Mapping) -> list[date]:
    """Dates from a fred/release/dates JSON response."""
    if "error_message" in payload or "release_dates" not in payload:
        raise ValueError(f"FRED error: {payload.get('error_message', 'no release_dates')}")
    listed = payload["release_dates"]
    if int(payload.get("count", len(listed))) > len(listed):
        raise ValueError(f"FRED response is truncated ({len(listed)} of "
                         f"{payload['count']} dates)")
    return sorted(date.fromisoformat(r["date"]) for r in listed)


def parse_fred_observations(payload: Mapping) -> dict[date, float]:
    """{date: value} from a fred/series/observations JSON response. FRED marks
    days without a value (market holidays) with '.'; they are left out, not
    filled."""
    if "error_message" in payload or "observations" not in payload:
        raise ValueError(f"FRED error: {payload.get('error_message', 'no observations')}")
    listed = payload["observations"]
    if int(payload.get("count", len(listed))) > len(listed):
        raise ValueError(f"FRED response is truncated ({len(listed)} of "
                         f"{payload['count']} observations)")
    return {date.fromisoformat(o["date"]): float(o["value"])
            for o in listed if o["value"] != "."}


def fetch_fred_observations(series_id: str, api_key: str) -> dict[date, float]:
    """Every observation FRED holds for a series."""
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "limit": FRED_OBSERVATIONS_MAX_LIMIT, "sort_order": "asc"}
    return parse_fred_observations(
        _fred_json(FRED_SERIES_OBSERVATIONS_URL, params, f"series {series_id}"))


def fetch_fred_release_dates(release_id: int, api_key: str) -> list[date]:
    """Every date a FRED release came out."""
    params = {"release_id": release_id, "api_key": api_key, "file_type": "json",
              "limit": FRED_MAX_LIMIT, "sort_order": "asc"}
    return parse_fred_release_dates(
        _fred_json(FRED_RELEASE_DATES_URL, params, f"release {release_id}"))
