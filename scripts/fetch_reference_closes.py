#!/usr/bin/env python3
"""Build data/sp500_close.csv from FRED for SPY point scaling.

    python scripts/fetch_reference_closes.py

The setups' thresholds are ES points. On a SPY session each one is multiplied
by SPY's prior-day close over the S&P 500's close that same day; this file is
the S&P 500 side. Needs FRED_API_KEY in the environment or on a FRED_API_KEY=
line in .env (never printed). FRED keeps 10 years of the index, so older SPY
sessions get no scale and generate nothing. Re-run before each SPY block.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mesproto.config import SPY_REFERENCE_SERIES  # noqa: E402
from mesproto.levels import load_reference_closes  # noqa: E402
from mesproto.sources import fetch_fred_observations, read_api_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/sp500_close.csv")
    ap.add_argument("--env-file", default=".env",
                    help="read FRED_API_KEY from here if it is not in the environment")
    args = ap.parse_args()

    key = read_api_key(Path(args.env_file))
    if not key:
        sys.exit("no FRED API key: set FRED_API_KEY, or add FRED_API_KEY=... to "
                 f"{args.env_file}. Free at https://fredaccount.stlouisfed.org/apikeys")

    closes = fetch_fred_observations(SPY_REFERENCE_SERIES, key)
    if not closes:
        sys.exit(f"FRED returned no {SPY_REFERENCE_SERIES} closes; nothing written")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# S&P 500 index daily close (FRED {SPY_REFERENCE_SERIES}), the ES stand-in "
             f"for SPY point scaling; fetched {date.today().isoformat()}",
             "date,close"] + [f"{d.isoformat()},{c}" for d, c in sorted(closes.items())]
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    loaded = load_reference_closes(args.out)       # round-trip: the file must load

    first, last = min(loaded), max(loaded)
    print(f"wrote {args.out}: {len(loaded)} closes, {first} .. {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
