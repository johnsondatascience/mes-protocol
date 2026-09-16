#!/usr/bin/env python3
"""Check an Optimus Flow (Quantower) CSV export against our Databento bars.

    python scripts/compare_optimus_export.py --export optimus_2026-08-12.csv \
        --dbn data/tbbo_ESU6_2026-07-30T2200_2026-08-31T2100.dbn.zst

Reads the saved DBN file, so it costs nothing. Exports the platform's display
time, so pass --tz if that is not America/New_York. Compares volume minute by
minute and, when the export carries delta (or ask/bid volume), reports whether
its delta runs the same way as ours or the opposite way — the hand-logged
setups are read in Optimus Flow, so the two must agree on the delta sign.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mesproto.levels import load_csv_bars, load_databento_tbbo  # noqa: E402
from mesproto.optimus import compare_delta, describe, load_optimus_export  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, metavar="PATH",
                    help="the CSV written by Optimus Flow's History Exporter")
    ap.add_argument("--tz", default="America/New_York",
                    help="timezone the export's timestamps are in (default ET)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dbn", metavar="PATH", help="our saved Databento download")
    src.add_argument("--csv", metavar="PATH", help="our bars as a 1-minute CSV instead")
    ap.add_argument("--symbol", default="ESU6", help="contract in the DBN file (default ESU6)")
    args = ap.parse_args()

    theirs = load_optimus_export(args.export, tz=args.tz)
    if "delta" not in theirs:
        print("note: no delta or ask/bid-volume column in the export — only volume "
              "can be compared. Re-export with volume analysis if your build offers it.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.dbn:
            if not Path(args.dbn).is_file():
                sys.exit(f"{args.dbn} does not exist — this script never buys data")
            bars = load_databento_tbbo(symbols=args.symbol, start="", end="", path=args.dbn)
        else:
            bars = load_csv_bars(args.csv, source="FUTURES")

    ours = pd.DataFrame({
        "volume": bars["volume"],
        "delta": bars["buy_volume"] - bars["sell_volume"],
    }, index=bars.index).loc[theirs.index.min():theirs.index.max()]

    result = compare_delta(theirs, ours)
    print(describe(theirs, ours, result))
    return 0 if result["verdict"] in ("SAME", "NO_DELTA") else 1


if __name__ == "__main__":
    raise SystemExit(main())
