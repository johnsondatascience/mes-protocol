#!/usr/bin/env python3
"""End-to-end: bars -> levels -> signals -> trade log -> evaluation.

    # from a CSV of 1-minute bars (SPY proxy or exported futures bars)
    python scripts/run_pipeline.py --csv data/spy_1min.csv --contract SPY

    # from Databento TBBO (needs DATABENTO_API_KEY)
    python scripts/run_pipeline.py --databento ESZ5 \
        --start 2025-10-01 --end 2025-10-31 --contract MES

Writes data/generated_trades.csv and prints the evaluation report.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mesproto import (  # noqa: E402
    CONTRACTS, build_sessions, fills_to_log, load_csv_bars,
    load_databento_tbbo, load_news_dates, sessions_to_frame, validate,
)
from mesproto.config import ALPHA, BURN_IN_TRADES, MIN_EXPECTANCY_R  # noqa: E402
from mesproto.evaluate import compute_r, report  # noqa: E402
from mesproto.signals import run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="1-minute bar CSV: timestamp,open,high,low,close,volume")
    src.add_argument("--databento", metavar="SYMBOL",
                     help="single contract month, e.g. ESZ5 (NOT a back-adjusted series)")
    ap.add_argument("--contract", default="MES", choices=sorted(CONTRACTS))
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2025-10-31")
    ap.add_argument("--tz", default="America/New_York",
                    help="timezone the CSV timestamps are in")
    ap.add_argument("--s3-entry", default="LIMIT", choices=["LIMIT", "STOP"],
                    help="pre-registered S3 entry style; never change mid-sample")
    ap.add_argument("--news-dates", metavar="PATH",
                    help="FOMC/CPI/NFP dates, one YYYY-MM-DD per line. Without it, "
                         "S1 signals before 10:30 cannot verify the stand-down and "
                         "are flagged out of the primary sample")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--burn-in", type=int, default=BURN_IN_TRADES,
                    help="burn-in trades per setup, flagged (not excluded), applied uniformly")
    ap.add_argument("--out", default="data/generated_trades.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    contract = CONTRACTS[args.contract]

    if args.csv:
        source = "SPY" if args.contract == "SPY" else "FUTURES"
        bars = load_csv_bars(args.csv, source=source, tz=args.tz)
    else:
        bars = load_databento_tbbo(symbols=args.databento, start=args.start,
                                   end=args.end)

    sessions = build_sessions(bars, tick=contract.tick)
    print(f"built {len(sessions)} sessions from {len(bars):,} bars "
          f"({bars.index[0]} .. {bars.index[-1]})")
    sf = sessions_to_frame(sessions)
    if not sf.empty:
        print("\nday types:")
        print(sf["day_type"].value_counts().to_string())
        print(f"\ngate hit rates: S1 {sf['s1_gate'].mean():.0%}  "
              f"S2 {sf['s2_gate'].mean():.0%}  S3 {sf['s3_gate'].mean():.0%}  "
              f"S5 {sf['s5_gate'].mean():.0%}")

    news_dates = load_news_dates(args.news_dates) if args.news_dates else None
    if news_dates is None:
        print("\nwarn: no --news-dates calendar. S1 signals before 10:30 ET get "
              "no_news_stand_down=None and are excluded from the primary analysis.")

    fills = run_all(bars, sessions, contract, s3_entry_style=args.s3_entry,
                    news_dates=news_dates)
    log = fills_to_log(fills, contract, contracts=args.contracts)
    if log.empty:
        print("\nno signals generated — check the gates above before assuming a bug")
        return 0

    res = validate(log)
    for w in res.warnings:
        print(f"warn: {w}")
    res.raise_if_bad()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(args.out, index=False)
    print(f"\nwrote {len(log)} trades -> {args.out}")

    scored = compute_r(log)
    scored["session_date"] = scored["session_date"].astype("datetime64[ns]")
    report(scored, min_exp=MIN_EXPECTANCY_R, alpha=ALPHA, n_boot=args.n_boot,
           burn_in=args.burn_in)

    amb = log["notes"].str.contains("ambiguous_bar").sum()
    if amb:
        print(f"NOTE: {amb} of {len(log)} trades ({amb / len(log):.0%}) resolved on "
              f"a bar whose intrabar order OHLC cannot show (stop and target in "
              f"one bar, or a stop entry's fill bar spanning the stop). Those were "
              f"scored as losses by convention; if they are a large share of the "
              f"sample, the bar-level result is unreliable and only replay can "
              f"settle it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
