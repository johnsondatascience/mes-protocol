#!/usr/bin/env python3
"""
mesproto.evaluate — evaluate MES trade logs (hand-logged or generated).

Reads a trade log CSV (see SCHEMA below), and for each setup reports:
  - n, win rate, mean R, median R
  - session-block bootstrap CI on mean R (trades within a session are NOT
    independent — same regime, same trader state — so we resample sessions,
    not trades)
  - the futility-gate verdict (kill / continue / promote)
  - day-type conditional breakdown

Usage:
    python -m mesproto.evaluate trades.csv
    python -m mesproto.evaluate trades.csv --min-expectancy 0.15 --alpha 0.01

SCHEMA (CSV columns, all required unless noted):
    trade_id        int
    session_date    YYYY-MM-DD          (replay session date, NOT calendar date)
    setup           str                 one of: IB_BREAK, LEVEL2LEVEL, VWAP_CONT,
                                        ABSORPTION, ON_INVENTORY
    direction       LONG|SHORT
    entry_time      HH:MM:SS ET
    entry_px        float
    stop_px         float               initial stop, as bracketed at entry
    exit_px         float
    exit_reason     TARGET|STOP|TIME|MANUAL|BREAKEVEN
    contracts       int
    day_type        TREND_UP|TREND_DOWN|BALANCE|DOUBLE_DIST|UNCLASSIFIED
    ib_range_pts    float               initial balance range, 09:30-10:00 ET
    checklist_ok    0|1                 all pre-registered conditions met?
    grade           A|B|C               your execution grade (not setup quality)
    notes           str, optional

R is computed from the log, not entered by hand:
    risk_pts = |entry_px - stop_px|
    R = (exit_px - entry_px) * sign / risk_pts   minus modeled costs
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ALPHA, ASSUMED_SIGMA_R, MES, MIN_EXPECTANCY_R

# --- cost model: single source of truth is config.Contract --------------------
POINT_VALUE = MES.point_value
TICK = MES.tick
COMMISSION_RT = MES.commission_rt
SLIPPAGE_TICKS_ENTRY = MES.slippage_ticks_entry
SLIPPAGE_TICKS_STOP = MES.slippage_ticks_stop

REQUIRED = [
    "trade_id", "session_date", "setup", "direction", "entry_px", "stop_px",
    "exit_px", "exit_reason", "contracts", "day_type", "checklist_ok",
]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"missing required columns: {missing}")
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


def compute_r(df):
    """R-multiple net of modeled commission and stop slippage."""
    sign = np.where(df["direction"].str.upper() == "LONG", 1.0, -1.0)
    risk_pts = (df["entry_px"] - df["stop_px"]).abs()
    if (risk_pts <= 0).any():
        sys.exit("found trades with zero/negative risk distance — fix the log")

    gross_pts = (df["exit_px"] - df["entry_px"]) * sign

    # cost model, in index points per contract
    slip = np.where(df["exit_reason"].str.upper() == "STOP",
                    SLIPPAGE_TICKS_STOP * TICK, 0.0)
    slip = slip + SLIPPAGE_TICKS_ENTRY * TICK
    commission_pts = COMMISSION_RT / POINT_VALUE

    net_pts = gross_pts - slip - commission_pts
    df = df.copy()
    df["risk_pts"] = risk_pts
    df["risk_usd"] = risk_pts * POINT_VALUE * df["contracts"]
    df["gross_R"] = gross_pts / risk_pts
    df["R"] = net_pts / risk_pts
    df["pnl_usd"] = net_pts * POINT_VALUE * df["contracts"]
    return df


def block_bootstrap_mean(df, n_boot=10000, seed=0):
    """Resample whole sessions with replacement; returns bootstrap dist of mean R."""
    rng = np.random.default_rng(seed)
    groups = [g["R"].values for _, g in df.groupby("session_date")]
    k = len(groups)
    if k < 2:
        return np.array([df["R"].mean()])
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        out[i] = np.concatenate([groups[j] for j in idx]).mean()
    return out


def futility_verdict(n, mean_r, sigma, min_exp, conf=0.80):
    """Pre-registered gate: kill if the upper conf-bound on mu is below min_exp."""
    if n < 40:
        return "COLLECT (n<40, no gate yet)"
    se = sigma / np.sqrt(n)
    upper = mean_r + norm.ppf(conf) * se
    if upper < min_exp:
        return f"KILL (upper {conf:.0%} bound {upper:+.3f}R < {min_exp:+.2f}R)"
    return f"CONTINUE (upper {conf:.0%} bound {upper:+.3f}R)"


def confirm_n(mu, sigma, alpha, power=0.80):
    return (norm.ppf(1 - alpha) + norm.ppf(power)) ** 2 * sigma ** 2 / mu ** 2


def report(df, min_exp, alpha, n_boot):
    print(f"\n{'='*78}")
    print(f"MES REPLAY EVALUATION — {len(df)} trades, "
          f"{df['session_date'].nunique()} sessions, "
          f"{df['session_date'].min():%Y-%m-%d} to {df['session_date'].max():%Y-%m-%d}")
    print(f"minimum interesting expectancy: {min_exp:+.2f}R    alpha: {alpha}")
    print("=" * 78)

    off_plan = (df["checklist_ok"] == 0).sum()
    if off_plan:
        print(f"\n!! {off_plan} trades logged with checklist_ok=0. These are excluded "
              f"from the\n   primary analysis — they measure your discipline, not the setup.")
    primary = df[df["checklist_ok"] == 1]

    for setup, g in primary.groupby("setup"):
        n = len(g)
        mean_r = g["R"].mean()
        sigma = g["R"].std(ddof=1) if n > 1 else np.nan
        wins = (g["R"] > 0).sum()

        boot = block_bootstrap_mean(g, n_boot=n_boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_le_0 = (boot <= 0).mean()

        print(f"\n--- {setup} " + "-" * (72 - len(setup)))
        print(f"  n={n}  sessions={g['session_date'].nunique()}  "
              f"win rate={wins/n:.1%}  mean={mean_r:+.3f}R  median={g['R'].median():+.3f}R")
        print(f"  sd={sigma:.3f}R  worst={g['R'].min():+.2f}R  best={g['R'].max():+.2f}R")
        print(f"  net P&L=${g['pnl_usd'].sum():,.0f}  "
              f"avg risk=${g['risk_usd'].mean():,.0f}/trade")
        print(f"  session-block bootstrap 95% CI on mean R: "
              f"[{lo:+.3f}, {hi:+.3f}]   P(mean<=0)={p_le_0:.3f}")
        print(f"  gate: {futility_verdict(n, mean_r, max(sigma, ASSUMED_SIGMA_R * 0.6), min_exp)}")
        if not np.isnan(sigma) and mean_r > 0:
            need = confirm_n(max(mean_r, 0.05), max(sigma, 1.0), alpha)
            print(f"  n to confirm this effect at alpha={alpha}, 80% power: {need:.0f} "
                  f"({need - n:+.0f} more)")

        # day-type conditioning
        if g["day_type"].nunique() > 1:
            print("  by day type:")
            for dt, gg in g.groupby("day_type"):
                if len(gg) >= 5:
                    print(f"    {dt:<14} n={len(gg):>4}  mean={gg['R'].mean():+.3f}R  "
                          f"win={((gg['R']>0).mean()):.0%}")

        # exit reason mix — a fast read on whether targets are reachable
        mix = g["exit_reason"].value_counts(normalize=True)
        print("  exits: " + "  ".join(f"{k}={v:.0%}" for k, v in mix.items()))

    print(f"\n{'='*78}")
    if primary.empty:
        print("PORTFOLIO: no trades passed the checklist. Nothing to evaluate.")
        print("This is the expected result when the data source cannot verify a")
        print("condition — bar-only data leaves delta_confirmed=None, which is")
        print("recorded as a failed checklist rather than silently passed.")
        print("=" * 78 + "\n")
        return

    print("PORTFOLIO (checklist_ok trades, all setups)")
    boot = block_bootstrap_mean(primary, n_boot=n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  n={len(primary)}  mean={primary['R'].mean():+.3f}R  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  net ${primary['pnl_usd'].sum():,.0f}")
    daily = primary.groupby("session_date")["pnl_usd"].sum()
    print(f"  daily P&L: mean ${daily.mean():,.0f}  sd ${daily.std():,.0f}  "
          f"worst ${daily.min():,.0f}  win days {((daily>0).mean()):.0%}")
    eq = daily.cumsum()
    print(f"  max drawdown on daily equity: ${(eq.cummax()-eq).max():,.0f}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--min-expectancy", type=float, default=MIN_EXPECTANCY_R,
                    help=f"minimum expectancy in R worth trading (default {MIN_EXPECTANCY_R})")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"significance level (default {ALPHA}, Bonferroni over 5 setups)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    df = compute_r(load(args.csv))
    report(df, args.min_expectancy, args.alpha, args.n_boot)


if __name__ == "__main__":
    main()
