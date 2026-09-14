#!/usr/bin/env python3
"""
mesproto.evaluate — evaluate MES trade logs (hand-logged or generated).

Reads a trade log CSV (see SCHEMA below), validates it, and for each setup
reports:
  - n, win rate, mean R, median R, net of the contract's modeled costs
  - session-block bootstrap CI on mean R (trades within a session are NOT
    independent — same regime, same trader state — so we resample sessions,
    not trades)
  - the futility-gate verdict (collect / continue / kill / passed), decided
    only at the pre-registered checkpoints
  - day-type conditional breakdown, exit mix, ambiguous-bar share

The primary sample excludes, per setup, the pre-registered burn-in (the
chronologically first BURN_IN_TRADES trades) and then every checklist_ok=0
trade.

Usage:
    python -m mesproto.evaluate trades.csv
    python -m mesproto.evaluate trades.csv --contract MES --min-expectancy 0.15 --alpha 0.01

SCHEMA (CSV columns, all required unless noted):
    trade_id        int
    session_date    YYYY-MM-DD          (replay session date, NOT calendar date)
    setup           str                 one of: IB_BREAK, LEVEL2LEVEL, VWAP_CONT,
                                        ABSORPTION, ON_INVENTORY, IB_FAIL
    direction       LONG|SHORT
    entry_time      HH:MM:SS ET         (orders trades within a session)
    entry_px        float
    stop_px         float               initial stop, as bracketed at entry
    exit_px         float               the MECHANICAL exit (see protocol §03)
    exit_reason     TARGET|STOP|TIME|MANUAL|BREAKEVEN
    contracts       int
    day_type        TREND_UP|TREND_DOWN|BALANCE|DOUBLE_DIST|UNCLASSIFIED
    ib_range_pts    float               initial balance range, 09:30-10:00 ET
    on_range_pos    float 0-1           S5 covariate
    checklist_ok    0|1                 all pre-registered conditions met?
    grade           A|B|C               your execution grade (not setup quality)
    source          str, optional       e.g. REPLAY, GENERATED
    notes           str, optional

R is computed from the log, not entered by hand:
    risk_pts = |entry_px - stop_px|
    R = (exit_px - entry_px) * sign / risk_pts   minus modeled costs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    ALPHA, BOOTSTRAP_CI, BURN_IN_TRADES, CONFIRM_N_MIN_EFFECT_R,
    CONFIRM_N_MIN_SIGMA_R, CONFIRM_POWER, CONTRACTS, DAY_TYPE_MIN_TRADES,
    FUTILITY_GATES, GATE_CONFIDENCE,
    GATE_SIGMA_FLOOR_R, MES, MIN_EXPECTANCY_R, Contract,
)
from .schema import validate

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


def compute_r(df, contract: Contract = MES):
    """R-multiple net of the contract's modeled commission and slippage.

    `contract` is the execution vehicle, not the chart the setup was read
    on: trades read on ES and executed in MES are scored with MES costs.
    """
    sign = np.where(df["direction"].str.upper() == "LONG", 1.0, -1.0)
    risk_pts = (df["entry_px"] - df["stop_px"]).abs()
    if (risk_pts <= 0).any():
        raise ValueError("found trades with zero/negative risk distance — fix the log")

    gross_pts = (df["exit_px"] - df["entry_px"]) * sign

    # cost model, in index points per contract
    slip = np.where(df["exit_reason"].str.upper() == "STOP",
                    contract.slippage_ticks_stop * contract.tick, 0.0)
    slip = slip + contract.slippage_ticks_entry * contract.tick
    commission_pts = contract.commission_rt / contract.point_value

    net_pts = gross_pts - slip - commission_pts
    df = df.copy()
    df["risk_pts"] = risk_pts
    df["risk_usd"] = risk_pts * contract.point_value * df["contracts"]
    df["gross_R"] = gross_pts / risk_pts
    df["R"] = net_pts / risk_pts
    df["pnl_usd"] = net_pts * contract.point_value * df["contracts"]
    return df


def chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Trades in the order they happened — never the order of rows in a file."""
    keys = ["session_date"] + (["entry_time"] if "entry_time" in df.columns else []) \
        + ["trade_id"]
    return df.sort_values(keys, kind="mergesort")


def primary_sample(df: pd.DataFrame, burn_in: int = BURN_IN_TRADES
                   ) -> tuple[pd.DataFrame, dict]:
    """The primary-analysis sample, in chronological order, plus exclusion counts.

    Burn-in is the first `burn_in` trades of each setup by time, counted
    whatever their checklist: reading skill drifts with every trade taken, so
    an off-checklist trade uses up burn-in like any other. After burn-in,
    checklist_ok=0 trades are excluded — they measure discipline (or, for
    generated trades, a condition the data could not verify), not the setup.
    """
    ordered = chronological(df)
    rank = ordered.groupby("setup").cumcount()
    is_burn = (rank < burn_in).to_numpy()
    ok = (ordered["checklist_ok"] == 1).to_numpy()
    info = {"burn_in": int(is_burn.sum()), "off_checklist": int((~is_burn & ~ok).sum())}
    return ordered[~is_burn & ok], info


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


def kill_threshold(n: int, sigma: float, min_exp: float,
                   conf: float = GATE_CONFIDENCE) -> float:
    """Mean R below which the gate kills at sample size n (protocol §06 table)."""
    return min_exp - norm.ppf(conf) * sigma / np.sqrt(n)


@dataclass(frozen=True)
class GateResult:
    verdict: str                      # COLLECT | CONTINUE | KILL | PASSED
    n: int
    min_exp: float
    checkpoint: Optional[int] = None  # last checkpoint evaluated
    upper: Optional[float] = None     # its upper confidence bound on mean R
    next_gate: Optional[int] = None

    def __str__(self) -> str:
        if self.verdict == "COLLECT":
            return f"COLLECT (n={self.n}; first gate at n={self.next_gate})"
        bound = (f"upper {GATE_CONFIDENCE:.0%} bound at n={self.checkpoint}: "
                 f"{self.upper:+.3f}R")
        if self.verdict == "KILL":
            return f"KILL at n={self.checkpoint} ({bound} < {self.min_exp:+.2f}R)"
        if self.verdict == "CONTINUE":
            return f"CONTINUE ({bound}; next gate at n={self.next_gate})"
        return f"PASSED all gates ({bound}) — collect the confirmation sample (§01)"


def futility_verdict(r_chronological: Sequence[float], min_exp: float,
                     checkpoints: Sequence[int] = FUTILITY_GATES,
                     conf: float = GATE_CONFIDENCE,
                     sigma_floor: float = GATE_SIGMA_FLOOR_R) -> GateResult:
    """Pre-registered one-sided futility gate, applied only at fixed checkpoints.

    Checkpoint c is decided on the first c primary trades, and the decision
    never changes afterwards. Re-deciding on the whole sample every time the
    report runs would turn each run into another uncorrected look — the
    informal sequential test the protocol warns about — and a setup that was
    dead at n=60 could be revived by whatever happened next.

    Kill when mean + z(conf) * sigma / sqrt(c) < min_exp, with sigma the
    observed standard deviation of those c trades, floored so a quiet start
    cannot shrink the bound.
    """
    r = np.asarray(r_chronological, dtype=float)
    n = len(r)
    last: Optional[tuple[int, float]] = None
    for c in checkpoints:
        if n < c:
            if last is None:
                return GateResult("COLLECT", n, min_exp, next_gate=c)
            return GateResult("CONTINUE", n, min_exp, last[0], last[1], next_gate=c)
        head = r[:c]
        sigma = max(float(head.std(ddof=1)), sigma_floor)
        upper = float(head.mean() + norm.ppf(conf) * sigma / np.sqrt(c))
        if upper < min_exp:
            return GateResult("KILL", n, min_exp, c, upper)
        last = (c, upper)
    if last is None:
        return GateResult("COLLECT", n, min_exp)
    return GateResult("PASSED", n, min_exp, last[0], last[1])


def bootstrap_ci(boot: np.ndarray, level: float = BOOTSTRAP_CI) -> tuple[float, float]:
    """Two-sided percentile interval of a bootstrap distribution."""
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(boot, [tail, 100.0 - tail])
    return float(lo), float(hi)


def confirm_n(mu, sigma, alpha, power=CONFIRM_POWER):
    return (norm.ppf(1 - alpha) + norm.ppf(power)) ** 2 * sigma ** 2 / mu ** 2


def report(df, min_exp, alpha, n_boot, burn_in: int = BURN_IN_TRADES):
    print(f"\n{'='*78}")
    print(f"MES REPLAY EVALUATION — {len(df)} trades, "
          f"{df['session_date'].nunique()} sessions, "
          f"{df['session_date'].min():%Y-%m-%d} to {df['session_date'].max():%Y-%m-%d}")
    print(f"minimum interesting expectancy: {min_exp:+.2f}R    alpha: {alpha}")
    print("=" * 78)

    if "source" in df.columns and df["source"].nunique() > 1:
        print(f"\n!! this log mixes sources {sorted(df['source'].dropna().unique())}. "
              f"Generated and hand-logged\n   trades are different experiments — "
              f"evaluate them separately.")

    primary, info = primary_sample(df, burn_in)
    if info["burn_in"]:
        print(f"\n!! {info['burn_in']} trades are pre-registered burn-in (the first "
              f"{burn_in} of each setup)\n   and are excluded from the primary analysis.")
    if info["off_checklist"]:
        print(f"\n!! {info['off_checklist']} post-burn-in trades logged with "
              f"checklist_ok=0. These are excluded from the\n   primary analysis — "
              f"they measure discipline (or, for generated trades, a\n   condition the "
              f"data could not verify), not the setup.")

    for setup, g in primary.groupby("setup", sort=True):
        n = len(g)
        mean_r = g["R"].mean()
        sigma = g["R"].std(ddof=1) if n > 1 else np.nan
        wins = (g["R"] > 0).sum()

        boot = block_bootstrap_mean(g, n_boot=n_boot)
        lo, hi = bootstrap_ci(boot)
        p_le_0 = (boot <= 0).mean()

        print(f"\n--- {setup} " + "-" * (72 - len(setup)))
        print(f"  n={n}  sessions={g['session_date'].nunique()}  "
              f"win rate={wins/n:.1%}  mean={mean_r:+.3f}R  median={g['R'].median():+.3f}R")
        print(f"  sd={sigma:.3f}R  worst={g['R'].min():+.2f}R  best={g['R'].max():+.2f}R")
        print(f"  net P&L=${g['pnl_usd'].sum():,.0f}  "
              f"avg risk=${g['risk_usd'].mean():,.0f}/trade")
        print(f"  session-block bootstrap {BOOTSTRAP_CI:.0%} CI on mean R: "
              f"[{lo:+.3f}, {hi:+.3f}]   P(mean<=0)={p_le_0:.3f}")
        print(f"  gate: {futility_verdict(g['R'].to_numpy(), min_exp)}")
        if not np.isnan(sigma) and mean_r > 0:
            need = confirm_n(max(mean_r, CONFIRM_N_MIN_EFFECT_R),
                             max(sigma, CONFIRM_N_MIN_SIGMA_R), alpha)
            print(f"  n to confirm this effect at alpha={alpha}, "
                  f"{CONFIRM_POWER:.0%} power: {need:.0f} ({need - n:+.0f} more)")

        # day-type conditioning
        if g["day_type"].nunique() > 1:
            print("  by day type:")
            for dt, gg in g.groupby("day_type"):
                if len(gg) >= DAY_TYPE_MIN_TRADES:
                    print(f"    {dt:<14} n={len(gg):>4}  mean={gg['R'].mean():+.3f}R  "
                          f"win={((gg['R']>0).mean()):.0%}")

        # exit reason mix — a fast read on whether targets are reachable
        mix = g["exit_reason"].value_counts(normalize=True)
        print("  exits: " + "  ".join(f"{k}={v:.0%}" for k, v in mix.items()))

        if "notes" in g.columns:
            amb = g["notes"].fillna("").astype(str).str.contains("ambiguous_bar").mean()
            if amb > 0:
                print(f"  ambiguous bars: {amb:.0%} of trades were scored STOP by "
                      f"convention — if large, only replay can settle them")

    print(f"\n{'='*78}")
    if primary.empty:
        print("PORTFOLIO: no trades in the primary sample. Nothing to evaluate.")
        print("Expected while a setup is still in burn-in, or when the data source")
        print("cannot verify a condition — bar-only data leaves delta_confirmed=None,")
        print("which is recorded as a failed checklist rather than silently passed.")
        print("=" * 78 + "\n")
        return

    print("PORTFOLIO (primary sample, all setups)")
    boot = block_bootstrap_mean(primary, n_boot=n_boot)
    lo, hi = bootstrap_ci(boot)
    print(f"  n={len(primary)}  mean={primary['R'].mean():+.3f}R  "
          f"{BOOTSTRAP_CI:.0%} CI [{lo:+.3f}, {hi:+.3f}]  net ${primary['pnl_usd'].sum():,.0f}")
    daily = primary.groupby("session_date")["pnl_usd"].sum()
    print(f"  daily P&L: mean ${daily.mean():,.0f}  sd ${daily.std():,.0f}  "
          f"worst ${daily.min():,.0f}  win days {((daily>0).mean()):.0%}")
    eq = daily.cumsum()
    print(f"  max drawdown on daily equity: ${(eq.cummax()-eq).max():,.0f}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--contract", default=MES.symbol, choices=sorted(CONTRACTS),
                    help="execution vehicle whose cost model scores the log (default MES)")
    ap.add_argument("--min-expectancy", type=float, default=MIN_EXPECTANCY_R,
                    help=f"minimum expectancy in R worth trading (default {MIN_EXPECTANCY_R})")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help=f"significance level (default {ALPHA}, Bonferroni over 5 setups)")
    ap.add_argument("--burn-in", type=int, default=BURN_IN_TRADES,
                    help=f"pre-registered burn-in trades per setup (default {BURN_IN_TRADES})")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    df = load(args.csv)
    res = validate(df)
    for w in res.warnings:
        print(f"warn: {w}")
    if not res.ok:
        sys.exit("trade log failed validation:\n  - " + "\n  - ".join(res.errors))
    df = compute_r(df, contract=CONTRACTS[args.contract])
    report(df, args.min_expectancy, args.alpha, args.n_boot, burn_in=args.burn_in)


if __name__ == "__main__":
    main()
