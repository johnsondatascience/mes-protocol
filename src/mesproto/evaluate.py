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

Setups in EXPLORATORY_SETUPS (IB_FAIL) are reported but never judged: no gate
verdict, no confirmation sample size, and they are left out of the portfolio.
They sit outside the Bonferroni family of N_SETUPS_TESTED.

The primary sample excludes every checklist_ok=0 trade. Burn-in (per setup,
the chronologically first BURN_IN_TRADES trades) is included and flagged —
protocol amendment 2026-09-15.

Usage:
    python -m mesproto.evaluate trades.csv
    python -m mesproto.evaluate trades.csv --min-expectancy 0.15 --alpha 0.01

SCHEMA (CSV columns, all required unless noted):
    trade_id          int
    session_date      YYYY-MM-DD        (replay session date, NOT calendar date)
    setup             str               one of: IB_BREAK, LEVEL2LEVEL, VWAP_CONT,
                                        ABSORPTION, ON_INVENTORY, IB_FAIL
    direction         LONG|SHORT
    contract          MES|ES|SPY        execution vehicle; its costs score the row
    entry_time        HH:MM:SS ET       (orders trades within a session)
    entry_px          float
    stop_px           float             initial stop, as bracketed at entry
    exit_px           float             the exit you actually took (managed)
    exit_reason       TARGET|STOP|TIME|MANUAL|BREAKEVEN
    mech_exit_px      float             the fixed 1:2 bracket's exit (protocol §03):
                                        stop_px, entry +/- 2R, or the time-exit price
    mech_exit_reason  TARGET|STOP|TIME
    contracts         int
    day_type          TREND_UP|TREND_DOWN|BALANCE|DOUBLE_DIST|UNCLASSIFIED
    ib_range_pts      float             initial balance range, 09:30-10:00 ET
    on_range_pos      float 0-1         S5 covariate
    gap_pct           float, optional   open vs prior close; gap-open sessions reported
    checklist_ok      0|1               all pre-registered conditions met?
    failed_checks     str, optional     conditions that were not True, ';'-separated
    grade             A|B|C             your execution grade (not setup quality)
    source            str, optional     e.g. REPLAY, GENERATED
    notes             str, optional

R is computed from the log, never entered by hand:
    risk_pts  = |entry_px - stop_px|
    R         = (mech_exit_px - entry_px) * sign / risk_pts   minus modeled costs
              — the protocol's mech_exit_R; every statistic runs on it
    managed_R = the same for exit_px, reported beside it
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
    EXPLORATORY_SETUPS, FUTILITY_GATES, GAP_OPEN_PCT, GATE_CONFIDENCE,
    GATE_SIGMA_FLOOR_R, MIN_EXPECTANCY_R, PRICE_EPS, STOP_ORDER_EXITS,
    VALIDATION_PERIODS,
)
from .schema import validate

REQUIRED = [
    "trade_id", "session_date", "setup", "direction", "contract", "entry_px",
    "stop_px", "exit_px", "exit_reason", "mech_exit_px", "mech_exit_reason",
    "contracts", "day_type", "checklist_ok",
]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"missing required columns: {missing}")
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


def compute_r(df: pd.DataFrame) -> pd.DataFrame:
    """R-multiples net of each row's contract costs (commission and slippage).

    The `contract` column is the execution vehicle, not the chart the setup
    was read on: trades read on ES and executed in MES are scored with MES
    costs. Adds:
      R, gross_R, pnl_usd   the mechanical exit (protocol's mech_exit_R) —
                            what every statistic runs on
      managed_R             the exit actually taken, which measures discretion
    """
    unknown = set(df["contract"].astype(str)) - set(CONTRACTS)
    if unknown:
        raise ValueError(f"unknown contract values {sorted(unknown)} — fix the log")
    spec = df["contract"].astype(str).map(CONTRACTS)
    tick = spec.map(lambda c: c.tick)
    point_value = spec.map(lambda c: c.point_value)
    commission_pts = spec.map(lambda c: c.commission_rt) / point_value
    entry_slip = spec.map(lambda c: c.slippage_ticks_entry) * tick
    stop_slip = spec.map(lambda c: c.slippage_ticks_stop) * tick

    sign = np.where(df["direction"].str.upper() == "LONG", 1.0, -1.0)
    risk_pts = (df["entry_px"] - df["stop_px"]).abs()
    if (risk_pts <= 0).any():
        raise ValueError("found trades with zero/negative risk distance — fix the log")

    def net_points(exit_px: pd.Series, reason: pd.Series) -> pd.Series:
        slip = np.where(reason.astype(str).str.upper().isin(STOP_ORDER_EXITS), stop_slip, 0.0)
        return (exit_px - df["entry_px"]) * sign - slip - entry_slip - commission_pts

    mech_net = net_points(df["mech_exit_px"], df["mech_exit_reason"])
    out = df.copy()
    out["risk_pts"] = risk_pts
    out["risk_usd"] = risk_pts * point_value * df["contracts"]
    out["gross_R"] = (df["mech_exit_px"] - df["entry_px"]) * sign / risk_pts
    out["R"] = mech_net / risk_pts
    out["pnl_usd"] = mech_net * point_value * df["contracts"]
    out["managed_R"] = net_points(df["exit_px"], df["exit_reason"]) / risk_pts
    return out


def chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Trades in the order they happened — never the order of rows in a file."""
    keys = ["session_date"] + (["entry_time"] if "entry_time" in df.columns else []) \
        + ["trade_id"]
    return df.sort_values(keys, kind="mergesort")


def primary_sample(df: pd.DataFrame, burn_in: int = BURN_IN_TRADES
                   ) -> tuple[pd.DataFrame, dict]:
    """The primary-analysis sample, in chronological order, plus counts.

    Burn-in is the first `burn_in` trades of each setup by time, counted
    whatever their checklist: reading skill drifts with every trade taken, so
    an off-checklist trade uses up burn-in like any other. Burn-in trades stay
    in the sample with burn_in=True (protocol amendment 2026-09-15), so the
    drift the protocol worries about is shown beside the result rather than
    removed from it. checklist_ok=0 trades are excluded — they measure
    discipline (or, for generated trades, a condition the data could not
    verify), not the setup.
    """
    ordered = chronological(df).copy()
    ordered["burn_in"] = (ordered.groupby("setup").cumcount() < burn_in).to_numpy()
    days = pd.to_datetime(ordered["session_date"]).dt.date
    validation = days.map(lambda d: any(a <= d <= b for a, b, _ in VALIDATION_PERIODS)).to_numpy()
    ok = (ordered["checklist_ok"] == 1).to_numpy() & ~validation
    info = {"burn_in": int((ordered["burn_in"].to_numpy() & ok).sum()),
            "off_checklist": int(((ordered["checklist_ok"] != 1).to_numpy() & ~validation).sum()),
            "validation": int(validation.sum())}
    return ordered[ok], info


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
    if info["validation"]:
        print(f"\n!! {info['validation']} trades fall in validation-only periods and are "
              f"excluded from\n   everything below:")
        for start, end, why in VALIDATION_PERIODS:
            print(f"     {start} .. {end}: {why}")
    if info["burn_in"]:
        print(f"\n!! {info['burn_in']} trades are burn-in (the first {burn_in} of each "
              f"setup). They are INCLUDED in\n   the primary analysis and flagged "
              f"burn_in=True; each setup also shows its result\n   without them, "
              f"for comparison only.")
    if info["off_checklist"]:
        print(f"\n!! {info['off_checklist']} trades logged with "
              f"checklist_ok=0. These are excluded from the\n   primary analysis — "
              f"they measure discipline (or, for generated trades, a\n   condition the "
              f"data could not verify), not the setup.")
        if "failed_checks" in df.columns:
            reasons = (df.loc[df["checklist_ok"] == 0, "failed_checks"].fillna("")
                       .astype(str).str.split(";").explode())
            reasons = reasons[reasons != ""].value_counts()
            if len(reasons):
                print("   excluded by condition (a trade can fail more than one):")
                for name, count in reasons.items():
                    print(f"     {name}: {count}")

    for setup, g in primary.groupby("setup", sort=True):
        n = len(g)
        mean_r = g["R"].mean()
        sigma = g["R"].std(ddof=1) if n > 1 else np.nan
        wins = (g["R"] > 0).sum()

        boot = block_bootstrap_mean(g, n_boot=n_boot)
        lo, hi = bootstrap_ci(boot)
        p_le_0 = (boot <= 0).mean()

        exploratory = setup in EXPLORATORY_SETUPS
        title = f"{setup} (EXPLORATORY)" if exploratory else setup
        print(f"\n--- {title} " + "-" * (72 - len(title)))
        print(f"  n={n}  sessions={g['session_date'].nunique()}  "
              f"win rate={wins/n:.1%}  mean={mean_r:+.3f}R  median={g['R'].median():+.3f}R")
        if g["burn_in"].any():
            later = g.loc[~g["burn_in"], "R"]
            shown = f"mean={later.mean():+.3f}R" if len(later) else "no trades yet"
            print(f"  without burn-in: n={len(later)} {shown}  "
                  f"(comparison only; the gate uses every trade)")
        if "gap_pct" in g.columns:
            gap = g[pd.to_numeric(g["gap_pct"], errors="coerce").abs() > GAP_OPEN_PCT]
            if len(gap):
                print(f"  gap-open sessions (|gap| > {GAP_OPEN_PCT:.0%}): n={len(gap)} "
                      f"win={(gap['R'] > 0).mean():.0%} mean={gap['R'].mean():+.3f}R  "
                      f"(included; a different regime)")
        if "managed_R" in g.columns and ((g["managed_R"] - g["R"]).abs() > PRICE_EPS).any():
            print(f"  managed exits (your discretion, not the test): "
                  f"mean={g['managed_R'].mean():+.3f}R vs mechanical {mean_r:+.3f}R")
        print(f"  sd={sigma:.3f}R  worst={g['R'].min():+.2f}R  best={g['R'].max():+.2f}R")
        print(f"  net P&L=${g['pnl_usd'].sum():,.0f}  "
              f"avg risk=${g['risk_usd'].mean():,.0f}/trade")
        print(f"  session-block bootstrap {BOOTSTRAP_CI:.0%} CI on mean R: "
              f"[{lo:+.3f}, {hi:+.3f}]   P(mean<=0)={p_le_0:.3f}")
        if exploratory:
            # described, never judged: a PASS here would read as a path to live
            # capital for a setup outside the tested family
            print("  exploratory: no gate verdict and no confirmation claim — a promising "
                  "result\n  needs its own pre-registered sample before it counts")
        else:
            print(f"  gate: {futility_verdict(g['R'].to_numpy(), min_exp)}")
        if not exploratory and not np.isnan(sigma) and mean_r > 0:
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

        # mechanical exit mix — a fast read on whether targets are reachable
        mix = g["mech_exit_reason"].value_counts(normalize=True)
        print("  exits: " + "  ".join(f"{k}={v:.0%}" for k, v in mix.items()))

        if "notes" in g.columns:
            amb = g["notes"].fillna("").astype(str).str.contains("ambiguous_bar").mean()
            if amb > 0:
                print(f"  ambiguous bars: {amb:.0%} of trades were scored STOP by "
                      f"convention — if large, only replay can settle them")

    print(f"\n{'='*78}")
    left_out = primary[primary["setup"].isin(EXPLORATORY_SETUPS)]
    tested = primary[~primary["setup"].isin(EXPLORATORY_SETUPS)]
    if tested.empty:
        print("PORTFOLIO: no trades from tested setups in the primary sample. Nothing to evaluate.")
        print("Expected when every trade fell in a validation-only period (named above),")
        print("when no trade passed its checklist, or when the data source")
        print("cannot verify a condition — bar-only data leaves delta_confirmed=None,")
        print("which is recorded as a failed checklist rather than silently passed.")
        print("=" * 78 + "\n")
        return

    print("PORTFOLIO (primary sample, tested setups)")
    if not left_out.empty:
        counts = ", ".join(f"{s}: n={len(g)}" for s, g in left_out.groupby("setup"))
        print(f"  excludes exploratory setups ({counts})")
    boot = block_bootstrap_mean(tested, n_boot=n_boot)
    lo, hi = bootstrap_ci(boot)
    print(f"  n={len(tested)}  mean={tested['R'].mean():+.3f}R  "
          f"{BOOTSTRAP_CI:.0%} CI [{lo:+.3f}, {hi:+.3f}]  net ${tested['pnl_usd'].sum():,.0f}")
    daily = tested.groupby("session_date")["pnl_usd"].sum()
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
    ap.add_argument("--burn-in", type=int, default=BURN_IN_TRADES,
                    help=f"burn-in trades per setup, flagged not excluded (default {BURN_IN_TRADES})")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    df = load(args.csv)
    res = validate(df)
    for w in res.warnings:
        print(f"warn: {w}")
    if not res.ok:
        sys.exit("trade log failed validation:\n  - " + "\n  - ".join(res.errors))
    df = compute_r(df)
    report(df, args.min_expectancy, args.alpha, args.n_boot, burn_in=args.burn_in)


if __name__ == "__main__":
    main()
