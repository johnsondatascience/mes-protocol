"""Trade-log schema — the contract between hand-logged replay and generated signals.

Both paths write this shape, so `mesproto.evaluate` treats them identically and
you can compare a generator's bar-level result against your own replay of the
same sessions. That comparison is the point: where they disagree is where the
order flow was carrying the signal, or where you were overriding your rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from .config import (
    CONTRACTS, MECHANICAL_TARGET_R, OFF_CHECKLIST_WARN_FRAC, PRICE_EPS, Contract,
)
from .signals import Fill

SETUPS = ("IB_BREAK", "LEVEL2LEVEL", "VWAP_CONT", "ABSORPTION", "ON_INVENTORY",
          "IB_FAIL")
EXIT_REASONS = ("TARGET", "STOP", "TIME", "MANUAL", "BREAKEVEN")
MECH_EXIT_REASONS = ("TARGET", "STOP", "TIME")   # all a fixed 1:2 bracket can do
DAY_TYPES = ("TREND_UP", "TREND_DOWN", "BALANCE", "DOUBLE_DIST", "UNCLASSIFIED")
DIRECTIONS = ("LONG", "SHORT")
GRADES = ("A", "B", "C")
OPTIONAL_COLUMNS = ("notes", "source", "gap_pct", "failed_checks")

# exit_px / exit_reason: what actually happened (the managed exit).
# mech_exit_px / mech_exit_reason: the fixed 1:2 bracket's outcome, logged for
# every trade — protocol §03, and what the statistics run on.
COLUMNS = [
    "trade_id", "session_date", "setup", "direction", "contract", "entry_time",
    "entry_px", "stop_px", "exit_px", "exit_reason",
    "mech_exit_px", "mech_exit_reason", "contracts",
    "day_type", "ib_range_pts", "on_range_pos", "gap_pct", "checklist_ok",
    "failed_checks", "grade", "source", "notes",
]


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError("trade log failed validation:\n  - "
                             + "\n  - ".join(self.errors))


def validate(df: pd.DataFrame) -> ValidationResult:
    """Reject logs the evaluator would silently misread.

    The costly failures are the quiet ones: a direction typo that compute_r
    scores as SHORT, a stop on the wrong side that still has positive risk
    distance, a checklist flag of 2 that is neither in nor out of the sample.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns and c not in OPTIONAL_COLUMNS]
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationResult(False, errors, warnings)

    bad_setup = set(df["setup"]) - set(SETUPS)
    if bad_setup:
        errors.append(f"unknown setup values: {sorted(bad_setup)}")
    bad_exit = set(df["exit_reason"].dropna()) - set(EXIT_REASONS)
    if bad_exit:
        errors.append(f"unknown exit_reason values: {sorted(bad_exit)}")
    bad_contract = set(df["contract"].astype(str)) - set(CONTRACTS)
    if bad_contract:
        errors.append(f"unknown contract values: {sorted(bad_contract)} "
                      f"(known: {sorted(CONTRACTS)})")
    mech_reason = df["mech_exit_reason"].astype(str).str.upper()
    bad_mech = set(mech_reason) - set(MECH_EXIT_REASONS)
    if bad_mech:
        errors.append(f"mech_exit_reason must be one of {MECH_EXIT_REASONS}, got "
                      f"{sorted(bad_mech)} — MANUAL and BREAKEVEN belong in exit_reason")
    if df["mech_exit_px"].isna().any():
        errors.append(f"{int(df['mech_exit_px'].isna().sum())} rows have no mech_exit_px")
    bad_day = set(df["day_type"].dropna()) - set(DAY_TYPES)
    if bad_day:
        errors.append(f"unknown day_type values: {sorted(bad_day)}")
    direction = df["direction"].astype(str).str.upper()
    bad_dir = set(direction) - set(DIRECTIONS)
    if bad_dir:
        errors.append(f"unknown direction values: {sorted(bad_dir)}")
    bad_flag = ~df["checklist_ok"].isin([0, 1])
    if bad_flag.any():
        errors.append(f"{int(bad_flag.sum())} rows have checklist_ok not in {{0, 1}}")
    bad_grade = set(df["grade"].dropna()) - set(GRADES)
    if bad_grade:
        errors.append(f"unknown grade values: {sorted(bad_grade)}")
    if (df["contracts"] <= 0).any():
        errors.append(f"{int((df['contracts'] <= 0).sum())} rows have contracts <= 0")

    risk = (df["entry_px"] - df["stop_px"]).abs()
    if (risk <= 0).any():
        errors.append(f"{int((risk <= 0).sum())} rows have zero/negative risk distance")
    wrong_side = ((direction == "LONG") & (df["stop_px"] > df["entry_px"])) \
        | ((direction == "SHORT") & (df["stop_px"] < df["entry_px"]))
    if wrong_side.any():
        errors.append(f"{int(wrong_side.sum())} rows have the stop on the wrong side "
                      "of entry for their direction")
    # A bracket exits at its stop or at exactly MECHANICAL_TARGET_R; anything
    # else under those labels is a typo, and a high one inflates R.
    sign = direction.map({"LONG": 1.0, "SHORT": -1.0})
    target = df["entry_px"] + sign * MECHANICAL_TARGET_R * risk
    off_target = (mech_reason == "TARGET") & ((df["mech_exit_px"] - target).abs() > PRICE_EPS)
    if off_target.any():
        errors.append(f"{int(off_target.sum())} rows log a TARGET mech_exit_px that is not "
                      f"{MECHANICAL_TARGET_R:g}R from entry")
    off_stop = (mech_reason == "STOP") & ((df["mech_exit_px"] - df["stop_px"]).abs() > PRICE_EPS)
    if off_stop.any():
        errors.append(f"{int(off_stop.sum())} rows log a STOP mech_exit_px that is not "
                      "the stop_px")
    if df["trade_id"].duplicated().any():
        errors.append("duplicate trade_id values")

    if "UNCLASSIFIED" in set(df["day_type"].dropna()):
        warnings.append("some rows are UNCLASSIFIED — day-type conditioning "
                        "will exclude them")
    off = (df["checklist_ok"] == 0).mean()
    if off > OFF_CHECKLIST_WARN_FRAC:
        generated = "source" in df.columns and (df["source"] == "GENERATED").all()
        why = ("conditions the data could not verify (None) or that failed"
               if generated else "a discipline signal, not a data problem")
        warnings.append(f"{off:.0%} of rows have checklist_ok=0 — {why}; "
                        "the primary sample will be small")
    return ValidationResult(not errors, errors, warnings)


def fills_to_log(fills: Sequence[Fill], contract: Contract, contracts: int = 1,
                 source: str = "GENERATED",
                 start_id: int = 1) -> pd.DataFrame:
    """Convert simulated fills into the canonical trade-log shape."""
    rows = []
    tid = start_id
    for f in fills:
        if not f.filled:
            continue
        s = f.signal
        rows.append({
            "trade_id": tid,
            "session_date": s.session_date,
            "setup": s.setup,
            "direction": s.direction,
            "contract": contract.symbol,
            "entry_time": f.entry_time.time().isoformat() if f.entry_time else None,
            "entry_px": round(f.entry_px, 4),
            "stop_px": round(s.stop_px, 4),
            # a generated trade has no discretion: its managed exit IS the bracket
            "exit_px": round(f.exit_px, 4),
            "exit_reason": f.exit_reason,
            "mech_exit_px": round(f.exit_px, 4),
            "mech_exit_reason": f.exit_reason,
            "contracts": contracts,
            "day_type": s.context.get("day_type", "UNCLASSIFIED"),
            "ib_range_pts": s.context.get("ib_range_pts"),
            "on_range_pos": s.context.get("on_range_pos"),
            "gap_pct": s.context.get("gap_pct"),
            "checklist_ok": int(s.checklist_ok),
            # which conditions kept the trade out, so the report can say why
            "failed_checks": ";".join(
                f"{k}={v}" for k, v in s.checklist.items() if v is not True),
            "grade": "A",   # generated trades have no execution quality to grade
            "source": source,
            "notes": "ambiguous_bar" if f.ambiguous_bar else "",
        })
        tid += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def write_template(path: str, n_rows: int = 0) -> None:
    """Write an empty CSV with the right header, for hand logging."""
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
