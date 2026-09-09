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

from .config import Contract
from .signals import Fill

SETUPS = ("IB_BREAK", "LEVEL2LEVEL", "VWAP_CONT", "ABSORPTION", "ON_INVENTORY",
          "IB_FAIL")
EXIT_REASONS = ("TARGET", "STOP", "TIME", "MANUAL", "BREAKEVEN")
DAY_TYPES = ("TREND_UP", "TREND_DOWN", "BALANCE", "DOUBLE_DIST", "UNCLASSIFIED")

COLUMNS = [
    "trade_id", "session_date", "setup", "direction", "entry_time",
    "entry_px", "stop_px", "exit_px", "exit_reason", "contracts",
    "day_type", "ib_range_pts", "on_range_pos", "checklist_ok", "grade",
    "source", "notes",
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
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns and c != "notes"]
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationResult(False, errors, warnings)

    bad_setup = set(df["setup"]) - set(SETUPS)
    if bad_setup:
        errors.append(f"unknown setup values: {sorted(bad_setup)}")
    bad_exit = set(df["exit_reason"].dropna()) - set(EXIT_REASONS)
    if bad_exit:
        errors.append(f"unknown exit_reason values: {sorted(bad_exit)}")
    bad_day = set(df["day_type"].dropna()) - set(DAY_TYPES)
    if bad_day:
        errors.append(f"unknown day_type values: {sorted(bad_day)}")

    risk = (df["entry_px"] - df["stop_px"]).abs()
    if (risk <= 0).any():
        errors.append(f"{int((risk <= 0).sum())} rows have zero/negative risk distance")
    if df["trade_id"].duplicated().any():
        errors.append("duplicate trade_id values")

    if "UNCLASSIFIED" in set(df["day_type"].dropna()):
        warnings.append("some rows are UNCLASSIFIED — day-type conditioning "
                        "will exclude them")
    if (df["checklist_ok"] == 0).mean() > 0.25:
        warnings.append(f"{(df['checklist_ok'] == 0).mean():.0%} of rows have "
                        "checklist_ok=0 — that is a discipline signal, not a "
                        "data problem, but the primary sample will be small")
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
            "entry_time": f.entry_time.time().isoformat() if f.entry_time else None,
            "entry_px": round(f.entry_px, 4),
            "stop_px": round(s.stop_px, 4),
            "exit_px": round(f.exit_px, 4),
            "exit_reason": f.exit_reason,
            "contracts": contracts,
            "day_type": s.context.get("day_type", "UNCLASSIFIED"),
            "ib_range_pts": s.context.get("ib_range_pts"),
            "on_range_pos": s.context.get("on_range_pos"),
            "checklist_ok": int(s.checklist_ok),
            "grade": "A",   # generated trades have no execution quality to grade
            "source": source,
            "notes": ("ambiguous_bar" if f.ambiguous_bar else "")
            + ("|" + ";".join(f"{k}={v}" for k, v in s.checklist.items()
                              if v is not True)),
        })
        tid += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def write_template(path: str, n_rows: int = 0) -> None:
    """Write an empty CSV with the right header, for hand logging."""
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
