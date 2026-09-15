#!/usr/bin/env python3
"""Tests for mesproto.evaluate and the trade-log schema.

The gate tests pin the protocol's §06 table: checkpoints are fixed sample
sizes, evaluated on the trades that existed at that checkpoint. Re-deciding
on the whole sample every time the report runs is the informal sequential
test the protocol warns against.
"""
import contextlib
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

from mesproto.config import BURN_IN_TRADES, FUTILITY_GATES, MES
from mesproto.evaluate import (
    compute_r, futility_verdict, kill_threshold, primary_sample, report,
)
from mesproto.schema import COLUMNS, validate


def make_log(r_values, setup="VWAP_CONT", start=date(2026, 1, 5), per_session=2,
             checklist_ok=1, source="REPLAY"):
    """A trade log whose mechanical R (gross, before costs) is `r_values`,
    in chronological order: LONG MES, entry 5000, 5-point stop. +2R is a
    TARGET, -1R a STOP, anything else a TIME exit; the managed exit matches."""
    rows = []
    for k, r in enumerate(r_values):
        d = start + timedelta(days=k // per_session)
        reason = "TARGET" if r == 2.0 else "STOP" if r == -1.0 else "TIME"
        rows.append({
            "trade_id": k + 1, "session_date": d, "setup": setup, "direction": "LONG",
            "contract": "MES", "entry_time": f"{10 + k % per_session:02d}:15:00",
            "entry_px": 5000.0, "stop_px": 4995.0,
            "exit_px": 5000.0 + 5.0 * r, "exit_reason": reason,
            "mech_exit_px": 5000.0 + 5.0 * r, "mech_exit_reason": reason, "contracts": 1,
            "day_type": "TREND_UP", "ib_range_pts": 20.0, "on_range_pos": 0.5,
            "checklist_ok": checklist_ok, "grade": "A", "source": source, "notes": "",
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


# --- costs -------------------------------------------------------------------

def test_compute_r_uses_each_rows_contract():
    """A SPY trade must not be charged MES commission in MES points. Costs
    come from the row's own contract column, not a command-line default."""
    log = pd.concat([make_log([2.0]), make_log([2.0])], ignore_index=True)
    log.loc[1, "trade_id"] = 2
    log.loc[0, ["contract", "entry_px", "stop_px", "exit_px", "mech_exit_px"]] = \
        ["SPY", 500.0, 499.0, 502.0, 502.0]
    scored = compute_r(log)
    spy, mes = scored.iloc[0], scored.iloc[1]
    assert abs(spy["R"] - 2.0) < 1e-12, spy["R"]                     # SPY: no commission
    assert abs(spy["pnl_usd"] - 2.0) < 1e-12, spy["pnl_usd"]
    expected = (10.0 - MES.commission_rt / MES.point_value) / 5.0
    assert abs(mes["R"] - expected) < 1e-12, mes["R"]
    print(f"  SPY R={spy['R']:.3f} ${spy['pnl_usd']:.2f}; MES R={mes['R']:.3f}")


def test_statistics_run_on_the_mechanical_exit():
    """§03: the fixed 1:2 outcome is what the statistics run on; the managed
    exit measures discretion and is kept apart."""
    log = make_log([2.0])
    log.loc[0, ["exit_px", "exit_reason"]] = [5002.5, "MANUAL"]     # took +0.5R by hand
    scored = compute_r(log)
    comm = MES.commission_rt / MES.point_value
    assert abs(scored["R"].iloc[0] - (10.0 - comm) / 5.0) < 1e-12, scored["R"].iloc[0]
    assert abs(scored["managed_R"].iloc[0] - (2.5 - comm) / 5.0) < 1e-12, \
        scored["managed_R"].iloc[0]
    print(f"  mechanical {scored['R'].iloc[0]:+.3f}R, managed {scored['managed_R'].iloc[0]:+.3f}R")


def test_breakeven_exit_pays_stop_slippage():
    """A breakeven exit is a moved stop being hit — a stop order, so it pays
    the stop's slippage like any other."""
    log = make_log([2.0])
    log.loc[0, ["exit_px", "exit_reason"]] = [5000.0, "BREAKEVEN"]
    managed = compute_r(log)["managed_R"].iloc[0]
    expected = (-MES.slippage_ticks_stop * MES.tick - MES.commission_rt / MES.point_value) / 5.0
    assert abs(managed - expected) < 1e-12, managed
    print(f"  breakeven exit: {managed:+.3f}R")


def test_compute_r_is_net_of_stop_slippage():
    mes = compute_r(make_log([-1.0]))
    expected = (-5.0 - MES.slippage_ticks_stop * MES.tick
                - MES.commission_rt / MES.point_value) / 5.0
    assert abs(mes["R"].iloc[0] - expected) < 1e-12
    assert mes["R"].iloc[0] < mes["gross_R"].iloc[0]
    print(f"  stopped trade: gross {mes['gross_R'].iloc[0]:+.3f}R net {mes['R'].iloc[0]:+.3f}R")


# --- burn-in -----------------------------------------------------------------
# Amended 2026-09-15: burn-in trades stay in the primary sample, flagged, so
# the report can show the result with and without them.

def test_burn_in_is_flagged_on_first_trades_of_each_setup_chronologically():
    s3 = make_log([-1.0] * BURN_IN_TRADES + [2.0] * 10, setup="VWAP_CONT")
    s1 = make_log([0.5] * (BURN_IN_TRADES + 5), setup="IB_BREAK")
    s1["trade_id"] += 1000
    log = pd.concat([s3, s1]).sample(frac=1.0, random_state=3)      # file order scrambled

    primary, info = primary_sample(compute_r(log))
    counts = primary.groupby("setup").size().to_dict()
    assert counts == {"VWAP_CONT": BURN_IN_TRADES + 10, "IB_BREAK": BURN_IN_TRADES + 5}, \
        "burn-in trades are included in the primary sample"
    s3p = primary[primary["setup"] == "VWAP_CONT"]
    assert s3p.loc[s3p["burn_in"], "gross_R"].eq(-1.0).all() and \
        s3p.loc[~s3p["burn_in"], "gross_R"].eq(2.0).all(), \
        "burn-in must be the chronologically first trades, not the first rows in the file"
    assert info["burn_in"] == 2 * BURN_IN_TRADES, info
    print(f"  burn-in flagged {info['burn_in']} trades; primary {counts}")


def test_burn_in_counts_off_checklist_trades():
    """Burn-in exists because reading skill drifts with every trade taken,
    checklist or not — so off-checklist trades use up burn-in too."""
    log = make_log([1.0] * (BURN_IN_TRADES + 4))
    log.loc[:9, "checklist_ok"] = 0                                    # 10 of the first 30
    primary, info = primary_sample(compute_r(log))
    assert len(primary) == BURN_IN_TRADES + 4 - 10, len(primary)
    assert int(primary["burn_in"].sum()) == BURN_IN_TRADES - 10, primary["burn_in"].sum()
    assert info == {"burn_in": BURN_IN_TRADES - 10, "off_checklist": 10}, info
    print(f"  {info}")


# --- futility gates ----------------------------------------------------------

def test_kill_thresholds_match_protocol_table():
    """§06: kill if mean R below -0.013 at n=60 and +0.047 at n=150 (sigma 1.5)."""
    assert FUTILITY_GATES == (60, 150)
    assert abs(kill_threshold(60, 1.5, 0.15) - (-0.013)) < 5e-4, kill_threshold(60, 1.5, 0.15)
    assert abs(kill_threshold(150, 1.5, 0.15) - 0.047) < 5e-4, kill_threshold(150, 1.5, 0.15)
    print(f"  n=60 {kill_threshold(60, 1.5, 0.15):+.3f}R, "
          f"n=150 {kill_threshold(150, 1.5, 0.15):+.3f}R")


def _series(mean, n, sd=1.5, seed=0):
    x = np.random.default_rng(seed).normal(size=n)
    return mean + sd * (x - x.mean()) / x.std(ddof=1)


def test_no_gate_before_first_checkpoint():
    v = futility_verdict(_series(-0.8, FUTILITY_GATES[0] - 10), 0.15)
    assert v.verdict == "COLLECT", v
    print(f"  {v}")


def test_gate_is_decided_on_trades_at_the_checkpoint():
    """A dead first 60 is a kill at n=60, even if later trades would lift the
    whole-sample mean above the bound. Otherwise every re-run of the report
    is a fresh, uncorrected chance to survive."""
    first = _series(-0.30, FUTILITY_GATES[0], seed=1)
    later = _series(+1.00, 40, seed=2)
    whole = np.concatenate([first, later])
    v = futility_verdict(whole, 0.15)
    assert v.verdict == "KILL" and v.checkpoint == FUTILITY_GATES[0], v
    print(f"  {v}")


def test_gate_continue_and_pass():
    mid = futility_verdict(_series(0.30, 100, seed=4), 0.15)
    assert mid.verdict == "CONTINUE" and mid.next_gate == FUTILITY_GATES[1], mid
    done = futility_verdict(_series(0.30, 160, seed=5), 0.15)
    assert done.verdict == "PASSED", done
    print(f"  {mid}\n  {done}")


# --- schema validation -------------------------------------------------------

def test_validate_rejects_bad_direction_and_flags():
    log = make_log([1.0, -1.0, 2.0])
    assert validate(log).ok
    bad = log.copy()
    bad.loc[0, "direction"] = "LNOG"          # would be scored as SHORT
    bad.loc[1, "checklist_ok"] = 2
    res = validate(bad)
    assert not res.ok
    text = " ".join(res.errors)
    assert "direction" in text and "checklist_ok" in text, res.errors
    print(f"  errors: {res.errors}")


def test_validate_checks_mechanical_exit_against_the_bracket():
    """A 1:2 bracket can only exit at the stop, at 2R, or on time. A logged
    mechanical TARGET anywhere but 2R is a typo that would inflate R."""
    log = make_log([2.0, -1.0, 0.3])
    assert validate(log).ok, validate(log).errors
    cases = {
        "mech_exit_px": ("mech_exit_px", 5011.0, 0),        # TARGET not at 2R
        "stop": ("mech_exit_px", 4994.0, 1),                # STOP not at stop_px
        "mech_exit_reason": ("mech_exit_reason", "MANUAL", 2),
        "contract": ("contract", "MNQ", 0),
    }
    for needle, (col, value, row) in cases.items():
        bad = log.copy()
        bad.loc[row, col] = value
        res = validate(bad)
        assert not res.ok and any(needle in e for e in res.errors), (col, res.errors)
    print("  TARGET off 2R, STOP off stop_px, MANUAL mech reason, unknown contract: rejected")


def test_validate_rejects_stop_on_wrong_side():
    log = make_log([1.0])
    log.loc[0, "stop_px"] = 5005.0             # a LONG with its stop above entry
    res = validate(log)
    assert not res.ok and any("wrong side" in e for e in res.errors), res.errors
    print(f"  errors: {res.errors}")


# --- report ------------------------------------------------------------------

def test_report_includes_burn_in_and_names_the_gate():
    # gross +0.6R: comfortably clear of the n=60 gate after costs
    log = make_log(list(_series(0.6, BURN_IN_TRADES + 70, seed=9)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(compute_r(log), min_exp=0.15, alpha=0.01, n_boot=200)
    out = buf.getvalue()
    assert f"n={BURN_IN_TRADES + 70} " in out, "the gate sample includes burn-in"
    assert f"without burn-in: n=70 " in out, out
    assert "gate:" in out and "CONTINUE" in out, out
    print("  " + "\n  ".join(line for line in out.splitlines()
                             if "gate" in line or "burn" in line))


def test_report_shows_gap_open_sessions_separately():
    """Amended 2026-09-15: gap-open sessions are in the primary sample, and
    the report shows how they did on their own — a different regime."""
    log = make_log([2.0, -1.0] * 10)
    log["gap_pct"] = 0.001
    log.loc[:5, "gap_pct"] = [0.02, -0.015, 0.02, -0.015, 0.02, -0.015]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(compute_r(log), min_exp=0.15, alpha=0.01, n_boot=200)
    out = buf.getvalue()
    assert "n=20 " in out, "gap-open trades stay in the primary sample"
    line = next((ln for ln in out.splitlines() if "gap-open" in ln and "n=" in ln), None)
    assert line is not None and "n=6 " in line, out
    print(f" {line}")


def test_report_shows_managed_exits_separately():
    log = make_log([2.0, -1.0] * 5)
    log.loc[log["exit_reason"] == "TARGET", ["exit_px", "exit_reason"]] = [5005.0, "MANUAL"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(compute_r(log), min_exp=0.15, alpha=0.01, n_boot=200)
    line = next((ln for ln in buf.getvalue().splitlines() if "managed" in ln), None)
    assert line is not None and "mechanical" in line, buf.getvalue()
    print(f" {line}")


def test_report_breaks_down_exclusions_by_condition():
    log = make_log([1.0] * 6)
    log["failed_checks"] = ""
    log.loc[:2, "checklist_ok"] = 0
    log.loc[:2, "failed_checks"] = ["delta_confirmed=None",
                                    "delta_confirmed=None;no_news_stand_down=None",
                                    "no_news_stand_down=None"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(compute_r(log), min_exp=0.15, alpha=0.01, n_boot=200)
    out = buf.getvalue()
    assert "delta_confirmed=None: 2" in out and "no_news_stand_down=None: 2" in out, out
    print("  " + "\n  ".join(ln for ln in out.splitlines() if "=None:" in ln))


def test_gate_counts_burn_in_trades():
    """With burn-in included, the first checkpoint is reached at 60 trades in
    total, not 60 after burn-in."""
    log = make_log(list(_series(0.6, FUTILITY_GATES[0], seed=11)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(compute_r(log), min_exp=0.15, alpha=0.01, n_boot=200)
    gate = next(line for line in buf.getvalue().splitlines() if "gate:" in line)
    assert "COLLECT" not in gate and f"n={FUTILITY_GATES[0]}" in gate, gate
    print(f" {gate}")


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
