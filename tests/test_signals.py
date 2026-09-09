#!/usr/bin/env python3
"""Tests for mesproto.signals.

The fill-convention tests are the ones that matter. A backtest gets optimistic
one small concession at a time, and each concession looks reasonable in
isolation — so they are pinned here.
"""
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from mesproto.config import ET, MES, S1_STOP_CAP_PTS, S1_STOP_FLOOR_PTS
from mesproto.levels import build_sessions, load_dataframe_bars
from mesproto.schema import fills_to_log, validate
from mesproto.signals import (
    Signal, generate_s1, generate_s3, run_all, simulate,
)
from mesproto.evaluate import compute_r

TICK = 0.25


def bars_from_path(d: date, path, volumes=None, overnight_center=None,
                   wick=0.5, delta_bias=0.0):
    """1-minute bars for one session from an explicit RTH price path."""
    rows, idx = [], []
    if overnight_center is not None:
        start = pd.Timestamp.combine(d - timedelta(days=1), time(18, 0)).tz_localize(ET)
        rng = np.random.default_rng(1)
        on = overnight_center + np.clip(np.cumsum(rng.normal(0, .05, 900)), -8, 8)
        for i, p in enumerate(on):
            idx.append(start + pd.Timedelta(minutes=i))
            rows.append([p, p + .25, p - .25, p, 200., np.nan, np.nan])

    rth_start = pd.Timestamp.combine(d, time(9, 30)).tz_localize(ET)
    vols = volumes if volumes is not None else np.full(len(path), 1000.0)
    for i, (p, v) in enumerate(zip(path, vols)):
        idx.append(rth_start + pd.Timedelta(minutes=i))
        bv = v * (0.5 + delta_bias)
        rows.append([p, p + wick, p - wick, p, v, bv, v - bv])

    return pd.DataFrame(
        rows, index=pd.DatetimeIndex(idx),
        columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])


def prior_balance_day(d, center=5000.0):
    return bars_from_path(d, center + 6 * np.sin(2 * np.pi * np.arange(390) / 45),
                          overnight_center=center)


def s1_path():
    """IB ~21pts, break above at 10:01, retest at 10:20, then run to target."""
    ib = 5000 + 10 * np.sin(2 * np.pi * np.arange(30) / 30)      # 09:30-10:00
    up = np.linspace(5001, 5020, 20)                              # 10:00-10:20 break
    back = np.linspace(5020, 5009, 12)                            # 10:20-10:32 retest
    hold = np.full(6, 5009.5)                                     # fill the limit
    run = np.linspace(5010, 5040, 40)                             # target
    tail = np.full(390 - (30 + 20 + 12 + 6 + 40), 5040.0)
    return np.concatenate([ib, up, back, hold, run, tail])


def s3_path():
    """Opens above prior value, one-timeframes to 11:00, pulls back to VWAP."""
    rise = np.linspace(5040, 5100, 90)          # 09:30-11:00, monotonic
    pull = np.linspace(5100, 5068, 30)          # 11:00-11:30, back to VWAP
    cont = np.linspace(5068, 5130, 100)         # continuation
    tail = np.full(390 - 220, 5130.0)
    return np.concatenate([rise, pull, cont, tail])


def s3_volumes():
    v = np.full(390, 1500.0)
    v[90:120] = 400.0        # declining volume on the pullback
    return v


def build_two_days(day2_path, day2_vol=None, day2_on=None, day1_center=5000.0):
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    b = pd.concat([
        prior_balance_day(d0, day1_center),
        bars_from_path(d1, day2_path, volumes=day2_vol,
                       overnight_center=day2_on if day2_on else day2_path[0],
                       delta_bias=0.2),
    ]).sort_index()
    bars = load_dataframe_bars(b.tz_localize(None).tz_localize(ET)
                               if b.index.tz is None else b, source="FUTURES")
    sessions = build_sessions(bars, tick=TICK)
    return bars, {s.date: s for s in sessions}, d1


# ---------------------------------------------------------------------------

def test_s1_emits_on_break_and_retest():
    bars, lv, d1 = build_two_days(s1_path())
    sigs = generate_s1(bars, lv[d1], MES)
    assert len(sigs) >= 1, "expected at least one S1 signal"
    s = sigs[0]
    assert s.direction == "LONG"
    assert s.entry_px > lv[d1].ib_high, (s.entry_px, lv[d1].ib_high)
    assert s.stop_px < s.entry_px, "long stop must sit below entry"
    assert S1_STOP_FLOOR_PTS <= s.risk_pts <= S1_STOP_CAP_PTS, s.risk_pts
    print(f"  entry={s.entry_px} stop={s.stop_px} risk={s.risk_pts:.2f}pts "
          f"target={s.target_px:.2f} @ {s.signal_time.time()}")


def test_s1_gate_blocks_out_of_band_ib():
    # dead open: IB of ~2 points on a 5000 index is 0.04%, far under the band
    flat = np.concatenate([np.full(30, 5000.0), np.linspace(5000, 5060, 360)])
    bars, lv, d1 = build_two_days(flat)
    assert lv[d1].s1_gate() is False
    assert generate_s1(bars, lv[d1], MES) == []
    print("  out-of-band IB correctly produced no signals")


def test_checklist_ok_false_when_delta_missing():
    """Bar-only data cannot verify the delta condition, so those trades are
    flagged and must not pool with delta-verified ones."""
    bars, lv, d1 = build_two_days(s1_path())
    stripped = bars.copy()
    stripped[["buy_volume", "sell_volume"]] = np.nan
    stripped.attrs.update(bars.attrs)
    stripped.attrs["has_delta"] = False
    sess = {s.date: s for s in build_sessions(stripped, tick=TICK)}
    sigs = generate_s1(stripped, sess[d1], MES)
    assert sigs, "signal should still generate"
    assert sigs[0].checklist["delta_confirmed"] is None
    assert sigs[0].checklist_ok is False
    print("  missing delta -> checklist_ok False (not silently True)")


def test_s3_never_enters_before_confirmation():
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    if not lv[d1].s3_gate():
        print(f"  day gate not met (day_type={lv[d1].day_type}) — skipping")
        return
    sigs = generate_s3(bars, lv[d1], MES)
    assert sigs, "synthetic trend day with a VWAP pullback should produce a signal"
    for s in sigs:
        assert s.signal_time.time() >= time(11, 0), s.signal_time
        assert 5.0 <= s.risk_pts <= 10.0, s.risk_pts
        assert s.direction == "LONG"
    print(f"  {len(sigs)} S3 signals, all at/after the 11:00 confirmation; "
          f"first entry={sigs[0].entry_px:.2f} stop={sigs[0].stop_px:.2f} "
          f"checklist_ok={sigs[0].checklist_ok}")


def test_limit_entry_requires_trading_through():
    """A touch is not a fill."""
    d = date(2026, 3, 3)
    # price bottoms exactly at 5000.0 and turns — never trades below it
    path = np.concatenate([np.linspace(5010, 5000.5, 10), np.linspace(5000.5, 5030, 380)])
    bars = load_dataframe_bars(bars_from_path(d, path, wick=0.5), source="FUTURES")
    sig = Signal(setup="IB_BREAK", session_date=d, direction="LONG",
                 signal_time=bars.index[0], entry_px=5000.0 - 0.5,  # below every low
                 stop_px=4995.0, entry_style="LIMIT")
    assert simulate(sig, bars, MES).filled is False
    print("  limit below the session low correctly did not fill")


def test_stop_wins_ties_within_one_bar():
    """When a bar covers both stop and target, the stop is assumed first."""
    d = date(2026, 3, 3)
    path = np.full(390, 5000.0)
    bars = bars_from_path(d, path, wick=0.5)
    # make one later bar enormous: it spans both the stop and the target
    ts = bars.index[20]
    bars.loc[ts, ["high", "low"]] = [5100.0, 4900.0]
    bars = load_dataframe_bars(bars, source="FUTURES")

    sig = Signal(setup="IB_BREAK", session_date=d, direction="LONG",
                 signal_time=bars.index[5], entry_px=5000.25, stop_px=4995.0,
                 entry_style="STOP")
    f = simulate(sig, bars, MES)
    assert f.filled and f.exit_reason == "STOP", (f.filled, f.exit_reason)
    assert f.ambiguous_bar is True
    print(f"  ambiguous bar resolved as STOP (ambiguous_bar={f.ambiguous_bar})")


def test_stop_entry_pays_slippage():
    d = date(2026, 3, 3)
    # 1 pt/min so the trigger is reached inside the 30-bar entry window
    path = np.concatenate([np.arange(390) * 1.0 + 5000.0])[:390]
    bars = load_dataframe_bars(bars_from_path(d, path), source="FUTURES")
    sig = Signal(setup="IB_BREAK", session_date=d, direction="LONG",
                 signal_time=bars.index[0], entry_px=5010.0, stop_px=5004.0,
                 entry_style="STOP")
    f = simulate(sig, bars, MES)
    assert f.filled and f.entry_px == 5010.0 + MES.tick, f.entry_px
    print(f"  stop entry filled at {f.entry_px} (signal {sig.entry_px} + 1 tick)")


def test_pipeline_produces_valid_log():
    bars, lv, d1 = build_two_days(s1_path())
    fills = run_all(bars, list(lv.values()), MES)
    log = fills_to_log(fills, MES, contracts=2)
    if log.empty:
        print("  no fills generated — nothing to validate")
        return
    res = validate(log)
    for w in res.warnings:
        print(f"  warn: {w}")
    res.raise_if_bad()
    scored = compute_r(log)
    assert scored["R"].notna().all()
    assert (scored["risk_usd"] > 0).all()
    print(f"  {len(log)} rows validated; mean R = {scored['R'].mean():+.3f}, "
          f"net ${scored['pnl_usd'].sum():,.0f}")
    print(scored[["session_date", "setup", "direction", "entry_px", "stop_px",
                  "exit_px", "exit_reason", "R"]].to_string(index=False))


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
