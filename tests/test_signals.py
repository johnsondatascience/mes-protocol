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


def test_limit_fill_bar_through_stop_is_a_loss():
    """A limit long that fills on a bar whose low also takes out the stop was
    filled on the way down and stopped on the same way down. Skipping the
    entry bar and letting a later rally score a TARGET turns a certain loss
    into a win."""
    d = date(2026, 3, 3)
    path = np.concatenate([np.full(11, 5010.0), np.full(379, 5020.0)])
    bars = bars_from_path(d, path, wick=0.5)
    ts = bars.index[10]
    bars.loc[ts, ["high", "low"]] = [5010.5, 4990.0]   # through entry AND stop
    bars = load_dataframe_bars(bars, source="FUTURES")

    sig = Signal(setup="IB_BREAK", session_date=d, direction="LONG",
                 signal_time=bars.index[5], entry_px=5000.0, stop_px=4996.0,
                 entry_style="LIMIT")
    f = simulate(sig, bars, MES)
    assert f.filled and f.entry_time == ts, (f.filled, f.entry_time)
    assert f.exit_reason == "STOP", f.exit_reason
    assert f.exit_time == ts and f.exit_px == 4996.0, (f.exit_time, f.exit_px)
    print(f"  limit filled and stopped on the same bar -> {f.exit_reason}")


def test_stop_entry_bar_spanning_stop_is_a_flagged_loss():
    """A stop entry whose fill bar also spans the protective stop cannot be
    sequenced from OHLC. The stop is assumed and the bar is flagged."""
    d = date(2026, 3, 3)
    path = np.concatenate([np.full(11, 5000.0), np.full(379, 5030.0)])
    bars = bars_from_path(d, path, wick=0.5)
    ts = bars.index[10]
    bars.loc[ts, ["high", "low"]] = [5011.0, 5003.0]
    bars = load_dataframe_bars(bars, source="FUTURES")

    sig = Signal(setup="IB_BREAK", session_date=d, direction="LONG",
                 signal_time=bars.index[5], entry_px=5010.0, stop_px=5004.0,
                 entry_style="STOP")
    f = simulate(sig, bars, MES)
    assert f.filled and f.entry_time == ts, (f.filled, f.entry_time)
    assert f.exit_reason == "STOP", f.exit_reason
    assert f.ambiguous_bar is True
    print(f"  stop entry bar spanned the stop -> {f.exit_reason}, flagged")


# --- look-ahead in generators ------------------------------------------------

S1_LOOKAHEAD_DATE = date(2026, 3, 3)


def s1_quick_retest_bars(break_delta, retest_deltas):
    """IB 4989.5-5010.5, close-through break at 10:00, retests from 10:01.

    Cumulative delta climbs to +3000 through the IB. The break bar and the
    bars after it get the explicit per-bar deltas passed in, so a test can
    place the new delta extreme exactly where it wants it.
    """
    d0, d1 = date(2026, 3, 2), S1_LOOKAHEAD_DATE
    ib = np.concatenate([np.linspace(5000, 5010, 8), np.linspace(5010, 4990, 15),
                         np.linspace(4990, 5002, 7)])
    brk = np.array([5012.0])
    retest = np.full(5, 5011.0)
    rally = np.linspace(5011, 5040, 30)
    path = np.concatenate([ib, brk, retest, rally])
    path = np.concatenate([path, np.full(390 - len(path), 5040.0)])

    day = bars_from_path(d1, path, overnight_center=5000.0)
    rth = (day.index.date == d1) & (day.index.time >= time(9, 30))
    deltas = np.full(390, 50.0)
    deltas[:30] = 100.0
    deltas[30] = break_delta
    deltas[31:31 + len(retest_deltas)] = retest_deltas
    vol = day.loc[rth, "volume"].to_numpy()
    day.loc[rth, "buy_volume"] = (vol + deltas) / 2
    day.loc[rth, "sell_volume"] = (vol - deltas) / 2

    bars = load_dataframe_bars(pd.concat([prior_balance_day(d0), day]).sort_index(),
                               source="FUTURES")
    return bars


def _decision(sig):
    """The fields a trader acts on. Context (e.g. day_type, which the protocol
    defines at 11:00 for every trade) is descriptive and excluded."""
    return (sig.signal_time, sig.direction, sig.entry_px, sig.stop_px,
            sig.entry_style, dict(sig.checklist))


def assert_signals_survive_truncation(generator, bars, d):
    """Re-run the generator on the tape cut at each signal's bar. A signal
    that changes, or disappears, was using bars that had not printed yet."""
    full = [s for s in generator(bars, {x.date: x for x in
                                        build_sessions(bars, tick=TICK)}[d], MES)]
    assert full, "fixture should produce at least one signal"
    for s in full:
        cut = bars[bars.index <= s.signal_time].copy()
        cut.attrs.update(bars.attrs)
        lv_cut = {x.date: x for x in build_sessions(cut, tick=TICK)}[d]
        seen = [_decision(x) for x in generator(cut, lv_cut, MES)
                if x.signal_time == s.signal_time]
        assert _decision(s) in seen, (
            f"signal at {s.signal_time.time()} not reproducible from bars "
            f"through that time: full={_decision(s)} truncated={seen}")
    return full


def test_s1_delta_confirmation_never_reads_future_bars():
    # new cumulative-delta high prints at 10:02; retest bars start at 10:01
    bars = s1_quick_retest_bars(break_delta=-200.0,
                                retest_deltas=[-100.0, 1000.0, 50.0, 50.0, 50.0])
    sigs = generate_s1(bars, {x.date: x for x in
                              build_sessions(bars, tick=TICK)}[S1_LOOKAHEAD_DATE], MES)
    for s in sigs:
        if s.checklist["delta_confirmed"] is True:
            assert s.signal_time.time() >= time(10, 2), s.signal_time
    assert_signals_survive_truncation(generate_s1, bars, S1_LOOKAHEAD_DATE)
    print("  " + "; ".join(f"{s.signal_time.time()} delta={s.checklist['delta_confirmed']}"
                           for s in sigs))


def test_s1_break_bar_itself_can_confirm_delta():
    """'New session extreme within 3 bars of the break' includes the break bar."""
    bars = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    sigs = generate_s1(bars, {x.date: x for x in
                              build_sessions(bars, tick=TICK)}[S1_LOOKAHEAD_DATE], MES)
    assert sigs, "expected an S1 signal on the quick retest"
    assert sigs[0].checklist["delta_confirmed"] is True, sigs[0].checklist
    print(f"  break-bar delta extreme confirmed at {sigs[0].signal_time.time()}")


def test_s1_one_signal_per_break():
    """Price sitting above the IB after a signal is the same break, not a new
    one. A break is a close crossing the edge from inside; re-arming on every
    close beyond it would log one idea as several correlated trades."""
    bars = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    sigs = generate_s1(bars, {x.date: x for x in
                              build_sessions(bars, tick=TICK)}[S1_LOOKAHEAD_DATE], MES)
    assert len(sigs) == 1, [s.signal_time.time() for s in sigs]
    print(f"  one break, one signal at {sigs[0].signal_time.time()}")


def test_s3_signals_survive_truncation():
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = assert_signals_survive_truncation(generate_s3, bars, d1)
    print(f"  {len(sigs)} S3 signals reproducible from truncated tape")


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
