#!/usr/bin/env python3
"""Tests for mesproto.signals.

The fill-convention tests are the ones that matter. A backtest gets optimistic
one small concession at a time, and each concession looks reasonable in
isolation — so they are pinned here.
"""
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from mesproto.config import (
    ET, GAP_OPEN_PCT, MES, S1_NEWS_EVENTS, S1_STOP_CAP_PTS, S1_STOP_FLOOR_PTS, SPY,
)
from mesproto.levels import build_sessions, load_dataframe_bars
from mesproto.news import NewsCalendar
from mesproto.schema import fills_to_log, validate
from mesproto.signals import (
    Fill, Signal, generate_s1, generate_s3, run_all, simulate,
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


def s3_chop_path():
    """Confirmed trend at 11:00, then two full swings through VWAP (4 crosses
    by 12:45), then an otherwise-textbook low-volume pullback into VWAP."""
    rise = np.linspace(5040, 5100, 90)                        # 09:30-11:00
    swings = np.concatenate([np.linspace(5100, 5050, 30), np.linspace(5050, 5100, 30),
                             np.linspace(5100, 5050, 30), np.linspace(5050, 5100, 30)])
    pull = np.concatenate([np.linspace(5100, 5074, 20), np.full(5, 5074.0)])
    cont = np.linspace(5074, 5130, 60)
    path = np.concatenate([rise, swings, pull, cont])
    return np.concatenate([path, np.full(390 - len(path), 5130.0)])


def s3_chop_volumes():
    v = np.full(390, 1500.0)
    v[210:235] = 300.0
    return v


def running_vwap_crosses(close: pd.Series, vwap: pd.Series) -> np.ndarray:
    """Crosses of close through VWAP counted through each bar (independent of
    the production helper on purpose)."""
    out, last, n = [], 0.0, 0
    for s in np.sign((close - vwap).to_numpy()):
        if s != 0:
            n += int(last != 0 and s != last)
            last = s
        out.append(n)
    return np.array(out)


def build_two_days(day2_path, day2_vol=None, day2_on=None, day1_center=5000.0,
                   day2_delta_bias=0.2):
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    b = pd.concat([
        prior_balance_day(d0, day1_center),
        bars_from_path(d1, day2_path, volumes=day2_vol,
                       overnight_center=day2_on if day2_on else day2_path[0],
                       delta_bias=day2_delta_bias),
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


def test_s1_gap_open_is_included_and_flagged():
    """Amended 2026-09-15: gap opens beyond 1% of the prior close are a
    different regime, so they are flagged and reported separately — but they
    no longer fail the checklist."""
    bars, lv, d1 = build_two_days(s1_path(), day1_center=4900.0)   # ~2% gap
    assert abs(lv[d1].gap_pct) > GAP_OPEN_PCT
    sigs = generate_s1(bars, lv[d1], MES)
    assert sigs, "gapped sessions still trade"
    assert "gap_within_limit" not in sigs[0].checklist, sigs[0].checklist
    assert sigs[0].context["gap_pct"] == lv[d1].gap_pct

    log = fills_to_log([simulate(s, bars, MES) for s in sigs], MES)
    assert abs(log["gap_pct"].iloc[0] - lv[d1].gap_pct) < 1e-9, log["gap_pct"].tolist()
    print(f"  gap {lv[d1].gap_pct:+.2%}: not a checklist condition; logged as gap_pct")


def test_s1_news_day_stands_down_before_1030():
    bars = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    d = S1_LOOKAHEAD_DATE
    lv = {x.date: x for x in build_sessions(bars, tick=TICK)}[d]

    carried = tuple(S1_NEWS_EVENTS) + ("GDP",)

    def calendar(days, start=d - timedelta(days=30), end=d + timedelta(days=30)):
        return NewsCalendar(start=start, end=end, events=frozenset(carried),
                            days={k: frozenset(v) for k, v in days.items()})

    on_news = generate_s1(bars, lv, MES, news=calendar({d: {S1_NEWS_EVENTS[0]}}))
    assert all(s.signal_time.time() >= time(10, 30) for s in on_news), \
        [s.signal_time.time() for s in on_news]

    quiet = generate_s1(bars, lv, MES, news=calendar({date(2026, 3, 6): {"CPI"}}))
    assert quiet and quiet[0].checklist["no_news_stand_down"] is True

    other = generate_s1(bars, lv, MES, news=calendar({d: {"GDP"}}))
    assert other and other[0].checklist["no_news_stand_down"] is True, \
        "a release outside the S1 list is not an S1 stand-down"

    unknown = generate_s1(bars, lv, MES)
    assert unknown[0].signal_time.time() < time(10, 30)
    assert unknown[0].checklist["no_news_stand_down"] is None, \
        "no calendar supplied: the stand-down could not be verified"
    assert unknown[0].checklist_ok is False

    stale = generate_s1(bars, lv, MES, news=calendar({}, end=d - timedelta(days=1)))
    assert stale and stale[0].checklist["no_news_stand_down"] is None, \
        "a session past the calendar's coverage is unknown, not a quiet day"
    print(f"  news day: {len(on_news)} signals before 10:30 suppressed; "
          f"no calendar or uncovered session -> None")


def test_s3_checklist_maps_every_protocol_condition():
    """Day gate (3 conditions + VWAP-chop stand-down) and trigger, each named."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = generate_s3(bars, lv[d1], MES)
    assert sigs
    expected = {"opened_outside_value", "one_timeframing", "delta_at_extreme",
                "vwap_crosses_ok", "pullback_to_vwap", "volume_declining",
                "counter_delta_ok", "stop_within_cap"}
    assert set(sigs[0].checklist) == expected, sorted(sigs[0].checklist)
    assert sigs[0].checklist_ok is True
    print(f"  checklist keys: {sorted(sigs[0].checklist)}")


def test_s3_delta_diverging_from_trend_fails_day_gate():
    """Up-trend with cumulative delta at its session LOW is not 'delta at a
    session extreme with price'."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0,
                                  day2_delta_bias=-0.2)
    assert lv[d1].s3_gate() is True, "fixture must pass the bar-level gate"
    assert generate_s3(bars, lv[d1], MES) == []
    print("  diverging delta -> no S3 signals")


def _with_day2_deltas(bars, d, deltas):
    """Overwrite day-2 RTH buy/sell volume so bar deltas equal `deltas`."""
    out = bars.copy()
    rth = (out.index.date == d) & (out.index.time >= time(9, 30))
    vol = out.loc[rth, "volume"].to_numpy()
    out.loc[rth, "buy_volume"] = (vol + deltas) / 2
    out.loc[rth, "sell_volume"] = (vol - deltas) / 2
    out.attrs.update(bars.attrs)
    return out, {s.date: s for s in build_sessions(out, tick=TICK)}


def test_s3_delta_high_inside_confirming_bar_passes_gate():
    """Delta makes its session high at 10:45, inside the 10:30-11:00 bar that
    confirms one-timeframing, then eases for a few minutes. Flow agreed with
    price on the bar that confirmed the trend; a one-minute pause before the
    close of that bar must not fail the day."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    deltas = np.full(390, 100.0)
    deltas[:76] = 300.0          # 09:30-10:45 buying, new highs
    deltas[76:90] = -100.0       # 10:46-10:59 eases off its high
    bars, lv = _with_day2_deltas(bars, d1, deltas)
    sigs = generate_s3(bars, lv[d1], MES)
    assert sigs, "delta high inside the confirming bar should pass the day gate"
    assert all(s.checklist["delta_at_extreme"] is True for s in sigs)
    print(f"  high at 10:45, eased by 10:59 -> {len(sigs)} S3 signal(s)")


def test_s3_delta_high_before_confirming_bar_fails_gate():
    """Loosened is not 'anything goes': delta whose session high was set
    before 10:30 and not exceeded in the confirming bar has stopped agreeing."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    deltas = np.full(390, 100.0)
    deltas[:46] = 300.0          # 09:30-10:15 buying
    deltas[46:90] = -50.0        # 10:16-10:59 no new high
    bars, lv = _with_day2_deltas(bars, d1, deltas)
    assert lv[d1].s3_gate() is True
    assert generate_s3(bars, lv[d1], MES) == []
    print("  high at 10:15, none in the confirming bar -> no S3 signals")


def test_s3_delta_extreme_unverifiable_without_delta():
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    stripped = bars.copy()
    stripped[["buy_volume", "sell_volume"]] = np.nan
    stripped.attrs.update(bars.attrs)
    stripped.attrs["has_delta"] = False
    sess = {s.date: s for s in build_sessions(stripped, tick=TICK)}
    sigs = generate_s3(stripped, sess[d1], MES)
    assert sigs, "bar-only data should still generate, flagged"
    assert sigs[0].checklist["delta_at_extreme"] is None
    assert sigs[0].checklist_ok is False
    print("  no delta -> delta_at_extreme None, checklist_ok False")


def test_s3_stands_down_once_vwap_crossed_too_often():
    """'Any day where VWAP has been crossed more than 3 times' is knowable bar
    by bar. Counting only through 11:00 lets an afternoon of chop through."""
    from mesproto.config import S3_MAX_VWAP_CROSSES
    from mesproto.levels import rth_slice
    bars, lv, d1 = build_two_days(s3_chop_path(), day2_vol=s3_chop_volumes(),
                                  day2_on=5040.0)
    s = lv[d1]
    assert s.s3_gate() is True and s.vwap_crosses <= S3_MAX_VWAP_CROSSES
    rth = rth_slice(bars, d1)
    crosses = running_vwap_crosses(rth["close"], s.vwap)
    assert crosses[-1] > S3_MAX_VWAP_CROSSES, "fixture must chop through VWAP"
    for sig in generate_s3(bars, s, MES):
        n = crosses[rth.index.get_loc(sig.signal_time)]
        assert n <= S3_MAX_VWAP_CROSSES, (sig.signal_time.time(), n)
    print(f"  {crosses[-1]} crosses by the close; no entry after the "
          f"{S3_MAX_VWAP_CROSSES + 1}th")


def test_s3_log_carries_ib_range_covariate():
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    fills = [simulate(s, bars, MES) for s in generate_s3(bars, lv[d1], MES)]
    log = fills_to_log(fills, MES)
    assert not log.empty
    assert log["ib_range_pts"].notna().all(), log["ib_range_pts"].tolist()
    print(f"  S3 rows carry ib_range_pts={log['ib_range_pts'].iloc[0]:.2f}")


def test_s3_signals_survive_truncation():
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = assert_signals_survive_truncation(generate_s3, bars, d1)
    print(f"  {len(sigs)} S3 signals reproducible from truncated tape")


def test_resting_entry_cancelled_at_stand_down():
    """A stand-down applies to fills, not just signals. An S3 order resting
    at 14:55 must not fill at 15:05, after the 15:00 cutoff."""
    import dataclasses
    from mesproto.config import S3_ENTRY_CUTOFF
    d = date(2026, 3, 3)
    before = (15 * 60 + 5) - (9 * 60 + 30)                 # bars up to 15:05
    path = np.concatenate([np.full(before, 5010.0), np.full(390 - before, 4999.0)])
    bars = load_dataframe_bars(bars_from_path(d, path, wick=0.5), source="FUTURES")
    at_1455 = bars.index[bars.index.time == time(14, 55)][0]
    sig = Signal(setup="VWAP_CONT", session_date=d, direction="LONG",
                 signal_time=at_1455, entry_px=5000.0, stop_px=4994.0,
                 entry_style="LIMIT", entry_deadline=S3_ENTRY_CUTOFF)
    assert simulate(dataclasses.replace(sig, entry_deadline=None), bars, MES).filled, \
        "fixture must fill when no deadline applies"
    assert simulate(sig, bars, MES).filled is False

    bars3, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = generate_s3(bars3, lv[d1], MES)
    assert sigs and all(s.entry_deadline == S3_ENTRY_CUTOFF for s in sigs)
    print("  order resting past 15:00 cancelled; S3 signals carry the deadline")


def test_log_failed_checks_and_notes():
    d = date(2026, 3, 3)
    ts = pd.Timestamp.combine(d, time(10, 30)).tz_localize(ET)
    clean = Signal(setup="IB_BREAK", session_date=d, direction="LONG", signal_time=ts,
                   entry_px=5000.0, stop_px=4996.0, entry_style="LIMIT",
                   checklist={"a": True, "b": True})
    flagged = Signal(setup="IB_BREAK", session_date=d, direction="LONG", signal_time=ts,
                     entry_px=5000.0, stop_px=4996.0, entry_style="LIMIT",
                     checklist={"a": True, "delta_confirmed": None})
    fills = [Fill(clean, True, ts, 5000.0, ts, 5008.0, "TARGET", 3, False),
             Fill(clean, True, ts, 5000.0, ts, 4996.0, "STOP", 1, True),
             Fill(flagged, True, ts, 5000.0, ts, 5008.0, "TARGET", 3, False)]
    log = fills_to_log(fills, MES)
    assert log["notes"].tolist() == ["", "ambiguous_bar", ""], log["notes"].tolist()
    assert log["failed_checks"].tolist() == ["", "", "delta_confirmed=None"], \
        log["failed_checks"].tolist()
    print(f"  notes: {log['notes'].tolist()}  failed_checks: {log['failed_checks'].tolist()}")


def test_generated_log_records_contract_and_mechanical_exit():
    """A generated trade has no discretion, so its managed exit is the bracket;
    and the contract it was scored for travels with the row."""
    bars, lv, d1 = build_two_days(s1_path())
    log = fills_to_log(run_all(bars, list(lv.values()), MES), SPY)
    assert not log.empty
    assert (log["contract"] == "SPY").all(), log["contract"].tolist()
    assert (log["mech_exit_px"] == log["exit_px"]).all()
    assert (log["mech_exit_reason"] == log["exit_reason"]).all()
    print(f"  {len(log)} rows: contract=SPY, mech exit == exit")


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
