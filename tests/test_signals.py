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
    ET, GAP_OPEN_PCT, IB_FAIL_STOP_BEYOND_EXTREME_PTS, IB_FAIL_STOP_CAP_PTS,
    IB_FAIL_STOP_FLOOR_PTS, MES, S1_NEWS_EVENTS, S1_STOP_CAP_PTS, S1_STOP_FLOOR_PTS, SPY,
)
from mesproto.levels import build_sessions, load_dataframe_bars
from mesproto.news import NewsCalendar
from mesproto.schema import fills_to_log, validate
from mesproto.signals import (
    Fill, Signal, generate_ib_fail, generate_s1, generate_s3, run_all, simulate,
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
    v[33:40] = 300.0         # a quick, thin climb through ~5062-5066: the low-volume node
    v[90:120] = 400.0        # declining volume on the pullback
    return v


def s3_va_edge_path(value_from=5068.0):
    """Trend day that builds value from `value_from` to 5075 on heavy volume by
    10:30, runs to 5100 on light volume, then pulls back to ~5076: inside 2
    points of the developing value-area high (~5075), but more than 2 points
    above VWAP. The light climb below `value_from` is the low-volume node."""
    rise1 = np.linspace(5040, value_from, 10)             # 09:30-09:40
    value = np.linspace(value_from, 5075, 50)             # 09:40-10:30, heavy
    rise2 = np.linspace(5075, 5100, 30)                   # 10:30-11:00, light
    pull = np.concatenate([np.linspace(5100, 5077, 15), np.full(6, 5076.5)])
    cont = np.linspace(5077, 5130, 60)
    path = np.concatenate([rise1, value, rise2, pull, cont])
    return np.concatenate([path, np.full(390 - len(path), 5130.0)])


def s3_va_edge_volumes():
    v = np.full(390, 600.0)
    v[10:60] = 3000.0
    v[90:111] = 250.0
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


def _bar_at(bars, d, hh, mm):
    return pd.Timestamp.combine(d, time(hh, mm)).tz_localize(ET)


def test_s1_retest_fails_on_close_not_wick():
    """Amended 2026-09-15: 'without closing back inside by more than 2 points'
    is read on the close. A wick deeper inside that closes within 2 points is
    still a retest that held — and the stop goes beyond that wick."""
    d = S1_LOOKAHEAD_DATE
    bars = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    lv = {x.date: x for x in build_sessions(bars, tick=TICK)}[d]
    edge = lv.ib_high
    ts = _bar_at(bars, d, 10, 1)

    wick = bars.copy()
    wick.attrs.update(bars.attrs)
    wick.loc[ts, "low"] = edge - 3.5                  # pokes 3.5 inside, closes above
    sigs = generate_s1(wick, lv, MES)
    assert sigs and sigs[0].signal_time == ts, [s.signal_time.time() for s in sigs]
    assert abs(sigs[0].stop_px - (edge - 3.5 - 1.0)) < 1e-9, sigs[0].stop_px

    closed = bars.copy()
    closed.attrs.update(bars.attrs)
    closed.loc[ts, ["low", "close"]] = [edge - 3.0, edge - 2.5]   # closes 2.5 inside
    sigs = generate_s1(closed, lv, MES)
    assert all(s.context["break_time"] != "10:00:00" for s in sigs), \
        [(s.signal_time.time(), s.context["break_time"]) for s in sigs]
    print(f"  wick 3.5 inside -> signal, stop {edge - 4.5:.2f}; close 2.5 inside -> break failed")


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


def test_s1_news_events_are_the_amended_list():
    """Amended 2026-09-15: PPI and retail sales join FOMC, CPI and NFP. Pinned
    so the list cannot drift without a test — and a restarted S1 sample."""
    assert set(S1_NEWS_EVENTS) == {"FOMC", "CPI", "NFP", "PPI", "RETAIL_SALES"}, S1_NEWS_EVENTS

    bars = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    d = S1_LOOKAHEAD_DATE
    lv = {x.date: x for x in build_sessions(bars, tick=TICK)}[d]
    for event in ("PPI", "RETAIL_SALES"):
        cal = NewsCalendar(start=d, end=d, events=frozenset(S1_NEWS_EVENTS),
                           days={d: frozenset({event})})
        sigs = generate_s1(bars, lv, MES, news=cal)
        assert all(s.signal_time.time() >= time(10, 30) for s in sigs), \
            (event, [s.signal_time.time() for s in sigs])
    print("  PPI and retail-sales days stand S1 down before 10:30")


# --- IB_FAIL: the break whose retest fails -----------------------------------

def ib_fail_bars(spike_high=5013.0, retest_low=None, break_delta=None):
    """IB 4989.5-5010.5; upside break closes 5012 at 10:00; 10:01 spikes to
    `spike_high` (and, if given, wicks down to `retest_low`); 10:02 closes
    5007.5 (3 points back inside, low 5007.0); then price falls through the IB.
    `break_delta` puts a new cumulative-delta high on the break bar."""
    d0, d1 = date(2026, 3, 2), S1_LOOKAHEAD_DATE
    ib = np.concatenate([np.linspace(5000, 5010, 8), np.linspace(5010, 4990, 15),
                         np.linspace(4990, 5002, 7)])
    path = np.concatenate([ib, [5012.0, 5011.5, 5007.5], np.linspace(5007, 4985, 30)])
    path = np.concatenate([path, np.full(390 - len(path), 4985.0)])
    day = bars_from_path(d1, path, overnight_center=5000.0)
    day.loc[_bar_at(day, d1, 10, 1), "high"] = spike_high
    if retest_low is not None:
        day.loc[_bar_at(day, d1, 10, 1), "low"] = retest_low
    if break_delta is not None:
        rth = (day.index.date == d1) & (day.index.time >= time(9, 30))
        deltas = np.full(390, -20.0)
        deltas[:30] = 100.0
        deltas[30] = break_delta
        vol = day.loc[rth, "volume"].to_numpy()
        day.loc[rth, "buy_volume"] = (vol + deltas) / 2
        day.loc[rth, "sell_volume"] = (vol - deltas) / 2
    return load_dataframe_bars(pd.concat([prior_balance_day(d0), day]).sort_index(),
                               source="FUTURES")


def _sessions(bars):
    return {x.date: x for x in build_sessions(bars, tick=TICK)}


def _ib_day(after_ib, deltas_after=None):
    """IB 4989.5-5010.5 (09:30-10:00), then `after_ib` closes from 10:00.
    Cumulative delta climbs +100/bar through the IB; `deltas_after` sets the
    bar deltas from 10:00 on (default -20)."""
    d0, d1 = date(2026, 3, 2), S1_LOOKAHEAD_DATE
    ib = np.concatenate([np.linspace(5000, 5010, 8), np.linspace(5010, 4990, 15),
                         np.linspace(4990, 5002, 7)])
    path = np.concatenate([ib, after_ib])
    path = np.concatenate([path, np.full(390 - len(path), path[-1])])
    day = bars_from_path(d1, path, overnight_center=5000.0)
    rth = (day.index.date == d1) & (day.index.time >= time(9, 30))
    deltas = np.full(390, -20.0)
    deltas[:30] = 100.0
    if deltas_after is not None:
        deltas[30:30 + len(deltas_after)] = deltas_after
    vol = day.loc[rth, "volume"].to_numpy()
    day.loc[rth, "buy_volume"] = (vol + deltas) / 2
    day.loc[rth, "sell_volume"] = (vol - deltas) / 2
    return load_dataframe_bars(pd.concat([prior_balance_day(d0), day]).sort_index(),
                               source="FUTURES")


def test_s1_one_trade_per_ib_edge_per_session():
    """Amended 2026-09-15: price wobbling back across the IB high makes a new
    crossing, but the edge has had its S1 trade. On real August 2026 tape the
    old rule sold one edge six times in an hour, several positions at once."""
    after = [5012.0, 5011.0, 5011.0, 5009.5,        # break, retest (signal), dip back inside
             5012.0, 5011.0, 5011.0]                 # re-break and a second clean retest
    deltas = [500.0, -50.0, -50.0, -50.0, 800.0, -50.0, -50.0]
    bars = _ib_day(np.concatenate([after, np.linspace(5011, 5040, 30)]), deltas)
    lv = _sessions(bars)[S1_LOOKAHEAD_DATE]
    sigs = generate_s1(bars, lv, MES)
    assert len(sigs) == 1 and sigs[0].signal_time.time() == time(10, 1), \
        [(s.signal_time.time(), s.context["break_time"]) for s in sigs]
    print("  break at 10:00 and re-break at 10:04 on the same edge -> one S1 signal (10:01)")


def test_ib_fail_one_per_ib_edge_per_session():
    after = [5012.0, 5007.5,                         # break, fails 3 points back inside
             5012.0, 5007.5]                         # re-break, fails again
    bars = _ib_day(np.concatenate([after, np.linspace(5007, 4985, 30)]))
    lv = _sessions(bars)[S1_LOOKAHEAD_DATE]
    fails = generate_ib_fail(bars, lv, MES)
    assert len(fails) == 1 and fails[0].signal_time.time() == time(10, 1), \
        [f.signal_time.time() for f in fails]
    print("  two failed breaks of the IB high -> one IB_FAIL (10:01)")


def test_ib_fail_enters_when_the_retest_fails():
    """Entry: stop order 1 tick beyond the failure bar's far end. Stop: 1 point
    beyond the false break's extreme. Floor 4, cap 8."""
    bars = ib_fail_bars()
    d = S1_LOOKAHEAD_DATE
    sigs = generate_ib_fail(bars, _sessions(bars)[d], MES)
    assert len(sigs) == 1, [s.signal_time.time() for s in sigs]
    s = sigs[0]
    assert (s.setup, s.direction, s.entry_style) == ("IB_FAIL", "SHORT", "STOP"), s
    assert s.signal_time == _bar_at(bars, d, 10, 2), s.signal_time
    assert abs(s.entry_px - (5007.0 - TICK)) < 1e-9, s.entry_px
    assert abs(s.stop_px - (5013.0 + IB_FAIL_STOP_BEYOND_EXTREME_PTS)) < 1e-9, s.stop_px
    assert IB_FAIL_STOP_FLOOR_PTS <= s.risk_pts <= IB_FAIL_STOP_CAP_PTS, s.risk_pts
    assert set(s.checklist) == {"ib_range_in_band", "break_in_window", "failed_in_window",
                                "stop_within_cap", "no_news_stand_down"}, sorted(s.checklist)
    assert s.context["break_time"] == "10:00:00" and "gap_pct" in s.context
    print(f"  SHORT stop-entry {s.entry_px} stop {s.stop_px} risk {s.risk_pts:.2f} "
          f"@ {s.signal_time.time()}")


def test_ib_fail_structure_beyond_cap_is_no_trade():
    bars = ib_fail_bars(spike_high=5016.0)          # stop 5017, entry 5006.75: 10.25 pts
    assert generate_ib_fail(bars, _sessions(bars)[S1_LOOKAHEAD_DATE], MES) == []
    print("  10.25-point structure -> no trade, never a widened cap")


def test_ib_fail_returns_nothing_when_gate_fails():
    flat = np.concatenate([np.full(30, 5000.0), np.linspace(5000, 5060, 360)])
    bars, lv, d1 = build_two_days(flat)
    assert lv[d1].s1_gate() is False
    assert generate_ib_fail(bars, lv[d1], MES) == []
    print("  out-of-band IB -> []")


def test_ib_fail_and_s1_never_take_the_same_break():
    """S1 takes the retest that holds; IB_FAIL the one that fails. One break,
    at most one of the two."""
    held = s1_quick_retest_bars(break_delta=500.0,
                                retest_deltas=[-50.0, -50.0, -50.0, -50.0, -50.0])
    failed = ib_fail_bars()
    for bars in (held, failed):
        lv = _sessions(bars)[S1_LOOKAHEAD_DATE]
        s1 = {s.context["break_time"] for s in generate_s1(bars, lv, MES)}
        fail = {s.context["break_time"] for s in generate_ib_fail(bars, lv, MES)}
        assert not s1 & fail, (s1, fail)
        assert s1 or fail, "each fixture should produce one of the two"
    lv = _sessions(held)[S1_LOOKAHEAD_DATE]
    assert generate_s1(held, lv, MES) and not generate_ib_fail(held, lv, MES)
    print("  held retest -> S1 only; failed retest -> IB_FAIL only")


def test_break_too_wide_for_s1_can_still_fail():
    """A retest whose swing is too wide for S1's cap ends S1's interest, not
    the break: if it then closes back inside, that is an IB_FAIL."""
    bars = ib_fail_bars(retest_low=5003.5, break_delta=500.0)   # S1 stop 5002.5: 8.25 pts
    lv = _sessions(bars)[S1_LOOKAHEAD_DATE]
    assert generate_s1(bars, lv, MES) == [], "S1 must stand down on the wide swing"
    fails = generate_ib_fail(bars, lv, MES)
    assert len(fails) == 1 and fails[0].signal_time.time() == time(10, 2), \
        [s.signal_time.time() for s in fails]
    print("  S1 capped out at 10:01; the 10:02 failure is still an IB_FAIL")


def test_ib_fail_news_day_stands_down_before_1030():
    bars = ib_fail_bars()
    d = S1_LOOKAHEAD_DATE
    lv = _sessions(bars)[d]
    cal = NewsCalendar(start=d, end=d, events=frozenset(S1_NEWS_EVENTS),
                       days={d: frozenset({"CPI"})})
    assert generate_ib_fail(bars, lv, MES, news=cal) == []
    unknown = generate_ib_fail(bars, lv, MES)
    assert unknown[0].checklist["no_news_stand_down"] is None
    print("  CPI day: 10:02 failure stood down; no calendar -> None")


def test_ib_fail_signals_survive_truncation():
    assert_signals_survive_truncation(generate_ib_fail, ib_fail_bars(), S1_LOOKAHEAD_DATE)
    print("  IB_FAIL reproducible from the tape cut at its own bar")


def test_ib_fail_flows_through_run_all_and_validates():
    bars = ib_fail_bars()
    log = fills_to_log(run_all(bars, build_sessions(bars, tick=TICK), MES), MES)
    assert "IB_FAIL" in set(log["setup"]), log["setup"].tolist()
    res = validate(log)
    assert res.ok, res.errors
    row = log[log["setup"] == "IB_FAIL"].iloc[0]
    print(f"  IB_FAIL {row['direction']} {row['entry_px']} -> {row['exit_reason']} "
          f"{row['exit_px']}")


# --- SPY: ES-point thresholds scaled by the day (added 2026-09-15) -----------

def to_spy(futures_bars, scale=10.0):
    """The same tape as a SPY proxy: RTH only, prices / `scale`, no delta."""
    t = futures_bars.index.time
    rth = futures_bars[(t >= time(9, 30)) & (t < time(16, 0))].copy()
    rth[["open", "high", "low", "close"]] /= scale
    rth[["buy_volume", "sell_volume"]] = np.nan
    return load_dataframe_bars(rth, source="SPY")


def _spy_sessions(bars, ratio=10.0):
    """Sessions whose S&P reference close on each prior day is `ratio` x SPY's."""
    import warnings
    closes = bars.groupby(bars.index.date)["close"].last()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {s.date: s for s in build_sessions(
            bars, tick=SPY.tick, reference_closes={d: c * ratio for d, c in closes.items()})}


def test_spy_s1_and_ib_fail_thresholds_scale_by_the_day():
    """With a 1:10 ratio, S1's 4-point stop floor is 0.40 on SPY, and IB_FAIL's
    stop sits 0.10 (not 1.00) beyond the false break."""
    d = S1_LOOKAHEAD_DATE
    bars = to_spy(s1_quick_retest_bars(break_delta=500.0,
                                       retest_deltas=[-50.0] * 5))
    lv = _spy_sessions(bars)[d]
    assert abs(lv.point_scale - 0.1) < 1e-12
    s1 = generate_s1(bars, lv, SPY)
    assert s1 and abs(s1[0].risk_pts - S1_STOP_FLOOR_PTS * 0.1) < 1e-9, \
        [(s.entry_px, s.stop_px) for s in s1]

    bars = to_spy(ib_fail_bars())
    lv = _spy_sessions(bars)[d]
    fail = generate_ib_fail(bars, lv, SPY)
    assert fail, "the scaled failure should still trade"
    assert abs(fail[0].stop_px - (501.3 + IB_FAIL_STOP_BEYOND_EXTREME_PTS * 0.1)) < 1e-9, \
        fail[0].stop_px
    assert fail[0].context["point_scale"] == lv.point_scale
    print(f"  S1 risk {s1[0].risk_pts:.2f}; IB_FAIL stop {fail[0].stop_px:.2f} "
          f"(risk {fail[0].risk_pts:.2f})")


def test_spy_s3_thresholds_scale_by_the_day():
    from mesproto.config import S3_STOP_CAP_PTS, S3_STOP_FLOOR_PTS
    bars = to_spy(build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)[0])
    lv = _spy_sessions(bars)[date(2026, 3, 3)]
    sigs = generate_s3(bars, lv, SPY)
    assert sigs, "the scaled S3 day should still signal"
    for s in sigs:
        assert S3_STOP_FLOOR_PTS * 0.1 - 1e-9 <= s.risk_pts <= S3_STOP_CAP_PTS * 0.1 + 1e-9, \
            s.risk_pts
    print(f"  S3 on SPY: risk {sigs[0].risk_pts:.2f} within 0.50-1.00")


def test_spy_sessions_without_a_ratio_generate_nothing():
    """No reference close for the prior day: every point threshold is unknown,
    so nothing is generated — never thresholds in the wrong units."""
    import warnings
    d = S1_LOOKAHEAD_DATE
    for bars, gen in ((to_spy(s1_quick_retest_bars(break_delta=500.0,
                                                   retest_deltas=[-50.0] * 5)), generate_s1),
                      (to_spy(ib_fail_bars()), generate_ib_fail),
                      (to_spy(build_two_days(s3_path(), day2_vol=s3_volumes(),
                                             day2_on=5040.0)[0]), generate_s3)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lv = {s.date: s for s in build_sessions(bars, tick=SPY.tick)}[d]
        assert lv.point_scale is None
        assert gen(bars, lv, SPY) == [], gen.__name__
    print("  S1, IB_FAIL, S3 -> [] without a ratio")


def test_s3_checklist_maps_every_protocol_condition():
    """Day gate (3 conditions + VWAP-chop stand-down) and trigger, each named."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = generate_s3(bars, lv[d1], MES)
    assert sigs
    expected = {"opened_outside_value", "one_timeframing", "delta_at_extreme",
                "vwap_crosses_ok", "pullback_to_level", "volume_declining",
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


def test_s3_pullback_to_developing_value_area_edge():
    """§05: 'Pullback to session VWAP ± 2 points, or to the developing
    value-area edge'. The edge is today's value-area high for a long, from
    the session profile through the signal bar; a LIMIT entry rests there."""
    from mesproto.config import S3_VA_EDGE_TOLERANCE_PTS, S3_VWAP_TOLERANCE_PTS
    from mesproto.levels import rth_slice, volume_profile
    bars, lv, d1 = build_two_days(s3_va_edge_path(), day2_vol=s3_va_edge_volumes(),
                                  day2_on=5040.0)
    s = lv[d1]
    assert s.s3_gate() is True, "fixture must pass the day gate"
    sigs = generate_s3(bars, s, MES)
    assert sigs, "a pullback to the developing VAH should signal"
    sig = sigs[0]
    rth = rth_slice(bars, d1)
    i = rth.index.get_loc(sig.signal_time)
    vah = volume_profile(rth.iloc[:i + 1], tick=TICK).vah
    assert abs(rth["low"].iloc[i] - s.vwap.iloc[i]) > S3_VWAP_TOLERANCE_PTS, "not a VWAP touch"
    assert abs(rth["low"].iloc[i] - vah) <= S3_VA_EDGE_TOLERANCE_PTS
    assert sig.context["pullback_level"] == "VA_EDGE", sig.context
    assert sig.checklist["pullback_to_level"] is True
    assert abs(sig.entry_px - vah) < 1e-9, (sig.entry_px, vah)
    assert_signals_survive_truncation(generate_s3, bars, d1)
    print(f"  {sig.signal_time.time()}: low {rth['low'].iloc[i]:.2f}, developing VAH {vah:.2f}, "
          f"VWAP {s.vwap.iloc[i]:.2f} -> LIMIT at {sig.entry_px:.2f}")


def _first_thin_level_beyond(rth_through_signal, anchor, direction):
    """Independent of production: walk tick by tick from `anchor` away from
    the trade until a level has under S3_LVN_MAX_FRAC_OF_POC of the POC's volume."""
    from mesproto.config import S3_LVN_MAX_FRAC_OF_POC
    from mesproto.levels import volume_profile
    prof = volume_profile(rth_through_signal, tick=TICK)
    vols = {int(round(p / TICK)): v for p, v in prof.bins.items()}
    step = -1 if direction == "LONG" else 1
    k = int(round(anchor / TICK)) + step
    while vols.get(k, 0.0) >= S3_LVN_MAX_FRAC_OF_POC * max(vols.values()):
        k += step
    return k * TICK


def test_s3_stop_goes_beyond_the_low_volume_node():
    """§05: 'Beyond the low-volume node under the pullback.' The node is the
    first price past the pullback (and entry) where under 25% as much has
    traded today as at the busiest price; the stop sits 1 tick beyond it."""
    from mesproto.levels import rth_slice
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sig = generate_s3(bars, lv[d1], MES)[0]
    rth = rth_slice(bars, d1)
    i = rth.index.get_loc(sig.signal_time)
    pullback_low = rth["low"].iloc[rth.index.get_loc(sig.signal_time.replace(hour=11, minute=0)):i + 1].min()
    assert pullback_low >= sig.entry_px, "fixture: the pullback stays above the VWAP limit"
    node = _first_thin_level_beyond(rth.iloc[:i + 1], sig.entry_px, "LONG")
    assert abs(sig.stop_px - (node - TICK)) < 1e-9, (sig.stop_px, node)
    assert 5.0 <= sig.risk_pts <= 10.0, sig.risk_pts
    print(f"  entry {sig.entry_px:.2f}, low-volume node {node:.2f} -> stop {sig.stop_px:.2f} "
          f"({sig.risk_pts:.2f} pts)")


def test_s3_no_low_volume_node_within_cap_is_no_trade():
    """Value built all the way down 5060-5075: nothing thin within 10 points of
    a 5074 entry. Structure beyond the cap is no trade, never a widened stop."""
    bars, lv, d1 = build_two_days(s3_va_edge_path(value_from=5060.0),
                                  day2_vol=s3_va_edge_volumes(), day2_on=5040.0)
    assert lv[d1].s3_gate() is True
    assert generate_s3(bars, lv[d1], MES) == []
    print("  nearest thin price ~15 pts below entry -> no trade")


def test_s3_price_already_at_a_level_is_not_a_pullback():
    """On a steady climb the developing value-area high sits at the session
    high. A bar there has not pulled back to anything: price must first be
    away from a level before coming back to it counts."""
    bars, lv, d1 = build_two_days(s3_path(), day2_vol=s3_volumes(), day2_on=5040.0)
    sigs = generate_s3(bars, lv[d1], MES)
    assert sigs, "the VWAP pullback should still signal"
    assert sigs[0].context["pullback_level"] == "VWAP", \
        [(s.signal_time.time(), s.context["pullback_level"], s.entry_px) for s in sigs]
    print(f"  first signal {sigs[0].signal_time.time()} at VWAP {sigs[0].entry_px:.2f}, "
          f"not at the session high")


def s3_hug_bars():
    """Pullback reaches VWAP at ~11:20 and sits on it for 25 quiet bars."""
    rise, pull = np.linspace(5040, 5100, 90), np.linspace(5100, 5070, 25)
    path = np.concatenate([rise, pull, np.full(25, 5070.0), np.linspace(5070, 5130, 80)])
    path = np.concatenate([path, np.full(390 - len(path), 5130.0)])
    vol = np.full(390, 1500.0)
    vol[90:115] = 400.0
    vol[115:140] = 150.0
    return build_two_days(path, day2_vol=vol, day2_on=5040.0)


def test_s3_level_rising_to_meet_price_is_not_a_pullback():
    """After 12:10 price sits flat at 5130 and the developing value-area high
    climbs up to it. Price never came back to the level — the level came to
    price — so the afternoon gives no signal."""
    bars, lv, d1 = build_two_days(s3_va_edge_path(value_from=5060.0),
                                  day2_vol=s3_va_edge_volumes(), day2_on=5040.0)
    late = [(s.signal_time.time(), s.context["pullback_level"], s.entry_px)
            for s in generate_s3(bars, lv[d1], MES) if s.signal_time.time() >= time(12, 10)]
    assert late == [], late
    print("  flat afternoon, VAH rising into price -> no signal")


def test_s3_one_signal_per_pullback():
    """Sitting on VWAP is one pullback, not a trade every few bars. After a
    signal, price must leave the level before another pullback can count."""
    bars, lv, d1 = s3_hug_bars()
    sigs = generate_s3(bars, lv[d1], MES)
    assert len(sigs) == 1, [(s.signal_time.time(), round(s.entry_px, 2)) for s in sigs]
    print(f"  25 bars on VWAP -> one signal at {sigs[0].signal_time.time()}")


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
