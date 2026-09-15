#!/usr/bin/env python3
"""Tests for mesproto.levels — synthetic sessions with known properties.

The load-bearing test is `test_no_lookahead`: it mutates the tape *after* the
classification cutoff and asserts nothing upstream of the cutoff changes.
"""
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mesproto.levels import (
    ET, build_sessions, load_dataframe_bars, sessions_to_frame,
    volume_profile, volume_profile_from_trades, _profile_from_bins,
)

TICK = 0.25


def make_session(d: date, path: np.ndarray, *, overnight=True,
                 on_center=5000.0, on_span=20.0, vol=1000.0, delta_bias=0.0):
    """Build 1-minute bars for one futures session from an RTH price path."""
    rows, idx = [], []
    if overnight:
        on_start = pd.Timestamp.combine(d - timedelta(days=1), time(18, 0)).tz_localize(ET)
        n_on = 15 * 60
        rng = np.random.default_rng(int(d.strftime("%Y%m%d")))
        on_path = on_center + np.cumsum(rng.normal(0, 0.05, n_on))
        on_path = np.clip(on_path, on_center - on_span / 2, on_center + on_span / 2)
        for i, p in enumerate(on_path):
            idx.append(on_start + pd.Timedelta(minutes=i))
            rows.append([p, p + 0.25, p - 0.25, p, vol * 0.2, np.nan, np.nan])

    rth_start = pd.Timestamp.combine(d, time(9, 30)).tz_localize(ET)
    for i, p in enumerate(path):
        idx.append(rth_start + pd.Timedelta(minutes=i))
        bv = vol * (0.5 + delta_bias)
        rows.append([p, p + 0.5, p - 0.5, p, vol, bv, vol - bv])

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                      columns=["open", "high", "low", "close", "volume",
                               "buy_volume", "sell_volume"])
    return df


def trend_path(start=5000.0, minutes=390, slope=0.03):
    """Monotonic climb — one-timeframes on every 30-min bar."""
    return start + slope * np.arange(minutes)


def balance_path(center=5000.0, minutes=390, amp=6.0, period=45):
    """Oscillation around a center — no one-timeframing, many VWAP crosses."""
    return center + amp * np.sin(2 * np.pi * np.arange(minutes) / period)


def build(frames, source="FUTURES"):
    bars = pd.concat(frames).sort_index()
    return load_dataframe_bars(bars, source=source)


# ---------------------------------------------------------------------------

def test_value_area_covers_target():
    # symmetric triangular distribution centered at 100.0
    vols = np.array([1, 2, 3, 5, 8, 13, 21, 13, 8, 5, 3, 2, 1], dtype=float)
    prices = np.round(99.0 + 0.25 * np.arange(len(vols)), 2)  # 99.00 .. 102.00
    p = _profile_from_bins(pd.Series(vols, index=prices), 0.70)
    covered = p.bins[(p.bins.index >= p.val) & (p.bins.index <= p.vah)].sum()
    frac = covered / p.bins.sum()
    assert p.poc == 100.5, p.poc  # peak of 21 sits at index 6
    assert 0.70 <= frac <= 0.85, frac
    assert p.val < p.poc < p.vah
    print(f"  value area {frac:.1%} of volume, POC={p.poc}, "
          f"VA=[{p.val}, {p.vah}]")


def test_profile_from_trades_matches_bins():
    price = np.array([100.0, 100.0, 100.25, 99.75, 100.0])
    size = np.array([10.0, 5.0, 3.0, 2.0, 7.0])
    p = volume_profile_from_trades(price, size, tick=0.25)
    assert p.poc == 100.0
    assert p.total_volume == 27.0
    print(f"  trade profile POC={p.poc}, total={p.total_volume}")


def test_trend_day_classified_and_gates():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    # prior day balances near 5000; next day opens well above prior value
    prior = make_session(d0, balance_path(5000.0))
    trend = make_session(d1, trend_path(5040.0, slope=0.05), on_center=5040.0)
    sessions = build_sessions(build([prior, trend]), tick=TICK)
    s = [x for x in sessions if x.date == d1][0]

    assert s.day_type == "TREND_UP", s.day_type
    assert s.opened_inside_value is False
    assert s.otf_up_bars >= 2, s.otf_up_bars
    assert s.s3_gate() is True
    assert s.s2_gate() is False
    print(f"  {d1}: {s.day_type}, otf_up={s.otf_up_bars}, "
          f"crosses={s.vwap_crosses}, s3_gate={s.s3_gate()}")


def test_balance_day_classified():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(5000.0))
    today = make_session(d1, balance_path(5000.0, amp=5.0, period=37))
    sessions = build_sessions(build([prior, today]), tick=TICK)
    s = [x for x in sessions if x.date == d1][0]

    assert s.day_type in ("BALANCE", "DOUBLE_DIST"), s.day_type
    assert s.opened_inside_value is True
    assert s.s3_gate() is False
    print(f"  {d1}: {s.day_type}, opened_inside={s.opened_inside_value}, "
          f"crosses={s.vwap_crosses}, s2_gate={s.s2_gate()}")


def test_ib_range_exact():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    path = trend_path(5000.0, slope=0.10)  # 0.10/min -> 30 min = 3.0 pts + wick
    prior = make_session(d0, balance_path(5000.0))
    today = make_session(d1, path)
    s = [x for x in build_sessions(build([prior, today]), tick=TICK)
         if x.date == d1][0]
    # first 30 bars span 5000.00 .. 5002.90, wicks +/- 0.5
    assert abs(s.ib_high - 5003.40) < 1e-6, s.ib_high
    assert abs(s.ib_low - 4999.50) < 1e-6, s.ib_low
    assert abs(s.ib_range - 3.90) < 1e-6, s.ib_range
    print(f"  IB = [{s.ib_low}, {s.ib_high}] range={s.ib_range}")


def test_no_lookahead():
    """Mutating the tape after the cutoff must not change anything at the cutoff."""
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(5000.0))
    base_path = balance_path(5000.0, amp=5.0, period=37)
    today = make_session(d1, base_path)
    bars = build([prior, today])
    before = [x for x in build_sessions(bars, tick=TICK) if x.date == d1][0]

    # violently spike everything after 11:00 ET — a huge afternoon trend
    mutated = bars.copy()
    cutoff_ts = pd.Timestamp.combine(d1, time(11, 0)).tz_localize(ET)
    mask = (mutated.index >= cutoff_ts) & (mutated.index.date == d1)
    bump = np.arange(mask.sum()) * 0.5
    for col in ["open", "high", "low", "close"]:
        mutated.loc[mask, col] = mutated.loc[mask, col].to_numpy() + bump
    mutated.attrs.update(bars.attrs)
    after = [x for x in build_sessions(mutated, tick=TICK) if x.date == d1][0]

    assert before.day_type == after.day_type, (before.day_type, after.day_type)
    assert before.vwap_crosses == after.vwap_crosses
    assert before.otf_up_bars == after.otf_up_bars
    assert before.otf_down_bars == after.otf_down_bars
    assert before.ib_range == after.ib_range
    assert before.opened_inside_value == after.opened_inside_value
    # sanity: the outcome fields SHOULD have changed
    assert after.rth_high > before.rth_high
    print(f"  day_type stable ({before.day_type}) while rth_high moved "
          f"{before.rth_high:.2f} -> {after.rth_high:.2f}")


def test_zero_width_overnight_range_has_no_position():
    """A range with no width has no 'where in the range'. 0.5 is a plausible
    substitute, which invariant 2 forbids."""
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    today = make_session(d1, trend_path(5060.0), on_center=5060.0)
    on = today.index < pd.Timestamp.combine(d1, time(9, 30)).tz_localize(ET)
    today.loc[on, ["open", "high", "low", "close"]] = 5060.0
    s = {x.date: x for x in build_sessions(
        build([make_session(d0, balance_path(5000.0)), today]), tick=TICK)}[d1]
    assert s.on_high == s.on_low == 5060.0
    assert s.on_range_pos is None, s.on_range_pos
    assert s.s5_gate() is False
    print("  flat overnight -> on_range_pos None")


def test_prior_close_and_gap():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(5000.0))
    today = make_session(d1, trend_path(5060.0), on_center=5060.0)
    sessions = {s.date: s for s in build_sessions(build([prior, today]), tick=TICK)}
    assert sessions[d0].prior_close is None, "first session has no prior close"
    assert sessions[d0].gap_pct is None
    assert sessions[d1].prior_close == sessions[d0].rth_close
    expected = (sessions[d1].open_px - sessions[d0].rth_close) / sessions[d0].rth_close
    assert abs(sessions[d1].gap_pct - expected) < 1e-12
    print(f"  prior_close={sessions[d1].prior_close:.2f} gap={sessions[d1].gap_pct:+.3%}")


def test_csv_loader_offsets_across_dst():
    """ET exports with UTC offsets change offset at DST. Both offset-carrying
    and naive (exchange-local) timestamps must land on the same ET clock."""
    import os
    import tempfile
    from mesproto.levels import load_csv_bars

    rows = "open,high,low,close,volume\n"
    with_offsets = ("timestamp," + rows
                    + "2026-03-06 09:30:00-05:00,1,2,0.5,1.5,10\n"
                    + "2026-03-09 09:30:00-04:00,1,2,0.5,1.5,10\n")
    naive = ("timestamp," + rows
             + "2026-03-06 09:30:00,1,2,0.5,1.5,10\n"
             + "2026-03-09 09:30:00,1,2,0.5,1.5,10\n")
    for body in (with_offsets, naive):
        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            bars = load_csv_bars(path, source="FUTURES")
        finally:
            os.remove(path)
        assert [ts.time() for ts in bars.index] == [time(9, 30), time(9, 30)], bars.index
        assert str(bars.index.tz) == "America/New_York"
    print("  offset-carrying and naive timestamps both land on 09:30 ET across DST")


def _tbbo_frame(rows):
    """rows: (ts_recv, ts_event, price, size, side, instrument_id)."""
    df = pd.DataFrame(rows, columns=["ts_recv", "ts_event", "price", "size", "side",
                                     "instrument_id"])
    for c in ("ts_recv", "ts_event"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df.set_index("ts_recv")


def test_tbbo_unsided_trades_are_not_buys():
    """Databento marks side N where the source gives none (auctions, some
    opening prints). Counting those as buy aggression biases every delta up."""
    from mesproto.levels import tbbo_to_bars
    t0 = "2026-03-03 14:30:10"
    trades = _tbbo_frame([
        (t0, t0, 5000.00, 5, "B", 1),
        (t0, t0, 5000.25, 3, "A", 1),
        (t0, t0, 5000.00, 40, "N", 1),
    ])
    bars = tbbo_to_bars(trades)
    bar = bars.iloc[0]
    assert bar["volume"] == 48, bar["volume"]
    assert bar["buy_volume"] == 5 and bar["sell_volume"] == 3, bar
    assert bars.attrs["has_delta"]
    print(f"  B=5 A=3 N=40 -> buy={bar['buy_volume']:.0f} sell={bar['sell_volume']:.0f} "
          f"volume={bar['volume']:.0f}")


def test_tbbo_bars_are_timed_by_ts_event():
    """to_df() indexes by ts_recv (capture server). The matching-engine time
    is ts_event; a trade at 09:30:59.999 must not slide into the 09:31 bar."""
    from mesproto.levels import tbbo_to_bars
    trades = _tbbo_frame([
        ("2026-03-03 14:31:00.004", "2026-03-03 14:30:59.999", 5000.0, 1, "B", 1),
        ("2026-03-03 14:31:10", "2026-03-03 14:31:10", 5001.0, 1, "B", 1),
    ])
    bars = tbbo_to_bars(trades)
    assert [ts.time() for ts in bars.index] == [time(9, 30), time(9, 31)], bars.index
    print("  trade stamped 09:30:59.999 by the exchange lands in the 09:30 bar")


def test_tbbo_rejects_unknown_aggressor_code():
    from mesproto.levels import tbbo_to_bars
    trades = _tbbo_frame([("2026-03-03 14:30:10", "2026-03-03 14:30:10", 5000.0, 1, "B", 1)])
    try:
        tbbo_to_bars(trades, sell_aggressor_code="S")
    except ValueError:
        print("  sell_aggressor_code must be A or B")
        return
    raise AssertionError("an unknown aggressor code would silently zero one side")


def test_tbbo_roll_inside_range_warns():
    import warnings
    from mesproto.levels import tbbo_to_bars
    trades = _tbbo_frame([
        ("2026-03-12 14:30:10", "2026-03-12 14:30:10", 5000.0, 1, "B", 101),
        ("2026-03-13 14:30:10", "2026-03-13 14:30:10", 5040.0, 1, "B", 202),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tbbo_to_bars(trades)
    assert any("roll" in str(x.message).lower() for x in w), [str(x.message) for x in w]
    print("  two instrument_ids in one tape -> roll warning")


def test_databento_symbology_guards_single_expiry():
    import warnings
    from mesproto.levels import databento_symbology

    assert databento_symbology("ESZ5") == ("ESZ5", "raw_symbol")
    assert databento_symbology(["MESH6"]) == ("MESH6", "raw_symbol")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert databento_symbology("ES.c.0") == ("ES.c.0", "continuous")
    assert any("continuous" in str(x.message).lower() for x in w), \
        "a continuous symbol must warn loudly"

    for bad in (["ESZ5", "ESH6"], "ES.FUT", "ESZ5-ESH6", "ESZ5,ESH6"):
        try:
            databento_symbology(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} mixes instruments into one tape and must be refused")
    print("  single expiry ok; continuous warns; parent/spread/multi refused")


def test_spy_has_no_overnight():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(500.0), overnight=False)
    today = make_session(d1, balance_path(500.0), overnight=False)
    bars = build([prior, today], source="SPY")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = [x for x in build_sessions(bars, tick=0.01) if x.date == d1][0]
        assert any("S5 is untestable" in str(x.message) for x in w)
    assert s.on_high is None and s.on_range_pos is None
    assert s.s5_gate() is False
    print("  SPY: overnight fields None, s5_gate False, warning raised")


# --- SPY point scaling (added 2026-09-15) -----------------------------------

def _spy_two_days():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(500.0), overnight=False)
    today = make_session(d1, trend_path(504.0, slope=0.003), overnight=False)
    return build([prior, today], source="SPY"), d0, d1


def _sessions_quiet(bars, **kw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {s.date: s for s in build_sessions(bars, **kw)}


def test_spy_point_scale_uses_prior_day_closes():
    """ES-point thresholds become SPY points by the ratio of the two closes on
    the PRIOR trading day — today's close is not known at 10:00."""
    bars, d0, d1 = _spy_two_days()
    spy_close_d0 = float(bars[bars.index.date == d0]["close"].iloc[-1])
    ref = {d0: spy_close_d0 * 10.05, d1: 99999.0}      # d1's own close must not be used
    s = _sessions_quiet(bars, tick=0.01, reference_closes=ref)
    assert s[d0].point_scale is None, "no prior day, no ratio"
    assert abs(s[d1].point_scale - 1 / 10.05) < 1e-12, s[d1].point_scale
    print(f"  SPY {spy_close_d0:.2f} / S&P {ref[d0]:.2f} -> scale {s[d1].point_scale:.5f}")


def test_spy_point_scale_is_none_without_a_reference_close():
    """Invariant 2: no reference close for the prior day means no ratio — never
    a guessed 1/10."""
    bars, d0, d1 = _spy_two_days()
    assert _sessions_quiet(bars, tick=0.01)[d1].point_scale is None
    assert _sessions_quiet(bars, tick=0.01,
                           reference_closes={d1: 5040.0})[d1].point_scale is None
    print("  no reference for the prior day -> point_scale None")


def test_futures_point_scale_is_one():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    bars = build([make_session(d0, balance_path(5000.0)), make_session(d1, trend_path(5040.0))])
    s = {x.date: x for x in build_sessions(bars, tick=TICK)}
    assert s[d0].point_scale == 1.0 and s[d1].point_scale == 1.0
    print("  futures: thresholds are already in ES points")


def test_load_reference_closes():
    import os
    import tempfile
    from mesproto.levels import load_reference_closes
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("# S&P 500 close (FRED SP500)\ndate,close\n2026-03-02,5012.5\n2026-03-03,5031.25\n")
        assert load_reference_closes(path) == {date(2026, 3, 2): 5012.5, date(2026, 3, 3): 5031.25}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("2026-03-04,n/a\n")
        try:
            load_reference_closes(path)
        except ValueError as e:
            assert ":5:" in str(e), e
        else:
            raise AssertionError("a malformed close must not be skipped")
    finally:
        os.remove(path)
    print("  comments/header skipped; malformed row raises with its line")


def test_futures_overnight_and_delta():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    prior = make_session(d0, balance_path(5000.0))
    today = make_session(d1, trend_path(5040.0), on_center=5040.0,
                         on_span=20.0, delta_bias=0.2)
    s = [x for x in build_sessions(build([prior, today]), tick=TICK)
         if x.date == d1][0]
    assert s.on_high is not None and s.on_low is not None
    assert 0.0 <= s.on_range_pos <= 1.0
    assert s.cum_delta is not None and s.cum_delta.iloc[-1] > 0
    print(f"  ON=[{s.on_low:.2f}, {s.on_high:.2f}] pos={s.on_range_pos:.2f} "
          f"cum_delta_close={s.cum_delta.iloc[-1]:,.0f}")


def test_frame_export():
    d0, d1 = date(2026, 3, 2), date(2026, 3, 3)
    sessions = build_sessions(build([make_session(d0, balance_path(5000.0)),
                                     make_session(d1, trend_path(5040.0),
                                                  on_center=5040.0)]), tick=TICK)
    df = sessions_to_frame(sessions)
    assert {"session_date", "ib_range_pts", "day_type", "s3_gate"} <= set(df.columns)
    print(f"  exported {len(df)} rows x {len(df.columns)} cols")
    print(df[["session_date", "day_type", "ib_range_pts", "opened_inside_value",
              "vwap_crosses", "s1_gate", "s3_gate"]].to_string(index=False))


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
