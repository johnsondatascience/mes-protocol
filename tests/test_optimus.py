#!/usr/bin/env python3
"""Tests for mesproto.optimus — reading an Optimus Flow (Quantower) CSV export
and checking its delta against ours.

Quantower does not document the export's columns or separator, and they differ
between panels and builds, so the loader sniffs both and says what it found.
"""
import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd

from mesproto.config import ET, EXTERNAL_DELTA_MIN_CORR, EXTERNAL_MIN_BARS
from mesproto.optimus import compare_delta, load_optimus_export


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_export_with_separate_date_and_time_and_semicolons():
    path = _write(
        "Date;Time;Open;High;Low;Close;Volume;Delta\n"
        "2026-08-03;09:30:00;7550.25;7551.00;7549.75;7550.50;1234;-56\n"
        "2026-08-03;09:31:00;7550.50;7552.00;7550.25;7551.75;2345;78\n")
    try:
        got = load_optimus_export(path)
    finally:
        os.remove(path)
    assert list(got.columns) == ["open", "high", "low", "close", "volume", "delta"]
    assert str(got.index.tz) == "America/New_York"
    assert got.index[0] == pd.Timestamp("2026-08-03 09:30", tz=ET)
    assert got["volume"].tolist() == [1234, 2345] and got["delta"].tolist() == [-56, 78]
    print(f"  {len(got)} bars, columns {list(got.columns)}")


def test_export_with_one_timestamp_column_and_ask_bid_volume():
    """Some panels write 'Ask volume' / 'Bid volume' instead of a delta: volume
    traded at the ask is buyer-initiated, so delta is ask minus bid."""
    path = _write(
        "DateTime,Open,High,Low,Close,Volume,Ask volume,Bid volume\n"
        "2026-08-03 09:30:00,7550.25,7551.0,7549.75,7550.5,1000,400,600\n"
        "2026-08-03 09:31:00,7550.5,7552.0,7550.25,7551.75,1000,700,300\n")
    try:
        got = load_optimus_export(path)
    finally:
        os.remove(path)
    assert got["delta"].tolist() == [-200, 400], got["delta"].tolist()
    print(f"  delta from ask/bid volume: {got['delta'].tolist()}")


def test_history_exporter_header_from_optimus_flow():
    """The real header from Optimus Flow's History Exporter (ESU6 Rithmic,
    1m): semicolon-separated with a trailing separator, the bar's own time in
    'Time left' and the bar's end in 'Time right', and no volume analysis."""
    path = _write(
        "Time left;Time right;Open;High;Median;Low;Close;Typical;Volume;"
        "Quote asset volume;Weighted;\n"
        "2026-08-12 13:30:00.000;2026-08-12 13:30:59.999;7765.5;7766.0;7765.25;"
        "7764.5;7765.75;7765.4;11725;0;7765.5\n"
        "2026-08-12 13:31:00.000;2026-08-12 13:31:59.999;7765.75;7767.0;7766.0;"
        "7765.0;7766.5;7766.1;8673;0;7766.2\n")
    try:
        got = load_optimus_export(path, tz="UTC")
    finally:
        os.remove(path)
    assert got.index[0] == pd.Timestamp("2026-08-12 09:30", tz=ET), got.index[0]
    assert "delta" not in got.columns, "this export carries no volume analysis"
    assert got["volume"].tolist() == [11725, 8673]
    assert got.attrs["columns_found"]["volume"] == "Volume"
    print(f"  'Time left' 13:30 UTC -> {got.index[0]:%H:%M %Z}, volume {got['volume'].tolist()}")


def test_export_timezone_is_explicit():
    """The export carries the platform's display time, not ours."""
    path = _write("DateTime,Open,High,Low,Close,Volume\n"
                  "2026-08-03 13:30:00,1,1,1,1,10\n")
    try:
        utc = load_optimus_export(path, tz="UTC")
    finally:
        os.remove(path)
    assert utc.index[0] == pd.Timestamp("2026-08-03 09:30", tz=ET), utc.index[0]
    print(f"  13:30 UTC -> {utc.index[0]:%H:%M %Z}")


def test_export_without_a_usable_timestamp_or_volume_raises():
    for text, want in (("Open,High,Low,Close,Volume\n1,1,1,1,5\n", "timestamp"),
                       ("DateTime,Open,High,Low,Close\n2026-08-03 09:30:00,1,1,1,1\n", "volume")):
        path = _write(text)
        try:
            load_optimus_export(path)
        except ValueError as e:
            assert want in str(e).lower(), e
        else:
            raise AssertionError(f"expected a {want} error")
        finally:
            os.remove(path)
    print("  missing timestamp or volume column raises")


def _bars(deltas, volume=1000, start="2026-08-03 09:30"):
    idx = pd.date_range(start, periods=len(deltas), freq="1min", tz=ET)
    return pd.DataFrame({"volume": float(volume), "delta": np.asarray(deltas, dtype=float)},
                        index=idx)


def test_compare_delta_reads_the_convention():
    rng = np.random.default_rng(7)
    deltas = rng.normal(0, 200, EXTERNAL_MIN_BARS + 40)
    ours = _bars(deltas)
    assert compare_delta(_bars(deltas), ours)["verdict"] == "SAME"
    assert compare_delta(_bars(-deltas), ours)["verdict"] == "INVERTED"
    noise = compare_delta(_bars(rng.normal(0, 200, len(deltas))), ours)
    assert noise["verdict"] == "INCONCLUSIVE", noise
    assert abs(compare_delta(_bars(deltas), ours)["correlation"] - 1.0) < 1e-9
    print(f"  same {compare_delta(_bars(deltas), ours)['correlation']:+.2f}, "
          f"inverted {compare_delta(_bars(-deltas), ours)['correlation']:+.2f}, "
          f"noise {noise['correlation']:+.2f} (need |r| >= {EXTERNAL_DELTA_MIN_CORR})")


def test_compare_delta_needs_enough_overlapping_bars():
    ours = _bars(np.arange(EXTERNAL_MIN_BARS + 10, dtype=float))
    few = compare_delta(ours.iloc[:EXTERNAL_MIN_BARS - 1], ours)
    assert few["verdict"] == "INSUFFICIENT" and few["bars"] == EXTERNAL_MIN_BARS - 1, few

    apart = compare_delta(_bars(np.arange(80, dtype=float), start="2026-08-04 09:30"), ours)
    assert apart["verdict"] == "INSUFFICIENT" and apart["bars"] == 0, apart
    print("  too few or non-overlapping bars -> INSUFFICIENT, never a verdict")


def test_compare_delta_reports_volume_agreement():
    """Rithmic and CME's own tape should agree on volume; a mismatch means the
    exports are not the same instrument, session or timezone."""
    ours = _bars(np.arange(EXTERNAL_MIN_BARS + 5, dtype=float), volume=1000)
    same = compare_delta(_bars(np.arange(EXTERNAL_MIN_BARS + 5, dtype=float), volume=1000), ours)
    assert same["volume_agreement"] == 1.0, same
    off = compare_delta(_bars(np.arange(EXTERNAL_MIN_BARS + 5, dtype=float), volume=1500), ours)
    assert off["volume_agreement"] == 0.0, off
    print(f"  volume agreement {same['volume_agreement']:.0%} vs {off['volume_agreement']:.0%}")


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
