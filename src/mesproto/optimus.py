"""Read an Optimus Flow (Quantower) CSV export and check it against our bars.

Optimus Flow is a desktop platform — a Quantower white label on a Rithmic
feed — with no API to query, so the comparison goes through its History
Exporter file. Quantower documents neither the separator nor the column names,
and they differ between panels and builds, so `load_optimus_export` sniffs the
separator and matches columns by alias, then reports what it found.

Why bother, when `levels.aggressor_convention` already reads the side
convention out of the Databento tape: the hand-logged setups (S2, S4) are read
in Optimus Flow. If its delta ran the other way, hand-logged and generated
trades would disagree on the one condition they share, and pooling or even
comparing them would be meaningless.
"""

from __future__ import annotations

import csv
from typing import Optional

import numpy as np
import pandas as pd

from .config import ET, EXTERNAL_DELTA_MIN_CORR, EXTERNAL_MIN_BARS

# lower-cased header -> our name. Quantower writes these differently depending
# on the panel: a chart export, the History Exporter and the footprint panel
# all spell volume analysis their own way.
_TIMESTAMP = ("datetime", "date time", "timestamp", "time stamp", "date/time")
_DATE, _TIME = ("date",), ("time",)
_ALIASES = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "last"),
    "volume": ("volume", "vol", "total volume"),
    "delta": ("delta", "volume delta", "cumulative delta bar", "bar delta"),
    "ask_volume": ("ask volume", "askvolume", "buy volume", "ask", "bought volume"),
    "bid_volume": ("bid volume", "bidvolume", "sell volume", "bid", "sold volume"),
}


def _sniff_separator(path: str) -> str:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_optimus_export(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Bars from an Optimus Flow / Quantower CSV export, indexed in ET.

    `tz` is the timezone the file's timestamps are *in* — the platform writes
    its display time, which is not necessarily ET. Delta comes from a delta
    column when there is one, else from ask minus bid volume (volume traded at
    the ask is buyer-initiated). Raises if the file has no usable timestamp or
    no volume: a silent guess here would be compared against real bars.
    """
    raw = pd.read_csv(path, sep=_sniff_separator(path), encoding="utf-8-sig")
    lower = {str(c).strip().lower(): c for c in raw.columns}

    stamp = next((lower[a] for a in _TIMESTAMP if a in lower), None)
    if stamp is not None:
        when = pd.to_datetime(raw[stamp].astype(str).str.strip(), format="mixed")
    elif any(a in lower for a in _DATE) and any(a in lower for a in _TIME):
        d = raw[next(lower[a] for a in _DATE if a in lower)].astype(str).str.strip()
        t = raw[next(lower[a] for a in _TIME if a in lower)].astype(str).str.strip()
        when = pd.to_datetime(d + " " + t, format="mixed")
    else:
        raise ValueError(
            f"{path}: no timestamp column (looked for {_TIMESTAMP} or a date and "
            f"a time column); found {list(raw.columns)}")

    found = {ours: lower[alias] for ours, aliases in _ALIASES.items()
             for alias in aliases if alias in lower}
    if "volume" not in found:
        raise ValueError(f"{path}: no volume column; found {list(raw.columns)}")

    out = pd.DataFrame(index=pd.DatetimeIndex(when))
    for name in ("open", "high", "low", "close", "volume"):
        if name in found:
            out[name] = pd.to_numeric(raw[found[name]], errors="coerce").to_numpy()
    if "delta" in found:
        out["delta"] = pd.to_numeric(raw[found["delta"]], errors="coerce").to_numpy()
    elif "ask_volume" in found and "bid_volume" in found:
        ask = pd.to_numeric(raw[found["ask_volume"]], errors="coerce").to_numpy()
        bid = pd.to_numeric(raw[found["bid_volume"]], errors="coerce").to_numpy()
        out["delta"] = ask - bid

    if out.index.tz is None:
        out.index = out.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer")
    out.index = out.index.tz_convert(ET)
    out.attrs["columns_found"] = {k: str(v) for k, v in found.items()}
    return out.sort_index()


def compare_delta(theirs: pd.DataFrame, ours: pd.DataFrame,
                  min_bars: int = EXTERNAL_MIN_BARS,
                  min_corr: float = EXTERNAL_DELTA_MIN_CORR) -> dict:
    """Compare an export against our bars on the minutes they share.

    Returns the overlap, the share of those bars whose volume matches, the
    correlation between the two deltas, and a verdict: SAME (their delta runs
    with ours), INVERTED (against it — every delta condition would flip), or
    INCONCLUSIVE. Too little overlap is INSUFFICIENT, never a verdict.
    """
    joined = theirs.join(ours, how="inner", lsuffix="_them", rsuffix="_us")
    bars = len(joined)
    out: dict = {"bars": bars, "verdict": "INSUFFICIENT", "correlation": float("nan"),
                 "volume_agreement": float("nan")}
    if bars:
        vol = np.isclose(joined["volume_them"].to_numpy(dtype=float),
                         joined["volume_us"].to_numpy(dtype=float))
        out["volume_agreement"] = float(vol.mean())
    if bars < min_bars:
        return out
    if "delta_them" not in joined or "delta_us" not in joined:
        out["verdict"] = "NO_DELTA"
        return out

    them = joined["delta_them"].to_numpy(dtype=float)
    us = joined["delta_us"].to_numpy(dtype=float)
    ok = ~(np.isnan(them) | np.isnan(us))
    if ok.sum() < min_bars or them[ok].std() == 0 or us[ok].std() == 0:
        return out
    corr = float(np.corrcoef(them[ok], us[ok])[0, 1])
    out["correlation"] = corr
    out["verdict"] = ("SAME" if corr >= min_corr else
                      "INVERTED" if corr <= -min_corr else "INCONCLUSIVE")
    return out


def describe(theirs: pd.DataFrame, ours: pd.DataFrame, result: Optional[dict] = None) -> str:
    """The comparison as plain lines for the script to print."""
    result = compare_delta(theirs, ours) if result is None else result
    lines = [
        f"export: {len(theirs)} bars {theirs.index.min()} .. {theirs.index.max()}",
        f"  columns recognised: {theirs.attrs.get('columns_found', {})}",
        f"ours:   {len(ours)} bars {ours.index.min()} .. {ours.index.max()}",
        f"overlapping minutes: {result['bars']}",
    ]
    if result["bars"]:
        lines.append(f"volume agrees on {result['volume_agreement']:.1%} of them")
    verdict = {
        "SAME": "delta runs the SAME way in both — the hand-logged and generated "
                "conditions agree",
        "INVERTED": "!! delta is INVERTED between them: every delta condition in the "
                    "protocol would flip. Do not pool or compare the two until this "
                    "is settled",
        "INCONCLUSIVE": "delta correlation is too weak to call either way — check the "
                        "symbol, session and timezone of the export",
        "NO_DELTA": "the export carried no delta (or ask/bid volume) column; volume "
                    "alignment above is all this file can check",
        "INSUFFICIENT": f"not enough overlapping minutes (need {EXTERNAL_MIN_BARS}); "
                        f"check the date range and the --tz of the export",
    }[result["verdict"]]
    corr = result["correlation"]
    lines.append(f"verdict: {result['verdict']}" + ("" if np.isnan(corr) else f" (r={corr:+.3f})"))
    lines.append(f"  {verdict}")
    return "\n".join(lines)
