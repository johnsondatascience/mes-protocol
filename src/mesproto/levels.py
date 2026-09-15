#!/usr/bin/env python3
"""
mesproto.levels — session levels for the MES replay protocol.

One interface over two very different data sources:

    SPY / equity 1-minute bars   -> free, no overnight session
    Databento TBBO (CME)         -> trades with aggressor side, full 23h session

Both produce a `SessionLevels` object exposing the same fields, so a rule
written against one runs unchanged against the other. Fields that a source
genuinely cannot supply are `None` — never silently zero or faked. Code that
consumes them must check.

    from mesproto.levels import load_csv_bars, load_databento_tbbo, build_sessions

    bars = load_csv_bars("spy_1min.csv", source="SPY")
    sessions = build_sessions(bars, tick=0.01)
    for s in sessions:
        print(s.date, s.day_type, s.ib_range, s.prior_vah, s.prior_val)

IMPORTANT — roll handling
-------------------------
Feed this ONE contract month at a time. Back-adjusted continuous series shift
historical prices by accumulated roll gaps, which silently corrupts every
level-based field here (value area, overnight range, IB edges). `load_databento_tbbo`
defaults to a single expiry symbol for that reason; if you pass a continuous
symbol like "ES.c.0", raw (unadjusted) stitching is what you want, and you must
still not compare levels across a roll boundary.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

from .config import (
    DAY_TYPE_CUTOFF, DOUBLE_DIST_RANGE_MULT, ET, IB_END, ON_OPEN,
    OTF_BAR_MINUTES, RTH_CLOSE, RTH_OPEN, S1_IB_RANGE_MAX_PCT,
    S1_IB_RANGE_MIN_PCT, S3_MAX_VWAP_CROSSES, S3_MIN_OTF_BARS,
    S5_ON_EXTREME_PCT, VALUE_AREA_PCT,
)

SourceKind = Literal["SPY", "FUTURES"]
DayType = Literal["TREND_UP", "TREND_DOWN", "BALANCE", "DOUBLE_DIST", "UNCLASSIFIED"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

REQUIRED_BAR_COLS = ["open", "high", "low", "close", "volume"]


def _finalize(df: pd.DataFrame, source: SourceKind) -> pd.DataFrame:
    """Common post-processing: ET index, sorted, typed, source tagged."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("bars must be indexed by a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError(
            "bar index is timezone-naive. Localize it explicitly — guessing the "
            "timezone of intraday data is how sessions end up off by an hour."
        )
    df = df.tz_convert(ET).sort_index()
    missing = [c for c in REQUIRED_BAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")
    if "buy_volume" not in df.columns:
        df["buy_volume"] = np.nan
        df["sell_volume"] = np.nan
    df.attrs["source"] = source
    df.attrs["has_delta"] = df["buy_volume"].notna().any()
    df.attrs["has_overnight"] = source == "FUTURES"
    return df


def load_csv_bars(
    path: str,
    source: SourceKind = "SPY",
    tz: str = "America/New_York",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Load OHLCV bars from CSV.

    Expects columns: timestamp, open, high, low, close, volume.
    `tz` is the timezone the timestamps are *in* — set it correctly; the
    default assumes exchange-local, which is what most free feeds ship.
    """
    df = pd.read_csv(path)
    raw = df[timestamp_col].astype(str)
    if pd.to_datetime(raw.iloc[:1]).dt.tz is not None:
        # offsets in the strings are authoritative, and they change at DST —
        # parse through UTC or pandas falls back to an object index
        idx = pd.DatetimeIndex(pd.to_datetime(raw, utc=True))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(raw)).tz_localize(
            ZoneInfo(tz), nonexistent="shift_forward", ambiguous="infer")
    df = df.drop(columns=[timestamp_col]).set_index(idx)
    return _finalize(df, source)


def load_dataframe_bars(df: pd.DataFrame, source: SourceKind = "SPY") -> pd.DataFrame:
    """Wrap an already-loaded bar frame (e.g. from an Alpaca or Polygon client)."""
    return _finalize(df.copy(), source)


_CONTINUOUS_SYMBOL = re.compile(r"^[A-Z0-9]+\.[cnv]\.\d+$")
_PARENT_SYMBOL = re.compile(r"^[A-Z0-9]+\.(FUT|OPT)$")


def databento_symbology(symbols: str | Sequence[str]) -> tuple[str, str]:
    """Validate a Databento request against invariant 6: one contract month.

    Returns (symbol, stype_in). Refuses anything that would interleave
    several instruments into one tape — multiple symbols, parent symbology
    (every expiry *and* every spread at once), calendar spreads (prices are
    spread values, not index levels). Continuous symbols (ES.c.0) are
    unadjusted, so levels are fine *within* one expiry but every level
    compared across the roll is wrong; that is allowed, loudly.
    """
    items = [symbols] if isinstance(symbols, str) else list(symbols)
    items = [p.strip() for s in items for p in str(s).split(",") if p.strip()]
    if len(items) != 1:
        raise ValueError(
            f"one contract month per request, got {items}. Bars are resampled "
            "from every trade in the response, so several instruments would be "
            "interleaved into one corrupt tape.")
    sym = items[0]
    if _PARENT_SYMBOL.match(sym):
        raise ValueError(
            f"{sym} is parent symbology: every expiry and spread in one stream. "
            "Request a single expiry such as ESZ5.")
    if "-" in sym or ":" in sym:
        raise ValueError(f"{sym} looks like a spread; its prices are not index levels.")
    if _CONTINUOUS_SYMBOL.match(sym):
        warnings.warn(
            f"!!! {sym} is a CONTINUOUS symbol. The series switches contract at "
            "each roll; any value area, overnight range or IB compared across "
            "that boundary is corrupt. Prefer a single expiry, and never pool "
            "levels across a roll. !!!",
            stacklevel=2,
        )
        return sym, "continuous"
    return sym, "raw_symbol"


def tbbo_to_bars(
    trades: pd.DataFrame, bar_minutes: int = 1, sell_aggressor_code: str = "A",
) -> pd.DataFrame:
    """Aggregate a Databento TBBO/trades frame (as from `DBNStore.to_df()`)
    into signed OHLCV bars.

    Separate from the download so a file you already paid for can be
    converted without re-requesting it, and so the conventions below are
    testable offline:

    * Bars are timed by `ts_event` (matching engine) when present. `to_df()`
      indexes by `ts_recv`, which lags and can push a trade into the next bar.
    * Only trades whose side is the buy or sell aggressor code count toward
      delta. Side 'N' (no aggressor: auctions, some opening prints) is
      counted in volume but in neither buy nor sell — treating it as a buy
      biases every cumulative-delta condition upward.
    * More than one instrument_id means the tape spans a roll (continuous
      symbology) and is warned about.
    """
    if sell_aggressor_code not in ("A", "B"):
        raise ValueError(f"sell_aggressor_code must be 'A' or 'B', got {sell_aggressor_code!r}")
    buy_code = "B" if sell_aggressor_code == "A" else "A"
    if trades.empty:
        raise ValueError("no trades to aggregate")

    ts_src = trades["ts_event"] if "ts_event" in trades.columns else trades.index
    ts = pd.DatetimeIndex(pd.to_datetime(ts_src, utc=True))

    if "instrument_id" in trades.columns and trades["instrument_id"].nunique() > 1:
        firsts = pd.Series(ts, index=trades["instrument_id"].to_numpy()) \
            .groupby(level=0).min().sort_values()
        warnings.warn(
            f"!!! tape contains {len(firsts)} instruments — a roll inside the "
            f"requested range (first prints: {', '.join(map(str, firsts))}). "
            "Levels computed across the roll are corrupt. !!!",
            stacklevel=2,
        )

    px = trades["price"].astype(float)
    if px.median() > 1e6:  # fixed-point 1e-9 (to_df(price_type="fixed"))
        px = px / 1e9

    size = trades["size"].astype(float).to_numpy()
    side = trades["side"].astype(str).to_numpy()
    t = pd.DataFrame({
        "price": px.to_numpy(),
        "size": size,
        "buy_volume": np.where(side == buy_code, size, 0.0),
        "sell_volume": np.where(side == sell_aggressor_code, size, 0.0),
    }, index=ts).tz_convert(ET).sort_index()

    rule = f"{bar_minutes}min"
    bars = t["price"].resample(rule).ohlc()
    bars["volume"] = t["size"].resample(rule).sum()
    bars["buy_volume"] = t["buy_volume"].resample(rule).sum()
    bars["sell_volume"] = t["sell_volume"].resample(rule).sum()
    bars = bars.dropna(subset=["open"])
    total = float(t["size"].sum())
    unsided = float(t["size"].sum() - t["buy_volume"].sum() - t["sell_volume"].sum())
    out = _finalize(bars, "FUTURES")
    out.attrs["unsided_volume_frac"] = unsided / total if total > 0 else 0.0
    return out


def load_databento_tbbo(
    dataset: str = "GLBX.MDP3",
    symbols: str | Sequence[str] = "ESZ5",
    start: str = "2025-10-01",
    end: str = "2025-10-31",
    bar_minutes: int = 1,
    api_key: Optional[str] = None,
    sell_aggressor_code: str = "A",
) -> pd.DataFrame:
    """Pull TBBO from Databento and aggregate to signed OHLCV bars.

    TBBO = every trade, with the book's best bid/offer at the moment of the
    trade, and the aggressor side. That side flag is what makes delta,
    cumulative delta, and footprint imbalance computable — plain OHLCV cannot
    produce them, no matter how fine the bars.

    Cost note: TBBO is a fraction of the size of full MBO for the same window.
    Price a request with Databento's estimator before running it.

    `sell_aggressor_code`: Databento marks `side` as the side of the *initiating*
    order — 'A' (ask) for a sell aggressor, 'B' (bid) for a buy aggressor.
    Verify against a session you know before trusting the sign of delta; an
    inverted convention flips every conclusion in the protocol.

    The symbol is checked by `databento_symbology` *before* any request is
    made, so a request that would corrupt the tape never costs money.
    """
    symbol, stype_in = databento_symbology(symbols)
    try:
        import databento as db
    except ImportError:
        raise ImportError("pip install databento")

    client = db.Historical(api_key) if api_key else db.Historical()
    data = client.timeseries.get_range(
        dataset=dataset, schema="tbbo", symbols=symbol, stype_in=stype_in,
        start=start, end=end,
    )
    trades = data.to_df()
    if trades.empty:
        raise ValueError("Databento returned no trades for that range/symbol")
    return tbbo_to_bars(trades, bar_minutes=bar_minutes,
                        sell_aggressor_code=sell_aggressor_code)


# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    poc: float
    vah: float
    val: float
    total_volume: float
    bins: pd.Series  # index = price level, value = volume

    @property
    def value_area_width(self) -> float:
        return self.vah - self.val


def volume_profile(bars: pd.DataFrame, tick: float,
                   value_area: float = VALUE_AREA_PCT) -> Optional[Profile]:
    """Volume profile from OHLCV bars, distributing each bar's volume evenly
    across the price levels it spanned.

    This is an approximation — true profiles are built from trades at price.
    It is stable enough for POC and value-area edges on 1-minute bars, which
    is what the protocol's levels need. If you have TBBO, prefer
    `volume_profile_from_trades` for the real thing.
    """
    if bars.empty or tick <= 0:
        return None

    lo_i = np.round(bars["low"].to_numpy() / tick).astype(np.int64)
    hi_i = np.round(bars["high"].to_numpy() / tick).astype(np.int64)
    vol = bars["volume"].to_numpy(dtype=float)

    lo_all, hi_all = lo_i.min(), hi_i.max()
    n_levels = int(hi_all - lo_all + 1)
    if n_levels <= 0 or n_levels > 5_000_000:
        return None

    acc = np.zeros(n_levels, dtype=float)
    spans = (hi_i - lo_i + 1).astype(float)
    per_level = np.divide(vol, spans, out=np.zeros_like(vol), where=spans > 0)

    # vectorized scatter-add over each bar's [low, high] span
    starts = (lo_i - lo_all).astype(np.int64)
    ends = (hi_i - lo_all + 1).astype(np.int64)
    diff = np.zeros(n_levels + 1, dtype=float)
    np.add.at(diff, starts, per_level)
    np.add.at(diff, ends, -per_level)
    acc = np.cumsum(diff)[:n_levels]

    prices = (np.arange(n_levels) + lo_all) * tick
    return _profile_from_bins(pd.Series(acc, index=prices), value_area)


def volume_profile_from_trades(
    price: np.ndarray, size: np.ndarray, tick: float,
    value_area: float = VALUE_AREA_PCT,
) -> Optional[Profile]:
    """Exact volume profile from trades at price (use with TBBO)."""
    if len(price) == 0:
        return None
    idx = np.round(np.asarray(price, dtype=float) / tick).astype(np.int64)
    lo = idx.min()
    acc = np.zeros(int(idx.max() - lo + 1), dtype=float)
    np.add.at(acc, idx - lo, np.asarray(size, dtype=float))
    prices = (np.arange(len(acc)) + lo) * tick
    return _profile_from_bins(pd.Series(acc, index=prices), value_area)


def _profile_from_bins(bins: pd.Series, value_area: float) -> Optional[Profile]:
    """Standard alternating expansion from the POC until `value_area` of volume."""
    total = float(bins.sum())
    if total <= 0:
        return None

    vals = bins.to_numpy(dtype=float)
    prices = bins.index.to_numpy(dtype=float)
    poc_i = int(np.argmax(vals))
    target = total * value_area

    lo_i = hi_i = poc_i
    covered = vals[poc_i]
    n = len(vals)
    while covered < target and (lo_i > 0 or hi_i < n - 1):
        # compare the next TWO levels on each side, take the heavier pair
        up = vals[hi_i + 1: hi_i + 3].sum() if hi_i < n - 1 else -1.0
        dn = vals[max(lo_i - 2, 0): lo_i].sum() if lo_i > 0 else -1.0
        if up >= dn:
            step = min(2, n - 1 - hi_i)
            covered += vals[hi_i + 1: hi_i + 1 + step].sum()
            hi_i += step
        else:
            step = min(2, lo_i)
            covered += vals[lo_i - step: lo_i].sum()
            lo_i -= step

    return Profile(
        poc=float(prices[poc_i]),
        vah=float(prices[hi_i]),
        val=float(prices[lo_i]),
        total_volume=total,
        bins=bins,
    )


# ---------------------------------------------------------------------------
# Session slicing
# ---------------------------------------------------------------------------

def _session_dates(bars: pd.DataFrame) -> list[date]:
    rth = bars.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")
    return sorted({ts.date() for ts in rth.index})


def rth_slice(bars: pd.DataFrame, d: date) -> pd.DataFrame:
    day = bars[bars.index.date == d]
    return day.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")


def overnight_slice(bars: pd.DataFrame, d: date) -> pd.DataFrame:
    """18:00 ET the previous calendar day through 09:30 ET on `d`.

    Empty for equity sources, which have no overnight session — callers get
    None-valued overnight fields rather than a silently wrong range.
    """
    if not bars.attrs.get("has_overnight", False):
        return bars.iloc[0:0]
    start = pd.Timestamp.combine(d - timedelta(days=1), ON_OPEN).tz_localize(ET)
    stop = pd.Timestamp.combine(d, RTH_OPEN).tz_localize(ET)
    return bars[(bars.index >= start) & (bars.index < stop)]


# ---------------------------------------------------------------------------
# Session levels
# ---------------------------------------------------------------------------

@dataclass
class SessionLevels:
    date: date
    source: SourceKind

    # opening
    open_px: float
    ib_high: float
    ib_low: float
    ib_range: float

    # session
    rth_high: float
    rth_low: float
    rth_close: float
    vwap: pd.Series                # session-anchored, indexed like the RTH bars
    vwap_crosses: int              # counted through the cutoff, not the close

    # prior day
    prior_vah: Optional[float]
    prior_val: Optional[float]
    prior_vpoc: Optional[float]
    prior_close: Optional[float]   # prior RTH session's last close
    opened_inside_value: Optional[bool]

    # overnight (futures only)
    on_high: Optional[float]
    on_low: Optional[float]
    on_mid: Optional[float]
    on_range_pos: Optional[float]  # where 09:30 sat in the ON range, 0..1
    on_volume: Optional[float]

    # structure
    otf_up_bars: int               # consecutive one-timeframing 30m bars, up
    otf_down_bars: int
    day_type: DayType
    classified_at: time

    # order flow (TBBO only)
    cum_delta: Optional[pd.Series]
    delta_at_extreme: Optional[bool]

    @property
    def gap_pct(self) -> Optional[float]:
        """Signed open-vs-prior-close gap as a fraction. None without a prior session."""
        if self.prior_close is None or self.prior_close <= 0:
            return None
        return (self.open_px - self.prior_close) / self.prior_close

    def s1_gate(self, index_level: Optional[float] = None) -> bool:
        """S1: IB range between 0.25% and 0.75% of index level."""
        ref = index_level or self.open_px
        if ref <= 0:
            return False
        return S1_IB_RANGE_MIN_PCT <= self.ib_range / ref <= S1_IB_RANGE_MAX_PCT

    def s2_gate(self) -> bool:
        """S2: opened inside prior value, and not one-timeframing."""
        return bool(self.opened_inside_value) and max(
            self.otf_up_bars, self.otf_down_bars) < S3_MIN_OTF_BARS

    def s3_gate(self) -> bool:
        """S3: opened outside prior value, one-timeframing 2+ bars, VWAP not chopped."""
        if self.opened_inside_value is None:
            return False
        return (not self.opened_inside_value
                and max(self.otf_up_bars, self.otf_down_bars) >= S3_MIN_OTF_BARS
                and self.vwap_crosses <= S3_MAX_VWAP_CROSSES)

    def s5_gate(self, on_volume_median: Optional[float] = None) -> bool:
        """S5: 09:30 in the top/bottom S5_ON_EXTREME_PCT of a well-traded overnight range."""
        if self.on_range_pos is None:
            return False
        extreme = (self.on_range_pos >= 1.0 - S5_ON_EXTREME_PCT
                   or self.on_range_pos <= S5_ON_EXTREME_PCT)
        if on_volume_median is None or self.on_volume is None:
            return extreme
        return extreme and self.on_volume > on_volume_median


def _session_vwap(bars: pd.DataFrame) -> pd.Series:
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    cum_v = bars["volume"].cumsum()
    cum_pv = (tp * bars["volume"]).cumsum()
    return (cum_pv / cum_v.replace(0, np.nan)).ffill()


def running_crosses(close: pd.Series, vwap: pd.Series) -> np.ndarray:
    """Closes through VWAP, counted through each bar.

    A close exactly on VWAP (or before VWAP exists) takes no side, so it
    neither counts as a cross nor resets the last side. One definition serves
    both the 11:00 day-type count and S3's bar-by-bar stand-down, so the two
    can never disagree about what a cross is.
    """
    side = np.sign((close - vwap).to_numpy(dtype=float))
    held = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flips = np.zeros(len(held), dtype=np.int64)
    if len(held) > 1:
        prev, cur = held[:-1], held[1:]
        flips[1:] = (cur != prev) & ~np.isnan(cur) & ~np.isnan(prev)
    return np.cumsum(flips)


def _count_crosses(close: pd.Series, vwap: pd.Series) -> int:
    counts = running_crosses(close, vwap)
    return int(counts[-1]) if len(counts) else 0


def _one_timeframing(bars30: pd.DataFrame) -> tuple[int, int]:
    """Longest run of consecutive 30-min bars extending in one direction.

    Up = higher high AND higher low versus the prior bar. Down = the mirror.
    """
    if len(bars30) < 2:
        return 0, 0
    hh = bars30["high"].diff() > 0
    hl = bars30["low"].diff() > 0
    lh = bars30["high"].diff() < 0
    ll = bars30["low"].diff() < 0
    up, down = (hh & hl).to_numpy(), (lh & ll).to_numpy()

    def longest(flags):
        best = run = 0
        for f in flags:
            run = run + 1 if f else 0
            best = max(best, run)
        return best

    return longest(up), longest(down)


def _classify(
    otf_up: int, otf_down: int, opened_inside: Optional[bool],
    pre: pd.DataFrame, vwap_crosses: int,
) -> DayType:
    """Day type from data available AT THE CUTOFF only."""
    if pre.empty:
        return "UNCLASSIFIED"
    if opened_inside is None:
        return "UNCLASSIFIED"
    trending = max(otf_up, otf_down) >= S3_MIN_OTF_BARS and not opened_inside
    if trending and vwap_crosses <= S3_MAX_VWAP_CROSSES:
        return "TREND_UP" if otf_up >= otf_down else "TREND_DOWN"
    # a wide pre-cutoff range with heavy two-sided rotation reads as two
    # distributions forming rather than one balanced one
    rng = pre["high"].max() - pre["low"].min()
    ib_rng = pre.between_time(RTH_OPEN, IB_END, inclusive="left")
    ib = (ib_rng["high"].max() - ib_rng["low"].min()) if not ib_rng.empty else np.nan
    if (not np.isnan(ib) and ib > 0 and rng > DOUBLE_DIST_RANGE_MULT * ib
            and vwap_crosses > S3_MAX_VWAP_CROSSES):
        return "DOUBLE_DIST"
    return "BALANCE"


def build_sessions(
    bars: pd.DataFrame,
    tick: float,
    cutoff: time = DAY_TYPE_CUTOFF,
    value_area: float = VALUE_AREA_PCT,
    dates: Optional[Iterable[date]] = None,
) -> list[SessionLevels]:
    """Build a SessionLevels record per RTH session.

    Every field that feeds day classification is computed from bars at or
    before `cutoff`. The full-session high/low/close are recorded for outcome
    evaluation but never touch the classification path.
    """
    source: SourceKind = bars.attrs.get("source", "SPY")
    all_dates = _session_dates(bars)
    wanted = sorted(set(dates)) if dates is not None else all_dates

    prior_profiles: dict[date, Optional[Profile]] = {}
    prior_closes: dict[date, float] = {}
    out: list[SessionLevels] = []

    for i, d in enumerate(all_dates):
        rth = rth_slice(bars, d)
        if rth.empty:
            continue
        prior_profiles[d] = volume_profile(rth, tick=tick, value_area=value_area)
        prior_closes[d] = float(rth["close"].iloc[-1])
        if d not in wanted:
            continue

        prior_d = all_dates[i - 1] if i > 0 else None
        pp = prior_profiles.get(prior_d) if prior_d else None
        prior_close = prior_closes.get(prior_d) if prior_d else None

        ib = rth.between_time(RTH_OPEN, IB_END, inclusive="left")
        if ib.empty:
            continue
        ib_high, ib_low = float(ib["high"].max()), float(ib["low"].min())
        open_px = float(rth["open"].iloc[0])

        vwap = _session_vwap(rth)
        pre = rth.between_time(RTH_OPEN, cutoff, inclusive="left")
        crosses = _count_crosses(pre["close"], vwap.loc[pre.index]) if not pre.empty else 0

        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        pre30 = pre.resample(f"{OTF_BAR_MINUTES}min").agg(agg).dropna(subset=["open"])
        otf_up, otf_down = _one_timeframing(pre30)

        opened_inside = None
        if pp is not None:
            opened_inside = bool(pp.val <= open_px <= pp.vah)

        on = overnight_slice(bars, d)
        if on.empty:
            on_high = on_low = on_mid = on_pos = on_vol = None
        else:
            on_high, on_low = float(on["high"].max()), float(on["low"].min())
            on_mid = (on_high + on_low) / 2.0
            span = on_high - on_low
            # a range with no width has no position in it — None, not 0.5
            on_pos = min(max(float((open_px - on_low) / span), 0.0), 1.0) \
                if span > 0 else None
            on_vol = float(on["volume"].sum())

        if bars.attrs.get("has_delta", False):
            cd = (rth["buy_volume"] - rth["sell_volume"]).cumsum()
            pre_cd = cd.loc[pre.index] if not pre.empty else cd.iloc[0:0]
            at_extreme = None
            if len(pre_cd) > 1:
                last = pre_cd.iloc[-1]
                at_extreme = bool(last >= pre_cd.max() or last <= pre_cd.min())
        else:
            cd, at_extreme = None, None

        out.append(SessionLevels(
            date=d, source=source, open_px=open_px,
            ib_high=ib_high, ib_low=ib_low, ib_range=ib_high - ib_low,
            rth_high=float(rth["high"].max()), rth_low=float(rth["low"].min()),
            rth_close=float(rth["close"].iloc[-1]),
            vwap=vwap, vwap_crosses=crosses,
            prior_vah=pp.vah if pp else None,
            prior_val=pp.val if pp else None,
            prior_vpoc=pp.poc if pp else None,
            prior_close=prior_close,
            opened_inside_value=opened_inside,
            on_high=on_high, on_low=on_low, on_mid=on_mid,
            on_range_pos=on_pos, on_volume=on_vol,
            otf_up_bars=otf_up, otf_down_bars=otf_down,
            day_type=_classify(otf_up, otf_down, opened_inside, pre, crosses),
            classified_at=cutoff,
            cum_delta=cd, delta_at_extreme=at_extreme,
        ))

    if source == "SPY":
        warnings.warn(
            "SPY source: overnight fields are None. S5 is untestable and S3's "
            "'opened outside prior value' gate is weaker than on futures, where "
            "the 18:00-09:30 session does real work.",
            stacklevel=2,
        )
    return out


def sessions_to_frame(sessions: Sequence[SessionLevels]) -> pd.DataFrame:
    """Flatten to one row per session for joining onto a trade log."""
    rows = []
    for s in sessions:
        rows.append({
            "session_date": s.date, "source": s.source, "open_px": s.open_px,
            "ib_high": s.ib_high, "ib_low": s.ib_low, "ib_range_pts": s.ib_range,
            "rth_high": s.rth_high, "rth_low": s.rth_low, "rth_close": s.rth_close,
            "vwap_crosses": s.vwap_crosses,
            "prior_vah": s.prior_vah, "prior_val": s.prior_val,
            "prior_vpoc": s.prior_vpoc, "prior_close": s.prior_close,
            "gap_pct": s.gap_pct, "opened_inside_value": s.opened_inside_value,
            "on_high": s.on_high, "on_low": s.on_low, "on_mid": s.on_mid,
            "on_range_pos": s.on_range_pos, "on_volume": s.on_volume,
            "otf_up_bars": s.otf_up_bars, "otf_down_bars": s.otf_down_bars,
            "day_type": s.day_type,
            "s1_gate": s.s1_gate(), "s2_gate": s.s2_gate(),
            "s3_gate": s.s3_gate(), "s5_gate": s.s5_gate(),
        })
    return pd.DataFrame(rows)
