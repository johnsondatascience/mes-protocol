"""Signal generators for S1 (IB break and retest) and S3 (VWAP continuation).

WHAT THIS IS
------------
A bar-level pre-screen. Each generator walks a session's 1-minute bars forward,
emits a Signal the moment its pre-registered conditions complete, and a
simulator resolves the mechanical 2R outcome. The output is a trade log in the
same schema as hand-logged replay, so both flow into the same evaluator.

WHAT THIS IS NOT
----------------
A substitute for manual replay. Bar data cannot see resting liquidity, cannot
distinguish absorption from a pause, and reconstructs fills optimistically no
matter how careful the convention. A setup that shows an edge here has cleared
the *cheapest* possible test, not a meaningful one. A setup that shows nothing
here, under generous assumptions, is very unlikely to reward 200 sessions of
hand replay — which is the actual point of running it.

FILL CONVENTIONS (deliberately pessimistic — do not "fix" these)
---------------------------------------------------------------
* A limit entry fills only if a later bar trades strictly through the level,
  never on a touch.
* A stop entry fills at the trigger price plus one tick of slippage against you.
* If a single bar's range covers both stop and target, the STOP is assumed to
  have been hit first. Bar data cannot resolve the sequence, and assuming the
  favorable ordering is how backtests manufacture edges that evaporate live.
* The fill bar is checked for the stop, never for the target. A stop entry
  whose fill bar spans the stop is scored a loss and flagged ambiguous; a
  limit fill bar that spans the stop is a certain loss.
* An entry still resting at the signal's entry_deadline (a setup's stand-down
  time) is cancelled, never filled.
* Any position still open at RTH close exits at the closing price (TIME).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    ENTRY_MAX_WAIT_BARS, ET, IB_FAIL_STOP_BEYOND_EXTREME_PTS, IB_FAIL_STOP_CAP_PTS,
    IB_FAIL_STOP_FLOOR_PTS, MECHANICAL_TARGET_R, OTF_BAR_MINUTES, RTH_CLOSE,
    RTH_OPEN, S1_BREAK_WINDOW, S1_DELTA_CONFIRM_BARS,
    S1_NEWS_EVENTS, S1_NEWS_STAND_DOWN_UNTIL, S1_RETEST_MAX_REENTRY_PTS,
    S1_STOP_BEYOND_SWING_PTS,
    S1_STOP_CAP_PTS, S1_STOP_FLOOR_PTS, S3_ENTRY_CUTOFF, S3_MAX_COUNTER_DELTA_FRAC,
    S3_MAX_VWAP_CROSSES, S3_MIN_OTF_BARS, S3_STOP_CAP_PTS, S3_STOP_FLOOR_PTS,
    S3_VWAP_TOLERANCE_PTS, Contract,
)
from .levels import SessionLevels, rth_slice, running_crosses
from .news import NewsCalendar

Direction = Literal["LONG", "SHORT"]
EntryStyle = Literal["LIMIT", "STOP"]


@dataclass
class Signal:
    """A trade the rules generated, before any outcome is known."""
    setup: str
    session_date: date
    direction: Direction
    signal_time: datetime          # when the last condition completed
    entry_px: float
    stop_px: float
    entry_style: EntryStyle
    checklist: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    # a resting entry is cancelled at this ET time: a stand-down that only
    # gated signals would still let an order placed at 14:59 fill at 15:28
    entry_deadline: Optional[time] = None

    @property
    def risk_pts(self) -> float:
        return abs(self.entry_px - self.stop_px)

    @property
    def sign(self) -> int:
        return 1 if self.direction == "LONG" else -1

    @property
    def target_px(self) -> float:
        return self.entry_px + self.sign * MECHANICAL_TARGET_R * self.risk_pts

    @property
    def checklist_ok(self) -> bool:
        """True only if every recorded condition is explicitly True.

        A None (undeterminable — usually missing delta) counts as NOT ok. A
        generator running on bar-only data will therefore mark its S1 signals
        checklist_ok=False, which is correct: those trades did not verify the
        delta condition and must not be pooled with ones that did.
        """
        return all(v is True for v in self.checklist.values())


@dataclass
class Fill:
    """Outcome of simulating a Signal forward through the session."""
    signal: Signal
    filled: bool
    entry_time: Optional[datetime] = None
    entry_px: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_px: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    # the exit depended on an intrabar order OHLC cannot show: stop and target
    # inside one bar, or a stop entry's fill bar that also spans the stop
    ambiguous_bar: bool = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _between(idx: pd.DatetimeIndex, lo: time, hi: time) -> np.ndarray:
    t = idx.time
    return (t >= lo) & (t < hi)


def _bar_delta(bars: pd.DataFrame) -> Optional[pd.Series]:
    if bars["buy_volume"].isna().all():
        return None
    return bars["buy_volume"] - bars["sell_volume"]


def _clamp_stop(entry: float, raw_stop: float, sign: int,
                floor: float, cap: float) -> Optional[float]:
    """Apply the setup's stop floor and cap. Returns None if structure exceeds cap."""
    dist = abs(entry - raw_stop)
    if dist > cap:
        return None                       # stand down, do not widen the target
    dist = max(dist, floor)               # never risk less than the floor
    return entry - sign * dist


# ---------------------------------------------------------------------------
# S1 — initial balance break and retest
# ---------------------------------------------------------------------------

def generate_s1(
    bars: pd.DataFrame, lv: SessionLevels, contract: Contract,
    news: Optional[NewsCalendar] = None,
) -> list[Signal]:
    """Break of the IB between 10:00 and 11:30, entered on the retest.

    State machine: WAITING -> BROKEN (close crosses the IB edge from inside)
    -> RETEST (price returns to the edge without closing back inside by more
    than S1_RETEST_MAX_REENTRY_PTS) -> emit.

    A break is a *crossing*. Price that stays beyond the edge after a signal,
    a failed retest, or a stand-down is still the same break; re-arming on
    every close beyond the edge would log one idea as several correlated
    trades and overweight whichever sessions hovered there.

    Stand-downs. On days carrying any release in S1_NEWS_EVENTS nothing is
    emitted before S1_NEWS_STAND_DOWN_UNTIL; `news` is that calendar. Without
    one — or on a session the calendar does not cover — the stand-down cannot
    be verified, so earlier signals carry no_news_stand_down=None.

    A gap open beyond GAP_OPEN_PCT of the prior close is a different regime,
    but since the 2026-09-15 amendment it is not a checklist condition: the
    trade is kept, and context["gap_pct"] lets the report show those sessions
    on their own.
    """
    return [s for s in _ib_break_signals(bars, lv, contract, news) if s.setup == "IB_BREAK"]


def generate_ib_fail(
    bars: pd.DataFrame, lv: SessionLevels, contract: Contract,
    news: Optional[NewsCalendar] = None,
) -> list[Signal]:
    """The S1 break whose retest fails, traded the other way (added 2026-09-15).

    Protocol §05: "The false break that reverses back through the entire IB.
    Log those as IB_FAIL." Waiting for the whole IB to be crossed would be
    chasing, so the trigger is the moment the break is known to have failed:
    the first bar in S1_BREAK_WINDOW that closes more than
    S1_RETEST_MAX_REENTRY_PTS back inside — exactly the condition that ends
    an S1 break. S1 and IB_FAIL share one walk of the tape, so a break yields
    at most one of them: S1 if a retest held first, IB_FAIL if it failed.

    Entry is a stop order 1 tick beyond the failure bar's far end (proof, not
    anticipation). The stop sits IB_FAIL_STOP_BEYOND_EXTREME_PTS beyond the
    false break's extreme; structure wider than IB_FAIL_STOP_CAP_PTS is no
    trade. S1's gate, news stand-down and gap flag apply unchanged.
    """
    return [s for s in _ib_break_signals(bars, lv, contract, news) if s.setup == "IB_FAIL"]


def _ib_break_signals(
    bars: pd.DataFrame, lv: SessionLevels, contract: Contract,
    news: Optional[NewsCalendar],
) -> list[Signal]:
    """One walk of the S1 window emitting both IB_BREAK and IB_FAIL signals."""
    out: list[Signal] = []
    if not lv.s1_gate():
        return out

    rth = rth_slice(bars, lv.date)
    win_mask = _between(rth.index, *S1_BREAK_WINDOW)
    win = rth[win_mask]
    if win.empty:
        return out
    prev_close = rth["close"].shift(1)[win_mask].to_numpy()

    delta = _bar_delta(rth)
    cum = delta.cumsum() if delta is not None else None
    news_day = None if news is None else news.is_news_day(lv.date, S1_NEWS_EVENTS)

    def news_status(ts: pd.Timestamp) -> Optional[bool]:
        """True clear, False stand down, None unverifiable."""
        if ts.time() >= S1_NEWS_STAND_DOWN_UNTIL:
            return True
        return None if news_day is None else not news_day

    def context(break_i: int, **extra) -> dict:
        return {"ib_high": lv.ib_high, "ib_low": lv.ib_low,
                "ib_range_pts": lv.ib_range, "day_type": lv.day_type,
                "on_range_pos": lv.on_range_pos,   # S5 as a covariate
                "gap_pct": lv.gap_pct,
                "break_time": str(win.index[break_i].time()), **extra}

    state = "WAITING"
    direction: Optional[Direction] = None
    edge = np.nan
    break_i = -1
    retest_extreme = np.nan   # swing an S1 stop goes beyond (pullback side)
    break_extreme = np.nan    # furthest the break reached (an IB_FAIL stop's anchor)
    s1_blocked = False        # S1's structure exceeded its cap on this break

    for i, (ts, bar) in enumerate(win.iterrows()):
        if state == "WAITING":
            pc = prev_close[i]
            if bar["close"] > lv.ib_high and not pc > lv.ib_high:
                state, direction, edge, break_i = "BROKEN", "LONG", lv.ib_high, i
            elif bar["close"] < lv.ib_low and not pc < lv.ib_low:
                state, direction, edge, break_i = "BROKEN", "SHORT", lv.ib_low, i
            if state == "BROKEN":
                # anchor is the swing the stop goes BEYOND: the pullback low on
                # a long break, the pullback high on a short one.
                retest_extreme = bar["low"] if direction == "LONG" else bar["high"]
                break_extreme = bar["high"] if direction == "LONG" else bar["low"]
                s1_blocked = False
            continue

        sign = 1 if direction == "LONG" else -1
        if direction == "LONG":
            retest_extreme = min(retest_extreme, bar["low"])
            break_extreme = max(break_extreme, bar["high"])
        else:
            retest_extreme = max(retest_extreme, bar["high"])
            break_extreme = min(break_extreme, bar["low"])

        # failure: a bar CLOSED back inside the IB by more than the tolerance.
        # A wick deeper inside that closes within it is a retest that held
        # (amended 2026-09-15); the wick still sets the swing the stop goes beyond.
        reentry = (edge - bar["close"]) if direction == "LONG" else (bar["close"] - edge)
        if reentry > S1_RETEST_MAX_REENTRY_PTS:
            fail_news = news_status(ts)
            if fail_news is not False:
                fail_sign = -sign
                entry = float(bar["low"] - contract.tick) if fail_sign < 0 \
                    else float(bar["high"] + contract.tick)
                raw_stop = break_extreme - fail_sign * IB_FAIL_STOP_BEYOND_EXTREME_PTS
                stop = _clamp_stop(entry, raw_stop, fail_sign,
                                   IB_FAIL_STOP_FLOOR_PTS, IB_FAIL_STOP_CAP_PTS)
                if stop is not None:
                    out.append(Signal(
                        setup="IB_FAIL", session_date=lv.date,
                        direction="SHORT" if fail_sign < 0 else "LONG",
                        signal_time=ts, entry_px=entry, stop_px=stop, entry_style="STOP",
                        checklist={
                            "ib_range_in_band": True,
                            "break_in_window": True,
                            "failed_in_window": True,
                            "stop_within_cap": True,
                            "no_news_stand_down": fail_news,
                        },
                        context=context(break_i, failed_break_extreme=float(break_extreme)),
                    ))
            state, direction = "WAITING", None
            continue

        # retest: this bar traded back to the edge without breaking the tolerance
        touched = (bar["low"] <= edge) if direction == "LONG" else (bar["high"] >= edge)
        if s1_blocked or not touched or i == break_i:
            continue

        news_ok = news_status(ts)
        if news_ok is False:
            continue                          # stand down; the break stays armed

        # delta confirmation: cumulative delta made a new session extreme in the
        # break direction on the break bar or within N bars after it. Only bars
        # through the current one are visible. While that window is still open
        # and unconfirmed the condition is pending, not failed — a later retest
        # bar may complete it — so nothing is emitted yet.
        delta_ok: Optional[bool] = None
        if cum is not None:
            b_ts = win.index[break_i]
            last_i = min(break_i + S1_DELTA_CONFIRM_BARS, i)
            seg = cum.loc[b_ts:win.index[last_i]]
            prior = cum[cum.index < b_ts]
            if len(seg) and len(prior):
                delta_ok = bool(seg.max() > prior.max()) if direction == "LONG" \
                    else bool(seg.min() < prior.min())
            if delta_ok is False and i < break_i + S1_DELTA_CONFIRM_BARS:
                continue

        entry = edge + sign * contract.tick
        raw_stop = retest_extreme - sign * S1_STOP_BEYOND_SWING_PTS
        # the swing must be on the correct side of entry to be a stop at all
        if sign * (entry - raw_stop) <= 0:
            raw_stop = entry - sign * S1_STOP_FLOOR_PTS
        stop = _clamp_stop(entry, raw_stop, sign, S1_STOP_FLOOR_PTS, S1_STOP_CAP_PTS)
        if stop is None:
            # The swing only gets wider on later bars, so S1 is done with this
            # break — but it stays armed, because it can still fail (IB_FAIL).
            s1_blocked = True
            continue

        out.append(Signal(
            setup="IB_BREAK", session_date=lv.date, direction=direction,
            signal_time=ts, entry_px=entry, stop_px=stop, entry_style="LIMIT",
            checklist={
                "ib_range_in_band": True,
                "break_in_window": True,
                "delta_confirmed": delta_ok,
                "retest_held": True,
                "stop_within_cap": True,
                "no_news_stand_down": news_ok,
            },
            context=context(break_i),
        ))
        state, direction = "WAITING", None   # one signal per break sequence
    return out


# ---------------------------------------------------------------------------
# S3 — VWAP pullback continuation on trend days
# ---------------------------------------------------------------------------

def _otf_confirm_time(rth: pd.DataFrame, cutoff: time) -> Optional[pd.Timestamp]:
    """Timestamp at which one-timeframing became confirmable.

    Entries before this are look-ahead: the day gate was not yet knowable.
    """
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    pre = rth.between_time(RTH_OPEN, cutoff, inclusive="left")
    b30 = pre.resample(f"{OTF_BAR_MINUTES}min").agg(agg).dropna(subset=["open"])
    if len(b30) < S3_MIN_OTF_BARS + 1:
        return None
    hh, hl = b30["high"].diff() > 0, b30["low"].diff() > 0
    lh, ll = b30["high"].diff() < 0, b30["low"].diff() < 0
    up, dn = (hh & hl).to_numpy(), (lh & ll).to_numpy()
    run_u = run_d = 0
    for k in range(len(b30)):
        run_u = run_u + 1 if up[k] else 0
        run_d = run_d + 1 if dn[k] else 0
        if max(run_u, run_d) >= S3_MIN_OTF_BARS:
            # confirmable only once that 30-min bar has closed
            return b30.index[k] + pd.Timedelta(minutes=OTF_BAR_MINUTES)
    return None


def _delta_extreme_in_confirming_bar(delta: Optional[pd.Series],
                                     confirm_at: pd.Timestamp,
                                     direction: Direction) -> Optional[bool]:
    """Did cumulative delta make a new session extreme, in the trend's
    direction, during the 30-minute bar that confirmed one-timeframing?

    Read at the same resolution as the price condition it accompanies: price
    one-timeframes when a 30-minute bar extends the range; flow agrees when
    that same bar carries cumulative delta to a new session extreme. Demanding
    the extreme on the last minute before confirmation failed trend days whose
    buying paused for a minute or two.

    Direction matters: an up-trend whose delta high was set earlier and not
    exceeded in the confirming bar has stopped agreeing with price. None when
    there is no delta, or no bars before the confirming bar to compare with.
    """
    if delta is None:
        return None
    cum = delta.cumsum()
    bar_start = confirm_at - pd.Timedelta(minutes=OTF_BAR_MINUTES)
    before = cum[cum.index < bar_start]
    during = cum[(cum.index >= bar_start) & (cum.index < confirm_at)]
    if before.empty or during.empty:
        return None
    if direction == "LONG":
        return bool(during.max() > before.max())
    return bool(during.min() < before.min())


def generate_s3(
    bars: pd.DataFrame, lv: SessionLevels, contract: Contract,
    entry_style: EntryStyle = "LIMIT",
) -> list[Signal]:
    """Pullback to session VWAP on a confirmed trend day.

    Day gate, all confirmed before any entry: opened outside prior value,
    one-timeframing on 30-minute bars, and cumulative delta at a session
    extreme with price — read as a new session extreme during the 30-minute
    bar that confirmed one-timeframing. A gate condition known to fail
    returns []; one the data cannot check (no delta) is recorded as None and
    flags every signal.

    Stand-downs are enforced bar by bar: no entries after S3_ENTRY_CUTOFF, and
    none once VWAP has been crossed more than S3_MAX_VWAP_CROSSES times — that
    count keeps running after the 11:00 classification, because an afternoon
    of chop is exactly the balance day in a trend costume the rule exists for.

    `entry_style` is pre-registered per the protocol: LIMIT at VWAP, or STOP
    beyond the pullback bar's extreme. Pick one and never mix within a sample.
    """
    out: list[Signal] = []
    if not lv.s3_gate():
        return out

    rth = rth_slice(bars, lv.date)
    confirm_at = _otf_confirm_time(rth, lv.classified_at)
    if confirm_at is None:
        return out

    direction: Direction = "LONG" if lv.otf_up_bars >= lv.otf_down_bars else "SHORT"
    sign = 1 if direction == "LONG" else -1
    vwap = lv.vwap
    delta = _bar_delta(rth)

    delta_at_extreme = _delta_extreme_in_confirming_bar(delta, confirm_at, direction)
    if delta_at_extreme is False:
        return out
    crosses = running_crosses(rth["close"], vwap)

    tradeable = rth[(rth.index >= confirm_at)
                    & (rth.index.time < S3_ENTRY_CUTOFF)]
    if tradeable.empty:
        return out

    in_pullback = False
    pb_start = 0
    # the impulse leg runs from the end of the previous pullback (or the open,
    # for the first one) to the start of the current pullback
    last_touch_i = 0
    idx_all = rth.index

    for ts, bar in tradeable.iterrows():
        i = idx_all.get_loc(ts)
        if crosses[i] > S3_MAX_VWAP_CROSSES:
            break                            # stand down for the rest of the session
        v = vwap.iloc[i]
        if np.isnan(v):
            continue
        near = (abs(bar["low"] - v) <= S3_VWAP_TOLERANCE_PTS) if direction == "LONG" \
            else (abs(bar["high"] - v) <= S3_VWAP_TOLERANCE_PTS)

        if not near:
            if in_pullback:
                in_pullback = False          # pullback ended without qualifying
            continue

        if not in_pullback:
            in_pullback, pb_start = True, i
            continue

        # --- pullback in progress: test the two flow conditions ---------------
        impulse = rth.iloc[last_touch_i:pb_start]
        pullback = rth.iloc[pb_start:i + 1]
        if impulse.empty or pullback.empty:
            continue

        imp_rate = impulse["volume"].mean()
        pb_rate = pullback["volume"].mean()
        vol_declining = bool(pb_rate < imp_rate) if imp_rate > 0 else None

        counter_ok: Optional[bool] = None
        if delta is not None:
            imp_d = abs(delta.iloc[last_touch_i:pb_start].sum())
            pb_d = abs(delta.iloc[pb_start:i + 1].sum())
            counter_ok = bool(pb_d < S3_MAX_COUNTER_DELTA_FRAC * imp_d) \
                if imp_d > 0 else None

        if vol_declining is not True:
            continue

        if entry_style == "LIMIT":
            entry = float(v)
        else:
            entry = float(bar["high"] + contract.tick) if direction == "LONG" \
                else float(bar["low"] - contract.tick)

        pb_extreme = float(pullback["low"].min()) if direction == "LONG" \
            else float(pullback["high"].max())
        raw_stop = pb_extreme - sign * contract.tick
        if sign * (entry - raw_stop) <= 0:
            raw_stop = entry - sign * S3_STOP_FLOOR_PTS
        stop = _clamp_stop(entry, raw_stop, sign, S3_STOP_FLOOR_PTS, S3_STOP_CAP_PTS)
        if stop is None:
            in_pullback = False
            continue

        out.append(Signal(
            setup="VWAP_CONT", session_date=lv.date, direction=direction,
            signal_time=ts, entry_px=entry, stop_px=stop, entry_style=entry_style,
            entry_deadline=S3_ENTRY_CUTOFF,
            checklist={
                "opened_outside_value": True,       # s3_gate
                "one_timeframing": True,            # s3_gate, at confirm_at
                "delta_at_extreme": delta_at_extreme,
                "vwap_crosses_ok": True,            # enforced bar by bar above
                "pullback_to_vwap": True,
                "volume_declining": vol_declining,
                "counter_delta_ok": counter_ok,
                "stop_within_cap": True,
            },
            context={
                "vwap": float(v), "day_type": lv.day_type,
                "ib_range_pts": lv.ib_range,
                "otf_bars": max(lv.otf_up_bars, lv.otf_down_bars),
                "vwap_crosses": int(crosses[i]),
                "on_range_pos": lv.on_range_pos,
                "gap_pct": lv.gap_pct,
                "confirmed_at": str(confirm_at.time()),
            },
        ))
        in_pullback = False
        last_touch_i = i
    return out


# ---------------------------------------------------------------------------
# Mechanical exit simulation
# ---------------------------------------------------------------------------

def simulate(signal: Signal, bars: pd.DataFrame, contract: Contract,
             max_wait_bars: int = ENTRY_MAX_WAIT_BARS) -> Fill:
    """Walk forward from the signal bar and resolve entry then exit.

    See FILL CONVENTIONS in the module docstring. They are intentionally
    unfavorable; loosening them is how a backtest starts lying.
    """
    rth = rth_slice(bars, signal.session_date)
    after = rth[rth.index > signal.signal_time]
    if after.empty:
        return Fill(signal, filled=False)

    sign = signal.sign
    entry_px: Optional[float] = None
    entry_time = None

    for j, (ts, bar) in enumerate(after.iterrows()):
        if j >= max_wait_bars:
            break
        if signal.entry_deadline is not None and ts.time() >= signal.entry_deadline:
            break                       # stand-down reached: the order is cancelled
        if signal.entry_style == "LIMIT":
            # must trade strictly through the limit, not merely touch it
            hit = bar["low"] < signal.entry_px if sign > 0 else bar["high"] > signal.entry_px
            if hit:
                entry_px, entry_time = signal.entry_px, ts
                break
        else:  # STOP entry — pay a tick of slippage
            hit = bar["high"] >= signal.entry_px if sign > 0 else bar["low"] <= signal.entry_px
            if hit:
                entry_px = signal.entry_px + sign * contract.tick
                entry_time = ts
                break

    if entry_px is None:
        return Fill(signal, filled=False)

    risk = abs(entry_px - signal.stop_px)
    target = entry_px + sign * MECHANICAL_TARGET_R * risk

    # The fill bar is checked for the stop and never for the target. A limit
    # long fills on the way down, so a fill bar whose low also takes out the
    # stop was stopped on that same move — certain, not ambiguous. A stop
    # entry's bar could have printed the stop before the trigger; the order
    # is unknowable, so the loss is assumed and the bar flagged. Either way,
    # skipping the fill bar would let a later bar score a TARGET on a trade
    # that was already dead.
    fill_bar = after.loc[entry_time]
    if (fill_bar["low"] <= signal.stop_px) if sign > 0 \
            else (fill_bar["high"] >= signal.stop_px):
        return Fill(signal, True, entry_time, entry_px, entry_time,
                    signal.stop_px, "STOP", 0, signal.entry_style == "STOP")

    held = 0
    ambiguous = False

    for ts, bar in rth[rth.index > entry_time].iterrows():
        held += 1
        hit_stop = bar["low"] <= signal.stop_px if sign > 0 else bar["high"] >= signal.stop_px
        hit_tgt = bar["high"] >= target if sign > 0 else bar["low"] <= target
        if hit_stop and hit_tgt:
            ambiguous = True
        if hit_stop:                                  # stop wins ties, always
            return Fill(signal, True, entry_time, entry_px, ts, signal.stop_px,
                        "STOP", held, ambiguous)
        if hit_tgt:
            return Fill(signal, True, entry_time, entry_px, ts, target,
                        "TARGET", held, ambiguous)

    last_ts, last_bar = rth.index[-1], rth.iloc[-1]
    return Fill(signal, True, entry_time, entry_px, last_ts,
                float(last_bar["close"]), "TIME", held, ambiguous)


def run_session(bars: pd.DataFrame, lv: SessionLevels, contract: Contract,
                s3_entry_style: EntryStyle = "LIMIT",
                news: Optional[NewsCalendar] = None) -> list[Fill]:
    signals = _ib_break_signals(bars, lv, contract, news) + \
        generate_s3(bars, lv, contract, entry_style=s3_entry_style)
    signals.sort(key=lambda s: s.signal_time)
    return [simulate(s, bars, contract) for s in signals]


def run_all(bars: pd.DataFrame, sessions: Sequence[SessionLevels],
            contract: Contract, s3_entry_style: EntryStyle = "LIMIT",
            news: Optional[NewsCalendar] = None) -> list[Fill]:
    fills: list[Fill] = []
    for lv in sessions:
        fills.extend(run_session(bars, lv, contract, s3_entry_style, news))
    return fills
