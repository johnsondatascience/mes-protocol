"""Contract specs, session boundaries, and the cost model.

Every magic number in this repo lives here. If you find a literal 0.25, 5.00,
or a hardcoded time anywhere else, that is a bug — move it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo("America/New_York")

# --- session boundaries, exchange time ---------------------------------------
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
IB_END = time(10, 0)
ON_OPEN = time(18, 0)            # CME equity index reopen, prior calendar day
DAY_TYPE_CUTOFF = time(11, 0)    # classify here. NEVER at the close.
OTF_BAR_MINUTES = 30             # one-timeframing is read on 30-minute bars

# --- levels -------------------------------------------------------------------
VALUE_AREA_PCT = 0.70            # share of session volume inside VAL..VAH
DOUBLE_DIST_RANGE_MULT = 2.0     # pre-cutoff range beyond this x IB (with chop) = DOUBLE_DIST
S5_ON_EXTREME_PCT = 0.15         # 09:30 in the top/bottom 15% of the overnight range

# --- fill simulation ----------------------------------------------------------
ENTRY_MAX_WAIT_BARS = 30         # a resting entry unfilled after this many bars is cancelled

# --- setup-specific windows ---------------------------------------------------
S1_BREAK_WINDOW = (time(10, 0), time(11, 30))
S1_NEWS_STAND_DOWN_UNTIL = time(10, 30)   # FOMC / CPI / NFP days
S3_ENTRY_CUTOFF = time(15, 0)
S4_NO_TRADE_OPEN = time(9, 35)   # first 5 minutes
S4_NO_TRADE_CLOSE = time(15, 45)  # last 15 minutes
S5_ENTRY_WINDOW = (time(9, 30), time(10, 0))
S5_TIME_STOP = time(10, 30)


@dataclass(frozen=True)
class Contract:
    """Instrument spec. Read ES for order flow; trade MES for size granularity."""
    symbol: str
    tick: float
    point_value: float
    commission_rt: float      # USD per contract, round turn, all-in
    slippage_ticks_stop: float
    slippage_ticks_entry: float

    @property
    def cost_points(self) -> float:
        """Round-turn cost in index points, assuming a stop exit."""
        return (self.commission_rt / self.point_value
                + (self.slippage_ticks_stop + self.slippage_ticks_entry) * self.tick)

    def cost_as_fraction_of_r(self, stop_points: float) -> float:
        """The tax a given stop distance pays. This is why tight stops lose."""
        return self.cost_points / stop_points if stop_points > 0 else float("inf")


MES = Contract(
    symbol="MES", tick=0.25, point_value=5.00, commission_rt=1.30,
    slippage_ticks_stop=1.0, slippage_ticks_entry=0.0,
)
ES = Contract(
    symbol="ES", tick=0.25, point_value=50.00, commission_rt=4.00,
    slippage_ticks_stop=1.0, slippage_ticks_entry=0.0,
)
SPY = Contract(  # proxy for prototyping only — no overnight session
    symbol="SPY", tick=0.01, point_value=1.00, commission_rt=0.00,
    slippage_ticks_stop=1.0, slippage_ticks_entry=0.0,
)

CONTRACTS = {c.symbol: c for c in (MES, ES, SPY)}

# exits filled by a stop order pay slippage_ticks_stop: a breakeven exit is a
# moved stop being hit, so it pays too. Market (MANUAL) exits are not charged.
STOP_ORDER_EXITS = ("STOP", "BREAKEVEN")

# --- statistical protocol -----------------------------------------------------
MIN_EXPECTANCY_R = 0.15   # below this, not worth trading after costs
N_SETUPS_TESTED = 5       # Bonferroni denominator
ALPHA = 0.05 / N_SETUPS_TESTED
ASSUMED_SIGMA_R = 1.5     # for planning only; the bootstrap uses the real thing
BURN_IN_TRADES = 30       # per setup, flagged and INCLUDED in primary (amended 2026-09-15)
FUTILITY_GATES = (60, 150)  # checkpoints at which the kill rule is applied
GATE_CONFIDENCE = 0.80    # kill when the one-sided upper bound on mean R < MIN_EXPECTANCY_R
GATE_SIGMA_FLOOR_R = 0.6 * ASSUMED_SIGMA_R  # a calm start cannot shrink the gate's bound
BOOTSTRAP_CI = 0.95       # §06: report the 2.5/97.5 session-block percentiles
CONFIRM_POWER = 0.80      # §01: power behind the confirmation sample size

# --- reporting only (never feed a verdict) -------------------------------------
CONFIRM_N_MIN_EFFECT_R = 0.05   # floor on the effect used for "n to confirm"
CONFIRM_N_MIN_SIGMA_R = 1.0     # floor on the sigma used for "n to confirm"
DAY_TYPE_MIN_TRADES = 5         # smallest day-type bucket worth printing
OFF_CHECKLIST_WARN_FRAC = 0.25  # validate() warns above this share of checklist_ok=0
GAP_OPEN_PCT = 0.01             # |open / prior close - 1| beyond this is a gap-open session:
                                # a different regime, reported separately but INCLUDED
                                # in the primary sample (amended 2026-09-15)

# --- setup parameters (pre-registered — changing one restarts the sample) -----
S1_IB_RANGE_MIN_PCT = 0.0025
S1_IB_RANGE_MAX_PCT = 0.0075
S1_DELTA_CONFIRM_BARS = 3
S1_RETEST_MAX_REENTRY_PTS = 2.0
S1_STOP_BEYOND_SWING_PTS = 1.0
S1_STOP_FLOOR_PTS = 4.0
S1_STOP_CAP_PTS = 8.0

S3_MIN_OTF_BARS = 2
S3_MAX_VWAP_CROSSES = 3
S3_VWAP_TOLERANCE_PTS = 2.0
S3_MAX_COUNTER_DELTA_FRAC = 0.40
S3_STOP_FLOOR_PTS = 5.0
S3_STOP_CAP_PTS = 10.0

MECHANICAL_TARGET_R = 2.0
PRICE_EPS = 1e-6          # float tolerance when a logged price must equal a computed one
