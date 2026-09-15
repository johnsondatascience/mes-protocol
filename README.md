# mes-protocol

Instrumentation for a pre-registered study of five intraday order-flow setups on
MES (Micro E-mini S&P 500 futures).

The study design — the setup rules, futility gates, and sample-size math — is in
[`docs/protocol.html`](docs/protocol.html). This repo is the code that supports
it: session levels computed without look-ahead, signal generation for the two
setups that are checkable from bars, pessimistic fill simulation, and a
bootstrap evaluator that treats hand-logged replay trades and generated trades
identically.

**This is a research harness, not a trading system.** Nothing here places orders.

## Why some setups are code and some are hand-logged

| Setup | Testable from bars? | How it is evaluated |
|---|---|---|
| S1 IB break and retest | Partly | `generate_s1`; delta condition needs TBBO |
| IB_FAIL (S1's failed retest, traded the other way) | Yes | `generate_ib_fail`; exploratory — reported, never judged |
| S2 Value-area edge to VPOC | No | Manual replay — needs DOM resting size |
| S3 VWAP pullback continuation | Yes | `generate_s3` |
| S4 HVN absorption reversal | No | Manual replay — needs the book |
| S5 Overnight inventory | As a covariate | Logged on every S1/S3 trade |

Bar data cannot see resting liquidity. A bar-level result is a cheap screen: a
setup that shows nothing here under generous assumptions is unlikely to reward
200 sessions of hand replay, which is the point of running it first.

## Install

```bash
git clone <this repo> && cd mes-protocol
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11+. Runtime deps: numpy, pandas, scipy. `databento` is optional and
only needed for the TBBO loader.

## Quick start

```bash
# 0. build the news calendar (needs FRED_API_KEY in the environment or .env);
#    re-run before each evaluation block — it covers through yesterday
python scripts/fetch_news_calendar.py

# 1. prototype on free SPY minute bars (no overnight session — S5 unavailable).
#    Thresholds are ES points, scaled per day by SPY / S&P 500 closes:
python scripts/fetch_reference_closes.py
python scripts/run_pipeline.py --csv data/spy_1min.csv --contract SPY

# 2. the real thing, one contract month at a time
export DATABENTO_API_KEY=...
python scripts/run_pipeline.py --databento ESZ5 --start 2025-10-01 \
    --end 2025-10-31 --contract MES

# 3. evaluate a hand-logged replay CSV
python -m mesproto.evaluate data/replay_trades.csv
```

Blank log template for hand logging:

```python
from mesproto import write_template
write_template("data/replay_trades.csv")
```

## Library use

```python
from mesproto import MES, build_sessions, load_csv_bars, sessions_to_frame
from mesproto.signals import run_all

bars = load_csv_bars("data/es_1min.csv", source="FUTURES")
sessions = build_sessions(bars, tick=MES.tick)

for s in sessions:
    print(s.date, s.day_type, s.ib_range, s.prior_vah, s.s3_gate())

fills = run_all(bars, sessions, MES)
```

`sessions_to_frame(sessions)` flattens to one row per session for joining onto a
trade log.

## Data

Nothing in `data/` is committed. Options, cheapest first:

- **Free equity minute bars** (Alpaca, Polygon free tiers) as a SPY proxy. Good
  for prototyping rules; no overnight session, so S5 is untestable and S3's
  "opened outside prior value" gate is weaker than on futures.
- **Databento signup credits** toward CME GLBX.MDP3. Buy `tbbo` (trades with the
  quote at the trade) rather than `mbo` — TBBO is a fraction of the size and
  carries the aggressor side, which is what makes delta computable. Price a
  request with their estimator first.
- **Optimus Flow Market Replay**, free with an account, for the hand-logged
  setups.

Always pull a single expiry. Back-adjusted continuous series shift historical
prices by accumulated roll gaps, which corrupts every level this repo computes.

### SPY point scaling

Every setup threshold in points (S1's 2-point retest and 4/8-point stops, S3's
±2-point tolerances and 5/10-point stops, IB_FAIL's stops) is written in ES
points. On a SPY session each is multiplied by that session's `point_scale`:
the **prior** day's SPY close divided by the S&P 500 index close on the same
day, from `data/sp500_close.csv` (`scripts/fetch_reference_closes.py`, FRED
series SP500, same free key). The prior day, because today's close is not
known at 10:00. The index stands in for ES: they differ by well under 1%,
about one SPY cent on the widest threshold. FRED keeps 10 years, so SPY runs
start no earlier than 2016-09-15. A session with no reference close for its
prior day has `point_scale=None` and S1, IB_FAIL and S3 generate nothing on
it. Percentage thresholds (the IB range filter, the gap flag) are unscaled.

### News calendar

`scripts/fetch_news_calendar.py` writes `data/news_calendar.csv`, which S1's
news stand-down reads. It needs a free FRED API key
([fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys))
in `FRED_API_KEY` or `.env`.

- **CPI, NFP, PPI, retail sales** come from FRED release dates: the days each
  release actually came out, shutdown delays included. FRED also lists about
  one annual-revision day a year per release; those count as release days.
- **FOMC** is the statement day of each *scheduled* meeting, from
  federalreserve.gov. Unscheduled meetings, notation votes and cancelled
  meetings are skipped — none could be planned around that morning.
- The file declares its date range and event types. A session outside the
  range is *unknown*, not quiet: its S1 signals before 10:30 get
  `no_news_stand_down=None` and leave the primary sample. The fetch refuses to
  write a calendar with a full year missing for any event, since that means a
  source page changed.
- Which events S1 watches is `S1_NEWS_EVENTS` in `config.py` (pre-registered).

## Verify the delta convention before trusting anything

Databento marks `side` as the **initiating** order's side: `A` (ask) is a sell
aggressor. `load_databento_tbbo(sell_aggressor_code=...)` exposes it because an
inverted convention flips the sign of every delta conclusion in the protocol.
`run_pipeline.py` checks it from the tape itself: TBBO carries the best bid and
offer at each trade, so trades at the ask must carry the buy code and trades at
the bid the sell code (`aggressor_convention`). The run stops unless they agree
on at least 95% of 1,000+ such trades. Comparing one session against Optimus
Flow is still a good second check.

### Second opinion from Optimus Flow

Optimus Flow is a desktop platform (a Quantower white label on a Rithmic feed)
with no API to query, so the check goes through a file it writes:

1. In Optimus Flow, open the **History Exporter** panel.
2. Pick the ES contract that was front month for the session, a **1-minute**
   timeframe, and a single session's date range. Include volume analysis
   (delta, or ask/bid volume) if your build offers it.
3. Export to CSV, and note the platform's display timezone.
4. `python scripts/compare_optimus_export.py --export <their.csv> --dbn <our.dbn.zst>`
   (add `--tz` if the export is not in ET).

It reports overlapping minutes, how often volume agrees, and whether their
delta runs the same way as ours or the opposite way — the hand-logged setups
(S2, S4) are read in that platform, so an inverted convention would make
hand-logged and generated trades disagree on the one condition they share.
Without a delta column it can only compare volume.

Downloads are billed. `run_pipeline.py --databento` streams the raw file to
`data/tbbo_<symbol>_<start>_<end>.dbn.zst` (or `--dbn PATH`) and reads that file
on every later run instead of requesting it again. Price a request first with
`databento.Historical().metadata.get_cost(...)`, which is free.

Trades with side `N` (no aggressor — auctions, some opening prints) count toward
volume but not delta. Bars are timed by `ts_event`, not the `ts_recv` index that
`to_df()` returns. Already have the file? `tbbo_to_bars(DBNStore.from_file(path).to_df())`
converts it without paying for the request again.

`load_databento_tbbo` refuses multiple symbols, parent symbology (`ES.FUT`) and
spreads before any request is sent, and warns loudly on continuous symbols
(`ES.c.0`) and on any tape that spans a roll.

## Tests

```bash
pytest                                    # or run the files directly
PYTHONPATH=src python3 tests/test_levels.py
PYTHONPATH=src python3 tests/test_signals.py
PYTHONPATH=src python3 tests/test_evaluate.py
```

Synthetic price paths only — no network, no data files. The load-bearing tests
are `test_no_lookahead` (mutates the tape after the classification cutoff and
asserts nothing upstream moves), the truncation tests in `test_signals.py`
(every signal must be reproducible from the tape cut at its own bar), the
fill-convention tests, and the gate tests in `test_evaluate.py` (thresholds
pinned to the protocol's §06 table, decided only at fixed checkpoints).

## Evaluation rules the CLI enforces

- R is computed from the **mechanical** exit (`mech_exit_px`, `mech_exit_reason`)
  net of the costs of each row's `contract`. The exit actually taken
  (`exit_px`) is scored as `managed_R` and reported beside it, never mixed in.
- A mechanical TARGET must sit exactly 2R from entry and a mechanical STOP at
  `stop_px`; anything else fails validation.
- The first `BURN_IN_TRADES` (30) trades of each setup, by time, are burn-in —
  whatever their checklist. They are included and flagged `burn_in=True`, and
  each setup's result is also printed without them for comparison.
- `checklist_ok=0` trades are excluded.
- The futility gate is decided at n = 60 and n = 150 on the trades that existed
  at that checkpoint. A kill at 60 stays a kill.
- Trades from `VALIDATION_PERIODS` (ES, 2026-07-30 to 2026-08-31) are dropped
  before anything else: that month validated the tooling and was read trade by
  trade, so its sessions belong to no sample. The report names them.
- IB_FAIL is exploratory (`EXPLORATORY_SETUPS`): its results are printed, but
  it gets no gate verdict or confirmation size, is left out of the portfolio,
  and is not one of the five setups in the alpha = 0.05 / 5 correction.
- Logs are validated first: unknown direction/setup/grade, checklist flags other
  than 0/1, and stops on the wrong side of entry are errors.

## Contributing

Read [`CLAUDE.md`](CLAUDE.md) first. It lists the invariants, the definition of
done for a new setup generator, and the specific "improvements" that would
quietly invalidate the study.

## Disclaimer

Research code. Not trading advice. Futures are leveraged and losses can exceed
intended risk on gaps.
