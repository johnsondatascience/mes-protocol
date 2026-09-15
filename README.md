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

# 1. prototype on free SPY minute bars (no overnight session — S5 unavailable)
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
Check one session against Optimus Flow before running a study on it.

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
- Logs are validated first: unknown direction/setup/grade, checklist flags other
  than 0/1, and stops on the wrong side of entry are errors.

## Contributing

Read [`CLAUDE.md`](CLAUDE.md) first. It lists the invariants, the definition of
done for a new setup generator, and the specific "improvements" that would
quietly invalidate the study.

## Disclaimer

Research code. Not trading advice. Futures are leveraged and losses can exceed
intended risk on gaps.
