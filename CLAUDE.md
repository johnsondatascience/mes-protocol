# Agent handoff — mes-protocol

Read this before changing anything. It exists because most of the "obvious
improvements" available in this codebase are ways of making a backtest lie.

## What this repo is

Instrumentation for a **pre-registered study** of five intraday futures setups
on MES. The study protocol — the actual rules, the futility gates, the
sample-size math — is in `docs/protocol.html`. This code exists to:

1. compute session levels (value area, IB, VWAP, overnight range, day type)
   without look-ahead,
2. generate signals for the setups that can be checked from bars (S1, its
   IB_FAIL mirror, and S3),
3. score them under deliberately pessimistic fill assumptions,
4. evaluate the resulting trade log with a session-block bootstrap.

## What this repo is NOT

- **Not a trading system.** Nothing here places orders and nothing should.
- **Not a substitute for manual replay.** S2 and S4 depend on resting liquidity
  and footprint imbalance that bar data cannot see. They are hand-logged in
  Optimus Flow. If you are asked to "implement S4", the answer is that it needs
  order-book data (Databento MBP-10 or MBO), not cleverer bar logic.
- **Not a place for parameter search.** Every threshold in `config.py` is
  pre-registered. Tuning them against the same data the study runs on destroys
  the study. See "Changing parameters" below.

## Architecture

```
src/mesproto/
  config.py     all constants: contract specs, cost model, session times,
                setup thresholds, statistical parameters. No magic numbers
                anywhere else — if you find one, move it here.
  levels.py     loaders (CSV / Databento TBBO / reference closes) -> bars;
                volume profile and value area; SessionLevels per RTH day
                (incl. point_scale for SPY); day-type classification.
  sources.py    network access to FRED and federalreserve.gov (scripts only).
  news.py       scheduled-release calendar (FRED + federalreserve.gov): parsers,
                the coverage-aware NewsCalendar, and the FOMC fetcher.
  signals.py    S1, IB_FAIL and S3 generators; pessimistic fill simulation.
                S1 and IB_FAIL share one walk of the tape, so a break yields
                at most one of them.
  schema.py     the canonical trade-log shape, shared by hand-logged replay
                and generated signals; validation.
  evaluate.py   R computation, session-block bootstrap, futility gates. CLI.
tests/          run directly (python tests/test_x.py) or with pytest.
scripts/        run_pipeline.py — bars to evaluation in one command.
                fetch_news_calendar.py — builds data/news_calendar.csv (network).
                fetch_reference_closes.py — builds data/sp500_close.csv for
                SPY point scaling (network).
docs/           protocol.html — the study design these tools serve.
```

Data flow: `bars -> build_sessions -> run_all -> fills_to_log -> compute_r -> report`.

## Invariants — do not violate these

1. **No look-ahead, ever.** Day type is classified at `DAY_TYPE_CUTOFF` (11:00
   ET) using only bars before it. S3 entries are gated on
   `_otf_confirm_time`, which is the close of the 30-minute bar that confirmed
   one-timeframing — not the moment the run began in hindsight.
   `tests/test_levels.py::test_no_lookahead` mutates the tape after the cutoff
   and asserts nothing upstream changes. **If you touch classification, that
   test must still pass.** Generators are held to the same standard by
   `assert_signals_survive_truncation` in `test_signals.py`: every signal must
   be reproducible, unchanged, from the tape cut at its own bar.
2. **Missing data is `None`, never a plausible substitute.** SPY has no
   overnight session, so `on_high`/`on_range_pos` are `None` and `s5_gate()` is
   `False`. Do not "fix" this by using the 09:30 open, the prior close, or
   extended-hours equity bars. The same holds for the news calendar: a
   session outside its declared coverage, or an event type it does not carry,
   is `None` — never a quiet day. And for SPY point scaling: a session with
   no prior-day reference close has `point_scale=None` and generates nothing;
   never substitute a fixed 1/10 or today's close.
3. **Fill conventions stay pessimistic.** Limit entries require trading
   *strictly through* the level. Stop entries pay a tick. When one bar contains
   both stop and target, the **stop** is assumed first and `ambiguous_bar` is
   flagged. The fill bar is checked for the stop and never for the target.
   Every one of these is pinned by a test in `test_signals.py`. A
   request to make fills "more realistic" is almost always a request to make
   them more favorable — push back and ask for the specific evidence.
4. **`checklist_ok` is False when a condition could not be verified.** A signal
   generated from bar data with no delta has `delta_confirmed=None`, therefore
   `checklist_ok=False`, therefore it is excluded from the primary analysis.
   Do not coerce `None` to `True`.
5. **Costs are always modeled.** `compute_r` subtracts commission and stop
   slippage. Never report gross R as if it were net.
6. **One contract month at a time.** Back-adjusted continuous series shift
   historical prices and silently corrupt every level in this repo. If someone
   passes a continuous symbol, warn loudly.
7. **The bootstrap resamples sessions, not trades.** Trades within a session
   share a regime and share the trader's state. Switching to an i.i.d. bootstrap
   tightens every interval by roughly a third and is wrong.
8. **One position per setup, one order per IB edge.** `run_session` resolves
   signals in time order through `_simulate_one_position_per_setup` and skips
   any whose setup still has an order working or a position open; each IB edge
   gives S1 and IB_FAIL one order per session. Both came from the first real
   tape (August 2026 ES), where one edge was sold six times in an hour. Logging
   several attempts at one idea as independent trades inflates the sample and
   the confidence in it.
9. **Gates are decided at fixed checkpoints, on the trades that existed then.**
   `futility_verdict` evaluates n = 60 on the first 60 primary trades and
   n = 150 on the first 150; a kill stays a kill. Burn-in (the first 30 trades
   of each setup, by time) counts toward those checkpoints: since the
   2026-09-15 amendment it is flagged `burn_in=True`, not removed. Re-deciding
   on the whole sample every time the report runs is an uncorrected
   sequential test.

## Changing parameters

Thresholds in `config.py` are pre-registered. Changing one **restarts the
sample for that setup** — old and new trades cannot be pooled. If you change
one:

- say so explicitly in the commit message,
- note which setups' samples are invalidated,
- do not quietly re-run the evaluation on the combined data.

The single exception is the cost model, which should be updated to match the
broker's actual fills whenever better information arrives. Raising costs is
always safe; lowering them needs a statement from the broker.

## Definition of done for a new setup

A setup generator is complete when all of the following hold:

- [ ] Every trigger condition from `docs/protocol.html` maps to a named key in
      `Signal.checklist`, and any condition the data cannot verify is `None`.
- [ ] It emits nothing before its conditions are knowable in real time, with a
      test proving it (mirror `test_s3_never_enters_before_confirmation`, and
      run it through `assert_signals_survive_truncation`).
- [ ] Stops respect the setup's floor and cap; structure beyond the cap means
      **no trade**, never a widened stop.
- [ ] The generator returns `[]` when its day gate fails, with a test.
- [ ] Output flows through `fills_to_log` and passes `schema.validate`.
- [ ] Its stand-down windows (news releases, time-of-day) are enforced from
      `config.py`, not hardcoded.
- [ ] Every threshold in points is multiplied by `lv.point_scale`, and the
      generator returns `[]` when that is `None`.

## Testing

```bash
PYTHONPATH=src python3 tests/test_levels.py
PYTHONPATH=src python3 tests/test_signals.py
PYTHONPATH=src python3 tests/test_evaluate.py
# or
pip install -e ".[dev]" && pytest
```

Tests use synthetic price paths with known properties — no network, no data
files. Keep it that way; a test suite that needs market data stops being run.

## Things people ask for that you should decline or push back on

| Request | Why it is wrong | What to offer instead |
|---|---|---|
| "Optimize the thresholds" | Destroys the pre-registration; guarantees overfit | An out-of-sample split, or a new pre-registered variant with its own sample |
| "Assume target hits first on ambiguous bars" | Manufactures edge | Report the ambiguous share; if it is large, the answer is replay |
| "Backtest S4 from bars" | Absorption is not visible in OHLCV | Databento MBP-10, or keep it in manual replay |
| "Use a continuous contract for more history" | Corrupts every level | Per-expiry data with explicit roll handling |
| "Drop the losing trades before the cutoff, they were learning" | Post-hoc exclusion | The 30-trade burn-in flag, applied uniformly, and the report's "without burn-in" comparison line |
| "Add live order routing" | Out of scope, and unreviewed | Nothing — this repo does not trade |

## Style

Python 3.11+, standard library plus numpy/pandas/scipy. Type hints on public
functions. Docstrings explain *why* a convention exists, not what the line does.
Keep modules importable without side effects; CLIs live behind
`if __name__ == "__main__"`.
