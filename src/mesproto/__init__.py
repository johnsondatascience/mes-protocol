"""mesproto — pre-screening and evaluation tooling for the MES replay protocol.

The protocol itself (the pre-registered rules, gates and sample-size math) is in
docs/protocol.html. This package is its instrumentation, not a trading system.

Pipeline:
    bars -> levels.build_sessions -> signals.run_all -> schema.fills_to_log
         -> evaluate.report
"""

from .config import CONTRACTS, ES, MES, SPY, Contract
from .levels import (
    Profile, SessionLevels, build_sessions, databento_symbology, load_csv_bars,
    load_databento_tbbo, load_dataframe_bars, load_news_dates, sessions_to_frame,
    tbbo_to_bars, volume_profile, volume_profile_from_trades,
)
from .schema import fills_to_log, validate, write_template
from .signals import Fill, Signal, generate_s1, generate_s3, run_all, simulate

__version__ = "0.1.0"

__all__ = [
    "CONTRACTS", "ES", "MES", "SPY", "Contract",
    "Profile", "SessionLevels", "build_sessions", "databento_symbology", "load_csv_bars",
    "load_databento_tbbo", "load_dataframe_bars", "load_news_dates", "sessions_to_frame",
    "tbbo_to_bars", "volume_profile", "volume_profile_from_trades",
    "Fill", "Signal", "generate_s1", "generate_s3", "run_all", "simulate",
    "fills_to_log", "validate", "write_template",
]
