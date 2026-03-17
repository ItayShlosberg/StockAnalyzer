"""
Centralized configuration for the momentum stock screener.

All tunable thresholds, constants, and environment loading live here
so the rest of the codebase imports from a single source of truth.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = "gpt-4o"
LLM_TEMPERATURE: float = 0.2

RESULTS_DIR: Path = _PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ScreenerConfig:
    """All numeric thresholds used by the quantitative screener."""

    # --- Universe filters (Phase 1: UniverseLoader) ---
    market_cap_min: int = 500_000_000
    market_cap_max: int = 25_000_000_000
    min_avg_volume_3m: int = 200_000

    # --- Technical filters (Phase 1: QuantitativeFilter) ---
    min_avg_daily_dollar_volume: int = 10_000_000
    max_pct_above_breakout: int = 30
    min_relative_volume: float = 1.5
    max_pct_from_52wk_high: float = 8.0
    breakout_lookback_days: int = 20
    recent_breakout_window: int = 5
    ohlcv_history_days: int = 260

    # --- Qualitative score thresholds (Phase 2/3: Validator) ---
    min_early_stage_timing_score: int = 6
    min_narrative_freshness_score: int = 5

    # --- Data fetching ---
    market_data_chunk_size: int = 100
    universe_page_size: int = 250
    universe_max_tickers: int = 500


DEFAULT_CONFIG = ScreenerConfig()

MEGA_CAP_BLACKLIST: set[str] = {
    # FAANG / Mag-7
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    # Large-cap stalwarts
    "BRK.A", "BRK.B", "UNH", "JNJ", "JPM", "V", "MA", "XOM", "PG",
    "HD", "AVGO", "COST", "LLY", "ABBV", "MRK", "PEP", "KO", "WMT",
    # Large-cap tech / SaaS consensus names
    "CRM", "ORCL", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "CSCO",
    # Crowd-favorite momentum names
    "PLTR", "SNOW", "PANW", "NOW", "UBER", "ABNB", "CRWD", "DDOG",
    "NET", "ZS", "COIN", "MSTR", "SMCI",
}
