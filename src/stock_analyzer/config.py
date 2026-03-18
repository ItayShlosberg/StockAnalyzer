"""
Centralized configuration for the momentum stock screener.

Loads settings from config.yaml (with sensible defaults if missing).
Exposes per-step LLM model selection so different pipeline phases can
use different models (e.g. gpt-4o for research, o3-mini for judgment).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


def _load_yaml() -> dict:
    """Load config.yaml from project root, returning empty dict if missing."""
    yaml_path = _PROJECT_ROOT / "config.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


_YAML = _load_yaml()
_LLM_YAML = _YAML.get("llm", {})
_SCREENER_YAML = _YAML.get("screener", {})
_VALIDATION_YAML = _YAML.get("validation", {})
_PATHS_YAML = _YAML.get("paths", {})


# ---------------------------------------------------------------------------
# Per-step LLM configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """Per-step model and temperature settings."""

    research_model: str = "gpt-4o"
    research_temperature: float = 0.2
    analysis_model: str = "gpt-4o"
    analysis_temperature: float = 0.2
    institutional_validation_model: str = "o3-mini"
    devils_advocate_model: str = "gpt-4o"
    devils_advocate_temperature: float = 0.2
    batch_size: int = 4
    max_llm_workers: int = 3


DEFAULT_LLM_CONFIG = LLMConfig(
    research_model=_LLM_YAML.get("research_model", "gpt-4o"),
    research_temperature=float(_LLM_YAML.get("research_temperature", 0.2)),
    analysis_model=_LLM_YAML.get("analysis_model", "gpt-4o"),
    analysis_temperature=float(_LLM_YAML.get("analysis_temperature", 0.2)),
    institutional_validation_model=_LLM_YAML.get("institutional_validation_model", "o3-mini"),
    devils_advocate_model=_LLM_YAML.get("devils_advocate_model", "gpt-4o"),
    devils_advocate_temperature=float(_LLM_YAML.get("devils_advocate_temperature", 0.2)),
    batch_size=int(_LLM_YAML.get("batch_size", 4)),
    max_llm_workers=int(_LLM_YAML.get("max_llm_workers", 3)),
)

# Legacy aliases used by report.py for logging which model was used
LLM_MODEL: str = DEFAULT_LLM_CONFIG.analysis_model
LLM_TEMPERATURE: float = DEFAULT_LLM_CONFIG.analysis_temperature


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR: Path = _PROJECT_ROOT / _PATHS_YAML.get("results_dir", "results")
TRACKING_DIR: Path = _PROJECT_ROOT / _PATHS_YAML.get("tracking_dir", "results/tracking")


# ---------------------------------------------------------------------------
# Screener thresholds
# ---------------------------------------------------------------------------

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
    max_pct_from_52wk_high: float = 15.0
    max_pct_below_200dma: float = 5.0
    breakout_lookback_days: int = 20
    recent_breakout_window: int = 10
    ohlcv_history_days: int = 260
    min_breakout_volume_ratio: float = 1.3

    # --- Qualitative score thresholds (Phase 2/3: Validator) ---
    min_early_stage_timing_score: int = 6
    min_narrative_freshness_score: int = 5

    # --- Data fetching ---
    market_data_chunk_size: int = 100
    universe_page_size: int = 250
    max_download_workers: int = 4


def _build_screener_config() -> ScreenerConfig:
    if not _SCREENER_YAML:
        return ScreenerConfig()
    return ScreenerConfig(**{
        k: type(getattr(ScreenerConfig, k))(v)
        for k, v in _SCREENER_YAML.items()
        if hasattr(ScreenerConfig, k)
    })


DEFAULT_CONFIG = _build_screener_config()


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for post-LLM validation steps."""

    min_institutional_validation_score: int = 6
    min_conviction_for_da_override: int = 80


DEFAULT_VALIDATION_CONFIG = ValidationConfig(
    min_institutional_validation_score=int(
        _VALIDATION_YAML.get("min_institutional_validation_score", 6)),
    min_conviction_for_da_override=int(
        _VALIDATION_YAML.get("min_conviction_for_da_override", 80)),
)


# ---------------------------------------------------------------------------
# Blacklist
# ---------------------------------------------------------------------------

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
