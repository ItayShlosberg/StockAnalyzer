"""
ScreenerPipeline — orchestrates the three-phase momentum screener.

Phase 1: Quantitative screening (universe → market data → technicals → filter)
Phase 2: Qualitative LLM analysis (web search + structured output)
Phase 3: Validation and export (cross-check → CSV/XLSX/console)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_analyzer.analyzer.llm_analyzer import LLMAnalyzer
from stock_analyzer.analyzer.validator import ResultValidator, ValidationResult
from stock_analyzer.config import DEFAULT_CONFIG, RESULTS_DIR, ScreenerConfig
from stock_analyzer.export.report import ReportExporter
from stock_analyzer.screener.market_data import MarketDataFetcher
from stock_analyzer.screener.quantitative_filter import QuantitativeFilter
from stock_analyzer.screener.technical import TechnicalCalculator
from stock_analyzer.screener.universe import UniverseLoader

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full output of a pipeline run, for programmatic access."""

    validation: ValidationResult
    quant_candidates: pd.DataFrame
    csv_path: Path
    xlsx_path: Path
    elapsed_seconds: float


class ScreenerPipeline:
    """Ties Phase 1 → 2 → 3 into a single run() call."""

    def __init__(
        self,
        config: ScreenerConfig = DEFAULT_CONFIG,
        output_dir: Path = RESULTS_DIR,
    ) -> None:
        self._config = config
        self._universe_loader = UniverseLoader(config)
        self._data_fetcher = MarketDataFetcher(config)
        self._technical_calc = TechnicalCalculator(config)
        self._quant_filter = QuantitativeFilter(config)
        self._llm_analyzer = LLMAnalyzer()
        self._validator = ResultValidator(config)
        self._exporter = ReportExporter(output_dir)

    def run(self) -> PipelineResult:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PIPELINE START — config: %s", self._config)
        logger.info("=" * 60)

        # ── Phase 1: Quantitative Screening ──────────────────────────────
        print("\n[Phase 1] Loading stock universe...")
        tickers, metadata = self._universe_loader.load()
        logger.debug("Universe tickers: %s", tickers[:20])
        print(f"  Universe: {len(tickers)} tickers (metadata from screener)")

        print("[Phase 1] Downloading market data...")
        bundle = self._data_fetcher.fetch(tickers, metadata)
        print(f"  OHLCV: {len(bundle.ohlcv)} tickers")

        print("[Phase 1] Computing technical indicators...")
        enriched = self._technical_calc.compute(bundle)
        print(f"  Enriched: {len(enriched)} tickers with full technical data")

        print("[Phase 1] Applying quantitative filters...")
        candidates = self._quant_filter.apply(enriched)
        print(f"  Candidates after quant filter: {len(candidates)}")

        if not candidates.empty:
            logger.info("Quant candidates:\n%s", candidates[
                ["last_close", "pct_from_52wk_high", "rel_volume_3d_max",
                 "pct_above_breakout", "market_cap", "sector", "short_name"]
            ].to_string())

        if candidates.empty:
            print("\n  No stocks passed quantitative filters. Try loosening thresholds.")
            elapsed = time.time() - t0
            empty_vr = ValidationResult()
            csv, xlsx = self._exporter.export(empty_vr, candidates)
            return PipelineResult(empty_vr, candidates, csv, xlsx, elapsed)

        # ── Phase 2: Qualitative LLM Analysis ────────────────────────────
        n_batches = -(-len(candidates) // 4)
        print(f"\n[Phase 2] Analyzing {len(candidates)} candidates in {n_batches} LLM batches...")
        print("  (Each batch triggers independent web searches)")
        llm_result = self._llm_analyzer.analyze(candidates)

        # ── Phase 3: Validation & Export ─────────────────────────────────
        print("\n[Phase 3] Validating LLM results against quantitative data...")
        validation = self._validator.validate(llm_result, candidates)

        print("[Phase 3] Exporting results...")
        csv_path, xlsx_path = self._exporter.export(validation, candidates)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(
            "PIPELINE COMPLETE in %.0fs — %d approved, %d rejected, %d watchlist",
            elapsed, len(validation.passed), len(validation.failed),
            len(llm_result.watchlist),
        )
        logger.info("=" * 60)
        print(f"\nPipeline complete in {elapsed:.0f}s")
        print(f"  CSV:  {csv_path}")
        print(f"  XLSX: {xlsx_path}")

        return PipelineResult(validation, candidates, csv_path, xlsx_path, elapsed)
