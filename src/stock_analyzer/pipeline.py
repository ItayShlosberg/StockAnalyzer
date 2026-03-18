"""
ScreenerPipeline — orchestrates the full momentum screener.

Phase 1:    Quantitative screening (universe -> market data -> technicals -> filter)
Phase 1.5:  News fetching + market regime context
Phase 2:    Two-step LLM analysis (research with web search -> structured analysis)
Phase 2.75: Institutional validation (reasoning model + web search)
Phase 2.5:  Devil's Advocate independent validation
Phase 3:    Programmatic validation, forward tracking, and export
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_analyzer.analyzer.devils_advocate import DevilsAdvocateValidator
from stock_analyzer.analyzer.institutional_validator import InstitutionalValidator
from stock_analyzer.analyzer.llm_analyzer import LLMAnalyzer
from stock_analyzer.analyzer.models import CandidateAnalysis
from stock_analyzer.analyzer.validator import ResultValidator, ValidationResult
from stock_analyzer.config import (
    DEFAULT_CONFIG,
    DEFAULT_LLM_CONFIG,
    DEFAULT_VALIDATION_CONFIG,
    RESULTS_DIR,
    ScreenerConfig,
)
from stock_analyzer.export.report import ReportExporter
from stock_analyzer.screener.market_data import MarketDataFetcher
from stock_analyzer.screener.news import NewsItem, fetch_news
from stock_analyzer.screener.quantitative_filter import QuantitativeFilter
from stock_analyzer.screener.technical import TechnicalCalculator
from stock_analyzer.screener.universe import UniverseLoader
from stock_analyzer.tracker.performance import ForwardTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full output of a pipeline run, for programmatic access."""

    validation: ValidationResult
    quant_candidates: pd.DataFrame
    csv_path: Path
    xlsx_path: Path
    elapsed_seconds: float
    track_record: pd.DataFrame | None = None


class ScreenerPipeline:
    """Ties all phases into a single run() call."""

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
        self._llm_analyzer = LLMAnalyzer(DEFAULT_LLM_CONFIG)
        self._institutional_validator = InstitutionalValidator(DEFAULT_LLM_CONFIG)
        self._devils_advocate = DevilsAdvocateValidator(DEFAULT_LLM_CONFIG)
        self._validator = ResultValidator(config)
        self._exporter = ReportExporter(output_dir)
        self._tracker = ForwardTracker()

    def run(self) -> PipelineResult:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PIPELINE START — config: %s", self._config)
        logger.info("=" * 60)

        # ── Track record of previous picks ────────────────────────────────
        print("\n[Pre-run] Evaluating previous picks performance...")
        track_record = self._tracker.evaluate_previous()
        if not track_record.empty:
            print(f"  Tracked {len(track_record)} previous picks (see log for details)")
        else:
            print("  No previous picks to track yet")

        # ── Phase 1: Quantitative Screening ──────────────────────────────
        print("\n[Phase 1] Loading full stock universe (exhaustive triple-query scan)...")
        tickers, metadata = self._universe_loader.load()
        logger.debug("Universe tickers (first 20): %s", tickers[:20])
        print(f"  Universe: {len(tickers)} unique tickers")

        print("[Phase 1] Downloading market data...")
        bundle = self._data_fetcher.fetch(tickers, metadata)
        print(f"  OHLCV: {len(bundle.ohlcv)} tickers")

        print("[Phase 1] Computing technical indicators (MAs, breakout, ATR, sector strength)...")
        enriched = self._technical_calc.compute(bundle)
        print(f"  Enriched: {len(enriched)} tickers with full technical data")

        print("[Phase 1] Applying quantitative filters...")
        candidates = self._quant_filter.apply(enriched)
        print(f"  Candidates after quant filter: {len(candidates)}")

        if not candidates.empty:
            display_cols = [
                "last_close", "pct_from_52wk_high", "rel_volume_3d_max",
                "pct_above_breakout", "breakout_timeframe", "breakout_volume_ratio",
                "sector_relative_strength", "market_cap", "sector", "short_name",
            ]
            available = [c for c in display_cols if c in candidates.columns]
            logger.info("Quant candidates:\n%s", candidates[available].to_string())

        if candidates.empty:
            print("\n  No stocks passed quantitative filters. Try loosening thresholds.")
            elapsed = time.time() - t0
            empty_vr = ValidationResult()
            csv, xlsx = self._exporter.export(empty_vr, candidates)
            return PipelineResult(empty_vr, candidates, csv, xlsx, elapsed, track_record)

        # ── Phase 1.5: News + Market Context ─────────────────────────────
        print(f"\n[Phase 1.5] Fetching news for {len(candidates)} candidates...")
        news_data = fetch_news(list(candidates.index))
        n_with_news = sum(1 for v in news_data.values() if v)
        print(f"  {n_with_news}/{len(candidates)} candidates have recent news headlines")

        market_context = self._compute_market_context(enriched)

        # ── Phase 2: Two-Step LLM Analysis ────────────────────────────────
        n_batches = -(-len(candidates) // DEFAULT_LLM_CONFIG.batch_size)
        print(f"\n[Phase 2] Two-step LLM analysis: {len(candidates)} candidates in {n_batches} batches")
        print("  Step 1: Web research (unstructured, model searches freely)")
        print("  Step 2: Structured analysis (using research results)")
        llm_output = self._llm_analyzer.analyze(candidates, news_data, market_context)
        llm_result = llm_output.result
        research_log = llm_output.research_log

        # ── Phase 2.75: Institutional Validation ─────────────────────────
        if llm_result.analyzed_candidates:
            print(f"\n[Phase 2.75] Institutional validation of {len(llm_result.analyzed_candidates)} "
                  f"candidates ({DEFAULT_LLM_CONFIG.institutional_validation_model})...")
            iv_result = self._institutional_validator.validate(llm_result.analyzed_candidates)

            iv_by_ticker = {a.ticker.upper(): a for a in iv_result.assessments}
            min_score = DEFAULT_VALIDATION_CONFIG.min_institutional_validation_score

            passed_iv: list[CandidateAnalysis] = []
            for c in llm_result.analyzed_candidates:
                assessment = iv_by_ticker.get(c.ticker.upper())
                if assessment is None:
                    logger.warning("  IV: no assessment for %s — passing through", c.ticker)
                    passed_iv.append(c)
                elif assessment.is_genuinely_early_stage and assessment.institutional_score >= min_score:
                    print(f"  IV PASS: {c.ticker} — score {assessment.institutional_score}/10")
                    passed_iv.append(c)
                else:
                    reason = (f"score {assessment.institutional_score}/10, "
                              f"consensus={assessment.analyst_consensus}, "
                              f"target={assessment.price_target_vs_current}")
                    print(f"  IV REJECT: {c.ticker} — {reason}")
                    logger.info("  IV REJECT %s: %s", c.ticker, assessment.reasoning[:200])

            print(f"  Institutional validation: {len(passed_iv)}/{len(llm_result.analyzed_candidates)} passed")
            approved_for_da = passed_iv
        else:
            approved_for_da = []

        # ── Phase 2.5: Devil's Advocate ───────────────────────────────────
        if approved_for_da:
            print(f"\n[Phase 2.5] Devil's Advocate challenging {len(approved_for_da)} candidates...")
            da_result = self._devils_advocate.challenge(
                approved_for_da, news_data, research_log,
            )
        else:
            da_result = None

        # ── Phase 3: Validation & Export ──────────────────────────────────
        print("\n[Phase 3] Validating against quant data + devil's advocate challenges...")
        llm_result_for_validation = llm_result
        llm_result_for_validation.analyzed_candidates = approved_for_da
        validation = self._validator.validate(llm_result_for_validation, candidates, da_result)

        self._tracker.save_picks(validation.passed, candidates)

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

        return PipelineResult(validation, candidates, csv_path, xlsx_path, elapsed, track_record)

    def _compute_market_context(self, enriched: pd.DataFrame) -> str:
        """Compute SPY-based market regime + breadth for LLM context."""
        lines: list[str] = []

        try:
            spy = yf.download("SPY", period="260d", progress=False)
            if not spy.empty:
                spy_close = spy["Close"].squeeze()
                spy_last = float(spy_close.iloc[-1])
                spy_sma50 = float(spy_close.rolling(50).mean().iloc[-1])
                spy_sma200 = float(spy_close.rolling(200).mean().iloc[-1])
                spy_20d_ret = ((spy_last / float(spy_close.iloc[-21])) - 1) * 100

                regime = "BULL" if spy_last > spy_sma50 > spy_sma200 else (
                    "CORRECTION" if spy_last < spy_sma50 else "MIXED"
                )
                lines.append(f"SPY: ${spy_last:.2f} | 50 DMA: ${spy_sma50:.2f} | 200 DMA: ${spy_sma200:.2f}")
                lines.append(f"SPY 20-day return: {spy_20d_ret:+.1f}% | Regime: {regime}")
        except Exception:
            logger.warning("Failed to fetch SPY data for market context", exc_info=True)

        if "above_50dma" in enriched.columns:
            breadth = (enriched["above_50dma"] == True).mean() * 100  # noqa: E712
            lines.append(f"Breadth: {breadth:.0f}% of universe above 50 DMA")
            if breadth < 30:
                lines.append("WARNING: Weak market breadth — fewer early-stage opportunities expected")

        context = "\n".join(lines)
        if context:
            logger.info("Market context:\n%s", context)
        return context
