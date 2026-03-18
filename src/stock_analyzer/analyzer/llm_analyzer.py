"""
LLMAnalyzer — two-step qualitative analysis of pre-screened candidates.

Step 1 (Research): Call the LLM with web search enabled but NO structured
output.  The model freely searches for news/catalysts per stock and returns
raw research notes.  Because there is no schema to fill, the model actually
triggers web searches instead of guessing from training data.

Step 2 (Analysis): Feed the real research from Step 1 (plus quant data and
yfinance news headlines) into a second LLM call WITH structured output.
The model now judges based on real, searched information.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from openai import OpenAI

from stock_analyzer.analyzer.models import (
    CandidateAnalysis,
    RejectedCandidate,
    ScreenerLLMResult,
    WatchlistCandidate,
)
from stock_analyzer.analyzer.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_research_prompt,
)
from stock_analyzer.config import DEFAULT_LLM_CONFIG, OPENAI_API_KEY, LLMConfig
from stock_analyzer.screener.news import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class _BatchResult:
    """Internal container for a single batch's output."""

    research: str
    n_searches: int
    result: ScreenerLLMResult


@dataclass
class LLMAnalysisOutput:
    """Wraps the LLM result together with accumulated research text."""

    result: ScreenerLLMResult
    research_log: str


class LLMAnalyzer:
    """Two-step qualitative analysis: research first, then structured judgment."""

    def __init__(self, llm_config: LLMConfig = DEFAULT_LLM_CONFIG) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set — check .env file")
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._llm = llm_config

    def _process_batch(
        self,
        batch_idx: int,
        n_batches: int,
        batch_df: pd.DataFrame,
        batch_news: dict[str, list[NewsItem]],
        today: str,
        market_context: str,
    ) -> _BatchResult:
        """Run Step 1 (research) + Step 2 (analysis) for a single batch."""
        tickers_str = ", ".join(batch_df.index)
        logger.info("LLM batch %d/%d: %s", batch_idx + 1, n_batches, tickers_str)

        research, n_searches = self._step1_research(batch_df, batch_news, today)
        result = self._step2_analyze(batch_df, batch_news, research, today, market_context)
        return _BatchResult(research=research, n_searches=n_searches, result=result)

    def analyze(
        self,
        candidates_df: pd.DataFrame,
        news_data: dict[str, list[NewsItem]],
        market_context: str = "",
    ) -> LLMAnalysisOutput:
        """
        Analyze candidates in small batches using a two-step process.
        Batches are processed concurrently via ThreadPoolExecutor.

        Returns an LLMAnalysisOutput containing both the structured result
        and the accumulated research log (for downstream DA / validation).
        """
        today = datetime.now().strftime("%B %d, %Y")
        tickers = list(candidates_df.index)
        n_batches = math.ceil(len(tickers) / self._llm.batch_size)
        max_workers = self._llm.max_llm_workers

        logger.info(
            "LLM analysis: %d candidates in %d batches (%d workers)",
            len(tickers), n_batches, max_workers,
        )

        batch_results: dict[int, _BatchResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for batch_idx in range(n_batches):
                start = batch_idx * self._llm.batch_size
                end = min(start + self._llm.batch_size, len(tickers))
                batch_tickers = tickers[start:end]
                batch_df = candidates_df.loc[batch_tickers]
                batch_news = {t: news_data.get(t, []) for t in batch_tickers}

                future = pool.submit(
                    self._process_batch,
                    batch_idx, n_batches, batch_df, batch_news, today, market_context,
                )
                futures[future] = batch_idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_results[idx] = future.result()
                except Exception:
                    logger.exception("LLM batch %d failed", idx + 1)

        all_approved: list[CandidateAnalysis] = []
        all_rejected: list[RejectedCandidate] = []
        all_watchlist: list[WatchlistCandidate] = []
        all_reasoning: list[str] = []
        all_red_flags: list[str] = []
        all_conditions: list[str] = []
        all_research: list[str] = []
        total_searches = 0

        for batch_idx in sorted(batch_results):
            br = batch_results[batch_idx]
            all_research.append(br.research)
            total_searches += br.n_searches
            all_approved.extend(br.result.analyzed_candidates)
            all_rejected.extend(br.result.rejected_candidates)
            all_watchlist.extend(br.result.watchlist)
            if br.result.best_picks_reasoning:
                all_reasoning.append(br.result.best_picks_reasoning)
            if br.result.red_flags_summary:
                all_red_flags.append(br.result.red_flags_summary)
            if br.result.market_conditions_note:
                all_conditions.append(br.result.market_conditions_note)

        logger.info(
            "LLM totals: %d approved, %d rejected, %d watchlist, %d web searches across %d batches",
            len(all_approved), len(all_rejected), len(all_watchlist), total_searches, n_batches,
        )

        llm_result = ScreenerLLMResult(
            analyzed_candidates=sorted(all_approved, key=lambda c: c.conviction_score, reverse=True),
            rejected_candidates=all_rejected,
            watchlist=all_watchlist,
            best_picks_reasoning="\n\n".join(all_reasoning) if all_reasoning else "No strong picks in this run.",
            red_flags_summary="\n".join(all_red_flags) if all_red_flags else "",
            market_conditions_note="\n".join(all_conditions) if all_conditions else None,
        )

        return LLMAnalysisOutput(
            result=llm_result,
            research_log="\n\n---\n\n".join(all_research) if all_research else "",
        )

    def _step1_research(
        self,
        batch_df: pd.DataFrame,
        batch_news: dict[str, list[NewsItem]],
        today: str,
    ) -> tuple[str, int]:
        """
        Step 1: Unstructured web search. No schema, so the model
        actually searches instead of guessing.

        Returns (research_text, n_web_searches).
        """
        prompt = build_research_prompt(batch_df, batch_news, today)
        tickers_str = ", ".join(batch_df.index)

        logger.info("  Step 1 (Research) for [%s]...", tickers_str)

        response = self._client.responses.create(
            model=self._llm.research_model,
            temperature=self._llm.research_temperature,
            input=[
                {"role": "developer", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "web_search_preview"}],
        )

        n_searches = sum(1 for item in response.output if item.type == "web_search_call")
        research_text = response.output_text

        logger.info("  Step 1 done: %d web searches, %d chars of research", n_searches, len(research_text))
        logger.debug("  Research output:\n%s", research_text[:2000])
        return research_text, n_searches

    def _step2_analyze(
        self,
        batch_df: pd.DataFrame,
        batch_news: dict[str, list[NewsItem]],
        research: str,
        today: str,
        market_context: str,
    ) -> ScreenerLLMResult:
        """
        Step 2: Structured analysis using the real research from Step 1.
        No web search tool — the model just judges the evidence.
        """
        prompt = build_analysis_prompt(batch_df, batch_news, research, today, market_context)
        tickers_str = ", ".join(batch_df.index)

        logger.info("  Step 2 (Analysis) for [%s]...", tickers_str)

        response = self._client.responses.parse(
            model=self._llm.analysis_model,
            temperature=self._llm.analysis_temperature,
            input=[
                {"role": "developer", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text_format=ScreenerLLMResult,
        )

        result: ScreenerLLMResult = response.output_parsed
        for c in result.analyzed_candidates:
            logger.info("  APPROVED %s (conviction %d)", c.ticker, c.conviction_score)
        for r in result.rejected_candidates:
            logger.info("  REJECTED %s: %s", r.ticker, r.rejection_reason[:80])

        return result
