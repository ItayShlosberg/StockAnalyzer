"""
LLMAnalyzer — sends pre-filtered candidates to the LLM for qualitative analysis.

Candidates are processed in small batches (3-4 at a time) to ensure the model
actually triggers web searches for each stock, rather than skipping search when
given a large list.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import pandas as pd
from openai import OpenAI

from stock_analyzer.analyzer.models import (
    CandidateAnalysis,
    RejectedCandidate,
    ScreenerLLMResult,
    WatchlistCandidate,
)
from stock_analyzer.analyzer.prompts import SYSTEM_PROMPT, build_user_prompt
from stock_analyzer.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

logger = logging.getLogger(__name__)

BATCH_SIZE = 4


class LLMAnalyzer:
    """Runs Phase 2 qualitative analysis on quantitatively-screened candidates."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set — check .env file")
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def analyze(self, candidates_df: pd.DataFrame) -> ScreenerLLMResult:
        """
        Analyze candidates in small batches so the LLM performs focused web
        searches per stock rather than skipping search for a large batch.
        """
        today = datetime.now().strftime("%B %d, %Y")
        tickers = list(candidates_df.index)
        n_batches = math.ceil(len(tickers) / BATCH_SIZE)

        all_approved: list[CandidateAnalysis] = []
        all_rejected: list[RejectedCandidate] = []
        all_watchlist: list[WatchlistCandidate] = []
        all_reasoning: list[str] = []
        all_red_flags: list[str] = []
        all_conditions: list[str] = []
        total_searches = 0

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(tickers))
            batch_tickers = tickers[start:end]
            batch_df = candidates_df.loc[batch_tickers]

            logger.info(
                "LLM batch %d/%d: analyzing %s",
                batch_idx + 1, n_batches, ", ".join(batch_tickers),
            )

            result, n_searches = self._analyze_batch(batch_df, today)
            total_searches += n_searches

            all_approved.extend(result.analyzed_candidates)
            all_rejected.extend(result.rejected_candidates)
            all_watchlist.extend(result.watchlist)
            if result.best_picks_reasoning:
                all_reasoning.append(result.best_picks_reasoning)
            if result.red_flags_summary:
                all_red_flags.append(result.red_flags_summary)
            if result.market_conditions_note:
                all_conditions.append(result.market_conditions_note)

        logger.info(
            "LLM totals: %d approved, %d rejected, %d watchlist, %d web searches",
            len(all_approved), len(all_rejected), len(all_watchlist), total_searches,
        )

        return ScreenerLLMResult(
            analyzed_candidates=sorted(all_approved, key=lambda c: c.conviction_score, reverse=True),
            rejected_candidates=all_rejected,
            watchlist=all_watchlist,
            best_picks_reasoning="\n\n".join(all_reasoning) if all_reasoning else "No strong picks in this run.",
            red_flags_summary="\n".join(all_red_flags) if all_red_flags else "",
            market_conditions_note="\n".join(all_conditions) if all_conditions else None,
        )

    def _analyze_batch(
        self, batch_df: pd.DataFrame, today: str,
    ) -> tuple[ScreenerLLMResult, int]:
        """Run a single LLM call for a small batch. Returns (result, n_searches)."""
        user_prompt = build_user_prompt(batch_df, today)
        tickers_str = ", ".join(batch_df.index)

        response = self._client.responses.parse(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            input=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[{"type": "web_search_preview"}],
            text_format=ScreenerLLMResult,
        )

        n_searches = sum(1 for item in response.output if item.type == "web_search_call")
        logger.info("  Batch [%s]: %d web searches", tickers_str, n_searches)

        result: ScreenerLLMResult = response.output_parsed
        for c in result.analyzed_candidates:
            logger.info("  APPROVED %s (conviction %d)", c.ticker, c.conviction_score)
        for r in result.rejected_candidates:
            logger.info("  REJECTED %s: %s", r.ticker, r.rejection_reason[:80])

        return result, n_searches
