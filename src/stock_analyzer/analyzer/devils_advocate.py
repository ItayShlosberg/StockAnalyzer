"""
DevilsAdvocateValidator — independent LLM cross-check on Phase 2 results.

Makes a separate LLM call with a skeptical persona that challenges each
approved candidate. Cross-references claimed catalysts against real news
headlines AND independently searches the web to verify claims.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from stock_analyzer.analyzer.models import CandidateAnalysis
from stock_analyzer.config import DEFAULT_LLM_CONFIG, OPENAI_API_KEY, LLMConfig
from stock_analyzer.screener.news import NewsItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for the devil's advocate response
# ---------------------------------------------------------------------------

class ChallengeItem(BaseModel):
    """Devil's advocate challenge for a single stock."""

    ticker: str = Field(description="Stock ticker being challenged")
    catalyst_verified: bool = Field(
        description="Does the claimed catalyst actually appear in the provided news headlines "
        "or web research? True = verified, False = unverified/fabricated",
    )
    strongest_bear_case: str = Field(
        description="The single strongest reason NOT to buy this stock right now",
    )
    crowdedness_assessment: str = Field(
        description="Evidence of whether this stock is already widely discussed. "
        "Reference specific sources if found.",
    )
    institutional_consensus_check: str = Field(
        description="Based on your web search: is this stock already covered by multiple "
        "analysts? What is the consensus rating? Are price targets near current levels? "
        "If so, the opportunity is NOT early-stage.",
    )
    reject_recommendation: bool = Field(
        description="Should this stock be rejected? True only if there is strong "
        "negative evidence (fabricated catalyst, clearly crowded, late stage, "
        "consensus play with limited upside). "
        "False if the bear case is speculative or minor.",
    )
    confidence: int = Field(
        description="How confident are you in the rejection/approval? 1-10", ge=1, le=10,
    )


class DevilsAdvocateResult(BaseModel):
    """Full response from the devil's advocate analysis."""

    challenges: list[ChallengeItem] = Field(
        description="One challenge per approved candidate, in the same order as input",
    )
    overall_assessment: str = Field(
        description="High-level assessment of the batch quality",
    )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_PROMPT = """\
You are a skeptical equity analyst tasked with finding flaws in stock picks.
For each approved candidate, you will receive:
1. The analyst's qualitative assessment (catalyst, narrative, scores)
2. Real news headlines from Yahoo Finance
3. Web research conducted earlier

YOUR JOB: Challenge each pick by answering:

1. CATALYST VERIFICATION: Does the claimed catalyst actually appear in the
   provided news headlines or research?  If the analyst claims "Company X won
   a new contract on March 10" but no headline or research mentions this,
   mark catalyst_verified=false.  This is critical — fabricated catalysts
   are the #1 failure mode.

2. BEAR CASE: What is the strongest reason NOT to buy this stock?

3. CROWDEDNESS: Is there evidence this stock is already widely discussed?
   Check if the headlines suggest mainstream coverage.

4. RECOMMENDATION: Only recommend rejection if you find STRONG evidence:
   - Catalyst is clearly fabricated (not in any provided source)
   - Stock is obviously crowded/late-stage based on the sources
   - Major risk the analyst missed
   Do NOT reject just because you can construct a theoretical bear case.
   Every stock has risks — only reject on hard evidence.

5. INSTITUTIONAL CONSENSUS: Search the web for this stock. Is it already
   covered by multiple analysts with "Buy" or "Hold" consensus? If analysts
   have price targets near current levels, the opportunity is NOT early-stage.
   If the stock has widespread analyst coverage and institutional ownership,
   note this as evidence AGAINST early-stage status.\
"""


class DevilsAdvocateValidator:
    """Runs an independent LLM challenge on Phase 2 approved candidates."""

    def __init__(self, llm_config: LLMConfig = DEFAULT_LLM_CONFIG) -> None:
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model = llm_config.devils_advocate_model
        self._temperature = llm_config.devils_advocate_temperature

    def challenge(
        self,
        approved: list[CandidateAnalysis],
        news_data: dict[str, list[NewsItem]],
        research_log: str,
    ) -> DevilsAdvocateResult:
        """
        Challenge each approved candidate with a skeptical second opinion.
        Now includes web search for independent verification.
        """
        if not approved:
            return DevilsAdvocateResult(challenges=[], overall_assessment="No candidates to challenge.")

        prompt = self._build_prompt(approved, news_data, research_log)

        logger.info("Devil's Advocate: challenging %d approved candidates...", len(approved))

        response = self._client.responses.parse(
            model=self._model,
            temperature=self._temperature,
            input=[
                {"role": "developer", "content": DEVILS_ADVOCATE_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "web_search_preview"}],
            text_format=DevilsAdvocateResult,
        )

        result: DevilsAdvocateResult = response.output_parsed

        for ch in result.challenges:
            status = "REJECT" if ch.reject_recommendation else "PASS"
            verified = "verified" if ch.catalyst_verified else "UNVERIFIED"
            logger.info(
                "  DA [%s] %s — catalyst %s — %s",
                ch.ticker, status, verified, ch.strongest_bear_case[:80],
            )

        return result

    @staticmethod
    def _build_prompt(
        approved: list[CandidateAnalysis],
        news_data: dict[str, list[NewsItem]],
        research_log: str,
    ) -> str:
        lines = [
            "Challenge each of these approved stock picks.",
            "Cross-reference the analyst's claims against the provided sources.",
            "Also search the web independently to verify catalyst claims and check",
            "analyst consensus / institutional positioning.",
            "",
        ]

        for c in approved:
            lines.append(f"--- {c.ticker} (Conviction: {c.conviction_score}/100) ---")
            lines.append(f"  Analyst's catalyst claim: {c.catalyst_summary}")
            lines.append(f"  Catalyst is new: {c.catalyst_is_new}")
            lines.append(f"  Narrative assessment: {c.narrative_summary}")
            lines.append(f"  Early-stage reasoning: {c.early_stage_reasoning}")
            lines.append(f"  Scores: timing={c.early_stage_timing_score} catalyst={c.catalyst_strength_score} "
                         f"freshness={c.narrative_freshness_score} excitement={c.momentum_excitement_score}")

            headlines = news_data.get(c.ticker, [])
            if headlines:
                lines.append(f"  Yahoo Finance headlines ({len(headlines)}):")
                for h in headlines:
                    lines.append(f"    [{h.published}] {h.title} ({h.publisher})")
            else:
                lines.append("  Yahoo Finance: no recent headlines")
            lines.append("")

        lines.extend([
            "=" * 50,
            "WEB RESEARCH (from earlier search):",
            "=" * 50,
            research_log if research_log else "(no web research available)",
            "",
            "Now challenge each candidate. Be skeptical but fair.",
            "Use web search to independently verify claims and check analyst consensus.",
        ])

        return "\n".join(lines)
