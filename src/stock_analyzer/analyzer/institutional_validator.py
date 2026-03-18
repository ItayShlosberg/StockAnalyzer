"""
InstitutionalValidator — Phase 2.75 professional investor validation.

Two-step process per stock (mirrors the main LLM analyzer design):
  Step 1: gpt-4o with web search gathers analyst consensus, price targets,
          institutional ownership, valuation, and earnings data.
  Step 2: o3-mini (reasoning model) judges whether the opportunity is
          genuinely early-stage or an already-priced consensus play.

Stocks are validated INDIVIDUALLY to ensure thorough web research per stock.
Batching all stocks into one search call produces shallow data and misses
analyst consensus information for less-covered names.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel, Field

from stock_analyzer.analyzer.models import CandidateAnalysis
from stock_analyzer.config import DEFAULT_LLM_CONFIG, OPENAI_API_KEY, LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InstitutionalAssessment(BaseModel):
    """Professional investor assessment of a single candidate."""

    ticker: str = Field(description="Stock ticker symbol")
    analyst_consensus: str = Field(
        description="Wall Street consensus: 'Strong Buy', 'Buy', 'Hold', 'Sell', or 'Strong Sell'. "
        "If unknown, say 'Unknown'.",
    )
    price_target_vs_current: str = Field(
        description="Consensus price target vs. current price, e.g. '15% above' or '5% below'. "
        "If unknown, say 'Unknown'.",
    )
    institutional_discovery_stage: str = Field(
        description="Reasoning about where this stock is in the institutional discovery cycle. "
        "Is this early accumulation, recognized momentum, or fully priced consensus?",
    )
    is_genuinely_early_stage: bool = Field(
        description="True ONLY if this represents a genuinely under-discovered opportunity "
        "that most professional investors have NOT yet recognized. "
        "False if analysts already cover it with Buy/Strong Buy, institutions are loaded, "
        "or it is a well-known sector play.",
    )
    institutional_score: int = Field(
        description="How undiscovered is this from an institutional perspective? "
        "10 = truly off the radar, minimal analyst coverage, no recent upgrades. "
        "7-9 = limited coverage, early institutional accumulation. "
        "4-6 = moderate coverage, some analyst attention, partly priced in. "
        "1-3 = widely covered, consensus Buy, fully priced by institutions.",
        ge=1, le=10,
    )
    reasoning: str = Field(
        description="Detailed reasoning tying together analyst consensus, institutional "
        "positioning, valuation, and earnings trajectory. Reference specific data found.",
    )


class InstitutionalValidationResult(BaseModel):
    """Full response from institutional validation."""

    assessments: list[InstitutionalAssessment] = Field(
        description="One assessment per candidate, in the same order as input",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
You are a senior equity research analyst. Search the web and report the
following factual data for the stock below. Do NOT provide opinions — just
the data. Be thorough — search specifically for each data point.

Search for and report:
1. ANALYST COVERAGE: How many analysts cover this stock? What is the
   consensus rating (Strong Buy/Buy/Hold/Sell)? List any recent
   upgrades/downgrades with analyst firm names and dates.
2. PRICE TARGETS: What is the consensus price target? What is the range
   (low to high)? How does the consensus compare to the current price?
3. INSTITUTIONAL OWNERSHIP: What percentage is institutionally owned?
   How many institutional holders? Any notable recent 13F additions?
4. VALUATION: P/E ratio, forward P/E, EV/EBITDA, or other relevant
   multiples. How do these compare to sector averages?
5. EARNINGS: Most recent quarter results — beat or miss vs consensus?
   Revenue growth rate? Any guidance changes? Next earnings date?

Report facts with numbers. If you cannot find data for a category, say
"Not found" explicitly — do NOT guess.\
"""

JUDGMENT_PROMPT = """\
You are a senior equity analyst at a top-tier institutional investment firm.
You will receive factual research data about a stock that passed a momentum
screen. Your job is to determine whether it represents a genuinely
EARLY-STAGE, under-discovered opportunity — or an already-recognized
consensus play that professional investors have already priced in.

JUDGMENT FRAMEWORK:
- A stock with "Buy" consensus from 10+ analysts, heavy institutional
  ownership (>60%), and a price near analyst targets is NOT early-stage.
  Score it 1-3.
- A stock with limited analyst coverage (<5 analysts), low institutional
  ownership, and improving fundamentals not yet in consensus IS potentially
  early-stage. Score it 7-10.
- Sector/macro tailwinds (oil prices, rate cuts, etc.) do NOT make a stock
  early-stage. Every peer benefits equally from macro tailwinds.
- If analyst consensus data is "Not found" or "Unknown", treat with caution:
  for mid/large-cap stocks (>$1B), lack of data likely means the research
  failed to find it, not that coverage doesn't exist. Score conservatively.

Be rigorous. The burden of proof is on "early-stage." Default to skepticism.\
"""


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class InstitutionalValidator:
    """Two-step Phase 2.75: per-stock web research (gpt-4o) + judgment (o3-mini)."""

    _MAX_RESEARCH_WORKERS = 3

    def __init__(self, llm_config: LLMConfig = DEFAULT_LLM_CONFIG) -> None:
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._research_model = llm_config.research_model
        self._research_temperature = llm_config.research_temperature
        self._judgment_model = llm_config.institutional_validation_model

    def validate(
        self,
        approved: list[CandidateAnalysis],
    ) -> InstitutionalValidationResult:
        """
        Per-stock two-step validation:
        1. gpt-4o + web search gathers institutional data (concurrent per stock)
        2. o3-mini makes structured early-stage judgment (batched)
        """
        if not approved:
            return InstitutionalValidationResult(assessments=[])

        logger.info(
            "Institutional Validation: assessing %d candidates concurrently (%d workers)...",
            len(approved), self._MAX_RESEARCH_WORKERS,
        )

        # Step 1: Research each stock concurrently for thorough coverage
        research_by_idx: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self._MAX_RESEARCH_WORKERS) as pool:
            futures = {
                pool.submit(self._step1_research_single, c): idx
                for idx, c in enumerate(approved)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    research = future.result()
                    research_by_idx[idx] = f"--- {approved[idx].ticker} ---\n{research}"
                except Exception:
                    logger.exception("IV research failed for %s", approved[idx].ticker)
                    research_by_idx[idx] = f"--- {approved[idx].ticker} ---\n(research failed)"

        all_research = [research_by_idx[i] for i in sorted(research_by_idx)]
        combined_research = "\n\n".join(all_research)

        # Step 2: Batch judgment with o3-mini using all gathered research
        result = self._step2_judge(approved, combined_research)

        for a in result.assessments:
            stage = "EARLY" if a.is_genuinely_early_stage else "LATE/CONSENSUS"
            logger.info(
                "  IV [%s] %s — score %d/10 — consensus: %s — target: %s",
                a.ticker, stage, a.institutional_score,
                a.analyst_consensus, a.price_target_vs_current,
            )

        return result

    def _step1_research_single(self, candidate: CandidateAnalysis) -> str:
        """Step 1: Gather institutional data for ONE stock via web search."""
        prompt = (
            f"Search the web for comprehensive institutional data on {candidate.ticker}.\n"
            f"Company context: {candidate.catalyst_summary}\n\n"
            f"Search specifically for:\n"
            f'- "{candidate.ticker} analyst rating consensus"\n'
            f'- "{candidate.ticker} price target"\n'
            f'- "{candidate.ticker} institutional ownership"\n'
            f'- "{candidate.ticker} earnings"\n\n'
            f"Report all data you find with numbers and sources."
        )

        logger.info("  IV researching %s with %s...", candidate.ticker, self._research_model)

        response = self._client.responses.create(
            model=self._research_model,
            temperature=self._research_temperature,
            input=[
                {"role": "developer", "content": RESEARCH_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "web_search_preview"}],
        )

        n_searches = sum(1 for item in response.output if item.type == "web_search_call")
        research_text = response.output_text

        logger.info(
            "  IV %s: %d web searches, %d chars",
            candidate.ticker, n_searches, len(research_text),
        )
        return research_text

    def _step2_judge(
        self,
        approved: list[CandidateAnalysis],
        research: str,
    ) -> InstitutionalValidationResult:
        """Step 2: Structured judgment using research data (o3-mini)."""
        prompt = self._build_judgment_prompt(approved, research)
        tickers_str = ", ".join(c.ticker for c in approved)

        logger.info(
            "  IV Step 2 (Judgment) for [%s] with %s...",
            tickers_str, self._judgment_model,
        )

        response = self._client.responses.parse(
            model=self._judgment_model,
            input=[
                {"role": "developer", "content": JUDGMENT_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text_format=InstitutionalValidationResult,
        )

        return response.output_parsed

    @staticmethod
    def _build_judgment_prompt(
        approved: list[CandidateAnalysis],
        research: str,
    ) -> str:
        lines = [
            "Below are stocks that passed our momentum screen, along with",
            "per-stock web research about their institutional profile.",
            "For each stock, determine if it is genuinely early-stage or",
            "an already-priced consensus play.",
            "",
            "=" * 50,
            "OUR SYSTEM'S ASSESSMENT:",
            "=" * 50,
            "",
        ]
        for c in approved:
            lines.append(f"--- {c.ticker} ---")
            lines.append(f"  Catalyst: {c.catalyst_summary}")
            lines.append(f"  Early-stage reasoning: {c.early_stage_reasoning}")
            lines.append(f"  Conviction: {c.conviction_score}/100")
            lines.append(f"  Scores: timing={c.early_stage_timing_score} "
                         f"catalyst={c.catalyst_strength_score} "
                         f"freshness={c.narrative_freshness_score}")
            lines.append("")

        lines.extend([
            "=" * 50,
            "WEB RESEARCH (per-stock institutional data gathered moments ago):",
            "=" * 50,
            "",
            research if research else "(no research available)",
            "",
            "Using the research data above, assess each stock.",
        ])
        return "\n".join(lines)
