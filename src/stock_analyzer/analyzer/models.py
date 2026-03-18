"""
Pydantic models for LLM structured output.

The LLM receives pre-computed quantitative data and real web research, then
fills ONLY the qualitative fields.  Self-check booleans have been removed
in favor of an independent Devil's Advocate validation step.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-candidate qualitative analysis (filled by LLM in Step 2)
# ---------------------------------------------------------------------------

class CandidateAnalysis(BaseModel):
    """LLM's qualitative judgment for a single pre-screened stock."""

    ticker: str = Field(description="Stock ticker symbol — must match one of the provided candidates")

    catalyst_summary: str = Field(
        description="The SPECIFIC recent catalyst found in the web research — what happened, "
        "when, and why it is or is not yet fully priced in.  Reference the research directly.",
    )
    catalyst_is_new: bool = Field(
        description="Is the catalyst genuinely NEW (last 1-2 weeks) and not yet fully digested by the market?",
    )
    narrative_summary: str = Field(
        description="How well-known is this stock's story based on news coverage density? "
        "Low coverage = under-discovered = positive.",
    )
    early_stage_reasoning: str = Field(
        description="Why this stock is (or is not) in the FIRST LEG of a momentum move. "
        "Reference the quantitative data provided.",
    )
    main_risks: str = Field(description="Key risks and reasons the trade may fail")

    catalyst_strength_score: int = Field(
        description="Catalyst strength. Scoring anchors: "
        "9-10 = Company-specific event in last 7 days (FDA, earnings surprise, contract win). "
        "7-8 = Company-specific event in last 14 days. "
        "5-6 = Plausible but unverified company-specific signal. "
        "3-4 = Sector/macro catalyst only (benefits all peers equally). "
        "1-2 = No catalyst found.",
        ge=1, le=10,
    )
    narrative_freshness_score: int = Field(
        description="How under-discovered is the story? 10=almost nobody talking, "
        "1=everyone knows. Low news coverage = higher score.",
        ge=1, le=10,
    )
    early_stage_timing_score: int = Field(
        description="Is this truly the beginning of the move? Scoring anchors: "
        "9-10 = First breakout from a multi-month base, almost no analyst coverage. "
        "7-8 = Recent breakout with limited institutional recognition. "
        "5-6 = Breakout recognized by some analysts, moderate coverage. "
        "3-4 = At/near 52-week highs with analyst upgrades and consensus coverage. "
        "1-2 = Widely discussed momentum name, everyone already positioned.",
        ge=1, le=10,
    )
    momentum_excitement_score: int = Field(
        description="Would a professional momentum trader get excited about this setup? "
        "10=definitely, 1=boring",
        ge=1, le=10,
    )

    conviction_score: int = Field(
        description="Overall conviction (1-100) combining quantitative data quality "
        "with qualitative judgment from the research",
        ge=1, le=100,
    )


class RejectedCandidate(BaseModel):
    """A candidate the LLM reviewed but rejected during qualitative analysis."""

    ticker: str = Field(description="Stock ticker symbol")
    rejection_reason: str = Field(
        description="Why this stock was rejected — stale catalyst, already crowded, boring setup, etc.",
    )


class WatchlistCandidate(BaseModel):
    """A stock with potential but not yet actionable."""

    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Why this stock is interesting but not yet ready")
    confirmation_needed: str = Field(
        description="What must happen before acting: another close above resistance, "
        "earnings hold, volume follow-through, etc.",
    )


# ---------------------------------------------------------------------------
# Top-level LLM response
# ---------------------------------------------------------------------------

class ScreenerLLMResult(BaseModel):
    """Full structured response from the Phase 2 LLM analysis."""

    analyzed_candidates: list[CandidateAnalysis] = Field(
        description="Candidates that PASSED qualitative checks, ranked best to worst by conviction. "
        "Only include genuinely early-stage stocks. Fewer is better than padding.",
    )
    rejected_candidates: list[RejectedCandidate] = Field(
        description="Candidates that were reviewed but FAILED qualitative checks. "
        "Include every candidate not in analyzed_candidates.",
    )
    watchlist: list[WatchlistCandidate] = Field(
        description="Stocks with potential but needing more confirmation before acting (0-5 names)",
    )
    best_picks_reasoning: str = Field(
        description="Why the top picks are genuinely EARLY — reference the web research "
        "and catalyst freshness specifically",
    )
    red_flags_summary: str = Field(
        description="Patterns observed among rejected candidates",
    )
    market_conditions_note: Optional[str] = Field(
        default=None,
        description="Note about current market conditions if they limit early-stage opportunities",
    )
