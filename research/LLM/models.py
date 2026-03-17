from typing import Literal, Optional

from pydantic import BaseModel, Field


class StockCandidate(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g. CRDO)")
    company_name: str = Field(description="Full company name")
    market_cap: str = Field(description="Approximate market capitalization (e.g. '$4.2B')")
    sector: str = Field(description="Market sector")
    industry: str = Field(description="Specific industry within the sector")

    breakout_date: str = Field(description="Approximate date the breakout occurred (e.g. 'March 14, 2025')")
    breakout_level: str = Field(description="The price level the stock broke out from (e.g. '$42 resistance')")
    pct_above_breakout: int = Field(description="How far the stock is above its breakout level as a percentage (e.g. 12 means 12%)", ge=0, le=100)

    early_stage_reasoning: str = Field(description="Why this stock is in the FIRST LEG of a move — not the second, third, or later. Why is the market still discovering this name?")
    technical_setup: str = Field(description="Summary of the technical breakout setup — base pattern, breakout quality, moving average structure")
    volume_summary: str = Field(description="Volume behavior: is the volume expansion RECENT (last 1-5 days)? Relative volume ratio?")
    fundamental_summary: str = Field(description="Revenue growth, EPS trajectory, margins, balance sheet quality")
    catalyst_summary: str = Field(description="The SPECIFIC recent catalyst — what happened, when, and why it is NOT yet fully priced in")
    main_risks: str = Field(description="Key risks and reasons the trade may fail")

    early_stage_timing_score: int = Field(description="Is this truly the beginning of the move? (1=late stage, 10=very first leg)", ge=1, le=10)
    technical_quality_score: int = Field(description="Technical quality: clean breakout, not overextended (1=ugly/extended, 10=textbook)", ge=1, le=10)
    volume_confirmation_score: int = Field(description="Volume confirmation: fresh recent expansion (1=stale/declining, 10=surging now)", ge=1, le=10)
    fundamental_quality_score: int = Field(description="Fundamental quality (1=junk, 10=excellent growth)", ge=1, le=10)
    catalyst_strength_score: int = Field(description="Catalyst strength: new, credible, not yet priced (1=old news, 10=fresh catalyst)", ge=1, le=10)
    liquidity_score: int = Field(description="Liquidity / tradability (1=illiquid, 10=very liquid)", ge=1, le=10)
    narrative_freshness_score: int = Field(description="How under-discovered is this story? (1=everyone knows, 10=almost nobody is talking about it)", ge=1, le=10)

    self_check_already_obvious: bool = Field(description="Self-check: Is this stock already obvious to most market participants? Must be false to pass.")
    self_check_ran_over_80pct: bool = Field(description="Self-check: Has this stock already run > 80% in the last 3 months? Must be false to pass.")
    self_check_on_mainstream_lists: bool = Field(description="Self-check: Would this stock appear on mainstream 'top momentum stocks' lists? Must be false to pass.")
    self_check_palantir_test: bool = Field(description="Self-check Palantir Test: Does this stock share traits with PLTR — already discovered, extended, consensus winner, multi-re-rated? Must be false to pass.")

    false_breakout_risk: Literal["Low", "Medium", "High"] = Field(description="Overall false breakout risk level")
    conviction_score: int = Field(description="Overall conviction score — accounts for all factors including timing and narrative freshness", ge=1, le=100)


class CrowdedStock(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    reason: str = Field(description="Why this stock is too late / too extended / too crowded — be specific about how far it has run and how well-known it is")


class WatchlistStock(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    reason: str = Field(description="Why this stock is interesting but not yet actionable")
    confirmation_needed: str = Field(description="Specific confirmation needed: another close above resistance, second day of volume, post-earnings hold, etc.")


class StockScreenerResult(BaseModel):
    top_candidates: list[StockCandidate] = Field(description="Up to 10 EARLY-STAGE stock candidates ranked best to worst. Every candidate must pass all self-checks. Fewer is better than padding with late-stage names.")
    best_3_reasoning: str = Field(description="Why the top 3 candidates are genuinely EARLY — focus on timing within the move, narrative freshness, and distance from breakout")
    too_late_crowded: list[CrowdedStock] = Field(description="Up to 5 stocks that look attractive but are too late / extended / crowded. Include well-known momentum names that were considered and rejected.")
    watchlist_not_ready: list[WatchlistStock] = Field(description="Up to 5 interesting early-stage names that need more confirmation before acting")
    red_flags_summary: str = Field(description="Recurring red-flag patterns among rejected names — overextension, narrative staleness, crowd ownership, etc.")
    self_correction_notes: str = Field(description="Notes from the self-correction loop: which candidates were initially considered but removed after the Palantir test or crowd test, and why")
    conclusion: str = Field(description="Final assessment: which names are genuinely early (not obvious yet but could become obvious soon) vs. noise")
    market_conditions_note: Optional[str] = Field(default=None, description="Note about current market conditions if they limit the number of genuine early-stage setups available")
