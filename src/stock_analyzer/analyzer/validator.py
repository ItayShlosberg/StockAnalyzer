"""
ResultValidator — programmatic post-LLM validation layer.

Cross-checks the LLM's qualitative output against real quantitative data,
enforces score thresholds and blacklist rules, and integrates the Devil's
Advocate independent challenge results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from stock_analyzer.analyzer.devils_advocate import ChallengeItem, DevilsAdvocateResult
from stock_analyzer.analyzer.models import CandidateAnalysis, ScreenerLLMResult
from stock_analyzer.config import MEGA_CAP_BLACKLIST, ScreenerConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Output of the validation pass."""

    passed: list[CandidateAnalysis] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    llm_result: ScreenerLLMResult | None = None
    devils_advocate: DevilsAdvocateResult | None = None


class ResultValidator:
    """Validates LLM output against quant data and devil's advocate challenges."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def validate(
        self,
        llm_result: ScreenerLLMResult,
        quant_df: pd.DataFrame,
        da_result: DevilsAdvocateResult | None = None,
    ) -> ValidationResult:
        """
        Run every approved candidate through validation checks including
        the devil's advocate cross-reference.
        """
        vr = ValidationResult(llm_result=llm_result, devils_advocate=da_result)

        da_by_ticker: dict[str, ChallengeItem] = {}
        if da_result:
            for ch in da_result.challenges:
                da_by_ticker[ch.ticker.upper()] = ch

        for candidate in llm_result.analyzed_candidates:
            challenge = da_by_ticker.get(candidate.ticker.upper())
            violations = self._check(candidate, quant_df, challenge)
            if violations:
                vr.failed.append({
                    "ticker": candidate.ticker,
                    "conviction": candidate.conviction_score,
                    "violations": violations,
                })
                logger.info("REJECTED %s: %s", candidate.ticker, "; ".join(violations))
            else:
                vr.passed.append(candidate)
                logger.info("PASSED   %s (conviction %d)", candidate.ticker, candidate.conviction_score)

        logger.info(
            "Validation: %d passed, %d failed out of %d LLM-approved",
            len(vr.passed), len(vr.failed), len(llm_result.analyzed_candidates),
        )
        return vr

    def _check(
        self,
        c: CandidateAnalysis,
        quant_df: pd.DataFrame,
        challenge: ChallengeItem | None,
    ) -> list[str]:
        violations: list[str] = []

        if c.ticker.upper() in MEGA_CAP_BLACKLIST:
            violations.append(f"BLACKLISTED: {c.ticker} is a known crowd favorite")

        if c.early_stage_timing_score < self._cfg.min_early_stage_timing_score:
            violations.append(
                f"LOW TIMING: {c.early_stage_timing_score}/10 "
                f"(min {self._cfg.min_early_stage_timing_score})",
            )
        if c.narrative_freshness_score < self._cfg.min_narrative_freshness_score:
            violations.append(
                f"STALE NARRATIVE: {c.narrative_freshness_score}/10 "
                f"(min {self._cfg.min_narrative_freshness_score})",
            )

        if not c.catalyst_is_new:
            violations.append("CATALYST NOT NEW: LLM determined catalyst is old/priced-in")

        if c.ticker.upper() in quant_df.index:
            row = quant_df.loc[c.ticker.upper()]
            pct_above = row.get("pct_above_breakout")
            if pd.notna(pct_above) and pct_above > self._cfg.max_pct_above_breakout:
                violations.append(
                    f"OVEREXTENDED (quant): {pct_above:.1f}% above breakout "
                    f"(max {self._cfg.max_pct_above_breakout}%)",
                )

        if challenge:
            if not challenge.catalyst_verified:
                violations.append(
                    "DA: Catalyst NOT VERIFIED — claimed catalyst not found in news or research"
                )
            if challenge.reject_recommendation and c.conviction_score < 80:
                violations.append(
                    f"DA: Rejection recommended (confidence {challenge.confidence}/10): "
                    f"{challenge.strongest_bear_case[:80]}"
                )

        return violations
