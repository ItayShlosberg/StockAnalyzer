"""
ResultValidator — programmatic post-LLM validation layer.

Cross-checks the LLM's qualitative output against the real quantitative data
and enforces score thresholds, self-check booleans, and blacklist rules.
Even if the LLM approves a stock, the validator can reject it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from stock_analyzer.analyzer.models import CandidateAnalysis, ScreenerLLMResult
from stock_analyzer.config import MEGA_CAP_BLACKLIST, ScreenerConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Output of the validation pass."""

    passed: list[CandidateAnalysis] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    llm_result: ScreenerLLMResult | None = None


class ResultValidator:
    """Validates LLM output against quant data and enforces hard rules."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def validate(
        self,
        llm_result: ScreenerLLMResult,
        quant_df: pd.DataFrame,
    ) -> ValidationResult:
        """
        Run every approved candidate through validation checks.

        Args:
            llm_result: The parsed LLM response.
            quant_df: The Phase 1 quantitative DataFrame (indexed by ticker).

        Returns:
            ValidationResult with passed/failed lists.
        """
        vr = ValidationResult(llm_result=llm_result)

        for candidate in llm_result.analyzed_candidates:
            violations = self._check(candidate, quant_df)
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
        self, c: CandidateAnalysis, quant_df: pd.DataFrame,
    ) -> list[str]:
        violations: list[str] = []

        if c.ticker.upper() in MEGA_CAP_BLACKLIST:
            violations.append(f"BLACKLISTED: {c.ticker} is a known crowd favorite")

        if c.self_check_already_obvious:
            violations.append("SELF-CHECK: LLM flagged as already obvious")
        if c.self_check_palantir_test:
            violations.append("PALANTIR TEST: shares traits with late-stage names")
        if c.self_check_on_mainstream_lists:
            violations.append("MAINSTREAM: would appear on popular momentum lists")

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

        return violations
