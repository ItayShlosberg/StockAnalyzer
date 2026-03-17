"""
ReportExporter — generates CSV, XLSX, and console output from pipeline results.

Merges quantitative data (Phase 1) with qualitative analysis (Phase 2) into
a single unified view per candidate.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_analyzer.analyzer.models import ScreenerLLMResult
from stock_analyzer.analyzer.validator import ValidationResult
from stock_analyzer.config import LLM_MODEL, RESULTS_DIR

logger = logging.getLogger(__name__)


class ReportExporter:
    """Exports screener results to CSV, XLSX, and console."""

    def __init__(self, output_dir: Path = RESULTS_DIR) -> None:
        self._output_dir = output_dir

    def export(
        self,
        validation: ValidationResult,
        quant_df: pd.DataFrame,
    ) -> tuple[Path, Path]:
        """
        Write results to disk and print a console summary.

        Returns:
            (csv_path, xlsx_path)
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        candidates_df = self._build_candidates_df(validation, quant_df)
        llm_result = validation.llm_result

        rejected_df = pd.DataFrame(validation.failed) if validation.failed else pd.DataFrame()
        llm_rejected_df = pd.DataFrame(
            [r.model_dump() for r in llm_result.rejected_candidates]
        ) if llm_result and llm_result.rejected_candidates else pd.DataFrame()
        watchlist_df = pd.DataFrame(
            [w.model_dump() for w in llm_result.watchlist]
        ) if llm_result and llm_result.watchlist else pd.DataFrame()

        summary_df = pd.DataFrame([{
            "best_picks_reasoning": llm_result.best_picks_reasoning if llm_result else "",
            "red_flags_summary": llm_result.red_flags_summary if llm_result else "",
            "market_conditions_note": llm_result.market_conditions_note if llm_result else "",
            "model": LLM_MODEL,
            "timestamp": timestamp,
            "candidates_passed": len(validation.passed),
            "candidates_failed_validation": len(validation.failed),
            "quant_candidates_input": len(quant_df),
        }])

        xlsx_path = self._output_dir / f"screener_{timestamp}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not candidates_df.empty:
                candidates_df.to_excel(writer, sheet_name="Top Candidates", index=False)
            if not watchlist_df.empty:
                watchlist_df.to_excel(writer, sheet_name="Watchlist", index=False)
            if not llm_rejected_df.empty:
                llm_rejected_df.to_excel(writer, sheet_name="LLM Rejected", index=False)
            if not rejected_df.empty:
                rejected_df.to_excel(writer, sheet_name="Validation Rejected", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        csv_path = self._output_dir / f"screener_{timestamp}.csv"
        if not candidates_df.empty:
            candidates_df.to_csv(csv_path, index=False)
        else:
            summary_df.to_csv(csv_path, index=False)

        self._print_console(validation, quant_df)

        logger.info("Results saved to %s and %s", csv_path, xlsx_path)
        return csv_path, xlsx_path

    def _build_candidates_df(
        self, validation: ValidationResult, quant_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge quant data with qualitative analysis into one flat table."""
        if not validation.passed:
            return pd.DataFrame()

        rows = []
        for c in validation.passed:
            qual = c.model_dump()
            quant = {}
            if c.ticker.upper() in quant_df.index:
                quant = quant_df.loc[c.ticker.upper()].to_dict()
                for k, v in quant.items():
                    qual[f"quant_{k}"] = v
            rows.append(qual)

        return pd.DataFrame(rows)

    @staticmethod
    def _print_console(validation: ValidationResult, quant_df: pd.DataFrame) -> None:
        sep = "=" * 72

        if validation.passed:
            print(f"\n{sep}")
            print("  EARLY-STAGE MOMENTUM CANDIDATES (passed all checks)")
            print(sep)

            for i, c in enumerate(validation.passed, 1):
                quant = quant_df.loc[c.ticker.upper()] if c.ticker.upper() in quant_df.index else {}
                cap = quant.get("market_cap", 0)
                cap_str = f"${cap / 1e9:.1f}B" if cap and cap >= 1e9 else f"${cap / 1e6:.0f}M" if cap else "N/A"

                print(f"\n  {i}. {c.ticker} — Conviction {c.conviction_score}/100")
                print(f"     Cap: {cap_str}  |  Sector: {quant.get('sector', 'N/A')}  |  {quant.get('industry', 'N/A')}")
                print(f"     Close: ${quant.get('last_close', 0):.2f}  |  Breakout: ${quant.get('breakout_level', 0)}  |  {quant.get('pct_above_breakout', 0):.1f}% above")
                print(f"     Rel Vol (3d max): {quant.get('rel_volume_3d_max', 0):.1f}x  |  % from 52wk high: {quant.get('pct_from_52wk_high', 0):.1f}%")
                print(f"     Catalyst: {c.catalyst_summary[:140]}")
                print(f"     Early-stage: {c.early_stage_reasoning[:140]}")
                print(f"     Scores: timing={c.early_stage_timing_score} catalyst={c.catalyst_strength_score} "
                      f"freshness={c.narrative_freshness_score} excitement={c.momentum_excitement_score}")
                print(f"     Risks: {c.main_risks[:120]}")
        else:
            print(f"\n{sep}")
            print("  No candidates passed all validation checks.")
            print(sep)

        llm_result = validation.llm_result
        if llm_result and llm_result.best_picks_reasoning:
            print(f"\n{sep}")
            print("  BEST PICKS REASONING")
            print(f"{sep}")
            print(f"  {llm_result.best_picks_reasoning}")

        if llm_result and llm_result.watchlist:
            print(f"\n{sep}")
            print("  WATCHLIST (need confirmation)")
            print(f"{sep}")
            for w in llm_result.watchlist:
                print(f"  {w.ticker} — needs: {w.confirmation_needed}")

        if llm_result and llm_result.rejected_candidates:
            print(f"\n{sep}")
            print("  LLM-REJECTED CANDIDATES")
            print(f"{sep}")
            for r in llm_result.rejected_candidates:
                print(f"  {r.ticker} — {r.rejection_reason[:100]}")

        if validation.failed:
            print(f"\n{sep}")
            print(f"  POST-LLM VALIDATION FAILURES ({len(validation.failed)})")
            print(f"{sep}")
            for f in validation.failed:
                print(f"  {f['ticker']}: {'; '.join(f['violations'])}")

        if llm_result and llm_result.red_flags_summary:
            print(f"\n{sep}")
            print("  RED FLAGS SUMMARY")
            print(f"{sep}")
            print(f"  {llm_result.red_flags_summary}")

        print()
