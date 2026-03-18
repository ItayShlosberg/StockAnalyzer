"""
QuantitativeFilter — applies hard numeric thresholds to the enriched DataFrame.

Every filter is derived from ScreenerConfig so thresholds are tunable without
touching code.  The output is a tight candidate list (~20-50 stocks) ready for
Phase 2 qualitative analysis.
"""

from __future__ import annotations

import logging

import pandas as pd

from stock_analyzer.config import ScreenerConfig

logger = logging.getLogger(__name__)


class QuantitativeFilter:
    """Reduces the enriched technical DataFrame to stocks passing all hard filters."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all quantitative filters and return the surviving candidates.

        Each filter is applied as a named step so the log shows exactly
        how many stocks each rule eliminates.
        """
        if df.empty:
            logger.warning("QuantitativeFilter received empty DataFrame")
            return df

        initial = len(df)
        logger.info("QuantitativeFilter: starting with %d stocks", initial)

        steps: list[tuple[str, pd.Series]] = [
            (
                "Price > 50 DMA",
                df["above_50dma"] == True,  # noqa: E712
            ),
            (
                "Price > 200 DMA or approaching from below",
                (df["above_200dma"] == True)  # noqa: E712
                | ((df["last_close"] / df["sma_200"] - 1) * 100 >= -self._cfg.max_pct_below_200dma),
            ),
            (
                f"Within {self._cfg.max_pct_from_52wk_high}% of 52-week high",
                df["pct_from_52wk_high"] >= -self._cfg.max_pct_from_52wk_high,
            ),
            (
                f"Relative volume (3d max) >= {self._cfg.min_relative_volume}",
                df["rel_volume_3d_max"] >= self._cfg.min_relative_volume,
            ),
            (
                f"Avg daily dollar volume >= ${self._cfg.min_avg_daily_dollar_volume:,.0f}",
                df["avg_daily_dollar_volume"] >= self._cfg.min_avg_daily_dollar_volume,
            ),
            (
                "Breakout detected in recent window",
                df["breakout_detected"] == True,  # noqa: E712
            ),
            (
                f"Distance from breakout <= {self._cfg.max_pct_above_breakout}%",
                df["pct_above_breakout"].fillna(999) <= self._cfg.max_pct_above_breakout,
            ),
            (
                "Sector-relative strength > 0 (outperforming sector)",
                df["sector_relative_strength"] > 0,
            ),
        ]

        if "breakout_volume_ratio" in df.columns:
            steps.append((
                f"Breakout volume >= {self._cfg.min_breakout_volume_ratio}x avg",
                df["breakout_volume_ratio"].fillna(0) >= self._cfg.min_breakout_volume_ratio,
            ))

        for name, mask in steps:
            before = len(df)
            df = df.loc[mask]
            dropped = before - len(df)
            if dropped:
                logger.info("  [%s] removed %d -> %d remain", name, dropped, len(df))

        logger.info(
            "QuantitativeFilter: %d / %d passed all filters",
            len(df), initial,
        )
        return df.sort_values("pct_from_52wk_high", ascending=False)
