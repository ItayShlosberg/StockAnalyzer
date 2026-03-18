"""
ForwardTracker — tracks performance of past picks to measure system accuracy.

On each run:
1. Saves current picks with entry prices to a JSON tracking file
2. Loads all previous tracking files and fetches current prices
3. Computes simple forward returns (1d, 3d, 5d) for past picks
4. Logs a performance report and provides data for the XLSX export
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_analyzer.analyzer.models import CandidateAnalysis
from stock_analyzer.config import TRACKING_DIR

logger = logging.getLogger(__name__)


class ForwardTracker:
    """Tracks forward performance of screener picks across runs."""

    def __init__(self, tracking_dir: Path = TRACKING_DIR) -> None:
        self._dir = tracking_dir

    def save_picks(
        self,
        picks: list[CandidateAnalysis],
        quant_df: pd.DataFrame,
    ) -> Path | None:
        """Save current run's picks for future performance tracking."""
        if not picks:
            return None

        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"picks_{timestamp}.json"

        records = []
        for c in picks:
            entry_price = 0.0
            if c.ticker.upper() in quant_df.index:
                entry_price = float(quant_df.loc[c.ticker.upper()].get("last_close", 0))

            records.append({
                "ticker": c.ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "conviction": c.conviction_score,
                "catalyst_strength": c.catalyst_strength_score,
                "narrative_freshness": c.narrative_freshness_score,
                "timing_score": c.early_stage_timing_score,
            })

        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info("ForwardTracker: saved %d picks to %s", len(records), path.name)
        return path

    def evaluate_previous(self) -> pd.DataFrame:
        """
        Load all previous tracking files, fetch current prices, and compute
        forward returns. Returns a DataFrame with performance data.
        """
        if not self._dir.exists():
            return pd.DataFrame()

        all_records: list[dict] = []
        for path in sorted(self._dir.glob("picks_*.json")):
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
                all_records.extend(records)
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read tracking file: %s", path.name)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])

        tickers = df["ticker"].unique().tolist()
        if not tickers:
            return pd.DataFrame()

        current_prices = self._fetch_current_prices(tickers)

        df["current_price"] = df["ticker"].map(current_prices)
        df["return_pct"] = ((df["current_price"] / df["entry_price"]) - 1) * 100
        df["days_held"] = (datetime.now() - df["date"]).dt.days

        df = df.dropna(subset=["current_price", "return_pct"])

        if not df.empty:
            logger.info("=" * 50)
            logger.info("PREVIOUS PICKS PERFORMANCE")
            logger.info("=" * 50)
            for _, row in df.iterrows():
                sign = "+" if row["return_pct"] >= 0 else ""
                logger.info(
                    "  %s (picked %s, %dd ago): $%.2f -> $%.2f (%s%.1f%%) conviction=%d",
                    row["ticker"], row["date"].strftime("%Y-%m-%d"),
                    row["days_held"], row["entry_price"], row["current_price"],
                    sign, row["return_pct"], row["conviction"],
                )

            avg_ret = df["return_pct"].mean()
            win_rate = (df["return_pct"] > 0).mean() * 100
            logger.info(
                "  SUMMARY: %d picks, avg return %.1f%%, win rate %.0f%%",
                len(df), avg_ret, win_rate,
            )

        return df

    @staticmethod
    def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
        """Batch fetch current prices for tracking tickers."""
        prices: dict[str, float] = {}
        try:
            data = yf.download(tickers, period="2d", progress=False)
            if data.empty:
                return prices

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        close = data[ticker]["Close"].dropna()
                        if not close.empty:
                            prices[ticker] = float(close.iloc[-1])
            elif len(tickers) == 1:
                close = data["Close"].dropna()
                if not close.empty:
                    prices[tickers[0]] = float(close.iloc[-1])
        except Exception:
            logger.warning("Failed to fetch current prices for tracking", exc_info=True)

        return prices
