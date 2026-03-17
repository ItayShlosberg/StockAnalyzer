"""
TechnicalCalculator — computes technical indicators from OHLCV data.

All calculations are pure pandas — no external TA library required.
Each ticker's OHLCV DataFrame is enriched with MAs, relative volume,
breakout detection, and trend structure, then collapsed into a single
summary row for the QuantitativeFilter.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from stock_analyzer.config import ScreenerConfig
from stock_analyzer.screener.market_data import MarketDataBundle

logger = logging.getLogger(__name__)


class TechnicalCalculator:
    """Computes technical features and produces a per-ticker summary DataFrame."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def compute(self, bundle: MarketDataBundle) -> pd.DataFrame:
        """
        Return a DataFrame with one row per ticker and columns:

        last_close, sma_50, sma_200, above_50dma, above_200dma,
        rel_volume_latest, rel_volume_3d_max, avg_daily_dollar_volume,
        pct_from_52wk_high, breakout_detected, breakout_level,
        pct_above_breakout, trend_50_above_200, sector, industry,
        market_cap, short_name.
        """
        rows: list[dict] = []

        for ticker, ohlcv in bundle.ohlcv.items():
            try:
                row = self._compute_single(ticker, ohlcv, bundle.metadata)
                if row is not None:
                    rows.append(row)
            except Exception:
                logger.warning("Technical calc failed for %s", ticker, exc_info=True)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")
        logger.info("TechnicalCalculator: computed features for %d tickers", len(df))
        return df

    def _compute_single(
        self, ticker: str, ohlcv: pd.DataFrame, metadata: pd.DataFrame,
    ) -> dict | None:
        df = ohlcv.copy()

        if "Close" not in df.columns or len(df) < 50:
            return None

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(dtype=float)

        if close.empty:
            return None

        last_close = float(close.iloc[-1])

        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(min(200, len(close))).mean().iloc[-1]) if len(close) >= 50 else np.nan

        avg_vol_20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else np.nan
        rel_vol_latest = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else 0.0
        rel_vol_3d = [
            float(volume.iloc[-i] / avg_vol_20)
            for i in range(1, min(4, len(volume)))
            if avg_vol_20 and avg_vol_20 > 0
        ]
        rel_vol_3d_max = max(rel_vol_3d) if rel_vol_3d else 0.0

        avg_dollar_vol = float((close.tail(20) * volume.tail(20)).mean()) if len(volume) >= 20 else 0.0

        high_52wk = float(close.tail(252).max()) if len(close) >= 50 else last_close
        if ticker in metadata.index and pd.notna(metadata.loc[ticker].get("fifty_two_week_high")):
            high_52wk = max(high_52wk, float(metadata.loc[ticker]["fifty_two_week_high"]))
        pct_from_52wk_high = ((last_close - high_52wk) / high_52wk) * 100 if high_52wk > 0 else 0.0

        lookback = self._cfg.breakout_lookback_days
        window = self._cfg.recent_breakout_window
        breakout_detected, breakout_level, pct_above_breakout = self._detect_breakout(
            close, lookback, window,
        )

        meta_row = metadata.loc[ticker] if ticker in metadata.index else {}
        market_cap = meta_row.get("market_cap") if isinstance(meta_row, pd.Series) else None

        return {
            "ticker": ticker,
            "last_close": last_close,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "above_50dma": last_close > sma_50 if pd.notna(sma_50) else False,
            "above_200dma": last_close > sma_200 if pd.notna(sma_200) else False,
            "rel_volume_latest": round(rel_vol_latest, 2),
            "rel_volume_3d_max": round(rel_vol_3d_max, 2),
            "avg_daily_dollar_volume": round(avg_dollar_vol, 0),
            "pct_from_52wk_high": round(pct_from_52wk_high, 2),
            "breakout_detected": breakout_detected,
            "breakout_level": round(breakout_level, 2) if breakout_level else None,
            "pct_above_breakout": round(pct_above_breakout, 2) if pct_above_breakout is not None else None,
            "trend_50_above_200": sma_50 > sma_200 if pd.notna(sma_50) and pd.notna(sma_200) else False,
            "market_cap": market_cap,
            "sector": meta_row.get("sector", "") if isinstance(meta_row, pd.Series) else "",
            "industry": meta_row.get("industry", "") if isinstance(meta_row, pd.Series) else "",
            "short_name": meta_row.get("short_name", "") if isinstance(meta_row, pd.Series) else "",
        }

    @staticmethod
    def _detect_breakout(
        close: pd.Series, lookback: int, window: int,
    ) -> tuple[bool, float | None, float | None]:
        """
        Detect whether the price broke above the highest close of the prior
        `lookback` days at any point in the last `window` sessions.

        Returns (breakout_detected, breakout_level, pct_above_breakout).
        """
        if len(close) < lookback + window:
            return False, None, None

        recent = close.iloc[-(lookback + window):]
        last_close = float(close.iloc[-1])

        for day_idx in range(window):
            pos = len(recent) - window + day_idx
            prior_window = recent.iloc[max(0, pos - lookback): pos]
            if prior_window.empty:
                continue
            resistance = float(prior_window.max())
            day_close = float(recent.iloc[pos])
            if day_close > resistance:
                pct = ((last_close - resistance) / resistance) * 100
                return True, resistance, pct

        return False, None, None
