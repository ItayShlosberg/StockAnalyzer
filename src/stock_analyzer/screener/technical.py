"""
TechnicalCalculator — computes technical indicators from OHLCV data.

All calculations are pure pandas — no external TA library required.
Each ticker's OHLCV DataFrame is enriched with MAs, relative volume,
multi-timeframe breakout detection, base quality metrics, and trend
structure, then collapsed into a single summary row.
"""

from __future__ import annotations

import logging

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
        Return a DataFrame with one row per ticker containing all technical
        indicators, breakout quality metrics, and metadata.
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
        df = self._add_sector_relative_strength(df)
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
        high = df["High"].squeeze() if "High" in df.columns else close
        low = df["Low"].squeeze() if "Low" in df.columns else close

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

        window = self._cfg.recent_breakout_window
        breakout = self._detect_breakout_multi(close, volume, avg_vol_20, window)

        atr_contraction = self._compute_atr_contraction(high, low, close)
        base_depth = self._compute_base_depth(close)
        return_20d = ((last_close / float(close.iloc[-21])) - 1) * 100 if len(close) >= 21 else 0.0

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
            "breakout_detected": breakout["detected"],
            "breakout_level": round(breakout["level"], 2) if breakout["level"] else None,
            "pct_above_breakout": round(breakout["pct_above"], 2) if breakout["pct_above"] is not None else None,
            "breakout_timeframe": breakout["timeframe"],
            "breakout_volume_ratio": round(breakout["volume_ratio"], 2) if breakout["volume_ratio"] else None,
            "atr_contraction_ratio": round(atr_contraction, 2) if atr_contraction is not None else None,
            "base_depth_pct": round(base_depth, 2) if base_depth is not None else None,
            "return_20d": round(return_20d, 2),
            "trend_50_above_200": sma_50 > sma_200 if pd.notna(sma_50) and pd.notna(sma_200) else False,
            "market_cap": market_cap,
            "sector": meta_row.get("sector", "") if isinstance(meta_row, pd.Series) else "",
            "industry": meta_row.get("industry", "") if isinstance(meta_row, pd.Series) else "",
            "short_name": meta_row.get("short_name", "") if isinstance(meta_row, pd.Series) else "",
        }

    @staticmethod
    def _detect_breakout_multi(
        close: pd.Series,
        volume: pd.Series,
        avg_vol_20: float,
        window: int,
    ) -> dict:
        """
        Multi-timeframe breakout detection. Checks 120-day, 60-day, and 20-day
        highs (preferring the highest timeframe). Within each timeframe, scans
        all days in the window and selects the breakout day with the strongest
        volume ratio — avoiding the trap of picking an early weak-volume day
        when a later high-conviction day exists.
        """
        result = {
            "detected": False, "level": None, "pct_above": None,
            "timeframe": None, "volume_ratio": None,
        }

        last_close = float(close.iloc[-1])

        for lookback in (120, 60, 20):
            if len(close) < lookback + window:
                continue

            recent = close.iloc[-(lookback + window):]
            recent_vol = volume.iloc[-(lookback + window):] if len(volume) >= lookback + window else pd.Series(dtype=float)

            best_in_window: dict | None = None

            for day_idx in range(window):
                pos = len(recent) - window + day_idx
                prior_window = recent.iloc[max(0, pos - lookback): pos]
                if prior_window.empty:
                    continue
                resistance = float(prior_window.max())
                day_close = float(recent.iloc[pos])

                if day_close > resistance:
                    pct = ((last_close - resistance) / resistance) * 100

                    vol_ratio = None
                    if not recent_vol.empty and avg_vol_20 and avg_vol_20 > 0:
                        try:
                            day_vol = float(recent_vol.iloc[pos])
                            vol_ratio = day_vol / avg_vol_20
                        except (IndexError, ValueError):
                            pass

                    if best_in_window is None or (vol_ratio or 0) > (best_in_window["volume_ratio"] or 0):
                        best_in_window = {
                            "detected": True,
                            "level": resistance,
                            "pct_above": pct,
                            "timeframe": lookback,
                            "volume_ratio": vol_ratio,
                        }

            if best_in_window is not None:
                return best_in_window

        return result

    @staticmethod
    def _compute_atr_contraction(
        high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> float | None:
        """
        ATR contraction ratio: compare recent ATR(14) to prior ATR(14).
        A ratio < 1.0 means the stock was tightening/consolidating (bullish
        for breakout quality). Returns None if insufficient data.
        """
        if len(close) < 60:
            return None

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        recent_atr = float(tr.iloc[-20:].rolling(14).mean().iloc[-1])
        prior_atr = float(tr.iloc[-60:-20].rolling(14).mean().iloc[-1])

        if prior_atr > 0:
            return recent_atr / prior_atr
        return None

    @staticmethod
    def _compute_base_depth(close: pd.Series) -> float | None:
        """
        Max drawdown in the 40 days before the most recent 5-day window,
        relative to the peak. A shallow base (< 15%) with tight action
        indicates higher quality.
        """
        if len(close) < 45:
            return None

        base_period = close.iloc[-45:-5]
        peak = float(base_period.max())
        trough = float(base_period.min())

        if peak > 0:
            return ((peak - trough) / peak) * 100
        return None

    @staticmethod
    def _add_sector_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute each stock's 20-day return relative to its sector median.
        Positive values mean the stock outperformed its sector.
        """
        if "return_20d" not in df.columns or "sector" not in df.columns:
            df["sector_relative_strength"] = 0.0
            return df

        sector_medians = df.groupby("sector")["return_20d"].transform("median")
        df["sector_relative_strength"] = (df["return_20d"] - sector_medians).round(2)

        return df
