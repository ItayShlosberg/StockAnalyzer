"""
MarketDataFetcher — batch-downloads OHLCV history.

Uses yf.download() for price/volume history in chunks.  Metadata (market cap,
sector, 52-week high) is provided by UniverseLoader from the screener response,
so NO per-ticker info calls are needed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from stock_analyzer.config import ScreenerConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketDataBundle:
    """Container for all data fetched in Phase 1."""

    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


class MarketDataFetcher:
    """Downloads OHLCV history for a list of tickers."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def fetch(self, tickers: list[str], metadata: pd.DataFrame) -> MarketDataBundle:
        """
        Download OHLCV for all tickers. Metadata is passed through from
        the UniverseLoader (already extracted from screener response).
        """
        period = f"{self._cfg.ohlcv_history_days}d"
        ohlcv = self._download_ohlcv(tickers, period)

        logger.info("MarketDataFetcher: %d tickers with OHLCV data", len(ohlcv))
        return MarketDataBundle(ohlcv=ohlcv, metadata=metadata)

    def _download_ohlcv(
        self, tickers: list[str], period: str,
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV in chunks to avoid request timeouts."""
        chunk_size = self._cfg.market_data_chunk_size
        result: dict[str, pd.DataFrame] = {}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            logger.info(
                "Downloading OHLCV chunk %d-%d / %d",
                i + 1, min(i + len(chunk), len(tickers)), len(tickers),
            )
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
            except Exception:
                logger.exception("OHLCV download failed for chunk starting at %d", i)
                continue

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in chunk:
                    if ticker in data.columns.get_level_values(0):
                        df = data[ticker].dropna(how="all")
                        if not df.empty and len(df) >= 50:
                            result[ticker] = df
            elif len(chunk) == 1:
                df = data.dropna(how="all")
                if not df.empty and len(df) >= 50:
                    result[chunk[0]] = df

        return result
