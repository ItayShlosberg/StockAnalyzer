"""
MarketDataFetcher — batch-downloads OHLCV history.

Uses yf.download() for price/volume history in chunks.  Metadata (market cap,
sector, 52-week high) is provided by UniverseLoader from the screener response,
so NO per-ticker info calls are needed here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _download_chunk(
        self, chunk: list[str], period: str,
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV for a single chunk of tickers."""
        result: dict[str, pd.DataFrame] = {}
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
            logger.exception("OHLCV download failed for chunk %s…", chunk[:3])
            return result

        if data.empty:
            return result

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

    def _download_ohlcv(
        self, tickers: list[str], period: str,
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV in parallel chunks to avoid request timeouts."""
        chunk_size = self._cfg.market_data_chunk_size
        chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        max_workers = self._cfg.max_download_workers

        logger.info(
            "Downloading OHLCV: %d tickers in %d chunks (%d workers)",
            len(tickers), len(chunks), max_workers,
        )

        result: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._download_chunk, chunk, period): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    chunk_result = future.result()
                    result.update(chunk_result)
                    logger.info(
                        "  OHLCV chunk %d/%d done: %d tickers",
                        idx + 1, len(chunks), len(chunk_result),
                    )
                except Exception:
                    logger.exception("OHLCV chunk %d failed", idx + 1)

        return result
