"""
UniverseLoader — builds the initial stock universe via yfinance screener.

Uses Yahoo Finance's EquityQuery API to pull all US-listed equities within
the target market-cap and liquidity range.  Extracts metadata (market cap,
sector, 52-week high) directly from the screener response to avoid slow
per-ticker info calls.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from stock_analyzer.config import MEGA_CAP_BLACKLIST, ScreenerConfig

logger = logging.getLogger(__name__)


class UniverseLoader:
    """Loads the investable universe of US equities matching size/liquidity criteria."""

    def __init__(self, config: ScreenerConfig) -> None:
        self._cfg = config

    def load(self) -> tuple[list[str], pd.DataFrame]:
        """
        Return (tickers, metadata_df).

        metadata_df is indexed by ticker and contains market_cap, sector,
        industry, fifty_two_week_high, fifty_two_week_low, short_name —
        all extracted from the screener response (zero extra API calls).
        """
        query = EquityQuery("and", [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", ["lastclosemarketcap.lasttwelvemonths", self._cfg.market_cap_min]),
            EquityQuery("lt", ["lastclosemarketcap.lasttwelvemonths", self._cfg.market_cap_max]),
            EquityQuery("gt", ["avgdailyvol3m", self._cfg.min_avg_volume_3m]),
        ])

        all_quotes: list[dict] = []
        offset = 0
        page_size = self._cfg.universe_page_size

        while True:
            logger.info("Fetching universe page offset=%d size=%d", offset, page_size)
            try:
                response = yf.screen(
                    query,
                    sortField="percentchange",
                    sortAsc=False,
                    size=page_size,
                    offset=offset,
                )
            except Exception:
                logger.exception("yfinance screen() failed at offset %d", offset)
                break

            quotes = response.get("quotes", [])
            if not quotes:
                break

            all_quotes.extend(quotes)

            if len(quotes) < page_size:
                break
            if len(all_quotes) >= self._cfg.universe_max_tickers:
                break
            offset += page_size

        tickers, metadata = self._process_quotes(all_quotes)
        logger.info(
            "Universe: %d raw quotes -> %d tickers after dedup/blacklist",
            len(all_quotes), len(tickers),
        )
        return tickers, metadata

    @staticmethod
    def _process_quotes(quotes: list[dict]) -> tuple[list[str], pd.DataFrame]:
        """Extract tickers and metadata from screener quote objects."""
        seen: set[str] = set()
        rows: list[dict] = []

        for q in quotes:
            symbol = (q.get("symbol") or "").upper()
            if not symbol or symbol in seen or symbol in MEGA_CAP_BLACKLIST:
                continue
            seen.add(symbol)
            rows.append({
                "ticker": symbol,
                "market_cap": q.get("marketCap"),
                "sector": q.get("sector", ""),
                "industry": q.get("industry", ""),
                "fifty_two_week_high": q.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": q.get("fiftyTwoWeekLow"),
                "short_name": q.get("shortName", ""),
            })

        if not rows:
            return [], pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")
        tickers = list(df.index)
        return tickers, df
