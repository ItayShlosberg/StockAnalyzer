"""
UniverseLoader — builds the initial stock universe via yfinance screener.

Runs three exhaustive screener queries with different sort criteria. Each
query paginates until the API returns no more results, so the full set of
matching US equities is captured. Multiple sorts provide redundancy in case
the API silently caps pagination depth.

1. Sort by percent change descending
2. Sort by average daily volume descending
3. Sort by market cap ascending

Results are deduplicated and merged.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

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
        Return (tickers, metadata_df) by exhaustively scanning all matching
        US equities. Three sort orders run concurrently for speed.
        """
        query = EquityQuery("and", [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", ["lastclosemarketcap.lasttwelvemonths", self._cfg.market_cap_min]),
            EquityQuery("lt", ["lastclosemarketcap.lasttwelvemonths", self._cfg.market_cap_max]),
            EquityQuery("gt", ["avgdailyvol3m", self._cfg.min_avg_volume_3m]),
        ])

        scans = [
            ("1/3", "percentchange", False),
            ("2/3", "avgdailyvol3m", False),
            ("3/3", "lastclosemarketcap.lasttwelvemonths", True),
        ]

        logger.info("Universe: launching %d parallel scans", len(scans))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(self._run_scan, query, label, sort_field, sort_asc)
                for label, sort_field, sort_asc in scans
            ]
            scan_results = [f.result() for f in futures]

        all_quotes: list[dict] = []
        counts: list[int] = []
        for quotes in scan_results:
            all_quotes.extend(quotes)
            counts.append(len(quotes))

        tickers, metadata = self._process_quotes(all_quotes)
        logger.info(
            "Universe: %d raw quotes (%s) -> %d unique tickers after dedup/blacklist",
            len(all_quotes), " + ".join(str(c) for c in counts), len(tickers),
        )
        return tickers, metadata

    def _run_scan(
        self, query: EquityQuery, label: str, sort_field: str, sort_asc: bool,
    ) -> list[dict]:
        """Run a single exhaustive scan (called from thread pool)."""
        logger.info("Universe scan %s: sort by %s asc=%s (exhaustive)", label, sort_field, sort_asc)
        return self._fetch_all_pages(query, sort_field, sort_asc=sort_asc)

    def _fetch_all_pages(
        self, query: EquityQuery, sort_field: str,
        *, sort_asc: bool = False,
    ) -> list[dict]:
        """Paginate through ALL screener results until the API is exhausted."""
        all_quotes: list[dict] = []
        offset = 0
        page_size = self._cfg.universe_page_size

        while True:
            logger.info("  Fetching page offset=%d size=%d sort=%s asc=%s", offset, page_size, sort_field, sort_asc)
            try:
                response = yf.screen(
                    query,
                    sortField=sort_field,
                    sortAsc=sort_asc,
                    size=page_size,
                    offset=offset,
                )
            except Exception:
                logger.exception("  yfinance screen() failed at offset %d", offset)
                break

            quotes = response.get("quotes", [])
            if not quotes:
                break

            all_quotes.extend(quotes)

            if len(quotes) < page_size:
                break
            offset += page_size

        logger.info("  %s: fetched %d quotes", sort_field, len(all_quotes))
        return all_quotes

    @staticmethod
    def _process_quotes(quotes: list[dict]) -> tuple[list[str], pd.DataFrame]:
        """Extract tickers and metadata, deduplicating across queries."""
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
