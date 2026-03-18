"""
NewsFetcher — retrieves recent news headlines for quant-filtered candidates.

Uses yfinance's Ticker.get_news() to fetch real headlines. News is used
ASYMMETRICALLY: presence of crowded/stale coverage can disqualify a stock,
but absence of news is treated as a POSITIVE signal (narrative freshness).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

NEWS_MAX_AGE_DAYS = 14
NEWS_COUNT_PER_TICKER = 10


@dataclass(frozen=True)
class NewsItem:
    """A single news headline with metadata."""

    title: str
    published: str
    publisher: str
    link: str


def fetch_news(tickers: list[str]) -> dict[str, list[NewsItem]]:
    """
    Fetch recent news for each ticker. Returns a mapping of ticker -> headlines.
    Tickers with no news or failed fetches get an empty list (which is fine —
    absence of news is not a penalty).
    """
    result: dict[str, list[NewsItem]] = {}
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)

    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).get_news(count=NEWS_COUNT_PER_TICKER)
            items = _parse_news(raw, cutoff)
            result[ticker] = items
            if items:
                logger.info("  %s: %d recent headlines", ticker, len(items))
            else:
                logger.debug("  %s: no recent news (positive freshness signal)", ticker)
        except Exception:
            logger.debug("  %s: news fetch failed (treating as no news)", ticker, exc_info=True)
            result[ticker] = []

    total = sum(len(v) for v in result.values())
    logger.info("NewsFetcher: %d headlines across %d tickers", total, len(tickers))
    return result


def _parse_news(raw: list[dict], cutoff: datetime) -> list[NewsItem]:
    """Extract structured NewsItems from yfinance's raw news response."""
    items: list[NewsItem] = []
    for entry in raw:
        content = entry.get("content", entry)
        title = content.get("title", "")
        if not title:
            continue

        pub_date = content.get("pubDate", "")
        publisher = ""
        provider = content.get("provider", {})
        if isinstance(provider, dict):
            publisher = provider.get("displayName", "")

        link = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else ""

        if pub_date:
            try:
                dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
                pub_date = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        items.append(NewsItem(title=title, published=pub_date, publisher=publisher, link=link))

    return items
