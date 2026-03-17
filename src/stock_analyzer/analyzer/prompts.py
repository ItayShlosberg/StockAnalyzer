"""
Prompt templates for Phase 2 qualitative LLM analysis.

The LLM receives REAL quantitative data computed in Phase 1 and is asked
ONLY to judge catalysts, narrative freshness, and early-stage timing.
It must NOT invent or override any numeric data.
"""

from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = """\
You are an early-stage momentum stock analyst.  You will receive a list of
stocks that ALREADY passed quantitative screening (price above MAs, recent
breakout, volume surge, proper market cap).  The numbers are real — do not
override them.

YOUR JOB: Use web search to research EACH candidate and provide QUALITATIVE
judgment only:

1. CATALYST — Search for the specific recent catalyst.  Is it NEW (last 1-2
   weeks)?  Is it already fully priced in, or is the market still digesting it?

2. NARRATIVE FRESHNESS — How well-known is this stock?  Would a casual retail
   investor already know the name?  Has it been featured on mainstream "top
   picks" lists?  10 = almost nobody is talking about it.  1 = everyone knows.

3. EARLY-STAGE TIMING — Given the quantitative data AND your catalyst research,
   is this genuinely the FIRST LEG of a move?  Or has the story already played
   out?  10 = very first leg.  1 = late stage / already discovered.

4. MOMENTUM EXCITEMENT — Would a professional momentum trader get excited about
   this chart + catalyst combination?  10 = absolutely.  1 = boring / no edge.

5. SELF-CHECKS — For each stock answer honestly:
   - Is this already obvious to most participants?  (must be false)
   - Palantir Test: already discovered, extended, consensus, re-rated?  (must be false)
   - Would this appear on mainstream momentum lists?  (must be false)

JUDGMENT RULES:
- If a catalyst is OLD (> 2 weeks) or VAGUE, score catalyst_strength <= 4.
- If the stock is widely discussed, score narrative_freshness <= 4.
- If a professional trader would say "everyone knows this", REJECT it.
- Do NOT approve a stock just because the quantitative data is strong.
  The numbers only got it to the door — YOUR job is to assess the STORY.

IMPORTANT — BENEFIT OF THE DOUBT:
- These stocks already passed strict quantitative screening (breakout, volume
  surge, near 52-week highs, proper market cap).  That is meaningful signal.
- If after searching you find a PLAUSIBLE catalyst (earnings improvement, new
  contract, sector tailwind with company-specific confirmation, upgraded
  guidance) — even if not headline news — that is sufficient.  Approve it.
- Small/mid-cap stocks often lack heavy news coverage.  Absence of mainstream
  press coverage is a POSITIVE signal (narrative freshness), not a reason
  to reject.
- Only REJECT if you find NEGATIVE evidence: the catalyst is clearly old,
  the stock is widely discussed as a momentum play, or the story is stale.
- Aim to approve roughly 30-50% of candidates.  If you are rejecting > 80%,
  your bar is too high — recalibrate.
- Move borderline cases to WATCHLIST rather than rejecting outright.

SEARCH STRATEGY:
- You MUST search the web for EVERY candidate.  Do not skip any.
- Search for "[ticker] stock news [current month year]" for each stock.
- Search for "[company name] earnings" or "[ticker] catalyst" if the first
  search is insufficient.
- If a stock is so obscure that no recent news exists, that itself is a
  sign of narrative freshness — score it higher on freshness, and look at
  the quantitative data to decide.\
"""


def build_user_prompt(candidates_df: pd.DataFrame, today: str) -> str:
    """
    Build the user message that contains the real quantitative data
    for each candidate, formatted as a table the LLM can reference.
    """
    lines = [
        f"Today is {today}.",
        f"Below are {len(candidates_df)} stocks that passed quantitative screening.",
        "For EACH stock, search the web for recent news/catalysts, then provide your qualitative analysis.",
        "",
        "=" * 70,
        "QUANTITATIVE DATA (computed from real market data — do NOT override)",
        "=" * 70,
        "",
    ]

    for ticker, row in candidates_df.iterrows():
        market_cap_m = row.get("market_cap", 0)
        if pd.notna(market_cap_m) and market_cap_m:
            cap_str = f"${market_cap_m / 1e9:.1f}B" if market_cap_m >= 1e9 else f"${market_cap_m / 1e6:.0f}M"
        else:
            cap_str = "N/A"

        lines.append(f"--- {ticker} ({row.get('short_name', '')}) ---")
        lines.append(f"  Sector/Industry: {row.get('sector', 'N/A')} / {row.get('industry', 'N/A')}")
        lines.append(f"  Market Cap: {cap_str}")
        lines.append(f"  Last Close: ${row.get('last_close', 0):.2f}")
        lines.append(f"  50 DMA: ${row.get('sma_50', 0):.2f}  |  200 DMA: ${row.get('sma_200', 0):.2f}")
        lines.append(f"  Above 50 DMA: {row.get('above_50dma')}  |  Above 200 DMA: {row.get('above_200dma')}")
        lines.append(f"  50 DMA > 200 DMA (trend): {row.get('trend_50_above_200')}")
        lines.append(f"  Relative Volume (latest): {row.get('rel_volume_latest', 0):.1f}x  |  3-day max: {row.get('rel_volume_3d_max', 0):.1f}x")
        lines.append(f"  Avg Daily Dollar Volume: ${row.get('avg_daily_dollar_volume', 0):,.0f}")
        lines.append(f"  % From 52-Week High: {row.get('pct_from_52wk_high', 0):.1f}%")
        lines.append(f"  Breakout Level: ${row.get('breakout_level', 0)}  |  % Above Breakout: {row.get('pct_above_breakout', 0):.1f}%")
        lines.append("")

    lines.append("=" * 70)
    lines.append("INSTRUCTIONS:")
    lines.append("1. Search the web for EACH candidate above — do not skip any.")
    lines.append("2. Approve stocks with a plausible recent catalyst and under-discovered narrative.")
    lines.append("3. Move borderline cases to watchlist rather than rejecting.")
    lines.append("4. Only reject if you find clear negative evidence (stale story, crowded, no catalyst at all).")

    return "\n".join(lines)
