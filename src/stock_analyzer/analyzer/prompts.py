"""
Prompt templates for the two-step LLM analysis.

Step 1 (Research): Unstructured — model searches the web freely.
Step 2 (Analysis): Structured — model fills Pydantic schema using research.
"""

from __future__ import annotations

import pandas as pd

from stock_analyzer.screener.news import NewsItem

# ---------------------------------------------------------------------------
# Step 1: Research (unstructured, web search enabled)
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are a stock market research assistant.  Your ONLY job is to search the
web for recent news and catalysts about the stocks you are given.

For EACH stock:
1. Search for "[ticker] stock news March 2026" (use the current month/year)
2. Search for "[company name] earnings" or "[ticker] catalyst" if needed
3. Report what you found: specific events, dates, analyst actions, earnings
   results, contract wins, FDA decisions, insider activity, etc.

RULES:
- You MUST search the web for EVERY stock.  Do NOT rely on your training data.
- Report FACTS from search results, not opinions.
- If you find nothing recent, say so explicitly.
- Include dates for every piece of news you report.
- Keep each stock's research to 3-5 bullet points.
- Do NOT provide investment advice or analysis — just report what you found.\
"""


def build_research_prompt(
    candidates_df: pd.DataFrame,
    news_data: dict[str, list[NewsItem]],
    today: str,
) -> str:
    """Build the Step 1 research prompt with quant context and yfinance headlines."""
    lines = [
        f"Today is {today}.",
        f"Search the web for recent news about each of these {len(candidates_df)} stocks.",
        "I have some headlines from Yahoo Finance below — use them as a starting point,",
        "but you MUST search for more recent and detailed information.",
        "",
    ]

    for ticker, row in candidates_df.iterrows():
        cap = row.get("market_cap", 0)
        cap_str = f"${cap / 1e9:.1f}B" if cap and cap >= 1e9 else f"${cap / 1e6:.0f}M" if cap else "N/A"

        lines.append(f"--- {ticker} ({row.get('short_name', '')}) ---")
        lines.append(f"  Sector: {row.get('sector', 'N/A')} | Cap: {cap_str}")
        lines.append(f"  Price: ${row.get('last_close', 0):.2f} | Near 52wk high: {row.get('pct_from_52wk_high', 0):.1f}%")

        headlines = news_data.get(ticker, [])
        if headlines:
            lines.append("  Yahoo Finance headlines:")
            for h in headlines[:5]:
                lines.append(f"    - [{h.published}] {h.title} ({h.publisher})")
        else:
            lines.append("  No Yahoo Finance headlines found (stock may be under-covered)")
        lines.append("")

    lines.append("Now search the web for EACH stock above and report your findings.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2: Analysis (structured output, no web search)
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are an early-stage momentum stock analyst.  You will receive:
1. Real quantitative data from Phase 1 screening
2. Real news headlines from Yahoo Finance
3. Real web research conducted moments ago (with actual search results)

YOUR JOB: Using ALL of the above evidence, provide qualitative judgment:

1. CATALYST — Based on the research provided, what is the specific recent
   catalyst?  Is it NEW (last 1-2 weeks)?  Is it already priced in?

2. NARRATIVE FRESHNESS — Based on news coverage density and the research,
   how well-known is this stock?  10 = almost nobody talking.  1 = everyone
   knows.  IMPORTANT: low news coverage is a POSITIVE signal, not negative.

3. EARLY-STAGE TIMING — Given quant data + research, is this the FIRST LEG
   of a move?  10 = very first leg.  1 = late stage / discovered.

4. MOMENTUM EXCITEMENT — Would a professional momentum trader get excited?
   10 = absolutely.  1 = boring / no edge.

JUDGMENT RULES:
- Only APPROVE if you find POSITIVE evidence that this is genuinely EARLY-STAGE
  and under-discovered. The burden of proof is on approval, not rejection.
- A stock at or near 52-week highs with recent analyst upgrades is NOT early-stage
  — it is a recognized, consensus play. Score early_stage_timing <= 4.
- If in doubt, REJECT or move to WATCHLIST. False positives are worse than
  false negatives for this system.

MACRO vs. ALPHA:
- A sector-wide catalyst (oil prices, rate expectations, geopolitical events)
  is NOT a company-specific catalyst. If the catalyst benefits ANY stock in
  the sector equally, score catalyst_strength <= 4 and note "macro/sector
  catalyst, not company-specific."
- Only score catalyst_strength >= 7 for company-specific events: earnings
  surprises, contract wins, FDA decisions, management changes, M&A, etc.

CATALYST SCORING:
- If the research found a SPECIFIC, COMPANY-SPECIFIC recent catalyst with a
  date, that is strong evidence. Score catalyst_strength >= 7.
- If the research found NO specific catalyst but the quant data shows a
  breakout on high volume, score catalyst_strength 5-6 and note it as
  "unverified quant signal."
- If the catalyst is purely sector/macro (benefits all peers equally), score
  catalyst_strength <= 4 regardless of recency.
- If the research shows the stock is widely discussed as a momentum favorite,
  score narrative_freshness <= 4 and consider rejecting.
- Low news coverage + strong quant breakout = high narrative freshness.
- Do NOT invent or fabricate any information.  If the research didn't find
  something, say so.  Reference the research directly.\
"""


def build_analysis_prompt(
    candidates_df: pd.DataFrame,
    news_data: dict[str, list[NewsItem]],
    research: str,
    today: str,
    market_context: str = "",
) -> str:
    """Build the Step 2 analysis prompt with quant data, news, and research."""
    lines = [
        f"Today is {today}.",
        f"Below are {len(candidates_df)} stocks that passed quantitative screening.",
        "Analyze each one using the quantitative data, news headlines, AND the",
        "web research provided below.",
    ]

    if market_context:
        lines.extend(["", "MARKET CONTEXT:", market_context, ""])

    lines.extend([
        "",
        "=" * 70,
        "QUANTITATIVE DATA (real — do NOT override)",
        "=" * 70,
        "",
    ])

    for ticker, row in candidates_df.iterrows():
        cap = row.get("market_cap", 0)
        if pd.notna(cap) and cap:
            cap_str = f"${cap / 1e9:.1f}B" if cap >= 1e9 else f"${cap / 1e6:.0f}M"
        else:
            cap_str = "N/A"

        lines.append(f"--- {ticker} ({row.get('short_name', '')}) ---")
        lines.append(f"  Sector/Industry: {row.get('sector', 'N/A')} / {row.get('industry', 'N/A')}")
        lines.append(f"  Market Cap: {cap_str}")
        lines.append(f"  Last Close: ${row.get('last_close', 0):.2f}")
        lines.append(f"  50 DMA: ${row.get('sma_50', 0):.2f}  |  200 DMA: ${row.get('sma_200', 0):.2f}")
        lines.append(f"  Above 50 DMA: {row.get('above_50dma')}  |  Above 200 DMA: {row.get('above_200dma')}")
        lines.append(f"  Trend (50>200): {row.get('trend_50_above_200')}")
        lines.append(f"  Rel Volume (latest): {row.get('rel_volume_latest', 0):.1f}x  |  3-day max: {row.get('rel_volume_3d_max', 0):.1f}x")
        lines.append(f"  Avg Daily Dollar Volume: ${row.get('avg_daily_dollar_volume', 0):,.0f}")
        lines.append(f"  % From 52-Week High: {row.get('pct_from_52wk_high', 0):.1f}%")
        lines.append(f"  Breakout Level: ${row.get('breakout_level', 0)}  |  % Above Breakout: {row.get('pct_above_breakout', 0):.1f}%")

        if "breakout_timeframe" in row:
            lines.append(f"  Breakout Timeframe: {row.get('breakout_timeframe', 'N/A')}-day high")
        if "breakout_volume_ratio" in row:
            lines.append(f"  Breakout Volume Ratio: {row.get('breakout_volume_ratio', 0):.1f}x avg")
        if "atr_contraction_ratio" in row:
            lines.append(f"  ATR Contraction (base tightness): {row.get('atr_contraction_ratio', 0):.2f}")
        if "sector_relative_strength" in row:
            lines.append(f"  Sector-Relative Strength (20d): {row.get('sector_relative_strength', 0):.1f}%")

        headlines = news_data.get(ticker, [])
        if headlines:
            lines.append(f"  Yahoo Finance headlines ({len(headlines)}):")
            for h in headlines[:5]:
                lines.append(f"    [{h.published}] {h.title}")
        else:
            lines.append("  Yahoo Finance: no recent headlines (under-covered — positive freshness signal)")
        lines.append("")

    lines.extend([
        "=" * 70,
        "WEB RESEARCH (conducted moments ago with real search results)",
        "=" * 70,
        "",
        research,
        "",
        "=" * 70,
        "INSTRUCTIONS:",
        "1. Use the web research above as your PRIMARY source of catalyst information.",
        "2. Cross-reference with Yahoo Finance headlines where available.",
        "3. Approve stocks with a plausible catalyst (from research) and under-discovered narrative.",
        "4. Move borderline cases to watchlist rather than rejecting.",
        "5. Only reject if the research reveals clear negative evidence.",
        "6. Do NOT fabricate information — if the research didn't find it, say so.",
    ])

    return "\n".join(lines)
