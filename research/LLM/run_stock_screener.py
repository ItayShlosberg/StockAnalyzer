"""
Early-stage momentum stock screener.
Uses LLM with web search + structured output, then applies hard programmatic
validation to reject any late-stage / crowded / overextended names that slip through.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from models import StockCandidate, StockScreenerResult
from prompt import STOCK_SCREENER_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MODEL = "gpt-4o"

MEGA_CAP_BLACKLIST = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK.A", "BRK.B", "UNH", "JNJ", "JPM", "V", "MA", "XOM", "PG",
    "HD", "AVGO", "COST", "LLY", "ABBV", "MRK", "PEP", "KO", "WMT",
    "CRM", "ORCL", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "CSCO",
    "PLTR", "SNOW", "PANW", "NOW", "UBER", "ABNB",
}

MAX_PCT_ABOVE_BREAKOUT = 40


def validate_candidate(stock: StockCandidate) -> list[str]:
    """Return list of violation reasons. Empty list = passed."""
    violations = []

    if stock.ticker.upper() in MEGA_CAP_BLACKLIST:
        violations.append(f"BLACKLISTED: {stock.ticker} is a known mega-cap / crowd favorite")

    if stock.pct_above_breakout > MAX_PCT_ABOVE_BREAKOUT:
        violations.append(f"OVEREXTENDED: {stock.pct_above_breakout}% above breakout (max {MAX_PCT_ABOVE_BREAKOUT}%)")

    if stock.self_check_already_obvious:
        violations.append("FAILED SELF-CHECK: LLM flagged as already obvious to most participants")

    if stock.self_check_ran_over_80pct:
        violations.append("FAILED SELF-CHECK: LLM flagged as having run > 80% in last 3 months")

    if stock.self_check_on_mainstream_lists:
        violations.append("FAILED SELF-CHECK: LLM flagged as appearing on mainstream momentum lists")

    if stock.self_check_palantir_test:
        violations.append("FAILED PALANTIR TEST: shares traits with late-stage discovered names")

    if stock.early_stage_timing_score < 6:
        violations.append(f"LOW TIMING SCORE: {stock.early_stage_timing_score}/10 (min 6)")

    if stock.narrative_freshness_score < 5:
        violations.append(f"STALE NARRATIVE: freshness {stock.narrative_freshness_score}/10 (min 5)")

    return violations


def run_screener() -> tuple[StockScreenerResult, list[dict]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    today = datetime.now().strftime("%B %d, %Y")
    user_message = (
        f"Today is {today}. Use web search extensively to find REAL current U.S. stock "
        f"market data. Search for RECENT breakouts, unusual volume surges, earnings "
        f"surprises, and fresh momentum setups from the LAST 1-5 TRADING DAYS.\n"
        f"You MUST search multiple queries to cover different sectors and setups.\n"
        f"Do NOT invent placeholder tickers — every ticker must be a real, currently-traded "
        f"U.S. stock with verifiable recent price action.\n"
        f"Do NOT return well-known mega-cap or large-cap momentum leaders (NVDA, PLTR, META, etc.).\n"
        f"Focus on $500M-$15B market cap stocks that are JUST NOW breaking out.\n\n"
        f"{STOCK_SCREENER_PROMPT}"
    )

    print(f"Calling {MODEL} with web search + structured output...")
    print("(This may take 60-120s as the model searches for real market data)\n")

    response = client.responses.parse(
        model=MODEL,
        temperature=0.2,
        input=[{"role": "user", "content": user_message}],
        tools=[{"type": "web_search_preview"}],
        text_format=StockScreenerResult,
    )

    search_calls = [item for item in response.output if item.type == "web_search_call"]
    print(f"Web searches performed: {len(search_calls)}")

    result: StockScreenerResult = response.output_parsed
    print(f"Raw candidates from LLM: {len(result.top_candidates)}")

    rejected = []
    passed = []
    for stock in result.top_candidates:
        violations = validate_candidate(stock)
        if violations:
            rejected.append({"ticker": stock.ticker, "company": stock.company_name, "violations": violations})
            print(f"  REJECTED {stock.ticker}: {'; '.join(violations)}")
        else:
            passed.append(stock)
            print(f"  PASSED   {stock.ticker} ({stock.company_name}) — conviction {stock.conviction_score}/100")

    result.top_candidates = passed
    print(f"\nFinal candidates after validation: {len(passed)}")

    return result, rejected


def save_results(result: StockScreenerResult, rejected: list[dict]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidates_df = pd.DataFrame([s.model_dump() for s in result.top_candidates])
    crowded_df = pd.DataFrame([s.model_dump() for s in result.too_late_crowded])
    watchlist_df = pd.DataFrame([s.model_dump() for s in result.watchlist_not_ready])
    rejected_df = pd.DataFrame(rejected) if rejected else pd.DataFrame()

    meta_df = pd.DataFrame([{
        "best_3_reasoning": result.best_3_reasoning,
        "red_flags_summary": result.red_flags_summary,
        "self_correction_notes": result.self_correction_notes,
        "conclusion": result.conclusion,
        "market_conditions_note": result.market_conditions_note,
        "model": MODEL,
        "timestamp": timestamp,
    }])

    xlsx_path = OUTPUT_DIR / f"screener_{timestamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not candidates_df.empty:
            candidates_df.to_excel(writer, sheet_name="Top Candidates", index=False)
        if not crowded_df.empty:
            crowded_df.to_excel(writer, sheet_name="Too Late - Crowded", index=False)
        if not watchlist_df.empty:
            watchlist_df.to_excel(writer, sheet_name="Watchlist", index=False)
        if not rejected_df.empty:
            rejected_df.to_excel(writer, sheet_name="Rejected by Validation", index=False)
        meta_df.to_excel(writer, sheet_name="Summary", index=False)

    csv_path = OUTPUT_DIR / f"screener_{timestamp}.csv"
    if not candidates_df.empty:
        candidates_df.to_csv(csv_path, index=False)
    else:
        meta_df.to_csv(csv_path, index=False)

    return csv_path, xlsx_path


def main():
    result, rejected = run_screener()

    if result.top_candidates:
        print("\n" + "=" * 70)
        print("TOP EARLY-STAGE CANDIDATES")
        print("=" * 70)
        for i, stock in enumerate(result.top_candidates, 1):
            print(f"\n{i}. {stock.ticker} - {stock.company_name}")
            print(f"   Conviction: {stock.conviction_score}/100 | Cap: {stock.market_cap} | {stock.sector} / {stock.industry}")
            print(f"   Breakout: {stock.breakout_date} at {stock.breakout_level} ({stock.pct_above_breakout}% above)")
            print(f"   Early-stage: {stock.early_stage_reasoning[:150]}")
            print(f"   Catalyst: {stock.catalyst_summary[:150]}")
            print(f"   Scores: timing={stock.early_stage_timing_score} tech={stock.technical_quality_score} "
                  f"vol={stock.volume_confirmation_score} fund={stock.fundamental_quality_score} "
                  f"catalyst={stock.catalyst_strength_score} freshness={stock.narrative_freshness_score}")
            print(f"   Risk: {stock.false_breakout_risk} | Risks: {stock.main_risks[:120]}")

        print(f"\n{'=' * 70}")
        print(f"BEST 3 REASONING\n{'=' * 70}")
        print(result.best_3_reasoning)
    else:
        print("\nNo candidates passed validation — the market may not offer genuine early-stage setups right now.")

    if result.too_late_crowded:
        print(f"\n{'=' * 70}")
        print(f"TOO LATE / CROWDED (correctly rejected)")
        print("=" * 70)
        for s in result.too_late_crowded:
            print(f"  {s.ticker} - {s.reason}")

    if result.watchlist_not_ready:
        print(f"\n{'=' * 70}")
        print(f"WATCHLIST (need confirmation)")
        print("=" * 70)
        for s in result.watchlist_not_ready:
            print(f"  {s.ticker} ({s.company_name}) - needs: {s.confirmation_needed}")

    if result.self_correction_notes:
        print(f"\n{'=' * 70}")
        print(f"SELF-CORRECTION NOTES")
        print("=" * 70)
        print(result.self_correction_notes)

    print(f"\n{'=' * 70}")
    print(f"CONCLUSION")
    print("=" * 70)
    print(result.conclusion)

    if rejected:
        print(f"\n{'=' * 70}")
        print(f"POST-LLM VALIDATION REJECTIONS ({len(rejected)} stocks)")
        print("=" * 70)
        for r in rejected:
            print(f"  {r['ticker']} ({r['company']}): {'; '.join(r['violations'])}")

    csv_path, xlsx_path = save_results(result, rejected)
    print(f"\nResults saved to:\n  CSV:  {csv_path}\n  XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
