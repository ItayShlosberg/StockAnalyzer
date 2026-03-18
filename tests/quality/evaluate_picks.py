"""
Quality Evaluation Framework — automated false-positive detection.

Replicates the "copy to ChatGPT and ask" workflow as an automated, repeatable
script.  For each approved stock in a screener CSV, makes an independent LLM
call with web search to assess whether the pick is genuinely early-stage.

Usage:
    python -m tests.quality.evaluate_picks results/screener_YYYYMMDD_HHMMSS.csv

Output:
    - Per-stock comparison: system assessment vs. evaluator assessment
    - Aggregate false-positive rate and average institutional score
    - JSON saved to tests/quality/results/eval_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from stock_analyzer.config import DEFAULT_LLM_CONFIG, OPENAI_API_KEY

EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Pydantic model for the evaluator's structured response
# ---------------------------------------------------------------------------

class StockEvaluation(BaseModel):
    """Independent evaluator assessment of a single stock pick."""

    ticker: str = Field(description="Stock ticker symbol")
    is_attractive_entry: bool = Field(
        description="Does this currently represent an attractive EARLY-STAGE entry "
        "point for professional investors? True only if genuinely under-discovered.",
    )
    is_consensus_play: bool = Field(
        description="Is this an already-recognized consensus play that professional "
        "investors have already priced in?",
    )
    analyst_consensus: str = Field(
        description="Wall Street consensus if found (Strong Buy/Buy/Hold/Sell), or 'Unknown'",
    )
    institutional_score: int = Field(
        description="How undiscovered is this? 10=truly off radar, 1=fully consensus. "
        "Same scale as institutional validator.",
        ge=1, le=10,
    )
    reasoning: str = Field(
        description="Detailed reasoning with specific data from web search",
    )


class EvaluationBatch(BaseModel):
    """Batch of evaluations."""

    evaluations: list[StockEvaluation] = Field(
        description="One evaluation per stock, in the same order as input",
    )


# ---------------------------------------------------------------------------
# Evaluator prompt
# ---------------------------------------------------------------------------

EVALUATOR_PROMPT = """\
You are an independent equity analyst evaluating stock picks from an automated
momentum screener.  For each stock below, search the web and provide a
comprehensive analysis answering:

1. Does this stock currently represent an attractive EARLY-STAGE entry point
   for professional investors?
2. Or is it an already-recognized consensus play that the market has priced in?

For each stock, search for and assess:
- Analyst consensus ratings and number of covering analysts
- Price targets vs. current price
- Institutional ownership levels and recent changes
- Valuation multiples vs. sector peers
- Recent earnings trajectory
- Whether the catalyst is company-specific or sector/macro-driven

Be honest and rigorous. An early-stage pick that is actually a consensus play
is a false positive — identify it clearly.\
"""


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_csv(csv_path: str) -> dict:
    """Evaluate all approved picks in a screener CSV."""
    df = pd.read_csv(csv_path)

    if "ticker" not in df.columns:
        print("ERROR: CSV does not have a 'ticker' column.")
        sys.exit(1)

    tickers = df["ticker"].tolist()
    if not tickers:
        print("No picks to evaluate.")
        return {}

    print(f"\nEvaluating {len(tickers)} picks from {csv_path}")
    print("=" * 60)

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_scores = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        system_scores[t] = {
            "conviction": row.get("conviction_score", "N/A"),
            "early_stage_timing": row.get("early_stage_timing_score", "N/A"),
            "catalyst_strength": row.get("catalyst_strength_score", "N/A"),
            "narrative_freshness": row.get("narrative_freshness_score", "N/A"),
        }

    prompt_lines = [
        f"Evaluate each of these {len(tickers)} stocks picked by our momentum screener.",
        "Search the web for each one and determine if it's genuinely early-stage",
        "or an already-priced consensus play.",
        "",
    ]
    for t in tickers:
        prompt_lines.append(f"- {t}")
    prompt_lines.append("")
    prompt_lines.append("Now search the web for each stock and provide your assessment.")

    print("Step 1: Gathering institutional data via web search (gpt-4o)...")

    research_response = client.responses.create(
        model=DEFAULT_LLM_CONFIG.research_model,
        temperature=DEFAULT_LLM_CONFIG.research_temperature,
        input=[
            {"role": "developer", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": "\n".join(prompt_lines)},
        ],
        tools=[{"type": "web_search_preview"}],
    )
    research_text = research_response.output_text

    print(f"  Research: {len(research_text)} chars gathered")
    print(f"Step 2: Structured evaluation ({DEFAULT_LLM_CONFIG.institutional_validation_model})...")

    eval_prompt = (
        "Based on the web research below, evaluate each stock.\n\n"
        + research_text
        + "\n\nNow provide your structured assessment for each stock."
    )

    response = client.responses.parse(
        model=DEFAULT_LLM_CONFIG.institutional_validation_model,
        input=[
            {"role": "developer", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": eval_prompt},
        ],
        text_format=EvaluationBatch,
    )

    result: EvaluationBatch = response.output_parsed

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    false_positives = 0
    total = 0
    scores = []
    output_data = {"timestamp": datetime.now().isoformat(), "source_csv": csv_path, "evaluations": []}

    for ev in result.evaluations:
        total += 1
        scores.append(ev.institutional_score)
        sys_data = system_scores.get(ev.ticker, {})

        status = "EARLY-STAGE" if ev.is_attractive_entry else "CONSENSUS/LATE"
        if ev.is_consensus_play:
            false_positives += 1
            status = "FALSE POSITIVE"

        print(f"\n  {ev.ticker}: {status}")
        print(f"    System: conviction={sys_data.get('conviction', '?')} "
              f"timing={sys_data.get('early_stage_timing', '?')} "
              f"catalyst={sys_data.get('catalyst_strength', '?')}")
        print(f"    Evaluator: institutional_score={ev.institutional_score}/10 "
              f"consensus={ev.analyst_consensus}")
        print(f"    Reasoning: {ev.reasoning[:200]}")

        output_data["evaluations"].append({
            "ticker": ev.ticker,
            "system_scores": sys_data,
            "evaluator_verdict": status,
            "is_attractive_entry": ev.is_attractive_entry,
            "is_consensus_play": ev.is_consensus_play,
            "institutional_score": ev.institutional_score,
            "analyst_consensus": ev.analyst_consensus,
            "reasoning": ev.reasoning,
        })

    fp_rate = (false_positives / total * 100) if total > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE METRICS")
    print(f"{'=' * 60}")
    print(f"  Total picks evaluated: {total}")
    print(f"  False positives: {false_positives} ({fp_rate:.0f}%)")
    print(f"  Average institutional score: {avg_score:.1f}/10")
    print()

    output_data["metrics"] = {
        "total": total,
        "false_positives": false_positives,
        "false_positive_rate": fp_rate,
        "avg_institutional_score": avg_score,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to {out_path}")
    return output_data


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m tests.quality.evaluate_picks <path_to_screener_csv>")
        sys.exit(1)
    evaluate_csv(sys.argv[1])


if __name__ == "__main__":
    main()
