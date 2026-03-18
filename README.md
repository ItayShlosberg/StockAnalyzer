# StockAnalyzer — Early-Stage Momentum Stock Screener

A multi-phase pipeline that identifies **early-stage** momentum stocks before they become obvious.

Phase 1 uses **real market data** (yfinance) for quantitative screening; Phase 2 uses a **two-step LLM process** (OpenAI GPT-4o with web search) for qualitative catalyst/narrative analysis; Phase 2.75 uses a **reasoning model** (o3-mini) for institutional-grade validation; a **Devil's Advocate** LLM independently challenges every approved pick.

## Architecture

```
Phase 1: Quantitative       Phase 1.5: Intel     Phase 2: LLM Analysis       Phase 2.75              Phase 2.5          Phase 3: Output
┌─────────────────┐         ┌────────────┐       ┌────────────────────┐      ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│ UniverseLoader   │         │ NewsFetcher │       │ Step 1: Web Search │      │ Institutional    │    │ Devil's      │    │ Validator    │
│(triple-query scan│         │ (yf news)   │       │ (unstructured)     │      │ Validator        │    │ Advocate     │    │ (quant + DA) │
│        ↓         │         │      ↓      │       │          ↓         │      │ (o3-mini + web)  │    │ (web search) │    │      ↓       │
│ MarketDataFetcher│         │ Market      │  ──→  │ Step 2: Structured │ ──→  │ "Is this truly   │──→ │ independent  │──→ │ Forward      │
│        ↓         │   ──→   │ Regime      │       │ Analysis (Pydantic)│      │  early-stage?"   │    │ challenge    │    │ Tracker      │
│ TechnicalCalc    │         │ (SPY-based) │       └────────────────────┘      └──────────────────┘    └──────────────┘    │      ↓       │
│        ↓         │         └────────────┘                                                                                │ Report       │
│ QuantFilter      │                                                                                                       │ Exporter     │
│ (~20-40 stocks)  │                                                                                                       └──────────────┘
└─────────────────┘
```

### Phase 1 — Quantitative (pure Python, no LLM)
- Exhaustive triple-query universe scan: all matching US equities ($500M-$25B market cap)
- 50/200-day moving averages, trend structure
- Multi-timeframe breakout detection (20/60/120-day highs)
- Volume confirmation on breakout day
- ATR contraction (base tightness before breakout)
- Sector-relative strength (must outperform own sector)
- Hard filters: market cap $500M-$25B, within 15% of 52wk high, relative vol > 1.5x

### Phase 1.5 — Intelligence Gathering
- Yahoo Finance news headlines per candidate (asymmetric: used to disqualify, not qualify)
- SPY-based market regime detection (bull/correction/mixed)
- Market breadth (% of universe above 50 DMA)

### Phase 2 — Two-Step LLM Analysis
- **Step 1 (Research)**: Unstructured call with `web_search_preview`. No schema, so the model actually searches.
- **Step 2 (Analysis)**: Structured call with Pydantic models, using Step 1's research as input.
- Prompt calibration: burden of proof on approval (not rejection), macro vs. company-specific catalyst distinction, anchored scoring rubrics.

### Phase 2.75 — Institutional Validation (NEW)
- Uses **o3-mini** (reasoning model) with web search for each approved candidate
- Checks analyst consensus, price targets, institutional ownership, valuation, earnings trajectory
- Rejects stocks that are already consensus plays with limited upside
- Only genuinely under-discovered opportunities proceed

### Phase 2.5 — Devil's Advocate
- Independent LLM call with **web search** that challenges every approved pick
- Cross-checks catalyst claims against real headlines AND independent web research
- Receives the full research log from Phase 2 (not an empty string)
- Checks institutional consensus as part of its challenge

### Phase 3 — Validation & Output
- Programmatic cross-checks (scores, blacklist, overextension)
- Devil's advocate integration (unverified catalysts get rejected)
- Forward performance tracker (saves picks, checks returns on next run)
- Export to CSV, XLSX (multi-sheet), and detailed per-run log file

## Configuration

All settings are in **`config.yaml`** at the project root. Key sections:

```yaml
llm:
  research_model: "gpt-4o"          # Step 1: web research
  analysis_model: "gpt-4o"          # Step 2: structured analysis
  institutional_validation_model: "o3-mini"  # Phase 2.75: reasoning model
  devils_advocate_model: "gpt-4o"   # Phase 2.5: skeptical check

screener:
  max_pct_from_52wk_high: 15.0      # Quantitative thresholds
  min_relative_volume: 1.5
  # ... all other thresholds

validation:
  min_institutional_validation_score: 6  # 1-10, institutional bar
```

See `config.yaml` for the full reference. If the file is missing, defaults are used.

## Quick Start

```bash
# 1. Create venv with Python 3.11+
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
echo OPENAI_API_KEY=sk-your-key-here > .env

# 4. Run the screener
set PYTHONPATH=src        # Windows
# export PYTHONPATH=src   # macOS/Linux
python -m stock_analyzer
```

Results are saved to `results/` (CSV, XLSX, per-run `.log`).
Pick tracking data is saved to `results/tracking/`.

## Quality Evaluation

After a run, evaluate the quality of approved picks:

```bash
python -m tests.quality.evaluate_picks results/screener_YYYYMMDD_HHMMSS.csv
```

This makes independent LLM calls with web search for each pick, replicating a professional investor's judgment. It outputs a per-stock comparison and an aggregate false-positive rate. Results are saved to `tests/quality/results/`.

## Project Structure

```
src/stock_analyzer/
├── config.py                   # Loads from config.yaml, exposes per-step LLM settings
├── pipeline.py                 # Orchestrator (all phases including 2.75)
├── __main__.py                 # Entry point with dual logging
├── screener/                   # Phase 1: quantitative
│   ├── universe.py             # UniverseLoader (triple-query yfinance screener)
│   ├── market_data.py          # MarketDataFetcher (batch OHLCV download)
│   ├── technical.py            # TechnicalCalculator (MAs, breakout, ATR, sector RS)
│   ├── quantitative_filter.py  # Hard numeric filters
│   └── news.py                 # NewsFetcher (yfinance headlines, asymmetric)
├── analyzer/                   # Phase 2: qualitative
│   ├── models.py               # Pydantic models (structured LLM output)
│   ├── prompts.py              # Two-step prompt templates (research + analysis)
│   ├── llm_analyzer.py         # LLMAnalyzer (two-step: research → analyze)
│   ├── institutional_validator.py  # Phase 2.75: professional investor validation
│   ├── devils_advocate.py      # Independent LLM cross-check (with web search)
│   └── validator.py            # Post-LLM validation layer
├── export/                     # Phase 3: output
│   └── report.py               # CSV, XLSX, console formatting
└── tracker/                    # Performance tracking
    └── performance.py          # ForwardTracker (saves picks, checks returns)

tests/quality/
├── evaluate_picks.py           # Automated false-positive evaluation
└── results/                    # Evaluation output (JSON)

config.yaml                     # All settings (LLM models, thresholds, paths)
docs/
└── STRATEGY.md                 # Data strategy roadmap and improvement estimates
```

## Key Design Decisions

1. **Two-step LLM**: separating "explore" from "judge" forces the model to actually search
2. **Per-step model selection**: reasoning models (o3-mini) for judgment, cheaper models for data gathering
3. **Institutional validation**: catches consensus plays that quantitative + basic LLM analysis miss
4. **Inverted burden of proof**: approval requires positive evidence of early-stage, not just absence of negatives
5. **Macro vs. alpha distinction**: sector-wide catalysts are scored differently from company-specific events
6. **Devil's Advocate with web search**: independent verification, not just reasoning from provided data
7. **Asymmetric news**: absence of coverage = positive (under-discovered), not negative
8. **Sector-relative strength**: filters out stocks just riding sector ETF moves
9. **Forward tracking**: the only way to know if system changes actually improve results
10. **Quality evaluation**: automated false-positive detection replicating professional investor judgment

For the full data strategy and roadmap, see [docs/STRATEGY.md](docs/STRATEGY.md).
