# StockAnalyzer — Early-Stage Momentum Stock Screener

A two-phase pipeline that identifies **early-stage** momentum stocks before they become obvious.

Phase 1 uses **real market data** (yfinance) for quantitative screening; Phase 2 uses an **LLM with web search** (OpenAI) for qualitative catalyst/narrative analysis.

## Architecture

```
Phase 1: Quantitative Screening          Phase 2: Qualitative Analysis        Phase 3: Output
┌───────────────┐                        ┌──────────────────────┐             ┌──────────────┐
│ UniverseLoader │→ MarketDataFetcher →  │ LLMAnalyzer          │→ Validator →│ ReportExporter│
│ (yf screener)  │  TechnicalCalculator  │ (GPT-4o + web search)│             │ (CSV/XLSX/log)│
│                │→ QuantitativeFilter → │ (batched, 4 at a time│             │              │
└───────────────┘   (~20-50 candidates)  └──────────────────────┘             └──────────────┘
```

### Phase 1 — what code computes (no LLM)
- 50/200-day moving averages, trend structure
- Relative volume (current vs 20-day avg)
- Breakout detection (price crossed N-day high in last 5 sessions)
- Distance from 52-week high, distance from breakout
- Hard filters: market cap $500M–$25B, dollar volume > $10M, within 8% of 52wk high, relative vol > 1.5x

### Phase 2 — what the LLM judges (with web search)
- Is there a **specific, recent catalyst**? (not old news)
- Is the narrative **under-discovered**? (not consensus)
- Would a momentum trader call this **early**? (first leg, not third)
- Palantir test: is this already widely known and re-rated?

### Phase 3 — programmatic validation
- Cross-checks LLM scores against quant data
- Enforces minimum score thresholds and self-check booleans
- Blacklist enforcement
- Exports CSV, XLSX (multi-sheet), and detailed log file

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

Results are saved to `results/` (CSV, XLSX, and a `.log` file per run).

## Project Structure

```
src/stock_analyzer/
├── config.py              # All tunable thresholds in one place
├── pipeline.py            # Orchestrator (Phase 1 → 2 → 3)
├── __main__.py            # Entry point with dual logging
├── screener/              # Phase 1: quantitative
│   ├── universe.py        # UniverseLoader (yfinance screener API)
│   ├── market_data.py     # MarketDataFetcher (batch OHLCV download)
│   ├── technical.py       # TechnicalCalculator (MAs, volume, breakout)
│   └── quantitative_filter.py  # Hard numeric filters
├── analyzer/              # Phase 2: qualitative
│   ├── models.py          # Pydantic models (structured LLM output)
│   ├── prompts.py         # System + user prompt templates
│   ├── llm_analyzer.py    # LLMAnalyzer (batched OpenAI calls)
│   └── validator.py       # Post-LLM validation layer
└── export/                # Phase 3: output
    └── report.py          # CSV, XLSX, console formatting
```

## Configuration

All thresholds live in `src/stock_analyzer/config.py` → `ScreenerConfig`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `market_cap_min` | $500M | Minimum market cap |
| `market_cap_max` | $25B | Maximum market cap |
| `max_pct_from_52wk_high` | 8% | Must be within this % of 52-week high |
| `min_relative_volume` | 1.5x | Minimum 3-day max relative volume |
| `max_pct_above_breakout` | 30% | Maximum extension above breakout |
| `min_avg_daily_dollar_volume` | $10M | Liquidity floor |

## Design Principles

1. **Quantitative data is real** — all numbers come from yfinance, not from LLM guesses
2. **LLM only does what it's good at** — qualitative judgment, narrative assessment, catalyst research
3. **Defense in depth** — even if the LLM approves a stock, programmatic validation can reject it
4. **Batched LLM calls** — candidates are sent 4 at a time so the model actually searches per stock
5. **Logging for investigation** — every run produces a detailed `.log` file in `results/`
