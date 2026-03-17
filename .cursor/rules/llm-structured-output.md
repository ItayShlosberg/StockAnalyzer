# LLM Structured Output Pattern

## When This Applies

Any time you build a new analysis, pipeline, or script that calls an LLM (GPT, Gemini, etc.) and needs structured results (JSON that maps to DataFrame rows, CSV export, or downstream processing).

## Required Pattern: Pydantic + LangChain Structured Output

Always use the **Pydantic + `with_structured_output()`** pattern. Never use raw prompt-only JSON instructions with manual `json.loads` parsing.

### Step 1 — Define a Pydantic output model

Create a `BaseModel` subclass with typed, described fields. Use `Field(description=...)` and validation constraints (`ge`, `le`, `min_length`, etc.) to tighten the schema. Each field becomes a flat column when converted to a DataFrame — design field names as if they are column headers.

Use `Literal` for fields with a known set of valid values — this constrains the LLM at schema level, not just via prompt instructions.

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MyAnalysisResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the decision")
    category: Literal["category_a", "category_b", "category_c"] = Field(description="Classification category")
    confidence: int = Field(description="Confidence score", ge=0, le=100)
    match: bool = Field(description="Whether the condition is met")
```

> **Tip — `model_dump()`**: Call `result.model_dump()` to get a plain dict. No need to add a custom `to_dict()` method.

#### Optional and nested fields

Use `Optional[T]` when a field may legitimately be absent (e.g., an explanation that only applies when a condition is met). Use nested `BaseModel` subclasses or `list[NestedModel]` for structured sub-results:

```python
class Finding(BaseModel):
    issue: str = Field(description="Description of the issue found")
    severity: Literal["low", "medium", "high"] = Field(description="Severity level")

class AuditResult(BaseModel):
    summary: str = Field(description="Overall audit summary")
    passed: bool = Field(description="Whether the audit passed")
    findings: list[Finding] = Field(description="List of issues found, empty if passed")
    remediation: Optional[str] = Field(default=None, description="Suggested fix, only if audit failed")
```

### Step 2 — Write the prompt template

Use a `str` template with `{placeholder}` variables (LangChain `PromptTemplate` style). The prompt should describe the task and expected output semantics, but does **not** need to specify JSON format or output structure — `with_structured_output()` handles schema enforcement at the API level. The Pydantic field descriptions are automatically sent to the LLM as part of the schema.

```python
MY_PROMPT = """
You are an expert at [domain]. Analyze the following:

Input A:
{input_a}

Input B:
{input_b}

Provide your analysis.
"""
```

### Step 3 — Call `ChatClient.get_llm_response()`

Instantiate a `ChatClient` and call `get_llm_response()`, passing the template, Pydantic model, and input variables. The method returns a 3-tuple on success or `None` on failure.

```python
from enterprise_company_er.utils.chat_client import ChatClient

client = ChatClient()

result = client.get_llm_response(
    template=MY_PROMPT,
    output_model=MyAnalysisResult,
    input_variable_names=["input_a", "input_b"],
    input_variable_values=["some value", "another value"],
)

if result is not None:
    result_dict, content_hash, full_prompt = result
    # result_dict is a plain dict with validated, typed values
else:
    # LLM call failed after retries — handle gracefully
    pass
```

### Step 4 — Parallelize and collect into a DataFrame

Use `fetch_parallel` for batch processing, then flatten results into a DataFrame:

```python
import pandas as pd
from functools import partial
from enterprise_company_er.utils.parallel_processing import fetch_parallel

def process_single_row(row, llm_client):
    result = llm_client.get_llm_response(
        template=MY_PROMPT,
        output_model=MyAnalysisResult,
        input_variable_names=["input_a", "input_b"],
        input_variable_values=[row["col_a"], row["col_b"]],
    )
    if result is not None:
        result_dict, _, _ = result
        for key, value in result_dict.items():
            row[f"llm_{key}"] = value
    else:
        for key in MyAnalysisResult.model_fields:
            row[f"llm_{key}"] = None
    return row

label_func = partial(process_single_row, llm_client=client)
labeled_rows = fetch_parallel(df, label_func, mode="df", max_workers=10, desc="LLM analysis")
results_df = pd.DataFrame(labeled_rows)
```

---

## How It Works Under the Hood

Understanding the internals so you don't reimplement them:

### `ChatClient` capabilities

Import: `from enterprise_company_er.utils.chat_client import ChatClient`

- **Token management**: Automatically generates and refreshes M2M auth tokens via `generate_m2m_token()` / `validate_m2m_token()`. If a 401 occurs, it regenerates the token and retries.
- **Connection reuse**: Caches the `ChatOpenAI` instance and reuses it when the config (model, web_search, request_id, token) hasn't changed.
- **Retry with backoff**: Uses `tenacity` — retries on `RateLimitError`, `APIConnectionError`, `APITimeoutError` with exponential backoff (2s–60s), up to 5 attempts.
- **Error recovery**: If the LLM returns unparseable output, `ChatClient` sends the error message to a recovery prompt that asks the LLM to fix its response while still using `with_structured_output()`.
- **Request tracking**: Each `ChatClient` instance has a `request_id` (UUID) sent as a `jll-request-id` header for tracing.


### `get_llm_response()` return value

Returns `None` on failure, or a 3-tuple on success:

```python
(result_dict, content_hash, full_prompt)
```

- `result_dict`: A plain Python dict from `pydantic_model.model_dump()` — all fields validated and typed.
- `content_hash`: A SHA-256 hash of the full prompt prefixed with model/websearch info (e.g. `"gpt-5-chat-noweb_abc123..."`). Useful for caching/deduplication.
- `full_prompt`: The fully rendered prompt string (template with variables substituted).

### The LangChain chain

Internally, `ChatClient` builds a chain: `PromptTemplate | llm.with_structured_output(pydantic_model)`. The `with_structured_output()` call converts the Pydantic model's JSON schema into the API's `response_format` parameter (for OpenAI-compatible models) or function-calling schema. This means the LLM is **constrained at generation time** to produce valid JSON matching the schema — it cannot return malformed output.


---

## Why This Pattern (vs. Prompt-Only JSON)

| Concern | Prompt-only `json.loads` | Pydantic + `with_structured_output()` |
|---------|--------------------------|---------------------------------------|
| Schema enforcement | None — LLM may ignore instructions | API-level — LLM is constrained to the schema |
| Type safety | None — everything is `str` | Pydantic validates types and constraints |
| Malformed JSON | Crashes or silent errors | Cannot happen — API guarantees valid output |
| Markdown fences | Must strip `` ``` `` manually | Not needed |
| Error handling | Manual try/except on `json.loads` | Built-in retries, backoff, recovery chain |
| Token refresh | Manual | Automatic inside `ChatClient` |
| Caching support | Must build yourself | `content_hash` returned for free |

---

## Checklist for New LLM Analyses

1. Define a Pydantic `BaseModel` with typed fields and `Field(description=...)`
2. Use `Literal[...]` for any field with a known set of valid values — not `str` with enum listed in the description
3. Use `Optional[T]` for fields that may be absent; use nested models or `list[NestedModel]` for structured sub-results
4. Write a prompt template with `{placeholder}` variables — no JSON format instructions needed
5. Use `ChatClient.get_llm_response(template, output_model, ...)` — never raw `requests.post`
6. The return is `(result_dict, content_hash, full_prompt)` or `None` on failure
7. Prefix LLM output columns with `llm_` when merging into existing DataFrames
8. Handle `None` results (LLM call failures) gracefully — set all `llm_*` columns to `None`
9. Use `fetch_parallel` for batch processing with configurable `max_workers`
10. Use `content_hash` if you need to cache results or avoid re-processing identical prompts