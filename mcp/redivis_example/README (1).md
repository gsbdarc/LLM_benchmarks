# Redivis MCP — Revelio Record Linkage

An agent pipeline that matches person records (e.g. from a Drexel alumni sample) to users in the **Revelio Labs Workforce** dataset on Redivis. An LLM drives the search by calling Redivis queries exposed as tools over the Model Context Protocol (MCP).

```
┌─────────────┐    OpenAI-compatible     ┌──────────────┐
│  Notebook   │ ───────────────────────▶ │ LLM backend  │
│ (orchestr.) │                          │  Playground  │
│             │ ◀─── tool_calls ──────── │   or NIM     │
└─────┬───────┘                          └──────────────┘
      │ MCP (streamable HTTP)
      ▼
┌─────────────┐    SQL queries           ┌──────────────┐
│ MCP server  │ ───────────────────────▶ │   Redivis    │
│ (FastMCP)   │ ◀──── result rows ────── │ StanfordGSB  │
└─────────────┘                          └──────────────┘
```

## Files

| File | Role |
| --- | --- |
| `server_http.py` | FastMCP server exposing two tools over streamable HTTP. |
| `redivis_tools.py` | Redivis SQL helpers — input validation, retry, result shaping. |
| `run_redivis_mcp.sh` | Starts the MCP server on `127.0.0.1:33061`. |
| `RecordLinkage.ipynb` | Clean agent notebook. Switch backend with one flag. |
| `AiPlayground.ipynb` | Original notebook (kept for reference). |
| `api_darc.txt` | Stanford AI Playground API key (untracked, you provide it). |
| `212054_Sample_rev.xlsx` | Sample input records. |

## Prerequisites

- Python venv at `./venv` with `mcp`, `openai`, `redivis`, `pandas`, `python-dotenv`, `openpyxl`.
- Redivis credentials (env or `.env`) — `redivis_tools.py` calls `load_dotenv()` and then uses the `redivis` Python client.
- For the Playground backend: API key in `api_darc.txt`.
- For the NIM backend: a vLLM/NIM server reachable at `http://yen-gpu4:8000/v1` (or whatever you set).

## Pipeline step by step

### 1. Start the MCP server

```bash
./run_redivis_mcp.sh
```

This runs `server_http.py`, which creates a `FastMCP` instance and registers two tools with `@mcp.tool()`:

- `search_person(firstname, lastname, user_country, limit)` — finds candidate user_ids from `individual_user`.
- `fetch_user_positions(user_id, limit)` — pulls position history from `individual_position`.

Both tool bodies are wrapped to return **structured error dicts** (`{"error": "...", "rows": [], ...}`) instead of raising, so the LLM can recover in-loop. The server listens on `http://127.0.0.1:33061/mcp`.

### 2. Tool discovery in the notebook

Open `RecordLinkage.ipynb`. The `load-tools` cell connects to the MCP server and calls `session.list_tools()`, then converts each tool's MCP schema into the OpenAI Chat Completions `tools=[...]` format:

```python
{"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}
```

This means the LLM's view of the tools is **generated from the same `@mcp.tool()` decorators** that define the server — no hand-written schema to drift.

### 3. Pick a backend

In the `config` cell, set `BACKEND = "playground"` or `"nim"`. The config dict controls:

- `base_url` and API key — `https://aiapi-prod.stanford.edu/v1` vs. `http://yen-gpu4:8000/v1`.
- `model` — `gpt-5.2` vs. `google/gemma-4-31b-it`.
- `completion_kwargs` — backend-specific arguments. Only the Playground gets `reasoning_effort="high"`; only NIM gets `temperature`, `parallel_tool_calls=False`, and `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`.

### 4. Run the agent loop

`run_agent(user_prompt, system_prompt, max_steps)` in the `agent` cell is the core loop:

1. Build `messages = [system, user]`.
2. Call `llm_client.chat.completions.create(..., tools=tools, tool_choice="auto", **COMPLETION_KWARGS)`.
3. If the reply has no `tool_calls`, return it as the final answer.
4. Otherwise, for each requested tool call:
   - Parse `tc.function.arguments` (JSON string) into `args`.
   - Open an MCP session via `streamablehttp_client(MCP_URL)` and `session.call_tool(name, args)`.
   - Convert the MCP `content` blocks to plain text and append a `role="tool"` message referencing `tc.id`.
5. Loop back to step 2 until the model returns a final answer or `max_steps` is hit.

The LLM never talks to the MCP server directly — the notebook is both the OpenAI client and the MCP client.

### 5. Reasoning visibility

Two sources of model reasoning are surfaced in the log:

- **NIM / Gemma**: `enable_thinking=True` emits reasoning inside `<think>...</think>` inside `message.content`. `split_thinking()` extracts those blocks and prints them as `REASONING (<think>)`. The raw message (tags intact) is still appended to `messages`, because Gemma's chat template expects to see them on subsequent turns.
- **Playground / gpt-5.2**: `extract_reasoning()` checks `message.reasoning_content`, `message.reasoning`, and `response.reasoning`. If the gateway forwards any of them, it prints as `REASONING (api)`; otherwise nothing extra appears.

### 6. Data flow for one record

```
df.iloc[ROW_INDEX]            →  "user" string (all fields)
run_agent(user_prompt, sys)   →  LLM step 1  →  tool_call search_person("Antelo", "Devereux", "United States")
                              →  MCP → Redivis SQL on individual_user → candidate user_ids
                              →  LLM step 2  →  tool_call fetch_user_positions(user_id=123…)
                              →  MCP → Redivis SQL on individual_position → position rows
                              →  LLM step N  →  final match + profile
```

## Resilience details

`redivis_tools._run_query` retries up to 3 times on transient failures — `ConnectionError`, `TimeoutError`, or messages containing `timeout|temporarily|unavailable|connection|502|503|504`. Backoff is exponential (0.5s, 1s, 2s). Non-transient errors (SQL syntax, auth) bubble immediately. Input validation rejects empty names and bad limits with clear `ValueError`s, which `server_http.py` turns into `{"error": "invalid input: ..."}` responses.

## Running a batch

```python
results = await run_batch([99, 210, 350])
```

`run_batch` runs `run_agent` for a list of row indices with `verbose=False` and returns `[{row, answer, usage}, ...]`.

## Troubleshooting

- **`No module named 'mcp'`** → activate the venv (`source venv/bin/activate`).
- **`ConnectionRefusedError` on MCP** → server isn't running; start `./run_redivis_mcp.sh`.
- **LLM says "no tools available"** → re-run the `load-tools` cell; it reads from the live server.
- **`tool_calls` arrive but fail with `invalid input`** → the model called a tool with an empty/malformed arg; the error dict is the signal to the model to retry. If it recurs, tighten the tool description or system prompt.
- **Playground rejects `extra_body`** → ensure `BACKEND="playground"` so NIM-only kwargs don't get sent.
