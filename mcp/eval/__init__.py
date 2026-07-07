"""
eval — agentic metric-evaluation package.

Extracted from mcp/MetricEval.ipynb so the agent loop can be run as a script,
batched over a SLURM array, and instrumented consistently. The notebook is now a
thin demo that imports from here.

Modules:
  config         backend selection, OpenAI client, paths, Weave project
  observability  hashes, reasoning/token derivations, vLLM /v1/metrics scrape
  agent          MCP tool discovery + the agent loop (persistent MCP session)
  integrity      post-run consistency / retry checks
  scorers        weave.Evaluation scorers
  sink           flatten one run -> one Parquet row (the OLAP layer)
"""
