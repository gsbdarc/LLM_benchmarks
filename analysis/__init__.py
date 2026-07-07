"""analysis — DuckDB queries, the concurrency sweep, and the dashboard.

The OLAP/visualization layer over the Parquet run-summaries written by
mcp/eval/sink.py. Kept separate from the eval package: it only reads Parquet,
so it has no dependency on the agent runtime.
"""
