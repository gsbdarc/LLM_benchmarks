"""analysis — DuckDB queries, the concurrency sweep, and the dashboard.

The OLAP/visualization layer over the Parquet run-summaries written by
agent_eval/reporting/sink.py. Kept separate from the eval package: it only reads Parquet,
so it has no dependency on the agent runtime.
"""
