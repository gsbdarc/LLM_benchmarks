"""
conftest.py — shared test setup for the eval package.

Ensures `import eval.<module>` resolves no matter where pytest is invoked from by
putting mcp/ (the package parent) on sys.path. pytest also does this via rootdir,
but doing it explicitly makes the tests runnable with plain `python -m pytest`
from any directory.
"""

import os
import sys
from pathlib import Path

# Disable Weave tracing for the whole test suite: the `op` decorator becomes a
# no-op, so unit tests never call weave.init or hit the network.
os.environ.setdefault("EVAL_DISABLE_WEAVE", "1")

MCP_DIR = Path(__file__).resolve().parents[2]  # eval/tests/ -> eval/ -> mcp/
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

# Repo root too, so `import analysis.queries` resolves in tests.
REPO_ROOT = MCP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
