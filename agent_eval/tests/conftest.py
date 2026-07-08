"""
conftest.py — shared test setup for the agent_eval package.

Puts the repo root on sys.path so `import agent_eval.<module>` and
`import analysis.queries` resolve no matter where pytest is invoked from. pytest
also does this via rootdir, but doing it explicitly keeps `python -m pytest`
runnable from any directory.
"""

import os
import sys
from pathlib import Path

# Disable Weave tracing for the whole test suite: the `op` decorator becomes a
# no-op, so unit tests never call weave.init or hit the network.
os.environ.setdefault("EVAL_DISABLE_WEAVE", "1")

# agent_eval/tests/ -> agent_eval/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
