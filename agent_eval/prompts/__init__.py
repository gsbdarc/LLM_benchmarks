"""
prompts — the metric-eval agent's prompt registry.

Each prompt variant is a JSON file in this directory:

    {"index": <int>, "name": <version>, "system": [<system lines>], "user": "<template>"}

`index` gives the variant a stable integer key, mirroring the int-keyed `models` map
in `backends/*.json`: a run records the derived `prompt_key` so prompt variants group
cleanly in the dashboard, and a sweep is a pure data change (drop in a new JSON file).
`system` is stored line-by-line (a JSON array) purely so the prompt stays readable in
review; the loader joins it with newlines. `user` is a template whose only placeholders
are {task_id} and {run_id}.

`resolve_prompt` is the prompt analogue of `config.resolve_model`: it accepts an int
index, a name string, or None (the default = lowest index) and returns
(name, system, user, key). The default variant is also exposed as PROMPT_NAME /
METRIC_EVAL_SYSTEM / eval_user_prompt() — the same public API the package used when
this was a single prompts.py module, so nothing downstream changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent


def _load_file(path: Path) -> dict[str, Any]:
    """Load one prompt JSON, joining the line-array `system` into a string."""
    data = json.loads(path.read_text())
    system = data["system"]
    if isinstance(system, list):
        system = "\n".join(system)
    return {"index": int(data["index"]), "name": data["name"], "system": system, "user": data["user"]}


def _load_registry() -> dict[int, dict[str, Any]]:
    """Scan prompts/*.json into {index: variant}. Errors on a duplicate index."""
    reg: dict[int, dict[str, Any]] = {}
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        variant = _load_file(path)
        idx = variant["index"]
        if idx in reg:
            raise ValueError(f"duplicate prompt index {idx} ({path.name} vs {reg[idx]['name']})")
        reg[idx] = variant
    if not reg:
        raise FileNotFoundError(f"no prompt JSON files found in {PROMPTS_DIR}")
    return reg


_REGISTRY = _load_registry()
_BY_NAME = {v["name"]: v for v in _REGISTRY.values()}
DEFAULT_KEY = min(_REGISTRY)  # lowest index is the default, mirroring resolve_model


def resolve_prompt(prompt: int | str | None = None) -> tuple[str, str, str, int]:
    """Resolve a prompt selector to (name, system, user, key).

    `prompt` accepts an int index, a name string, an integer-like string, or None
    (the default = the lowest index). Mirrors `config.resolve_model`.
    """
    if prompt is None:
        v = _REGISTRY[DEFAULT_KEY]
    elif isinstance(prompt, int) or (isinstance(prompt, str) and prompt.isdigit()):
        key = int(prompt)
        if key not in _REGISTRY:
            raise KeyError(f"no prompt with index {key} (have {sorted(_REGISTRY)})")
        v = _REGISTRY[key]
    else:
        if prompt not in _BY_NAME:
            raise KeyError(f"no prompt named {prompt!r} (have {sorted(_BY_NAME)})")
        v = _BY_NAME[prompt]
    return v["name"], v["system"], v["user"], v["index"]


def prompt_names() -> list[str]:
    """All registered prompt-variant names, ordered by index (default first)."""
    return [_REGISTRY[k]["name"] for k in sorted(_REGISTRY)]


# The default variant, exposed with the pre-registry API for back-compat.
PROMPT_NAME, METRIC_EVAL_SYSTEM, _DEFAULT_USER, PROMPT_KEY = resolve_prompt()


def eval_user_prompt(
    task_id: str, run_id: int | str | None, prompt: str | None = None
) -> str:
    """The per-output instruction the agent receives (the resolved prompt's `user`).

    `run_id` is only interpolated into the text, so it accepts an int (real runs),
    a str (e.g. the dashboard's `"<run_id>"` placeholder), or None. `prompt` optionally
    selects a non-default variant by name or index (defaults to the active one).
    """
    _, _, template, _ = resolve_prompt(prompt)
    return template.format(task_id=task_id, run_id=run_id)
