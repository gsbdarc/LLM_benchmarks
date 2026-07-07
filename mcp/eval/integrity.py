"""
integrity.py — post-run consistency checks (notebook cell 65889249, unchanged logic).

The headline check is score-consistency: every numeric score the agent saved must
have appeared in a prior metric-tool result. This catches the worst silent
failure — the agent inventing scores instead of forwarding what the tools
returned.
"""

from __future__ import annotations

import json

from .observability import op

# The agent-facing scoring tools whose results are the source of truth for the
# score-consistency check. With the composite consolidation these are the three
# type-tools (each returns composite_score + flattened sub-scores).
METRIC_TOOLS = {
    "evaluate_raw_string", "evaluate_extracted_string", "evaluate_list",
}


def extract_saved_evaluation(messages):
    """Return (list_of_save_args, call_count) by scanning assistant tool_calls."""
    saves = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.function.name == "save_evaluation":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                saves.append(args)
    return saves, len(saves)


def check_score_consistency(messages):
    """Verify every saved numeric score appeared in a prior metric-tool result."""
    observed = set()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("name") in METRIC_TOOLS:
            try:
                data = json.loads(msg["content"])
                for v in data.values():
                    if isinstance(v, (int, float)):
                        observed.add(float(v))
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    saves, _ = extract_saved_evaluation(messages)
    if not saves:
        return {"consistent": None, "missing": [], "note": "no save_evaluation found"}

    missing = []
    for fe in saves[0].get("field_evaluations", []):
        for key, val in fe.get("scores", {}).items():
            if isinstance(val, (int, float)):
                if not any(abs(float(val) - v) < 1e-9 for v in observed):
                    missing.append({"field": fe.get("field"), "score_key": key, "value": val})
    return {"consistent": len(missing) == 0, "missing": missing}


def check_retry_rule(messages):
    """For each tool error, check whether the same tool was re-called (any args)."""
    call_map = {}  # tool_call_id -> (name, args)
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            call_map[tc.id] = (tc.function.name, args)

    error_names = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
            if '"error"' in content or "isError=True" in content:
                tc_id = msg.get("tool_call_id", "")
                if tc_id in call_map:
                    error_names.append(call_map[tc_id][0])

    retry_info = {}
    for name in set(error_names):
        total_calls = sum(1 for n, _ in call_map.values() if n == name)
        error_count = error_names.count(name)
        retry_info[name] = {
            "errors": error_count,
            "total_calls": total_calls,
            "retried": total_calls > error_count,
        }
    return {"tool_errors_with_retry": retry_info, "total_errors": len(error_names)}


@op
def run_integrity_report(result, verbose=True):
    """Run all checks, optionally print a summary, and return a dict (Weave-traced)."""
    messages = result.get("messages", [])
    saves, save_count = extract_saved_evaluation(messages)
    consistency = check_score_consistency(messages)
    retries = check_retry_rule(messages)
    save_success = save_count == 1 and result.get("stopped_reason") == "answered"

    if verbose:
        print("\n--- INTEGRITY REPORT ---")
        print(f"  save_evaluation called : {save_count}x  (save_success={save_success})")
        print(f"  score consistency      : consistent={consistency['consistent']}  "
              f"missing={consistency.get('missing', [])}")
        if retries["total_errors"]:
            print(f"  retry behavior         : {retries['tool_errors_with_retry']}")
        else:
            print("  retry behavior         : no tool errors")
        print("------------------------")

    return {
        "save_count": save_count,
        "save_success": save_success,
        "score_consistency": consistency,
        "retry_behavior": retries,
    }
