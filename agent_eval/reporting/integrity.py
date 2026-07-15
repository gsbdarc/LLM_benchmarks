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


def count_failed_saves(messages):
    """Number of save_evaluation TOOL RESULTS that reported failure.

    Reads the results (not just the calls) so we can tell a save that errored on
    the first attempt from a redundant second save — save_agentic_evaluation
    returns {"saved": false, "error": ...} on failure.
    """
    failed = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("name") == "save_evaluation":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                errored = isinstance(data, dict) and (data.get("error") or data.get("saved") is False)
            except (json.JSONDecodeError, TypeError):
                errored = '"error"' in str(content) or "isError=True" in str(content)
            if errored:
                failed += 1
    return failed


def save_outcome(messages):
    """(attempts, failed, succeeded) for save_evaluation across a run."""
    _, attempts = extract_saved_evaluation(messages)
    failed = count_failed_saves(messages)
    return attempts, failed, max(0, attempts - failed)


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
    save_count, save_failed, save_ok = save_outcome(messages)
    consistency = check_score_consistency(messages)
    retries = check_retry_rule(messages)
    # Success = the evaluation was saved (>=1 successful save) and the run answered.
    # save_failed (a save attempt that errored) is tracked separately so the
    # "couldn't save on the first go" nuance isn't lost when a retry succeeds.
    save_success = save_ok >= 1 and result.get("stopped_reason") == "answered"

    if verbose:
        print("\n--- INTEGRITY REPORT ---")
        print(f"  save_evaluation calls  : {save_count}x  (failed={save_failed}, "
              f"save_success={save_success})")
        print(f"  score consistency      : consistent={consistency['consistent']}  "
              f"missing={consistency.get('missing', [])}")
        if retries["total_errors"]:
            print(f"  retry behavior         : {retries['tool_errors_with_retry']}")
        else:
            print("  retry behavior         : no tool errors")
        print("------------------------")

    return {
        "save_count": save_count,
        "save_failed": save_failed,
        "save_success": save_success,
        "score_consistency": consistency,
        "retry_behavior": retries,
    }
