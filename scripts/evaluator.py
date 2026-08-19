"""
evaluator.py — Data-driven evaluation engine for OCR benchmark pipeline.
 
Metric functions are registered by type. The benchmark spec declares which
metrics apply to each field, and this module routes accordingly.
 
Usage:
    from evaluator import evaluate_task
 
    result = evaluate_task(
        model_output={"first_channel_raw": "2 CBS", "first_channel_name": "CBS", ...},
        ground_truth_doc={"first_channel_raw": "2 CBS WCBS", ...},
        benchmark_spec=benchmarks["7"]
    )
"""
 
from __future__ import annotations
 
import re
from collections import Counter
from typing import Any
 
 
# ---------------------------------------------------------------------------
#  Normalisation helpers
# ---------------------------------------------------------------------------
 
def _normalize_text(s: str | None) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
 
 
def _tokenize(s: str) -> list[str]:
    """Split on whitespace after normalisation."""
    normed = _normalize_text(s)
    return normed.split() if normed else []
 
 
def _is_null(val: Any) -> bool:
    """Determine if a value should be treated as null/absent."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "null", "none", "n/a"):
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    return False
 
 
# ---------------------------------------------------------------------------
#  RAW STRING metrics  (word-level IoU / Jaccard)
# ---------------------------------------------------------------------------
 
def word_iou(predicted: str | None, expected: str | None) -> dict:
    """
    Word-Level Intersection-over-Union (Jaccard Index).
 
    |intersection(pred_words, gt_words)| / |union(pred_words, gt_words)|
 
    Returns dict with 'word_iou' score in [0.0, 1.0].
    """
    pred_tokens = set(_tokenize(predicted))
    gt_tokens = set(_tokenize(expected))
 
    if not pred_tokens and not gt_tokens:
        return {"word_iou": 1.0}  # both empty = perfect match
    if not pred_tokens or not gt_tokens:
        return {"word_iou": 0.0}
 
    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    score = len(intersection) / len(union)
    return {"word_iou": round(score, 4)}
 
 
# ---------------------------------------------------------------------------
#  EXTRACTED STRING metrics  (null accuracy, Levenshtein, char-level F1)
# ---------------------------------------------------------------------------
 
def null_accuracy(predicted: Any, expected: Any) -> dict:
    """
    Binary classification: Is-Null vs Is-Not-Null.
 
    Returns TP (True Positive), FP (False Positive), FN (False Negative), TN (True Negative) counts and derived precision, recall, F1.
    Treats a single (pred, gt) pair as one sample.
    """
    pred_null = _is_null(predicted)
    gt_null = _is_null(expected)
 
    tp = int(not pred_null and not gt_null)  # both present
    tn = int(pred_null and gt_null)           # both null
    fp = int(not pred_null and gt_null)       # predicted present, gt null
    fn = int(pred_null and not gt_null)       # predicted null, gt present
 
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
 
    return {
        "null_accuracy": round(f1, 4),
        "null_tp": tp, "null_tn": tn, "null_fp": fp, "null_fn": fn,
        "null_precision": round(precision, 4),
        "null_recall": round(recall, 4),
    }
 
 
def levenshtein_distance(s1: str, s2: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
 
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # substitute
            ))
        prev_row = curr_row
    return prev_row[-1]
 
 
def levenshtein_similarity(predicted: str | None, expected: str | None) -> dict:
    """
    Normalised Levenshtein similarity: 1 - (edit_dist / max_len).
 
    Both values are lowercased + stripped before comparison.
    If both are null/empty, returns 1.0 (perfect match on absence).
    """
    p = _normalize_text(predicted)
    e = _normalize_text(expected)
 
    if not p and not e:
        return {"levenshtein_similarity": 1.0, "levenshtein_distance": 0}
    if not p or not e:
        max_len = max(len(p), len(e))
        return {"levenshtein_similarity": 0.0, "levenshtein_distance": max_len}
 
    dist = levenshtein_distance(p, e)
    max_len = max(len(p), len(e))
    similarity = 1.0 - (dist / max_len)
    return {
        "levenshtein_similarity": round(similarity, 4),
        "levenshtein_distance": dist,
    }
 
 
def char_f1(predicted: str | None, expected: str | None) -> dict:
    """
    Character-level F1 score.
 
    Treats each string as a multiset of characters, computes precision/recall/F1.
    Useful for catching partial correctness vs complete hallucination.
    """
    p = _normalize_text(predicted)
    e = _normalize_text(expected)
 
    if not p and not e:
        return {"char_f1": 1.0, "char_precision": 1.0, "char_recall": 1.0}
    if not p or not e:
        return {"char_f1": 0.0, "char_precision": 0.0, "char_recall": 0.0}
 
    pred_counts = Counter(p)
    gt_counts = Counter(e)
 
    # intersection = sum of min counts for each char
    common = sum((pred_counts & gt_counts).values())
 
    precision = common / sum(pred_counts.values()) if sum(pred_counts.values()) > 0 else 0.0
    recall = common / sum(gt_counts.values()) if sum(gt_counts.values()) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
 
    return {
        "char_f1": round(f1, 4),
        "char_precision": round(precision, 4),
        "char_recall": round(recall, 4),
    }
 
 
# ---------------------------------------------------------------------------
#  LIST metrics  (set-based F1, sequence LCS)
# ---------------------------------------------------------------------------
 
def _normalize_list(val: Any) -> list[str]:
    """Coerce to list of normalised strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [_normalize_text(val)]
    if isinstance(val, list):
        return [_normalize_text(str(v)) for v in val]
    return [_normalize_text(str(val))]
 
 
def set_f1(predicted: Any, expected: Any) -> dict:
    """
    Set-based Precision / Recall / F1.
 
    Treats predicted and expected as sets of strings (order-independent).
    """
    pred_set = set(_normalize_list(predicted))
    gt_set = set(_normalize_list(expected))
 
    # Remove empty strings from sets
    pred_set.discard("")
    gt_set.discard("")
 
    if not pred_set and not gt_set:
        return {"set_f1": 1.0, "set_precision": 1.0, "set_recall": 1.0}
    if not pred_set or not gt_set:
        return {"set_f1": 0.0, "set_precision": 0.0, "set_recall": 0.0}
 
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
 
    return {
        "set_f1": round(f1, 4),
        "set_precision": round(precision, 4),
        "set_recall": round(recall, 4),
    }
 
 
def sequence_lcs(predicted: Any, expected: Any) -> dict:
    """
    Longest Common Subsequence (LCS) normalised similarity.
 
    LCS_length / max(len(pred), len(gt))
 
    Captures whether the model got the right items in the right order.
    """
    pred_list = _normalize_list(predicted)
    gt_list = _normalize_list(expected)
 
    # Filter out empty strings
    pred_list = [x for x in pred_list if x]
    gt_list = [x for x in gt_list if x]
 
    if not pred_list and not gt_list:
        return {"sequence_lcs": 1.0, "lcs_length": 0}
    if not pred_list or not gt_list:
        return {"sequence_lcs": 0.0, "lcs_length": 0}
 
    m, n = len(pred_list), len(gt_list)
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_list[i - 1] == gt_list[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
 
    lcs_len = dp[m][n]
    max_len = max(m, n)
    similarity = lcs_len / max_len
 
    return {
        "sequence_lcs": round(similarity, 4),
        "lcs_length": lcs_len,
    }

def set_inclusion(predicted: Any, expected: Any) -> dict:
    """
    Set-based Inclusion Precision / Recall / F1.

    Checks if the expected string is a substring of ANY predicted string (Recall),
    and if a predicted string contains ANY expected string (Precision).
    """
    pred_list = _normalize_list(predicted)
    gt_list = _normalize_list(expected)

    # Filter out empty strings
    pred_list = [x for x in pred_list if x]
    gt_list = [x for x in gt_list if x]

    if not pred_list and not gt_list:
        return {"set_inclusion": 1.0, "inclusion_precision": 1.0, "inclusion_recall": 1.0}
    if not pred_list or not gt_list:
        return {"set_inclusion": 0.0, "inclusion_precision": 0.0, "inclusion_recall": 0.0}

    # Recall: What fraction of ground truth items are found inside at least one predicted item?
    # e.g. Is "a&e" found inside "a&e | 52 | 50"?
    matched_gt = sum(1 for gt in gt_list if any(gt in pred for pred in pred_list))
    recall = matched_gt / len(gt_list)

    # Precision: What fraction of predicted items contain at least one ground truth item?
    matched_pred = sum(1 for pred in pred_list if any(gt in pred for gt in gt_list))
    precision = matched_pred / len(pred_list)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "set_inclusion": round(f1, 4),
        "inclusion_precision": round(precision, 4),
        "inclusion_recall": round(recall, 4),
    }
 
# ---------------------------------------------------------------------------
#  Metric registry — maps metric name → function
# ---------------------------------------------------------------------------
 
METRIC_REGISTRY = {
    "word_iou":     word_iou,
    "null_accuracy": null_accuracy,
    "levenshtein":  levenshtein_similarity,
    "char_f1":      char_f1,
    "set_f1":       set_f1,
    "sequence_lcs": sequence_lcs,
    "set_inclusion": set_inclusion,
}


# ---------------------------------------------------------------------------
#  Per-type composite scoring (single source of truth)
#
#  Each composite bundles the per-type metrics and applies the same formula
#  used historically inside ``evaluate_task``. Both the deterministic pipeline
#  (``evaluate_task`` -> ``scripts/6b_mongo_eval.py``) and the agentic MCP
#  type-tools (``mcp/metric_server.py``) call these, so the agent and the
#  baseline produce identical composites by construction.
#
#  Return shape:
#      {
#        "composite_score": float,          # the per-type composite
#        "components": {<metric>: value},   # all sub-scores that fed it
#        "field_type": "raw_string" | "extracted_string" | "list",
#      }
# ---------------------------------------------------------------------------

def composite_raw_string(predicted: Any, expected: Any) -> dict:
    """raw_string composite: word IoU is the sole signal."""
    components = word_iou(predicted, expected)
    return {
        "composite_score": float(components.get("word_iou", 0.0)),
        "components": components,
        "field_type": "raw_string",
    }


def composite_extracted_string(predicted: Any, expected: Any) -> dict:
    """extracted_string composite: null_accuracy gates the content score.

    If presence detection is wrong the content score is irrelevant; if it is
    right, use the better of levenshtein_similarity / char_f1.
    """
    components: dict = {}
    components.update(null_accuracy(predicted, expected))
    components.update(levenshtein_similarity(predicted, expected))
    components.update(char_f1(predicted, expected))
    null_score = components.get("null_accuracy", 0.0)
    content_score = max(
        components.get("levenshtein_similarity", 0.0),
        components.get("char_f1", 0.0),
    )
    return {
        "composite_score": float(null_score * content_score),
        "components": components,
        "field_type": "extracted_string",
    }


def composite_list(predicted: Any, expected: Any) -> dict:
    """list composite: best of set_f1 / sequence_lcs / set_inclusion."""
    components: dict = {}
    components.update(set_f1(predicted, expected))
    components.update(sequence_lcs(predicted, expected))
    components.update(set_inclusion(predicted, expected))
    composite = max(
        components.get("set_f1", 0.0),
        components.get("sequence_lcs", 0.0),
        components.get("set_inclusion", 0.0),
    )
    return {
        "composite_score": float(composite),
        "components": components,
        "field_type": "list",
    }


COMPOSITE_BY_TYPE = {
    "raw_string": composite_raw_string,
    "extracted_string": composite_extracted_string,
    "list": composite_list,
}


def evaluate_field_composite(
    field_type: str,
    predicted: Any,
    expected: Any,
    fallback_metrics: list | None = None,
) -> dict:
    """Dispatch to the correct per-type composite.

    Unknown ``field_type`` falls back to the first declared metric (matching the
    legacy behavior in ``evaluate_task``).
    """
    fn = COMPOSITE_BY_TYPE.get(field_type)
    if fn is not None:
        return fn(predicted, expected)

    components: dict = {}
    composite: Any = 0.0
    if fallback_metrics:
        primary = fallback_metrics[0]
        mfn = METRIC_REGISTRY.get(primary)
        if mfn is not None:
            components.update(mfn(predicted, expected))
            composite = components.get(primary, 0.0)
    if isinstance(composite, dict):  # metric returned an error payload
        composite = 0.0
    return {
        "composite_score": float(composite),
        "components": components,
        "field_type": field_type,
    }
 
 
# ---------------------------------------------------------------------------
#  Ground truth field resolver
# ---------------------------------------------------------------------------
 
def _resolve_gt_value(ground_truth_doc: dict, gt_field_path: str) -> Any:
    """
    Resolve a possibly-dotted field path from the ground truth document.
 
    e.g. "all_times.hours" → ground_truth_doc["all_times"]["hours"]
    """
    parts = gt_field_path.split(".")
    val = ground_truth_doc
    for part in parts:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            return None
    return val
 
 
# ---------------------------------------------------------------------------
#  Main evaluation entry point
# ---------------------------------------------------------------------------
 
def evaluate_task(
    model_output: dict,
    ground_truth_doc: dict,
    benchmark_spec: dict,
) -> dict:
    """
    Evaluate a single task result against ground truth using the benchmark spec.
 
    Args:
        model_output:      The 'output' object from the task_result doc.
        ground_truth_doc:  The ground truth doc for this image_id.
        benchmark_spec:    The enriched benchmark spec (with typed ground_truth).
 
    Returns:
        {
            "field_details": {
                "first_channel_raw": {
                    "predicted": "2 CBS",
                    "expected": "2 CBS WCBS",
                    "scores": {"word_iou": 0.67}
                }, ...
            },
            "weighted_score": 0.78,
            "weights_used": {"first_channel_raw": 0.5, ...}
        }
    """
    gt_spec = benchmark_spec.get("ground_truth", {})
 
    field_details = {}
    weighted_sum = 0.0
    total_weight = 0.0
 
    for field_key, field_def in gt_spec.items():
        gt_field_path = field_def["gt_field"]
        output_field = field_def["output_field"]
        metrics = field_def["metrics"]
        weight = field_def.get("weight", 1.0)
 
        # Resolve values
        predicted = model_output.get(output_field)
        expected = _resolve_gt_value(ground_truth_doc, gt_field_path)
 
        # Compute each declared metric for this field (kept for the per-field
        # `scores` detail that downstream consumers read).
        scores = {}
        for metric_name in metrics:
            fn = METRIC_REGISTRY.get(metric_name)
            if fn is None:
                scores[metric_name] = {"error": f"Unknown metric: {metric_name}"}
                continue
            scores.update(fn(predicted, expected))

        # Composite score via the shared per-type helper — the single source of
        # truth also used by the agentic MCP type-tools, so the deterministic
        # baseline and the agent produce identical composites by construction.
        field_type = field_def.get("type", "raw_string")
        composite = evaluate_field_composite(field_type, predicted, expected, metrics)
        composite_score = composite["composite_score"]
        # Surface any sub-scores the composite computed that weren't declared.
        for sub_key, sub_val in composite["components"].items():
            scores.setdefault(sub_key, sub_val)

        weighted_composite_score = composite_score * weight
        weighted_sum += composite_score * weight
        total_weight += weight

        field_details[field_key] = {
            "predicted": predicted,
            "expected": expected,
            "composite_score": composite_score,
            "weighted_composite_score": weighted_composite_score,
            "scores": scores,
        }
    
    weighted_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # Override the score to be the max if we are evaluating the all_times_1 benchmark
    task_name = benchmark_spec.get("task_name")
    
    if task_name == "all_times_1": 
        if field_details:
            # Extract the max composite_score from the evaluated fields
            weighted_score = max(detail["composite_score"] for detail in field_details.values())
        else:
            weighted_score = 0.0
    # --------------------------

    return {
        "field_details": field_details,
        "weighted_score": round(weighted_score, 4),
        "weights_used": {k: v["weight"] for k, v in gt_spec.items()},
    }
 
 