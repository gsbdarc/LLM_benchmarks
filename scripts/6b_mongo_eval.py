"""
pipeline.py — MongoDB evaluation pipeline with backfill + change stream.

Connects the evaluator to MongoDB: reads task_results, joins with benchmarks
and ground_truths, computes metrics, and writes to an evaluations collection.

Usage:
    # Backfill all existing results
    python pipeline.py --backfill

    # Watch for new results in real-time
    python pipeline.py --watch

    # Both: backfill then watch
    python pipeline.py --backfill --watch

Environment:
    MONGO_URI  — MongoDB connection string (default: mongodb://localhost:27017)
    MONGO_DB   — Database name (default: tv_guide_benchmarks)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dotenv import load_dotenv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient, UpdateOne, ReplaceOne
from pymongo.server_api import ServerApi
from pymongo.errors import PyMongoError

from evaluator import evaluate_task, aggregate_scores

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

username = os.getenv("MONGO_DB_USERNAME")
password = os.getenv("MONGO_DB_PASSWORD")

hosts = [
    'darc-data-shard-00-00.9fjam.mongodb.net:27017',
    'darc-data-shard-00-01.9fjam.mongodb.net:27017',
    'darc-data-shard-00-02.9fjam.mongodb.net:27017'
]
setName = 'DARC-Data-shard-0'
uri = (
    f"mongodb://{username}:{password}@{','.join(hosts)}/"
    f"?tls=true&replicaSet={setName}&authSource=admin&retryWrites=true&w=majority&appName=DARC-Data"
)


# Collections
TASK_RESULTS_COLL = "llm_outputs"
BENCHMARKS_COLL = "benchmarks"
GROUND_TRUTHS_COLL = "ground_truths"
EVALUATIONS_COLL = "evaluations"

# Which benchmark IDs to evaluate
BENCHMARK_IDS = [str(i) for i in range(4, 13)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Database helpers
# ---------------------------------------------------------------------------

def get_db():
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["usf-internship"]
    return db


def ensure_indexes(db):
    """Create indexes for efficient lookups and change stream filtering."""
    db[TASK_RESULTS_COLL].create_index([("benchmark_id", 1), ("status", 1)])
    db[TASK_RESULTS_COLL].create_index([("image_id", 1)])
    db[EVALUATIONS_COLL].create_index(
        [("task_id", 1), ("benchmark_id", 1), ("model_id", 1)],
        unique=True,
    )
    db[EVALUATIONS_COLL].create_index([("model_id", 1), ("benchmark_id", 1)])
    logger.info("Indexes ensured.")


# ---------------------------------------------------------------------------
#  Seed reference data from JSON files
# ---------------------------------------------------------------------------

def seed_benchmarks(db, filepath: str = str(os.path.join(BASE_DIR, "inputs", "benchmarks.json"))):
    """Load enriched benchmarks into MongoDB."""
    with open(filepath) as f:
        benchmarks = json.load(f)

    ops = []
    for bid, spec in benchmarks.items():
        doc = {"_id": bid, **spec}
        ops.append(UpdateOne({"_id": bid}, {"$set": doc}, upsert=True))

    if ops:
        db[BENCHMARKS_COLL].delete_many({})  # drop existing records
        result = db[BENCHMARKS_COLL].bulk_write(ops)
        logger.info(
            f"Seeded {result.upserted_count} new / {result.modified_count} "
            f"updated benchmarks."
        )


def seed_ground_truths(db, filepath: str = str(os.path.join(BASE_DIR, "inputs", "ground_truth.json"))):
    """Load ground truths into MongoDB (one doc per image_id)."""
    with open(filepath) as f:
        ground_truths = json.load(f)

    ops = []
    for image_id, gt_doc in ground_truths.items():
        doc = {"_id": image_id, "image_id": image_id, **gt_doc}
        ops.append(UpdateOne({"_id": image_id}, {"$set": doc}, upsert=True))

    if ops:
        result = db[GROUND_TRUTHS_COLL].bulk_write(ops)
        logger.info(
            f"Seeded {result.upserted_count} new / {result.modified_count} "
            f"updated ground_truth docs."
        )


# ---------------------------------------------------------------------------
#  Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_single_result(
    task_result: dict,
    benchmarks_cache: dict,
    ground_truths_cache: dict,
) -> dict | None:
    """
    Evaluate one task_result document.
 
    Returns an evaluation document ready for upsert, or None if data is missing.
    """
    benchmark_id = task_result.get("benchmark_id")
    image_id = task_result.get("image_id")
 
    # Validate benchmark first (needed for output parsing below)
    if benchmark_id not in BENCHMARK_IDS:
        return None
 
    benchmark_spec = benchmarks_cache.get(benchmark_id)
    if benchmark_spec is None:
        logger.warning(f"Benchmark {benchmark_id} not found in cache. Skipping.")
        return None
 
    # --- Parse model output ---
    model_output = task_result.get("output", {})
 
    # Try JSON-parsing if stored as a string
    if isinstance(model_output, str):
        try:
            model_output = json.loads(model_output)
        except (json.JSONDecodeError, TypeError):
            pass  # still a raw string — fall through to wrapping below
 
    # Wrap any non-dict output (raw string, int, float, list, etc.)
    if not isinstance(model_output, dict):
        schema_fields = benchmark_spec.get("schema", {}).get("fields", {})
        if len(schema_fields) == 1:
            field_name = list(schema_fields.keys())[0]
            logger.debug(
                f"Task {task_result.get('task_id')}: wrapped {type(model_output).__name__} into '{field_name}'"
            )
            model_output = {field_name: model_output}
        else:
            logger.warning(
                f"Task {task_result.get('task_id')}: output is {type(model_output).__name__} but "
                f"benchmark expects {len(schema_fields)} fields, skipping."
            )
            return None
 
    # --- Ground truth lookup ---
    gt_doc = ground_truths_cache.get(image_id)
    if gt_doc is None:
        logger.warning(f"Ground truth for image {image_id} not found. Skipping.")
        return None
 
    # Run evaluation
    eval_result = evaluate_task(model_output, gt_doc, benchmark_spec)
 
    # Build evaluation document
    eval_doc = {
        "task_id": task_result.get("task_id"),
        "task_result_id": task_result.get("_id"),
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_spec.get("task_name"),
        "model_id": task_result.get("model_id"),
        "model_name": task_result.get("model_name"),
        "image_id": image_id,
        "field_details": eval_result["field_details"],
        "weighted_score": eval_result["weighted_score"],
        "weights_used": eval_result["weights_used"],
        "evaluated_at": datetime.now(timezone.utc),
        "source_updated_at": task_result.get("updated_at"),
    }
 
    return eval_doc


# ---------------------------------------------------------------------------
#  Backfill: evaluate all existing processed results
# ---------------------------------------------------------------------------

def backfill(db):
    """Evaluate all processed task_results for benchmark IDs 4–12."""
    logger.info("Starting backfill...")

    # Load caches
    benchmarks_cache = {
        doc["_id"]: doc for doc in db[BENCHMARKS_COLL].find()
    }
    ground_truths_cache = {
        doc["_id"]: doc for doc in db[GROUND_TRUTHS_COLL].find()
    }

    logger.info(
        f"Loaded {len(benchmarks_cache)} benchmarks, "
        f"{len(ground_truths_cache)} ground truth docs."
    )

    # Query all processed results for our benchmark range
    query = {
        "benchmark_id": {"$in": BENCHMARK_IDS},
        "status": "processed",
    }
    cursor = db[TASK_RESULTS_COLL].find(query)

    ops = []
    evaluated = 0
    skipped = 0

    for task_result in cursor:
        eval_doc = evaluate_single_result(
            task_result, benchmarks_cache, ground_truths_cache
        )
        if eval_doc is None:
            skipped += 1
            continue

        # Upsert keyed on (task_id, benchmark_id, model_id) to be idempotent
        filter_key = {
            "task_id": eval_doc["task_id"],
            "benchmark_id": eval_doc["benchmark_id"],
            "model_id": eval_doc["model_id"],
        }
        ops.append(ReplaceOne(filter_key, eval_doc, upsert=True))
        evaluated += 1

        # Flush in batches of 500
        if len(ops) >= 500:
            db[EVALUATIONS_COLL].bulk_write(ops)
            ops = []
            logger.info(f"  ... flushed {evaluated} evaluations so far")

    # Final flush
    if ops:
        db[EVALUATIONS_COLL].bulk_write(ops)

    logger.info(f"Backfill complete: {evaluated} evaluated, {skipped} skipped.")


# ---------------------------------------------------------------------------
#  Change stream: watch for new task_results in real-time
# ---------------------------------------------------------------------------

def watch(db):
    """
    Watch the task_results collection for new inserts/updates
    and evaluate them incrementally.
    """
    logger.info("Starting change stream watcher...")

    # Load caches (will be refreshed periodically in production)
    benchmarks_cache = {
        doc["_id"]: doc for doc in db[BENCHMARKS_COLL].find()
    }
    ground_truths_cache = {
        doc["_id"]: doc for doc in db[GROUND_TRUTHS_COLL].find()
    }

    # Only watch for inserts and updates to relevant benchmarks
    pipeline = [
        {
            "$match": {
                "$or": [
                    {"operationType": "insert"},
                    {"operationType": "update"},
                    {"operationType": "replace"},
                ],
                "fullDocument.benchmark_id": {"$in": BENCHMARK_IDS},
                "fullDocument.status": "processed",
            }
        }
    ]

    try:
        with db[TASK_RESULTS_COLL].watch(
            pipeline,
            full_document="updateLookup",
        ) as stream:
            logger.info("Change stream opened. Waiting for events...")

            for change in stream:
                task_result = change.get("fullDocument")
                if task_result is None:
                    continue

                op = change.get("operationType")
                task_id = task_result.get("task_id", "?")
                logger.info(f"Change detected [{op}]: task_id={task_id}")

                eval_doc = evaluate_single_result(
                    task_result, benchmarks_cache, ground_truths_cache
                )
                if eval_doc is None:
                    logger.info(f"  Skipped (missing reference data).")
                    continue

                filter_key = {
                    "task_id": eval_doc["task_id"],
                    "benchmark_id": eval_doc["benchmark_id"],
                    "model_id": eval_doc["model_id"],
                }
                db[EVALUATIONS_COLL].replace_one(
                    filter_key, eval_doc, upsert=True
                )
                logger.info(
                    f"  Evaluated: score={eval_doc['weighted_score']:.4f} "
                    f"(benchmark={eval_doc['benchmark_name']}, "
                    f"model={eval_doc['model_name']})"
                )

    except PyMongoError as e:
        logger.error(f"Change stream error: {e}")
        raise


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR Benchmark Evaluation Pipeline"
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Evaluate all existing processed task_results.",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch for new task_results via change stream.",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Seed benchmarks and ground_truths into MongoDB.",
    )
    parser.add_argument(
        "--benchmarks-file", default=str(os.path.join(BASE_DIR, "inputs", "benchmarks.json")),
        help="Path to benchmarks JSON.",
    )
    parser.add_argument(
        "--ground-truths-file", default=str(os.path.join(BASE_DIR, "inputs", "ground_truth.json")),
        help="Path to ground truths JSON.",
    )
    args = parser.parse_args()

    if not any([args.backfill, args.watch, args.seed]):
        parser.print_help()
        sys.exit(1)

    db = get_db()
    ensure_indexes(db)

    if args.seed:
        seed_benchmarks(db, args.benchmarks_file)
        seed_ground_truths(db, args.ground_truths_file)

    if args.backfill:
        backfill(db)

    if args.watch:
        watch(db)


if __name__ == "__main__":
    main()