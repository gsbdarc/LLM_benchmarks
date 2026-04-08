"""
Goal: create a mapping file that:
(1) finds all unique combinations of benchmarks, models, and images
(2) assigns a unique task id to each one
(3) saves these results into a csv file to be used in main.py
"""

# Setup

from pathlib import Path
from dotenv import load_dotenv
import json
import csv
import os

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[1]

# Load JSON mapping files for images, benchmarks, and models

image_index_path = os.path.join(BASE_DIR, "inputs", "image_index.json")

with open(image_index_path, "r") as f:
    image_index = json.load(f)
    # randomly generated 10 integers from 0 to 34 to sample
    sample_ids = ["0", "3", "7", "11", "14", "19", "22", "26", "28", "31"]
    image_index = {key: value for key, value in image_index.items() if key in sample_ids}

benchmark_path = os.path.join(BASE_DIR, "inputs", "benchmarks.json")

with open(benchmark_path, "r") as f:
    benchmarks = json.load(f)

model_path = os.path.join(BASE_DIR, "inputs", "models.json")

with open(model_path, "r") as f:
    models = json.load(f)
    model_keys = list(models.keys())
    retired_models = {"0", "8", "9"}
    model_keys = [x for x in model_keys if x not in retired_models] # filter out retired models

# Generate all combinations of benchmarks, models, and images

mapping = []

for benchmark_id in benchmarks.keys():
    for model_id in model_keys:
        for image_id in image_index.keys():
            benchmark_name = benchmarks[benchmark_id]["task_name"]
            model_name = models[model_id]["model"]
            image_name = image_index[image_id]['png']
            row = [
                benchmark_id,
                benchmark_name,
                model_id,
                model_name,
                image_id,
                image_name]
            if row not in mapping:
                mapping.append(row)

# Create or append mapping file

filename = os.path.join(BASE_DIR, "inputs", "mapping.csv")

headers = [
    "task_id",
    "benchmark_id",
    "benchmark_name",
    "model_id",
    "model_name",
    "image_id",
    "image_name"]

if os.path.exists(filename) and os.path.getsize(filename) > 0:
    # Read existing data
    with open(filename, "r") as file:
        reader = csv.reader(file)
        all_rows = list(reader)

    # Separate header from data
    existing_header = all_rows[0]
    existing_data = all_rows[1:]

    # Get just the non-task_id parts (columns 1-6) for comparison
    existing_without_id = [row[1:] for row in existing_data]

    # Find the highest task_id so we can continue from there
    if existing_data:
        next_id = max(int(row[0]) for row in existing_data) + 1
    else:
        next_id = 1

    # Append new unique rows
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        for row in mapping:
            if row not in existing_without_id:
                full_row = [next_id] + row
                writer.writerow(full_row)
                print(f"Added with task_id {next_id}: {row}")
                next_id += 1
            else:
                print(f"Skipped (already exists): {row}")

else:
    # Create new file
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        seen = []  # Track what we've already added
        task_id = 0

        for row in mapping:
            if row not in seen:
                full_row = [task_id] + row
                writer.writerow(full_row)
                # print(f"Added with task_id {task_id}: {row}")
                seen.append(row)
                task_id += 1

print("Done!")
