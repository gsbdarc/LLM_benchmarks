# Goal: create a mapping file that:
# (1) finds all unique combinations of benchmarks, models, and images
# (2) assigns a unique task id to each one
# (3) saves these results into a csv file to be used in main.py

# Setup

from pathlib import Path
import json
import csv
import os

# Load JSON mapping files for images, benchmarks, and models

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/image_index.json", "r") as f:
    image_index = json.load(f)

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/benchmarks.json", "r") as f:
    benchmarks = json.load(f)

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/models.json", "r") as f:
    models = json.load(f)
    model_keys = list(models.keys())
    model_keys = model_keys[1:15]  # take out llama

# Generate all combinations of benchmarks, models, and images

mapping = []

for benchmark_id in benchmarks.keys():
    for model_id in model_keys:
        for image_id in image_index.keys():
            benchmark_name = benchmarks[benchmark_id]["task_name"]
            model_name = models[model_id]["model"]
            image_path = image_index[image_id]['png']
            row = [
                benchmark_id,
                benchmark_name,
                model_id,
                model_name,
                image_id,
                image_path]
            if row not in mapping:
                mapping.append(row)

# Create or append mapping file

filename = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/mapping.csv"

headers = [
    "task_id",
    "benchmark_id",
    "benchmark_name",
    "model_id",
    "model_name",
    "image_id",
    "image_path"]

if os.path.exists(filename):
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

        for i, row in enumerate(mapping, start=1):
            # Check for duplicates within the new data itself
            if row not in mapping[:new_rows.index(row)]:
                full_row = [i] + row
                writer.writerow(full_row)
                print(f"Added with task_id {i}: {row}")
            else:
                print(f"Skipped (duplicate in new data): {row}")

print("Done!")
