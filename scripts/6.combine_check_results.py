"""Combine all individual JSON task results into one DataFrame
and save as a JSON file for downstream analysis and visualization."""

# Setup

import pandas as pd
import glob
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root/".env")
BASE_DIR = os.getenv("BASE_DIR")

# Helper Functions


def compile_results(results_path: str, output_path: str) -> None:
    """Combine individual JSON task results into a single DataFrame.

    Each JSON file is expected to contain a single key (task_id)
    mapping to a dict of output fields.

    Args:
        results_path: Directory containing individual JSON result files.
        output_path: File path where the combined JSON will be saved.

    Output:
        Saves DataFrame as JSON file.
    """

    results_json = glob.glob(os.path.join(results_path, "*.json"))

    llm_results = []
    for f in results_json:
        with open(f) as file:
            llm_output = json.load(file)
            # Each file has one key (task_id) mapping to one dict (task output).
            # Creates an iterator over task_output grabs the first element
            task_id, task_output = next(iter(llm_output.items()))

            # Build the row: start with task_id, then unpack the output fields
            # alongside it so everything is flat and on the same level
            row = {'task_id': task_id, **task_output}
            llm_results.append(row)

    llm_results_df = pd.DataFrame(llm_results)

    return llm_results_df


def check_results(llm_results_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Returns
    - The count of tasks with errors.
    - Dictionary of unique errors and how frequently they appeared.
    """

    error_count = llm_results_df["error"].notna().sum()
    error_dict = llm_results_df["error"].value_counts().to_dict()

    return error_count, error_dict

# main


def main():
    results_path = os.path.join(BASE_DIR, "outputs", "results")
    output_path = os.path.join(BASE_DIR, "outputs", "metrics", "combined_results.json")
    llm_results_df = compile_results(results_path, output_path)
    error_count, error_dict = check_results(llm_results_df)
    total_results = len(llm_results_df)
    processed_results = total_results - error_count

    print(f"{error_count} tasks had errors, {processed_results} processed succesfully.")
    print("===== Error Breakdown =====")
    for key, value in error_dict.items():
        print("=====Error Message======")
        print(key)
        print("=====Error Count=====")
        print(value)

    llm_results_df.to_json(output_path,
                           orient='records')


if __name__ == "__main__":
    main()
