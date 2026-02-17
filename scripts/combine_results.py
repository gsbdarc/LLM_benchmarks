"""Combine all individual JSON task results into one DataFrame
and save as a JSON file for downstream analysis and visualization."""

# Setup

import pandas as pd
import glob
import json
import os

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

    llm_results_df.to_json(output_path,
                           orient='records')

# main


def main():
    results_path = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/results/"
    output_path = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/metrics/combined_results.json"
    compile_results(results_path, output_path)


if __name__ == "__main__":
    main()
