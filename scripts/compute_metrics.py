"""
- Load combined_results.json file from outputs/metric folder
- Load ground_truth.csv and join based on image_id and benchmark_name
- Designate if output was correct or not
"""

# Set up

import pandas as pd
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")

BASE_DIR = os.getenv("BASE_DIR")

# Helper Functions


def load_and_filter(combined_results: str) -> pd.DataFrame:
    """
    Loads file and returns DataFrame of tasks where the status is "processed"
    """
    with open(combined_results, "r") as f:
        llm_outputs = json.load(f)

    results_df = pd.DataFrame(llm_outputs)
    filtered_df = results_df[results_df["status"] == "processed"]
    return filtered_df


def lookup_truth(results_df: pd.DataFrame,
                 ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in DataFrame on llm_results and ground truth DataFrame.
    Returns new DataFrame that has original results_df data and a new ground_truth column on benchmark_name and image_id.
    """
    results_copy = results_df.copy(
        deep=True)  # copy results_df so orignal is untouched
    merged_df = results_copy.merge(
        ground_truth_df,
        on='image_id',
        how='left')  # create a merged df

    results_copy['ground_truth'] = merged_df.apply(
        lambda row: row.get(
            row['benchmark_name']),
        axis=1)  # which column ground_truth comes from depends on benchmark_name

    return results_copy


def compare_simple(llm_output: str, ground_truth: str) -> int:
    """
    Compares simple string outputs.
    """
    return int(llm_output.strip().lower() == ground_truth.strip().lower())


def compare_cleaned_name(llm_output: str, ground_truth: str) -> int:
    """
    Compares strings excluding parantheses.
    """
    cleaned = re.sub(r'\s*\(.*\)', '', llm_output)
    return int(cleaned == ground_truth)


def compare_date(llm_output: str, ground_truth: str) -> int:
    """
    Converts inputs to datetime objects and compares them.
    If input can not be converted into a datetime object then return 0.
    """
    try:
        return int(pd.to_datetime(llm_output) == pd.to_datetime(ground_truth))
    except BaseException:
        return 0


def compute_accuracy(row: pd.Series, benchmark_comparisons: dict) -> int:
    """
    Uses benchmark_comparison dictionary to calculate accuarcy based on benchmark_name.
    """
    benchmark = row['benchmark_name']  # identify benchmark of row
    # finds the appropriate comparison function
    compare_fn = benchmark_comparisons.get(benchmark)
    return compare_fn(
        row['output'],
        row['ground_truth'])  # compute the function


def main():

    # set variables

    combined_results = os.path.join(
        BASE_DIR,
        "outputs",
        "metrics",
        "combined_results.json")
    ground_truth = os.path.join(BASE_DIR, "inputs", "ground_truth.json")
    save_path = os.path.join(BASE_DIR, "outputs", "metrics", "metrics.json")

    # load llm_results

    filtered_df = load_and_filter(combined_results)

    # load ground truth

    with open(ground_truth, "r") as f:
        truth = json.load(f)

    ground_truth_df = pd.DataFrame.from_dict(truth, orient='index')

    # add ground truth to llm_results

    merged_df = lookup_truth(filtered_df, ground_truth_df)

    # use dictionary to define how each benchmark metric should be computed

    benchmark_comparisons = {
        "day_of_week": compare_simple,
        "newspaper_name": compare_cleaned_name,
        "newspaper_date": compare_date,
        "tv_guide_date": compare_date,
        "first_program": compare_simple,
        "first_channel": compare_simple
    }

    # compute accuracy

    merged_df['accuracy'] = merged_df.apply(
        lambda row: compute_accuracy(row, benchmark_comparisons), axis=1
    )

    # save dataframe as json

    merged_df.to_json(save_path, orient='records')


if __name__ == "__main__":
    main()
