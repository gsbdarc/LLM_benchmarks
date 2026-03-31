"""
- Load combined_results.json file from outputs/metric folder
- Load ground_truth.csv and join based on image_id and benchmark_name
- Calculate dollar costs for input tokens, output tokens, and total tokens used
- Designate if output was correct or not
"""

# Set up

import pandas as pd
import json
import re
import os
from dotenv import load_dotenv
from pathlib import Path

# Get project_root
project_root = Path(__file__).resolve().parents[1]

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


def calculate_dollar_costs(
        filtered_df: pd.DataFrame,
        benchmarks: str) -> pd.DataFrame:
    """
    Converts token usage into input, output, and total dollar costs.
    """
    with open(benchmarks, "r") as f:
        model_config = json.load(f)

    model_config_df = pd.DataFrame(model_config)
    config = model_config_df.T

    df = filtered_df.merge(config[['model', 'input_cost', 'output_cost']].rename(
        columns={'model': 'model_name'}), on='model_name', how='left')

    df['input_tokens'] = df['total_tokens'] - df['completion_tokens']
    df['input_dollar_cost'] = (df['input_tokens'] * df['input_cost'] / 1000000)
    df['output_dollar_cost'] = (
        df['completion_tokens'] *
        df['output_cost'] /
        1000000)
    df['total_dollar_cost'] = (
        (df['input_tokens'] * df['input_cost'] / 1000000)
        + (df['completion_tokens'] * df['output_cost'] / 1000000))

    return df


def load_ground_truth(image_path: str) -> pd.DataFrame:
    """
    Loads ground truth JSON file, coverts to DF and adds column for image index.
    """

    ground_truth = list()

    with open(image_path, "r") as f:
        truth = json.load(f)
        image_index = list(truth.keys())
        for image in image_index:
            row = {'image_id': image}
            truths = truth[image]
            row.update(truths)
            ground_truth.append(row)

    ground_truth_df = pd.DataFrame(ground_truth)
    return ground_truth_df


def lookup_truth(results_df: pd.DataFrame,
                 ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in DataFrame on llm_results and ground truth DataFrame.
    Returns new DataFrame that has original results_df data and a new ground_truth column on benchmark_name and image_id.
    """
    results_copy = results_df.copy(deep=True).reset_index(drop=True)
    merged_df = results_copy.merge(
        ground_truth_df,
        on='image_id',
        how='left')  # create a merged df

    results_copy['ground_truth'] = merged_df.apply(
        lambda row: row.get(
            row['benchmark_name']),
        axis=1)  # which column ground_truth comes from depends on benchmark_name

    columns_to_drop = ['status', 'error']
    results_copy = results_copy.drop(columns=columns_to_drop)

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
    cleaned = cleaned.replace('-', ' ')
    ground_truth = ground_truth.replace('-', ' ')
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
        project_root,
        "outputs",
        "metrics",
        "combined_results.json")
    ground_truth = os.path.join(project_root, "inputs", "ground_truth.json")
    save_path = os.path.join(
        project_root,
        "outputs",
        "metrics",
        "metrics.json")
    model_config = os.path.join(project_root, "inputs", "models.json")

    # load llm_results

    filtered_df = load_and_filter(combined_results)

    # calculate costs

    filtered_df = calculate_dollar_costs(filtered_df, model_config)

    # load ground truth

    ground_truth_df = load_ground_truth(ground_truth)

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
