# Objectives

# - Load results.json file from outputs folder
# - Determine if all tasks were completed successfully
# - For unsuccesful/incomplete tasks:
# - Add to "investigation" db and save
# - For succesful tasks:
# - Load ground_truth.csv and join based on image_id and benchmark_name
# - Designate if output was correct or not

# Set up

import pandas as pd
import os
import json
import csv
import re

# Key Inputs

# update with the same mapping file used in main.py
mapping_path = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/mapping_shuffled.csv"

# update based on the array used in the slurm script
array_size = 1000

# what name should the investigate_df be saved under? change each time you
# run this script to avoid overriding
investigate_json = "investigate_2"

# what name should the investigate_df be saved under? change each time you
# run this script to avoid overriding
metrics_json = "metrics_2"

# Functions


def sort_outputs(s):
    """
    Opens all JSON files within a file path.
    If JSON file has an "error" output then it will add output to "investigate" DF.
    Otherwise it will add the result to the "llm_results" DF.
    Returns 2 dataframes.
    """

    investigate = []
    llm_results = []

    for filename in os.listdir(s):
        if filename.endswith(".json"):
            with open(os.path.join(s, filename)) as f:
                llm_output = json.load(f)
                # extract tasks outputs as a list
                llm_output_values = list(llm_output.values())
                llm_output_fields = list(
                    llm_output_values[0].keys())  # get a list of keys
                if 'error' in llm_output_fields:  # if there was an error processing the task
                    key = llm_output.keys()
                    task_id = list(key)[0]  # extract task_id from keys
                    row = {'task_id': task_id}
                    value = llm_output.values()
                    task_output = list(value)[0]  # extract output from values
                    # now task_id and output are on the same level
                    row.update(task_output)
                    investigate.append(row)  # add row to "investigate" df
                else:
                    key = llm_output.keys()
                    task_id = list(key)[0]  # extract task_id from keys
                    row = {'task_id': task_id}
                    value = llm_output.values()
                    task_output = list(value)[0]  # extract output from values
                    # now task_id and output are on the same level
                    row.update(task_output)
                    llm_results.append(row)  # add row to "llm_results" df

    investigate_df = pd.DataFrame(investigate)
    llm_results_df = pd.DataFrame(llm_results)

    return investigate_df, llm_results_df


def find_missing_tasks(
        llm_results_df,
        investigate_df,
        array_size: int,
        mapping_path: str):
    """
    Finds any task id's that were the slurm array but haven't been processed or triggered an error.
    Returns updated investigate_df and prints the number of missing tasks.
    """

    # list of all task id's that were processed
    processed_tasks = list(llm_results_df['task_id'])
    # list of all task id's that had an error
    error_tasks = list(investigate_df['task_id'])

    with open(mapping_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        n = array_size + 1
        mapping = list(reader)[0:n]  # get a list of tasks in the slurm array
        missing_tasks = 0

        for row in mapping:
            task = row[0]
            # for each task see if it exists in either list
            if (task not in processed_tasks and task not in error_tasks):
                # if not, add it to investigate df
                entry = pd.DataFrame(
                    {'task_id': [task], 'error': ['task did not process']})
                full_investigate_df = pd.concat(
                    [investigate_df, entry], ignore_index=True)
                missing_tasks += 1
            else:
                continue

    return full_investigate_df


def lookup_truth(results_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in DataFrame on llm_results and ground truth DataFrame.
    Returns new DataFrame that has original results_df data and a new ground_truth column on benchmark_name and image_id.
    """
    results_copy = results_df.copy(deep = True) # copy results_df so orignal is untouched
    merged_df = results_copy.merge(ground_truth_df, on='image_id', how = 'left') # create a merged df

    results_copy['ground_truth'] = merged_df.apply(lambda row: row.get(row['benchmark_name']), axis = 1) # which column ground_truth comes from depends on benchmark_name

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
    except:
        return 0

# use dictionary to define how each benchmark metric should be computed

benchmark_comparisons = {
    "day_of_week": compare_simple,
    "newspaper_name": compare_cleaned_name,
    "newspaper_date": compare_date,
    "tv_guide_date": compare_date,
    "first_program": compare_simple,
    "first_channel": compare_simple
}

def compute_accuracy(row: pd.Series) -> int:
    """
    Uses benchmark_comparison dictionary to calculate accuarcy based on benchmark_name.
    """
    benchmark = row['benchmark_name'] # identify benchmark of row
    compare_fn = benchmark_comparisons.get(benchmark) # finds the appropriate comparison function
    return compare_fn(row['output'], row['ground_truth']) # compute the function

# load llm results


folder = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/results"

# sort results

investigate_df, llm_results_df = sort_outputs(folder)

full_investigate_df = find_missing_tasks(
    llm_results_df, investigate_df, array_size, mapping_path)

# save investigate_db

full_investigate_df.to_json(
    f"/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/metrics/{investigate_json}.json",
    orient='records')

# load ground truth

ground_truth = list()

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/ground_truth.json", "r") as f:
    truth = json.load(f)
    image_index = list(truth.keys())
    for image in image_index:
        row = {'image_id': image}
        truths = truth[image]
        row.update(truths)
        ground_truth.append(row)

ground_truth_df = pd.DataFrame(ground_truth)

# add ground truth to llm_results

merged_df = lookup_truth(llm_results_df, ground_truth_df)

# compute accuracy

merged_df['accuracy'] = merged_df.apply(compute_accuracy, axis = 1)

# save dataframe as json

llm_results_df.to_json(
    f"/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/metrics/{metrics_json}.json",
    orient='records')
