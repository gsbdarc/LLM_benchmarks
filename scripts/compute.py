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
    # opens all JSON files within a file path
    # if JSON file has an "error" output then it will add output to "investigate" DF
    # otherwise it will add the result to the "llm_results" DF
    # return 2 dataframes

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
    # finds any task id's that were the slurm array but haven't been processed or triggered an error
    # returns updated investigate_df and prints the number of missing tasks

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


def lookup_truth(row):
    # looks up ground truth for a given row based on the benchmark name and
    # image id

    col_name = row['benchmark_name']  # extract benchmark name from df
    img_id = row['image_id']  # extract img id from df

    # find the corresponding row for that image_id in ground_truth_df
    match = ground_truth_df[ground_truth_df['image_id'] == img_id]

    if not match.empty and col_name in ground_truth_df.columns:
        # return the ground_truth_df value that corresponds to the benchmark
        # name
        return match[col_name].values[0]
    return None


def compute_accuracy(row):
    # compares accuracy of llm output to ground_truth based on benchmark_name

    benchmark = row['benchmark_name']
    llm_output = row['output']
    ground_truth = row['ground_truth']

    if benchmark == "day_of_week":
        if llm_output.strip().lower() == ground_truth.strip().lower():
            return 1
        else:
            return 0
    elif benchmark == "newspaper_name":
        cleaned_name = re.sub(r'\s*\(.*\)', '', llm_output)
        if cleaned_name == ground_truth:
            return 1
        else:
            return 0
    elif benchmark == "newspaper_date":
        llm_date = pd.to_datetime(llm_output)
        ground_truth_date = pd.to_datetime(ground_truth)
        if llm_date == ground_truth_date:
            return 1
        else:
            return 0
    elif benchmark == "tv_guide_date":
        try:
            llm_date = pd.to_datetime(llm_output)
            ground_truth_date = pd.to_datetime(ground_truth)
            if llm_date == ground_truth_date:
                return 1
            else:
                return 0
        except BaseException:
            return 0  # assign 0 if unable to be parsed

    elif benchmark == "first_program":
        if llm_output.strip().lower() == ground_truth.strip().lower():
            return 1
        else:
            return 0

    else:
        if llm_output.strip().lower() == ground_truth.strip().lower():
            return 1
        else:
            return 0

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

llm_results_df['ground_truth'] = llm_results_df.apply(lookup_truth, axis=1)

# compute accuracy

llm_results_df['accuracy'] = llm_results_df.apply(compute_accuracy, axis=1)

# save dataframe as json

llm_results_df.to_json(
    f"/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/metrics/{metrics_json}.json",
    orient='records')
