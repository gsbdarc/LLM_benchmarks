# Standard library imports
import base64
import json
import logging
import sys
import time
import os
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from typing import Type

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

# Local application imports
script_dir = os.path.abspath(
    '/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/scripts')
sys.path.append(script_dir)

import helper

# Configuring inputs, adjust model and examples as neede

data_path = '/zfs/projects/students/ltdarc-usf-intern-2025/data'
load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = "gpt-5"

example_1_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Austin_American_Statesman_Sun__Aug_3__2014_ (10).pdf"
example_2_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Chicago_Tribune_Sun__May_28__1995_ (30).pdf"

example_b64_1 = helper.pdf_to_b64(example_1_path)
example_b64_2 = helper.pdf_to_b64(example_2_path)

# Data Models

class meta_data(BaseModel):
    newspaper_name: str
    publication_date: str


# Prompts

role_prompt = (
    "You are a precise metadata extraction assistant. "
    "Your job is to identify and extract structured information from scanned newspaper TV guide images. "
    "You must always return the result as a valid JSON object that exactly matches the schema: "
    '{"newspaper": "<name>", "date": "<Month Day, Year>"}')

content_prompt = """
You are analyzing a scanned newspaper page that includes a TV guide.

Your task is to extract exactly two fields:

1. `"newspaper"` — the full name of the newspaper where the TV guide appears.
2. `"date"` — the date the TV guide is for, formatted as "Month Day, Year" (e.g., "January 5, 2023").

Follow the rules below exactly and output ONLY the JSON for that new image.

1. Exact Date Found
If the page shows a complete date (e.g., "August 9, 2014"), use it directly.

2. Day-of-Week Mismatch Rule
If the page shows a full date with a day-of-week (e.g., "Sun, Dec 17, 2000")
and elsewhere shows a different day-of-week (e.g., "Wednesday"),
assume the guide is for that other day within the same calendar week.

Example:
- Visible: "Sun, Dec 17, 2000"
- Elsewhere: "Wednesday"
→ Correct guide date: "December 20, 2000"

3. Date Range Rule
If the page shows a date range (e.g., "May 28–June 3, 1995"),
assume the guide is for the last date in that range.

Example:
- Visible: "May 28–June 3, 1995"
→ Correct guide date: June 3, 1995"

4. Final Output Format
Always return the date in the format: "Month Day, Year".

Below are a few labeled examples for guidance.
Study them carefully before analyzing the final image.

---

### Example 1
Image:
(Shown below)
Expected Output:
{
  "newspaper": "Austin American-Statesman",
  "date": "August 9, 2014"
}

---

### Example 2
Image:
(Shown below)
Expected Output:
{
  "newspaper": "The Chicago Tribune",
  "date": "June 3, 1995"
}

---

Now analyze the next image and output only valid JSON with the same keys:
{
  "newspaper": "...",
  "date": "..."
}
"""


# Core LLM Functions

def call_llm(
        b64: str,
        model: str,
        role_prompt: str,
        content_prompt: str,
        structured_output: Type[BaseModel]) -> BaseModel:
    """
    Inputs:
    - b64: Base64 image for LLM to extact metadata
    - model: Which model of chatgpt sould be used
    - role prompt: What role the model should take on
    - content prompt: What shoud the model do with the input
    - structured output: What should the model do with the input

    Function will feed inputs as well as a two few shot examples into the LLM.

    Outputs:
    - response.output_parsed: structured output in json format
    - elapsed: time it took to call LLM and recieve the structured output
    - input_tokens: number of tokens the inputs translated to
    - output_tokens: number of tokens the output translated to
    - tps_output: rate of output tokens per second.
    
    """

    start = time.time()

    response = client.responses.parse(
        model=model,
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": role_prompt}]},

            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Example 1: Extract metadata from this page."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{example_b64_1}"},
                    {"type": "input_text", "text": '{"newspaper": "Austin American-Statesman", "date": "August 9, 2014"}'}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Example 2: Extract metadata from this page."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{example_b64_2}"},
                    {"type": "input_text", "text": '{"newspaper": "The Chicago Tribune", "date": "May 28, 1995"}'}
                ]
            },

            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": content_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    },
                ],
            },
        ],
        text_format=structured_output,
    )

    end = time.time()
    elapsed = end - start

    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens

    tps_output = output_tokens / elapsed if elapsed > 0 else 0

    return response.output_parsed, elapsed, input_tokens, output_tokens, tps_output


def process_file(file):

    """
    Converts input file into Base64
    Feeds this and other inputs definied earlier in script into ChatGPT
    Extracts structure output and well as time/token metrics
    returns a dictionary of these outputs
    """
    
    b64 = helper.pdf_to_b64(file)
    llm_meta, elapsed, input_tokens, output_tokens, tps_output = call_llm(b64, model, role_prompt, content_prompt, meta_data)
    
    date_str = llm_meta.publication_date
    if date_str:
        date_obj = helper.convert_string_to_date(date_str)
    else:
        date_obj = date_str

    return {
        "LLM_Newspaper_Name": llm_meta.newspaper_name,
        "LLM_Newspaper_Date": date_obj,
        "Time_Taken": elapsed,
        "Input_Tokens": input_tokens,
        "Output_Tokens": output_tokens,
        "Tokens_Per_Second": tps_output
    }

# Evaluation Functions
    
def compare_results(results_df, index):

    """
    Inputs
    - results_df: DataFrame of outputs from process_file
    - index: DataFrame of PDF's and CSV source of truth

    For each row in the results_df the script find the corresponding row in the index df
    This points to the location of CSV source of truth file for a given pdf 
    It extracts the newspaper name and tv guide date from ths CSV file
    These values are then compared to the structured output from the LLM

    - name_match: does the name the LLM extracted equal the name in the CSV
    - date_match: does the date the LLM extracted equal the date in the CSV


    Output:
    - Data Frame with the following columns:
        - Index
        - LLM Newspaper Name
        - LLM Newspaper Date
        - Actual Name (from CSV)
        - Actual Date (from CSV)
        - Time taken (for single call of LLM)
        - Input tokens
        - Output token 
        - Tokens/second
    """

    comparison_results = list()

    for _, row in results_df.iterrows():
        ground_truth_path = index.loc[row["Index"], "ground_truth"]
        true_name = helper.extract_newspaper_name(ground_truth_path)
        true_date = helper.get_date_csv(ground_truth_path)
    
        name_match = helper.normalize_name(row["LLM_Newspaper_Name"]) == helper.normalize_name(true_name)
        date_match = row["LLM_Newspaper_Date"] == true_date
    
        comparison_results.append({
            "Index": row["Index"],
            "LLM_Newspaper_Name": row["LLM_Newspaper_Name"],
            "LLM_Newspaper_Date": row["LLM_Newspaper_Date"],
            "Actual_Name": true_name,
            "Actual_Date": true_date,
            "Name_Match": name_match,
            "Date_Match": date_match,
            "Time_Taken": row["Time_Taken"],
            "Input_Tokens": row["Input_Tokens"],
            "Output_Tokens": row["Output_Tokens"],
            "Tokens_Per_Second": row["Tokens_Per_Second"]
        })

    comparison_df = pd.DataFrame(comparison_results)

    return comparison_df

# main/orchestration

def main():

    """
    Creates "index" data frame where each row has the locations of a PDF and it's CSV source of truth file
    Make a list of only pdf locations called "files"
    Using parallel processing feed files into process_files function (calls LLM)
    Append each result to a list called "results"
    Add an index and convert to a dataframe
    Using the compare_results function match LLM outputs to source of truth
    calc_accuracy looks at the proportion of "True" for each metric
    """
    index = helper.make_index(data_path)

    files = list(index['pdf_files'])

    with Pool(25) as p: #parallel processing across 25 threads
        results = p.map(process_file, files)

    # Add index after the fact
    for i, r in enumerate(results):
        r["Index"] = i

    results_df = pd.DataFrame(results)

    comparison_df = compare_results(results_df, index)

    name_accuracy = helper.calc_accuracy(comparison_df, "Name_Match")

    date_accuracy = helper.calc_accuracy(comparison_df, "Date_Match")

    """
    Once accuracy has been evaluated we need to log inputs and results
    Creates a "logs" folder in current scripts folder
    Each time the script is called the following is logged:
    - model type
    - role prompt
    - content prompt
    - reasoning effort
    - dataframe of results
    - name accuracy rate
    - date accuracy rate
    """

    os.makedirs("logs", exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_path = f"logs/llm_call_{today}.log"

    logger = logging.getLogger("llm_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=== LLM CALL ===")
    logger.info(f"Model: {model}")
    logger.info(f"Role Prompt:\n{role_prompt}")
    logger.info(f"Content Prompt:\n{content_prompt}")
    logger.info(f"Reasoning Effort:\n{'high'}")

    logger.info("=== Dataframe ===")
    logger.info("\n" + comparison_df.to_markdown())

    logger.info("=== Name Accuracy ===")
    logger.info(name_accuracy)

    logger.info("=== Date Accuracy ===")
    logger.info(date_accuracy)


if __name__ == "__main__":
    print("Starting program")
    main()
    print("Finished program, please check logs for results")
