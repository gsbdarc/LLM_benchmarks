# Standard library imports
import base64
import json
import logging
import sys
import os
import re
import requests
from datetime import datetime
from io import BytesIO
from IPython.display import display, Markdown, JSON
from multiprocessing import Pool
from typing import Type

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

# Local application imports
script_dir = os.path.abspath('/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/scripts')
sys.path.append(script_dir)
import helper

#Inputs

model = "llama3.2-vision:11b"
data_path='/zfs/projects/students/ltdarc-usf-intern-2025/data'

example_1_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Austin_American_Statesman_Sun__Aug_3__2014_ (10).pdf"
example_b64_1 = helper.pdf_to_b64(example_1_path)

#Prompts

system_prompt = (
    "You are a precise metadata extraction assistant. "
    "Your job is to identify and extract structured information from scanned newspaper TV guide images. "
    "You must always return the result as a valid JSON object that exactly matches the schema: "
    '{"newspaper": "<name>", "date": "<Month Day, Year>"}'
)

user_prompt = f"""
Here is the image, base64-encoded:

{example_b64_1}

Please determine the name of the newspaper and the date the TV guide is for.


"""

#Data Structures

output_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "newspaper": {"type": "string"},
        "date": {"type": "string"}
    },
    "required": ["newspaper", "date"]
}

# General LLM Functions

def build_llama_prompt(messages):
    """Builds a Llama prompt using the required template."""
    out = []
    n = len(messages)

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        out.append(f"<|start_header_id|>{role}<|end_header_id|>\n")
        out.append(content)
        out.append("\n")

        # If this is not the last message → append <|eot_id|>
        if i < n - 1:
            out.append("<|eot_id|>\n")
        # If this is the last message AND role != assistant → append assistant header
        elif role != "assistant":
            out.append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

    return "".join(out)

def call_llm(system_prompt, user_prompt, output_schema):
    LLM_API_URL = f"{ollama_url}/api/chat"

    shste
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
        "format": output_schema,
        "stream": False
    }
        
    response = requests.post(LLM_API_URL, headers=headers, data=json.dumps(payload))
    result = response.json()
    content_dict = json.loads(result["message"]["content"])
    
    #print(content_dict["newspaper"])
    #print(content_dict["date"])
    return content_dict
    
def get_date_csv(path: str) -> str:
    """Builds date object based on year, month, and date columns in csv."""
    truth_df = pd.read_csv(path)
    
    true_year = str(truth_df.loc[0, 'Year']).strip()
    true_month = str(truth_df.loc[0, 'Month']).strip()
    true_date_raw = str(truth_df.loc[0, 'Date']).strip()

    match = re.search(r"\d+", true_date_raw)
    true_day = int(match.group())
    date_str = f"{true_month} {true_day}, {int(true_year)}"
    date_obj = datetime.strptime(date_str, "%B %d, %Y")

    return date_obj

def process_file(file, system_prompt, user_prompt):
    b64 = pdf_to_b64(file)
    llm_meta = call_llm(user_prompt)
        
    date_str = llm_meta["date"]
    if date_str:
        date_obj = convert_string_to_date(date_str)
    else:
        date_obj = date_str

    return {
        "LLM_Newspaper_Name": llm_meta["newspaper"],
        "LLM_Newspaper_Date": date_obj
    }

def compare_results(results_df, index):

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

def main()
    
    try:
        r = requests.get(ollama_url)
        r.raise_for_status()
        print("✅ Success:", r.text)
    except requests.exceptions.RequestException as e:
        print("❌ Failed to connect:", e)

    output_prompt = build_llama_prompt(output_prompt)

    files = list(index['pdf_files'])
    
    with Pool(34) as p:
        results = p.map(process_file, files)
    
    for i, r in enumerate(results):
        r["Index"] = i

    results_df = pd.DataFrame(results)

    comparison_df = compare_results(results_df, index)

    name_accuracy = helper.calc_accuracy(comparison_df, "Name_Match")

    date_accuracy = helper.calc_accuracy(comparison_df, "Date_Match")

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
    logger.info(f"Role Prompt:\n{system_prompt}")
    logger.info(f"Content Prompt:\n{user_prompt}")

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
