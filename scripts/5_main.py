# Goals

# - Load mapping.csv for list of tasks
# - Check if task has been processed succesfully in MongoDB, exit if so
# - Load inputs (benchmark, model, and image) based on mapping
# - Process task
# - Write results to MongoDB

# Load Python packages and modules

from dotenv import load_dotenv
from PIL import Image
import os
import requests
from pydantic import BaseModel, ValidationError, create_model
from typing import Literal, List, Dict, Any, Type
import json
import pandas as pd
import csv
import base64
from pathlib import Path
import sys
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# Task selection from slurm array
#task_selection = sys.argv[1]
# cast to int so we can use this to index mapping.csv
#task_selection = int(task_selection)
task_selection = 3930

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# Stanford AI Gateway
STANFORD_API_KEY = os.getenv("STANFORD_API_KEY")
ENDPOINT = "https://aiapi-prod.stanford.edu/v1/chat/completions"

# MongoDB
username = os.getenv("MONGO_DB_USERNAME")
password = os.getenv("MONGO_DB_PASSWORD")

hosts = [
    'darc-data-shard-00-00.9fjam.mongodb.net:27017',
    'darc-data-shard-00-01.9fjam.mongodb.net:27017',
    'darc-data-shard-00-02.9fjam.mongodb.net:27017'
]
setName = 'DARC-Data-shard-0'
uri = (
    f"mongodb://{username}:{password}@{','.join(hosts)}/"
    f"?tls=true&replicaSet={setName}&authSource=admin&retryWrites=true&w=majority&appName=DARC-Data"
)
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["usf-internship"]
collection = db["llm_outputs"]

# Helper Functions

def check_task_mongo(collection, task_id: str, run_id: int) -> bool:
    """Check if tasks already exists in Mongo. If the task does exist, check if the status key has a value of processed."""
    result = collection.find_one(
        {"_id": f"{task_id}_{run_id}_processed"},
    )
    if result is None:
        return False
    else:
        return True


def create_pydantic_model(
        benchmark_id: str) -> tuple[dict, list, str, str]:
    """
    Builds pydantic and other prompt related model inputs based off of benchmark_id.

    Returns:
        properties: JSON schema properties dict
        required: List of required field names
        system_prompt: The system prompt string
        user_prompt: The user prompt string
    """

    benchmark_path = os.path.join(BASE_DIR, "inputs", "benchmarks.json")
    with open(benchmark_path, "r") as f:
        benchmarks = json.load(f)

    benchmark_name = benchmarks[benchmark_id]['task_name']
    SYSTEM_PROMPT = benchmarks[benchmark_id]['system_prompt']
    USER_PROMPT = benchmarks[benchmark_id]['user_prompt']
    properties = benchmarks[benchmark_id]['schema']['fields']
    required = list(properties.keys())

    return properties, required, SYSTEM_PROMPT, USER_PROMPT


def encode_image(image_path: str) -> str:
    """ Converts PNG image into b64. """

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_model(model_name: str, b64: str, SYSTEM_PROMPT: str,
              USER_PROMPT: str, properties: dict, required: list,
              benchmark_name: str, benchmark_id: str, model_id: str,
              image_id: str, run_number: int) -> dict:
    """
    Builds payload and sends request to Stanford AI API.
    Parses output and returns a dictionary.
    """

    # ---------------------------------------------
    # Build payload
    # ---------------------------------------------

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "DynamicModel",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
    }

    # ---------------------------------------------
    # Send request
    # ---------------------------------------------

    output_dict = dict()
    status = "unprocessed"

    max_retries = 3

    for attempt in range(max_retries):

        try:
            r = requests.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {STANFORD_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=600,
            )

            resp_json = r.json()

            try:

                # -----------------------------------------
                # Extract usage (context accounting)
                # -----------------------------------------

                usage = resp_json.get("usage")

                message = resp_json["choices"][0]["message"]
                content = message.get("content")

                if not content or not isinstance(content, str):
                    raise ValueError("Empty or non-text content returned")

                # -----------------------------------------
                # Parse output
                # -----------------------------------------
                cleaned = (
                    content
                    .strip()
                    .removeprefix("```json")
                    .removesuffix("```")
                    .strip()
                )
                llm_output = json.loads(cleaned)

                # -----------------------------------------
                # Attach metadata
                # -----------------------------------------

                output_dict["output"] = llm_output
                output_dict["model_name"] = model_name
                output_dict["model_id"] = model_id
                output_dict["image_id"] = image_id
                output_dict["benchmark_name"] = benchmark_name
                output_dict["benchmark_id"] = benchmark_id

                if usage:
                    output_dict["completion_tokens"] = usage["completion_tokens"]
                    output_dict["total_tokens"] = usage["total_tokens"]

                output_dict["status"] = "processed"

                return output_dict  # if success exit loop and return output_dict

            except Exception as e:
                try:
                    # try to return more detailed message if possible
                    error_msg = resp_json['error']['message']
                    output_dict["error"] = error_msg
                    output_dict["status"] = status
                    return output_dict

                except BaseException:
                    output_dict["error"] = str(e)
                    output_dict["status"] = status
                    return output_dict

        except Exception as e:
            if attempt < max_retries - 1:
                continue  # try again
            else:
                output_dict["error"] = str(e)
                output_dict["status"] = status

        output_dict["run_id"] = run_number

        return output_dict  # all retries failed, exit loop


def write_results_mongo(collection, task_id: str, result: dict, run_id: int) -> None:
    """ Writes a single result ot MongoDB, replaces existing record if it already exists."""
    status = result["status"]
    doc = {
        "_id": f"{task_id}_{run_id}_{status}",
        "task_id": task_id,
        **result,
        "updated_at": datetime.now(),
    }
    collection.replace_one(
        {"_id": doc["_id"]},
        doc,
        upsert=True
    )


def main():
    """Load a task from the mapping, process it through the LLM pipeline, and save results."""

    run_number = 1  # future iteration: add this as a slurm script variable, have a default

    mapping_path = os.path.join(BASE_DIR, "inputs", "mapping.csv")

    mapping = pd.read_csv(mapping_path)
    # mapping = mapping[~mapping['model_name'].isin(['claude-3-5-sonnet',
    # 'claude-3-7-sonnet'])] # filter out retired models
    # mapping = mapping[mapping['model_name'].isin(
    # ['claude-4-5-sonnet', 'claude-opus-4-6', 'gpt-5.2', 'Llama-4'])]  # filter for new models

    selected_task = mapping[mapping['task_id'] == task_selection]

    if selected_task.empty:
        sys.exit(0)  # if the task_id is not in mapping then exit the program

    results_json = os.path.join(
        BASE_DIR,
        "outputs",
        "results",
        f"results_{task_selection}_{run_number}.json")

    # check if we need to process this task

    if check_task_mongo(collection, str(task_selection), run_number):
        print("Task has already been processed.")
        sys.exit(0)

    # load inputs from mapping

    benchmark_id = str(selected_task['benchmark_id'].iloc[0])
    benchmark_name = str(selected_task['benchmark_name'].iloc[0])
    model_id = str(selected_task['model_id'].iloc[0])
    model_name = str(selected_task['model_name'].iloc[0])
    image_id = str(selected_task['image_id'].iloc[0])
    image_name = str(selected_task['image_name'].iloc[0])

    image_path = os.path.join(BASE_DIR, "inputs", "data", "pngs", image_name)

    # built prompt based model inputs

    properties, required, system_prompt, user_prompt = create_pydantic_model(
        benchmark_id)

    # encode image

    b64 = encode_image(image_path)

    model_output = run_model(model_name, b64, system_prompt,
                             user_prompt, properties, required,
                             benchmark_name, benchmark_id, model_id,
                             image_id, run_number)

    # write LLM results to MongoDB

    write_results_mongo(collection, str(task_selection), model_output, run_number)


if __name__ == "__main__":
    main()
