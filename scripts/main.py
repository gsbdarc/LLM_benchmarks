# Goals

# - Load mapping.csv for list of tasks
# - Check if task has been processed succesfully, exit if so
# - Load inputs (benchmark, model, and image) based on mapping
# - Process task
# - Save results as json file in output folder

# Load Python packages and modules

from dotenv import load_dotenv
from PIL import Image
import os
import requests
from pydantic import BaseModel, ValidationError, create_model
from typing import Literal, List, Dict, Any, Type
import json
import csv
import base64
from pathlib import Path
import sys

# Task selection from slurm array
# task_selection = sys.argv[1]
# cast to int so we can use this to index mapping.csv
# task_selection = int(task_selection)
task_selection = 2087

# Load environment variables from .env file
load_dotenv("../.env")

STANFORD_API_KEY = os.getenv("STANFORD_API_KEY")
BASE_DIR = os.getenv("BASE_DIR")

ENDPOINT = "https://aiapi-prod.stanford.edu/v1/chat/completions"

# create dictionary to convert strings to actual python types

type_map = {
    "str": [str, "string"]
}


# Helper Functions

def already_processed(json_path: str) -> bool:
    """Check if a task's result JSON already has status: processed."""
    try:
        with open(json_path, "r") as f:
            result = json.load(f)
            for task_data in result.values():
                if isinstance(task_data, dict) and task_data.get(
                        "status") == "processed":
                    return True
            return False
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def create_pydantic_model(
        benchmark_id: str) -> tuple[Type[BaseModel], dict, list, str, str]:
    """
    Builds pydantic model and other prompt related model inputs based off of benchmark_id.

    Returns:
        DynamicModel: The constructed Pydantic model class
        properties: JSON schema properties dict
        required: List of required field names
        system_prompt: The system prompt string
        user_prompt: The user prompt string
    """

    benchmark_path = os.path.join(BASE_DIR, "inputs", "benchmarks.json")
    with open(benchmark_path, "r") as f:
        benchmarks = json.load(f)

    SYSTEM_PROMPT = benchmarks[benchmark_id]['system_prompt']
    USER_PROMPT = benchmarks[benchmark_id]['user_prompt']
    class_name = benchmarks[benchmark_id]['schema']['class_name']
    fields = benchmarks[benchmark_id]['schema']['fields']

    pydantic_fields = {}

    for field_name, field_type in fields.items():
        python_type = type_map[field_type][0]  # converts "str" to str
        # "..." makes the python type mandatory vs optional
        pydantic_fields[field_name] = (python_type, ...)

    DynamicModel = create_model(
        class_name,
        **pydantic_fields
    )

    properties = {}
    required = []

    for field_name, field_type in fields.items():
        json_type = type_map[field_type][1]  # converts "str" to "string"
        properties[field_name] = {"type": json_type}
        required.append(field_name)

    return DynamicModel, properties, required, SYSTEM_PROMPT, USER_PROMPT


def encode_image(image_path: str) -> str:
    """ Converts PNG image into b64. """

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_model(model_name: str, b64: str, SYSTEM_PROMPT: str,
              USER_PROMPT: str, properties: dict, required: list,
              benchmark_name: str, benchmark_id: str, model_id: str,
              image_id: str) -> dict:
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

                output_dict["output"] = llm_output[benchmark_name]
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

        return output_dict  # all retries failed, exit loop


def main():
    """Load a task from the mapping, process it through the LLM pipeline, and save results."""

    mapping_path = os.path.join(BASE_DIR, "inputs", "mapping.csv")

    with open(mapping_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        mapping = list(reader)

    try:
        selected_task = mapping[task_selection]

    except BaseException:
        sys.exit(0)  # if the task_id is not in mapping then exit the program

    task_id = selected_task[0]  # extract unique task id from mapping

    results_json = os.path.join(
        BASE_DIR,
        "outputs",
        "results",
        f"results_{task_id}.json")

    # check if we need to process this task

    if already_processed(results_json):
        sys.exit(0)

    # load inputs from mapping

    benchmark_id = selected_task[1]
    benchmark_name = selected_task[2]
    model_id = selected_task[3]
    model_name = selected_task[4]
    image_id = selected_task[5]
    image_name = selected_task[6]

    image_path = os.path.join(BASE_DIR, "inputs", "data", "pngs", image_name)

    # built prompt based model inputs

    DynamicModel, properties, required, system_prompt, user_prompt = create_pydantic_model(
        benchmark_id)

    # encode image

    b64 = encode_image(image_path)

    model_output = run_model(model_name, b64, system_prompt,
                             user_prompt, properties, required,
                             benchmark_name, benchmark_id, model_id,
                             image_id)

    # assign LLM results to a dictionary

    results = dict()
    results[task_id] = model_output
    # print(results)

    # save as a JSON file

    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
