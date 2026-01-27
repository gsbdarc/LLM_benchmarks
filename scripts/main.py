# Goals

# - Load mapping.csv for list of tasks
# - For each row check if the task already exists in processed.csv
# - If the task does not already exist in processed.csv then load inputs (benchmark, model, and image)
# - Process image
# - Save results to results.json in output folder
# - Update processed.csv

# Setup Inputs

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
import os
import requests
from pydantic import BaseModel, ValidationError, create_model
from typing import Literal, List, Dict, Any
import json
import csv
import base64
import matplotlib.pyplot as plt
from pathlib import Path

# Load environment variables from .env file
load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")

STANFORD_API_KEY = os.getenv("STANFORD_API_KEY")

ENDPOINT = "https://aiapi-prod.stanford.edu/v1/chat/completions"

# create dictionary to convert strings to actual python types

type_map = {
    "str" : [str, "string"]
}

# Load JSON's and CSV's

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/image_index.json", "r") as f:
    image_index = json.load(f)

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/benchmarks.json", "r") as f:
    benchmarks = json.load(f)

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/models.json", "r") as f:
    models = json.load(f)

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/mapping.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    mapping = list(reader)

mapping = mapping[0:5] # select first 5 unique tasks ids

# if processed.csv exists load the file, else create one

processed_csv = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/processed.csv" 

if os.path.exists(processed_csv):
    
    with open(processed_csv, "r") as file:
        reader = csv.reader(file)
        processed = list(reader)

else:
    with open(processed_csv, "w", newline="") as file:
        writer = csv.writer(file)
        processed = list()

# if results.json exists load the file, else create one

results_json = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/outputs/results.json"

if os.path.exists(results_json):
    with open(results_json, "r") as f:
        results = json.load(f)
else:
    with open(results_json, "w") as f:
        results = {}
        json.dump(results, f, indent=2)

# Helper Functions

def encode_image(image_path: str) -> str:
    # converts PNG image into b64
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_model(model_name: str, test_b64):

    # ---------------------------------------------
    # Build payload
    # ---------------------------------------------

    if model_family == "gpt":
        # Adjusting detail level for GPT models
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_b64}"
                            }
                            #,"detail": detail_level
                        }
                    ],
                },
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT,
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

    else:
        # All other vision-capable models
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
                                "url": f"data:image/png;base64,{test_b64}"
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
    
    try:
        r = requests.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {STANFORD_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )        
        r.raise_for_status()

        resp_json = r.json()

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

        output_dict["output"] =  llm_output[benchmark_name]
        output_dict["model_name"] = model_name
        output_dict["image_id"] = image_id
        output_dict["benchmark_name"] = benchmark_name
        output_dict["benchmark_id"] = benchmark_id

        if usage:
            output_dict["completion_tokens"] = usage["completion_tokens"]
            output_dict["total_tokens"] = usage["total_tokens"]

        #for field in list(properties.keys()):
            #output[field] = data[field]
            

        #print(json.dumps(data, indent=2))
    
    except Exception as e:
        return resp_json

    return output_dict

# Process Images

for row in mapping:
    task_id = row[0] # extract unique task id from mapping

    if task_id not in processed:
        benchmark_id = row[1]
        benchmark_name = row[2]
        model_id = row[3]
        model_name = row[4]
        image_id = row[5]
        image_path = row[6]

        # load benchmark inputs

        SYSTEM_PROMPT = benchmarks[benchmark_id]['system_prompt']
        USER_PROMPT = benchmarks[benchmark_id]['user_prompt']
        class_name = benchmarks[benchmark_id]['schema']['class_name']
        fields = benchmarks[benchmark_id]['schema']['fields']

        #create pydantic model and other prompt related model inputs

        pydantic_fields = {}

        for field_name, field_type in fields.items():
            python_type = type_map[field_type][0] # converts "str" to str
            pydantic_fields[field_name] = (python_type, ...) # "..." makes the python type mandatory vs optional

        DynamicModel = create_model(
            class_name,
            **pydantic_fields
        )

        properties = {}
        required = []
        
        for field_name, field_type in fields.items():
            json_type = type_map[field_type][1] # converts "str" to "string"
            properties[field_name] = {"type": json_type}
            required.append(field_name)

        # encode image

        b64 = encode_image(image_path)

        # load model inputs

        model_family = models[model_id]["family"]
        
        if "detail" in models[model_id]:
            detail_level = models[model_id]["detail"]

        model_output = run_model(model_name, b64)

        # update results dictionary
        
        results[task_id] = model_output

        # update processed.csv

        with open(processed_csv, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(task_id)

with open(results_json, "w") as f:
    json.dump(results, f, indent=2)
    