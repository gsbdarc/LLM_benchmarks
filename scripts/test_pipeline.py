## Goals

# - Test feasability of project pipeline using a 1 prompt, 1 Stanford API LLM, and 10 sample PNGs
# - Load models & prompts from models.json + benchmarks.json
# - Load PNGs from data folder
# - Pass inputs into model within Stanford playground
# - Record outputs as a .json file

# Input Setup

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
import os
import requests
from pydantic import BaseModel, ValidationError, create_model
from typing import Literal, List, Dict, Any
import json
import base64
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# model & task selection
model_id = sys.argv[1][0]
task_id = sys.argv[1][1]

# cast to string to match models.json & benchmarks.json
model_id = str(model_id)
task_id = str(task_id)

# load models.json
with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/models.json", "r") as f:
    models = json.load(f)

# assign model specific variables
model_name = models[model_id]["model"]
model_family = models[model_id]["family"]
if "detail" in models[model_id]:
    detail_level = models[model_id]["detail"]


# Load environment variables from .env file
load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")

STANFORD_API_KEY = os.getenv("STANFORD_API_KEY")

# Benchmark selection

task = "newspaper_name"

with open("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/benchmarks.json", "r") as f:
    benchmarks = json.load(f) #rename (dont just use data)

SYSTEM_PROMPT = benchmarks[task_id]['system_prompt']
USER_PROMPT = benchmarks[task_id]['user_prompt']
class_name = benchmarks[task_id]['schema']['class_name']
fields = benchmarks[task_id]['schema']['fields']
task_name = benchmarks[task_id]["task_name"]

if "aliases" in benchmarks[task_id]:
    aliases = benchmarks[task_id]["aliases"]

# create mapping to convert strings to actual python types

type_map = {
    "str" : [str, "string"]
}

# dynamically build pydantic model

pydantic_fields = {}

for field_name, field_type in fields.items():
    python_type = type_map[field_type][0] # converts "str" to str
    pydantic_fields[field_name] = (python_type, ...) # "..." makes the python type mandatory vs optional

DynamicModel = create_model(
    class_name,
    **pydantic_fields
)

# dynamically build "properties" and "required fields" for payload

properties = {}
required = []

for field_name, field_type in fields.items():
    json_type = type_map[field_type][1] # converts "str" to "string"
    properties[field_name] = {"type": json_type}
    required.append(field_name)

# image setup

def encode_image(image_path: str) -> str:
    # converts PNG image into b64
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_dir = Path("/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pngs")

# create iterable list of PNGs

image_paths = sorted(
    str(p) for p in image_dir.iterdir()
    if p.suffix.lower() == ".png"
)

# use first ten images only

test_images = image_paths[0:11]

# pass inputs into playground

ENDPOINT = "https://aiapi-prod.stanford.edu/v1/chat/completions"

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
                            },
                            "detail": detail_level
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
        }
    else:
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

    results = dict()

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

        # clean content so json is able to load without error
        
        cleaned = (
            content
            .strip()
            .removeprefix("```json")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)

        # -----------------------------------------
        # Attach metadata
        # -----------------------------------------

        first_key = list(data.keys())[0]
        
        
        if first_key in aliases:
            results[task_name] =  data[first_key]
        else:
            return (f"add {first_key} to aliases")

        if usage:
            results["completion_tokens"] = usage["completion_tokens"]
            results["total_tokens"] = usage["total_tokens"]

        results["model"] = model_name
        results["task_id"] = task_id

    except Exception as e:
        results["error"] = str(e)

    return results

json_results = dict()

for img_path in test_images:
    b64 = encode_image(img_path)
    output = run_model(model_name, b64)
    json_results[img_path] = output

save_path = f"/zfs/projects/students/ltdarc-usf-intern-2025/dev/development_notebooks/test_pipeline/results/test_pipeline_{task_id}_{task_name}_{model_id}_{model_name}"

with open(save_path, "w") as f:
    json.dump(json_results, f, indent=2)