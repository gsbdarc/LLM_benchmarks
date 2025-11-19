#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard library imports
import helper
import base64
import json
import logging
import sys
import os
import re
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


# In[2]:


os.makedirs("logs", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
log_path = f"logs/llm_call_{today}.log"

logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# In[3]:


def pdf_to_b64(pdf_path: str) -> str:
    """ Converts pdf image to b64."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=100)
    buffer = BytesIO()
    pages[0].save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()

    return img_b64


# In[4]:


class meta_data(BaseModel):
    newspaper_name: str
    publication_date: str


# In[5]:

example_1_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Austin_American_Statesman_Sun__Aug_3__2014_ (10).pdf"
example_2_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Chicago_Tribune_Sun__May_28__1995_ (30).pdf"

example_b64_1 = pdf_to_b64(example_1_path)
example_b64_2 = pdf_to_b64(example_2_path)


# In[62]:


def call_llm(
        b64: str,
        model: str,
        role_prompt: str,
        content_prompt: str,
        structured_output: Type[BaseModel]) -> BaseModel:
    """Feeds B64 image into LLM and returns structured output."""

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

    return response.output_parsed


# In[7]:


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


# In[8]:


def convert_string_to_date(input: str) -> datetime:
    """converts string to date object"""
    date_obj = pd.to_datetime(input, format="%B %d, %Y")
    return date_obj


# In[9]:


def calc_accuracy(df: pd.DataFrame, column_name) -> str:
    """calculates the accuracy rate of a boolean column."""
    accuracy = df[column_name].mean()

    msg = f"{column_name} accuracy: {accuracy:.2%}"

    logger.info("=== Accuracy ===")
    logger.info(msg)

    print(msg)


# In[10]:


def extract_newspaper_name(file_path: str) -> str:
    """
    Extracts the newspaper name from a file path like:
    'Arizona_Republic_Sun__Dec_17__2000_ (15).csv'
    and returns 'Arizona Republic Sun'.
    """
    # Get just the filename, e.g. "Arizona_Republic_Sun__Dec_17__2000_
    # (15).csv"
    filename = os.path.basename(file_path)
    # Remove the extension (.csv)
    filename = os.path.splitext(filename)[0]

    # Use regex to capture everything before the first '__'
    match = re.match(r"^(.*?)__", filename)
    name_part = match.group(1)
    parts = name_part.split("_")

    if parts[-1].lower() in ["sun"]:
        parts = parts[:-1]

    return " ".join(parts).strip()


# In[11]:


def normalize_name(name: str) -> str:
    """cleans newspaper name ahead of comparison"""
    name = name.lower()                     # ignore capitalization
    name = re.sub(r"\(.*?\)", "", name)     # remove parentheses and contents
    name = name.replace("-", " ")           # replace dashes with spaces
    # remove 'the' at start (if present)
    name = re.sub(r"^\s*the\s+", "", name)
    name = re.sub(r"\s+", " ", name)        # collapse multiple spaces
    return name.strip()


# In[12]:


load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")


# In[13]:


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# In[56]:


model = "gpt-5"


# In[15]:


role_prompt = (
    "You are a precise metadata extraction assistant. "
    "Your job is to identify and extract structured information from scanned newspaper TV guide images. "
    "You must always return the result as a valid JSON object that exactly matches the schema: "
    '{"newspaper": "<name>", "date": "<Month Day, Year>"}')


# In[16]:


content_prompt = """
You are analyzing a scanned newspaper page that includes a TV guide.

Your task is to extract exactly two fields:

1. `"newspaper"` — the full name of the newspaper where the TV guide appears.
2. `"date"` — the date the TV guide is for, formatted as "Month Day, Year" (e.g., "January 5, 2023").

Important extraction rules:
- If you see a mismatch (e.g., “Sun, Dec 17, 2000” in the upper right and “Wednesday” heading), assume the guide is for the **day immediately after** that date.
- If you see a range of dates (e.g. "May 28-June 3, 1995), assume the guide is for the first day within that range.
- If uncertain, output `null` for that field.

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
  "date": "May 28, 1995"
}

---

Now analyze the next image and output only valid JSON with the same keys:
{
  "newspaper": "...",
  "date": "..."
}
"""


# In[17]:


data_path = '/zfs/projects/students/ltdarc-usf-intern-2025/data'

index = helper.make_index(data_path)


# In[18]:


gpt_models = ["gpt-4o-mini", "gpt-4.1", "gpt-5-nano", "gpt-5-mini", "gpt-5"]


# In[63]:


# --- log prompts ---
logger.info("=== LLM CALL ===")
logger.info(f"Model: {model}")
logger.info(f"Role Prompt:\n{role_prompt}")
logger.info(f"Content Prompt:\n{content_prompt}")
logger.info(f"Reasoning Effort:\n{'high'}")

# Function to process ONE file (this will run in parallel)


def process_file(file):
    b64 = pdf_to_b64(file)
    llm_meta = call_llm(b64, model, role_prompt, content_prompt, meta_data)

    date_str = llm_meta.publication_date
    if date_str:
        date_obj = convert_string_to_date(date_str)
    else:
        date_obj = date_str

    return {
        "LLM_Newspaper_Name": llm_meta.newspaper_name,
        "LLM_Newspaper_Date": date_obj
    }


# Run the work in parallel
files = list(index['pdf_files'])

with Pool(34) as p:
    results = p.map(process_file, files)

# Add index after the fact
for i, r in enumerate(results):
    r["Index"] = i


# In[64]:


results_df = pd.DataFrame(results)

comparison_results = list()

for _, row in results_df.iterrows():
    ground_truth_path = index.loc[row["Index"], "ground_truth"]
    true_name = extract_newspaper_name(ground_truth_path)
    true_date = get_date_csv(ground_truth_path)

    name_match = normalize_name(
        row["LLM_Newspaper_Name"]) == normalize_name(true_name)
    date_match = row["LLM_Newspaper_Date"] == true_date

    comparison_results.append({
        "Index": row["Index"],
        "LLM_Newspaper_Name": row["LLM_Newspaper_Name"],
        "LLM_Newspaper_Date": row["LLM_Newspaper_Date"],
        "Actual_Name": true_name,
        "Actual_Date": true_date,
        "Name_Match": name_match,
        "Date_Match": date_match,
    })

comparison_df = pd.DataFrame(comparison_results)


# In[65]:


comparison_df


# In[66]:


calc_accuracy(comparison_df, "Name_Match")


# In[67]:


calc_accuracy(comparison_df, "Date_Match")


# In[ ]:
