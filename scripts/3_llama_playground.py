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
script_dir = os.path.abspath(
    '/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/scripts')
sys.path.append(script_dir)


# In[12]:


def pdf_to_b64(pdf_path: str) -> str:
    """ Converts pdf image to b64."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=100)
    buffer = BytesIO()
    pages[0].save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()

    return img_b64


# In[23]:


def call_llm(user_prompt):
    LLM_API_URL = f"{ollama_url}/api/chat"

    payload = {
        "model": "llama3.2-vision:11b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "format": output_schema,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        LLM_API_URL,
        headers=headers,
        data=json.dumps(payload))
    result = response.json()
    content_dict = json.loads(result["message"]["content"])
    # print(content_dict["newspaper"])
    # print(content_dict["date"])
    return content_dict


# In[34]:


def convert_string_to_date(input: str) -> datetime:
    """Convert many common newspaper date formats to datetime."""

    formats = [
        "%B %d, %Y",       # January 17, 1999
        "%b %d, %Y",       # Jan 17, 1999
        "%a, %b %d, %Y",   # Sun, Mar 5, 2000
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(input, format=fmt)
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date format: {input}")


# In[27]:


data_path = '/zfs/projects/students/ltdarc-usf-intern-2025/data'

index = helper.make_index(data_path)


# In[2]:


ollama_url = None
SCRATCH_BASE = f"/scratch/shared/{os.environ['USER']}"
# Attempt to read the host and port from scratch folder
try:
    with open(f"{SCRATCH_BASE}/ollama/host.txt") as f:
        HOST = f.read().strip()
    with open(f"{SCRATCH_BASE}/ollama/port.txt") as f:
        PORT = f.read().strip()
    ollama_url = f"http://{HOST}:{PORT}"
except Exception as e:
    print("[⚠️] Could not read host/port from scratch. If using someone else's server, manually set the `ollama_url` below.")
    # You can optionally provide a fallback manually here
    # ollama_url = 'http://HOST:PORT'  # Uncomment and modify if needed

# If no URL is available, raise an exception or pause
if ollama_url is None:
    raise ValueError(
        "No Ollama server URL available. Please set `ollama_url` manually and run cell again")

# Parse server and port from the URL
server = ':'.join(ollama_url.split(':')[0:2])
port = ollama_url.split(':')[-1]

# Display interaction information
display(Markdown(f"""
# How can we interact with this server?

Once the server is running, you can interact with it by sending HTTP requests to its URL.

The server is running on **{server}**, and you need to contact it through port **{port}**.
"""))


# In[3]:


try:
    r = requests.get(ollama_url)
    r.raise_for_status()
    print("✅ Success:", r.text)
except requests.exceptions.RequestException as e:
    print("❌ Failed to connect:", e)


# In[4]:


os.makedirs("logs", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
log_path = f"logs/llm_call_{today}.log"

logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# In[7]:


system_prompt = (
    "You are a precise metadata extraction assistant. "
    "Your job is to identify and extract structured information from scanned newspaper TV guide images. "
    "You must always return the result as a valid JSON object that exactly matches the schema: "
    '{"newspaper": "<name>", "date": "<Month Day, Year>"}')


# In[8]:


content_prompt = """You are analyzing a scanned newspaper page that includes a TV guide.

Your task is to extract exactly two fields:

1. `"newspaper"` — the full name of the newspaper where the TV guide appears.
2. `"date"` — the date the TV guide is for, formatted as "Month Day, Year" (e.g., "January 5, 2023").

Important extraction rules:
- Ignore any day-of-week labels such as "Sunday" or "Wed".
- If there is a mismatch (for example, a date appears in one corner and a different day of week appears elsewhere), assume the guide is for the **day immediately after** that date.
  Example: if you see “Sun, Dec 17, 2000” with a “Wednesday” heading, the correct date is “December 20, 2000”.
- If you cannot confidently determine the value, output `null` for that field.

Formatting rules:
- Return only a valid JSON object.
- Do **not** include text such as “newspaper name”, “publication date”, “field name”, or any placeholders.
- Do **not** include explanations, markdown, or additional commentary.

Expected output schema:
{
  "newspaper": "<actual newspaper name>",
  "date": "<Month Day, Year>"
}"""


# In[13]:


example_1_path = "/zfs/projects/students/ltdarc-usf-intern-2025/data/Austin_American_Statesman_Sun__Aug_3__2014_ (10).pdf"
example_b64_1 = pdf_to_b64(example_1_path)


# In[11]:


output_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "newspaper": {"type": "string"},
        "date": {"type": "string"}
    },
    "required": ["newspaper", "date"]
}


# In[43]:


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


# In[45]:


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


# In[44]:


def calc_accuracy(df: pd.DataFrame, column_name) -> str:
    """calculates the accuracy rate of a boolean column."""
    accuracy = df[column_name].mean()

    msg = f"{column_name} accuracy: {accuracy:.2%}"

    logger.info("=== Accuracy ===")
    logger.info(msg)

    print(msg)


# In[46]:


def normalize_name(name: str) -> str:
    """cleans newspaper name ahead of comparison"""
    name = name.lower()                     # ignore capitalization
    name = re.sub(r"\(.*?\)", "", name)     # remove parentheses and contents
    name = name.replace("-", " ")           # replace dashes with spaces
    # remove 'the' at start (if present)
    name = re.sub(r"^\s*the\s+", "", name)
    name = re.sub(r"\s+", " ", name)        # collapse multiple spaces
    return name.strip()


# In[41]:


def process_file(file):
    b64 = pdf_to_b64(file)
    user_prompt = f"""
    Here is the image, base64-encoded:

    {file}

    Please determine the name of the newspaper and the date the TV guide is for.

    """
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


# Run the work in parallel
files = list(index['pdf_files'])

with Pool(34) as p:
    results = p.map(process_file, files)

for i, r in enumerate(results):
    r["Index"] = i


# In[14]:


user_prompt = f"""
Here is the image, base64-encoded:

{example_b64_1}

Please determine the name of the newspaper and the date the TV guide is for.


"""


# In[42]:


print(results)


# In[47]:


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


# In[48]:


comparison_df


# In[49]:


calc_accuracy(comparison_df, "Name_Match")


# In[50]:


calc_accuracy(comparison_df, "Date_Match")
