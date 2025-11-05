#!/usr/bin/env python
# coding: utf-8

# In[80]:


# Standard library imports
import base64
import json
import logging
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Type

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

# Local application imports
import Validity_Functions


# In[50]:


def pdf_to_b64(pdf_path: str) -> str:
    """ Converts pdf image to b64."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=100)
    buffer = BytesIO()
    pages[0].save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()

    return img_b64


# In[101]:


class meta_data(BaseModel):
    newspaper_name: str
    publication_date: str


# In[102]:


def call_llm(
        b64: str,
        model: str,
        role_prompt: str,
        content_prompt: str,
        structured_output: Type[BaseModel]) -> BaseModel:
    """Feeds B64 image into LLM and returns structured output."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.parse(model=model,
                                      input=[{"role": "system",
                                              "content": [{"type": "input_text",
                                                           "text": role_prompt}]},
                                             {"role": "user",
                                              "content": [{"type": "input_text",
                                                           "text": content_prompt},
                                                          {"type": "input_image",
                                                           "image_url": f"data:image/png;base64,{b64}",
                                                           "detail": "low"},
                                                          ],
                                              },
                                             ],
                                      text_format=structured_output,
                                      )

    return response.output_parsed


# In[104]:


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


# In[105]:


def convert_string_to_date(input: str) -> datetime:
    """converts string to date object"""
    date_obj = pd.to_datetime(input, format="%B %d, %Y")
    return date_obj


# In[99]:


def calc_accuracy(df: pd.DataFrame, column_name) -> str:
    """calculates the accuracy rate of a boolean column."""
    accuracy = df[column_name].mean()
    msg = f"{column_name} accuracy: {accuracy:.2%}"

    logger.info("=== Accuracy ===")
    logger.info(msg)

    print(msg)


# In[66]:


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


# In[106]:


def normalize_name(name: str) -> str:
    """cleans newspaper name ahead of comparison"""
    name = name.lower()                     # ignore capitalization
    name = re.sub(r"\(.*?\)", "", name)     # remove parentheses and contents
    name = name.replace("-", " ")           # replace dashes with spaces
    # remove 'the' at start (if present)
    name = re.sub(r"^\s*the\s+", "", name)
    name = re.sub(r"\s+", " ", name)        # collapse multiple spaces
    return name.strip()


# In[97]:


today = datetime.now().strftime("%Y-%m-%d")
log_path = f"logs/llm_call_{today}.log"

logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# In[51]:


load_dotenv("/zfs/projects/students/ltdarc-usf-intern-2025/.env")


# In[52]:


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# In[53]:


model = "gpt-4o-mini"


# In[54]:


role_prompt = "You are a metadata extraction assistant that specializes reading the TV guides of newspapers."


# In[55]:


content_prompt = """Look at this newspaper page and extract two fields:
1. The name of the newspaper
2. The date the tv guide is for (do not include day of week)

Return your answer ONLY as a valid JSON object with the following keys:
  - newspaper
  - date

Example:
{
  "newspaper": "The New York Times",
  "date": "January 5, 2023"
}
"""


# In[59]:


data_path = '/zfs/projects/students/ltdarc-usf-intern-2025/data'

index = Validity_Functions.make_index(data_path)


# In[98]:


results = []

# --- log prompts ---
logger.info("=== LLM CALL ===")
logger.info(f"Model: {model}")
logger.info(f"Role Prompt:\n{role_prompt}")
logger.info(f"Content Prompt:\n{content_prompt}")

for idx, row in index.iterrows():
    pdf_path = row["pdf_files"]
    b64 = pdf_to_b64(pdf_path)

    llm_meta = call_llm(b64, model, role_prompt, content_prompt, MetaData)
    date_str = llm_meta.publication_date
    date_obj = convert_string_to_date(date_str)

    results.append({
        "Index": idx,
        "LLM_Newspaper_Name": llm_meta.newspaper_name,
        "LLM_Newspaper_Date": date_obj
    })

results_df = pd.DataFrame(results)


# In[73]:


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


# In[74]:


print(comparison_df)


# In[100]:


calc_accuracy(comparison_df, "Name_Match")


# In[107]:


calc_accuracy(comparison_df, "Date_Match")
