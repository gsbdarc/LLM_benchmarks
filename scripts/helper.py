import pandas as pd
import os
import logging
import base64
import re

from datetime import datetime
from io import BytesIO
from pdf2image import convert_from_path

os.makedirs("logs", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
log_path = f"logs/llm_call_{today}.log"

logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def make_index(path: str):

    """
    Creates a DataFrame with path locations of
    PDF and CSV files.
    """

    os.makedirs("/zfs/projects/students/ltdarc-usf-intern-2025/data/snapshots", exist_ok = True)
    files=os.listdir(path)
    
    pdf_files = [path+'/'+f for f in files if f.endswith('.pdf')]
    csv_files = [path+'/'+f for f in files if f.endswith('.csv')]

    pdf_files.sort()
    csv_files.sort()
    
    df_final=pd.DataFrame(zip(pdf_files, csv_files),columns=['pdf_files','ground_truth'])
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot_path = os.path.join("/zfs/projects/students/ltdarc-usf-intern-2025/code/snapshots", f"index_{today}.csv")
    df_final.to_csv(snapshot_path)

    return df_final

def check_index(df: pd.DataFrame):
    """
    Function checks paths in every row to ensure they exist.
    """
    # Generate today's date
    today = datetime.now().strftime("%Y-%m-%d")

    # Ensure logs folder exists 
    os.makedirs("logs", exist_ok = True)

    #filename should include the date
    log_filename = f"logs/missing_files_{today}.log"

    #set up logging
    logger = logging.getLogger("file_validator")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    error_count = 0
    
    for idx,row in df.iterrows():
        #Check PDF loadability
        try:
            pages = convert_from_path(row['pdf_files'], first_page=1, last_page=1)
        except Exception as e:
            logger.error(f"Row {idx} — PDF load failed: {row['pdf_files']} — {e}")
            error_count += 1

        #Check CSV loadability
        try:
            ground_truth= pd.read_csv(row['ground_truth'], nrows=1)  # lightweight test
        except Exception as e:
            logger.error(f"Row {idx} — CSV load failed: {row['ground_truth']} — {e}")
            error_count += 1

    if error_count == 0:
        return False, "All files are loadable"
    else:
        return True, f"{error_count} errors, logs created."
        
def file_viewer(df: pd.DataFrame, index: int):
    """
    Generates PDF image and loads CSV table based
    on the index value of the DataFrame.
    """
    pdf_path = df.iloc[index]['pdf_files']
    csv_path = df.iloc[index]['ground_truth']

    pages = convert_from_path(pdf_path)
    img = pages[0]

    display(img)

    truth=pd.read_csv(csv_path)

    return truth

def pdf_to_b64(pdf_path: str) -> str:
    """ Converts pdf image to b64."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=100)
    buffer = BytesIO()
    pages[0].save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode()

    return img_b64

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

def convert_string_to_date(input: str) -> datetime:
    """converts string to date object"""
    date_obj = pd.to_datetime(input, format="%B %d, %Y")
    return date_obj

def calc_accuracy(df: pd.DataFrame,  column_name) -> str:
    """calculates the accuracy rate of a boolean column."""
    accuracy = df[column_name].mean()
    #msg = f"{column_name} accuracy: {accuracy:.2%}"

    return accuracy

def extract_newspaper_name(file_path: str) -> str:
    """
    Extracts the newspaper name from a file path like:
    'Arizona_Republic_Sun__Dec_17__2000_ (15).csv'
    and returns 'Arizona Republic Sun'.
    """
    # Get just the filename, e.g. "Arizona_Republic_Sun__Dec_17__2000_ (15).csv"
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

def normalize_name(name: str) -> str:
    """cleans newspaper name ahead of comparison"""
    name = name.lower()                     # ignore capitalization
    name = re.sub(r"\(.*?\)", "", name)     # remove parentheses and contents
    name = name.replace("-", " ")           # replace dashes with spaces
    name = re.sub(r"^\s*the\s+", "", name)  # remove 'the' at start (if present)
    name = re.sub(r"\s+", " ", name)        # collapse multiple spaces
    return name.strip()

