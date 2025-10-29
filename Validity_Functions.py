import pandas as pd
import os
import logging
import base64

from datetime import datetime
from pdf2image import convert_from_path

def make_index(path: str):

    """
    Creates a DataFrame with path locations of
    PDF and CSV files.
    """
    
    files=os.listdir(path)
    
    pdf_files = [path+'/'+f for f in files if f.endswith('.pdf')]
    csv_files = [path+'/'+f for f in files if f.endswith('.csv')]

    pdf_files.sort()
    csv_files.sort()
    
    df_final=pd.DataFrame(zip(pdf_files, csv_files),columns=['pdf_files','ground_truth'])

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
    logging.basicConfig(
        filename=log_filename,
        level=logging.ERROR, #log should only be created if an error occurs
        format="%(asctime)s - %(levelname)s - %(message)s") #what kind of message is included in the log

    error_count = 0
    
    for idx,row in df.iterrows():
        if not os.path.exists(row['pdf_files']):
            logging.error(f"File not found: {row['pdf_files']}")
            error_count += 1
        if not os.path.exists(row['ground_truth']):
            logging.error(f"Ground truth file not found: {row['ground_truth']}")
            error_count += 1

    if error_count > 0:
        return f"{error_count} missing file(s) logged."
    else:
        return f"All files exist, no log file created."

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
