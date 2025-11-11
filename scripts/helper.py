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

    os.makedirs("/zfs/projects/students/ltdarc-usf-intern-2025/code/snapshots", exist_ok = True)
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
