"""
- Goal: iterate through CSVs and extract ground truth
- Output: json file
- Extracted Fields:
    - Newspaper Name
    - Newspaper Publish Date
    - TV Guide Day of Week
    - TV Guide Date
    - First Program (manual)
    - First Channel (manual)
"""

# Setup

import pandas as pd
import json
import re
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Functions


def load_ground_truth(ground_truth_path: str) -> dict:
    """
    Loads existing ground_truth JSON or instantiates a new dictionary.
    """
    try:
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
    except BaseException:
        ground_truth = dict()  # initalize new dicitionary if unable to find ground_truth.json

    return ground_truth


def parse_newspaper_string(filename: str) -> tuple[str, str]:
    """
    Extracts newspaper name and date object from 'File Name' field in CSVs
    """

    filename = filename.strip()

    # Double underscore before month, single underscores within the date
    match = re.match(
        r'(.+)__(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_(\d{1,2})__(\d{4})_(?:\s*\((\d+)\))?',
        filename)

    if match:
        name = match.group(1).replace('_', ' ')
        name = re.sub(r'\s*Sun$', '', name)

        month = match.group(2)
        day = match.group(3)
        year = match.group(4)

        date = f"{month} {day} {year}"

        return name, date

    return None


def update_ground_truth(
        BASE_DIR: str,
        ground_truth: dict,
        image_index_path: str) -> dict:
    """
    Updates ground truth dictionary with newspaper_name, newsppaer_date, day_of_week, and tv_guide_date from CSVs.
    """

    with open(image_index_path, "r") as f:
        image_index = json.load(f)

    for i in range(0, len(image_index)):
        index = str(i)  # consistent with how other inputs are indexed

        if index in ground_truth:
            continue

        image = image_index[index]['csv']
        csv_path = os.path.join(BASE_DIR, "inputs", "data", "csvs", image)

        csv_df = pd.read_csv(csv_path)

        day_of_week = csv_df['Day'][0]

        year = csv_df['Year'][0]
        month = csv_df['Month'][0]
        day = csv_df['Date'][0]
        tv_guide_date = f"{month} {day} {year}"

        file_name = csv_df['File Name'][0]
        result = parse_newspaper_string(file_name)

        if result:  # if able to parse
            name, date_obj = result
            ground_truth[index] = {
                'newspaper_name': name,
                'newspaper_date': date_obj,
                'day_of_week': day_of_week,
                'tv_guide_date': tv_guide_date}
        else:
            print(f"Could not parse: {file_name}")

    return ground_truth


def manually_update_ground_truth(ground_truth: dict) -> dict:
    """
    Updates ground truth with wiht First Program and First Channel values.
    """
    ground_truth['0']['first_program'] = 'Good Morning Arizona 94204'
    ground_truth['0']['first_channel'] = '3'

    ground_truth['1']['first_program'] = 'Good Day Austin Saturday'
    ground_truth['1']['first_channel'] = 'Fox/7.1'

    ground_truth['2'][
        'first_program'] = '2015 Daytona 500 The 57th running of the event. The race consists of 200 laps and is the first race of the season. (N) (cc)'
    ground_truth['2']['first_channel'] = 'Fox/7.1'

    ground_truth['3']['first_program'] = 'Good Day Austin Saturday'
    ground_truth['3']['first_channel'] = 'Fox/7.1'

    ground_truth['4']['first_program'] = 'News'
    ground_truth['4']['first_channel'] = 'Fox/7.1'

    ground_truth['5']['first_program'] = 'Ancient Mysteries'
    ground_truth['5']['first_channel'] = 'AE'

    ground_truth['6']['first_program'] = 'Ancient Mysteries'
    ground_truth['6']['first_channel'] = 'AE'

    ground_truth['7']['first_program'] = 'News at Noon'
    ground_truth['7']['first_channel'] = '2'

    ground_truth['8']['first_program'] = 'CBS 2 News'
    ground_truth['8']['first_channel'] = '2 WCBS'

    ground_truth['9']['first_program'] = 'CBS 2 News Sunday (N)'
    ground_truth['9']['first_channel'] = '2 WCBS'

    ground_truth['10']['first_program'] = 'CBS 2 News Sunday (N)'
    ground_truth['10']['first_channel'] = '2 WCBS'

    ground_truth['11']['first_program'] = '(4:30) CBS 2 News This Morning (N) (cc)'
    ground_truth['11']['first_channel'] = '2 WCBS'

    ground_truth['12']['first_program'] = 'NCAA Basketball'
    ground_truth['12']['first_channel'] = '2 WCBS'

    ground_truth['13']['first_program'] = 'CBS 2 News Saturday (N) (cc)'
    ground_truth['13']['first_channel'] = '2 WCBS'

    ground_truth['14']['first_program'] = '30-Minute Meals'
    ground_truth['14']['first_channel'] = 'FOOD'

    ground_truth['15']['first_program'] = 'SNAKES ON A PLANE (\'06) Samuel L. Jackson, Kenan Thompson. 3066705'
    ground_truth['15']['first_channel'] = 'FX'

    ground_truth['16']['first_program'] = 'Wake Up 2Day'
    ground_truth['16']['first_channel'] = '2 KHON 3 3 3'

    ground_truth['17']['first_program'] = 'News'
    ground_truth['17']['first_channel'] = '7 WTVW'

    ground_truth['18']['first_program'] = 'ABC World News Tonight (CC) 173'
    ground_truth['18']['first_channel'] = '2 KATU ABC'

    ground_truth['19']['first_program'] = 'Politically 9292145'
    ground_truth['19']['first_channel'] = '2 KATU'

    ground_truth['20']['first_program'] = 'News (CC) 85682'
    ground_truth['20']['first_channel'] = '2 KATU'

    ground_truth['21']['first_program'] = 'Biography: Gene Hackman. 413091'
    ground_truth['21']['first_channel'] = 'A&E'

    ground_truth['22']['first_program'] = 'News'
    ground_truth['22']['first_channel'] = 'KCBS 2'

    ground_truth['23']['first_program'] = '(6:00) News (CC) 89164'
    ground_truth['23']['first_channel'] = '2 3 2 Fox'

    ground_truth['24']['first_program'] = 'Futurerama 8712'
    ground_truth['24']['first_channel'] = '2 3 003 2'

    ground_truth['25']['first_program'] = '(6:00) News (CC) 52978'
    ground_truth['25']['first_channel'] = '2 3 003 2'

    ground_truth['26']['first_program'] = '(11:00) Auto Racing NASCAR Winston Cup -- Coca-Cola 600. (S Live) (CC) 193997'
    ground_truth['26']['first_channel'] = '2 3 003 2'

    ground_truth['27']['first_program'] = '(6:00) News (CC) 34666'
    ground_truth['27']['first_channel'] = '3 2 Fox'

    ground_truth['28']['first_program'] = 'King of Hill'
    ground_truth['28']['first_channel'] = '4 WDAF'

    ground_truth['29']['first_program'] = 'Biography: Alan Alda 718318'
    ground_truth['29']['first_channel'] = 'A&E'

    ground_truth['30']['first_program'] = 'Noticiero Telemundo: Edición Matutina 31717'
    ground_truth['30']['first_channel'] = 'Telemundo 2 XRIO'

    ground_truth['31']['first_program'] = 'Jimmy Kimmel Live (CC) (HD)'
    ground_truth['31']['first_channel'] = 'KOMO ABC 4 4'

    ground_truth['32']['first_program'] = 'Born Losers (1967) Tom Laughlin. Part-Indian foot fighter Billy Jack vs. outlaw bikers. (Crime drama, PG, (2:00) +9493577'
    ground_truth['32']['first_channel'] = 'TBS'

    ground_truth['33']['first_program'] = 'Spider-Man (CC) 61721 7363'
    ground_truth['33']['first_channel'] = '43 5'

    ground_truth['34']['first_program'] = 'Sesame Street (CC) 123017'
    ground_truth['34']['first_channel'] = '33'

    return ground_truth


def main():
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root/".env")
    BASE_DIR = os.getenv("BASE_DIR")
    image_index_path = os.path.join(BASE_DIR, "inputs", "image_index.json")
    ground_truth_path = os.path.join(
        BASE_DIR, "inputs", "ground_truth_test.json")
    ground_truth = load_ground_truth(ground_truth_path)
    updated_ground_truth = update_ground_truth(
        BASE_DIR, ground_truth, image_index_path)
    full_ground_truth = manually_update_ground_truth(updated_ground_truth)

    with open(ground_truth_path, "w") as f:
        json.dump(full_ground_truth, f, indent=2)


if __name__ == "__main__":
    main()
