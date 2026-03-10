"""
Make a snapshot of PNG and CSV names, used to create mapping.csv.
"""

# Setup

from pathlib import Path
from dotenv import load_dotenv
import json
import os
import glob

# Functions


def make_index(image_paths: str, index_file: str) -> None:
    """
    Loads or creates new image_index dictionary, updates based on images in image_paths.
    Saves updated index as json file.
    """

    if os.path.exists(index_file) and os.path.getsize(
            index_file) > 0:  # if image index already exists and isn't empty

        with open(index_file, "r") as f:
            image_dict = json.load(f)

    else:  # if image dictionary does not already exist
        image_dict = {}

    for image in image_paths:
        filename = os.path.basename(image)  # get the filename itself
        if filename not in [v['png'] for v in image_dict.values(
        )]:  # if the file isn't already in the image_dictionary
            index = len(image_dict)
            csv_filename = filename.replace(
                '.png', '.csv')  # create a path for the CSV
            image_dict[index] = {
                'png': filename,
                'csv': csv_filename}  # index should be len(dict)

    with open(index_file, "w") as f:
        json.dump(image_dict, f, indent=2)


def main():
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root/".env")
    BASE_DIR = os.getenv("BASE_DIR")

    image_folder = os.path.join(BASE_DIR, "inputs", "data", "pngs")
    image_paths = list(Path(image_folder).glob("*.png"))

    index_file = os.path.join(BASE_DIR, "inputs", "image_index.json")

    make_index(image_paths, index_file)


if __name__ == "__main__":
    main()
