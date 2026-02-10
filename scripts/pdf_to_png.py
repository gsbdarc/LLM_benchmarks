# Description
# - Converts PDF images to Greyscale PNG

# Setup

import os
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path

# Inputs

# where pdfs are located
image_dir = Path(
    "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pdfs")

# where new pngs should be saved
output_dir = "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pngs"

# Functions


def pdf_to_bw_png(pdf_path: str, output_dir: str, dpi: int = 300) -> None:
    """
    Converts a single page pdf into a greyscale PNG.
    Saves PNG to output_dir, keeps the orignal pdf name.
    Prints path of PNG and file size in MB's.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    bw_image = pages[0].convert('L')  # grey scale

    # Get original filename and save to new directory
    filename = os.path.basename(pdf_path).rsplit('.', 1)[0] + '.png'
    output_path = os.path.join(output_dir, filename)

    bw_image.save(output_path)

    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"{output_path}: {size_mb:.2f} MB")

    if os.path.getsize(output_path) > 5242880:
        print("Warning: file exceeds 5 MBs")


def main() -> None:
    """
    Converts and saves all pdfs within the image_dir to png.
    """

    for pdf_path in image_dir.iterdir():
        if pdf_path.suffix.lower() == ".pdf":
            pdf_to_bw_png(str(pdf_path), output_dir)


if __name__ == "__main__":
    main()
