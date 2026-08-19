# Description
# - Converts PDF images to Greyscale PNG

# Setup

import os
from pdf2image import convert_from_path
from dotenv import load_dotenv
from pathlib import Path

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


def main(base_dir: str | Path | None = None) -> None:
    """
    Converts and saves all PDFs in the configured input directory to PNG.
    """
    if base_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")
        configured_base_dir = os.getenv("BASE_DIR")
        if not configured_base_dir:
            raise RuntimeError(
                "BASE_DIR is not configured; set it in the repository .env"
            )
        base_dir = configured_base_dir

    image_dir = Path(base_dir) / "inputs" / "data" / "pdfs"
    output_dir = Path(base_dir) / "inputs" / "data" / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in image_dir.iterdir():
        if pdf_path.suffix.lower() == ".pdf":
            pdf_to_bw_png(str(pdf_path), output_dir)


if __name__ == "__main__":
    main()
