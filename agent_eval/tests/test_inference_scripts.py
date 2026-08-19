import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(filename: str):
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdf_conversion_main_creates_output_and_processes_only_pdfs(
    tmp_path, monkeypatch
):
    script = load_script("1_pdf_to_png.py")
    pdf_dir = tmp_path / "inputs" / "data" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "first.pdf").touch()
    (pdf_dir / "second.PDF").touch()
    (pdf_dir / "notes.txt").touch()
    calls = []
    monkeypatch.setattr(
        script,
        "pdf_to_bw_png",
        lambda pdf_path, output_dir: calls.append(
            (Path(pdf_path).name, Path(output_dir))
        ),
    )

    script.main(tmp_path)

    output_dir = tmp_path / "inputs" / "data" / "pngs"
    assert output_dir.is_dir()
    assert set(calls) == {
        ("first.pdf", output_dir),
        ("second.PDF", output_dir),
    }


def test_inference_main_requires_an_integer_task_id():
    script = load_script("5_main.py")

    assert script.parse_args(["42"]).task_id == 42
    with pytest.raises(SystemExit):
        script.parse_args([])
    with pytest.raises(SystemExit):
        script.parse_args(["not-an-integer"])


def test_ground_truth_update_leaves_existing_rows_unchanged(tmp_path):
    script = load_script("4_extract_ground_truth.py")
    image_index_path = tmp_path / "image_index.json"
    image_index_path.write_text(
        '{"0": {"csv": "already-loaded.csv"}}', encoding="utf-8"
    )
    existing = {"0": {"newspaper_name": "Existing value"}}

    updated = script.update_ground_truth(tmp_path, existing, image_index_path)

    assert updated == existing
