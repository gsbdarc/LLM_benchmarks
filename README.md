# LLM Benchmarking Project

## Project Goal

The goal of this project is to **develop and evaluate benchmark suites for multimodal large language models (LLMs)**, with a primary focus on **image-based extraction, processing, and reasoning tasks**.

- **Current focus**: Evaluating models available through the **Stanford AI Playground API**
- **Future-ready design**: The architecture is intentionally flexible to support:
  - Non-Playground LLMs
  - Additional multimodal tasks
  - New evaluation metrics
  - Changes in prompt or schema design

This repository separates **configuration, data, and execution logic** so that benchmarks can evolve without major code refactors.

---

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/gsbdarc/LLM_benchmarks
cd LLM_benchmarks
```

---

### Create and Activate a Virtual Environment

```bash
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### (Optional but Recommended) Create a Jupyter Kernel

```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv
```

---

## Environment Variables

Create a `.env` file in the project root with:

```text
OPENAI_API_KEY=your_key_here
STANFORD_API_KEY=your_key_here
```
---

## High-Level Directory Structure

```
Input Files/
├── venv/
├── data/
│   ├── pdf_images/
│   ├── png_images/
│   ├── logs/
│   └── csv_files/
├── dev/
│   ├── development_notebooks/
│   └── archive/
└── LLM_benchmarks/
    ├── input_files/
    ├── output_files/
    └── scripts/
```

Each directory has a single responsibility, described below.

---

## Data Overview (`data/`)

This directory contains **all raw and processed data assets** used during benchmarking.

### `pdf_images/`
- Original scanned **PDF newspaper TV guide pages**
- Treated as immutable source files

### `png_images/`
- PNG images converted from PDFs
- Used as inputs for multimodal LLM calls

### `logs/`
- Results from prior evaluations of LLM calls, mostly GPT and LLAMA

### `csv_files/`
- Human-transcribed **ground truth CSVs**
- Serve as the source of truth for evaluation

---

## Development Workspace (`dev/`)

This folder is used for **iteration, experimentation, and debugging**.

### `development_notebooks/`
- Jupyter notebooks for:
  - Prompt prototyping
  - Model behavior exploration
  - Debugging Base64 encoding or schemas
  - Testing metric logic

### `archive/`
- Older or deprecated notebooks
- Retained for historical context only

> ⚠️ Code in `dev/` is not considered production-ready.

---

## Production Codebase (`LLM_benchmarks/`)

This directory contains the **authoritative, production-ready benchmarking system**.

All benchmark runs should be executable **without notebooks**, using scripts only.

---

## Input Configuration (`LLM_benchmarks/input_files/`)

These files define **what gets evaluated** and **how evaluation is performed**.

### `models.json`

Defines supported LLMs and model-specific configuration.

Example:
```json
{
  "gpt-5": {
    "detail": "low",
    "max_context_len": 272000
  },
  "gpt-4": {
    "detail": "low",
    "max_context_len": 1047576
  }
}
```

This file allows you to:
- Add or remove models
- Adjust multimodal parameters and details
- Support non-Playground models in the future

---

### `benchmarks.json`

Defines **tasks** executed by LLMs.

Each task includes:
- A system prompt
- A user prompt
- A task description
- An expected output schema

Example:
```json
{
  "newspaper_name": {
    "system_prompt": "You are ...",
    "user_prompt": "...",
    "task_description": "Extract the newspaper name from the image",
    "schema": {
      "class_name": "NewspaperName",
      "fields": {
        "newspaper_name": "str"
      }
    }
  }
}
```

Adding a new task typically requires **no changes to core code**, only this file.

---

### `ground_truth.json`

Stores **ground truth values** per task.

Example:
```json
{
  "newspaper_name": {
    "metric": "accuracy",
    "images": [
      {
        "image_path": "/zfs/project/...",
        "ground_truth": "Arizona Paper"
      }
    ]
  }
}
```

This structure supports:
- Task-specific metrics
- Multiple images per task

---

## Output Artifacts (`LLM_benchmarks/output_files/`)

This directory is populated automatically after benchmark runs.

### `results.json`

Stores raw model outputs and metadata.

Example:
```json
{
  "newspaper_name": {
    "gpt-5": {
      "results": {
        "image_path_1": {
          "NewspaperName": "Some Output",
          "completion_tokens": 123,
          "error_message": null
        }
      }
    }
  }
}
```

This file enables:
- Metric computation
- Debugging failed runs
- Cross-model comparison
- Reproducibility

---

## Scripts (`LLM_benchmarks/scripts/`)

Contains **production-ready Python scripts**, including:
- `main.py` — orchestrates benchmark runs
- `compute_metrics.py` — evaluates model outputs against ground truth

---

## End-to-End Execution Flow

A visual pipeline overview is available here:  
**Figma Flowchart**  
https://www.figma.com/board/4hsI8oro2HJvl5MPIxHBPC/Pipeline-Visualization?node-id=25-77&p=f&t=c5xVyTikN39ckBle-0

### Execution Steps

1. **User runs `main.py`**
   - Selects images, models, and tasks via arguments

2. **Configuration loading**
   - Model details from `models.json`
   - Task definitions from `benchmarks.json`

3. **Image preprocessing**
   - Images are encoded into Base64
   - Model- and task-specific payloads are created

4. **Model inference**
   - LLM responses are captured
   - Metadata (tokens, errors, timing) is recorded
   - Results are saved to `results.json`

5. **Evaluation**
   - `compute_metrics.py` compares outputs to `ground_truth.json`
   - Metrics are computed per task and model




  