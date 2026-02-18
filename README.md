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

### Create and Activate a Virtual Environment (YENs)

```bash
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Create and Activate a Virtual Environment (Sherlock)

Recommended: create new git branch for Sherlock

```
git checkout -b sherlock
```

Request a compute resources to create a venv via a slurm script

```
module load python/3.12
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
├── venv/
├── logs/
├── dev/
│   ├── development_notebooks/
│   └── archive/
└── LLM_benchmarks/
    ├── inputs/
    │   ├── models.json
    │   ├── benchmarks.json
    │   └── data/
    │       ├── pdfs/
    │       ├── pngs/
    │       └── csvs/
    ├── outputs/
    │   ├── results/
    │   └── metrics/
    └── scripts/
```

## Logs (`logs/`)

Results from prior evaluations of LLM calls, mostly GPT and LLAMA

---

## Development Workspace (`dev/`)

This folder is used for **iteration, experimentation, and debugging**.

### `development_notebooks/`
- Jupyter notebooks for:
  - Prompt prototyping
  - Model behavior exploration
  - Debugging Base64 encoding or schemas
  - Testing metric logic
  - Testing experiment pipelines

### `archive/`
- Older or deprecated notebooks
- Retained for historical context only

> ⚠️ Code in `dev/` is not considered production-ready.

---

## Input Configuration (`LLM_benchmarks/inputs/`)

These files define **what gets evaluated** and **how evaluation is performed**.

### `models.json`

Defines supported LLMs and model-specific configuration.

Example:
```json
{
    "0" : {
        "model": "llama-3.2",
        "family": "llama",
        "max_context_window": 128000},
    "1" : {
        "model": "gpt-4",
        "family": "gpt",
        "max_context_input": 128000,
        "max_context_output": 4096,
        "max_context_window": 132096,
        "detail": "low"}
}
```

This file allows you to:
- Add or remove models
- Adjust multimodal parameters and details
- Support non-Playground models in the future

---

### `benchmarks.json`

Defines **benchmark tasks** executed by LLMs.

Each benchmark includes:
- A unique ID
- A benchmark task name
- A system prompt
- A user prompt
- A benchmark task description
- An expected output schema

Example:
```json
{
    "0" : {
        "task_name": "newspaper_name",
        "system_prompt": "You are a metadata extraction assistant. Extract information from newspaper TV guide image. Always return valid JSON matching the exact schema provided.",
        "user_prompt": "Extract the newspaper name from this image.",
        "task_description": "Extraction: LLM should extract the name of the newspaper the TV guide is published in.",
        "schema":{
            "class_name": "NewspaperName", 
            "fields":{
                "newspaper_name": "str"}}}
}
```

Adding a new benchmark task typically requires **no changes to core code**, only this file. If updating prompts **do not** edit an existing benchmark, add a new benchmark instead.

### `image_index.json`

Defines **png images** to be processed by LLMs, creates a snapshot of all images to be processed.

Each benchmark includes:
- A unique ID
- Image PNG Path
- Ground Truth CSV Path

Example:
```json
{
    "0" : {
        "png": "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pngs/Austin_American_Statesman_Sun__Aug_3__2014_ (10).png",
        "csv": "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/csvs/Austin_American_Statesman_Sun__Aug_3__2014_ (10).csv"
}
```

### `mapping.csv`

Defines combinations of **benchmark tasks, models, and images** to be evaluted in the pipeline.

Dependencies (the following must be created prior to this file):
- models.json
- benchmarks.json
- image_index.json

Each combination includes:
- A unique ID
- Benchmark ID
- Benchmark Name
- Model ID
- Model Name
- Image ID
- Image Path

Example:
```csv
['0', 'newspaper_name', '2', 'gpt-4', '0', '/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pngs/Arizona_Republic_Sun__Dec_17__2000_ (15).png']
```

### `mapping_shuffled.csv`

Copy of mapping.csv where rows have been shuffled randomly.
Used to get a sample of results rather than all tasks in one go.
Same dependencies and file structure as mapping.csv.

### `ground_truth.json`

Stores **ground truth values** per image id.

Dependencies (the following must be created prior to this file):
- image_index.csv

Example:
```json
{
  "0": {
    "newspaper_name": "Arizona Republic",
    "newspaper_date": "Dec 17 2000",
    "day_of_week": "Wednesday",
    "tv_guide_date": "December 20 2000",
    "first_program": "Good Morning Arizona 94204",
    "first_channel": "3"
    }
}
```

## Data Overview (`LLM_benchmarks/inputs/data/`)

This directory contains **all raw and processed data assets** used during benchmarking.

### `pdfs/`
- Original scanned **PDF newspaper TV guide pages**
- Treated as immutable source files

### `pngs/`
- Greyscale PNG images converted from PDFs
- Used as inputs for multimodal LLM calls

### `csvs/`
- Human-transcribed **ground truth CSVs**
- Serve as the source of truth for evaluation

---

## Outputs (`LLM_benchmarks/outputs/`)

This directory stores the results and metrics from the pipeline.

### Results (`LLM_benchmarks/outputs/results`)

This directory stores the results of each run.

#### `results_{task_id}.json`

Stores raw model outputs and metadata by task id.

Example:
```json
{
  "0": {
      "output": "Arizona Republic",
      "completion_tokens": 9,
      "total_tokens": 1,
      "model": "gpt-4",
      "image_id": "0",
      "image_path": "/zfs/projects/students/ltdarc-usf-intern-2025/LLM_benchmarks/inputs/data/pngs/Arizona_Republic_Sun__Dec_17__2000_ (15).png",
      "task_id": "1",
      "task_name": "newspaper_name",
      "status": "processed"
  }
}
```

This file enables:
- Metric computation
- Debugging failed runs
- Cross-model comparison

### Metrics (`LLM_benchmarks/outputs/metrics`)

This directory stores metrics from successful runs and, separately, unsuccesful tasks that need further investigation.

#### `metrics_combined_results.json`

Stores all combined task outputs in "records" orient for easy conversion to DataFrames.

Example:
```json
{
  "0": {
      "task_id" : "0",
      "output": "Arizona Republic",
      "status": "processed",
      "completion_tokens": 9,
      "total_tokens": 1,
      "model": "gpt-4",
      "model_id": "1",
      "image_id": "0",
      "task_id": "1",
      "task_name": "newspaper_name",
      "ground_truth": "Arizona Republic",
      "accuracy":  1
  }
}
```

#### `metrics_{version_number}.json`

Stores successful task_id's, LLM outputs, ground truth, and accuracy results in "records" orient for easy conversion to DataFrames. Note that the keys are indices and not the task_id's.

Example:
```json
{
  "0": {
      "task_id" : "0",
      "output": "Arizona Republic",
      "completion_tokens": 9,
      "total_tokens": 1,
      "model": "gpt-4",
      "image_id": "0",
      "ground_truth": "Arizona Republic",
      "accuracy":  1
  }
}
```

This file enables:
- Analysis of LLM results


---

## Scripts (`LLM_benchmarks/scripts/`)

Contains **production-ready Python scripts**, including:

### `create_mapping.py`

Create a mapping file that: 
(1) finds all unique combinations of selected benchmarks, models, and images
(2) assigns a unique task id to each one
(3) saves these results into a csv file to be used in main.py

### `pdf_to_png.py`

Converts all PDFs in a given directoy to greyscale PNGs.
Saves PNGS to LLM_Benchmarks/iputs/data/pngs/ directory.
Prints PNG paths and file sizes in MBs.

### `main.py`

Orchestrates processing of a single task.
Tasks are loaded via the mapping.csv file
If the task has not already been processed then the corresponding benchmark, model, and image data is loaded from their respective JSONs.
A pydantic model is dynamically generated and inputs are passed into an LLM via Stanford API.
The following outputs are saved as an individual JSON file

- Task ID
- Image ID
- LLM output
- Completion tokens
- Total tokens
- Model ID
- Model Name
- Benchmark ID
- Benchmark Name
- Status

### `combine_results.py`

Loads all results within a directory.
Combines results into a single DataFrame.
Saves DataFrame as a JSON.
Prints the total number of successful and unsuccessful tasks, returns dictionary of error messages with counts.

### `compute.py`

Loads all results within a directory.
Separates unsuccessful tasks into a "investigate_df", saved as a JSON.
Loads ground truth file.
Evaluates model outputs compared to ground truth, assigns a accuracy score.
Saves results as a JSON.

---

## End-to-End Execution Flow

![Flowchart](./images/pipeline.png)

### Execution Steps

1. **User runs `main.py`**
   - Selects images, models, and tasks from `mapping.csv` via task_id

2. **Configuration loading**
   - Task definitions from `benchmarks.json`

3. **Image preprocessing**
   - Images are encoded into Base64
   - Task-specific payloads are created

4. **Model inference**
   - LLM responses are captured
   - Metadata (tokens, errors, status) is recorded
   - Results are saved to `results_{task_id}.json`

5. **Combining results**
   - `combine_results.py` compiles all results into a single JSON.

6. **Evaluation**
   - `compute_metrics.py` compares outputs to `ground_truth.json`
   - Metrics are computed per task and model
