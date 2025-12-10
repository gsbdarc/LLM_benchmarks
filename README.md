# LLM_benchmarks

Purpose: Evaluate open- and closed-source LLMs with vision capabilities, both explicitly vision-labeled (i.e. Qwen3-vl) and general multimodal models (i.e. GPT-5), on their ability to extract data, interpret tables, and perform reasoning tasks on TV guide PDFs collected from historical newspapers.

Start by cloning repo:

```
git clone https://github.com/gsbdarc/LLM_benchmarks
```

Create and activate virtual environment:

```
/usr/bin/python3  -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Making a kernel (optional but highly recommended):

```
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv
```

Configure environment variables:

- Create an .env file
- Add OPENAI_API_Key=your_key_here
- Add STANFORD_API_KEY=your_key_here

Directory Structure:

LLM_benchmarks/
    scripts/
        helper.py
        run_ollama_server.slurm
        1_snapshot.py
    notebooks/
    data/
        <pdf files>
        <csv files>

Note: Your PDF filenames must match the format expected by helper.make_index(), where each PDF is in the same order as its correct ground-truth CSV.

Script Overview:

- `1_snapshot.py`
 - Creates a full index of a pdf and csv files in /data
 - Confirms that every file loads correctly via pdf2image
 - Generates a snapshot DataFrame used by later scripts

- `run_ollama_server.slurm`

```
sbatch run_ollama_server.slurm
```
 - Starts ollama server
 - Cancel job once you're done with ollama LLMs

```
squeue
scancel JOBID
```

Noteboks Overview:

- For code that is still in testing/WIP

- `Instructor_Playground.ipynb`
 - Testing compatability of Instructor module with Stanford AI Playground API 
 - Passing PNGs into models via image input, asking models to return
  -  Newspaper Publishing Date
  -  TV Guide Day of Week
  -  Date of TV Guide
 - Conclusion: Stanford API does not support direct image processesing, Instructor can not be used as a way a circumvent this. Will call API's directly with Instructor module.

How to run scripts:

- `1_snapshot.py`: run this first, creates snapshot of all files in your data folder and checks that they are able to load without issue
