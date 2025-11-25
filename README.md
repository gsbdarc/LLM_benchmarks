# LLM_benchmarks

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

Directory Structure:

LLM_benchmarks/
    scripts/
        helper.py
        1_snapshot.py
        2_gpt_playground.py
        3_llama_playground.py
    data/
        <pdf files>
        <csv files>

Note: Your PDF filenames must match the format expected by helper.make_index(), where each PDF is in the same order as its correct ground-truth CSV.

Script Overview:

- `1_snapshot.py`
 - Creates a full index of a pdf and csv files in /data
 - Confirms that ever file loads correctly via pdf2image
 - Generates a snapshot DataFrame used by later scripts 

```
python scripts/1_snapshot.py
```

- `2_gpt_playground.py`
 -  Reads all PDFs in /data
 -  Converts each to Base 64
 -  Sends few-shot examples + target PDF to GPT
 -  Extracts:
  -  Newspaper name
  -  Publication date
 - Measures:
  - Runtime per file
  - Input/output tokens
  - Tokens per second
 - Compares results to ground truth CSVs
 - Saves a detailed log in /logs

Key inputs (need to be changed before running, located at top of script): 

- data_path: full path to data folder
- load_dotenv(env_path)
- model: which GPT model (can use any OpenAI model that supports responses.parse)
- example_1_path/example_2_path: which pdfs to include as part of few-shot examples
- role_prompt: what role the llm should take on
- content_prompt: what the llm is expected to do with the data provided

```
python scripts/2_gpt_playground.py
```

Output appears in:

logs/llm_call_YYYY-MM-DD.log



Changing the model: 

Running llama3.2-vision via ollama on yen:

https://ollama.com/library/llama3.2-vision

https://rcpedia.stanford.edu/blog/2025/05/12/running-ollama-on-stanford-computing-clusters/

How to run scripts:

- `1_snapshot.py`: run this first, creates snapshot of all files in your data folder and checks that they are able to load without issue
- `2_gpt_playground.py`: feeds pdfs into chatgpt, with newspaper name and date extracted. Compares to source of truth csv files.
- `3_llama_playground.py`: feeds pdfs into llama, with newspaper name and date extracted. Compares to source of truth csv files.
  - Note: ollama GPU server should be run first. Following that the script can be run from the cpu node.