Retired scripts from LLM Benchmarking project.

Script Overview:

```
python scripts/old/gpt.py
```

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

Findings:
- GPT Models were relatively good at extracting both the name of the newspaper and using reasoning to get the date of the TV guide.
- GPT-5 with reasoning turned "high" had the best results with 97% name accuracy and 69% date accuracy.

```
python scripts/old/llama.py
```

Running llama3.2-vision via ollama on yen:

https://ollama.com/library/llama3.2-vision

https://rcpedia.stanford.edu/blog/2025/05/12/running-ollama-on-stanford-computing-clusters/

-  Reads all PDFs in /data
 -  Converts each to Base 64
 -  Sends zero shot example and target PDF to llama3.2-vision:11b
 -  Extracts:
  -  Newspaper name
  -  Publication date
 - Compares results to ground truth CSVs
 - Saves a detailed log in /logs

Key inputs (need to be changed before running, located at top of script): 
- data_path: full path to data folder
- load_dotenv(env_path)
- role_prompt: what role the llm should take on
- content_prompt: what the llm is expected to do with the data provided

Findings:
- LLAMA struggled with both name extraction and date reasoning, scoring 0% and 11% respectively.

How to run scripts:
- `gpt_playground.py`: feeds pdfs into GPT models, with newspaper name and date extracted. Compares to source of truth csv files.
- `llama_playground.py`: feeds pdfs into llama, with newspaper name and date extracted. Compares to source of truth csv files.
  - Note: ollama GPU server should be run first. Following that the script can be run from the cpu node.