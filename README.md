# LLM_benchmarks

Start by cloning repo:

```
git clone https://github.com/gsbdarc/LLM_benchmarks
```

Make a venv (assuming you're using Yen):

```
/usr/bin/python3  -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

rcpedia: how to make this into a kernel

How to run scripts:

- `1_snapshot.py`: run this first, creates daily snapshot of all files in your data folder and checks that they are able to load without issue
- `2_llm_playground.py`: feeds pdfs into LLMs, with newspaper name and date extracted. Compares to source of truth csv files. 