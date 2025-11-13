# LLM_benchmarks

Start by cloning repo:

```
git clone https://github.com/gsbdarc/LLM_benchmarks
```

Make a venv:

```
/usr/bin/python3  -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Making a kernel:

```
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv
```

Run ollama on yen:

https://rcpedia.stanford.edu/blog/2025/05/12/running-ollama-on-stanford-computing-clusters/

How to run scripts:

- `1_snapshot.py`: run this first, creates snapshot of all files in your data folder and checks that they are able to load without issue
- `2_gpt_playground.py`: feeds pdfs into chatgpt, with newspaper name and date extracted. Compares to source of truth csv files.
- `3_llama_playground.py`: feeds pdfs into llama, with newspaper name and date extracted. Compares to source of truth csv files. 