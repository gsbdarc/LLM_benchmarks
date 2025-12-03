# LLM_benchmarks

Purpose: Evaluating open and closed vision/vison supported LLMs data extraction, table understanding, and reasoning capabilties on PDF TV 

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
    data/
        <pdf files>
        <csv files>

Note: Your PDF filenames must match the format expected by helper.make_index(), where each PDF is in the same order as its correct ground-truth CSV.

Script Overview:

- `1_snapshot.py`
 - Creates a full index of a pdf and csv files in /data
 - Confirms that ever file loads correctly via pdf2image
 - Generates a snapshot DataFrame used by later scripts 

How to run scripts:

- `1_snapshot.py`: run this first, creates snapshot of all files in your data folder and checks that they are able to load without issue
