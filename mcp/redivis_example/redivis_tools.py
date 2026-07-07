import sys
import time
import redivis
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

MAX_LIMIT = 100
ORG_NAME = "StanfordGSBLibrary"
DATASET_NAME = "revelio_labs_workforce_data"

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

def sql_escape(value: str) -> str:
    return value.replace("'", "''")

def _validate_name(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()

TRANSIENT_MARKERS = ("timeout", "temporarily", "unavailable", "connection", "503", "502", "504")

def _is_transient(err: BaseException) -> bool:
    if isinstance(err, (ConnectionError, TimeoutError)):
        return True
    msg = str(err).lower()
    return any(m in msg for m in TRANSIENT_MARKERS)

def _run_query(query: str, max_attempts: int = 3) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            log("entered _run_query")
            log("starting redivis.query()")
            t0 = time.time()
            job = redivis.query(query)
            log(f"redivis.query() returned after {time.time() - t0:.2f}s")

            log("starting to_pandas_dataframe()")
            t1 = time.time()
            df = job.to_pandas_dataframe()
            log(f"to_pandas_dataframe() returned after {time.time() - t1:.2f}s")

            df = df.where(df.notnull(), None)

            return {
                "row_count": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "rows": df.astype(object).to_dict(orient="records"),
            }
        except Exception as e:
            if attempt < max_attempts and _is_transient(e):
                sleep = 0.5 * (2 ** (attempt - 1))
                log(f"transient redivis error (attempt {attempt}/{max_attempts}): {e} — retrying in {sleep:.1f}s")
                time.sleep(sleep)
                continue
            raise

def query_person(firstname: str, lastname: str, user_country: str = "United States", limit: int = 25) -> dict:
    firstname = _validate_name("firstname", firstname)
    lastname = _validate_name("lastname", lastname)
    user_country = _validate_name("user_country", user_country)

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    limit = min(limit, MAX_LIMIT)

    firstname_esc = sql_escape(firstname)
    lastname_esc = sql_escape(lastname)
    country_esc = sql_escape(user_country)

    # query is an exact match 

    query = f"""
    SELECT user_id, firstname, lastname, user_country, profile_linkedin_url,user_location, profile_summary
    FROM `{ORG_NAME}.{DATASET_NAME}.individual_user`
    WHERE firstname = '{firstname_esc}'
      AND lastname = '{lastname_esc}'
      AND user_country = '{country_esc}'
    LIMIT {int(limit)}
    """
    log("about to run query_person")
    log(query)
    return _run_query(query)

def get_user_by_id(user_id: int, limit: int = 25) -> dict:
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise ValueError("user_id must be an integer")

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    # For record linkage, 10 is usually plenty.
    limit = min(limit, 10)

    query = f"""
    SELECT
      user_id,
      position_id,
      company_raw,
      company_cleaned,
      location_raw,
      country,
      state,
      metro_area,
      startdate,
      enddate,
      title_raw,
      title_translated,
      seniority
    FROM `{ORG_NAME}.{DATASET_NAME}.individual_position`
    WHERE user_id = {user_id}
    LIMIT {limit}
    """

    log("about to run get_user_by_id")
    log(query)
    return _run_query(query)

def scrape_website(url: str, max_chars: int = 3000) -> dict:
    t0 = time.time()
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        log(f"scrape_website: timeout after {time.time()-t0:.2f}s — {url}")
        return {"error": "request timed out", "url": url, "content": ""}
    except requests.exceptions.HTTPError as e:
        log(f"scrape_website: HTTP {e.response.status_code} — {url}")
        return {"error": f"HTTP {e.response.status_code}", "url": url, "content": ""}
    except Exception as e:
        log(f"scrape_website: unexpected error — {e}")
        return {"error": str(e), "url": url, "content": ""}

    soup = BeautifulSoup(response.text, "html.parser")

    # Strip noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Clean up whitespace
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines()]
    clean = "\n".join(l for l in lines if l)

    truncated = len(clean) > max_chars
    log(f"scrape_website: got {len(clean)} chars in {time.time()-t0:.2f}s truncated={truncated}")
    return {
        "url": url,
        "content": clean[:max_chars],
        "truncated": truncated,
        "chars_returned": min(len(clean), max_chars),
    }

