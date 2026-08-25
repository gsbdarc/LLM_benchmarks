import os
import logging
import sys

# FastMCP is similar to FastAPI
#

from mcp.server.fastmcp import FastMCP
from LLM_benchmarks.mcp.redivis_example.redivis_tools import query_person, get_user_by_id, scrape_website

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))

mcp = FastMCP(
    "redivis-revelio-http",
    json_response=True,
    host=HOST,
    port=PORT,
)

def _error_response(msg: str) -> dict:
    return {
        "error": msg,
        "matched_rows": 0,
        "returned_preview_rows": 0,
        "columns": [],
        "rows": [],
    }

def _success_response(result: dict) -> dict:
    return {
        "matched_rows": int(result["row_count"]),
        "returned_preview_rows": min(len(result["rows"]), 5),
        "columns": result["columns"],
        "rows": result["rows"],
    }

@mcp.tool()
def search_person(
    firstname: str,
    lastname: str,
    user_country: str = "United States",
    limit: int = 40,
) -> dict:
    """
    Search Revelio individual_user records by first name, last name, and country.
    Use this to find candidate user_ids for record linkage.
    """
    try:
        result = query_person(firstname, lastname, user_country, limit)
    except ValueError as e:
        return _error_response(f"invalid input: {e}")
    except Exception as e:
        logging.exception("search_person failed")
        return _error_response(f"redivis error: {e}")
    return _success_response(result)

@mcp.tool()
def fetch_user_positions(user_id: int, limit: int = 5) -> dict:
    """
    Fetch compact position history for a candidate user_id.
    Returns linkage-relevant fields such as company, title, dates, and location.
    """
    try:
        result = get_user_by_id(user_id, limit)
    except ValueError as e:
        return _error_response(f"invalid input: {e}")
    except Exception as e:
        logging.exception("fetch_user_positions failed")
        return _error_response(f"redivis error: {e}")
    return _success_response(result)

@mcp.tool()
def fetch_website(website_address: str, max_chars: int = 3000) -> dict:
    """
    Scrape a website using BeautifulSoup and return cleaned text content.
    Use this when the user provides a URL and needs the page content read.
    Truncates output to max_chars characters.
    """
    try:
        result = scrape_website(website_address, max_chars)
    except ValueError as e:
        return {"error": f"invalid input: {e}", "url": website_address, "content": ""}
    except Exception as e:
        logging.exception("fetch_website failed")
        return {"error": str(e), "url": website_address, "content": ""}
    return result


if __name__ == "__main__":
    logging.info("Starting MCP server on http://%s:%s/mcp", HOST, PORT)
    mcp.run(transport="streamable-http")