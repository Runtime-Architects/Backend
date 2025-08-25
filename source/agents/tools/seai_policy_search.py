"""
seai_policy_search.py

This module consists of the HTTP request made to Azure AI search to fetch SEAI Policy Data
"""

import os
import httpx
import yaml


# def load_config(path: str = "../../config.yaml") -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


async def run_curl_search(query: str) -> str:
    """Run a search query against a specified search index using an asynchronous HTTP request.

    This function constructs a search request to a configured search endpoint using the provided query string. It retrieves necessary configuration values from environment variables and raises an error if any are missing. The function sends a POST request to the search API and returns the response as a string.

    Args:
        query (str): The search query string to be executed.

    Returns:
        str: The response from the search API as a string.

    Raises:
        ValueError: If any required environment variables are missing.
        httpx.HTTPStatusError: If the HTTP request returns an unsuccessful status code.
    """
    search_index_name = os.getenv("POLICY_SEARCH_INDEX_NAME")
    search_api_key = os.getenv("POLICY_SEARCH_API_KEY")
    search_api_version = os.getenv("POLICY_SEARCH_API_VERSION")
    search_endpoint = os.getenv("POLICY_SEARCH_ENDPOINT")

    # config = load_config()
    # top_n = config.get("azure_search", {}).get("top", 5)
    top_n = 5

    if not all(
        [search_index_name, search_api_key, search_api_version, search_endpoint]
    ):
        raise ValueError("Missing one or more required environment variables.")

    url = f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}"

    headers = {"Content-Type": "application/json", "api-key": search_api_key}

    payload = {"search": query, "top": top_n}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.text
