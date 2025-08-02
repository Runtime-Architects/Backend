import os
import httpx
import yaml

def load_config(path: str = "../../config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

async def run_curl_search(query: str) -> str:
    """
    Sends a search query to Azure AI Search using httpx.
    Reads config from environment variables.
    Returns raw JSON string of the response.
    """
    search_index_name = os.getenv("POLICY_SEARCH_INDEX_NAME")
    search_api_key = os.getenv("POLICY_SEARCH_API_KEY")
    search_api_version = os.getenv("POLICY_SEARCH_API_VERSION")
    search_endpoint = os.getenv("POLICY_SEARCH_ENDPOINT")

    config = load_config()
    top_n = config.get("azure_search", {}).get("top", 5)

    if not all([search_index_name, search_api_key, search_api_version, search_endpoint]):
        raise ValueError("Missing one or more required environment variables.")

    url = f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key
    }

    payload = {
        "search": query,
        "top": top_n
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.text 
