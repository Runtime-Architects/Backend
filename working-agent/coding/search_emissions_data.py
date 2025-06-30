# filename: search_emissions_data.py
import subprocess
import sys

# Install necessary packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["duckduckgo-search", "requests"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

from duckduckgo_search import DDGS

def find_emissions_data():
    query = "Ireland CO2 emissions daily or monthly 2023-2024 CSV"
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=10):
            results.append({'title': r['title'], 'href': r['href']})
    print("Search results for Ireland CO2 emissions data 2023-2024:")
    for idx, result in enumerate(results):
        print(f"{idx+1}. {result['title']} - {result['href']}")

find_emissions_data()
