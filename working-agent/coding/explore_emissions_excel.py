# filename: explore_emissions_excel.py
import subprocess
import sys

# Install pandas and openpyxl if needed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["pandas", "openpyxl"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import pandas as pd

# Download the file
url = "https://www.epa.ie/media/epa-2020/monitoring-amp-assessment/climate-change/greenhouse-gases/Change-2022-2023-table_MAY2025.xlsx"
xlsx_fname = "Change-2022-2023-table_MAY2025.xlsx"
pd.read_excel(url, engine='openpyxl')  # This line ensures dependencies are initialized

try:
    df = pd.read_excel(url, engine='openpyxl', sheet_name=None)
    print(f"Sheet names in the Excel file: {list(df.keys())}")
    # Print first 10 rows of each sheet for inspection
    for sheet, data in df.items():
        print(f"\n=== Sheet: {sheet} ===")
        print(data.head(10).to_string())
except Exception as e:
    print(f"Error reading Excel file: {e}")
