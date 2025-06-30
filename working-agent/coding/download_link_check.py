# filename: download_link_check.py
import subprocess
import sys

# Install the bs4 package if not available
try:
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup

import requests

def print_download_links(url):
    print(f"Checking for downloadable emission datasets at: {url}")
    resp = requests.get(url, timeout=30)
    if resp.ok:
        soup = BeautifulSoup(resp.text, "html.parser")
        found = False
        for link in soup.find_all("a", href=True):
            href = link['href']
            text = link.get_text(strip=True)
            if href.lower().endswith((".csv", ".xlsx", ".xls", ".ods")) or "download" in href.lower():
                print(f"Possible download link: {text or href} -> {href}")
                found = True
        if not found:
            print("No direct CSV/Excel download links detected on this page.")
    else:
        print("Failed to fetch the page.")

print_download_links("https://www.epa.ie/our-services/monitoring--assessment/climate-change/ghg/latest-emissions-data/")
