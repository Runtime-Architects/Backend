import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque
import time
import unicodedata

def is_valid_href(href):
    return href and not href.startswith(('mailto:', 'tel:', 'javascript:', '#'))

def extract_clean_text(soup):
    # Removes scripts, styles, noscript
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    
    # Extract visible text
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned_text = '\n'.join(lines)
    
    # Normalize weird unicode
    return unicodedata.normalize('NFKC', cleaned_text)

def bfs_crawl_and_save_text(starting_url, output_file='crawled_text.txt', max_pages=50):
    visited_links = set()
    broken_links = set()
    queue = deque([starting_url])

    with open(output_file, 'w', encoding='utf-8') as f:
        while queue and len(visited_links) < max_pages:
            current_url = queue.popleft()

            if current_url in visited_links:
                continue

            print(f"Visiting: {current_url}")
            try:
                response = requests.get(current_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code != 200:
                    print(f"Broken: {current_url} - Status Code: {response.status_code}")
                    broken_links.add(current_url)
                    continue
            except requests.RequestException as e:
                print(f"Request failed: {current_url} - Error: {e}")
                broken_links.add(current_url)
                continue

            visited_links.add(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Gets clean readable text
            page_text = extract_clean_text(soup)

            # Writing content to file
            f.write(f"\n=== URL: {current_url} ===\n")
            f.write(page_text)
            f.write("\n\n")

            # Find and queue sub-links
            for tag in soup.find_all('a'):
                href = tag.get('href')
                if is_valid_href(href):
                    full_url = urljoin(current_url, href)
                    if full_url not in visited_links and full_url not in queue:
                        queue.append(full_url)

            time.sleep(1)  

    print(f"\nCrawling Complete. Clean text saved to '{output_file}'.")



start_url = "https://www.seai.ie/"
output_file_path = "seai_crawled_text.txt"
bfs_crawl_and_save_text(start_url, output_file_path)
