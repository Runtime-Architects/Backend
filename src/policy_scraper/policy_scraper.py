import os
import json
import time
import asyncio
import requests
import hashlib
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import textwrap


def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and not any(
        url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
    )


def get_all_links(base_url, soup):
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = urljoin(base_url, a_tag['href'])
        if is_valid_url(href):
            links.add(href)
    return links


def extract_metadata_from_soup(soup, url):
    metadata = {}

    title_tag = soup.find("title")
    metadata['title'] = title_tag.get_text().strip() if title_tag else ""

    meta_desc = soup.find("meta", attrs={"name": "description"})
    metadata['description'] = meta_desc.get("content", "").strip() if meta_desc else ""

    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    metadata['keywords'] = meta_keywords.get("content", "").strip() if meta_keywords else ""

    headings = []
    for i in range(1, 7):
        for heading in soup.find_all(f"h{i}"):
            headings.append({
                "level": i,
                "text": heading.get_text().strip()
            })
    metadata['headings'] = headings

    main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main"))
    if main_content:
        metadata['main_content'] = main_content.get_text(separator=" ").strip()

    url_lower = url.lower()
    if "/news" in url_lower or "/events" in url_lower:
        metadata['category'] = "news"
    elif "/grant" in url_lower or "/funding" in url_lower:
        metadata['category'] = "grants"
    elif "/business" in url_lower:
        metadata['category'] = "business"
    elif "/home" in url_lower or "/residential" in url_lower:
        metadata['category'] = "residential"
    elif "/data" in url_lower or "/research" in url_lower:
        metadata['category'] = "research"
    elif "/about" in url_lower:
        metadata['category'] = "about"
    else:
        metadata['category'] = "general"

    services = []
    service_keywords = [
        "energy upgrade", "BER", "building energy rating", "heat pump", "solar",
        "insulation", "grant", "loan", "electric vehicle", "renewable energy",
        "energy efficiency", "retrofit", "sustainable energy", "climate action"
    ]
    full_text = soup.get_text().lower()
    for keyword in service_keywords:
        if keyword in full_text:
            services.append(keyword)
    metadata['services'] = list(set(services))

    date_patterns = [
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        r'\b(\d{4})-(\d{2})-(\d{2})\b'
    ]
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, soup.get_text(), re.IGNORECASE)
        dates_found.extend(matches)
    metadata['dates_mentioned'] = dates_found[:5]

    return metadata


def clean_text_for_pdf(text):
    """Clean and format text for PDF generation"""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]{}\'\"\/\n]', '', text)
    
    return text.strip()


def create_pdf_from_content(document, output_dir):
    """Create a PDF file from the document content"""
    try:
        # Create filename from URL
        url_hash = hashlib.md5(document['url'].encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^\w\s-]', '', document['title'])[:50]
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{document['category']}-{safe_title}-{url_hash}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4, 
                               rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Build PDF content
        story = []
        
        # Title
        if document['title']:
            story.append(Paragraph(clean_text_for_pdf(document['title']), title_style))
            story.append(Spacer(1, 12))
        
        # URL
        story.append(Paragraph(f"<b>URL:</b> {document['url']}", normal_style))
        story.append(Spacer(1, 12))
        
        # Description
        if document['description']:
            story.append(Paragraph("<b>Description:</b>", heading_style))
            story.append(Paragraph(clean_text_for_pdf(document['description']), normal_style))
            story.append(Spacer(1, 12))
        
        # Category and Services
        story.append(Paragraph(f"<b>Category:</b> {document['category']}", normal_style))
        if document['services']:
            story.append(Paragraph(f"<b>Services:</b> {', '.join(document['services'])}", normal_style))
        story.append(Spacer(1, 12))
        
        # Keywords
        if document['keywords']:
            story.append(Paragraph(f"<b>Keywords:</b> {', '.join(document['keywords'])}", normal_style))
            story.append(Spacer(1, 12))
        
        # Headings
        if document['headings']:
            story.append(Paragraph("<b>Page Headings:</b>", heading_style))
            for heading in document['headings'][:10]:  # Limit to first 10 headings
                story.append(Paragraph(f"• {clean_text_for_pdf(heading)}", normal_style))
            story.append(Spacer(1, 12))
        
        # Main content
        story.append(Paragraph("<b>Content:</b>", heading_style))
        
        # Split content into paragraphs and clean
        content_paragraphs = document['content'].split('\n')
        for para in content_paragraphs:
            clean_para = clean_text_for_pdf(para)
            if clean_para and len(clean_para) > 10:  # Only include substantial paragraphs
                # Wrap long paragraphs
                if len(clean_para) > 500:
                    wrapped_lines = textwrap.wrap(clean_para, width=80)
                    for line in wrapped_lines:
                        story.append(Paragraph(line, normal_style))
                else:
                    story.append(Paragraph(clean_para, normal_style))
                story.append(Spacer(1, 6))
        
        # Metadata footer
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Metadata:</b>", heading_style))
        story.append(Paragraph(f"Word Count: {document['wordCount']}", normal_style))
        story.append(Paragraph(f"Last Modified: {document['lastModified']}", normal_style))
        story.append(Paragraph(f"Domain: {document['domain']}", normal_style))
        
        # Build PDF
        doc.build(story)
        
        return filepath
        
    except Exception as e:
        print(f"Error creating PDF for {document['url']}: {e}")
        return None


def extract_optimized_content(html_content, url):
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "aside"]):
        tag.decompose()

    metadata = extract_metadata_from_soup(soup, url)

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    searchable_text = " ".join([
        metadata.get('title', ''),
        metadata.get('description', ''),
        metadata.get('keywords', ''),
        " ".join([h['text'] for h in metadata.get('headings', [])]),
        " ".join(metadata.get('services', [])),
        clean_text[:2000]
    ]).strip()

    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    doc_id = f"seai-{metadata['category']}-{url_hash}"

    document = {
        "id": doc_id,
        "url": url,
        "title": metadata.get('title', ''),
        "description": metadata.get('description', ''),
        "content": clean_text,
        "searchableText": searchable_text,
        "category": metadata.get('category', 'general'),
        "services": metadata.get('services', []),
        "headings": [h['text'] for h in metadata.get('headings', [])],
        "keywords": metadata.get('keywords', '').split(',') if metadata.get('keywords') else [],
        "organization": "SEAI",
        "contentType": "webpage",
        "pagelanguage": "en",
        "lastModified": datetime.utcnow().isoformat() + "Z",
        "wordCount": len(clean_text.split()),
        "datesMentioned": metadata.get('dates_mentioned', []),
        "domain": urlparse(url).netloc
    }

    return document


def bfs_crawl_and_save_pdf(starting_url, output_dir, max_pages=50):
    """Crawl websites and save each page as a PDF"""
    visited = set()
    queue = deque([starting_url])
    crawled_data = []
    count = 0

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting crawl from {starting_url}")
    print(f"Target: {max_pages} pages")
    print(f"Output directory: {output_dir}")

    while queue and count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        try:
            print(f"Crawling ({count + 1}/{max_pages}): {url}")
            response = requests.get(url, timeout=10)
            if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
                continue

            document = extract_optimized_content(response.text, url)
            crawled_data.append(document)
            visited.add(url)
            
            # Create PDF for this page
            pdf_path = create_pdf_from_content(document, output_dir)
            if pdf_path:
                print(f"  → PDF saved: {os.path.basename(pdf_path)}")
            
            count += 1

            # Find more links to crawl
            soup = BeautifulSoup(response.text, "html.parser")
            links = get_all_links(url, soup)
            for link in links:
                if link not in visited and link.startswith(starting_url):
                    queue.append(link)

            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"Error crawling {url}: {e}")
            continue

    # Save summary JSON file
    summary_file = os.path.join(output_dir, "crawl_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=2)

    print(f"\nCrawling completed. {len(crawled_data)} pages processed")
    print(f"PDFs saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

    # Print statistics
    categories = {}
    total_services = set()
    for doc in crawled_data:
        cat = doc.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        total_services.update(doc.get('services', []))

    print(f"\nCrawl Summary:")
    print(f"Categories found: {dict(categories)}")
    print(f"Unique services identified: {len(total_services)}")
    print(f"Average word count: {sum(doc.get('wordCount', 0) for doc in crawled_data) / len(crawled_data):.0f}")


# Example usage
if __name__ == "__main__":
    # Install required packages first:
    # pip install reportlab beautifulsoup4 requests
    
    bfs_crawl_and_save_pdf('https://www.seai.ie/', 'seai_pdfs', max_pages=200)