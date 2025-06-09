#!/usr/bin/env python
import sys
import os
from pathlib import Path
import time

# Add the project to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from datetime import datetime
import logging

# better logging - prevents double output from Scrapy
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Override any existing logging configuration
)

# Disable Scrapy's duplicate logging
logging.getLogger('scrapy').setLevel(logging.WARNING)
logging.getLogger('scrapy.utils.log').setLevel(logging.WARNING)
logging.getLogger('scrapy.middleware').setLevel(logging.WARNING)
logging.getLogger('scrapy.extensions').setLevel(logging.WARNING)
logging.getLogger('scrapy.core.engine').setLevel(logging.WARNING)
logging.getLogger('scrapy.crawler').setLevel(logging.WARNING)

def main():
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        date_from = sys.argv[1]
        date_to = sys.argv[2] if len(sys.argv) > 2 else date_from
        region = sys.argv[3] if len(sys.argv) > 3 else 'all'
    else:
        # Default to today
        date_from = date_to = datetime.now().strftime('%d-%b-%Y')
        region = 'all'
    
    logger.info("=" * 50)
    logger.info("Starting CO2 Data Scraper")
    logger.info(f"Date From: {date_from}")
    logger.info(f"Date To: {date_to}")
    logger.info(f"Region: {region}")
    logger.info("=" * 50)
    
    # Create necessary directories
    Path('cache').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Get project settings
    settings = get_project_settings()
    
    # Add some debugging settings if needed
    if '--debug' in sys.argv:
        settings.set('LOG_LEVEL', 'DEBUG')
        logger.info("Debug mode enabled")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Create process
        process = CrawlerProcess(settings)
        
        # Run spider
        logger.info("Starting spider...")
        process.crawl('co2_intensity', date_from=date_from, date_to=date_to, region=region)
        
        logger.info("Spider launched, waiting for completion...")
        process.start()
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"\nScraping completed in {elapsed_time:.2f} seconds!")
    
    # Show results
    show_results(logger)
    
    return 0

def show_results(logger):
    """Display scraping results"""
    
    # Check cache directory
    cache_dir = Path('cache')
    if cache_dir.exists():
        files = list(cache_dir.glob('*.json'))
        if files:
            logger.info(f"\nFound {len(files)} cached files:")
            for file in sorted(files, key=lambda x: x.stat().st_mtime)[-5:]:  # Show last 5 files
                size_kb = file.stat().st_size / 1024
                mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"  - {file.name} ({size_kb:.1f} KB) - {mtime}")
        else:
            logger.warning("No cache files found")
    
    # Check data directory
    data_dir = Path('data')
    if data_dir.exists():
        files = list(data_dir.glob('*.json'))
        if files:
            logger.info(f"\nFound {len(files)} data export files:")
            for file in sorted(files, key=lambda x: x.stat().st_mtime)[-5:]:  # Show last 5 files
                size_kb = file.stat().st_size / 1024
                mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"  - {file.name} ({size_kb:.1f} KB) - {mtime}")
                
                # Show preview of latest file
                if file == sorted(files, key=lambda x: x.stat().st_mtime)[-1]:
                    try:
                        import json
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        time_series_count = len(data.get('time_series', []))
                        logger.info(f"    └─ Contains {time_series_count} time series data points")
                        
                        if 'metadata' in data and 'region' in data['metadata']:
                            region = data['metadata']['region']
                            date_range = data['metadata'].get('date_range', {})
                            logger.info(f"    └─ Region: {region}, Date: {date_range.get('from', 'Unknown')}")
                            
                    except Exception as e:
                        logger.warning(f"    └─ Could not preview file: {e}")
        else:
            logger.warning("No data export files found")
    
    # Performance tips
    logger.info("\n" + "=" * 50)
    logger.info("Tips for better performance:")
    logger.info("- Use specific date ranges instead of 'all'")
    logger.info("- Run during off-peak hours for faster response")
    logger.info("- Check network connection if scraping is slow")
    logger.info("=" * 50)

def validate_date_format(date_str):
    """Validate date format"""
    try:
        datetime.strptime(date_str, '%d-%b-%Y')
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    # Basic argument validation
    if len(sys.argv) > 1:
        date_from = sys.argv[1]
        if not validate_date_format(date_from):
            print(f"Error: Invalid date format '{date_from}'. Use format: DD-Mon-YYYY (e.g., 08-Jun-2025)")
            print("Usage: python run_scraper.py [date_from] [date_to] [region]")
            print("       python run_scraper.py 08-Jun-2025 08-Jun-2025 all")
            sys.exit(1)
        
        if len(sys.argv) > 2:
            date_to = sys.argv[2]
            if not validate_date_format(date_to):
                print(f"Error: Invalid date format '{date_to}'. Use format: DD-Mon-YYYY (e.g., 08-Jun-2025)")
                sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code)