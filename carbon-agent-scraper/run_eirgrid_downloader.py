#!/usr/bin/env python
"""
EirGrid Data Downloader CLI - UPDATED VERSION
Professional command-line interface for downloading energy metrics data
Now properly prioritizes API calls with CSV download as fallback
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import time

from unified_downloader import UnifiedEirGridDownloader


def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Set up professional logging configuration"""
    
    # Clear any existing handlers to prevent duplicates
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Base configuration
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=handlers,
        force=True
    )
    
    # Suppress verbose libraries
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('webdriver_manager').setLevel(logging.WARNING)


def validate_date(date_str: str) -> str:
    """Validate and return date in YYYY-MM-DD format"""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def get_available_areas() -> List[str]:
    """Get list of available data areas"""
    return [
        'co2_emissions',
        'co2_intensity',
        'snsp',
        'wind_generation',
        'solar_generation',
        'total_generation',
        'demand',
        'fuel_mix',
        'frequency',
        'interconnection'
    ]


def get_api_supported_areas() -> List[str]:
    """Get list of areas supported by API"""
    return [
        'co2_emissions',
        'co2_intensity',
        'snsp',
        'wind_generation',
        'solar_generation',
        'total_generation',
        'demand',
        'fuel_mix',
        'frequency',
        'interconnection'
    ]


def get_csv_supported_areas() -> List[str]:
    """Get list of areas supported by CSV download"""
    return [
        'co2_emissions',
        'co2_intensity',
        'wind_generation',
        'solar_generation',
        'demand'
    ]


def calculate_date_range(duration: str) -> Tuple[str, str]:
    """Calculate date range based on duration"""
    today = datetime.now()
    
    if duration == 'day':
        return today.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif duration == 'week':
        start = today - timedelta(days=6)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif duration == 'month':
        start = today - timedelta(days=29)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif duration == 'year':
        start = today - timedelta(days=364)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    else:
        raise ValueError(f"Invalid duration: {duration}")


def print_method_strategy(areas: List[str], include_forecast: bool, force_scraping: bool):
    """Print information about the download strategy that will be used"""
    
    logger = logging.getLogger(__name__)
    
    api_supported = get_api_supported_areas()
    csv_supported = get_csv_supported_areas()
    
    if force_scraping:
        logger.info("DOWNLOAD STRATEGY: Forced CSV download for all areas")
        return
    
    if include_forecast:
        logger.info("DOWNLOAD STRATEGY: CSV download required for forecast data")
        return
    
    logger.info("DOWNLOAD STRATEGY: API first, CSV fallback")
    
    for area in areas:
        if area in api_supported:
            if area in csv_supported:
                logger.info(f"  {area}: API primary, CSV fallback available")
            else:
                logger.info(f"  {area}: API only")
        elif area in csv_supported:
            logger.info(f"  {area}: CSV only")
        else:
            logger.warning(f"  {area}: Not supported by either method")


def print_summary(results: Dict[str, Any], total_time: float, stats: Dict[str, int]):
    """Print professional summary of download results"""
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    # Overall statistics
    total_areas = len(results)
    successful = sum(1 for r in results.values() if r['success'])
    failed = total_areas - successful
    
    logger.info(f"Total areas processed: {total_areas}")
    logger.info(f"Successful downloads: {successful}")
    logger.info(f"Failed downloads: {failed}")
    logger.info(f"Success rate: {(successful/total_areas*100):.1f}%")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Method statistics
    logger.info("")
    logger.info("METHOD STATISTICS:")
    logger.info(f"  API successes: {stats['api_success']}")
    logger.info(f"  API failures: {stats['api_fail']}")
    logger.info(f"  CSV download successes: {stats['csv_success']}")
    logger.info(f"  CSV download failures: {stats['csv_fail']}")
    
    # Method efficiency
    total_attempts = sum(stats.values())
    if total_attempts > 0:
        api_rate = (stats['api_success'] / total_attempts) * 100
        logger.info(f"  API success rate: {api_rate:.1f}%")
    
    # Detailed results by method
    api_areas = []
    csv_areas = []
    failed_areas = []
    
    for area, result in results.items():
        if result['success']:
            if result['method'] == 'api':
                api_areas.append(area)
            elif result['method'] == 'csv_download':
                csv_areas.append(area)
        else:
            failed_areas.append(area)
    
    if api_areas:
        logger.info(f"  API successful: {', '.join(api_areas)}")
    if csv_areas:
        logger.info(f"  CSV successful: {', '.join(csv_areas)}")
    
    # Failed downloads
    if failed_areas:
        logger.info("")
        logger.info("FAILED DOWNLOADS:")
        for area in failed_areas:
            result = results[area]
            logger.error(f"  {area}: {result['error']}")
    
    # Files saved
    logger.info("")
    logger.info("FILES SAVED:")
    for area, result in results.items():
        if result['success'] and 'file_path' in result:
            file_size = Path(result['file_path']).stat().st_size / 1024
            method_indicator = result['method'].upper()
            logger.info(f"  {area}: {result['file_path']} ({file_size:.1f} KB, via {method_indicator})")


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Download EirGrid energy metrics data (API first, CSV fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all areas for today (API first, CSV fallback)
  python run_eirgrid_downloader.py
  
  # Download specific areas (API first, CSV fallback)
  python run_eirgrid_downloader.py --areas co2_intensity,wind_generation
  
  # Download with forecast data (forces CSV method)
  python run_eirgrid_downloader.py --areas co2_intensity --forecast
  
  # Force CSV download method for all areas
  python run_eirgrid_downloader.py --areas co2_emissions --force-scraping
  
  # Download last week's data for Ireland only
  python run_eirgrid_downloader.py --duration week --region roi

Available areas:
  API + CSV Support:     co2_emissions, co2_intensity, wind_generation, 
                        solar_generation, demand
  API Only:             snsp, total_generation, fuel_mix, frequency, 
                        interconnection

Method Priority:
  1. API (fast, actual data only)
  2. CSV download (slower, includes forecast if requested)

Regions: all, roi (Republic of Ireland), ni (Northern Ireland)
        """
    )
    
    # Area selection
    parser.add_argument('--areas', '-a',
                       help='Comma-separated list of areas to download (default: all)',
                       default='all')
    
    # Date options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--duration', '-d',
                           choices=['day', 'week', 'month', 'year'],
                           help='Predefined duration to download')
    date_group.add_argument('--start', '-s',
                           type=validate_date,
                           help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end', '-e',
                       type=validate_date,
                       help='End date (YYYY-MM-DD)')
    
    # Region selection
    parser.add_argument('--region', '-r',
                       choices=['all', 'roi', 'ni'],
                       default='all',
                       help='Region to download (default: all)')
    
    # Options
    parser.add_argument('--forecast', '-f',
                       action='store_true',
                       help='Include forecast data (forces CSV download method)')
    
    parser.add_argument('--force-scraping',
                       action='store_true',
                       help='Force use of CSV download method for all areas')
    
    parser.add_argument('--no-update',
                       action='store_true',
                       help='Create new files instead of updating existing ones')
    
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for data files (default: ./data)')
    
    parser.add_argument('--headless',
                       action='store_true',
                       default=True,
                       help='Run browser in headless mode (default: True)')
    
    parser.add_argument('--show-browser',
                       action='store_true',
                       help='Show browser window (disables headless mode)')
    
    # Testing options
    parser.add_argument('--test-api',
                       action='store_true',
                       help='Test API connectivity for all areas and exit')
    
    # Logging options
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--log-file',
                       help='Log to file in addition to console')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Parse areas
        if args.areas.lower() == 'all':
            areas = get_available_areas()
        else:
            areas = [a.strip() for a in args.areas.split(',')]
            # Validate areas
            available = get_available_areas()
            invalid = [a for a in areas if a not in available]
            if invalid:
                logger.error(f"Invalid areas: {', '.join(invalid)}")
                logger.info(f"Available areas: {', '.join(available)}")
                return 1
        
        # Create downloader for testing if needed
        headless = not args.show_browser
        downloader = UnifiedEirGridDownloader(data_dir=args.output_dir, headless=headless)
        
        # Test API connectivity if requested
        if args.test_api:
            logger.info("Testing API connectivity...")
            api_areas = get_api_supported_areas()
            
            for area in api_areas:
                logger.info(f"Testing {area}...")
                is_working = downloader.test_api_connection(area)
                status = "✓ Working" if is_working else "✗ Failed"
                logger.info(f"  {area}: {status}")
            
            downloader.cleanup()
            return 0
        
        # Determine date range
        if args.duration:
            date_from, date_to = calculate_date_range(args.duration)
        elif args.start:
            date_from = args.start
            date_to = args.end if args.end else args.start
        else:
            # Default to today
            date_from = date_to = datetime.now().strftime('%Y-%m-%d')
        
        # Validate date range
        if datetime.strptime(date_from, '%Y-%m-%d') > datetime.strptime(date_to, '%Y-%m-%d'):
            logger.error("Start date must be before or equal to end date")
            return 1
        
        # Log configuration
        logger.info("=" * 60)
        logger.info("EIRGRID DATA DOWNLOADER")
        logger.info("=" * 60)
        logger.info(f"Areas: {', '.join(areas)}")
        logger.info(f"Date range: {date_from} to {date_to}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Include forecast: {args.forecast}")
        logger.info(f"Force CSV download: {args.force_scraping}")
        logger.info(f"Update existing files: {not args.no_update}")
        
        if args.output_dir:
            logger.info(f"Output directory: {args.output_dir}")
        
        logger.info("=" * 60)
        
        # Show download strategy
        print_method_strategy(areas, args.forecast, args.force_scraping)
        logger.info("=" * 60)
        
        # Process each area
        start_time = time.time()
        results = {}
        
        for i, area in enumerate(areas, 1):
            logger.info(f"Processing {area} ({i}/{len(areas)})...")
            
            result = downloader.download_area(
                area=area,
                region=args.region,
                date_from=date_from,
                date_to=date_to,
                include_forecast=args.forecast,
                force_scraping=args.force_scraping
            )
            
            if result['success']:
                # Save data
                file_path = downloader.save_data(
                    result['data'],
                    area,
                    args.region,
                    update_existing=not args.no_update
                )
                result['file_path'] = file_path
                logger.info(f"  ✓ SUCCESS: {area} downloaded via {result['method'].upper()}")
            else:
                logger.error(f"  ✗ FAILED: {area} - {result['error']}")
            
            results[area] = result
            
            # Small delay between areas to be respectful
            if i < len(areas):
                time.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # Print summary
        print_summary(results, total_time, downloader.get_statistics())
        
        # Clean up temporary files
        downloader.cleanup()
        
        # Return appropriate exit code
        successful = sum(1 for r in results.values() if r['success'])
        if successful == len(areas):
            logger.info("All downloads completed successfully!")
            return 0
        elif successful > 0:
            logger.warning("Some downloads failed. Check the summary above.")
            return 2
        else:
            logger.error("All downloads failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())