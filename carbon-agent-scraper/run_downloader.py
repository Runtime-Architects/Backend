#!/usr/bin/env python
"""
Main script to run CO2 data downloader
Usage: python run_downloader.py [duration] [date_from] [date_to] [region]
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from co2_downloader import CO2DataDownloader

def setup_logging(debug: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

def validate_date_format(date_str: str) -> bool:
    """Validate date format DD-Mon-YYYY"""
    try:
        datetime.strptime(date_str, '%d-%b-%Y')
        return True
    except ValueError:
        return False

def calculate_date_range(duration: str) -> tuple:
    """Calculate appropriate date range for duration"""
    today = datetime.now()
    
    if duration == "day":
        date_from = today.strftime('%d-%b-%Y')
        date_to = date_from
    elif duration == "week":
        # Last 7 days
        date_from = (today - timedelta(days=6)).strftime('%d-%b-%Y')
        date_to = today.strftime('%d-%b-%Y')
    elif duration == "month":
        # Last 30 days
        date_from = (today - timedelta(days=29)).strftime('%d-%b-%Y')
        date_to = today.strftime('%d-%b-%Y')
    else:
        raise ValueError(f"Invalid duration: {duration}")
    
    return date_from, date_to

def show_results(json_file: str, logger):
    """Display results summary"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        time_series = data.get('time_series', [])
        
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Region: {metadata.get('region', 'Unknown')}")
        logger.info(f"Total data points: {metadata.get('total_points', len(time_series))}")
        logger.info(f"Source file: {metadata.get('source_file', 'Unknown')}")
        
        if time_series:
            first_point = time_series[0]
            last_point = time_series[-1]
            logger.info(f"Time range: {first_point.get('time')} to {last_point.get('time')}")
            
            # Show some sample data
            logger.info("\nSample data points:")
            for i, point in enumerate(time_series[:3]):
                intensity = point.get('intensity', 'N/A')
                forecast = point.get('intensity_forecast')
                forecast_str = f", Forecast: {forecast}" if forecast else ""
                logger.info(f"  {point.get('time')}: {intensity} gCO2/kWh{forecast_str}")
            
            if len(time_series) > 6:
                logger.info("  ...")
                for point in time_series[-3:]:
                    intensity = point.get('intensity', 'N/A')
                    forecast = point.get('intensity_forecast')
                    forecast_str = f", Forecast: {forecast}" if forecast else ""
                    logger.info(f"  {point.get('time')}: {intensity} gCO2/kWh{forecast_str}")
        
        logger.info(f"\nData saved to: {json_file}")
        file_size = Path(json_file).stat().st_size / 1024
        logger.info(f"File size: {file_size:.1f} KB")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error displaying results: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download CO2 intensity data from Smart Grid Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_downloader.py                                    # Today's data
  python run_downloader.py day                               # Today's data
  python run_downloader.py week                              # Last week's data
  python run_downloader.py month                             # Last month's data
  python run_downloader.py day 08-Jun-2025                   # Specific day
  python run_downloader.py week 03-Jun-2025 09-Jun-2025      # Specific week
  python run_downloader.py day 08-Jun-2025 08-Jun-2025 roi   # Ireland only
  python run_downloader.py --no-merge day                    # Don't merge with existing

Duration options: day, week, month
Region options: all, roi, ni
Date format: DD-Mon-YYYY (e.g., 08-Jun-2025)
        """
    )
    
    parser.add_argument('duration', nargs='?', default='day',
                       choices=['day', 'week', 'month'],
                       help='Time duration (default: day)')
    
    parser.add_argument('date_from', nargs='?', 
                       help='Start date in DD-Mon-YYYY format (optional)')
    
    parser.add_argument('date_to', nargs='?',
                       help='End date in DD-Mon-YYYY format (optional)')
    
    parser.add_argument('region', nargs='?', default='all',
                       choices=['all', 'roi', 'ni'],
                       help='Region (default: all)')
    
    parser.add_argument('--no-merge', action='store_true',
                       help='Don\'t merge with existing data')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--visible', action='store_true',
                       help='Show browser window (for debugging)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate and process arguments
        duration = args.duration
        region = args.region
        merge_existing = not args.no_merge
        
        # Handle date arguments
        if args.date_from:
            if not validate_date_format(args.date_from):
                logger.error(f"Invalid date format: {args.date_from}. Use DD-Mon-YYYY (e.g., 08-Jun-2025)")
                return 1
            date_from = args.date_from
            
            if args.date_to:
                if not validate_date_format(args.date_to):
                    logger.error(f"Invalid date format: {args.date_to}. Use DD-Mon-YYYY (e.g., 08-Jun-2025)")
                    return 1
                date_to = args.date_to
            else:
                date_to = date_from
        else:
            # Calculate dates based on duration
            date_from, date_to = calculate_date_range(duration)
        
        # Log configuration
        logger.info("=" * 60)
        logger.info("CO2 DATA DOWNLOADER")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Date range: {date_from} to {date_to}")
        logger.info(f"Region: {region}")
        logger.info(f"Merge with existing: {merge_existing}")
        logger.info("=" * 60)
        
        # Create downloader
        downloader = CO2DataDownloader(headless=not args.visible)
        
        # Download and process
        logger.info("Starting download...")
        start_time = datetime.now()
        
        result = downloader.download_and_process(
            duration=duration,
            date_from=date_from,
            date_to=date_to,
            region=region,
            merge_existing=merge_existing
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        if result:
            logger.info(f"Download completed in {elapsed:.1f} seconds!")
            show_results(result, logger)
            return 0
        else:
            logger.error("Download failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)