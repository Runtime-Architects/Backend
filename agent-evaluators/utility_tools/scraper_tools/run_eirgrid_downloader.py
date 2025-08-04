"""
EirGrid Data Downloader CLI - Updated for Organized File Structure
Professional command-line interface for downloading energy metrics data
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import time

from .unified_downloader import UnifiedEirGridDownloader


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


def print_summary(results: Dict[str, Any], total_time: float, stats: Dict[str, int], file_structure: Dict[str, str]):
    """Print professional summary of download results with new file structure info"""
    
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
    
    # Detailed results
    if failed > 0:
        logger.info("")
        logger.info("FAILED DOWNLOADS:")
        for area, result in results.items():
            if not result['success']:
                logger.error(f"  {area}: {result['error']} (method: {result['method']})")
    
    # Files saved with new organized structure
    logger.info("")
    logger.info("FILES SAVED (ORGANIZED STRUCTURE):")
    for area, file_path in file_structure.items():
        if file_path:
            try:
                file_size = Path(file_path).stat().st_size / 1024
                # Show relative path from data directory for clarity
                relative_path = Path(file_path).relative_to(Path("data"))
                logger.info(f"  {area}: data/{relative_path} ({file_size:.1f} KB)")
            except:
                logger.info(f"  {area}: {file_path}")
    
    # Show directory structure overview
    logger.info("")
    logger.info("DIRECTORY STRUCTURE:")
    data_dir = Path("data")
    if data_dir.exists():
        for metric_dir in sorted(data_dir.iterdir()):
            if metric_dir.is_dir():
                file_count = len(list(metric_dir.glob("*.json")))
                logger.info(f"  data/{metric_dir.name}/ ({file_count} files)")


def list_available_data_command(data_dir: str = None):
    """Command to list all available data files"""
    downloader = UnifiedEirGridDownloader(data_dir=data_dir)
    available = downloader.list_available_data()
    
    if not available:
        print("No data files found.")
        return
    
    print("\n📁 AVAILABLE DATA FILES:")
    print("=" * 50)
    
    for area, files in available.items():
        print(f"\n🔹 {area.upper()}:")
        for file_info in files:
            print(f"   📄 {file_info}")
    
    print(f"\n💡 Files are organized in: data/{{metric}}/{{metric}}_{{start-date}}_{{end-date}}.json")


def main():
    """Main CLI function with updated file structure support"""
    
    parser = argparse.ArgumentParser(
        description="Download EirGrid energy metrics data with organized file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all areas for today (organized structure)
  python run_eirgrid_downloader.py
  
  # Download specific areas with forecast data
  python run_eirgrid_downloader.py --areas co2_intensity,wind_generation --forecast
  
  # Download last week's data for Ireland only
  python run_eirgrid_downloader.py --duration week --region roi
  
  # Download specific date range
  python run_eirgrid_downloader.py --start 2025-06-01 --end 2025-06-15
  
  # List all available data files
  python run_eirgrid_downloader.py --list-data
  
  # Force CSV download method (more reliable for some areas)
  python run_eirgrid_downloader.py --areas co2_emissions --force-scraping

File Organization:
  Data is now organized as: data/{metric}/{metric}_{start-date}_{end-date}.json
  Examples:
    data/co2_intensity/co2_intensity_2025-06-23_2025-06-23.json
    data/wind_generation/wind_generation_all_2025-06-20_2025-06-26.json
    data/demand/demand_roi_2025-06-01_2025-06-30.json

Available areas:
  co2_emissions       - CO2 Emissions (tCO2/hr)
  co2_intensity       - CO2 Intensity (gCO2/kWh)
  snsp                - System Non-Synchronous Penetration (%)
  wind_generation     - Wind Generation (MW)
  solar_generation    - Solar Generation (MW)
  total_generation    - Total Generation (MW)
  demand              - System Demand (MW)
  fuel_mix            - Fuel Mix breakdown (MWh)
  frequency           - System Frequency (Hz)
  interconnection     - Interconnection flows (MW)

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
                       help='Include forecast data (uses CSV download method)')
    
    parser.add_argument('--force-scraping',
                       action='store_true',
                       help='Force use of CSV download method (more reliable)')
    
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
    
    # List data command
    parser.add_argument('--list-data', '-l',
                       action='store_true',
                       help='List all available data files and exit')
    
    # Logging options
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--log-file',
                       help='Log to file in addition to console')
    
    args = parser.parse_args()
    
    # Set up logging ONCE at the beginning
    setup_logging(args.debug, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle list data command
        if args.list_data:
            list_available_data_command(args.output_dir)
            return 0
        
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
        logger.info("EIRGRID DATA DOWNLOADER - ORGANIZED STRUCTURE")
        logger.info("=" * 60)
        logger.info(f"Areas: {', '.join(areas)}")
        logger.info(f"Date range: {date_from} to {date_to}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Include forecast: {args.forecast}")
        logger.info(f"Force CSV download: {args.force_scraping}")
        logger.info(f"Update existing files: {not args.no_update}")
        
        if args.output_dir:
            logger.info(f"Output directory: {args.output_dir}")
        
        logger.info(f"File structure: data/{{metric}}/{{metric}}_{date_from}_{date_to}.json")
        logger.info("=" * 60)
        
        # Create downloader
        headless = not args.show_browser
        downloader = UnifiedEirGridDownloader(data_dir=args.output_dir, headless=headless)
        
        # Process each area
        start_time = time.time()
        results = {}
        file_structure = {}
        
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
                # Save data with new organized structure
                file_path = downloader.save_data(
                    result['data'],
                    area,
                    args.region,
                    date_from,
                    date_to,
                    update_existing=not args.no_update
                )
                result['file_path'] = file_path
                file_structure[area] = file_path
                
                # Log success with file location
                relative_path = Path(file_path).relative_to(Path("data")) if "data" in file_path else Path(file_path).name
                logger.info(f"  ✅ SUCCESS: {area} downloaded via {result['method']} → data/{relative_path}")
            else:
                logger.error(f"  ❌ FAILED: {area} - {result['error']}")
                file_structure[area] = None
            
            results[area] = result
            
            # Small delay between areas to be respectful
            if i < len(areas):
                time.sleep(1)
        
        total_time = time.time() - start_time
        
        # Print summary with file structure info
        print_summary(results, total_time, downloader.get_statistics(), file_structure)
        
        # Show available data after download
        logger.info("")
        logger.info("📊 DATA ORGANIZATION SUMMARY:")
        available_data = downloader.list_available_data()
        for area, files in available_data.items():
            if area in areas:  # Only show areas we just processed
                logger.info(f"  {area}: {len(files)} file(s) available")
        
        # Clean up temporary files
        downloader.cleanup()
        
        # Return appropriate exit code
        successful = sum(1 for r in results.values() if r['success'])
        if successful == len(areas):
            logger.info("🎉 All downloads completed successfully!")
            logger.info(f"💾 Data organized in: data/{{metric}}/{{metric}}_{date_from}_{date_to}.json format")
            return 0
        elif successful > 0:
            logger.warning("⚠️  Some downloads failed. Check the summary above.")
            return 2
        else:
            logger.error("💥 All downloads failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️  Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())