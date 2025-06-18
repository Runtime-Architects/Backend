"""
Quick Test Script - Validates Unified Download Logic
For comprehensive testing, use comprehensive_test.py instead
"""

import sys
import logging
from datetime import datetime, timedelta
from unified_downloader import UnifiedEirGridDownloader


def setup_logging():
    """Set up simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def quick_validation_test():
    """Quick validation of the unified download prioritization logic"""
    
    logger = logging.getLogger(__name__)
    
    # Use yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    downloader = UnifiedEirGridDownloader(headless=True)
    
    logger.info("🚀 QUICK VALIDATION TEST")
    logger.info("=" * 50)
    logger.info("Testing that API is prioritized over CSV download")
    logger.info("=" * 50)
    
    # Test 1: Normal download (should prefer API)
    logger.info("Test 1: Normal download (should use API first)")
    result1 = downloader.download_area(
        area='co2_intensity',
        region='all',
        date_from=yesterday,
        date_to=yesterday,
        include_forecast=False,
        force_scraping=False
    )
    
    if result1['success']:
        method = result1['method']
        points = len(result1['data']['time_series'])
        logger.info(f"✅ SUCCESS: Used {method.upper()} method, {points} data points")
        
        if method == 'api':
            logger.info("   ✓ Correctly prioritized API method")
        else:
            logger.warning(f"   ⚠️  Used {method} instead of API (API might have failed)")
    else:
        logger.error(f"❌ FAILED: {result1['error']}")
    
    logger.info("-" * 30)
    
    # Test 2: Forecast requested (should use CSV)
    logger.info("Test 2: With forecast flag (should use CSV)")
    result2 = downloader.download_area(
        area='co2_intensity',
        region='all',
        date_from=yesterday,
        date_to=yesterday,
        include_forecast=True,
        force_scraping=False
    )
    
    if result2['success']:
        method = result2['method']
        points = len(result2['data']['time_series'])
        actual = sum(1 for p in result2['data']['time_series'] if not p.get('is_forecast', False))
        forecast = points - actual
        logger.info(f"✅ SUCCESS: Used {method.upper()} method")
        logger.info(f"   Data: {points} total ({actual} actual, {forecast} forecast)")
        
        if method == 'csv_download':
            logger.info("   ✓ Correctly used CSV for forecast data")
        else:
            logger.warning(f"   ⚠️  Expected CSV method but used {method}")
    else:
        logger.error(f"❌ FAILED: {result2['error']}")
    
    logger.info("-" * 30)
    
    # Test 3: Force scraping (should use CSV)
    logger.info("Test 3: Force scraping flag (should use CSV)")
    result3 = downloader.download_area(
        area='co2_intensity',
        region='all',
        date_from=yesterday,
        date_to=yesterday,
        include_forecast=False,
        force_scraping=True
    )
    
    if result3['success']:
        method = result3['method']
        points = len(result3['data']['time_series'])
        logger.info(f"✅ SUCCESS: Used {method.upper()} method, {points} data points")
        
        if method == 'csv_download':
            logger.info("   ✓ Correctly forced CSV method")
        else:
            logger.warning(f"   ⚠️  Expected CSV method but used {method}")
    else:
        logger.error(f"❌ FAILED: {result3['error']}")
    
    # Show statistics
    stats = downloader.get_statistics()
    logger.info("=" * 50)
    logger.info("METHOD USAGE STATISTICS:")
    logger.info(f"API successes: {stats['api_success']}")
    logger.info(f"API failures: {stats['api_fail']}")
    logger.info(f"CSV successes: {stats['csv_success']}")
    logger.info(f"CSV failures: {stats['csv_fail']}")
    
    total_attempts = sum(stats.values())
    if total_attempts > 0:
        api_rate = (stats['api_success'] / total_attempts) * 100
        logger.info(f"API success rate: {api_rate:.1f}%")
    
    # Determine if prioritization is working
    logger.info("=" * 50)
    
    # Check if API was used when it should be
    api_used_appropriately = (
        result1['success'] and result1.get('method') == 'api'
    ) or (
        result1['success'] and result1.get('method') == 'csv_download' and stats['api_fail'] > 0
    )
    
    # Check if CSV was used when requested
    csv_used_when_needed = (
        result2['success'] and result2.get('method') == 'csv_download'
    ) and (
        result3['success'] and result3.get('method') == 'csv_download'
    )
    
    if api_used_appropriately and csv_used_when_needed:
        logger.info("✅ VALIDATION PASSED: Method prioritization working correctly!")
    elif api_used_appropriately:
        logger.warning("⚠️  PARTIAL: API prioritization works, but CSV flags not respected")
    elif csv_used_when_needed:
        logger.warning("⚠️  PARTIAL: CSV flags work, but API prioritization failed")
    else:
        logger.error("❌ VALIDATION FAILED: Method prioritization not working as expected")
    
    logger.info("\n💡 For comprehensive testing, run: python comprehensive_test.py")
    
    downloader.cleanup()


def main():
    """Main test function"""
    setup_logging()
    
    print("EirGrid Downloader - Quick Validation Test")
    print("=" * 50)
    
    try:
        quick_validation_test()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())