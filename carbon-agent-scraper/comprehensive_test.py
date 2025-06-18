#!/usr/bin/env python
"""
Comprehensive EirGrid Data Collection Test Script
Tests both API and CSV download methods for all supported areas
"""

import sys
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Import our downloader classes
try:
    from unified_downloader import UnifiedEirGridDownloader
    downloader_available = True
except ImportError:
    downloader_available = False
    print("⚠️  UnifiedEirGridDownloader not available")


class ComprehensiveEirGridTester:
    """Comprehensive tester for both API and CSV download methods"""
    
    def __init__(self, headless: bool = True, debug: bool = False):
        self.debug = debug
        self.headless = headless
        
        # Initialize downloader if available
        self.downloader = UnifiedEirGridDownloader(headless=headless) if downloader_available else None
        
        # All supported areas with their API mappings
        self.api_areas = {
            'co2_emissions': 'co2emission',
            'co2_intensity': 'co2intensity',
            'snsp': 'SnspAll',
            'wind_generation': 'windactual',
            'solar_generation': 'solaractual',
            'total_generation': 'generationactual',
            'demand': 'demandactual',
            'fuel_mix': 'fuelMix',
            'frequency': 'frequency',
            'interconnection': 'interconnection'
        }
        
        # Areas that support CSV download
        self.csv_areas = [
            'co2_emissions',
            'co2_intensity', 
            'wind_generation',
            'solar_generation',
            'demand'
        ]
        
        # Rate limiting settings
        self.api_delay = 2.0  # 2 seconds between API requests
        self.csv_delay = 3.0  # 3 seconds between CSV tests
        self.retry_attempts = 2
        self.retry_delay = 3.0
        
        # Test results storage
        self.results = {
            'api_tests': {},
            'csv_tests': {},
            'unified_tests': {}
        }
    
    def setup_logging(self):
        """Set up logging for the test"""
        level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        
        # Suppress verbose libraries unless in debug mode
        if not self.debug:
            logging.getLogger('selenium').setLevel(logging.ERROR)
            logging.getLogger('urllib3').setLevel(logging.ERROR)
            logging.getLogger('webdriver_manager').setLevel(logging.ERROR)
    
    def format_date_for_api(self, date_str: str, is_end_date: bool = False) -> str:
        """Convert YYYY-MM-DD format to API format"""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        time_part = '23%3A59' if is_end_date else '00%3A00'
        base_format = dt.strftime('%d-%b-%Y').lower()
        return f'{base_format}+{time_part}'
    
    def test_api_area(self, area: str, api_area: str, region: str = 'ALL', date_str: str = None) -> Dict:
        """Test API connectivity for a specific area"""
        
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        start_time = self.format_date_for_api(date_str, False)
        end_time = self.format_date_for_api(date_str, True)
        
        url = f"https://www.smartgriddashboard.com/DashboardService.svc/data?area={api_area}&region={region}&datefrom={start_time}&dateto={end_time}"
        
        result = {
            'area': area,
            'api_area': api_area,
            'region': region,
            'date': date_str,
            'method': 'api',
            'success': False,
            'data_points': 0,
            'error': None,
            'sample_value': None,
            'sample_time': None,
            'response_time': None,
            'attempts': 0
        }
        
        for attempt in range(self.retry_attempts):
            result['attempts'] = attempt + 1
            
            try:
                if attempt > 0:
                    time.sleep(self.retry_delay)
                
                start_request = time.time()
                response = requests.get(url, timeout=15)
                result['response_time'] = round(time.time() - start_request, 2)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('Status') == 'Success':
                        rows = data.get('Rows', [])
                        result['data_points'] = len(rows)
                        
                        if rows:
                            result['success'] = True
                            sample_row = rows[0]
                            result['sample_value'] = sample_row.get('Value')
                            result['sample_time'] = sample_row.get('EffectiveTime')
                            return result
                        else:
                            result['error'] = 'No data rows returned'
                    else:
                        result['error'] = data.get('ErrorMessage', 'Unknown API error')
                        
                elif response.status_code == 503:
                    result['error'] = 'Service temporarily unavailable (503)'
                    if attempt < self.retry_attempts - 1:
                        continue
                else:
                    result['error'] = f'HTTP {response.status_code}: {response.reason}'
                    break
                    
            except requests.exceptions.Timeout:
                result['error'] = 'Request timeout'
                if attempt < self.retry_attempts - 1:
                    continue
            except Exception as e:
                result['error'] = str(e)
                break
        
        return result
    
    def test_csv_area(self, area: str, region: str = 'all', date_str: str = None) -> Dict:
        """Test CSV download for a specific area"""
        
        if not self.downloader:
            return {
                'area': area,
                'method': 'csv_download',
                'success': False,
                'error': 'Downloader not available',
                'data_points': 0
            }
        
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        result = {
            'area': area,
            'region': region,
            'date': date_str,
            'method': 'csv_download',
            'success': False,
            'data_points': 0,
            'actual_points': 0,
            'forecast_points': 0,
            'error': None,
            'response_time': None
        }
        
        try:
            start_time = time.time()
            
            # Use the unified downloader's CSV method directly
            csv_result = self.downloader._download_via_csv(area, region, date_str, date_str, False)
            
            result['response_time'] = round(time.time() - start_time, 2)
            result['success'] = csv_result['success']
            
            if csv_result['success']:
                time_series = csv_result['data'].get('time_series', [])
                result['data_points'] = len(time_series)
                result['actual_points'] = sum(1 for p in time_series if not p.get('is_forecast', False))
                result['forecast_points'] = result['data_points'] - result['actual_points']
            else:
                result['error'] = csv_result.get('error', 'Unknown CSV error')
                
        except Exception as e:
            result['error'] = str(e)
            result['response_time'] = round(time.time() - start_time, 2) if 'start_time' in locals() else None
        
        return result
    
    def test_unified_method(self, area: str, region: str = 'all', date_str: str = None, include_forecast: bool = False) -> Dict:
        """Test the unified download method"""
        
        if not self.downloader:
            return {
                'area': area,
                'method': 'unified',
                'success': False,
                'error': 'Downloader not available'
            }
        
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        result = {
            'area': area,
            'region': region,
            'date': date_str,
            'method': 'unified',
            'success': False,
            'actual_method_used': None,
            'data_points': 0,
            'error': None,
            'response_time': None
        }
        
        try:
            start_time = time.time()
            
            unified_result = self.downloader.download_area(
                area=area,
                region=region,
                date_from=date_str,
                date_to=date_str,
                include_forecast=include_forecast,
                force_scraping=False
            )
            
            result['response_time'] = round(time.time() - start_time, 2)
            result['success'] = unified_result['success']
            result['actual_method_used'] = unified_result.get('method', 'unknown')
            
            if unified_result['success']:
                time_series = unified_result['data'].get('time_series', [])
                result['data_points'] = len(time_series)
            else:
                result['error'] = unified_result.get('error', 'Unknown unified error')
                
        except Exception as e:
            result['error'] = str(e)
            result['response_time'] = round(time.time() - start_time, 2) if 'start_time' in locals() else None
        
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all methods and areas"""
        
        print("🔬 COMPREHENSIVE EIRGRID DATA COLLECTION TEST")
        print("=" * 80)
        print(f"📅 Testing with yesterday's date: {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}")
        print(f"⏱️  API delay: {self.api_delay}s, CSV delay: {self.csv_delay}s")
        print(f"🔄 Retry attempts: {self.retry_attempts}")
        print("=" * 80)
        
        # Phase 1: Test API connectivity for all areas
        print("\n🚀 PHASE 1: API CONNECTIVITY TEST")
        print("-" * 50)
        
        api_results = []
        for i, (area, api_area) in enumerate(self.api_areas.items()):
            print(f"[{i+1}/{len(self.api_areas)}] Testing API: {area} ({api_area})")
            
            if i > 0:
                time.sleep(self.api_delay)
            
            result = self.test_api_area(area, api_area)
            api_results.append(result)
            self.results['api_tests'][area] = result
            
            if result['success']:
                print(f"   ✅ SUCCESS - {result['data_points']} points ({result['response_time']}s)")
            else:
                print(f"   ❌ FAILED - {result['error']} ({result['response_time']}s)")
        
        # Phase 2: Test CSV download for supported areas
        print(f"\n📊 PHASE 2: CSV DOWNLOAD TEST")
        print("-" * 50)
        
        csv_results = []
        for i, area in enumerate(self.csv_areas):
            print(f"[{i+1}/{len(self.csv_areas)}] Testing CSV: {area}")
            
            if i > 0:
                time.sleep(self.csv_delay)
            
            result = self.test_csv_area(area)
            csv_results.append(result)
            self.results['csv_tests'][area] = result
            
            if result['success']:
                print(f"   ✅ SUCCESS - {result['data_points']} points ({result['actual_points']} actual, {result['forecast_points']} forecast) ({result['response_time']}s)")
            else:
                print(f"   ❌ FAILED - {result['error']} ({result['response_time']}s)")
        
        # Phase 3: Test unified method for key areas
        print(f"\n🎯 PHASE 3: UNIFIED METHOD TEST")
        print("-" * 50)
        
        key_test_areas = ['co2_intensity', 'wind_generation', 'demand', 'snsp']
        unified_results = []
        
        for i, area in enumerate(key_test_areas):
            if area not in self.api_areas:
                continue
                
            print(f"[{i+1}/{len(key_test_areas)}] Testing Unified: {area}")
            
            if i > 0:
                time.sleep(1.0)
            
            result = self.test_unified_method(area)
            unified_results.append(result)
            self.results['unified_tests'][area] = result
            
            if result['success']:
                print(f"   ✅ SUCCESS - Used {result['actual_method_used']} method, {result['data_points']} points ({result['response_time']}s)")
            else:
                print(f"   ❌ FAILED - {result['error']} ({result['response_time']}s)")
        
        # Generate comprehensive summary
        self.print_comprehensive_summary()
    
    def print_comprehensive_summary(self):
        """Print detailed summary of all test results"""
        
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        # API Results Summary
        api_success = sum(1 for r in self.results['api_tests'].values() if r['success'])
        api_total = len(self.results['api_tests'])
        api_rate = (api_success / api_total * 100) if api_total > 0 else 0
        
        print(f"\n🚀 API CONNECTIVITY RESULTS:")
        print(f"   Success Rate: {api_rate:.1f}% ({api_success}/{api_total} areas)")
        
        print(f"\n   ✅ Working API Areas ({api_success}):")
        for area, result in self.results['api_tests'].items():
            if result['success']:
                print(f"      🟢 {area:<20} - {result['data_points']:>3} points ({result['response_time']}s)")
        
        print(f"\n   ❌ Failed API Areas ({api_total - api_success}):")
        for area, result in self.results['api_tests'].items():
            if not result['success']:
                print(f"      🔴 {area:<20} - {result['error']}")
        
        # CSV Results Summary
        if self.results['csv_tests']:
            csv_success = sum(1 for r in self.results['csv_tests'].values() if r['success'])
            csv_total = len(self.results['csv_tests'])
            csv_rate = (csv_success / csv_total * 100) if csv_total > 0 else 0
            
            print(f"\n📊 CSV DOWNLOAD RESULTS:")
            print(f"   Success Rate: {csv_rate:.1f}% ({csv_success}/{csv_total} areas)")
            
            print(f"\n   ✅ Working CSV Areas ({csv_success}):")
            for area, result in self.results['csv_tests'].items():
                if result['success']:
                    print(f"      🟢 {area:<20} - {result['data_points']:>3} points ({result['actual_points']} actual, {result['forecast_points']} forecast)")
            
            if csv_success < csv_total:
                print(f"\n   ❌ Failed CSV Areas ({csv_total - csv_success}):")
                for area, result in self.results['csv_tests'].items():
                    if not result['success']:
                        print(f"      🔴 {area:<20} - {result['error']}")
        
        # Unified Method Results
        if self.results['unified_tests']:
            unified_success = sum(1 for r in self.results['unified_tests'].values() if r['success'])
            unified_total = len(self.results['unified_tests'])
            
            print(f"\n🎯 UNIFIED METHOD RESULTS:")
            print(f"   Success Rate: {(unified_success/unified_total*100):.1f}% ({unified_success}/{unified_total} areas)")
            
            for area, result in self.results['unified_tests'].items():
                status = "✅" if result['success'] else "❌"
                method = result.get('actual_method_used', 'unknown')
                print(f"      {status} {area:<20} - Used {method} method")
        
        # Method Comparison for overlapping areas
        print(f"\n🔍 METHOD COMPARISON:")
        print("   Area                 API    CSV    Best Method")
        print("   " + "-" * 50)
        
        for area in self.csv_areas:
            api_status = "✅" if self.results['api_tests'].get(area, {}).get('success', False) else "❌"
            csv_status = "✅" if self.results['csv_tests'].get(area, {}).get('success', False) else "❌"
            
            # Determine best method
            api_ok = self.results['api_tests'].get(area, {}).get('success', False)
            csv_ok = self.results['csv_tests'].get(area, {}).get('success', False)
            
            if api_ok and csv_ok:
                api_time = self.results['api_tests'][area].get('response_time', 999)
                csv_time = self.results['csv_tests'][area].get('response_time', 999)
                best = "API (faster)" if api_time < csv_time else "CSV"
            elif api_ok:
                best = "API only"
            elif csv_ok:
                best = "CSV only"
            else:
                best = "Neither"
            
            print(f"   {area:<20} {api_status}    {csv_status}    {best}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("   " + "-" * 30)
        
        if api_rate >= 80:
            print("   🎯 Excellent API connectivity! Prioritize API method.")
        elif api_rate >= 50:
            print("   ⚖️  Mixed API results. Use API with CSV fallback.")
        else:
            print("   🚨 Poor API connectivity. Focus on CSV method.")
        
        # Areas that need CSV fallback
        needs_csv = [area for area in self.csv_areas 
                    if not self.results['api_tests'].get(area, {}).get('success', False)
                    and self.results['csv_tests'].get(area, {}).get('success', False)]
        
        if needs_csv:
            print(f"   📊 These areas need CSV fallback: {', '.join(needs_csv)}")
        
        # Areas with no working method
        no_method = [area for area in self.api_areas.keys()
                    if not self.results['api_tests'].get(area, {}).get('success', False)
                    and not self.results['csv_tests'].get(area, {}).get('success', False)]
        
        if no_method:
            print(f"   🚨 These areas have no working method: {', '.join(no_method)}")
        
        print(f"\n🔄 To retest specific area: python comprehensive_test.py [area_name]")
        print(f"🐛 For debugging: python comprehensive_test.py --debug")
    
    def test_specific_area(self, area: str):
        """Run detailed test for a specific area"""
        
        if area not in self.api_areas:
            print(f"❌ Unknown area: {area}")
            print(f"Available areas: {', '.join(self.api_areas.keys())}")
            return
        
        print(f"🔬 DETAILED TEST: {area}")
        print("=" * 50)
        
        # Test different dates
        dates_to_test = [
            ('today', datetime.now().strftime('%Y-%m-%d')),
            ('yesterday', (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')),
            ('2_days_ago', (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'))
        ]
        
        for date_name, date_str in dates_to_test:
            print(f"\n📅 Testing {date_name} ({date_str}):")
            
            # Test API
            api_result = self.test_api_area(area, self.api_areas[area], date_str=date_str)
            api_status = "✅" if api_result['success'] else "❌"
            api_info = api_result.get('error', f"{api_result.get('data_points', 0)} points")
            print(f"   API: {api_status} {api_info}")
            
            # Test CSV if supported
            if area in self.csv_areas:
                csv_result = self.test_csv_area(area, date_str=date_str)
                csv_status = "✅" if csv_result['success'] else "❌"
                csv_info = csv_result.get('error', f"{csv_result.get('data_points', 0)} points")
                print(f"   CSV: {csv_status} {csv_info}")
            else:
                print(f"   CSV: ➖ Not supported")
            
            # Test unified
            unified_result = self.test_unified_method(area, date_str=date_str)
            unified_status = "✅" if unified_result['success'] else "❌"
            method_used = unified_result.get('actual_method_used', 'unknown')
            unified_info = unified_result.get('error', f"{unified_result.get('data_points', 0)} points")
            print(f"   Unified: {unified_status} Used {method_used} ({unified_info})")
            
            time.sleep(2.0)  # Delay between date tests
    
    def cleanup(self):
        """Clean up resources"""
        if self.downloader:
            self.downloader.cleanup()


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test of EirGrid data collection methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comprehensive test
  python comprehensive_test.py
  
  # Test specific area in detail
  python comprehensive_test.py co2_intensity
  
  # Run with debug logging and visible browser
  python comprehensive_test.py --debug --show-browser
        """
    )
    
    parser.add_argument('area', nargs='?',
                       help='Specific area to test in detail (optional)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--show-browser', action='store_true',
                       help='Show browser during CSV tests')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ComprehensiveEirGridTester(
        headless=not args.show_browser,
        debug=args.debug
    )
    
    tester.setup_logging()
    
    try:
        if args.area:
            # Test specific area
            tester.test_specific_area(args.area)
        else:
            # Run comprehensive test
            tester.run_comprehensive_test()
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        tester.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())