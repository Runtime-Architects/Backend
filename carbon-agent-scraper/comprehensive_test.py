"""
Comprehensive EirGrid Data Collection Test Script
Tests both API and CSV download methods
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
    """Comprehensive tester for both API and CSV download methods with organized file structure"""
    
    def __init__(self, headless: bool = True, debug: bool = False, data_dir: str = None):
        self.debug = debug
        self.headless = headless
        self.data_dir = data_dir or "test_data"
        
        # Initialize downloader if available
        self.downloader = UnifiedEirGridDownloader(data_dir=self.data_dir, headless=headless) if downloader_available else None
        
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
            'unified_tests': {},
            'file_structure_tests': {}
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
        """Test the unified download method with file structure testing"""
        
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
            'response_time': None,
            'file_path': None,
            'file_structure_correct': False
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
                
                # Test file saving with new structure
                try:
                    file_path = self.downloader.save_data(
                        unified_result['data'],
                        area,
                        region,
                        date_str,
                        date_str,
                        update_existing=True
                    )
                    result['file_path'] = file_path
                    
                    # Verify file structure
                    expected_dir = Path(self.data_dir) / area
                    expected_filename = f"{area}_{date_str}_{date_str}.json" if region.lower() == 'all' else f"{area}_{region}_{date_str}_{date_str}.json"
                    expected_path = expected_dir / expected_filename
                    
                    result['file_structure_correct'] = Path(file_path) == expected_path and expected_path.exists()
                    
                except Exception as e:
                    result['error'] = f"File save error: {e}"
            else:
                result['error'] = unified_result.get('error', 'Unknown unified error')
                
        except Exception as e:
            result['error'] = str(e)
            result['response_time'] = round(time.time() - start_time, 2) if 'start_time' in locals() else None
        
        return result
    
    def test_file_structure(self):
        """Test the organized file structure"""
        
        print(f"\n📁 PHASE 4: FILE STRUCTURE TEST")
        print("-" * 50)
        
        if not self.downloader:
            print("   ❌ Downloader not available for file structure testing")
            return
        
        # Test creating directories and files for different scenarios
        test_scenarios = [
            ('co2_intensity', 'all', '2025-06-23', '2025-06-23'),
            ('wind_generation', 'roi', '2025-06-20', '2025-06-25'),
            ('demand', 'ni', '2025-06-01', '2025-06-30')
        ]
        
        structure_results = {}
        
        for i, (area, region, date_from, date_to) in enumerate(test_scenarios, 1):
            print(f"[{i}/{len(test_scenarios)}] Testing file structure: {area} ({region}) {date_from} to {date_to}")
            
            try:
                # Create mock data
                mock_data = {
                    'time_series': [
                        {'time': f'{date_from} 12:00:00', 'value': 100.0, 'is_forecast': False}
                    ],
                    'extracted_at': datetime.now().isoformat(),
                    'metadata': {'area': area, 'total_points': 1}
                }
                
                # Save using the new structure
                file_path = self.downloader.save_data(
                    mock_data, area, region, date_from, date_to, update_existing=False
                )
                
                # Verify structure
                expected_dir = Path(self.data_dir) / area
                expected_filename = f"{area}_{date_from}_{date_to}.json" if region.lower() == 'all' else f"{area}_{region}_{date_from}_{date_to}.json"
                expected_path = expected_dir / expected_filename
                
                structure_correct = Path(file_path) == expected_path and expected_path.exists()
                
                structure_results[f"{area}_{region}_{date_from}_{date_to}"] = {
                    'success': True,
                    'file_path': file_path,
                    'expected_path': str(expected_path),
                    'structure_correct': structure_correct,
                    'directory_exists': expected_dir.exists(),
                    'file_exists': expected_path.exists()
                }
                
                if structure_correct:
                    print(f"   ✅ SUCCESS - Correct structure: {expected_path.relative_to(Path(self.data_dir).parent)}")
                else:
                    print(f"   ❌ FAILED - Expected: {expected_path}, Got: {file_path}")
                    
            except Exception as e:
                structure_results[f"{area}_{region}_{date_from}_{date_to}"] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ FAILED - {e}")
        
        self.results['file_structure_tests'] = structure_results
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all methods and areas with file structure testing"""
        
        print("🔬 COMPREHENSIVE EIRGRID DATA COLLECTION TEST - ORGANIZED STRUCTURE")
        print("=" * 80)
        print(f"📅 Testing with yesterday's date: {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}")
        print(f"📁 Test data directory: {self.data_dir}")
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
        
        # Phase 3: Test unified method for key areas with file structure
        print(f"\n🎯 PHASE 3: UNIFIED METHOD TEST WITH FILE STRUCTURE")
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
                file_status = "✅ correct structure" if result['file_structure_correct'] else "⚠️ incorrect structure"
                relative_path = Path(result['file_path']).relative_to(Path(self.data_dir).parent) if result['file_path'] else "unknown"
                print(f"   ✅ SUCCESS - Used {result['actual_method_used']} method, {result['data_points']} points ({result['response_time']}s)")
                print(f"      📁 File: {relative_path} ({file_status})")
            else:
                print(f"   ❌ FAILED - {result['error']} ({result['response_time']}s)")
        
        # Phase 4: Test file structure specifically
        self.test_file_structure()
        
        # Generate comprehensive summary
        self.print_comprehensive_summary()
    
    def print_comprehensive_summary(self):
        """Print detailed summary of all test results including file structure"""
        
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST SUMMARY - ORGANIZED STRUCTURE")
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
                file_status = ""
                if result['success'] and 'file_structure_correct' in result:
                    file_status = " 📁✅" if result['file_structure_correct'] else " 📁⚠️"
                print(f"      {status} {area:<20} - Used {method} method{file_status}")
        
        # File Structure Test Results
        if self.results['file_structure_tests']:
            structure_success = sum(1 for r in self.results['file_structure_tests'].values() if r.get('success', False) and r.get('structure_correct', False))
            structure_total = len(self.results['file_structure_tests'])
            
            print(f"\n📁 FILE STRUCTURE TEST RESULTS:")
            print(f"   Success Rate: {(structure_success/structure_total*100):.1f}% ({structure_success}/{structure_total} tests)")
            
            for test_name, result in self.results['file_structure_tests'].items():
                if result.get('success', False):
                    status = "✅" if result.get('structure_correct', False) else "⚠️"
                    relative_path = Path(result['file_path']).relative_to(Path(self.data_dir).parent) if result.get('file_path') else "unknown"
                    print(f"      {status} {test_name:<30} - {relative_path}")
                else:
                    print(f"      ❌ {test_name:<30} - {result.get('error', 'Unknown error')}")
        
        # Directory Structure Overview
        print(f"\n📂 GENERATED DIRECTORY STRUCTURE:")
        data_path = Path(self.data_dir)
        if data_path.exists():
            print(f"   {self.data_dir}/")
            for metric_dir in sorted(data_path.iterdir()):
                if metric_dir.is_dir():
                    file_count = len(list(metric_dir.glob("*.json")))
                    print(f"   ├── {metric_dir.name}/ ({file_count} files)")
                    for json_file in sorted(metric_dir.glob("*.json"))[:3]:  # Show first 3 files
                        print(f"   │   ├── {json_file.name}")
                    if file_count > 3:
                        print(f"   │   └── ... ({file_count - 3} more files)")
        
        # Method Comparison for overlapping areas
        print(f"\n🔍 METHOD COMPARISON:")
        print("   Area                 API    CSV    Unified  File Structure")
        print("   " + "-" * 65)
        
        for area in self.csv_areas:
            api_status = "✅" if self.results['api_tests'].get(area, {}).get('success', False) else "❌"
            csv_status = "✅" if self.results['csv_tests'].get(area, {}).get('success', False) else "❌"
            unified_status = "✅" if self.results['unified_tests'].get(area, {}).get('success', False) else "❌"
            
            # File structure status
            file_status = "➖"
            if area in self.results['unified_tests']:
                if self.results['unified_tests'][area].get('file_structure_correct', False):
                    file_status = "✅"
                elif self.results['unified_tests'][area].get('success', False):
                    file_status = "⚠️"
                else:
                    file_status = "❌"
            
            print(f"   {area:<20} {api_status}    {csv_status}    {unified_status}      {file_status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("   " + "-" * 30)
        
        if api_rate >= 80:
            print("   🎯 Excellent API connectivity! Prioritize API method.")
        elif api_rate >= 50:
            print("   ⚖️  Mixed API results. Use API with CSV fallback.")
        else:
            print("   🚨 Poor API connectivity. Focus on CSV method.")
        
        # File structure recommendations
        structure_success_rate = 0
        if self.results['file_structure_tests']:
            structure_success = sum(1 for r in self.results['file_structure_tests'].values() if r.get('structure_correct', False))
            structure_total = len(self.results['file_structure_tests'])
            structure_success_rate = (structure_success / structure_total * 100) if structure_total > 0 else 0
        
        if structure_success_rate == 100:
            print("   📁 Perfect file organization! New structure working correctly.")
        elif structure_success_rate >= 80:
            print("   📁 Good file organization with minor issues.")
        else:
            print("   📁 File organization needs attention - check directory structure.")
        
        print(f"\n🔄 To retest specific area: python comprehensive_test.py [area_name]")
        print(f"🐛 For debugging: python comprehensive_test.py --debug")
        print(f"📁 Test data saved in: {self.data_dir}/")
    
    def test_specific_area(self, area: str):
        """Run detailed test for a specific area including file structure"""
        
        if area not in self.api_areas:
            print(f"❌ Unknown area: {area}")
            print(f"Available areas: {', '.join(self.api_areas.keys())}")
            return
        
        print(f"🔬 DETAILED TEST: {area} (with file structure)")
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
            
            # Test unified with file structure
            unified_result = self.test_unified_method(area, date_str=date_str)
            unified_status = "✅" if unified_result['success'] else "❌"
            method_used = unified_result.get('actual_method_used', 'unknown')
            unified_info = unified_result.get('error', f"{unified_result.get('data_points', 0)} points")
            
            file_info = ""
            if unified_result['success'] and unified_result.get('file_path'):
                file_status = "✅" if unified_result.get('file_structure_correct', False) else "⚠️"
                relative_path = Path(unified_result['file_path']).relative_to(Path(self.data_dir).parent)
                file_info = f" → {relative_path} {file_status}"
            
            print(f"   Unified: {unified_status} Used {method_used} ({unified_info}){file_info}")
            
            time.sleep(2.0)  # Delay between date tests
    
    def cleanup(self):
        """Clean up resources"""
        if self.downloader:
            self.downloader.cleanup()


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test of EirGrid data collection methods with organized file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comprehensive test
  python comprehensive_test.py
  
  # Test specific area in detail
  python comprehensive_test.py co2_intensity
  
  # Run with debug logging and visible browser
  python comprehensive_test.py --debug --show-browser
  
  # Use custom test data directory
  python comprehensive_test.py --data-dir ./test_organized_data
        """
    )
    
    parser.add_argument('area', nargs='?',
                       help='Specific area to test in detail (optional)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--show-browser', action='store_true',
                       help='Show browser during CSV tests')
    
    parser.add_argument('--data-dir', 
                       help='Test data directory (default: test_data)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ComprehensiveEirGridTester(
        headless=not args.show_browser,
        debug=args.debug,
        data_dir=args.data_dir
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