"""
Unified EirGrid Data Downloader
Combines API and CSV download methods with intelligent fallback
Includes fixes for demand data and enhanced method selection
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import the updated CSV downloader
try:
    from csv_downloader import CSVDataDownloader
    csv_downloader_available = True
except ImportError:
    csv_downloader_available = False
    logging.warning("CSV downloader not available")


class UnifiedEirGridDownloader:
    """Unified downloader that combines API and CSV download methods with enhanced intelligence"""
    
    def __init__(self, data_dir: Optional[str] = None, headless: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Set up data directory
        if data_dir is None:
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(exist_ok=True)
        
        # CSV downloader with fixes
        self.csv_downloader = CSVDataDownloader(data_dir=str(self.data_dir), headless=headless) if csv_downloader_available else None
        
        # API rate limiting
        self.api_delay = 2.0  # Delay between API calls
        self.last_api_call = 0
        
        # Data area mappings for API
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
        
        # UPDATED: Enhanced area preferences based on comprehensive test results
        self.csv_preferred_areas = ['solar_generation']  # Areas where API fails but CSV works
        self.csv_required_areas = ['solar_generation']   # Areas that ONLY work with CSV
        self.problematic_csv_areas = ['co2_emissions']   # Areas where CSV often fails but API works
        self.api_preferred_areas = ['co2_intensity', 'wind_generation', 'demand', 'snsp', 'frequency', 'interconnection']
        
        # Track download statistics
        self.stats = {
            'api_success': 0,
            'api_fail': 0,
            'csv_success': 0,
            'csv_fail': 0
        }
    
    def download_area(self, 
                     area: str,
                     region: str = "all",
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None,
                     include_forecast: bool = False,
                     force_scraping: bool = False) -> Dict[str, Any]:
        """
        Download data for a specific area using intelligent method selection
        
        Args:
            area: Data area name
            region: 'all', 'roi', or 'ni'
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            include_forecast: Whether to include forecast data (requires CSV download)
            force_scraping: Force use of CSV download method
            
        Returns:
            Dictionary with data and metadata
        """
        
        # Default to yesterday if no dates provided
        if date_from is None:
            date_from = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if date_to is None:
            date_to = datetime.now().strftime('%Y-%m-%d')
        
        result = {
            'area': area,
            'region': region,
            'date_from': date_from,
            'date_to': date_to,
            'method': None,
            'success': False,
            'data': None,
            'error': None,
            'method_attempted': []
        }
        
        # UPDATED: Intelligent method selection based on comprehensive test results
        self.logger.info(f"Starting download for {area} (region: {region}, dates: {date_from} to {date_to})")
        
        # Force CSV for areas that require it or when explicitly requested
        if (include_forecast or 
            force_scraping or 
            area in self.csv_required_areas or
            (area in self.csv_preferred_areas and not force_scraping)):
            
            if csv_downloader_available:
                method_reason = "forecast requested" if include_forecast else "force scraping" if force_scraping else "CSV preferred"
                self.logger.info(f"Using CSV download method for {area} ({method_reason})")
                csv_result = self._download_via_csv(area, region, date_from, date_to, include_forecast)
                result['method_attempted'].append('csv_download')
                
                if csv_result['success']:
                    self.stats['csv_success'] += 1
                    return csv_result
                else:
                    self.stats['csv_fail'] += 1
                    self.logger.warning(f"CSV download failed for {area}: {csv_result['error']}")
                    
                    # For CSV-required areas, don't fall back to API
                    if area in self.csv_required_areas:
                        return csv_result
            else:
                self.logger.warning("CSV download not available, falling back to API")
        
        # Try API for areas where it works well (and we haven't already tried CSV)
        if (area in self.api_areas and 
            area not in self.csv_required_areas and
            'csv_download' not in result['method_attempted']):
            
            self.logger.info(f"Attempting API download for {area}")
            api_result = self._download_via_api(area, region.upper() if region.lower() != 'all' else 'ALL', date_from, date_to)
            result['method_attempted'].append('api')
            
            if api_result['success']:
                self.stats['api_success'] += 1
                return api_result
            else:
                self.stats['api_fail'] += 1
                self.logger.warning(f"API failed for {area}: {api_result['error']}")
        
        # Fallback to CSV download if API failed and CSV is available and we haven't tried it yet
        if (csv_downloader_available and 
            'csv_download' not in result['method_attempted'] and
            area not in self.problematic_csv_areas):
            
            self.logger.info(f"Falling back to CSV download for {area}")
            csv_result = self._download_via_csv(area, region, date_from, date_to, include_forecast)
            result['method_attempted'].append('csv_download_fallback')
            
            if csv_result['success']:
                self.stats['csv_success'] += 1
                return csv_result
            else:
                self.stats['csv_fail'] += 1
                self.logger.warning(f"CSV fallback failed for {area}: {csv_result['error']}")
        
        # If we get here, all available methods failed
        result['error'] = f"All available methods failed. Attempted: {', '.join(result['method_attempted'])}"
        self.logger.error(f"All methods failed for {area}: {result['error']}")
        return result
    
    def _download_via_api(self, area: str, region: str, date_from: str, date_to: str) -> Dict[str, Any]:
        """Download data using the API method with enhanced error handling"""
        
        result = {
            'area': area,
            'region': region,
            'date_from': date_from,
            'date_to': date_to,
            'method': 'api',
            'success': False,
            'data': None,
            'error': None
        }
        
        # Get API area name
        api_area = self.api_areas.get(area, area)
        
        # Format dates for API
        try:
            start_time = self._format_date_for_api(date_from, False)
            end_time = self._format_date_for_api(date_to, True)
        except Exception as e:
            result['error'] = f"Date formatting error: {e}"
            return result
        
        # Rate limiting
        time_since_last = time.time() - self.last_api_call
        if time_since_last < self.api_delay:
            time.sleep(self.api_delay - time_since_last)
        
        url = f"https://www.smartgriddashboard.com/DashboardService.svc/data?area={api_area}&region={region}&datefrom={start_time}&dateto={end_time}"
        self.logger.debug(f"API URL: {url}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                self.last_api_call = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('Status') == 'Success':
                        rows = data.get('Rows', [])
                        if rows:
                            # Parse API data to standard format
                            parsed_data = self._parse_api_data(rows, area)
                            result['data'] = parsed_data
                            result['success'] = True
                            
                            # Log success summary
                            point_count = len(parsed_data['time_series'])
                            self.logger.info(f"API download successful: {point_count} data points")
                            return result
                        else:
                            result['error'] = "No data returned from API"
                    else:
                        error_msg = data.get('ErrorMessage', 'Unknown API error')
                        if 'Invalid Area' in error_msg:
                            result['error'] = f"Invalid Area: '{api_area}'"
                        else:
                            result['error'] = error_msg
                
                elif response.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(f"Service unavailable, retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    result['error'] = f"HTTP {response.status_code}: {response.reason}"
                    
            except requests.exceptions.Timeout:
                result['error'] = "Request timeout"
                if attempt < max_retries - 1:
                    self.logger.warning(f"API timeout, retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                result['error'] = str(e)
                break
        
        return result
    
    def _download_via_csv(self, area: str, region: str, date_from: str, date_to: str, include_forecast: bool) -> Dict[str, Any]:
        """Download data using the enhanced CSV download method"""
        
        result = {
            'area': area,
            'region': region,
            'date_from': date_from,
            'date_to': date_to,
            'method': 'csv_download',
            'success': False,
            'data': None,
            'error': None,
            'debug_info': {}
        }
        
        if not csv_downloader_available:
            result['error'] = "CSV downloader not available"
            return result
        
        try:
            # Convert dates to format expected by CSV downloader
            date_from_fmt = self._convert_date_for_csv(date_from)
            date_to_fmt = self._convert_date_for_csv(date_to)
            
            # Determine duration
            days_diff = (datetime.strptime(date_to, '%Y-%m-%d') - datetime.strptime(date_from, '%Y-%m-%d')).days
            if days_diff == 0:
                duration = 'day'
            elif days_diff <= 7:
                duration = 'week'
            else:
                duration = 'month'
            
            self.logger.debug(f"CSV download: {area}, duration={duration}, dates={date_from_fmt} to {date_to_fmt}")
            
            # Use the enhanced CSV downloader
            csv_result = self.csv_downloader.download_area_csv(
                area=area,
                region=region,
                duration=duration,
                date_from=date_from_fmt,
                date_to=date_to_fmt
            )
            
            if csv_result['success']:
                result['data'] = csv_result['data']
                result['success'] = True
                result['debug_info'] = csv_result.get('debug_info', {})
                
                # Log summary
                ts = result['data'].get('time_series', [])
                actual_count = sum(1 for point in ts if not point.get('is_forecast', False))
                forecast_count = sum(1 for point in ts if point.get('is_forecast', False))
                self.logger.info(f"CSV download successful: {len(ts)} total points ({actual_count} actual, {forecast_count} forecast)")
            else:
                result['error'] = csv_result['error']
                result['debug_info'] = csv_result.get('debug_info', {})
                
                # Enhanced error logging for better debugging
                if result['debug_info']:
                    self.logger.debug(f"CSV download debug info: {result['debug_info']}")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"CSV download error: {e}")
        
        return result
    
    def _format_date_for_api(self, date_str: str, is_end_date: bool = False) -> str:
        """Convert YYYY-MM-DD to API format"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            # Use double %% to escape the % character in strftime
            time_part = '23%%3A59' if is_end_date else '00%%3A00'
            # Format: dd-mmm-yyyy+HH%3AMM (e.g., 15-jun-2025+00%3A00)
            return dt.strftime(f'%d-%b-%Y+{time_part}').lower()
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD. Error: {e}")
    
    def _convert_date_for_csv(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to CSV downloader format (DD-Mon-YYYY)"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%d-%b-%Y')
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD. Error: {e}")
    
    def _parse_api_data(self, rows: List[Dict], area: str) -> Dict[str, Any]:
        """Parse API response data to standard format"""
        
        time_series = []
        
        for row in rows:
            try:
                data_point = {
                    'time': row.get('EffectiveTime', ''),
                    'value': self._parse_number(row.get('Value')),
                    'is_forecast': False,
                    'field_name': row.get('FieldName', area)
                }
                
                # Only add if we have a valid value
                if data_point['value'] is not None:
                    time_series.append(data_point)
            except Exception as e:
                self.logger.warning(f"Error parsing API row: {e}")
                continue
        
        return {
            'time_series': time_series,
            'extracted_at': datetime.now().isoformat(),
            'metadata': {
                'area': area,
                'total_points': len(time_series)
            }
        }
    
    def _parse_number(self, value) -> Optional[float]:
        """Parse number from various formats"""
        if value is None or str(value).strip() == '':
            return None
        
        try:
            return float(str(value).strip().replace(',', ''))
        except ValueError:
            return None
    
    def save_data(self, data: Dict[str, Any], area: str, region: str, update_existing: bool = True) -> str:
        """
        Save data to JSON file with smart updating
        
        Args:
            data: Data dictionary from download
            area: Area name
            region: Region name
            update_existing: Whether to update existing files
            
        Returns:
            Path to saved file
        """
        
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{area}_{region}_{timestamp}.json"
        file_path = self.data_dir / filename
        
        if update_existing and file_path.exists():
            # Load existing data
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
            
            # Merge data intelligently
            merged_data = self._merge_data(existing_data, data)
            
            # Save merged data
            with open(file_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
                
            self.logger.info(f"Updated existing file: {file_path}")
        else:
            # Save new file
            save_data = {
                'metadata': {
                    'area': area,
                    'region': region,
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                },
                'data': data
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Created new file: {file_path}")
        
        return str(file_path)
    
    def _merge_data(self, existing: Dict, new: Dict) -> Dict:
        """
        Merge new data with existing, replacing forecast with actual values
        """
        
        # Update metadata
        if 'metadata' not in existing:
            existing['metadata'] = {}
        existing['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Get time series from both
        existing_ts = existing.get('data', {}).get('time_series', [])
        new_ts = new.get('time_series', [])
        
        # Create lookup for existing data by time
        existing_by_time = {point['time']: point for point in existing_ts}
        
        # Update with new data
        for new_point in new_ts:
            time_key = new_point['time']
            
            if time_key in existing_by_time:
                old_point = existing_by_time[time_key]
                
                # If old point was forecast and new is actual, replace
                if old_point.get('is_forecast', False) and not new_point.get('is_forecast', False):
                    existing_by_time[time_key] = new_point
                    self.logger.debug(f"Replaced forecast with actual for {time_key}")
                
                # Update if new value is different
                elif old_point.get('value') != new_point.get('value'):
                    existing_by_time[time_key] = new_point
                    self.logger.debug(f"Updated value for {time_key}")
            else:
                # Add new time point
                existing_by_time[time_key] = new_point
        
        # Convert back to list and sort by time
        merged_ts = list(existing_by_time.values())
        merged_ts.sort(key=lambda x: self._parse_time(x['time']))
        
        existing['data']['time_series'] = merged_ts
        existing['data']['extracted_at'] = new.get('extracted_at', datetime.now().isoformat())
        
        return existing
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime for sorting"""
        formats = [
            '%d-%b-%Y %H:%M:%S',
            '%d %B %Y %H:%M',
            '%d %b %Y %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%d %B %Y, %H:%M',  # For "18 June 2025, 00:00" format
            '%d %b %Y, %H:%M'   # For abbreviated month format
        ]
        
        # Clean up the time string
        time_str = time_str.strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse time format: '{time_str}'")
        return datetime.now()  # Fallback
    
    def get_statistics(self) -> Dict[str, int]:
        """Get download statistics"""
        return self.stats
    
    def get_method_recommendation(self, area: str) -> str:
        """Get recommended download method for an area"""
        if area in self.csv_required_areas:
            return "csv_only"
        elif area in self.csv_preferred_areas:
            return "csv_preferred"
        elif area in self.api_preferred_areas:
            return "api_preferred"
        elif area in self.problematic_csv_areas:
            return "api_only"
        else:
            return "api_first_with_csv_fallback"
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.csv_downloader:
            self.csv_downloader.cleanup_temp_files()


def test_unified_downloader():
    """Test function to verify the unified downloader works correctly"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    downloader = UnifiedEirGridDownloader(headless=False)
    
    # Test areas with different methods
    test_areas = ['demand', 'co2_intensity', 'solar_generation']
    
    print("🔬 Testing Unified EirGrid Downloader")
    print("=" * 50)
    
    for area in test_areas:
        print(f"\n📊 Testing {area}...")
        
        result = downloader.download_area(
            area=area,
            region='all',
            date_from='2025-06-18',
            date_to='2025-06-18'
        )
        
        if result['success']:
            point_count = len(result['data']['time_series'])
            print(f"   ✅ SUCCESS: {area} - {point_count} points via {result['method']}")
        else:
            print(f"   ❌ FAILED: {area} - {result['error']}")
            print(f"   Attempted methods: {result['method_attempted']}")
    
    # Print statistics
    stats = downloader.get_statistics()
    print(f"\n📈 Download Statistics:")
    print(f"   API Success: {stats['api_success']}")
    print(f"   API Failures: {stats['api_fail']}")
    print(f"   CSV Success: {stats['csv_success']}")
    print(f"   CSV Failures: {stats['csv_fail']}")
    
    downloader.cleanup()


if __name__ == "__main__":
    test_unified_downloader()