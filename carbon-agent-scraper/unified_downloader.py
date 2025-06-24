"""
Unified EirGrid Data Downloader
Updated with new organized file structure: data/{metric}/{metric}_{start-date}_{end-date}.json
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
    """Unified downloader with organized file structure by metric and date range"""
    
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
        
        # Enhanced area preferences based on comprehensive test results
        self.csv_preferred_areas = ['solar_generation']
        self.csv_required_areas = ['solar_generation']
        self.problematic_csv_areas = ['co2_emissions']
        self.api_preferred_areas = ['co2_intensity', 'wind_generation', 'demand', 'snsp', 'frequency', 'interconnection']
        
        # Track download statistics
        self.stats = {
            'api_success': 0,
            'api_fail': 0,
            'csv_success': 0,
            'csv_fail': 0
        }
    
    def _create_metric_directory(self, area: str) -> Path:
        """Create and return metric-specific directory"""
        metric_dir = self.data_dir / area
        metric_dir.mkdir(exist_ok=True)
        return metric_dir
    
    def _generate_filename(self, area: str, date_from: str, date_to: str, region: str = None) -> str:
        """Generate filename in new format: {area}_{start-date}_{end-date}.json"""
        if region and region.lower() != 'all':
            return f"{area}_{region}_{date_from}_{date_to}.json"
        else:
            return f"{area}_{date_from}_{date_to}.json"
    
    def _find_overlapping_files(self, metric_dir: Path, area: str, date_from: str, date_to: str, region: str = None) -> List[Path]:
        """Find files that might contain overlapping data for the given date range"""
        overlapping_files = []
        
        target_start = datetime.strptime(date_from, '%Y-%m-%d')
        target_end = datetime.strptime(date_to, '%Y-%m-%d')
        
        # Look for files in the metric directory
        pattern = f"{area}_*.json" if not region or region.lower() == 'all' else f"{area}_{region}_*.json"
        
        for file_path in metric_dir.glob(pattern):
            try:
                # Extract dates from filename
                filename = file_path.stem
                parts = filename.split('_')
                
                if region and region.lower() != 'all':
                    # Format: area_region_start_end
                    if len(parts) >= 4:
                        file_start_str = parts[-2]
                        file_end_str = parts[-1]
                else:
                    # Format: area_start_end
                    if len(parts) >= 3:
                        file_start_str = parts[-2]
                        file_end_str = parts[-1]
                
                file_start = datetime.strptime(file_start_str, '%Y-%m-%d')
                file_end = datetime.strptime(file_end_str, '%Y-%m-%d')
                
                # Check for overlap
                if (target_start <= file_end and target_end >= file_start):
                    overlapping_files.append(file_path)
                    
            except (ValueError, IndexError) as e:
                self.logger.debug(f"Could not parse filename {file_path.name}: {e}")
                continue
        
        return overlapping_files
    
    def download_area(self, 
                     area: str,
                     region: str = "all",
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None,
                     include_forecast: bool = False,
                     force_scraping: bool = False) -> Dict[str, Any]:
        """
        Download data for a specific area using intelligent method selection
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
        
        # Intelligent method selection based on comprehensive test results
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
        
        # Try API for areas where it works well
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
        
        # Fallback to CSV download if API failed
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
        """Parse API response data to standard format with normalized time"""
        
        time_series = []
        
        for row in rows:
            try:
                original_time = row.get('EffectiveTime', '')
                # Normalize time format for consistency
                normalized_time = self._normalize_time_format(original_time)
                
                data_point = {
                    'time': normalized_time,
                    'value': self._parse_number(row.get('Value')),
                    'is_forecast': False,
                    # 'field_name': row.get('FieldName', area) Don't need this for now
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
    
    def save_data(self, 
                  data: Dict[str, Any], 
                  area: str, 
                  region: str, 
                  date_from: str = None,
                  date_to: str = None,
                  update_existing: bool = True) -> str:
        """
        Save data to organized file structure: data/{metric}/{metric}_{start-date}_{end-date}.json
        
        Args:
            data: Data dictionary from download
            area: Area name (metric)
            region: Region name  
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            update_existing: Whether to update existing overlapping files
            
        Returns:
            Path to saved file
        """
        
        # Extract date range from data if not provided
        if not date_from or not date_to:
            date_from, date_to = self._extract_date_range_from_data(data)
        
        # Create metric-specific directory
        metric_dir = self._create_metric_directory(area)
        
        # Generate filename
        filename = self._generate_filename(area, date_from, date_to, region)
        target_file_path = metric_dir / filename
        
        # Check for exact file match first
        if target_file_path.exists() and update_existing:
            self.logger.info(f"Updating existing file: {target_file_path}")
            # Load existing data
            with open(target_file_path, 'r') as f:
                existing_data = json.load(f)
            
            # Merge data intelligently (prioritize actual over forecast)
            merged_data = self._merge_data(existing_data, data, area, region, date_from, date_to)
            
            # Save merged data
            with open(target_file_path, 'w') as f:
                json.dump(merged_data, f, indent=2)
                
            self.logger.info(f"Updated file: {target_file_path}")
            
        elif update_existing:
            # Check for overlapping files that might need to be consolidated
            overlapping_files = self._find_overlapping_files(metric_dir, area, date_from, date_to, region)
            
            if overlapping_files:
                self.logger.info(f"Found {len(overlapping_files)} overlapping files for date range {date_from} to {date_to}")
                
                # For exact date matches, merge into existing file
                # For partial overlaps, create new file and optionally consolidate later
                exact_match = None
                for file_path in overlapping_files:
                    try:
                        # Check if this file covers the exact same date range
                        filename_parts = file_path.stem.split('_')
                        if region and region.lower() != 'all':
                            file_start = filename_parts[-2]
                            file_end = filename_parts[-1]
                        else:
                            file_start = filename_parts[-2]
                            file_end = filename_parts[-1]
                        
                        if file_start == date_from and file_end == date_to:
                            exact_match = file_path
                            break
                    except (IndexError, ValueError):
                        continue
                
                if exact_match:
                    # Merge with exact match
                    with open(exact_match, 'r') as f:
                        existing_data = json.load(f)
                    
                    merged_data = self._merge_data(existing_data, data, area, region, date_from, date_to)
                    
                    with open(exact_match, 'w') as f:
                        json.dump(merged_data, f, indent=2)
                    
                    self.logger.info(f"Merged data into existing file: {exact_match}")
                    return str(exact_match)
                else:
                    # Create new file (partial overlap case)
                    self.logger.info(f"Creating new file for date range {date_from} to {date_to}")
            
            # Create new file
            save_data = {
                'metadata': {
                    'area': area,
                    'region': region,
                    'date_from': date_from,
                    'date_to': date_to,
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                },
                'data': data
            }
            
            with open(target_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Created new file: {target_file_path}")
        
        else:
            # Force create new file
            save_data = {
                'metadata': {
                    'area': area,
                    'region': region,
                    'date_from': date_from,
                    'date_to': date_to,
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                },
                'data': data
            }
            
            with open(target_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Created new file: {target_file_path}")
        
        return str(target_file_path)
    
    def _extract_date_range_from_data(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Extract date range from data time series"""
        time_series = data.get('time_series', [])
        
        if not time_series:
            # Default to today if no data
            today = datetime.now().strftime('%Y-%m-%d')
            return today, today
        
        # Parse all times to find min and max dates
        dates = []
        for point in time_series:
            try:
                parsed_time = self._parse_time(point['time'])
                dates.append(parsed_time.date())
            except:
                continue
        
        if dates:
            min_date = min(dates).strftime('%Y-%m-%d')
            max_date = max(dates).strftime('%Y-%m-%d')
            return min_date, max_date
        else:
            today = datetime.now().strftime('%Y-%m-%d')
            return today, today
    
    def _normalize_time_format(self, time_str: str) -> str:
        """
        Normalize time string to consistent format: 'YYYY-MM-DD HH:MM:SS'
        Handles various input formats from API and CSV sources
        """
        try:
            # Parse the time using existing _parse_time method
            dt = self._parse_time(time_str)
            # Return in standardized format
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            # If parsing fails, return original
            self.logger.warning(f"Could not normalize time format: {time_str}")
            return time_str
    
    def _merge_data(self, existing: Dict, new_data: Dict, area: str, region: str, date_from: str, date_to: str) -> Dict:
        """
        Merge new data with existing, replacing forecast with actual values
        FIXED: Handles time format inconsistencies between API and CSV sources
        
        Args:
            existing: Existing file data (with metadata wrapper)
            new_data: New data (time_series format)
            area: Area name
            region: Region name
            date_from: Date from
            date_to: Date to
        """
        
        # Update metadata
        if 'metadata' not in existing:
            existing['metadata'] = {}
        
        existing['metadata'].update({
            'area': area,
            'region': region,
            'date_from': date_from,
            'date_to': date_to,
            'last_updated': datetime.now().isoformat()
        })
        
        # Get time series from both (handle different data structures)
        if 'data' in existing and 'time_series' in existing['data']:
            existing_ts = existing['data']['time_series']
        elif 'time_series' in existing:
            existing_ts = existing['time_series']
        else:
            existing_ts = []
        
        # New data should always be in time_series format
        new_ts = new_data.get('time_series', [])
        
        # FIXED: Create lookup with normalized time keys to handle format differences
        existing_by_normalized_time = {}
        original_time_mapping = {}  # Map normalized time back to original format
        
        for point in existing_ts:
            original_time = point['time']
            normalized_time = self._normalize_time_format(original_time)
            existing_by_normalized_time[normalized_time] = point
            original_time_mapping[normalized_time] = original_time
        
        # Process new data points
        merge_count = 0
        update_count = 0
        add_count = 0
        
        for new_point in new_ts:
            original_new_time = new_point['time']
            normalized_new_time = self._normalize_time_format(original_new_time)
            
            if normalized_new_time in existing_by_normalized_time:
                old_point = existing_by_normalized_time[normalized_new_time]
                merged = False
                
                # If old point was forecast and new is actual, replace with actual
                if old_point.get('is_forecast', False) and not new_point.get('is_forecast', False):
                    # Keep the original time format from existing data for consistency
                    new_point_copy = new_point.copy()
                    new_point_copy['time'] = original_time_mapping[normalized_new_time]
                    existing_by_normalized_time[normalized_new_time] = new_point_copy
                    merge_count += 1
                    merged = True
                    self.logger.debug(f"Replaced forecast with actual for {normalized_new_time}")
                
                # If old point was actual and new is forecast, add forecast_value but keep actual
                elif (not old_point.get('is_forecast', False) and new_point.get('is_forecast', False) and
                      'forecast_value' not in old_point):
                    old_point['forecast_value'] = new_point.get('value')
                    merge_count += 1
                    merged = True
                    self.logger.debug(f"Added forecast value to actual data for {normalized_new_time}")
                
                # Update if values are different and not downgrading from actual to forecast
                elif (old_point.get('value') != new_point.get('value') and 
                      not (not old_point.get('is_forecast', False) and new_point.get('is_forecast', False))):
                    # Keep the original time format from existing data for consistency
                    new_point_copy = new_point.copy()
                    new_point_copy['time'] = original_time_mapping[normalized_new_time]
                    existing_by_normalized_time[normalized_new_time] = new_point_copy
                    update_count += 1
                    merged = True
                    self.logger.debug(f"Updated value for {normalized_new_time}")
                
                if not merged:
                    self.logger.debug(f"No merge needed for {normalized_new_time} (same value or downgrade)")
            else:
                # Add new time point (use consistent format)
                normalized_point = new_point.copy()
                normalized_point['time'] = self._normalize_time_format(original_new_time)
                existing_by_normalized_time[normalized_new_time] = normalized_point
                add_count += 1
        
        # Convert back to list and sort by time
        merged_ts = list(existing_by_normalized_time.values())
        merged_ts.sort(key=lambda x: self._parse_time(x['time']))
        
        # Log merge statistics
        self.logger.info(f"Data merge complete: {merge_count} merged, {update_count} updated, {add_count} added")
        
        # Ensure proper data structure
        if 'data' not in existing:
            existing['data'] = {}
        
        existing['data']['time_series'] = merged_ts
        existing['data']['extracted_at'] = new_data.get('extracted_at', datetime.now().isoformat())
        
        # Update metadata in data section if it exists
        if 'metadata' in new_data:
            if 'metadata' not in existing['data']:
                existing['data']['metadata'] = {}
            existing['data']['metadata'].update(new_data['metadata'])
        
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
    
    def list_available_data(self, area: str = None) -> Dict[str, List[str]]:
        """
        List all available data files in the organized structure
        
        Args:
            area: Optional specific area to list (if None, lists all)
            
        Returns:
            Dictionary mapping areas to list of available date ranges
        """
        available_data = {}
        
        if area:
            areas_to_check = [area]
        else:
            # Get all metric directories
            areas_to_check = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        for metric_area in areas_to_check:
            metric_dir = self.data_dir / metric_area
            if not metric_dir.exists():
                continue
            
            files = []
            for file_path in metric_dir.glob(f"{metric_area}_*.json"):
                # Extract date range from filename
                try:
                    filename = file_path.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Format could be: area_start_end or area_region_start_end
                        if len(parts) == 3:  # area_start_end
                            date_range = f"{parts[1]} to {parts[2]}"
                        else:  # area_region_start_end
                            date_range = f"{parts[-2]} to {parts[-1]} ({parts[-3]})"
                        files.append(date_range)
                except:
                    files.append(file_path.name)
            
            if files:
                available_data[metric_area] = sorted(files)
        
        return available_data
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.csv_downloader:
            self.csv_downloader.cleanup_temp_files()


def test_new_file_structure():
    """Test the new file structure"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    downloader = UnifiedEirGridDownloader(headless=False)
    
    print("🔬 Testing New Organized File Structure")
    print("=" * 50)
    
    # Test single day download
    result = downloader.download_area(
        area='co2_intensity',
        region='all',
        date_from='2025-06-23',
        date_to='2025-06-23'
    )
    
    if result['success']:
        # Save with new structure
        file_path = downloader.save_data(
            result['data'],
            'co2_intensity',
            'all',
            '2025-06-23',
            '2025-06-23'
        )
        print(f"✅ Data saved to: {file_path}")
        
        # List available data
        available = downloader.list_available_data()
        print(f"\n📁 Available data files:")
        for area, files in available.items():
            print(f"  {area}:")
            for file_info in files:
                print(f"    - {file_info}")
    else:
        print(f"❌ Download failed: {result['error']}")
    
    downloader.cleanup()


if __name__ == "__main__":
    test_new_file_structure()