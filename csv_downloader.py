"""
CSV Download Method for EirGrid Data - Updated for Organized File Structure
Integrates CSV download approach with the unified downloader architecture
Includes fixes for demand data and enhanced error handling
"""

import os
import json
import csv
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd


class CSVDataDownloader:
    """Enhanced CSV downloader that handles multiple data areas and pages with demand fixes"""
    
    def __init__(self, data_dir: Optional[str] = None, headless: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Set up directories - now consistent with organized structure
        if data_dir is None:
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(exist_ok=True)
        
        # Temporary download directory (separate from organized data structure)
        self.download_dir = Path("temp_downloads")
        self.download_dir.mkdir(exist_ok=True)
        
        self.headless = headless
        self.driver = None
        
        # FIXED: Updated area configuration with correct filename patterns and enhanced settings
        self.area_config = {
            'co2_intensity': {
                'page': 'co2',
                'button_index': 0,  # First "View In Table" button
                'expected_filename_patterns': ['Co2Intensity_*.csv', 'CO2Intensity_*.csv'],
                'table_title': 'CO2 Intensity Over Time',
                'download_timeout': 30,
                'requires_js_wait': False
            },
            'co2_emissions': {
                'page': 'co2', 
                'button_index': 1,  # Second "View In Table" button
                'expected_filename_patterns': ['Co2Emission_*.csv', 'CO2Emission_*.csv', 'Co2Emissions_*.csv'],
                'table_title': 'CO2 Emissions Over Time',
                'download_timeout': 30,
                'requires_js_wait': False
            },
            'wind_generation': {
                'page': 'wind',
                'button_index': 0,
                'expected_filename_patterns': ['WINDGeneration_*.csv', 'WindGeneration_*.csv', 'Wind_*.csv'],
                'table_title': 'Wind Generation',
                'download_timeout': 30,
                'requires_js_wait': False
            },
            'solar_generation': {
                'page': 'solar',
                'button_index': 0,
                'expected_filename_patterns': ['SOLARGeneration_*.csv', 'SolarGeneration_*.csv', 'Solar_*.csv'],
                'table_title': 'Solar Generation',
                'download_timeout': 30,
                'requires_js_wait': False
            },
            'demand': {
                'page': 'demand',
                'button_index': 0,  # First accordion for "Actual and Forecast System Demand"
                'expected_filename_patterns': [
                    'SystemDemand_*.csv', 
                    'Demand_*.csv', 
                    'DEMAND_*.csv',
                    'ActualDemand_*.csv',
                    'ForecastDemand_*.csv',
                    'DemandActual_*.csv',
                    'SystemDemandActual_*.csv',
                    'SystemDemandForecast_*.csv'
                ],
                'table_title': 'Actual and Forecast System Demand',
                'download_timeout': 45,  # Longer timeout for demand data
                'requires_js_wait': True  # Special handling for JavaScript processing
            }
        }
    
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome driver with enhanced download preferences"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Enhanced Chrome options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Enhanced download preferences
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,
            "download.extensions_to_open": "",
            "safebrowsing.disable_download_protection": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            self.logger.info("ChromeDriver setup completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup ChromeDriver: {e}")
            raise
        
        driver.implicitly_wait(10)
        
        # Execute script to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def download_area_csv(self,
                         area: str,
                         region: str = "all",
                         duration: str = "day", 
                         date_from: Optional[str] = None, 
                         date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Download CSV data for a specific area with enhanced error handling
        
        Args:
            area: Data area (e.g., 'co2_intensity', 'co2_emissions', 'demand')
            region: 'all', 'roi', or 'ni'
            duration: 'day', 'week', or 'month'
            date_from: Start date in 'DD-Mon-YYYY' format (optional)
            date_to: End date in 'DD-Mon-YYYY' format (optional)
            
        Returns:
            Dictionary with success status, data, and metadata
        """
        
        result = {
            'area': area,
            'region': region,
            'method': 'csv_download',
            'success': False,
            'data': None,
            'error': None,
            'csv_file': None,
            'debug_info': {}
        }
        
        # Check if area is supported
        if area not in self.area_config:
            result['error'] = f"Area '{area}' not configured for CSV download"
            return result
        
        config = self.area_config[area]
        
        self.driver = self._setup_driver()
        
        try:
            # Build URL
            url = self._build_url(config['page'], region, duration, date_from, date_to, area)
            self.logger.info(f"Navigating to: {url}")
            
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load completely
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-gray.text-white"))
            )
            self.logger.info("Page loaded successfully")
            
            # FIXED: Enhanced wait for pages that require JavaScript processing
            if config.get('requires_js_wait', False):
                self.logger.info("Waiting for JavaScript processing to complete...")
                time.sleep(5)  # Wait for any JavaScript processing
                
                # Wait for charts to be rendered if present
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "svg"))
                    )
                    self.logger.info("Charts rendered successfully")
                except:
                    self.logger.warning("Charts may not have rendered completely")
            
            time.sleep(3)  # Additional settling time
            
            # Find accordion headers
            accordion_headers = self._find_accordion_headers()
            
            if not accordion_headers:
                raise Exception("No 'View In Table' accordion headers found")
            
            self.logger.info(f"Found {len(accordion_headers)} accordion headers")
            result['debug_info']['accordion_count'] = len(accordion_headers)
            
            # Get the appropriate accordion header
            target_accordion = self._find_target_accordion(accordion_headers, config)
            
            if not target_accordion:
                # Enhanced error reporting for debugging
                accordion_texts = []
                for i, header in enumerate(accordion_headers):
                    try:
                        parent = header.find_element(By.XPATH, "./ancestor::div[contains(@class, 'bg-gray')]")
                        accordion_texts.append(f"Accordion {i}: {parent.text[:100]}...")
                    except:
                        accordion_texts.append(f"Accordion {i}: Could not get text")
                
                result['debug_info']['accordion_texts'] = accordion_texts
                raise Exception(f"Could not find accordion for area '{area}'. Available accordions: {accordion_texts}")
            
            # Expand accordion
            self._expand_accordion(target_accordion)
            
            # Wait for table section
            self._wait_for_table_section()
            
            # Find download button
            download_button = self._find_download_button_in_section(target_accordion)
            
            if not download_button:
                raise Exception("Download .CSV button not found in the expanded section")
            
            # Get button info for debugging
            try:
                button_href = download_button.get_attribute("href")
                result['debug_info']['download_url'] = button_href
                self.logger.info(f"Download button URL: {button_href}")
            except:
                pass
            
            # Clear downloads and attempt download
            self._clear_download_directory()
            
            # FIXED: Enhanced download with multiple pattern checking
            csv_file = self._click_download_and_wait_multiple_patterns(
                download_button, 
                config['expected_filename_patterns'], 
                config.get('download_timeout', 30)
            )
            
            if csv_file:
                # Parse CSV
                parsed_data = self._parse_csv_file(csv_file, area)
                
                result['success'] = True
                result['data'] = parsed_data
                result['csv_file'] = csv_file
                result['debug_info']['final_filename'] = Path(csv_file).name
                
                self.logger.info(f"Successfully downloaded and parsed CSV for {area}")
            else:
                # Enhanced error reporting with file listing
                available_files = list(self.download_dir.glob("*"))
                result['debug_info']['available_files'] = [f.name for f in available_files]
                result['error'] = f"Failed to download CSV file. Available files: {[f.name for f in available_files]}"
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error downloading CSV for {area}: {e}")
            
            # Enhanced debugging info
            try:
                result['debug_info']['page_title'] = self.driver.title
                result['debug_info']['current_url'] = self.driver.current_url
            except:
                pass
            
            # Take screenshot for debugging
            self._save_debug_screenshot(area)
            
        finally:
            if self.driver:
                self.driver.quit()
        
        return result
    
    def _find_accordion_headers(self) -> List:
        """Find all accordion headers (View In Table buttons)"""
        strategies = [
            "//h3[@data-orientation='vertical']//button[contains(text(), 'View In Table')]",
            "//button[contains(@class, 'flex-1') and contains(text(), 'View In Table')]",
            "//div[@data-orientation='vertical']//button[contains(text(), 'View In Table')]"
        ]
        
        for xpath in strategies:
            try:
                headers = self.driver.find_elements(By.XPATH, xpath)
                if headers:
                    self.logger.info(f"Found {len(headers)} accordion headers using strategy")
                    return headers
            except Exception as e:
                self.logger.debug(f"Strategy failed: {e}")
                continue
        
        return []
    
    def _find_target_accordion(self, accordion_headers: List, config: Dict) -> Optional[object]:
        """FIXED: Enhanced accordion finding with better debugging"""
        
        self.logger.info(f"Looking for accordion with button_index: {config.get('button_index')} and title: {config.get('table_title')}")
        
        # Try by button index first
        if 'button_index' in config:
            button_index = config['button_index']
            if button_index < len(accordion_headers):
                header = accordion_headers[button_index]
                self.logger.info(f"Selected accordion by index {button_index}")
                return header
        
        # Try by table title
        if 'table_title' in config:
            for i, header in enumerate(accordion_headers):
                try:
                    # Look for the title in the parent container
                    parent = header.find_element(By.XPATH, "./ancestor::div[contains(@class, 'bg-gray')]")
                    parent_text = parent.text
                    
                    if config['table_title'] in parent_text:
                        self.logger.info(f"Found accordion by title match at index {i}")
                        return header
                    
                    # Log what we found for debugging
                    self.logger.debug(f"Accordion {i} text snippet: {parent_text[:100]}...")
                    
                except Exception as e:
                    self.logger.debug(f"Could not get text for accordion {i}: {e}")
                    continue
        
        # If we get here, we couldn't find the right accordion
        self.logger.error("Could not find target accordion by index or title")
        return None
    
    def _expand_accordion(self, accordion_button):
        """Expand accordion if it's not already expanded"""
        try:
            # Check if accordion is already expanded
            state = accordion_button.get_attribute("data-state")
            if state != "open":
                self.logger.info("Expanding accordion section")
                # Scroll into view
                self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", 
                                         accordion_button)
                time.sleep(1)
                
                # Click to expand
                accordion_button.click()
                time.sleep(2)
            else:
                self.logger.info("Accordion already expanded")
        except Exception as e:
            self.logger.warning(f"Error checking accordion state: {e}")
            # Try clicking anyway
            accordion_button.click()
            time.sleep(2)
    
    def _wait_for_table_section(self):
        """Wait for table section to be visible"""
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )
            self.logger.info("Table section is visible")
        except Exception as e:
            self.logger.warning(f"Table might not be fully loaded: {e}")
    
    def _find_download_button_in_section(self, accordion_button) -> Optional[object]:
        """Find download button within the expanded accordion section"""
        try:
            # Find the accordion content div that follows the header
            accordion_content = accordion_button.find_element(
                By.XPATH, 
                "./ancestor::h3/following-sibling::div[@role='region']"
            )
            
            # Look for download button within this section
            strategies = [
                ".//a[contains(text(), 'Download .CSV')]",
                ".//a[contains(@class, 'bg-primary') and contains(text(), 'Download')]",
                ".//a[contains(text(), '.CSV')]"
            ]
            
            for xpath in strategies:
                try:
                    button = accordion_content.find_element(By.XPATH, xpath)
                    if button and button.is_displayed():
                        self.logger.info("Found download button in accordion section")
                        return button
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error finding download button in section: {e}")
        
        return None
    
    def _build_url(self, page: str, region: str, duration: str, date_from: Optional[str], date_to: Optional[str], area: str = None) -> str:
        """Build the URL for the specified page and parameters"""
        base_url = f"https://www.smartgriddashboard.com/{region}/{page}/"
        
        params = []
        
        # The CO2 page uses specific parameter names for intensity vs emissions
        if page == 'co2':
            if area == 'co2_intensity':
                # CO2 intensity uses 'intensity' prefix
                params.append(f"intensityduration={duration}")
                if date_from:
                    params.append(f"intensitydatefrom={date_from}")
                if date_to:
                    params.append(f"intensitydateto={date_to}")
            elif area == 'co2_emissions':
                # CO2 emissions uses 'emissions' prefix
                params.append(f"emissionsduration={duration}")
                if date_from:
                    params.append(f"emissionsdatefrom={date_from}")
                if date_to:
                    params.append(f"emissionsdateto={date_to}")
            else:
                # Default to intensity if area not specified
                params.append(f"intensityduration={duration}")
                if date_from:
                    params.append(f"intensitydatefrom={date_from}")
                if date_to:
                    params.append(f"intensitydateto={date_to}")
        else:
            # All other pages use standard parameter names
            params.append(f"duration={duration}")
            if date_from:
                params.append(f"datefrom={date_from}")
            if date_to:
                params.append(f"dateto={date_to}")
        
        if params:
            return f"{base_url}?{'&'.join(params)}"
        else:
            return base_url
    
    def _clear_download_directory(self):
        """Clear any existing CSV files in download directory"""
        try:
            for file in self.download_dir.glob("*.csv"):
                file.unlink()
            self.logger.debug("Cleared temporary download directory")
        except Exception as e:
            self.logger.warning(f"Error clearing download directory: {e}")
    
    def _click_download_and_wait_multiple_patterns(self, download_button, filename_patterns: List[str], timeout: int = 30) -> Optional[str]:
        """FIXED: Enhanced download with multiple filename pattern support"""
        try:
            # Scroll and ensure button is clickable
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", download_button)
            time.sleep(1)
            
            # Check button state
            if not download_button.is_displayed():
                raise Exception("Download button is not visible")
            
            if not download_button.is_enabled():
                raise Exception("Download button is not enabled")
            
            # Multiple click strategies
            success = False
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # Regular click
                        WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable(download_button)
                        )
                        download_button.click()
                    elif attempt == 1:
                        # JavaScript click
                        self.driver.execute_script("arguments[0].click();", download_button)
                    else:
                        # Action chains click
                        ActionChains(self.driver).move_to_element(download_button).click().perform()
                    
                    success = True
                    self.logger.info(f"Successfully clicked download button (attempt {attempt + 1})")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            if not success:
                raise Exception("Failed to click download button after multiple attempts")
            
            # Wait for download with multiple patterns
            return self._wait_for_download_multiple_patterns(filename_patterns, timeout)
            
        except Exception as e:
            self.logger.error(f"Error in download process: {e}")
            return None
    
    def _wait_for_download_multiple_patterns(self, filename_patterns: List[str], timeout: int) -> Optional[str]:
        """FIXED: Wait for download with multiple filename patterns"""
        start_time = time.time()
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            # Check for files matching any pattern
            for pattern in filename_patterns:
                csv_files = list(self.download_dir.glob(pattern))
                if csv_files:
                    # Get the most recent file
                    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                    if latest_file.stat().st_size > 0:
                        self.logger.info(f"Downloaded file: {latest_file.name} (matched pattern: {pattern})")
                        return str(latest_file)
            
            # Check for .crdownload files (Chrome partial downloads)
            crdownload_files = list(self.download_dir.glob("*.crdownload"))
            
            # Log progress every 10 seconds
            current_time = time.time()
            if current_time - last_log_time > 10:
                if crdownload_files:
                    self.logger.info(f"Download in progress... ({current_time - start_time:.1f}s elapsed)")
                else:
                    # List all files for debugging
                    all_files = list(self.download_dir.glob("*"))
                    self.logger.debug(f"Waiting for download... Available files: {[f.name for f in all_files]}")
                last_log_time = current_time
            
            time.sleep(0.5)
        
        # Final check for any CSV files (even if they don't match expected patterns)
        csv_files = list(self.download_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
            self.logger.warning(f"Found CSV file with unexpected name: {latest_file.name}")
            return str(latest_file)
        
        self.logger.error(f"Download timeout after {timeout} seconds")
        return None
    
    def _parse_csv_file(self, csv_file_path: str, area: str) -> Dict[str, Any]:
        """Parse downloaded CSV file and convert to standard format with normalized time"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            self.logger.info(f"CSV columns: {list(df.columns)}")
            self.logger.info(f"Total rows: {len(df)}")
            
            # Parse based on area type
            if area in ['co2_intensity', 'co2_emissions']:
                return self._parse_co2_csv(df, area)
            elif area == 'demand':
                return self._parse_demand_csv(df, area)
            else:
                return self._parse_standard_csv(df, area)
                
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise

    def _normalize_time_format(self, time_str: str) -> str:
        """
        Normalize time string to consistent format: 'YYYY-MM-DD HH:MM:SS'
        Handles various input formats from CSV sources
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

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime for normalization"""
        formats = [
            '%Y-%m-%d %H:%M:%S',        # Normalized format
            '%d-%b-%Y %H:%M:%S',        # API format: "24-Jun-2025 00:00:00"
            '%d %B %Y %H:%M',           # CSV format: "24 June 2025 00:00"
            '%d %b %Y %H:%M',           # CSV format abbreviated: "24 Jun 2025 00:00"
            '%Y-%m-%dT%H:%M:%S',        # ISO format
            '%d %B %Y, %H:%M',          # CSV format with comma: "24 June 2025, 00:00"
            '%d %b %Y, %H:%M',          # CSV format abbreviated with comma: "24 Jun 2025, 00:00"
            '%d-%m-%Y %H:%M:%S',        # Alternative format
            '%d/%m/%Y %H:%M:%S',        # Alternative format
        ]
        
        # Clean up the time string
        time_str = time_str.strip()
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.split('.')[0], fmt)  # Remove milliseconds if present
            except ValueError:
                continue
        
        self.logger.warning(f"Could not parse time format: '{time_str}' - using current time as fallback")
        return datetime.now()  # Fallback
        """Parse downloaded CSV file and convert to standard format"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            self.logger.info(f"CSV columns: {list(df.columns)}")
            self.logger.info(f"Total rows: {len(df)}")
            
            # Parse based on area type
            if area in ['co2_intensity', 'co2_emissions']:
                return self._parse_co2_csv(df, area)
            elif area == 'demand':
                return self._parse_demand_csv(df, area)
            else:
                return self._parse_standard_csv(df, area)
                
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise
    
    def _parse_co2_csv(self, df: pd.DataFrame, area: str) -> Dict[str, Any]:
        """Parse CO2-specific CSV format with normalized time"""
        time_series = []
        
        # Expected columns for CO2 data:
        # DATE & TIME, CO2 INTENSITY (gCO2/kWh), CO2 FORECAST (gCO2/kWh), *REGION*
        # or
        # DATE & TIME, CO2 EMISSION (tCO2/hr), CO2 FORECAST (tCO2/hr), *REGION*
        
        time_col = df.columns[0]  # DATE & TIME
        actual_col = df.columns[1]  # Actual values
        forecast_col = df.columns[2] if len(df.columns) > 2 else None  # Forecast values
        region_col = df.columns[3] if len(df.columns) > 3 else None  # Region
        
        region_value = None
        
        for _, row in df.iterrows():
            try:
                original_time_str = str(row[time_col]).strip()
                # Normalize time format for consistency
                normalized_time = self._normalize_time_format(original_time_str)
                
                # Parse actual value
                actual_val = self._parse_number(row[actual_col])
                
                # Parse forecast value  
                forecast_val = None
                if forecast_col and pd.notna(row[forecast_col]):
                    forecast_val = self._parse_number(row[forecast_col])
                
                # Get region from first row
                if region_value is None and region_col and pd.notna(row[region_col]):
                    region_value = str(row[region_col]).strip().replace('*', '')
                
                # Create data point based on what data is available
                if actual_val is not None:
                    # Has actual data
                    data_point = {
                        'time': normalized_time,
                        'value': actual_val,
                        'is_forecast': False
                    }
                    if forecast_val is not None:
                        data_point['forecast_value'] = forecast_val
                elif forecast_val is not None:
                    # Only forecast data
                    data_point = {
                        'time': normalized_time,
                        'value': forecast_val,
                        'is_forecast': True
                    }
                else:
                    # No valid data
                    continue
                
                time_series.append(data_point)
                
            except Exception as e:
                self.logger.warning(f"Error parsing row: {e}")
                continue
        
        return {
            'time_series': time_series,
            'extracted_at': datetime.now().isoformat(),
            'metadata': {
                'region': region_value or 'unknown',
                'area': area,
                'total_points': len(time_series)
            }
        }
    
    def _parse_demand_csv(self, df: pd.DataFrame, area: str) -> Dict[str, Any]:
        """Parse demand-specific CSV format with normalized time"""
        time_series = []
        
        # Expected columns for Demand data:
        # DATE & TIME, DEMAND (MW), DEMAND FORECAST (MW), *REGION*
        
        time_col = df.columns[0]  # DATE & TIME
        actual_col = df.columns[1] if len(df.columns) > 1 else None  # Actual demand
        forecast_col = df.columns[2] if len(df.columns) > 2 else None  # Forecast demand
        region_col = df.columns[3] if len(df.columns) > 3 else None  # Region
        
        region_value = None
        
        for _, row in df.iterrows():
            try:
                original_time_str = str(row[time_col]).strip()
                # Normalize time format for consistency
                normalized_time = self._normalize_time_format(original_time_str)
                
                # Parse actual value
                actual_val = self._parse_number(row[actual_col]) if actual_col else None
                
                # Parse forecast value  
                forecast_val = None
                if forecast_col and pd.notna(row[forecast_col]):
                    forecast_val = self._parse_number(row[forecast_col])
                
                # Get region from first row
                if region_value is None and region_col and pd.notna(row[region_col]):
                    region_value = str(row[region_col]).strip().replace('*', '')
                
                # Create data point based on what data is available
                if actual_val is not None:
                    # Has actual data
                    data_point = {
                        'time': normalized_time,
                        'value': actual_val,
                        'is_forecast': False
                    }
                    if forecast_val is not None:
                        data_point['forecast_value'] = forecast_val
                elif forecast_val is not None:
                    # Only forecast data
                    data_point = {
                        'time': normalized_time,
                        'value': forecast_val,
                        'is_forecast': True
                    }
                else:
                    # No valid data
                    continue
                
                time_series.append(data_point)
                
            except Exception as e:
                self.logger.warning(f"Error parsing row: {e}")
                continue
        
        return {
            'time_series': time_series,
            'extracted_at': datetime.now().isoformat(),
            'metadata': {
                'region': region_value or 'unknown',
                'area': area,
                'total_points': len(time_series)
            }
        }
    
    def _parse_standard_csv(self, df: pd.DataFrame, area: str) -> Dict[str, Any]:
        """Parse standard CSV format for other data types with normalized time"""
        time_series = []
        
        # Assume first column is time, second is value
        time_col = df.columns[0]
        value_col = df.columns[1]
        
        for _, row in df.iterrows():
            try:
                original_time_str = str(row[time_col]).strip()
                # Normalize time format for consistency
                normalized_time = self._normalize_time_format(original_time_str)
                
                value = self._parse_number(row[value_col])
                
                if value is not None:
                    time_series.append({
                        'time': normalized_time,
                        'value': value,
                        'is_forecast': False
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error parsing row: {e}")
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
        """Parse number from CSV value"""
        if pd.isna(value) or str(value).strip() == '':
            return None
        
        try:
            return float(str(value).strip().replace(',', ''))
        except ValueError:
            return None
    
    def _save_debug_screenshot(self, area: str):
        """Save screenshot for debugging"""
        try:
            screenshot_path = self.download_dir / f"error_screenshot_{area}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.driver.save_screenshot(str(screenshot_path))
            self.logger.info(f"Debug screenshot saved: {screenshot_path}")
        except Exception as e:
            self.logger.warning(f"Could not save debug screenshot: {e}")
    
    def cleanup_temp_files(self):
        """Clean up temporary download files"""
        try:
            for file in self.download_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            self.logger.debug("Cleaned up temporary download files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    downloader = CSVDataDownloader(headless=False)  # Set True for headless
    
    # Test demand download specifically
    print("Testing demand download with fixes...")
    result = downloader.download_area_csv(
        area='demand',
        region='all',
        duration='day'
    )
    
    if result['success']:
        print("SUCCESS!")
        print(f"Downloaded {len(result['data']['time_series'])} data points")
        print(f"Debug info: {result['debug_info']}")
        
        # Show first few points
        for i, point in enumerate(result['data']['time_series'][:5]):
            print(f"{i+1}: {point}")
    else:
        print(f"FAILED: {result['error']}")
        print(f"Debug info: {result['debug_info']}")
    
    # Cleanup
    downloader.cleanup_temp_files()


if __name__ == "__main__":
    main()