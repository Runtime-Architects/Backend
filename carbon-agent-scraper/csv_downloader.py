"""
CSV Download Method for EirGrid Data
Integrates CSV download approach with the unified downloader architecture
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
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd


class CSVDataDownloader:
    """Enhanced CSV downloader that handles multiple data areas and pages"""
    
    def __init__(self, data_dir: Optional[str] = None, headless: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Set up directories
        if data_dir is None:
            self.data_dir = Path("data")
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(exist_ok=True)
        
        # Temporary download directory
        self.download_dir = Path("temp_downloads")
        self.download_dir.mkdir(exist_ok=True)
        
        self.headless = headless
        self.driver = None
        
        # Map data areas to their pages and button indices
        self.area_config = {
            'co2_intensity': {
                'page': 'co2',
                'button_index': 0,  # First "View In Table" button
                'expected_filename_pattern': 'Co2Intensity_*.csv'
            },
            'co2_emissions': {
                'page': 'co2', 
                'button_index': 1,  # Second "View In Table" button
                'expected_filename_pattern': 'Co2Emission_*.csv'
            },
            'wind_generation': {
                'page': 'wind',
                'button_index': 0,
                'expected_filename_pattern': 'WindGeneration_*.csv'
            },
            'solar_generation': {
                'page': 'solar',
                'button_index': 0,
                'expected_filename_pattern': 'SolarGeneration_*.csv'
            },
            'demand': {
                'page': 'demand',
                'button_index': 0,
                'expected_filename_pattern': 'Demand_*.csv'
            }
            # Add more areas as needed
        }
    
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome driver with download preferences"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Configure download behavior
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set download directory
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1
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
        Download CSV data for a specific area
        
        Args:
            area: Data area (e.g., 'co2_intensity', 'co2_emissions')
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
            'csv_file': None
        }
        
        # Check if area is supported
        if area not in self.area_config:
            result['error'] = f"Area '{area}' not configured for CSV download"
            return result
        
        config = self.area_config[area]
        
        self.driver = self._setup_driver()
        
        try:
            # Build URL
            url = self._build_url(config['page'], region, duration, date_from, date_to)
            self.logger.info(f"Navigating to: {url}")
            
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load completely
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-gray.text-white"))
            )
            self.logger.info("Page loaded successfully")
            
            # Wait for content to settle
            time.sleep(3)
            
            # Find "View In Table" buttons
            view_table_buttons = WebDriverWait(self.driver, 15).until(
                EC.presence_of_all_elements_located((By.XPATH, "//button[contains(text(), 'View In Table')]"))
            )
            
            if not view_table_buttons:
                raise Exception("No 'View In Table' buttons found")
            
            self.logger.info(f"Found {len(view_table_buttons)} 'View In Table' buttons")
            
            # Get the appropriate button for this area
            button_index = config['button_index']
            if button_index >= len(view_table_buttons):
                raise Exception(f"Button index {button_index} not available (only {len(view_table_buttons)} buttons found)")
            
            target_button = view_table_buttons[button_index]
            self.logger.info(f"Using button index {button_index} for area '{area}'")
            
            # Scroll the button into view and click
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", target_button)
            time.sleep(2)
            
            # Click the button
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable(target_button)
                )
                target_button.click()
                self.logger.info(f"Clicked 'View In Table' button for {area}")
            except Exception as e:
                self.logger.warning(f"Regular click failed: {e}. Trying JavaScript click...")
                self.driver.execute_script("arguments[0].click();", target_button)
                self.logger.info("Used JavaScript click as fallback")
            
            # Wait for the table section to expand
            self.logger.info("Waiting for table section to expand...")
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )
            self.logger.info("Table section expanded successfully")
            
            # Wait for download button to become available
            time.sleep(3)
            
            # Find the download CSV button
            download_button = self._find_download_button()
            
            if not download_button:
                raise Exception("Download .CSV button not found")
            
            # Clear any existing downloads
            self._clear_download_directory()
            
            # Click download button
            csv_file = self._click_download_and_wait(download_button, config['expected_filename_pattern'])
            
            if csv_file:
                # Parse CSV to standard format
                parsed_data = self._parse_csv_file(csv_file, area)
                
                result['success'] = True
                result['data'] = parsed_data
                result['csv_file'] = csv_file
                
                self.logger.info(f"Successfully downloaded and parsed CSV for {area}")
            else:
                result['error'] = "Failed to download CSV file"
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error downloading CSV for {area}: {e}")
            
            # Take screenshot for debugging
            self._save_debug_screenshot(area)
            
        finally:
            if self.driver:
                self.driver.quit()
        
        return result
    
    def _build_url(self, page: str, region: str, duration: str, date_from: Optional[str], date_to: Optional[str]) -> str:
        """Build the URL for the specified page and parameters"""
        base_url = f"https://www.smartgriddashboard.com/{region}/{page}/"
        
        params = []
        
        # Add duration parameter based on page type
        if page == 'co2':
            params.append(f"intensityduration={duration}")
            if date_from:
                params.append(f"intensitydatefrom={date_from}")
            if date_to:
                params.append(f"intensitydateto={date_to}")
        elif page == 'wind':
            params.append(f"windduration={duration}")
            if date_from:
                params.append(f"winddatefrom={date_from}")
            if date_to:
                params.append(f"winddateto={date_to}")
        elif page == 'solar':
            params.append(f"solarduration={duration}")
            if date_from:
                params.append(f"solardatefrom={date_from}")
            if date_to:
                params.append(f"solardateto={date_to}")
        elif page == 'demand':
            params.append(f"demandduration={duration}")
            if date_from:
                params.append(f"demanddatefrom={date_from}")
            if date_to:
                params.append(f"demanddateto={date_to}")
        
        if params:
            return f"{base_url}?{'&'.join(params)}"
        else:
            return base_url
    
    def _find_download_button(self) -> Optional[object]:
        """Find the download CSV button using multiple strategies"""
        strategies = [
            "//a[contains(text(), 'Download .CSV')]",
            "//a[contains(@class, 'bg-primary') and contains(text(), 'Download')]",
            "//a[@download and contains(@href, 'blob:')]",
            "//a[contains(text(), '.CSV')]",
            "//button[contains(text(), 'Download .CSV')]"
        ]
        
        for i, xpath in enumerate(strategies):
            try:
                self.logger.debug(f"Trying download button strategy {i+1}/{len(strategies)}...")
                button = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                
                if button.is_displayed():
                    self.logger.info(f"Found visible download button using strategy {i+1}")
                    return button
                    
            except Exception as e:
                self.logger.debug(f"Strategy {i+1} failed: {e}")
                continue
        
        # Try scrolling to find hidden button
        self.logger.info("Download button not found, scrolling to find it...")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Try first strategy again after scrolling
        try:
            button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Download .CSV')]"))
            )
            return button
        except:
            return None
    
    def _clear_download_directory(self):
        """Clear any existing CSV files in download directory"""
        try:
            for file in self.download_dir.glob("*.csv"):
                file.unlink()
        except Exception as e:
            self.logger.warning(f"Error clearing download directory: {e}")
    
    def _click_download_and_wait(self, download_button, filename_pattern: str, timeout: int = 30) -> Optional[str]:
        """Click download button and wait for file to be downloaded"""
        try:
            # Scroll button into view
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", download_button)
            time.sleep(1)
            
            # Try multiple click methods
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
                        # Direct URL navigation
                        href = download_button.get_attribute("href")
                        if href and href.startswith("blob:"):
                            self.driver.get(href)
                    
                    success = True
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            if not success:
                raise Exception("Failed to click download button")
            
            # Wait for file to download
            return self._wait_for_download(filename_pattern, timeout)
            
        except Exception as e:
            self.logger.error(f"Error in download process: {e}")
            return None
    
    def _wait_for_download(self, filename_pattern: str, timeout: int) -> Optional[str]:
        """Wait for CSV file to be downloaded"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for files matching the pattern
            csv_files = list(self.download_dir.glob(filename_pattern))
            if csv_files:
                # Get the most recent file
                latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    self.logger.info(f"Downloaded file: {latest_file.name}")
                    return str(latest_file)
            
            # Check for .crdownload files (Chrome partial downloads)
            crdownload_files = list(self.download_dir.glob("*.crdownload"))
            if crdownload_files:
                self.logger.debug("Download in progress...")
            
            time.sleep(0.5)
        
        self.logger.error(f"Download timeout after {timeout} seconds")
        return None
    
    def _parse_csv_file(self, csv_file_path: str, area: str) -> Dict[str, Any]:
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
            else:
                return self._parse_standard_csv(df, area)
                
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise
    
    def _parse_co2_csv(self, df: pd.DataFrame, area: str) -> Dict[str, Any]:
        """Parse CO2-specific CSV format"""
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
                time_str = str(row[time_col]).strip()
                
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
                        'time': time_str,
                        'value': actual_val,
                        'is_forecast': False
                    }
                    if forecast_val is not None:
                        data_point['forecast_value'] = forecast_val
                elif forecast_val is not None:
                    # Only forecast data
                    data_point = {
                        'time': time_str,
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
        """Parse standard CSV format for other data types"""
        time_series = []
        
        # Assume first column is time, second is value
        time_col = df.columns[0]
        value_col = df.columns[1]
        
        for _, row in df.iterrows():
            try:
                time_str = str(row[time_col]).strip()
                value = self._parse_number(row[value_col])
                
                if value is not None:
                    time_series.append({
                        'time': time_str,
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
            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    downloader = CSVDataDownloader(headless=False)  # Set True for headless
    
    # Test CO2 intensity download
    result = downloader.download_area_csv(
        area='co2_intensity',
        region='all',
        duration='day'
    )
    
    if result['success']:
        print("Success!")
        print(f"Downloaded {len(result['data']['time_series'])} data points")
        
        # Show first few points
        for i, point in enumerate(result['data']['time_series'][:5]):
            print(f"{i+1}: {point}")
    else:
        print(f"Failed: {result['error']}")
    
    # Cleanup
    downloader.cleanup_temp_files()


if __name__ == "__main__":
    main()