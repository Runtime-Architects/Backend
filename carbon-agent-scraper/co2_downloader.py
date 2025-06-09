#!/usr/bin/env python
"""
CO2 Data Downloader using CSV download approach
Much more efficient than scraping individual table pages
"""

import os
import json
import csv
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

class CO2DataDownloader:
    """Downloads CO2 intensity data from Smart Grid Dashboard using CSV download"""
    
    def __init__(self, download_dir: Optional[str] = None, headless: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Set up download directory
        if download_dir is None:
            self.download_dir = Path("downloads")
        else:
            self.download_dir = Path(download_dir)
        
        self.download_dir.mkdir(exist_ok=True)
        
        # Set up data directory
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.headless = headless
        self.driver = None
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome driver with download preferences and automatic driver management"""
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
        
        # Automatically install and setup ChromeDriver
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
    
    def download_co2_data(self, 
                         duration: str = "day", 
                         date_from: Optional[str] = None, 
                         date_to: Optional[str] = None,
                         region: str = "all") -> Optional[str]:
        """
        Download CO2 data CSV from the website
        
        Args:
            duration: 'day', 'week', or 'month'
            date_from: Start date in 'DD-Mon-YYYY' format (optional)
            date_to: End date in 'DD-Mon-YYYY' format (optional)
            region: 'all', 'roi', or 'ni'
            
        Returns:
            Path to downloaded CSV file or None if failed
        """
        
        self.driver = self._setup_driver()
        
        try:
            # Build URL
            url = self._build_url(duration, date_from, date_to, region)
            self.logger.info(f"Navigating to: {url}")
            
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load completely
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-gray.text-white"))
            )
            self.logger.info("Page loaded successfully")
            
            # Wait a bit more for all elements to settle
            time.sleep(3)
            
            # Find "View In Table" buttons
            view_table_buttons = WebDriverWait(self.driver, 15).until(
                EC.presence_of_all_elements_located((By.XPATH, "//button[contains(text(), 'View In Table')]"))
            )
            
            if not view_table_buttons:
                raise Exception("No 'View In Table' buttons found")
            
            self.logger.info(f"Found {len(view_table_buttons)} 'View In Table' buttons")
            
            # Get the first button (CO2 Intensity table)
            first_button = view_table_buttons[0]
            
            # Scroll the button into view
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", first_button)
            time.sleep(2)
            
            # Try to click using JavaScript if regular click fails
            try:
                # Wait for button to be clickable
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable(first_button)
                )
                first_button.click()
                self.logger.info("Clicked 'View In Table' button")
            except Exception as e:
                self.logger.warning(f"Regular click failed: {e}. Trying JavaScript click...")
                # Use JavaScript click as fallback
                self.driver.execute_script("arguments[0].click();", first_button)
                self.logger.info("Clicked 'View In Table' button using JavaScript")
            
            # Wait for the accordion to expand and table to appear
            self.logger.info("Waiting for table section to expand...")
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )
            self.logger.info("Table section expanded successfully")
            
            # Wait a bit more for the download button to become available
            time.sleep(3)
            
            # Find the download CSV button with multiple strategies
            download_button = None
            strategies = [
                "//a[contains(text(), 'Download .CSV')]",
                "//a[contains(@class, 'bg-primary') and contains(text(), 'Download')]",
                "//a[@download and contains(@href, 'blob:')]",
                "//a[contains(text(), '.CSV')]"
            ]
            
            for i, xpath in enumerate(strategies):
                try:
                    self.logger.info(f"Trying download button strategy {i+1}/{len(strategies)}...")
                    download_button = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    
                    # Check if button is visible
                    if download_button.is_displayed():
                        self.logger.info(f"Found visible download button using strategy {i+1}")
                        break
                    else:
                        self.logger.info(f"Button found but not visible, trying next strategy...")
                        download_button = None
                        
                except Exception as e:
                    self.logger.info(f"Strategy {i+1} failed: {e}")
                    continue
            
            if not download_button:
                # Try scrolling down to find hidden download button
                self.logger.info("Download button not found, scrolling to find it...")
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Try again after scrolling
                download_button = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Download .CSV')]"))
                )
            
            if not download_button:
                raise Exception("Download .CSV button not found after multiple attempts")
            
            # Get the expected filename from the download attribute
            expected_filename = download_button.get_attribute("download")
            if not expected_filename:
                expected_filename = f"Co2Intensity_{datetime.now().strftime('%d.%b.%Y')}.csv"
            
            self.logger.info(f"Expected download filename: {expected_filename}")
            
            # Scroll download button into view
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", download_button)
            time.sleep(1)
            
            # Click download button with retry logic
            download_success = False
            for attempt in range(3):
                try:
                    self.logger.info(f"Attempting to click download button (attempt {attempt + 1}/3)...")
                    
                    # Wait for button to be clickable
                    WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable(download_button)
                    )
                    
                    download_button.click()
                    self.logger.info("Clicked download button successfully")
                    download_success = True
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                    if attempt < 2:  # Not the last attempt
                        # Try JavaScript click
                        try:
                            self.driver.execute_script("arguments[0].click();", download_button)
                            self.logger.info("Used JavaScript click as fallback")
                            download_success = True
                            break
                        except Exception as js_error:
                            self.logger.warning(f"JavaScript click also failed: {js_error}")
                            
                            # Try direct URL navigation as last resort
                            try:
                                href = download_button.get_attribute("href")
                                if href and href.startswith("blob:"):
                                    self.logger.info("Trying direct URL navigation...")
                                    self.driver.get(href)
                                    self.logger.info("Used URL navigation as fallback")
                                    download_success = True
                                    break
                            except Exception as nav_error:
                                self.logger.warning(f"URL navigation failed: {nav_error}")
                                time.sleep(1)
                    else:
                        raise Exception(f"Failed to click download button after 3 attempts: {e}")
            
            if not download_success:
                raise Exception("Failed to initiate download")
            
            # Wait for file to download
            downloaded_file = self._wait_for_download(expected_filename)
            
            if downloaded_file:
                self.logger.info(f"Successfully downloaded: {downloaded_file}")
                return downloaded_file
            else:
                raise Exception("Download failed or timed out")
                
        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            
            # Take a screenshot for debugging
            try:
                screenshot_path = Path("downloads") / f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.driver.save_screenshot(str(screenshot_path))
                self.logger.info(f"Error screenshot saved: {screenshot_path}")
            except:
                pass
                
            return None
            
        finally:
            if self.driver:
                self.driver.quit()
    
    def _build_url(self, duration: str, date_from: Optional[str], date_to: Optional[str], region: str) -> str:
        """Build the URL with appropriate parameters"""
        base_url = f"https://www.smartgriddashboard.com/{region}/co2/"
        
        params = [f"intensityduration={duration}"]
        
        if date_from:
            params.append(f"intensitydatefrom={date_from}")
        if date_to:
            params.append(f"intensitydateto={date_to}")
            
        if params:
            return f"{base_url}?{'&'.join(params)}"
        else:
            return base_url
    
    def _wait_for_download(self, expected_filename: str, timeout: int = 30) -> Optional[str]:
        """Wait for file to be downloaded"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if file exists
            file_path = self.download_dir / expected_filename
            if file_path.exists() and file_path.stat().st_size > 0:
                return str(file_path)
            
            # Check for .crdownload files (Chrome partial downloads)
            crdownload_files = list(self.download_dir.glob("*.crdownload"))
            if crdownload_files:
                self.logger.info("Download in progress...")
            
            time.sleep(0.5)
        
        self.logger.error(f"Download timeout after {timeout} seconds")
        return None
    
    def parse_csv_to_json(self, csv_file_path: str) -> Dict:
        """Parse downloaded CSV file and convert to JSON format"""
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            # Expected columns: DATE & TIME, CO2 INTENSITY (gCO2/kWh), CO2 FORECAST (gCO2/kWh), *REGION*
            time_col = df.columns[0]  # DATE & TIME
            intensity_col = df.columns[1]  # CO2 INTENSITY
            forecast_col = df.columns[2] if len(df.columns) > 2 else None  # CO2 FORECAST
            region_col = df.columns[3] if len(df.columns) > 3 else None  # REGION
            
            self.logger.info(f"CSV columns: {list(df.columns)}")
            self.logger.info(f"Total rows: {len(df)}")
            
            # Parse data
            time_series = []
            region_value = None
            
            for _, row in df.iterrows():
                try:
                    # Parse time
                    time_str = str(row[time_col]).strip()
                    
                    # Parse intensity
                    intensity_val = self._parse_number(row[intensity_col])
                    
                    # Parse forecast (optional)
                    forecast_val = None
                    if forecast_col and pd.notna(row[forecast_col]):
                        forecast_val = self._parse_number(row[forecast_col])
                    
                    # Get region (from first row)
                    if region_value is None and region_col and pd.notna(row[region_col]):
                        region_value = str(row[region_col]).strip().replace('*', '')
                    
                    time_series.append({
                        'time': time_str,
                        'intensity': intensity_val,
                        'intensity_forecast': forecast_val
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing row: {e}")
                    continue
            
            # Create JSON structure
            result = {
                'metadata': {
                    'scraped_at': datetime.now().isoformat(),
                    'region': region_value or 'all',
                    'source_file': Path(csv_file_path).name,
                    'total_points': len(time_series)
                },
                'time_series': time_series
            }
            
            self.logger.info(f"Parsed {len(time_series)} data points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV: {e}")
            raise
    
    def _parse_number(self, value) -> Optional[float]:
        """Parse number from CSV value"""
        if pd.isna(value) or str(value).strip() == '':
            return None
        
        try:
            return float(str(value).strip())
        except ValueError:
            return None
    
    def save_json_data(self, data: Dict, filename: Optional[str] = None) -> str:
        """Save parsed data to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            region = data['metadata'].get('region', 'unknown')
            filename = f"co2_data_{region}_{timestamp}.json"
        
        file_path = self.data_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved data to: {file_path}")
        return str(file_path)
    
    def merge_with_existing_data(self, new_data: Dict, existing_file: Optional[str] = None) -> Dict:
        """Merge new data with existing data, avoiding duplicates"""
        
        if existing_file and Path(existing_file).exists():
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Get existing time points
                existing_times = {point['time'] for point in existing_data.get('time_series', [])}
                
                # Add only new time points
                new_points = [
                    point for point in new_data['time_series'] 
                    if point['time'] not in existing_times
                ]
                
                if new_points:
                    # Merge data
                    merged_series = existing_data.get('time_series', []) + new_points
                    
                    # Sort by time
                    merged_series.sort(key=lambda x: self._parse_time(x['time']))
                    
                    # Update metadata
                    merged_data = {
                        'metadata': {
                            **existing_data.get('metadata', {}),
                            'last_updated': datetime.now().isoformat(),
                            'total_points': len(merged_series)
                        },
                        'time_series': merged_series
                    }
                    
                    self.logger.info(f"Merged {len(new_points)} new points with existing data")
                    return merged_data
                else:
                    self.logger.info("No new data points to add")
                    return existing_data
                    
            except Exception as e:
                self.logger.warning(f"Error reading existing data: {e}")
                return new_data
        else:
            return new_data
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime for sorting"""
        try:
            return datetime.strptime(time_str, '%d %B %Y %H:%M')
        except ValueError:
            try:
                return datetime.strptime(time_str, '%d %b %Y %H:%M')
            except ValueError:
                return datetime.now()  # Fallback
    
    def download_and_process(self, 
                           duration: str = "day",
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None,
                           region: str = "all",
                           merge_existing: bool = True) -> Optional[str]:
        """
        Complete workflow: download CSV, parse to JSON, and optionally merge with existing data
        
        Returns:
            Path to final JSON file or None if failed
        """
        
        try:
            # Download CSV
            csv_file = self.download_co2_data(duration, date_from, date_to, region)
            if not csv_file:
                return None
            
            # Parse to JSON
            json_data = self.parse_csv_to_json(csv_file)
            
            # Merge with existing data if requested
            if merge_existing:
                # Find existing file for this region and duration
                pattern = f"co2_data_{region}_*.json"
                existing_files = list(self.data_dir.glob(pattern))
                if existing_files:
                    latest_file = max(existing_files, key=lambda f: f.stat().st_mtime)
                    json_data = self.merge_with_existing_data(json_data, str(latest_file))
            
            # Save final JSON
            json_file = self.save_json_data(json_data)
            
            # Clean up CSV file
            try:
                os.remove(csv_file)
                self.logger.info("Cleaned up temporary CSV file")
            except:
                pass
            
            return json_file
            
        except Exception as e:
            self.logger.error(f"Error in download and process workflow: {e}")
            return None


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    downloader = CO2DataDownloader(headless=True)  # Set False to see browser
    
    # Example: Download today's data
    result = downloader.download_and_process(
        duration="day",
        region="all",
        merge_existing=True
    )
    
    if result:
        print(f"Success! Data saved to: {result}")
    else:
        print("Download failed")


if __name__ == "__main__":
    main()