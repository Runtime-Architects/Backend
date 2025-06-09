import scrapy
from scrapy_playwright.page import PageMethod
from carbon_scraper.items import CO2IntensityItem
from datetime import datetime
import re
import json
import logging
import asyncio

class CO2Spider(scrapy.Spider):
    name = 'co2_intensity'
    allowed_domains = ['smartgriddashboard.com']
    
    def __init__(self, date_from=None, date_to=None, region='all', *args, **kwargs):
        super(CO2Spider, self).__init__(*args, **kwargs)
        
        # Set default dates if not provided
        if date_from is None:
            self.date_from = datetime.now().strftime('%d-%b-%Y')
        else:
            self.date_from = date_from
            
        if date_to is None:
            self.date_to = datetime.now().strftime('%d-%b-%Y')
        else:
            self.date_to = date_to
            
        self.region = region
        self.logger.info(f"Scraping CO2 data for {self.region} from {self.date_from} to {self.date_to}")
    
    async def start(self):
        # Construct URL with parameters
        base_url = f'https://www.smartgriddashboard.com/{self.region}/co2/'
        
        # For specific date ranges, add query parameters
        url_params = {
            'intensitydatefrom': self.date_from,
            'intensitydateto': self.date_to,
            'intensityduration': 'day'
        }
        
        # Build URL with parameters
        param_string = '&'.join([f'{k}={v}' for k, v in url_params.items()])
        url = f"{base_url}?{param_string}"
        
        self.logger.info(f"Requesting URL: {url}")
        
        yield scrapy.Request(
            url,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_methods": [
                    # Wait for the page to load completely
                    PageMethod("wait_for_load_state", "networkidle", timeout=20000),
                    # Wait for the main content to appear
                    PageMethod("wait_for_selector", "div.bg-gray.text-white", timeout=15000),
                    # Wait for the table button to appear
                    PageMethod("wait_for_selector", "button:has-text('View In Table')", timeout=10000),
                    # Small wait to ensure everything is loaded
                    PageMethod("wait_for_timeout", 2000),
                ],
            },
            callback=self.parse,
            errback=self.errback,
        )
    
    async def parse(self, response):
        page = response.meta["playwright_page"]
        
        # Create item
        item = CO2IntensityItem()
        item['timestamp'] = datetime.now().isoformat()
        item['region'] = self.region
        item['date_from'] = self.date_from
        item['date_to'] = self.date_to
        
        try:
            self.logger.info("Page loaded successfully, starting data extraction...")
            
            # Extract current values from the cards
            self.logger.info("Extracting current values from cards...")
            item['latest_intensity'] = await self._extract_card_by_position(page, 0)
            item['todays_low_intensity'] = await self._extract_card_by_position(page, 1)
            item['latest_emissions'] = await self._extract_card_by_position(page, 2)
            
            # Extract all time series data with improved pagination
            self.logger.info("Starting time series data extraction...")
            item['time_series_data'] = await self._extract_all_table_data_improved(page)
            
            # Log summary
            self.logger.info(f"Extraction completed!")
            self.logger.info(f"Latest intensity: {item.get('latest_intensity')}")
            self.logger.info(f"Time series points: {len(item.get('time_series_data', []))}")
            
            yield item
            
        except Exception as e:
            self.logger.error(f"Error parsing CO2 data: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            await page.close()
    
    async def _extract_card_by_position(self, page, position):
        """Extract value from cards by their position (0=first, 1=second, 2=third)"""
        try:
            # Get all the metric cards
            cards = await page.query_selector_all("div.bg-gray.text-white.flex.flex-col")
            
            if len(cards) > position:
                card = cards[position]
                
                # Get the value element within the card
                value_element = await card.query_selector("p.font-metaWeb")
                if value_element:
                    value_text = await value_element.inner_text()
                    # Parse the value and unit
                    match = re.search(r'(\d+)\s*(gCO2/kWh|tCO2/hr)', value_text)
                    if match:
                        return {
                            'value': int(match.group(1)),
                            'unit': match.group(2)
                        }
                    else:
                        self.logger.warning(f"Could not parse value from card {position}: {value_text}")
                        
        except Exception as e:
            self.logger.warning(f"Failed to extract card at position {position}: {str(e)}")
        return None
    
    async def _extract_all_table_data_improved(self, page):
        """Extract all time series data with improved pagination handling"""
        all_time_series = []
        
        try:
            # Find and click the first "View In Table" button for intensity data
            self.logger.info("Looking for 'View In Table' button...")
            await page.wait_for_selector("button:has-text('View In Table')", timeout=15000)
            view_buttons = await page.query_selector_all("button:has-text('View In Table')")
            
            if not view_buttons:
                self.logger.warning("No 'View In Table' buttons found")
                return all_time_series
                
            self.logger.info(f"Found {len(view_buttons)} 'View In Table' buttons")
            
            # Click the first button (CO2 Intensity table)
            await view_buttons[0].click()
            self.logger.info("Clicked 'View In Table' button")
            
            # Wait for table to appear with better selector and longer timeout
            await page.wait_for_selector("table tbody tr", timeout=15000)
            self.logger.info("Table appeared successfully")
            
            # Small wait for table to fully load
            await page.wait_for_timeout(2000)
            
            # Get total pages more reliably
            total_pages = await self._get_total_pages(page)
            self.logger.info(f"Detected {total_pages} total pages to process")
            
            if total_pages == 0:
                self.logger.warning("No pages detected, attempting to extract current page only")
                page_data = await self._extract_current_page_data(page)
                return page_data
            
            # Extract data from all pages with improved error handling
            for page_num in range(1, total_pages + 1):
                try:
                    self.logger.info(f"Processing page {page_num}/{total_pages}")
                    
                    # Navigate to page if not the first page
                    if page_num > 1:
                        success = await self._navigate_to_page(page, page_num)
                        if not success:
                            self.logger.warning(f"Failed to navigate to page {page_num}, stopping pagination")
                            break
                    
                    # Extract data from current page
                    page_data = await self._extract_current_page_data(page)
                    all_time_series.extend(page_data)
                    
                    self.logger.info(f"Successfully extracted page {page_num}/{total_pages} - {len(page_data)} rows")
                    
                    # Small delay between pages to be respectful
                    if page_num < total_pages:  # Don't wait after the last page
                        await page.wait_for_timeout(800)
                    
                except Exception as e:
                    self.logger.error(f"Error extracting page {page_num}: {str(e)}")
                    # Continue with next page instead of breaking completely
                    continue
                    
        except asyncio.TimeoutError:
            self.logger.error("Timeout waiting for table elements")
        except Exception as e:
            self.logger.error(f"Error during table extraction: {str(e)}")
            
        self.logger.info(f"Total data points extracted: {len(all_time_series)}")
        return all_time_series
    
    async def _get_total_pages(self, page):
        """Get total number of pages more reliably"""
        try:
            # Try to find the "Last" button which contains total pages
            last_button = await page.query_selector("button:has-text('Last')")
            if last_button:
                last_text = await last_button.inner_text()
                match = re.search(r'Last\s*\((\d+)\)', last_text)
                if match:
                    return int(match.group(1))
            
            # Fallback: count visible page buttons
            page_buttons = await page.query_selector_all("button:not(:has-text('Last')):not(:has-text('First')):not(:has-text('<')):not(:has-text('>'))")
            # Filter to only numeric buttons
            numeric_buttons = []
            for button in page_buttons:
                text = await button.inner_text()
                if text.strip().isdigit():
                    numeric_buttons.append(int(text.strip()))
            
            if numeric_buttons:
                return max(numeric_buttons)
                
        except Exception as e:
            self.logger.warning(f"Could not determine total pages: {e}")
        
        return 1  # Default to 1 page if we can't determine
    
    async def _navigate_to_page(self, page, page_num):
        """Navigate to a specific page with better error handling"""
        try:
            # Find the page button by text
            page_buttons = await page.query_selector_all(f"button")
            target_button = None
            
            for button in page_buttons:
                text = await button.inner_text()
                if text.strip() == str(page_num):
                    # Check if this button is not already active
                    button_classes = await button.get_attribute('class') or ''
                    if 'bg-primary' not in button_classes:
                        target_button = button
                        break
            
            if target_button:
                # Scroll button into view
                await target_button.scroll_into_view_if_needed()
                
                # Wait a bit and then click
                await page.wait_for_timeout(200)
                await target_button.click()
                
                # Wait for table to update with a more specific condition
                await page.wait_for_function(
                    f"""() => {{
                        const activeButton = document.querySelector('button.bg-primary');
                        return activeButton && activeButton.textContent.trim() === '{page_num}';
                    }}""",
                    timeout=5000
                )
                
                # Additional wait for content to load
                await page.wait_for_timeout(1000)
                return True
            else:
                self.logger.warning(f"Could not find button for page {page_num}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to navigate to page {page_num}: {str(e)}")
            return False
    
    async def _extract_current_page_data(self, page):
        """Extract data from the currently visible page"""
        page_data = []
        
        try:
            # Wait for table rows to be present
            await page.wait_for_selector("tbody tr[data-state='false']", timeout=5000)
            
            # Get all data rows (excluding headers)
            rows = await page.query_selector_all("tbody tr[data-state='false']")
            
            for row in rows:
                try:
                    cells = await row.query_selector_all("td")
                    if len(cells) >= 2:
                        time_text = await cells[0].inner_text()
                        intensity_text = await cells[1].inner_text()
                        
                        # Check if there's forecast data (3rd column)
                        forecast_text = None
                        if len(cells) >= 3:
                            forecast_cell_text = await cells[2].inner_text()
                            if forecast_cell_text.strip():  # Only if not empty
                                forecast_text = forecast_cell_text
                        
                        point = {
                            'time': time_text.strip(),
                            'intensity': self._parse_number(intensity_text),
                            'intensity_forecast': self._parse_number(forecast_text) if forecast_text else None
                        }
                        page_data.append(point)
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting row data: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error extracting current page data: {e}")
            
        return page_data
    
    def _parse_number(self, text):
        """Parse number from text, handling empty strings"""
        if not text or text.strip() == '':
            return None
        try:
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', text.strip())
            if cleaned == '':
                return None
            if '.' in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            return None
    
    async def errback(self, failure):
        self.logger.error(f"Request failed: {failure}")
        if hasattr(failure.value, '__class__'):
            error_type = failure.value.__class__.__name__
            if 'TimeoutError' in error_type or 'timeout' in str(failure.value).lower():
                self.logger.error("The request timed out. This could be due to:")
                self.logger.error("- Slow internet connection")
                self.logger.error("- Website being overloaded")
                self.logger.error("- Firewall or proxy issues")
                self.logger.error("Try running the scraper again in a few minutes.")
        
        # Close any open pages
        page = failure.request.meta.get("playwright_page")
        if page:
            try:
                await page.close()
            except:
                pass