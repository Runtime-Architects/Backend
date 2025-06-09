#!/usr/bin/env python
"""
Debug script to help troubleshoot download issues
Shows exactly whats going on on the page
"""

import time
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [DEBUG] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def debug_page_elements(driver, step_name):
    """Debug helper to check page elements"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== DEBUGGING: {step_name} ===")
    
    # Check for View In Table buttons
    view_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'View In Table')]")
    logger.info(f"Found {len(view_buttons)} 'View In Table' buttons")
    
    for i, btn in enumerate(view_buttons):
        try:
            is_displayed = btn.is_displayed()
            is_enabled = btn.is_enabled()
            text = btn.text
            logger.info(f"  Button {i+1}: '{text}' - Displayed: {is_displayed}, Enabled: {is_enabled}")
        except Exception as e:
            logger.info(f"  Button {i+1}: Error checking properties - {e}")
    
    # Check for download buttons
    download_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Download') or contains(text(), '.CSV')]")
    logger.info(f"Found {len(download_buttons)} download-related buttons")
    
    for i, btn in enumerate(download_buttons):
        try:
            is_displayed = btn.is_displayed()
            text = btn.text
            href = btn.get_attribute("href")
            download_attr = btn.get_attribute("download")
            logger.info(f"  Download {i+1}: '{text}' - Displayed: {is_displayed}")
            logger.info(f"    href: {href}")
            logger.info(f"    download: {download_attr}")
        except Exception as e:
            logger.info(f"  Download {i+1}: Error checking properties - {e}")
    
    # Check for tables
    tables = driver.find_elements(By.TAG_NAME, "table")
    logger.info(f"Found {len(tables)} tables")
    
    # Check accordion/collapsible sections
    accordions = driver.find_elements(By.CSS_SELECTOR, "[data-state]")
    logger.info(f"Found {len(accordions)} accordion sections")
    
    for i, acc in enumerate(accordions):
        try:
            state = acc.get_attribute("data-state")
            logger.info(f"  Accordion {i+1}: state = {state}")
        except Exception as e:
            logger.info(f"  Accordion {i+1}: Error - {e}")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Setup Chrome driver (visible mode for debugging)
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Download directory
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    
    prefs = {
        "download.default_directory": str(download_dir.absolute()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("🔍 Starting debug session...")
        
        # Navigate to page
        url = "https://www.smartgriddashboard.com/all/co2/?intensityduration=day&intensitydatefrom=08-Jun-2025&intensitydateto=08-Jun-2025"
        logger.info(f"Navigating to: {url}")
        driver.get(url)
        
        # Wait for page load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-gray.text-white"))
        )
        logger.info("✅ Page loaded")
        
        # Debug initial state
        debug_page_elements(driver, "INITIAL PAGE STATE")
        
        # Find View In Table buttons
        view_buttons = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[contains(text(), 'View In Table')]"))
        )
        
        if not view_buttons:
            logger.error("❌ No 'View In Table' buttons found!")
            return
        
        logger.info(f"✅ Found {len(view_buttons)} 'View In Table' buttons")
        
        # Try to click the first button
        first_button = view_buttons[0]
        
        # Scroll into view
        logger.info("📜 Scrolling button into view...")
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", first_button)
        time.sleep(2)
        
        # Take screenshot before click
        screenshot_path = download_dir / "before_click.png"
        driver.save_screenshot(str(screenshot_path))
        logger.info(f"📸 Screenshot saved: {screenshot_path}")
        
        debug_page_elements(driver, "BEFORE CLICKING VIEW IN TABLE")
        
        # Try clicking
        try:
            logger.info("🖱️ Attempting to click 'View In Table' button...")
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(first_button))
            first_button.click()
            logger.info("✅ Successfully clicked 'View In Table' button")
        except Exception as e:
            logger.warning(f"⚠️ Regular click failed: {e}")
            logger.info("🔧 Trying JavaScript click...")
            driver.execute_script("arguments[0].click();", first_button)
            logger.info("✅ JavaScript click executed")
        
        # Wait for accordion to expand
        logger.info("⏳ Waiting for table section to expand...")
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )
            logger.info("✅ Table section expanded")
        except Exception as e:
            logger.warning(f"⚠️ Table didn't appear: {e}")
        
        time.sleep(3)  # Wait for everything to load
        
        # Debug after clicking
        debug_page_elements(driver, "AFTER CLICKING VIEW IN TABLE")
        
        # Take screenshot after expansion
        screenshot_path = download_dir / "after_expand.png"
        driver.save_screenshot(str(screenshot_path))
        logger.info(f"📸 Screenshot saved: {screenshot_path}")
        
        # Look for download button
        logger.info("🔍 Looking for download button...")
        
        strategies = [
            "//a[contains(text(), 'Download .CSV')]",
            "//a[contains(@class, 'bg-primary') and contains(text(), 'Download')]",
            "//a[@download and contains(@href, 'blob:')]",
            "//a[contains(text(), '.CSV')]"
        ]
        
        download_button = None
        for i, xpath in enumerate(strategies):
            try:
                logger.info(f"Trying strategy {i+1}: {xpath}")
                buttons = driver.find_elements(By.XPATH, xpath)
                
                for btn in buttons:
                    if btn.is_displayed():
                        download_button = btn
                        logger.info(f"✅ Found visible download button with strategy {i+1}")
                        logger.info(f"   Text: '{btn.text}'")
                        logger.info(f"   Download attr: {btn.get_attribute('download')}")
                        break
                
                if download_button:
                    break
                    
            except Exception as e:
                logger.info(f"Strategy {i+1} failed: {e}")
        
        if download_button:
            logger.info("🎉 Download button found and ready!")
            
            # Show final page state
            debug_page_elements(driver, "FINAL STATE - READY FOR DOWNLOAD")
            
            # Optional: Try the download
            try_download = input("\n🤔 Try downloading now? (y/n): ").lower().strip() == 'y'
            
            if try_download:
                logger.info("📥 Attempting download...")
                
                # Use the same enhanced clicking logic as the main downloader
                try:
                    # Scroll download button into view
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", download_button)
                    time.sleep(1)
                    
                    # Try regular click first
                    WebDriverWait(driver, 5).until(EC.element_to_be_clickable(download_button))
                    download_button.click()
                    logger.info("✅ Download initiated with regular click!")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Regular click failed: {e}")
                    logger.info("🔧 Trying JavaScript click...")
                    
                    try:
                        # Use JavaScript click as fallback
                        driver.execute_script("arguments[0].click();", download_button)
                        logger.info("✅ Download initiated with JavaScript click!")
                    except Exception as js_error:
                        logger.error(f"❌ JavaScript click also failed: {js_error}")
                        
                        # Try one more approach - direct URL navigation
                        try:
                            href = download_button.get_attribute("href")
                            if href and href.startswith("blob:"):
                                logger.info("🔧 Trying direct URL navigation...")
                                driver.get(href)
                                logger.info("✅ Download initiated via URL navigation!")
                            else:
                                logger.error("❌ No valid href found for direct navigation")
                        except Exception as nav_error:
                            logger.error(f"❌ URL navigation failed: {nav_error}")
                
                time.sleep(5)  # Wait for download
        else:
            logger.error("❌ Download button not found!")
            
            # Scroll to bottom to see if button is hidden
            logger.info("📜 Scrolling to bottom of page...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Take final screenshot
            screenshot_path = download_dir / "final_state.png"
            driver.save_screenshot(str(screenshot_path))
            logger.info(f"📸 Final screenshot saved: {screenshot_path}")
            
            debug_page_elements(driver, "AFTER SCROLLING TO BOTTOM")
        
        # Keep browser open for manual inspection
        input("\n🔍 Press Enter to close the browser and end debug session...")
        
    except Exception as e:
        logger.error(f"❌❌❌ Debug session failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            driver.quit()
        except:
            pass
        
        logger.info("Debug session ended")

if __name__ == "__main__":
    main()