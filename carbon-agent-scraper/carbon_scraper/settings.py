BOT_NAME = "carbon_scraper"

SPIDER_MODULES = ["carbon_scraper.spiders"]
NEWSPIDER_MODULE = "carbon_scraper.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure Playwright
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Optimized settings for faster scraping
DOWNLOAD_DELAY = 0  # Small delay to be respectful
CONCURRENT_ITEMS = 5
CONCURRENT_REQUESTS = 10  # Keep low for single-page scraping
DOWNLOAD_TIMEOUT = 30  # Increased timeout for slow loading

# User agent
USER_AGENT = 'carbon_scraper (+https://sustainable-city-ai.eu)'

# Configure pipelines
ITEM_PIPELINES = {
    'carbon_scraper.pipelines.ValidationPipeline': 300,
    'carbon_scraper.pipelines.CachePipeline': 400,
    'carbon_scraper.pipelines.JsonExportPipeline': 500,
}

# Cache settings
CACHE_EXPIRY_MINUTES = 15
CACHE_DIR = 'cache'

# Fixed Playwright settings - removed problematic route configuration
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": [
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor",
    ],
}

# Increased timeouts for better reliability
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 5000

# Simplified browser contexts - removed the problematic route parameter
PLAYWRIGHT_CONTEXTS = {
    "default": {
        "ignore_https_errors": True,
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
}

# Logging
LOG_LEVEL = 'INFO'

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Disable HTTP cache for real-time data
HTTPCACHE_ENABLED = False

# Feed export encoding
FEED_EXPORT_ENCODING = "utf-8"

# Additional performance optimizations
AUTOTHROTTLE_ENABLED = True  # Disable auto-throttling for single requests
COOKIES_ENABLED = True  # May be needed for the website

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Request fingerprinting - can help with caching
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"