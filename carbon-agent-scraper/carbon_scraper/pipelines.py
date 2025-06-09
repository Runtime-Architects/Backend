import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

class ValidationPipeline:
    """Validate scraped data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item, spider):
        # Validate required fields
        required_fields = ['timestamp', 'region']
        for field in required_fields:
            if field not in item or item[field] is None:
                self.logger.warning(f"Missing required field: {field}")
        
        # Log what we got
        self.logger.info(f"Processed item with {len(item.get('time_series_data', []))} time series points")
        
        return item

class CachePipeline:
    """Cache scraped data to avoid excessive requests"""
    
    def __init__(self):
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def process_item(self, item, spider):
        # Create cache filename based on region and date
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cache_file = self.cache_dir / f"co2_{item['region']}_{timestamp}.json"
        
        # Convert item to dict and handle any datetime objects
        item_dict = dict(item)
        
        # Remove raw_data to save space (optional)
        if 'raw_data' in item_dict:
            del item_dict['raw_data']
        
        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(item_dict, f, indent=2, default=str)
        
        self.logger.info(f"Cached data to {cache_file}")
        self.logger.info(f"File size: {cache_file.stat().st_size} bytes")
        
        return item
    
    def get_cached_data(self, region, date_from, date_to, max_age_minutes=15):
        """Retrieve cached data if it's still fresh"""
        # Look for most recent cache file
        pattern = f"co2_{region}_*.json"
        cache_files = list(self.cache_dir.glob(pattern))
        
        if not cache_files:
            return None
        
        # Get most recent file
        latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
        
        # Check if cache is still fresh
        file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
        if file_age < timedelta(minutes=max_age_minutes):
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None

class JsonExportPipeline:
    """Export data to JSON files in a structured format"""
    
    def __init__(self):
        self.export_dir = Path('data')
        self.export_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def process_item(self, item, spider):
        # Create a more readable export file
        date_str = item.get('date_from', datetime.now().strftime('%d-%b-%Y'))
        export_file = self.export_dir / f"co2_data_{item['region']}_{date_str}.json"
        
        # Structure the data nicely
        export_data = {
            'metadata': {
                'scraped_at': item['timestamp'],
                'region': item['region'],
                'date_range': {
                    'from': item.get('date_from'),
                    'to': item.get('date_to')
                }
            },
            'current_values': {
                'latest_intensity': item.get('latest_intensity'),
                'todays_low': item.get('todays_low_intensity'),
                'latest_emissions': item.get('latest_emissions')
            },
            'time_series': item.get('time_series_data', [])
        }
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported structured data to {export_file}")
        
        return item