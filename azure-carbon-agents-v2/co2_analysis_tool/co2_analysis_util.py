import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

def process_json_data(json_data: Dict) -> pd.DataFrame:
        time_series = json_data['data']['time_series']
        df = pd.DataFrame(time_series)
        
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        # Get the start, end date and region
        start_date = json_data['metadata']['date_from']
        end_date = json_data['metadata']['date_to']
        region= find_region_str(json_data['metadata']['region'])
        
        return (df.sort_values('timestamp')[['timestamp', 'value']], start_date, end_date, region)


def str_to_date(date_str: str) -> datetime:
    # Example format: "2023-07-11" (adjust as needed)
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def get_days(start_date_str: str, end_date_str: str) -> int:
    start_date = str_to_date(start_date_str)
    end_date = str_to_date(end_date_str)
    
    return (end_date - start_date).days + 1


def find_region_str(region: str):
      if region=="roi":
            return "Republic of Ireland"
      elif region=="all":
            return "Republic of Ireland and Nothern Ireland"
      elif region=="ni":
            return "Nothern Ireland"
      else:
            return ""
      

def calculate_thresholds(values: List) -> Dict[str, float]:
        q1, q3 = np.percentile(values, [33, 66])
        return {'low': q1, 'high': q3}

def classify_intensity(values: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    return pd.cut(values,
                 bins=[-np.inf, thresholds['low'], thresholds['high'], np.inf],
                 labels=['low', 'medium', 'high'])

def format_time_range(start: datetime, end: datetime) -> str:
        """
        Formats time range for weekly view with consistent date formatting.
        Examples:
        - Same day: "01-Jan-23 14:00-16:00"
        - Cross-day: "01-Jan-23 22:00-02-Jan-23 04:00"
        - Same timestamp: "01-Jan-23 14:00"
        """
        date_format = '%d-%b-%y %H:%M'  # Consistent with daily format
        
        start_str = start.strftime(date_format)
        end_str = end.strftime(date_format)
        
        if start == end:
            return start_str
        elif start.date() == end.date():
            # Same day, show date only once
            return f"{start_str.split()[0]} {start_str.split()[1]} to {end_str.split()[1]}"
        else:
            # Cross-day, show full timestamps
            return f"{start_str} to {end_str}"
        


