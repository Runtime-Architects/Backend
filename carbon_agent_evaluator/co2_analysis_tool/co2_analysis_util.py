import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import os
import sys
from scraper_tools.run_eirgrid_downloader import main as eirgrid_main


def get_emission_data(startdate: str, enddate: str, region: str) -> float:
    # Check if the file exists
    file_path = f'data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:  
            json_data = json.load(file)     
        return process_json_data(json_data)
    
    else:
        if(datetime.strptime(enddate, "%Y-%m-%d").date() == datetime.today()):
            call_as_cli(startdate, enddate, region, True)
        else:
            call_as_cli(startdate, enddate, region, False)
        
        # Try to load the file again after calling the CLI
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:  
                json_data = json.load(file)     
            return process_json_data(json_data)
        else:
            raise Exception(f"Failed to fetch data from scrapper. File not found at {file_path}")


def call_as_cli(startdate: str, enddate: str, region: str, forecast=False):
            # Simulate command line arguments

            if forecast:
                sys.argv = [
                    'run_eirgrid_downloader.py',
                    '--areas', 'co2_intensity',
                    '--start', startdate,
                    '--end', enddate,
                    '--region', region,
                    '--forecast',
                    '--output-dir', './data'
                ]
            else:
                sys.argv = [
                    'run_eirgrid_downloader.py',
                    '--areas', 'co2_intensity',
                    '--start', startdate,
                    '--end', enddate,
                    '--region', region,
                    '--output-dir', './data'
                ]
            
            # Run the main function
            return eirgrid_main()



def process_json_data(json_data: Dict) -> tuple:
    """
    Process JSON data and return (dataframe, start_date, end_date, region)
    """
    time_series = json_data['data']['time_series']
    df = pd.DataFrame(time_series)
    try:
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        # Get the start, end date and region from metadata
        start_date = json_data['metadata']['date_from']
        end_date = json_data['metadata']['date_to']
        region = json_data['metadata']['region']

    except Exception as e:
        raise Exception(f"Error processing json: {str(e)}")
    
    # Return tuple as expected by the analysis modules
    return (
        df.sort_values('timestamp')[['timestamp', 'value']], 
        start_date, 
        end_date, 
        region
    )



def get_view(start_date_str: str, end_date_str: str) -> str:
    """
    Determines the appropriate view based on the date range:
    - 'day' view: 1-6 days
    - 'week' view: 7-21 days (up to 3 weeks)
    - 'month' view: 22+ days (more than 3 weeks)
    
    Returns:
        str: One of 'day', 'week', or 'month'
    """

    start_date= datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date= datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # Calculate total days (adding 1 to include both start and end dates)
    day_diff = (end_date - start_date).days + 1
    
    if day_diff <= 6:  # Up to 6 days
        return "day"
    elif day_diff <= 21:  # 7-21 days (up to 3 weeks)
        return "week"
    else:  # 22+ days (more than 3 weeks)
        return "month"



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
        

