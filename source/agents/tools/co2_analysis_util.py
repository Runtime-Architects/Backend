"""
co2_analysis_util.py

Utility functions for carbon emission data analysis
"""

import os
import sys
import json
from datetime import datetime, date
from typing import Dict, List

import pandas as pd
import numpy as np

from eirgridscraper.run_eirgrid_downloader import main as eirgrid_main


def get_emission_data(startdate: str, enddate: str, region: str) -> float:
    """
    Fetch CO2 emission data for a given date range and region.

    If the corresponding JSON file exists locally, it reads the data from the file.
    Otherwise, it calls a CLI function to fetch the data and then processes it.

    Args:
        startdate (str): Start date in "YYYY-MM-DD" format.
        enddate (str): End date in "YYYY-MM-DD" format.
        region (str): Region code (e.g., "roi", "ni", "all").

    Returns:
        pd.DataFrame: DataFrame with columns ["timestamp", "value"] sorted by timestamp.

    Raises:
        Exception: If data could not be fetched or processed.
    """
    # Check if the file exists
    file_path = f"data/co2_intensity/co2_intensity_{region}_{startdate}_{enddate}.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            json_data = json.load(file)
        return process_json_data(json_data)

    else:
        if datetime.strptime(enddate, "%Y-%m-%d").date() == date.today():
            call_as_cli(startdate, enddate, region, True)
        else:
            call_as_cli(startdate, enddate, region, False)

        # Try to load the file again after calling the CLI
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                json_data = json.load(file)
            return process_json_data(json_data)
        else:
            raise Exception(
                f"Failed to fetch data from scrapper. File not found at {file_path}"
            )


def call_as_cli(startdate: str, enddate: str, region: str, forecast=False):
    """
    Simulate command line arguments to run the EirGrid downloader script.

    Args:
        startdate (str): Start date in "YYYY-MM-DD" format.
        enddate (str): End date in "YYYY-MM-DD" format.
        region (str): Region code.
        forecast (bool): If True, fetch forecast data; otherwise, fetch historical data.

    Returns:
        Any: Returns the result of eirgrid_main() function execution.
    """
    # Simulate command line arguments

    if forecast:
        sys.argv = [
            "run_eirgrid_downloader.py",
            "--areas",
            "co2_intensity",
            "--start",
            startdate,
            "--end",
            enddate,
            "--region",
            region,
            "--forecast",
            "--output-dir",
            "./data",
        ]
    else:
        sys.argv = [
            "run_eirgrid_downloader.py",
            "--areas",
            "co2_intensity",
            "--start",
            startdate,
            "--end",
            enddate,
            "--region",
            region,
            "--output-dir",
            "./data",
        ]

    # Run the main function
    return eirgrid_main()


def process_json_data(json_data: Dict) -> pd.DataFrame:
    """
    Process raw JSON CO2 emission data into a structured pandas DataFrame.

    Args:
        json_data (Dict): JSON data containing a "data" key with "time_series" entries.

    Returns:
        pd.DataFrame: DataFrame containing sorted timestamps and emission values.

    Raises:
        Exception: If the JSON data cannot be processed.
    """
    time_series = json_data["data"]["time_series"]
    df = pd.DataFrame(time_series)
    try:
        # Convert to datetime
        df["timestamp"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

    # Get the start, end date and region
    # start_date = json_data['metadata']['date_from']
    # end_date = json_data['metadata']['date_to']
    # region= find_region_str(json_data['metadata']['region'])

    except:
        raise Exception("Error processing json.")

    return df.sort_values("timestamp")[["timestamp", "value"]]


def get_view(start_date_str: str, end_date_str: str) -> str:
    """
    Determine the appropriate view type based on the date range.

    Args:
        start_date_str (str): Start date in "YYYY-MM-DD" format.
        end_date_str (str): End date in "YYYY-MM-DD" format.

    Returns:
        str: "day" for up to 6 days, "week" for 7-21 days, "month" for 22+ days.
    """

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Calculate total days (adding 1 to include both start and end dates)
    day_diff = (end_date - start_date).days + 1

    if day_diff <= 6:  # Up to 6 days
        return "day"
    elif day_diff <= 21:  # 7-21 days (up to 3 weeks)
        return "week"
    else:  # 22+ days (more than 3 weeks)
        return "month"


def str_to_date(date_str: str) -> datetime:
    """
    Convert a date string into a datetime.date object.

    Args:
        date_str (str): Date string in "YYYY-MM-DD" format.

    Returns:
        datetime.date: Corresponding date object.
    """
    # Example format: "2023-07-11" (adjust as needed)
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def get_days(start_date_str: str, end_date_str: str) -> int:
    """
    Calculate the number of days between two dates, inclusive.

    Args:
        start_date_str (str): Start date in "YYYY-MM-DD" format.
        end_date_str (str): End date in "YYYY-MM-DD" format.

    Returns:
        int: Number of days including both start and end dates.
    """
    start_date = str_to_date(start_date_str)
    end_date = str_to_date(end_date_str)

    return (end_date - start_date).days + 1


def find_region_str(region: str):
    """
    Convert a region code to its full name.

    Args:
        region (str): Region code. Expected values are:
                    - "roi" : Republic of Ireland
                    - "ni"  : Northern Ireland
                    - "all" : Republic of Ireland and Northern Ireland

    Returns:
        str: Full name of the region. Returns an empty string for unrecognized codes.
    """
    if region == "roi":
        return "Republic of Ireland"
    elif region == "all":
        return "Republic of Ireland and Nothern Ireland"
    elif region == "ni":
        return "Nothern Ireland"
    else:
        return ""


def calculate_thresholds(values: List) -> Dict[str, float]:
    """
    Calculate low and high thresholds for a list of numeric values using percentiles.

    Args:
        values (List[float]): A list of numeric values.

    Returns:
        Dict[str, float]: Dictionary containing "low" (33rd percentile) and "high" (66th percentile) thresholds.
    """
    q1, q3 = np.percentile(values, [33, 66])
    return {"low": q1, "high": q3}


def classify_intensity(values: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    """
    Classify numeric values into intensity categories based on thresholds.

    Args:
        values (pd.Series): Series of numeric values to classify.
        thresholds (Dict[str, float]): Dictionary with "low" and "high" threshold values.

    Returns:
        pd.Series: Series of categorical labels ("low", "medium", "high") corresponding to each value.
    """
    return pd.cut(
        values,
        bins=[-np.inf, thresholds["low"], thresholds["high"], np.inf],
        labels=["low", "medium", "high"],
    )


def format_time_range(start: datetime, end: datetime) -> str:
    """
    Format a time range for weekly view with consistent date formatting.

    Examples:
        - Same day: "01-Jan-23 14:00-16:00"
        - Cross-day: "01-Jan-23 22:00-02-Jan-23 04:00"
        - Same timestamp: "01-Jan-23 14:00"

    Args:
        start (datetime): Start of the range.
        end (datetime): End of the range.

    Returns:
        str: Formatted time range.
            - If start == end, returns only the start time.
            - If start and end are on the same date, returns "date start_time-end_time".
            - Otherwise, returns full "start_date start_time - end_date end_time".
    """
    date_format = "%d-%b-%y %H:%M"  # Consistent with daily format

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
