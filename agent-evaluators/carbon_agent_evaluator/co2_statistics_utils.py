"""
Utility functions for CO2 statistics calculations.
This module provides centralized functions to avoid duplicate calculations across the codebase.
"""

from typing import Dict, List


def calculate_co2_statistics(data_points: List[Dict]) -> Dict:
    """
    Centralized function to calculate CO2 statistics from data points.
    
    Args:
        data_points: List of dictionaries with 'value' and optionally 'time' keys
        
    Returns:
        Dictionary containing:
        - min_value: Minimum CO2 value
        - max_value: Maximum CO2 value  
        - avg_value: Average CO2 value (float)
        - daily_average: Average CO2 value as integer
        - min_point: Data point with minimum value
        - max_point: Data point with maximum value
        - optimal_times: List of times with minimum value (up to 3)
        - peak_times: List of times with maximum value (up to 3)
    """
    if not data_points:
        return {
            "min_value": 0,
            "max_value": 0,
            "avg_value": 0,
            "daily_average": 0,
            "min_point": None,
            "max_point": None,
            "optimal_times": [],
            "peak_times": []
        }
    
    values = [point.get('value', 0) for point in data_points if 'value' in point]
    
    if not values:
        return {
            "min_value": 0,
            "max_value": 0,
            "avg_value": 0,
            "daily_average": 0,
            "min_point": None,
            "max_point": None,
            "optimal_times": [],
            "peak_times": []
        }
    
    min_value = min(values)
    max_value = max(values)
    avg_value = sum(values) / len(values)
    
    # Find all points with min/max values
    min_entries = [point for point in data_points if point.get('value') == min_value]
    max_entries = [point for point in data_points if point.get('value') == max_value]
    
    # Get the first min/max points
    min_point = min_entries[0] if min_entries else None
    max_point = max_entries[0] if max_entries else None
    
    # Extract times
    optimal_times = [entry.get('time', '') for entry in min_entries if entry.get('time')][:3]
    peak_times = [entry.get('time', '') for entry in max_entries if entry.get('time')][:3]
    
    return {
        "min_value": min_value,
        "max_value": max_value,
        "avg_value": avg_value,
        "daily_average": int(avg_value),
        "min_point": min_point,
        "max_point": max_point,
        "optimal_times": optimal_times,
        "peak_times": peak_times
    }