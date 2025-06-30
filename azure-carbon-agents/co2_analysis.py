import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List
from run_eirgrid_downloader import main as eirgrid_main
import json

class CO2IntensityAnalyzer:
    """
    Analyzer that categorizes CO2 intensity periods into {low: [], med: [], high: []}
    with optimized time range display, showing only future values from current time.
    """
    
    def __init__(self, json_data: Dict):
        self.data = self._process_json_data(json_data)
        self.thresholds = self._calculate_thresholds()
    
    def _process_json_data(self, json_data: Dict) -> pd.DataFrame:
        time_series = json_data['data']['time_series']
        df = pd.DataFrame(time_series)
        
        # Convert to datetime - adjust format as needed based on your actual data
        df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        # Ensure timestamps are timezone-aware (UTC)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Filter to only include future values from current time
        now = pd.Timestamp.now(tz='UTC')  # Use pandas Timestamp for consistent comparison
        df = df[df['timestamp'] >= now].copy()
        
        return df.sort_values('timestamp')[['timestamp', 'value']]
    
    def _calculate_thresholds(self) -> Dict[str, float]:
        values = self.data['value'].values
        q1, q3 = np.percentile(values, [33, 66])
        return {'low': q1, 'high': q3}
    
    def _classify_intensity(self, value: float) -> str:
        if value <= self.thresholds['low']:
            return 'low'
        if value >= self.thresholds['high']:
            return 'high'
        return 'med'
    
    def get_combined_periods(self) -> Dict[str, List[str]]:
        """
        Returns combined time periods with optimized display:
        - Shows range when continuous (11:00-12:30)
        - Shows single time when isolated (11:30)
        - Only includes future time periods from current time
        """
        combined = {'low': [], 'med': [], 'high': []}
        
        if self.data.empty:
            return combined
            
        current_level = None
        start_time = None
        
        for i, row in self.data.iterrows():
            timestamp, value = row['timestamp'], row['value']
            level = self._classify_intensity(value)
            
            if current_level is None:
                current_level = level
                start_time = timestamp
                continue
                
            if level != current_level:
                # Format the time period
                prev_row = self.data.iloc[self.data.index.get_loc(i)-1] if self.data.index.get_loc(i) > 0 else None
                if prev_row is not None and start_time == prev_row['timestamp']:
                    # Single point
                    time_str = start_time.strftime('%H:%M')
                else:
                    # Range
                    end_time = prev_row['timestamp'] if prev_row is not None else start_time
                    time_str = f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"
                
                combined[current_level].append(time_str)
                current_level = level
                start_time = timestamp
        
        # Handle the last period
        if current_level is not None:
            if start_time == self.data.iloc[-1]['timestamp']:
                time_str = start_time.strftime('%H:%M')
            else:
                time_str = f"{start_time.strftime('%H:%M')}-{self.data.iloc[-1]['timestamp'].strftime('%H:%M')}"
            combined[current_level].append(time_str)
    
        return combined
    

# if __name__ == '__main__':

#     def call_as_cli():
#     # Simulate command line arguments
#         sys.argv = [
#             'run_eirgrid_downloader.py',
#             '--areas', 'co2_intensity',
#             '--start', '2025-06-25',
#             '--end', '2025-06-25',
#             '--region', 'all',
#             '--forecast',
#             '--output-dir', './data'
#         ]
    
#         # Run the main function
#         return eirgrid_main()
    
#     call_as_cli()

#     try:
#         with open(f'data/co2_intensity/co2_intensity_all_2025-06-25_2025-06-25.json', 'r') as file:
#             scraper_data = json.load(file)
#     except:
#         raise Exception
    

#     analyzer = CO2IntensityAnalyzer(scraper_data)
#     intensity_periods = analyzer.get_combined_periods()

#     print(intensity_periods)