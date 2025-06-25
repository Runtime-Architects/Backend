import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from run_eirgrid_downloader import main as eirgrid_main
import json

class CO2IntensityAnalyzer:
    """
    Analyzer that categorizes CO2 intensity periods into {low: [], med: [], high: []}
    with optimized time range display.
    """
    
    def __init__(self, json_data: Dict):
        self.data = self._process_json_data(json_data)
        self.thresholds = self._calculate_thresholds()
    
    def _process_json_data(self, json_data: Dict) -> pd.DataFrame:
        time_series = json_data['data']['time_series']
        df = pd.DataFrame(time_series)
        df['timestamp'] = pd.to_datetime(df['time'], format='%d %B %Y %H:%M')
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
        """
        combined = {'low': [], 'med': [], 'high': []}
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
                if start_time == self.data.iloc[i-1]['timestamp']:
                    # Single point
                    time_str = start_time.strftime('%H:%M')
                else:
                    # Range
                    time_str = f"{start_time.strftime('%H:%M')}-{self.data.iloc[i-1]['timestamp'].strftime('%H:%M')}"
                
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


# Example usage
# if __name__ == "__main__":

#     # def call_as_cli():
#     # # Simulate command line arguments
#     #     sys.argv = [
#     #         'run_eirgrid_downloader.py',
#     #         '--areas', 'co2_intensity,wind_generation',
#     #         '--start', '2025-06-22',
#     #         '--end', '2025-06-22',
#     #         '--region', 'all',
#     #         '--forecast',
#     #         '--output-dir', './data'
#     #     ]
    
#     #     # Run the main function
#     #     return eirgrid_main()
    
#     # call_as_cli()


#     with open('data/co2_intensity_all_20250622.json', 'r') as file:
#         scraper_data = json.load(file)


#     analyzer = CO2IntensityAnalyzer(scraper_data)
#     intensity_periods = analyzer.get_combined_periods()
    
#     print("CO2 Intensity Periods:")
#     for level, periods in intensity_periods.items():
#         print(f"{level.upper()}: {', '.join(periods)}")