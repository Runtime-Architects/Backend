import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
from typing import Dict, List
from pathlib import Path
import shutil
# from run_eirgrid_downloader import main as eirgrid_main
import json
from co2_plot import CO2plot as plot


class CO2IntensityAnalyzer:
    """
    Analyzer that categorizes CO2 intensity periods into {low: [], med: [], high: []}
    with date formatting as dd-mmm-yy for multi-day data.
    """
    
    def __init__(self, json_data: Dict):
        self.data = self._process_json_data(json_data)
    
    def _process_json_data(self, json_data: Dict) -> pd.DataFrame:
        time_series = json_data['data']['time_series']
        df = pd.DataFrame(time_series)
        
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        
        # Ensure timestamps are timezone-aware (Europe/Dublin)
        if df['timestamp'].dt.tz is None:
            ireland = timezone('Europe/Dublin')
            df['timestamp'] = df['timestamp'].dt.tz_localize(ireland)
        
        # Check if data spans multiple days
        start_date = df['timestamp'].min().date()
        end_date = df['timestamp'].max().date()
        today = datetime.now(tz=ireland).date()
        
        # Only filter future values if data is for current day only
        # if start_date == today and end_date == today:
        #     now = pd.Timestamp.now(tz=ireland)
        #     df = df[df['timestamp'] >= now].copy()
        
        return df.sort_values('timestamp')[['timestamp', 'value']]
    
    
    def _calculate_thresholds(self) -> Dict[str, float]:
        values = self.data['value'].values
        q1, q3 = np.percentile(values, [33, 66])
        return {'low': q1, 'high': q3}
    
    def _classify_intensity(self, value: float, thresholds: Dict[str, float]) -> str:
        if value <= thresholds['low']:
            return 'low'
        if value >= thresholds['high']:
            return 'high'
        return 'med'
        

    def _format_time_range(self, start: datetime, end: datetime) -> str:
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
        
    
    def get_intensity_periods(self, view: str = 'day') -> Dict[str, List[str]]:
        """
        Returns CO2 intensity periods with appropriate formatting and automatic aggregation.
        
        Args:
            view: 'day' (no aggregation), 'week' (hourly), or 'month' (daily)
        
        Returns:
            Dictionary of time periods categorized into {low: [], med: [], high: []}
            with formatting appropriate for the view type.
        """
        combined = {'low': [], 'med': [], 'high': []}
        
        if self.data.empty:
            return combined
        
        # Determine aggregation based on view type
        if view == 'week':
            processed_data = self.data.set_index('timestamp').resample('h').mean().reset_index()
        elif view == 'month':
            processed_data = self.data.set_index('timestamp').resample('D').mean().reset_index()
        else:  # day view (default)
            processed_data = self.data.copy()
        
        # Check if processed data is empty after resampling
        if processed_data.empty:
            return combined
        
        # Calculate thresholds based on processed data
        thresholds = self._calculate_thresholds()
        
        current_level = None
        start_time = None
        
        for i, row in processed_data.iterrows():
            timestamp, value = row['timestamp'], row['value']
            level = self._classify_intensity(value, thresholds)
            
            if current_level is None:
                current_level = level
                start_time = timestamp
                continue
                
            if level != current_level:
                # Get the previous timestamp safely
                if i > 0 and i-1 < len(processed_data):
                    end_time = processed_data.iloc[i-1]['timestamp']
                else:
                    end_time = start_time
                    
                time_str = self._format_time_range(start_time, end_time)
                combined[current_level].append(time_str)
                
                current_level = level
                start_time = timestamp
        
        # Handle the last period if there's any data
        if current_level is not None and not processed_data.empty:
            time_str = self._format_time_range(start_time, processed_data.iloc[-1]['timestamp'])
            combined[current_level].append(time_str)

        work_dir = Path("plots")
        work_dir.mkdir(exist_ok=True)

        # Plot based on view type
        if view == 'day':
            plot.plot_day_intensity(processed_data, work_dir)
        elif view == 'week':
            plot.plot_week_intensity(processed_data, work_dir)
        elif view == 'month':
            plot.plot_month_intensity(processed_data, work_dir)
        
        return combined

    
    

if __name__ == '__main__':

    # def call_as_cli():
    # # Simulate command line arguments
    #     sys.argv = [
    #         'run_eirgrid_downloader.py',
    #         '--areas', 'co2_intensity',
    #         '--start', '2025-06-25',
    #         '--end', '2025-06-25',
    #         '--region', 'all',
    #         '--forecast',
    #         '--output-dir', './data'
    #     ]
    
    #     # Run the main function
    #     return eirgrid_main()
    
    # call_as_cli()

    try:
        with open('C:\\Users\\nithy\\NK\\UCD\\Sem3\\SustainbleCityAI\\Backend\\azure-carbon-agents\\data\\co2_intensity\\co2_intensity_roi_2025-07-08_2025-07-08.json', 'r') as file:
            scraper_data = json.load(file)
    except:
        raise Exception
    

    analyzer = CO2IntensityAnalyzer(scraper_data)
    intensity_periods = analyzer.get_intensity_periods('day')

    print(intensity_periods)