import os
from pathlib import Path
from co2_analysis_tool.co2_analysis_util import (get_emission_data, get_view, calculate_thresholds,
                                                 classify_intensity, format_time_range)
from co2_analysis_tool.co2_plot import (plot_day_intensity, plot_weekly_intensity,
                      plot_monthly_intensity)
import json

from scraper_tools.run_eirgrid_downloader import main as eirgrid_main


class CO2IntensityAnalyzer:
    """
    Analyzer that categorizes CO2 intensity periods into {low: [], med: [], high: []}
    with date formatting as dd-mmm-yy for multi-day data.
    """

    def __init__(self, startdate: str, enddate: str, region: str):
        self.data = get_emission_data(startdate, enddate, region)
        self.region = region 
        self.start_date_str= startdate
        self.end_date_str= enddate
        self.view= get_view(self.start_date_str, self.end_date_str)


    def get_analysis_by_view(self):
        """
        Returns CO2 daily intensity periods with time ranges and their average emissions.
        """

        if self.view == 'week' or self.view == 'day':
            return self.day_weekly_analysis()
        elif self.view == 'month':
            return self.monthly_analysis()
        else:
            return "Wrong view called no graph generated !"

    
    def day_weekly_analysis(self):

        if self.view == 'week':
            self.data = self.data.set_index('timestamp').resample('h').mean().reset_index()

        # Calculate thresholds based on processed data
        thresholds = calculate_thresholds(self.data['value'].values)
    
        # Classify intensity 
        self.data['level'] = classify_intensity(self.data['value'], thresholds)

        combined = {'low': [], 'medium': [], 'high': []}
        current_level = None
        period_values = []  # To store emissions for the current period

        for i, row in self.data.iterrows():
            timestamp, level, value = row['timestamp'], row['level'], row['value']
            
            if current_level is None:
                current_level = level
                start_time = timestamp
                period_values.append(value)
                continue
                
            if level != current_level:
                # Calculate average emission for the period
                avg_emission = sum(period_values) / len(period_values) if period_values else 0
                
                # Get the previous timestamp safely
                if i > 0 and i-1 < len(self.data):
                    end_time = self.data.iloc[i-1]['timestamp']
                else:
                    end_time = start_time
                    
                time_str = format_time_range(start_time, end_time)
                combined[current_level].append({
                    'time': time_str,
                    'emission': round(avg_emission,3)
                })
                
                # Reset for new period
                current_level = level
                start_time = timestamp
                period_values = [value]
            else:
                period_values.append(value)
        
        # Handle the last period if there's any data
        if current_level is not None and not self.data.empty:
            avg_emission = sum(period_values) / len(period_values) if period_values else 0
            time_str = format_time_range(start_time, self.data.iloc[-1]['timestamp'])
            combined[current_level].append({
                'time': time_str,
                'emission': avg_emission
            })


        self.generate_plot()

        if self.view == "week":
            return {
            "analysis_type": "week",
            "results": combined
            }
        elif self.view == "day":
            return {
            "analysis_type": "day",
            "results": combined
            }

        return "Error processing data"

    
    def get_monthly_analysis(self):
        """
        Returns CO2 monthly intensity periods with dates and emissions.
        
        """

        combined = {'low': {}, 'medium': {}, 'high': {}}
        
        if self.data.empty:
            raise Exception(f"Failed processing json.")
        
        # reshape by day for monthly
        self.data = self.data.set_index('timestamp').resample('d').mean().reset_index()

        # Calculate thresholds based on processed data
        thresholds = calculate_thresholds(self.data['value'].values)
        
        # Classify intensity 
        self.data['level'] = classify_intensity(self.data['value'], thresholds)
        
        # Add date and emission to the appropriate level category
        for _, row in self.data.iterrows():
            date_str = row['timestamp'].strftime('%Y-%m-%d')
            emission = round(row['value'], 3)
            level = row['level']
            
            combined[level][date_str] = emission

        self.generate_plot()
            
        if not combined:
            return "Error processing data"
            
        return {
            "analysis_type": "month",
            "results": combined
            }


    def generate_plot(self):
        work_dir = Path("plots")
        work_dir.mkdir(exist_ok=True)

        # Plot based on view type
        if self.view == 'day':
            plot_day_intensity(self.data, self.start_date_str, self.end_date_str, self.region, work_dir)
        elif self.view == 'week':
            plot_weekly_intensity(self.data, self.start_date_str, self.end_date_str, self.region, work_dir)
        elif self.view == 'month':
            plot_monthly_intensity(self.data, self.start_date_str, self.end_date_str, self.region, work_dir)



if __name__ == '__main__':

    # def call_as_cli():
    # # Simulate command line arguments
    #     sys.argv = [
    #         'run_eirgrid_downloader.py',
    #         '--areas', 'co2_intensity',
    #         '--start', '2025-07-20',
    #         '--end', '2025-07-20',
    #         '--region', 'all',
    #         '--forecast',
    #         '--output-dir', './data'
    #     ]
    
    #     # Run the main function
    #     return eirgrid_main()
    
    # call_as_cli()

    try:
        # Use relative path to data directory
        data_file = 'data/co2_intensity/co2_intensity_all_2025-07-01_2025-07-14.json'
        if not os.path.exists(data_file):
            # Look for any available co2 intensity file
            import glob
            available_files = glob.glob('data/co2_intensity/co2_intensity_*.json')
            if available_files:
                data_file = available_files[0]
            else:
                raise FileNotFoundError("No CO2 intensity data files found in data/co2_intensity/")
        
        with open(data_file, 'r') as file:
            scraper_data = json.load(file)
    except Exception as e:
        raise Exception(f"Error loading CO2 data: {e}")
    

    analyzer = CO2IntensityAnalyzer("2025-07-01", "2025-07-14", "all")
    intensity_periods = analyzer.get_analysis_by_view()

    print(intensity_periods)