import json
from pathlib import Path
from datetime import datetime
from co2_analysis_tool.co2_analysis_util import (process_json_data, calculate_thresholds, 
                               classify_intensity, format_time_range, get_days)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



def get_daily_analysis(file_path: str):
    """
    Returns CO2 daily intensity periods with appropriate formatting and automatic aggregation.
    
    
    Returns:
        Dictionary of time periods categorized into {low: [], med: [], high: []}
        with formatting appropriate for the view type.
    """

    with open(file_path, 'r') as file:
        scraper_data = json.load(file)
    

    data, start_date_str, end_date_str, region= process_json_data(scraper_data)


    combined = {'low': [], 'medium': [], 'high': []}
    
    if data.empty:
        raise Exception(f"Failed proccessing json.")
    
    
    # Calculate thresholds based on processed data
    thresholds = calculate_thresholds(data['value'].values)
    
    # Classify intensity 
    data['level'] = classify_intensity(data['value'], thresholds)
    
    current_level = None

    for i, row in data.iterrows():
        timestamp, level = row['timestamp'], row['level']
        
        if current_level is None:
            current_level = level
            start_time = timestamp
            continue
            
        if level != current_level:
            # Get the previous timestamp safely
            if i > 0 and i-1 < len(data):
                end_time = data.iloc[i-1]['timestamp']
            else:
                end_time = start_time
                
            time_str = format_time_range(start_time, end_time)
            combined[current_level].append(time_str)
            
            current_level = level
            start_time = timestamp
    
    # Handle the last period if there's any data
    if current_level is not None and not data.empty:
        time_str = format_time_range(start_time, data.iloc[-1]['timestamp'])
        combined[current_level].append(time_str)

    work_dir = Path("plots")
    work_dir.mkdir(exist_ok=True)

    if(get_days(start_date_str, end_date_str) > 6):
        print("Wrong view called no graph generated !")
    else:
        plot_day_intensity(data, start_date_str, end_date_str, region, work_dir)
        
    return {
        "analysis_type": "daily",
        "date_range": f"{start_date_str} to {end_date_str}", 
        "region": region,
        "results": combined
    }




def plot_day_intensity(df_, start_date_str, end_date_str, region, workdir):
    """Visualizes CO2 intensity using 15-minute data for 1-6 days."""

    df= df_.copy()

    # Create a column to identify consecutive level groups
    df['level_group'] = (df['level'] != df['level'].shift(1)).cumsum()

    # Filter groups that last at least 90 minutes (6 intervals)
    min_intervals = 6
    highlight_regions = df.groupby('level_group').filter(
        lambda x: len(x) >= min_intervals
    ).groupby('level_group').agg(
        {'timestamp': ['first', 'last'], 'level': 'first'}
    ).reset_index(drop=True)

    # Define colors for each level
    level_colors = {
        'low': 'green',
        'medium': 'orange',
        'high': 'red'
    }

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'], df['value'], label='Actual', color='black')

    # Highlight regions
    for _, row in highlight_regions.iterrows():
        plt.axvspan(
            row[('timestamp', 'first')],
            row[('timestamp', 'last')],
            color=level_colors[row['level'].iloc[0]],
            alpha=0.3,
            label=f"{row['level'].iloc[0]} (≥90min)"
        )

    # Format x-axis
    ax = plt.gca()

    if(start_date_str == end_date_str):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)

        plt.title(f'CO₂ Intensity Trend ({region}) for {start_date_str}', fontsize=14)

    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
        plt.xticks(rotation=70)

        plt.title(f'CO₂ Intensity Trend ({region}) for {start_date_str} to {end_date_str}', fontsize=14)


    #tile and labels
    
    plt.ylabel("CO₂ Intensity (tCO₂/hr)")
    plt.xlabel("Time")

    # Legend handling
    legend_handles = []
    legend_labels = []

    # Manually create legend entries in desired order
    for level, color in [('low', 'green'), ('medium', 'orange'), ('high', 'red')]:
        # Add a dummy patch (rectangle) for each level
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=level)
        legend_handles.append(patch)
        legend_labels.append(f"{level} (≥90min)")

    # Add the actual data line (black) to the legend
    line = plt.Line2D([], [], color='black', label='CO₂ intensity')
    legend_handles.insert(0, line)  # Insert at the beginning
    legend_labels.insert(0, 'CO₂ intensity')


    # Apply the custom legend
    plt.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{workdir}/co2plot_day_{start_date_str}_{end_date_str}.png', bbox_inches='tight')
    plt.close()






# if __name__ == '__main__':

# # def call_as_cli():
# # # Simulate command line arguments
# #     sys.argv = [
# #         'run_eirgrid_downloader.py',
# #         '--areas', 'co2_intensity',
# #         '--start', '2025-06-25',
# #         '--end', '2025-06-25',
# #         '--region', 'all',
# #         '--forecast',
# #         '--output-dir', './data'
# #     ]

# #     # Run the main function
# #     return eirgrid_main()

# # call_as_cli()

    
#     fp= "C:\\Users\\nithy\\NK\\UCD\\Sem3\\SustainbleCityAI\\Backend\\azure-carbon-agents-v2\\data\\co2_intensity\\co2_intensity_roi_2025-07-11_2025-07-11.json"


#     analyzer = CO2IntensityAnalyzer()
#     intensity_periods= analyzer.get_daily_analysis(fp)

#     print(intensity_periods)