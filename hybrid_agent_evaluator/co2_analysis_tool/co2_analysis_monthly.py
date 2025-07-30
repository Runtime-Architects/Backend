import json
from pathlib import Path
from datetime import datetime
from co2_analysis_tool.co2_analysis_util import (process_json_data, calculate_thresholds, 
                               classify_intensity, format_time_range, get_days, str_to_date)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_monthly_analysis(file_path: str):
    """
    Returns CO2 monthly intensity periods with appropriate formatting and automatic aggregation.
    
    
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
    
    # reshape by day for monthly
    data = data.set_index('timestamp').resample('d').mean().reset_index()


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

    if(get_days(start_date_str, end_date_str) < 22):
        print("Wrong view called no graph generated !")
    else:
        plot_monthly_intensity(data, start_date_str, end_date_str, region, work_dir)
        
    return {
        "analysis_type": "monthly",
        "date_range": f"{start_date_str} to {end_date_str}", 
        "region": region,
        "results": combined
    }



def plot_monthly_intensity(df_, start_date_str, end_date_str, region, work_dir):
    """
    Plots the trend of CO2 intensity over a month with:
    - Background bands (green/yellow/red) based on existing classification
    - Scatter point colors matching the classification
    """

    # Prepare data (keep timestamp as a column)
    df = df_.copy()

    # Get thresholds from existing data 
    low_thresh = df[df['level'] == 'low']['value'].max()
    high_thresh = df[df['level'] == 'medium']['value'].max()

    # Map levels to colors
    color_map = {'low': 'green', 'medium': 'orange', 'high': 'red'}
    colors = df['level'].map(color_map)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Background bands
    ax.axhspan(high_thresh, df["value"].max(), color="red", alpha=0.2, label="High Intensity")
    ax.axhspan(low_thresh, high_thresh, color="orange", alpha=0.2, label="Medium Intensity")
    ax.axhspan(df["value"].min(), low_thresh, color="green", alpha=0.2, label="Low Intensity")

    # Trend line (use timestamp column directly)
    ax.plot(df["timestamp"], df["value"], color="b", alpha=0.5, linewidth=2, label="CO2 value")

    # Scatter plot with colors based on level
    sc = ax.scatter(
        df["timestamp"], 
        df["value"], 
        c=colors,
        edgecolor="white"
    )

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %a"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    start_date= str_to_date(start_date_str)
    ax.set_title(f"Monthly CO2 Intensity Trend - {start_date.strftime('%B %Y')}")

    # Y-axis
    ax.set_ylabel("tCO2/hr")
    ax.set_ylim(df["value"].min() - 20, df["value"].max() + 20)

    # Legend
    ax.legend(bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    
    # Save plot  
    plt.savefig(f'{work_dir}/co2plot_month_{start_date_str}_{end_date_str}.png', bbox_inches='tight')
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

    
#     fp= "C:\\Users\\nithy\\NK\\UCD\\Sem3\\SustainbleCityAI\\Backend\\azure-carbon-agents\\data\\co2_intensity\\co2_intensity_all_2025-06-01_2025-06-30.json"


#     analyzer = CO2IntensityAnalyzer()
#     intensity_periods= analyzer.get_monthly_analysis(fp)

#     print(intensity_periods)