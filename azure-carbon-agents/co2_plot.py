import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math


class CO2plot:
    """
    plot the CO2 analysis
    """

    def __init__(self):
        pass

    def plot_day_intensity(df_, workdir: str):
        """
        Analyzes and visualizes CO2 intensity using 15-minute data for one or multiple days (up to 6 days max).
        
        Parameters:
        - df_: DataFrame with 'timestamp' and 'value' columns.
        - workdir: Directory to save the plot.
        """
        df = df_.copy()

        max_days_to_plot = 6
        
        # Ensure timestamp is timezone-naive and in correct format
        if hasattr(df['timestamp'], 'dt'):
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Extract date from timestamp
        df['date'] = df['timestamp'].dt.date

        # Get unique days in the data
        unique_days = df['date'].unique()
        num_days = len(unique_days)
        
        # Check if data exceeds maximum allowed days
        if num_days > max_days_to_plot:
            print(f"Warning: Data spans {num_days} days which exceeds maximum plottable days ({max_days_to_plot}). No plot generated.")
            return
        
        # Proceed with plotting
        df_batch = df.copy()
        
        # Compute thresholds (normalized across all days)
        p33 = np.percentile(df_batch["value"], 33)
        p66 = np.percentile(df_batch["value"], 66)

        # Categorize function
        def categorize(val):
            if val <= p33:
                return "Low"
            elif val <= p66:
                return "Medium"
            else:
                return "High"

        df_batch["category"] = df_batch["value"].apply(categorize)

        # Identify consecutive groups with same category
        df_batch["group"] = (df_batch["category"] != df_batch["category"].shift()).cumsum()

        # Plotting
        fig, ax = plt.subplots(figsize=(18, 6))
        
        # Plot the line for the full data
        ax.plot(df_batch["timestamp"], df_batch["value"], color="black", linewidth=2, 
                label="CO2 Intensity (15-min)")

        # Color map for categories
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        added_to_legend = set()

        # First calculate all highlights to determine the maximum needed padding
        highlights = []
        for _, group_df in df_batch.groupby("group"):
            category = group_df["category"].iloc[0]
            start_time = group_df["timestamp"].iloc[0]
            end_time = group_df["timestamp"].iloc[-1]
            
            if len(group_df) >= 6:  # At least 6 consecutive points (1.5 hours)
                highlight_end = end_time + pd.Timedelta(minutes=15)  # Include full last interval
                highlights.append((start_time, highlight_end, category))

        # Set x-axis limits with padding
        x_min = df_batch["timestamp"].min()
        x_max = df_batch["timestamp"].max()
        
        # Calculate padding based on the data range and highlights
        if num_days == 1:
            # For single day, use 1 hour padding but ensure it doesn't go beyond highlights
            padding = pd.Timedelta(minutes=30)  # Start with 30 minutes
            # Check if any highlight extends beyond our current padding
            for start, end, _ in highlights:
                if end > x_max + padding:
                    padding = end - x_max + pd.Timedelta(minutes=15)
        else:
            # For multiple days, use 3 hours padding but check highlights
            padding = pd.Timedelta(hours=2)  # Start with 2 hours
            for start, end, _ in highlights:
                if end > x_max + padding:
                    padding = end - x_max + pd.Timedelta(hours=1)
        
        # Apply final padding (minimum of calculated padding and 1 hour for single day/3 hours for multi-day)
        if num_days == 1:
            padding = min(padding, pd.Timedelta(hours=1))
        else:
            padding = min(padding, pd.Timedelta(hours=3))
        
        ax.set_xlim(x_min - padding, x_max + padding)

        # Now plot the highlights with the correct x-limits
        for start_time, highlight_end, category in highlights:
            label = f"{category} Region" if category not in added_to_legend else None
            ax.axvspan(start_time, highlight_end,
                    color=color_map[category], alpha=0.2, label=label)
            if label:
                added_to_legend.add(category)

        # X-axis formatting
        if num_days == 1:
            # Single day format
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Every 2 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            date_str = unique_days[0].strftime("%Y-%m-%d")
            title = f"15-Minute CO2 Intensity Trend for {date_str}"
            # Every 2 hours markers and annotations
            annotation_points = df_batch[(df_batch["timestamp"].dt.hour % 2 == 0) & 
                                    (df_batch["timestamp"].dt.minute == 0)]
        else:
            # Multi-day format - show ticks at midnight and noon
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            date_range_str = f"{unique_days[0].strftime('%Y-%m-%d')} to {unique_days[-1].strftime('%Y-%m-%d')}"
            title = f"15-Minute CO2 Intensity Trend for {date_range_str}"
            # Every 12 hours markers and annotations
            annotation_points = df_batch[(df_batch["timestamp"].dt.hour % 12 == 0) & 
                                    (df_batch["timestamp"].dt.minute == 0)]

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add markers and annotations only at selected points
        ax.scatter(annotation_points["timestamp"], annotation_points["value"], 
                color="black", s=30, zorder=3)
        
        for x, y in zip(annotation_points["timestamp"], annotation_points["value"]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                    xytext=(-5,10), ha='center', fontsize=8)

        ax.set_ylabel("tCO2/hr")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

        plt.savefig(f'{workdir}/co2plot_day_{start_date}_{end_date}.png', bbox_inches='tight')
        plt.close()  # Close the figure to free memory


    def plot_week_intensity(df_, workdir: str):
        """Analyzes and visualizes weekly CO2 intensity with 15-min data aggregated to hourly."""
        
        df = df_.copy()
        
        # Ensure timestamp is timezone-naive if it's not already
        if hasattr(df['timestamp'], 'dt'):
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Extract date from timestamp
        df['date'] = df['timestamp'].dt.date
        
        start_date= df['date'].iloc[0]
        end_date= df['date'].iloc[-1]

        # Get number of full weeks between the two dates
        num_days = (end_date - start_date).days
        num_weeks = math.ceil(num_days / 7)

        # Compute thresholds on data
        p33 = np.percentile(df["value"], 33)
        p66 = np.percentile(df["value"], 66)

        # Categorize function
        def categorize(val):
            if val <= p33:
                return "Low"
            elif val <= p66:
                return "Medium"
            else:
                return "High"

        df["category"] = df["value"].apply(categorize)
        df["group"] = (df["category"] != df["category"].shift()).cumsum()

        # Plotting
        fig, ax = plt.subplots(figsize=(18, 8))
        
        ax.plot(df["timestamp"], df["value"], color="black", 
                linewidth=2, label="CO2 Intensity (hourly avg)")
        ax.scatter(df["timestamp"][::6], df["value"][::6], color="black", s=40, zorder=3)

        # Color map for categories
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        added_to_legend = set()

        # Loop over groups (checking for consecutive hours)
        for _, group_df in df.groupby("group"):
            category = group_df["category"].iloc[0]
            start_time = group_df["timestamp"].iloc[0]
            end_time = group_df["timestamp"].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Highlight regions lasting at least 6 consecutive hours
            if duration_hours >= 6:
                label = f"{category} Region" if category not in added_to_legend else None
                ax.axvspan(start_time - pd.Timedelta(minutes=30),  # Center the highlight
                        end_time + pd.Timedelta(minutes=30),
                        color=color_map[category], alpha=0.2, label=label)
                if label:
                    added_to_legend.add(category)

        # Format x-axis differently based on number of weeks
        if num_weeks == 1:
            # Single week: show hours at 00:00 and 12:00
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))  
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d\n%H:%M"))

            # Add value annotations for every 6 hours
            for x, y in zip(df["timestamp"][::6], df["value"][::6]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                        xytext=(-5,10), ha='center', fontsize=8)
        else:
            # Multiple weeks: only show 00:00 for each day
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))  
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d\n%H:%M"))

        # Rotate and align x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_ylabel("tCO2/hr")

        
        # Create accurate title with week count
        if num_weeks == 1:
            title = f"Hourly CO2 Intensity Trend ({start_date.strftime('%b %d')} to {end_date.strftime('%b %d')})"
        else:
            title = f"Hourly CO2 Intensity Trend - {num_weeks} Weeks ({start_date.strftime('%b %d')} to {end_date.strftime('%b %d')})"
        
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        # plt.show()
                
        plt.savefig(f'{workdir}/co2plot_week_{start_date}_{end_date}.png')


    def plot_month_intensity(df_, workdir:str, max_months=11):
        """
        Plots the trend of CO2 intensity over multiple months (up to 11) with:
        - Background bands (green/yellow/red) based on percentiles
        - Scatter point colors matching the percentile classification
        - Clear month separation and labeling
        
        Parameters:
        - df_: DataFrame with timestamp and value columns
        - max_months: Maximum number of months to display (1-11)
        """

        # Prepare data
        df = df_.copy()

        # Ensure timestamp is timezone-naive if it's not already
        if hasattr(df['timestamp'], 'dt'):
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df['date'] = df['timestamp'].dt.date
        
        start_date= df['date'].iloc[0]
        end_date= df['date'].iloc[-1]

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)

        
        # Limit to max_months if needed
        if len(df.resample('ME').count()) > max_months:
            start_date = df.index[-1] - pd.DateOffset(months=max_months-1)
            df = df[df.index >= start_date]
        
        # Calculate percentile thresholds for the entire period
        low_thresh = df["value"].quantile(0.33)
        high_thresh = df["value"].quantile(0.66)
        
        # Assign colors based on percentiles
        colors = []
        for val in df["value"]:
            if val <= low_thresh:
                colors.append("green")  # Low intensity
            elif val <= high_thresh:
                colors.append("orange")  # Medium intensity
            else:
                colors.append("red")  # High intensity
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Background bands for entire period
        ax.axhspan(high_thresh, df["value"].max(), color="red", alpha=0.15, label="High Intensity")
        ax.axhspan(low_thresh, high_thresh, color="orange", alpha=0.15, label="Medium Intensity")
        ax.axhspan(df["value"].min(), low_thresh, color="green", alpha=0.15, label="Low Intensity")
        
        # Plot trend line
        ax.plot(df.index, df["value"], color="navy", alpha=0.7, linewidth=1.5, label="CO2 value")
        
        # Scatter plot with colors
        sc = ax.scatter(
            df.index, 
            df["value"], 
            c=colors,
            s=40,
            edgecolor="white",
            linewidth=0.5,
            zorder=3
        )
        
        # Add vertical lines and labels for month separation
        months = pd.date_range(start=df.index[0].replace(day=1), 
                            end=df.index[-1], 
                            freq='MS')
        
        # for month_start in months:
        #     if month_start > df.index[0]:
        #         ax.axvline(month_start, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
        #     month_name = month_start.strftime('%b')
        #     ax.text(month_start, ax.get_ylim()[1]*1.02, month_name, 
        #             ha='left', va='bottom', fontsize=10)
        
        # Format x-axis based on timespan
        if len(months) <= 3:  # Few months - show days
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %a"))
        else:  # More months - show weeks
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        
        # Rotate and align x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Set title with date range
        ax.set_title(f"CO2 Intensity Trend: {start_date} to {end_date}", pad=20)
        
        # Y-axis settings
        ax.set_ylabel("tCO2/hr")
        buffer = (df["value"].max() - df["value"].min()) * 0.1
        ax.set_ylim(max(0, df["value"].min() - buffer), df["value"].max() + buffer)
        
        # Legend and grid
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{workdir}/co2plot_month_{start_date}_{end_date}.png')






