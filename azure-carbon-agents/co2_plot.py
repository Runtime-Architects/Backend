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
                label="CO2 Intensity (15-min intervals)")

        # Color map for categories
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        
        # Create custom legend entries for the highlights
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.2, 
                         label=f'Low Intensity (<{p33:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='orange', alpha=0.2, 
                         label=f'Medium Intensity ({p33:.1f}-{p66:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.2, 
                         label=f'High Intensity (>{p66:.1f} gCO₂/hr)')
        ]

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
            padding = pd.Timedelta(minutes=30)
        else:
            padding = pd.Timedelta(hours=2)
        
        # Apply final padding
        if num_days == 1:
            padding = min(padding, pd.Timedelta(hours=1))
        else:
            padding = min(padding, pd.Timedelta(hours=3))
        
        ax.set_xlim(x_min - padding, x_max + padding)

        # Plot the highlights
        for start_time, highlight_end, category in highlights:
            ax.axvspan(start_time, highlight_end,
                    color=color_map[category], alpha=0.2)

        # X-axis formatting
        if num_days == 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            date_str = unique_days[0].strftime("%Y-%m-%d")
            title = f"15-Minute CO₂ Intensity Trend for {date_str}"
            annotation_points = df_batch[(df_batch["timestamp"].dt.hour % 2 == 0) & 
                                    (df_batch["timestamp"].dt.minute == 0)]
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            date_range_str = f"{unique_days[0].strftime('%Y-%m-%d')} to {unique_days[-1].strftime('%Y-%m-%d')}"
            title = f"15-Minute CO₂ Intensity Trend for {date_range_str}"
            annotation_points = df_batch[(df_batch["timestamp"].dt.hour % 12 == 0) & 
                                    (df_batch["timestamp"].dt.minute == 0)]

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add markers and annotations
        ax.scatter(annotation_points["timestamp"], annotation_points["value"], 
                color="black", s=30, zorder=3)
        
        for x, y in zip(annotation_points["timestamp"], annotation_points["value"]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                    xytext=(-5,10), ha='center', fontsize=8)

        # Updated Y-axis label with units
        ax.set_ylabel("CO₂ Intensity (tCO₂/hr)")
        ax.set_title(title)
        
        # Combine both legends (line and highlights)
        line_legend = ax.legend(handles=[plt.Line2D([], [], color='black', linewidth=2, 
                                        label='CO₂ Intensity (15-min intervals)')], 
                              loc='upper left')
        ax.add_artist(line_legend)
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

        plt.savefig(f'{workdir}/co2plot_day_{start_date}_{end_date}.png', bbox_inches='tight')
        plt.close()


    def plot_week_intensity(df_, workdir: str):
        """Analyzes and visualizes weekly CO₂ intensity with 15-min data aggregated to hourly."""
        
        df = df_.copy()
        
        # Ensure timestamp is timezone-naive if it's not already
        if hasattr(df['timestamp'], 'dt'):
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Extract date from timestamp
        df['date'] = df['timestamp'].dt.date
        
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

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
        
        # Main plot line with updated label
        ax.plot(df["timestamp"], df["value"], color="black", 
                linewidth=2, label="CO₂ Intensity (hourly avg)")
        ax.scatter(df["timestamp"][::6], df["value"][::6], color="black", s=40, zorder=3)

        # Color map for categories
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        
        # Create custom legend entries for the highlights
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.2, 
                        label=f'Low Intensity (<{p33:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='orange', alpha=0.2, 
                        label=f'Medium Intensity ({p33:.1f}-{p66:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.2, 
                        label=f'High Intensity (>{p66:.1f} gCO₂/hr)')
        ]

        # Loop over groups (checking for consecutive hours)
        for _, group_df in df.groupby("group"):
            category = group_df["category"].iloc[0]
            start_time = group_df["timestamp"].iloc[0]
            end_time = group_df["timestamp"].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Highlight regions lasting at least 6 consecutive hours
            if duration_hours >= 6:
                ax.axvspan(start_time - pd.Timedelta(minutes=30),  # Center the highlight
                        end_time + pd.Timedelta(minutes=30),
                        color=color_map[category], alpha=0.2)

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

        # Updated Y-axis label with proper units and CO₂ formatting
        ax.set_ylabel("CO₂ Intensity (gCO₂/hr)")

        # Create accurate title with week count
        if num_weeks == 1:
            title = f"Hourly CO₂ Intensity Trend ({start_date.strftime('%b %d')} to {end_date.strftime('%b %d')})"
        else:
            title = f"Hourly CO₂ Intensity Trend - {num_weeks} Weeks ({start_date.strftime('%b %d')} to {end_date.strftime('%b %d')})"
        
        ax.set_title(title)
        
        # Combine both legends (line and highlights)
        line_legend = ax.legend(handles=[plt.Line2D([], [], color='black', linewidth=2, 
                                        label='CO₂ Intensity (hourly avg)')], 
                            loc='upper left')
        ax.add_artist(line_legend)
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{workdir}/co2plot_week_{start_date}_{end_date}.png', bbox_inches='tight')
        plt.close()


    def plot_month_intensity(df_, workdir: str, max_months=11):
        """
        Plots the trend of CO₂ intensity over multiple months (up to 11) with:
        - Background bands (green/yellow/red) based on percentiles
        - Scatter point colors matching the percentile classification
        - Clear month separation and labeling
        
        Parameters:
        - df_: DataFrame with timestamp and value columns
        - workdir: Directory to save the plot
        - max_months: Maximum number of months to display (1-11)
        """

        # Prepare data
        df = df_.copy()

        # Ensure timestamp is timezone-naive if it's not already
        if hasattr(df['timestamp'], 'dt'):
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df['date'] = df['timestamp'].dt.date
        
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

        # Calculate number of months in data
        num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        
        # Validate maximum months
        if num_months > max_months:
            print(f"Warning: Data spans {num_months} months which exceeds maximum plottable months ({max_months}). No plot generated.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)

        # Calculate percentile thresholds for the entire period
        p33 = df["value"].quantile(0.33)
        p66 = df["value"].quantile(0.66)
        
        # Assign colors based on percentiles
        colors = []
        for val in df["value"]:
            if val <= p33:
                colors.append("green")  # Low intensity
            elif val <= p66:
                colors.append("orange")  # Medium intensity
            else:
                colors.append("red")  # High intensity
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create custom legend entries for the bands
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.15, 
                        label=f'Low Intensity (<{p33:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='orange', alpha=0.15, 
                        label=f'Medium Intensity ({p33:.1f}-{p66:.1f} gCO₂/hr)'),
            plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.15, 
                        label=f'High Intensity (>{p66:.1f} gCO₂/hr)')
        ]

        # Background bands for entire period
        ax.axhspan(p66, df["value"].max(), color="red", alpha=0.15)
        ax.axhspan(p33, p66, color="orange", alpha=0.15)
        ax.axhspan(df["value"].min(), p33, color="green", alpha=0.15)
        
        # Plot trend line
        ax.plot(df.index, df["value"], color="navy", alpha=0.7, linewidth=1.5, 
            label="CO₂ Intensity (daily values)")
        
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
        if num_months <= 3:  # Few months - show days
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %a"))
        else:  # More months - show weeks
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        
        # Rotate and align x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Set title with date range
        if num_months == 1:
            title = f"CO₂ Intensity Trend: {start_date.strftime('%b %Y')}"
        else:
            title = f"CO₂ Intensity Trend: {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}"
        ax.set_title(title, pad=20)
        
        # Y-axis settings with proper units
        ax.set_ylabel("CO₂ Intensity (gCO₂/hr)")
        buffer = (df["value"].max() - df["value"].min()) * 0.1
        ax.set_ylim(max(0, df["value"].min() - buffer), df["value"].max() + buffer)
        
        # Combine both legends (line and bands)
        line_legend = ax.legend(handles=[plt.Line2D([], [], color='navy', linewidth=1.5, 
                                    label='CO₂ Intensity (daily values)')], 
                            loc='upper left')
        ax.add_artist(line_legend)
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{workdir}/co2plot_month_{start_date}_{end_date}.png', bbox_inches='tight')
        plt.close()






