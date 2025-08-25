"""
co2_plots.py

This module is consists of utility functions used to plot carbon emissions over different periods and regions
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from agents.tools.co2_analysis_util import get_days, str_to_date, find_region_str


def plot_day_intensity(df_, start_date_str, end_date_str, region, workdir):
    """Plots the CO₂ intensity trend over a specified date range, highlighting regions of significant intensity levels.

    Args:
        df_ (pd.DataFrame): DataFrame containing the CO₂ intensity data with columns 'timestamp', 'value', and 'level'.
        start_date_str (str): The start date for the plot in string format (e.g., 'YYYY-MM-DD').
        end_date_str (str): The end date for the plot in string format (e.g., 'YYYY-MM-DD').
        region (str): The region for which the CO₂ intensity trend is being plotted.
        workdir (str): The directory where the plot image will be saved.

    Returns:
        None: The function saves the plot as a PNG file in the specified work directory.
    """

    df = df_.copy()

    # Create a column to identify consecutive level groups
    df["level_group"] = (df["level"] != df["level"].shift(1)).cumsum()

    # Filter groups that last at least 90 minutes (6 intervals)
    min_intervals = 6
    highlight_regions = (
        df.groupby("level_group")
        .filter(lambda x: len(x) >= min_intervals)
        .groupby("level_group")
        .agg({"timestamp": ["first", "last"], "level": "first"})
        .reset_index(drop=True)
    )

    # Define colors for each level
    level_colors = {"low": "green", "medium": "orange", "high": "red"}

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["value"], label="Actual", color="black")

    # Highlight regions
    for _, row in highlight_regions.iterrows():
        plt.axvspan(
            row[("timestamp", "first")],
            row[("timestamp", "last")],
            color=level_colors[row["level"].iloc[0]],
            alpha=0.3,
            label=f"{row['level'].iloc[0]} (≥90min)",
        )

    # Format x-axis
    ax = plt.gca()

    if start_date_str == end_date_str:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45)

        plt.title(
            f"CO₂ Intensity Trend ({find_region_str(region)}) for {start_date_str}",
            fontsize=14,
        )

    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d %H:%M"))
        plt.xticks(rotation=70)

        plt.title(
            f"CO₂ Intensity Trend ({find_region_str(region)}) for {start_date_str} to {end_date_str}",
            fontsize=14,
        )

    # tile and labels

    plt.ylabel("CO₂ Intensity (tCO₂/hr)")
    plt.xlabel("Time")

    # Legend handling
    legend_handles = []
    legend_labels = []

    # Manually create legend entries in desired order
    for level, color in [("low", "green"), ("medium", "orange"), ("high", "red")]:
        # Add a dummy patch (rectangle) for each level
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=level)
        legend_handles.append(patch)
        legend_labels.append(f"{level} (≥90min)")

    # Add the actual data line (black) to the legend
    line = plt.Line2D([], [], color="black", label="CO₂ intensity")
    legend_handles.insert(0, line)  # Insert at the beginning
    legend_labels.insert(0, "CO₂ intensity")

    # Apply the custom legend
    plt.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()

    # Save plot
    plt.savefig(
        f"{workdir}/co2plot_day_all_{start_date_str}_{end_date_str}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_weekly_intensity(df_, start_date_str, end_date_str, region, workdir):
    """Plots the weekly CO₂ intensity trend for a specified region over a given date range.

    Args:
        df_ (pd.DataFrame): DataFrame containing the CO₂ intensity data with columns 'timestamp', 'value', and 'level'.
        start_date_str (str): The start date of the period to plot, formatted as a string (e.g., 'YYYY-MM-DD').
        end_date_str (str): The end date of the period to plot, formatted as a string (e.g., 'YYYY-MM-DD').
        region (str): The name of the region for which the CO₂ intensity trend is being plotted.
        workdir (str): The directory where the plot image will be saved.

    Returns:
        None: The function saves the plot as a PNG file in the specified work directory.

    Raises:
        ValueError: If the input DataFrame does not contain the required columns.
    """

    df = df_.copy()

    # Create a column to identify consecutive level groups
    df["level_group"] = (df["level"] != df["level"].shift(1)).cumsum()

    # Filter groups that last at least 6 hours (6 intervals)
    min_intervals = 6
    highlight_regions = (
        df.groupby("level_group")
        .filter(lambda x: len(x) >= min_intervals)
        .groupby("level_group")
        .agg({"timestamp": ["first", "last"], "level": "first"})
        .reset_index(drop=True)
    )

    # Define colors for each level
    level_colors = {"low": "green", "medium": "orange", "high": "red"}

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["value"], label="Actual", color="black")

    # Highlight regions
    for _, row in highlight_regions.iterrows():
        plt.axvspan(
            row[("timestamp", "first")],
            row[("timestamp", "last")],
            color=level_colors[row["level"].iloc[0]],
            alpha=0.3,
            label=f"{row['level'].iloc[0]} (≥ 6hours)",
        )

    # Format x-axis
    ax = plt.gca()

    ndays = get_days(start_date_str, end_date_str)

    if ndays == 7:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d\n%H:%M"))
        plt.xticks(rotation=45)

        plt.title(
            f"CO₂ Intensity Trend ({find_region_str(region)}) for {start_date_str} to {end_date_str}",
            fontsize=14,
        )

    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d %H:%M"))

        plt.xticks(rotation=70)

        plt.title(
            f"CO₂ Intensity Trend ({find_region_str(region)}) for {start_date_str} to {end_date_str} ({ndays // 7} weeks)",
            fontsize=14,
        )

    # tile and labels

    plt.ylabel("CO₂ Intensity (tCO₂/hr)")
    plt.xlabel("Time")

    # Legend handling
    legend_handles = []
    legend_labels = []

    # Manually create legend entries in desired order
    for level, color in [("low", "green"), ("medium", "orange"), ("high", "red")]:
        # Add a dummy patch (rectangle) for each level
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=level)
        legend_handles.append(patch)
        legend_labels.append(f"{level} (≥ 6 hours)")

    # Add the actual data line (black) to the legend
    line = plt.Line2D([], [], color="black", label="CO₂ intensity")
    legend_handles.insert(0, line)  # Insert at the beginning
    legend_labels.insert(0, "CO₂ intensity")

    # Apply the custom legend
    plt.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()

    # Save plot
    plt.savefig(
        f"{workdir}/co2plot_week_all_{start_date_str}_{end_date_str}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_monthly_intensity(df_, start_date_str, end_date_str, region, work_dir):
    """Plots the monthly CO2 intensity trend for a specified region over a given date range.

    Args:
        df_ (pd.DataFrame): DataFrame containing CO2 intensity data with columns 'timestamp', 'value', and 'level'.
        start_date_str (str): Start date of the period to plot in 'YYYY-MM-DD' format.
        end_date_str (str): End date of the period to plot in 'YYYY-MM-DD' format.
        region (str): The region for which the CO2 intensity is being plotted.
        work_dir (str): Directory where the plot image will be saved.

    Returns:
        None: The function saves the plot as a PNG file in the specified directory.
    """

    # Prepare data (keep timestamp as a column)
    df = df_.copy()

    # Get thresholds from existing data
    low_thresh = df[df["level"] == "low"]["value"].max()
    high_thresh = df[df["level"] == "medium"]["value"].max()

    # Map levels to colors
    color_map = {"low": "green", "medium": "orange", "high": "red"}
    colors = df["level"].map(color_map)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Background bands
    ax.axhspan(
        high_thresh, df["value"].max(), color="red", alpha=0.2, label="High Intensity"
    )
    ax.axhspan(
        low_thresh, high_thresh, color="orange", alpha=0.2, label="Medium Intensity"
    )
    ax.axhspan(
        df["value"].min(), low_thresh, color="green", alpha=0.2, label="Low Intensity"
    )

    # Trend line (use timestamp column directly)
    ax.plot(
        df["timestamp"],
        df["value"],
        color="b",
        alpha=0.5,
        linewidth=2,
        label="CO2 value",
    )

    # Scatter plot with colors based on level
    sc = ax.scatter(df["timestamp"], df["value"], c=colors, edgecolor="white")

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %a"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    start_date = str_to_date(start_date_str)
    ax.set_title(
        f"Monthly CO2 Intensity Trend ({find_region_str(region)}) - {start_date.strftime('%B %Y')}"
    )

    # Y-axis
    ax.set_ylabel("tCO2/hr")
    ax.set_ylim(df["value"].min() - 20, df["value"].max() + 20)

    # Legend
    ax.legend(bbox_to_anchor=(1.2, 1))

    plt.tight_layout()

    # Save plot
    plt.savefig(
        f"{work_dir}/co2plot_all_{region}_{start_date_str}_{end_date_str}.png",
        bbox_inches="tight",
    )
    plt.close()
