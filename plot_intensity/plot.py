import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime

# Your co2_plot_trend function (unchanged)
def co2_plot_trend(df_):
    """
    Plots the trend of CO2 intensity over time with full background bands indicating Low, Medium, and High intensity ranges.
    Categorization is based on the 33rd and 66th percentiles of CO2 intensity values.

    Parameters:
    - df_ : pandas.DataFrame
        The input DataFrame must contain columns for timestamps ('EffectiveTime') and CO2 values ('Value').

    Returns:
    - matplotlib.pyplot: Displays the plot with background bands.
    """
    # Set plot style
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # Prepare data
    df = df_.reset_index(inplace=False)
    df["EffectiveTime"] = pd.to_datetime(df["EffectiveTime"])
    df.sort_values("EffectiveTime", inplace=True)
    df.set_index("EffectiveTime", inplace=True)

    # Calculate thresholds
    low_thresh = df["Value"].quantile(0.33)
    high_thresh = df["Value"].quantile(0.66)

    # Format today's date
    today_date = datetime.datetime.now().strftime("%A %d/%m/%Y")

    # Color map for scatter points
    norm = plt.Normalize(df["Value"].min(), df["Value"].max())
    cmap = plt.get_cmap("RdYlGn_r")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Full background banding
    ax.axhspan(df["Value"].min(), low_thresh, color="green", alpha=0.2, label="Low Intensity")
    ax.axhspan(low_thresh, high_thresh, color="orange", alpha=0.2, label="Medium Intensity")
    ax.axhspan(high_thresh, df["Value"].max(), color="red", alpha=0.2, label="High Intensity")

    # Trend line
    ax.plot(df.index, df["Value"], color="b", alpha=0.5, linewidth=2, label="CO2 Value")

    # Scatter plot for intensity
    sc = ax.scatter(df.index, df["Value"], c=df["Value"], cmap=cmap, norm=norm, edgecolor="none")

    # X-axis formatting
    total_duration_hours = (df.index.max() - df.index.min()).total_seconds() / 3600
    interval = max(1, round(total_duration_hours / 24))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", label="Value intensity")
    cbar.set_ticks([df["Value"].min(), low_thresh, high_thresh, df["Value"].max()])
    cbar.set_ticklabels(
        [f"{df['Value'].min():.0f}", f"33rd % ({low_thresh:.0f})", f"66th % ({high_thresh:.0f})", f"{df['Value'].max():.0f}"]
    )

    # Y-axis & Title
    ax.set_ylabel("tCO2/hr")
    ax.set_ylim(df["Value"].min() - 20, df["Value"].max() + 20)
    ax.set_title(f"CO2 Intensity Forecast with Background Banding - {today_date}")

    # Legend
    ax.legend(loc="upper left")

    plt.tight_layout()
    return plt

# ✅ Generate random test data
np.random.seed(42)
n = 30  # Number of data points
timestamps = pd.date_range(start=pd.Timestamp.now().floor('H'), periods=n, freq='H')
categories = ["Low", "Medium", "High"]

df_test = pd.DataFrame({
    "EffectiveTime": timestamps,
    "Value": np.random.randint(100, 601, size=n),
    "category": np.random.choice(categories, size=n),
    "normalized": np.random.uniform(0, 4, size=n)
})

# ✅ Call the function to plot
plot = co2_plot_trend(df_test)

# ✅ Show plot if running in plain Python
plt.show()
