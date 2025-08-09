from pathlib import Path
from agents.tools.co2_analysis_util import (
    get_emission_data,
    get_view,
    calculate_thresholds,
    classify_intensity,
    format_time_range,
)
from agents.tools.co2_plot import (
    plot_day_intensity,
    plot_weekly_intensity,
    plot_monthly_intensity,
)

class CO2IntensityAnalyzer:
    """
    Analyzer that categorizes CO2 intensity periods into {low: [], med: [], high: []}
    with date formatting as dd-mmm-yy for multi-day data.
    """

    def __init__(self, startdate: str, enddate: str, region: str):
        self.data = get_emission_data(startdate, enddate, region)
        self.region = region
        self.start_date_str = startdate
        self.end_date_str = enddate
        self.view = get_view(self.start_date_str, self.end_date_str)

    def get_analysis_by_view(self):
        """
        Returns CO2 daily intensity periods with time ranges and their average emissions.
        """

        if self.view == "week" or self.view == "day":
            return self.day_weekly_analysis()
        elif self.view == "month":
            return self.monthly_analysis()
        else:
            return "Wrong view called no graph generated !"

    def day_weekly_analysis(self):

        if self.view == "week":
            self.data = (
                self.data.set_index("timestamp").resample("h").mean().reset_index()
            )

        # Calculate thresholds based on processed data
        thresholds = calculate_thresholds(self.data["value"].values)

        # Classify intensity
        self.data["level"] = classify_intensity(self.data["value"], thresholds)

        combined = {"low": [], "medium": [], "high": []}
        current_level = None
        period_values = []  # To store emissions for the current period

        for i, row in self.data.iterrows():
            timestamp, level, value = row["timestamp"], row["level"], row["value"]

            if current_level is None:
                current_level = level
                start_time = timestamp
                period_values.append(value)
                continue

            if level != current_level:
                # Calculate average emission for the period
                avg_emission = (
                    sum(period_values) / len(period_values) if period_values else 0
                )

                # Get the previous timestamp safely
                if i > 0 and i - 1 < len(self.data):
                    end_time = self.data.iloc[i - 1]["timestamp"]
                else:
                    end_time = start_time

                time_str = format_time_range(start_time, end_time)
                combined[current_level].append(
                    {"time": time_str, "emission": round(avg_emission, 3)}
                )

                # Reset for new period
                current_level = level
                start_time = timestamp
                period_values = [value]
            else:
                period_values.append(value)

        # Handle the last period if there's any data
        if current_level is not None and not self.data.empty:
            avg_emission = (
                sum(period_values) / len(period_values) if period_values else 0
            )
            time_str = format_time_range(start_time, self.data.iloc[-1]["timestamp"])
            combined[current_level].append({"time": time_str, "emission": avg_emission})

        self.generate_plot()

        if self.view == "week":
            return {"analysis_type": "week", "results": combined}
        elif self.view == "day":
            return {"analysis_type": "day", "results": combined}

        return "Error processing data"

    def monthly_analysis(self):
        """
        Returns CO2 monthly intensity periods with dates and emissions.

        """

        combined = {"low": {}, "medium": {}, "high": {}}

        if self.data.empty:
            raise Exception(f"Failed processing json.")

        # reshape by day for monthly
        self.data = self.data.set_index("timestamp").resample("d").mean().reset_index()

        # Calculate thresholds based on processed data
        thresholds = calculate_thresholds(self.data["value"].values)

        # Classify intensity
        self.data["level"] = classify_intensity(self.data["value"], thresholds)

        # Add date and emission to the appropriate level category
        for _, row in self.data.iterrows():
            date_str = row["timestamp"].strftime("%Y-%m-%d")
            emission = round(row["value"], 3)
            level = row["level"]

            combined[level][date_str] = emission

        self.generate_plot()

        if not combined:
            return "Error processing data"

        return {"analysis_type": "month", "results": combined}

    def generate_plot(self):
        work_dir = Path("plots")
        work_dir.mkdir(exist_ok=True)

        # Plot based on view type
        if self.view == "day":
            plot_day_intensity(
                self.data, self.start_date_str, self.end_date_str, self.region, work_dir
            )
        elif self.view == "week":
            plot_weekly_intensity(
                self.data, self.start_date_str, self.end_date_str, self.region, work_dir
            )
        elif self.view == "month":
            plot_monthly_intensity(
                self.data, self.start_date_str, self.end_date_str, self.region, work_dir
            )


if __name__ == "__main__":
    # Initialize here
    pass
