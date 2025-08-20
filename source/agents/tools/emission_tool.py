"""
emission_tool.py

This module consists of the async function used to get emission analysis
It calls utility functions defined in this package
"""

from agents.tools.co2_analysis import CO2IntensityAnalyzer


async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
    """Retrieve the CO2 emission analysis for a specified period and region.

    This asynchronous function analyzes CO2 intensity data between the given start and end dates for a specified region. It utilizes the `CO2IntensityAnalyzer` class to perform the analysis and returns the calculated intensity periods.

    Args:
        startdate (str): The start date for the analysis in 'YYYY-MM-DD' format.
        enddate (str): The end date for the analysis in 'YYYY-MM-DD' format.
        region (str): The region for which the CO2 intensity analysis is to be performed.

    Returns:
        float: The calculated CO2 intensity periods.

    Raises:
        Exception: If an error occurs during the analysis, an exception is raised with details about the error and the input parameters.
    """
    # Check if the file exists

    try:
        # Your existing tool logic
        analyzer = CO2IntensityAnalyzer(startdate, enddate, region)
        intensity_periods = analyzer.get_analysis_by_view()

        return intensity_periods
    except Exception as e:
        # Return detailed error info
        raise {
            "error": str(e),
            "input_params": {
                "startdate": startdate,
                "enddate": enddate,
                "region": region,
            },
        }
