from agents.tools.co2_analysis import CO2IntensityAnalyzer

async def get_emission_analysis(startdate: str, enddate: str, region: str) -> float:
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
    
    
