import scrapy
from datetime import datetime

class CO2IntensityItem(scrapy.Item):
    # Metadata
    timestamp = scrapy.Field()
    region = scrapy.Field()
    
    # Current values
    latest_intensity = scrapy.Field()  # gCO2/kWh
    todays_low_intensity = scrapy.Field()  # gCO2/kWh
    latest_emissions = scrapy.Field()  # tCO2/hr
    
    # Time series data
    time_series_data = scrapy.Field()  # List of {time, intensity, forecast}
    
    # Date range
    date_from = scrapy.Field()
    date_to = scrapy.Field()
    
    # Raw data for debugging
    raw_data = scrapy.Field()

class CO2TimeSeriesPoint(scrapy.Item):
    time = scrapy.Field()
    intensity = scrapy.Field()  # Actual intensity
    intensity_forecast = scrapy.Field()  # Forecast intensity
    emissions = scrapy.Field()  # CO2 emissions