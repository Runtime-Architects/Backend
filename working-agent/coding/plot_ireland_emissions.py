# filename: plot_ireland_emissions.py
import subprocess
import sys

# Install matplotlib if not present
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import pandas as pd

# Download and read the data
url = "https://www.epa.ie/media/epa-2020/monitoring-amp-assessment/climate-change/greenhouse-gases/Change-2022-2023-table_MAY2025.xlsx"
df = pd.read_excel(url, engine='openpyxl')

# Bar plot for sectors (2023)
plt.figure(figsize=(10,6))
plt.bar(df["Mt CO2 eq"], df[2023], color='forestgreen')
plt.xlabel("Sector")
plt.ylabel("Emissions (Mt CO2 eq, 2023)")
plt.title("Ireland CO2 Equivalent Emissions by Sector, 2023")
plt.xticks(rotation=30, ha='right')

# Show total emissions as a number on the plot
total_2023 = df[2023].sum()
plt.text(0.95, 0.95, f"Total 2023: {total_2023:.2f} Mt CO2 eq",
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("emissions.png")
print("Plot created and saved as emissions.png.")
