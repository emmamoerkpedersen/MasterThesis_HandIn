import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


raw_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_RAW_level.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
editted_level_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_EDT_level.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')

# Divide Value column by 1000
raw_data['Value'] = raw_data['Value'] / 1000
editted_level_data['Value'] = editted_level_data['Value'] / 1000

# Convert Date column to datetime with specified format (DD-MM-YYYY HH:MM)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y %H:%M')
editted_level_data['Date'] = pd.to_datetime(editted_level_data['Date'], format='%d-%m-%Y %H:%M')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(raw_data['Date'], raw_data['Value'], label='Raw Data', alpha=0.7)
plt.plot(editted_level_data['Date'], editted_level_data['Value'], label='Edited Data', alpha=0.7)

# Customize the plot
plt.title('Comparison of Raw and Edited Level Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()


