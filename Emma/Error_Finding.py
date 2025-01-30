import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


raw_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006846/VST_RAW.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
editted_level_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006846/VST_EDT.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
vinge_data = pd.read_excel('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Emma/Sample data/21006846/VINGE.xlsm', decimal = ',', header = 0)
precipitation_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Emma/Sample data/RainData_05225.csv', parse_dates=['datetime'])


# Convert Date column to datetime with specified format (DD-MM-YYYY HH:MM)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y %H:%M')
editted_level_data['Date'] = pd.to_datetime(editted_level_data['Date'], format='%d-%m-%Y %H:%M')
vinge_data['Date'] = pd.to_datetime(vinge_data['Date'], format='%d.%m.%Y %H:%M')

# Crop all dataframes to start from 2010-01-01
start_date = '2000-01-01'
raw_data = raw_data[raw_data['Date'] >= start_date]
editted_level_data = editted_level_data[editted_level_data['Date'] >= start_date]
vinge_data = vinge_data[vinge_data['Date'] >= start_date]
precipitation_data = precipitation_data[precipitation_data['datetime'] >= start_date]

# Multiply W.L [cm] by 100 to convert to mm
vinge_data['W.L [cm]'] = vinge_data['W.L [cm]']*10

# First merge the datasets on the nearest timestamp, but in reverse order
merged_data = pd.merge_asof(
    vinge_data[['Date', 'W.L [cm]']].sort_values('Date'),  # Make Vinge data the left dataframe
    raw_data.sort_values('Date'),
    on='Date',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=30)  # Allow 30 minutes tolerance for matching
)

# Calculate the absolute difference between values
merged_data['difference'] = abs(merged_data['Value'].astype(float) - merged_data['W.L [cm]'])

# Find significant discrepancies (you can adjust the threshold)
lower_threshold = 5  # difference of 5mm
upper_threshold = 150
discrepancies = merged_data[(merged_data['difference'] > lower_threshold) & (merged_data['difference'] < upper_threshold)]

# Sort by difference to see the largest discrepancies first
discrepancies = discrepancies.sort_values('difference', ascending=False)

# Display the results
print(f"Found {len(discrepancies)} points with significant differences")
print("\nTop 10 largest discrepancies:")
print(discrepancies[['Date', 'Value', 'W.L [cm]', 'difference']].head(10))

# Optional: Create a scatter plot to visualize the differences
plt.figure(figsize=(12, 6))
plt.scatter(discrepancies['Date'], discrepancies['difference'], alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Absolute Difference (mm)')
plt.title('Differences Between Raw Data and Vinge Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Optional: Create a scatter plot to visualize the differences
plt.figure(figsize=(12, 6))
plt.scatter(discrepancies['Value'], discrepancies['W.L [cm]'], alpha=0.5)
plt.xlabel('Value (mm)')
plt.ylabel('W.L (mm)')
plt.title('Differences Between Raw Data and Vinge Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Create a boolean mask for points with differences
has_difference = (merged_data['difference'] > lower_threshold) & (merged_data['difference'] < upper_threshold)

# Create groups of continuous drift periods
merged_data['drift_group'] = (~has_difference).cumsum()[has_difference]

# Calculate drift statistics
drift_stats = merged_data[has_difference].groupby('drift_group').agg({
    'Date': ['min', 'max', lambda x: (x.max() - x.min()).total_seconds() / (60 * 60 * 24)],  # Start, end, and duration in days
    'difference': ['mean', 'max', 'count']
}).reset_index()

# Rename columns for clarity
drift_stats.columns = ['drift_group', 'start_date', 'end_date', 'duration_days', 'mean_difference', 'max_difference', 'num_points']

# Calculate overall averages
avg_drift_duration = drift_stats['duration_days'].mean()
avg_drift_difference = drift_stats['mean_difference'].mean()

print("\nDrift Statistics:")
print(f"Number of distinct drifts: {len(drift_stats)}")
print(f"Average drift duration: {avg_drift_duration:.2f} days")
print(f"Average drift difference: {avg_drift_difference:.2f} mm")

# Display detailed drift statistics
print("\nDetailed drift statistics:")
print(drift_stats.sort_values('duration_days', ascending=False).head())

drift_stats.to_csv('drift_stats.csv', index=False)