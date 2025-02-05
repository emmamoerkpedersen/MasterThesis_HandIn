import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


raw_data = pd.read_csv('Sample data/21006846/VST_RAW.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
editted_level_data = pd.read_csv('Sample data/21006846/VST_EDT.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
vinge_data = pd.read_excel('Sample data/21006846/VINGE.xlsm', decimal = ',', header = 0)
#precipitation_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Emma/Sample data/RainData_05205.csv', parse_dates=['datetime'])

raw_data['Value'] = raw_data['Value'].astype(float)
# Convert Date column to datetime with specified format (YYYY-MM-DD HH:MM:SS)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d %H:%M:%S')
editted_level_data['Date'] = pd.to_datetime(editted_level_data['Date'], format='%d-%m-%Y %H:%M')
vinge_data['Date'] = pd.to_datetime(vinge_data['Date'], format='%d.%m.%Y %H:%M')

# Crop all dataframes to start from 2010-01-01
start_date = '2000-01-01'
raw_data = raw_data[raw_data['Date'] >= start_date]
editted_level_data = editted_level_data[editted_level_data['Date'] >= start_date]
vinge_data = vinge_data[vinge_data['Date'] >= start_date]
#precipitation_data = precipitation_data[precipitation_data['datetime'] >= start_date]

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

# Calculate summary statistics
summary_stats = pd.DataFrame({
    'drift_group': ['SUMMARY'],
    'start_date': [drift_stats['start_date'].min()],
    'end_date': [drift_stats['end_date'].max()],
    'duration_days': [drift_stats['duration_days'].mean()],
    'mean_difference': [drift_stats['mean_difference'].mean()],
    'max_difference': [drift_stats['max_difference'].max()],
    'num_points': [drift_stats['num_points'].sum()]
})

# Add percentile statistics
summary_stats['min_difference'] = drift_stats['mean_difference'].min()
summary_stats['25th_percentile'] = drift_stats['mean_difference'].quantile(0.25)
summary_stats['75th_percentile'] = drift_stats['mean_difference'].quantile(0.75)

# Concatenate the summary stats with the drift_stats
drift_stats_with_summary = pd.concat([drift_stats, summary_stats], ignore_index=True)

# Save to CSV
drift_stats_with_summary.to_csv('Data Errors/drift_stats_21006846.csv', index=False)

# Create an interactive plot using Plotly
fig = go.Figure()

# Add raw data line
fig.add_trace(go.Scatter(
    x=raw_data['Date'],
    y=raw_data['Value'],
    name='Raw Data',
    line=dict(color='blue', width=1),
    opacity=0.7
))

fig.add_trace(go.Scatter(
    x=editted_level_data['Date'],
    y=editted_level_data['Value'],
    name='Editted Data',
    line=dict(color='red', width=1),
    opacity=0.7
))
# Add vinge data points
fig.add_trace(go.Scatter(
    x=vinge_data['Date'],
    y=vinge_data['W.L [cm]'],
    name='Vinge Data',
    mode='markers',
    marker=dict(color='green', size=5),
    opacity=0.7
))

# Add colored rectangles for drift periods
for _, drift in drift_stats.iterrows():
    fig.add_vrect(
        x0=drift['start_date'],
        x1=drift['end_date'],
        fillcolor="red",
        opacity=0.2,
        layer="below",
        name="Drift Period",
        line_width=0
    )

# Update layout
fig.update_layout(
    title='Water Level Data Comparison with Highlighted Drift Periods',
    xaxis_title='Date',
    yaxis_title='Water Level (mm)',
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

# Save the plot to a specific location
output_path = 'plots/drift_analysis_21006846.html'  # Change this path as needed
os.makedirs('plots', exist_ok=True)  # Create the directory if it doesn't exist
fig.write_html(output_path)
webbrowser.open('file://' + os.path.abspath(output_path))
