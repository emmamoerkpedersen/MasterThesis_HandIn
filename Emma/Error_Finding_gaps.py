import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


raw_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006847/VST_RAW.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')


# Convert Date column to datetime with specified format (DD-MM-YYYY HH:MM)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y %H:%M')

# Crop all dataframes to start from 2010-01-01
start_date = '2000-01-01'
raw_data = raw_data[raw_data['Date'] >= start_date]

# Calculate time differences between consecutive measurements
time_differences = raw_data['Date'].diff()

# Find gaps larger than 15 minutes
gaps = raw_data[time_differences > pd.Timedelta(minutes=15)].copy()

# Add the previous timestamp to see the full gap interval
gaps['Previous_Date'] = raw_data['Date'].shift(1)[gaps.index]

# Create a DataFrame with gap information
gap_info = pd.DataFrame({
    'Gap_Start': gaps['Previous_Date'],
    'Gap_End': gaps['Date'],
    'Gap_Duration': gaps['Date'] - gaps['Previous_Date']
})

# Print the gaps
print(f"Found {len(gap_info)} gaps larger than 15 minutes:")
print(gap_info)

# Save gap information to CSV
output_path = 'data_gaps_21006847.csv'
gap_info.to_csv(output_path, index=False)
print(f"Gap information saved to {output_path}")

# # Create an interactive plot
# fig = go.Figure()

# # Add the raw data line
# fig.add_trace(go.Scatter(
#     x=raw_data['Date'],
#     y=raw_data['Value'],
#     name='Raw Data',
#     line=dict(color='blue')
# ))

# # Add colored background for gaps
# for _, gap in gap_info.iterrows():
#     fig.add_vrect(
#         x0=gap['Gap_Start'],
#         x1=gap['Gap_End'],
#         fillcolor="rgba(255, 0, 0, 0.1)",  # Light red color
#         layer="below",
#         line_width=0,
#         opacity=0.5
#     )

# # Update layout
# fig.update_layout(
#     title='Raw Data with Highlighted Gaps',
#     xaxis_title='Date',
#     yaxis_title='Value',
#     showlegend=True,
#     hovermode='x unified'
# )

# # Create plots directory if it doesn't exist
# if not os.path.exists('plots'):
#     os.makedirs('plots')

# # Save the plot as HTML
# fig.write_html('plots/raw_data_with_gaps.html')

# # Open the plot in browser (optional)
# webbrowser.open('plots/raw_data_with_gaps.html')

