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

