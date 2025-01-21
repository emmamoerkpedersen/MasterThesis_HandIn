import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch

# List of folder names
folders = ['21006845', '21006846', '21006847']

# Dictionary to store dataframes
vst_dfs = {}

# Get the repository root path (assuming we're in the Oliver directory)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo_root, 'Sample data')

# Load each VST_RAW.txt file
for folder in folders:
    file_path = os.path.join(data_dir, folder, 'VST_RAW.txt')
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                # First try to read a few lines to check the format
                with open(file_path, 'r', encoding=encoding) as f:
                    first_lines = [next(f) for _ in range(5)]
                    print(f"\nFirst few lines of {folder}:")
                    print(''.join(first_lines))
                
                # Then try to read the file, skipping metadata and setting column names
                df = pd.read_csv(file_path, 
                               encoding=encoding,
                               delimiter=';',
                               decimal=',',
                               skiprows=3,  # Skip the metadata lines
                               names=['Date', 'Value'])  # Set column names
                vst_dfs[folder] = df
                break  # If successful, break the encoding loop
            except UnicodeDecodeError:
                continue  # Try next encoding
            except StopIteration:
                print(f"File {folder} has fewer than 5 lines")
                break
        if folder not in vst_dfs:
            print(f"Warning: Could not read file in folder {folder} with any supported encoding")
    except FileNotFoundError:
        print(f"Warning: VST_RAW.txt not found in folder {folder}")

# Load VINGE_LEVEL.txt data
vinge_df = pd.read_csv(os.path.join(data_dir, '21006845', 'VINGE.txt'), 
                       delimiter='\t',
                       encoding='latin1',
                       decimal=',',
                       quotechar='"')

# Clean up the data
vinge_df['Date'] = pd.to_datetime(vinge_df['Date'], format='%d.%m.%Y %H:%M')
# Convert from cm to mm (multiply by 10)
vinge_df['W.L [cm]'] = pd.to_numeric(vinge_df['W.L [cm]'], errors='coerce') * 10

# Filter for dates after 1990
vinge_df = vinge_df[vinge_df['Date'].dt.year >= 1990]

# Inspect the data
print("\nVINGE_LEVEL Data Inspection:")
print("Number of records:", len(vinge_df))
print("\nWater Level values (mm) summary:")
print(vinge_df['W.L [cm]'].describe())
print("\nFirst few rows of Water Level data (mm):")
print(vinge_df[['Date', 'W.L [cm]']].head())

# Create a figure with 3 subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
axes = [ax1, ax2, ax3]

# Plot each dataset in its own subplot
for (folder, df), ax in zip(vst_dfs.items(), axes):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    
    # Convert value column to float, replacing any invalid values with NaN
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Plot on the specific subplot
    ax.plot(df['Date'], df['Value'])
    
    # Customize each subplot
    ax.set_title(f'Dataset {folder}')
    ax.set_ylabel('Water level (mm)')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# Only show x-label on bottom subplot
axes[-1].set_xlabel('Date')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
#plt.show()

# Define anomalies descriptions
"""
Spikes (Single-point Anomalies)
    Sudden, extreme values that immediately return to normal
    Usually single points or very short duration
    Likely causes: Sensor glitches, electrical interference, measurement errors

Linear Interpolation Segments
    Perfectly straight lines in the data
    Often used to fill gaps in data
    Easy to identify due to their unnatural linear nature
    Problem: Doesn't represent real measurements

Offset Errors (Level Shifts)
    Sudden jump to different level
    Normal variation continues at new level
    Eventually returns to original baseline
    Data remains "active" but at wrong magnitude

Missing Data
    Gaps in the time series
    May be represented as NaN values or zeros
    Could be regular (scheduled) or irregular (failures)

Flat Line Segments
    Periods where values remain constant
    Could indicate sensor freezing
    May represent "stuck" readings or default values
    Different from linear interpolation as it's typically horizontal

Calibration Issues
    Gradual drift in values
    Step changes between different levels
    May be related to sensor recalibration events
"""

# Choose one dataset (first one in the dictionary)
folder = list(vst_dfs.keys())[0]
df = vst_dfs[folder]

# Convert Date column to datetime and value column to float (if not already done)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Before defining time_windows, let's analyze data availability
print("\nData Availability Analysis:")
print(f"Dataset starts at: {df['Date'].min()}")
print(f"Dataset ends at: {df['Date'].max()}")

# Find gaps in the data
df_sorted = df.sort_values('Date')
time_diff = df_sorted['Date'].diff()
gaps = time_diff[time_diff > pd.Timedelta(days=7)]  # Find gaps longer than 7 days
print("\nMajor gaps in data:")
for idx in gaps.index:
    gap_start = df_sorted['Date'][idx - 1]
    gap_end = df_sorted['Date'][idx]
    print(f"Gap from {gap_start} to {gap_end} ({(gap_end - gap_start).days} days)")

time_windows = [
    {
        "title": "Data gaps",
        "start_date": pd.to_datetime('1994-01-01'),
        "end_date": pd.to_datetime('1995-01-01'),
        "y_range": (250, 2000)
    },
    {
        "title": "Linear Interpolation segment",
        "start_date": pd.to_datetime('1998-01-01'),
        "end_date": pd.to_datetime('2002-11-01'),
        "y_range": None
    },
    {
        "title": "Offset error",
        "start_date": pd.to_datetime('2007-03-21'),
        "end_date": pd.to_datetime('2007-03-23'),
        "y_range": (-7820, -7700)
    },
    {
        "title": "Spike error",
        "start_date": pd.to_datetime('2011-01-01'),
        "end_date": pd.to_datetime('2011-05-01'),
        "y_range": None
    },
    {
        "title": "Long offset error",
        "start_date": pd.to_datetime('2016-08-16'),
        "end_date": pd.to_datetime('2016-09-02'),
        "y_range": (10, 45)
    },
    {
        "title": "Spike fluctuations & flatline",
        "start_date": pd.to_datetime('2016-12-11'),
        "end_date": pd.to_datetime('2016-12-23'),
        "y_range": None
    }
]

# If fewer than 5 windows specified, fill rest with empty dates
while len(time_windows) < 6:
    time_windows.append({
        "title": "Empty Section",
        "start_date": pd.to_datetime('1900-01-01'),
        "end_date": pd.to_datetime('1900-01-02'),
        "y_range": None  # Use default y-axis range
    })

# Create figure with subplots - 2 rows, 6 columns for bottom row
fig = plt.figure(figsize=(24, 12))  # Increased width to accommodate 6 plots
gs = fig.add_gridspec(2, 6, height_ratios=[2, 1], hspace=0.3)

# Define colors for the subplots (added one more color)
colors = ['red', 'green', 'orange', 'yellow', 'purple', 'lightblue']

# Main plot spanning all columns
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(df['Date'], df['Value'], 'b-', label='Water Level')
ax_main.scatter(vinge_df['Date'], vinge_df['W.L [cm]'], color='red', s=30, label='Manual Measurements')
ax_main.set_title(f'Dataset: {folder}')
ax_main.grid(True)
ax_main.legend()

# Add dynamic date formatter that changes based on zoom level
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax_main.xaxis.set_major_locator(locator)
ax_main.xaxis.set_major_formatter(formatter)
ax_main.tick_params(axis='x', rotation=45)
ax_main.set_ylabel('Value')

# Plot each subplot without sharing the y-axis
for i in range(6):
    window = time_windows[i]
    start_date = window["start_date"]
    end_date = window["end_date"]
    title = window["title"]
    y_range = window["y_range"]
    
    # Create subplot in bottom row
    ax = fig.add_subplot(gs[1, i])
    
    # Get data for the time window
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    window_data = df.loc[mask]
    
    # Get board data for the time window
    board_mask = (vinge_df['Date'] >= start_date) & (vinge_df['Date'] <= end_date)
    board_window_data = vinge_df.loc[board_mask]
    
    if len(window_data) > 0:
        ax.plot(window_data['Date'], window_data['Value'], color=colors[i])
        # Add board data points
        ax.scatter(board_window_data['Date'], board_window_data['W.L [cm]'], 
                  color='red', s=30)
        
        # Set custom y-axis range if specified
        if y_range is not None:
            ax.set_ylim(y_range[0], y_range[1])
    
    # Remove gridlines
    ax.grid(False)
    
    # Always set the x-axis limits to the requested time window
    ax.set_xlim(start_date, end_date)
    
    # Calculate time span and set ticks
    time_span = end_date - start_date
    days_span = time_span.days
    
    # Use 4 ticks for all plots by setting appropriate locator
    if days_span > 365:
        # For periods longer than a year, show ticks every 6 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    else:
        # For shorter periods, show quarterly ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Show every other month
    
    # Always use the same date format (YYYY-MM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Customize subplot
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    if i == 0:
        ax.set_ylabel('Water level (mm)')

# Adjust layout first
plt.tight_layout()

# Now add rectangles and connection lines after tight_layout
for i in range(6):
    ax = fig.axes[i+1]
    window = time_windows[i]
    start_date = window["start_date"]
    end_date = window["end_date"]
    title = window["title"]
    y_range = window["y_range"]
    
    # Get data for the time window
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    window_data = df.loc[mask]
    
    # Calculate y limits with valid data only
    valid_values = window_data['Value'].dropna()
    if len(valid_values) > 0:
        y_min = valid_values.min() if y_range is None else y_range[0]
        y_max = valid_values.max() if y_range is None else y_range[1]
        
        if np.isfinite(y_min) and np.isfinite(y_max):
            # Calculate the width of the time window in x-axis units
            x_width = mdates.date2num(end_date) - mdates.date2num(start_date)
            
            # Calculate the height of the rectangle to make it square-like
            # Scale the height to match the width, considering the y-axis range
            y_range_global = ax_main.get_ylim()[1] - ax_main.get_ylim()[0]
            x_range_global = mdates.date2num(df['Date'].max()) - mdates.date2num(df['Date'].min())
            scaling_factor = y_range_global / x_range_global
            y_height = x_width * scaling_factor
            
            # Apply a scale factor only to the last 4 subplots (index 2, 3, 4, 5)
            if i >= 2 and y_range is not None:
                scale_factor = 2.0  # Adjust this value to make the rectangle larger
                y_height = (y_max - y_min) * scale_factor  # Scale the height of the rectangle
            
            # Draw rectangle in main plot
            rect = plt.Rectangle(
                (mdates.date2num(start_date), y_min),  # Bottom-left corner
                x_width,  # Width of the rectangle
                y_height,  # Height of the rectangle (scaled for selected subplots)
                fill=False, 
                color=colors[i],
                linewidth=1.5,
                transform=ax_main.transData,
                zorder=5
            )
            ax_main.add_patch(rect)
            
            # Get the center point of the data section
            rect_center_x = mdates.date2num(start_date + (end_date - start_date) / 2)
            rect_center_y = y_min + (y_height / 2)  # Center the arrow within the scaled rectangle
            
            # Create dotted connection lines with higher endpoint
            con = ConnectionPatch(
                xyA=(rect_center_x, rect_center_y),
                coordsA=ax_main.transData,
                xyB=(0.5, 1.15),  # Moved up from 1.05 to 1.15
                coordsB=ax.transAxes,
                arrowstyle='->',
                color=colors[i],
                linewidth=0.8,
                linestyle=':',
                axesA=ax_main,
                axesB=ax
            )
            fig.add_artist(con)

plt.show()