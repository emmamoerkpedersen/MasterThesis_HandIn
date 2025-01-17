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

# Load each VST_RAW.txt file
for folder in folders:
    file_path = os.path.join('Data', 'Sample data', folder, 'VST_RAW.txt')
    try:
        df = pd.read_csv(file_path, delimiter=';', decimal=',')
        vst_dfs[folder] = df
    except FileNotFoundError:
        print(f"Warning: VST_RAW.txt not found in folder {folder}")

# Create a figure with 3 subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
axes = [ax1, ax2, ax3]

# Plot each dataset in its own subplot
for (folder, df), ax in zip(vst_dfs.items(), axes):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    
    # Convert value column to float, replacing any invalid values with NaN
    df[' value'] = pd.to_numeric(df[' value'], errors='coerce')
    
    # Plot on the specific subplot
    ax.plot(df['Date'], df[' value'])
    
    # Customize each subplot
    ax.set_title(f'Dataset {folder}')
    ax.set_ylabel('Value')
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
df[' value'] = pd.to_numeric(df[' value'], errors='coerce')

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

# Define time windows based on specific periods of interest with custom titles
time_windows = [
    ((pd.to_datetime('1994-04-01'), pd.to_datetime('1994-10-01')), "Spike Error"),  # 1994 period
    ((pd.to_datetime('2002-10-01'), pd.to_datetime('2004-03-01')), "Missing Data Error"),  # 2003-2004 period
    ((pd.to_datetime('2011-01-01'), pd.to_datetime('2011-06-01')), "Offset Error"),  # 2011 period
    ((pd.to_datetime('2012-01-01'), pd.to_datetime('2012-06-01')), "Linear Interpolation"),  # 2012 period
    ((pd.to_datetime('2013-01-01'), pd.to_datetime('2013-06-01')), "Flat Line Error"),  # 2013 period
]

# If fewer than 5 windows specified, fill rest with empty dates
while len(time_windows) < 5:
    time_windows.append(((pd.to_datetime('1900-01-01'), pd.to_datetime('1900-01-02')), "Empty Section"))

# Create figure with subplots - 2 rows, 5 columns for bottom row
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 5, height_ratios=[2, 1], hspace=0.3)

# Define colors for the subplots
colors = ['red', 'green', 'black', 'purple', 'orange']

# Main plot spanning all columns
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(df['Date'], df[' value'], 'b-', label='Full Dataset')
ax_main.set_title(f'Complete Dataset {folder}')
ax_main.grid(True)

# Add major (year) and minor (month) ticks to main plot
ax_main.xaxis.set_major_locator(mdates.YearLocator())
ax_main.xaxis.set_minor_locator(mdates.MonthLocator())
ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_main.tick_params(axis='x', rotation=45)
ax_main.set_ylabel('Value')

# Find global y-limits for all subplots
y_min = float('inf')
y_max = float('-inf')
for (start_date, end_date), title in time_windows:
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    window_data = df.loc[mask]
    if len(window_data) > 0:
        valid_values = window_data[' value'].dropna()
        if len(valid_values) > 0:
            y_min = min(y_min, valid_values.min())
            y_max = max(y_max, valid_values.max())

# Add padding to y-limits
y_padding = (y_max - y_min) * 0.1
y_min -= y_padding
y_max += y_padding

# Calculate total duration and default window size for random windows
total_duration = df['Date'].max() - df['Date'].min()
window_size = pd.Timedelta(days=30)  # Default to 30-day windows

# If fewer than 5 windows specified, fill rest with random windows
while len(time_windows) < 5:
    # Random start time for the window
    random_start = df['Date'].min() + pd.Timedelta(seconds=np.random.random() * total_duration.total_seconds())
    random_end = random_start + window_size
    time_windows.append((random_start.strftime('%Y-%m-%d'), random_end.strftime('%Y-%m-%d')))

# Convert string dates to datetime for all windows
time_windows = [
    ((pd.to_datetime(start) if isinstance(start, str) else start,
     pd.to_datetime(end) if isinstance(end, str) else end), title)
    for (start, end), title in time_windows
]

for i in range(5):
    (start_date, end_date), title = time_windows[i]  # Unpack both dates and title
    
    # Get y-range for the window
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    window_data = df.loc[mask].copy()
    
    # Enhanced debug information
    print(f"\nSection {i+1} ({colors[i]}):")
    print(f"Time window requested: {start_date} to {end_date}")
    if len(window_data) > 0:
        print(f"Actual data range: {window_data['Date'].min()} to {window_data['Date'].max()}")
        print(f"Number of data points: {len(window_data)}")
        print(f"Value range: {window_data[' value'].min():.2f} to {window_data[' value'].max():.2f}")
    
    # Create subplot in bottom row
    if i == 0:
        ax = fig.add_subplot(gs[1, i])
        first_ax = ax
    else:
        ax = fig.add_subplot(gs[1, i], sharey=first_ax)
        plt.setp(ax.get_yticklabels(), visible=False)
    
    if len(window_data) > 0:
        ax.plot(window_data['Date'], window_data[' value'], color=colors[i])
        ax.set_ylim(y_min, y_max)
    
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
        ax.set_ylabel('Value')

# Adjust layout first
plt.tight_layout()

# Now add rectangles and connection lines after tight_layout
for i in range(5):
    ax = fig.axes[i+1]
    (start_date, end_date), title = time_windows[i]  # Unpack both dates and title
    
    # Get data for the time window
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    window_data = df.loc[mask]
    
    # Calculate y limits with valid data only
    valid_values = window_data[' value'].dropna()
    if len(valid_values) > 0:
        y_min = valid_values.min()
        y_max = valid_values.max()
        
        if np.isfinite(y_min) and np.isfinite(y_max):
            padding = (y_max - y_min) * 0.1
            
            # Draw rectangle in main plot
            rect = plt.Rectangle(
                (mdates.date2num(start_date), y_min - padding), 
                mdates.date2num(end_date) - mdates.date2num(start_date), 
                (y_max - y_min + 2*padding),
                fill=False, 
                color=colors[i],
                linewidth=1.5,
                transform=ax_main.transData,
                zorder=5
            )
            ax_main.add_patch(rect)
            
            # Get the center point of the data section
            rect_center_x = mdates.date2num(start_date + (end_date - start_date) / 2)
            rect_center_y = y_min + (y_max - y_min) / 2
            
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