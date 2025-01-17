import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define folder names
folders = ['21006845', '21006846', '21006847']

# Initialize an empty list to store dataframes
dfs = []

# Loop through each folder and read the data
for folder in folders:
    data_path = os.path.join('Sample data', folder, 'VST_RAW.txt')
    
    # Read the standardized data file
    df = pd.read_csv(data_path, sep=';', encoding='latin-1')
    
    # Convert Time column to datetime format using ISO format
    df['Time'] = pd.to_datetime(df['Time'])  # Let pandas automatically detect the format
    
    # Add a column to identify the source folder
    df['Source'] = folder
    
    dfs.append(df)

# Create figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
axes = [ax1, ax2, ax3]

# Define thresholds
lower_threshold = 0
upper_threshold = 2000
linear_window = 100  # Number of consecutive points to check for linear pattern
tolerance = 0.1  # Tolerance for detecting linear patterns

# Plot each dataset in its own subplot
for df, ax, folder in zip(dfs, axes, folders):
    # Plot the main data
    ax.plot(df['Time'], df['VST_raw'])
    
    # Highlight outliers (values < 0 or > 2000)
    outliers_mask = (df['VST_raw'] < lower_threshold) | (df['VST_raw'] > upper_threshold)
    if outliers_mask.any():
        outlier_idx = df.index[outliers_mask]
        for idx in outlier_idx:
            ax.axvspan(df['Time'].iloc[idx], df['Time'].iloc[idx+1] if idx < len(df)-1 else df['Time'].iloc[idx], 
                      color='red', alpha=0.2, label='Outlier')
    
    # Highlight NaN values
    nan_mask = df['VST_raw'].isna()
    if nan_mask.any():
        nan_idx = df.index[nan_mask]
        for idx in nan_idx:
            ax.axvspan(df['Time'].iloc[idx], df['Time'].iloc[idx+1] if idx < len(df)-1 else df['Time'].iloc[idx], 
                      color='yellow', alpha=0.2, label='Missing Data')
    
    # Detect and highlight linear patterns
    for i in range(len(df) - linear_window):
        window = df['VST_raw'].iloc[i:i+linear_window]
        if not window.isna().any():  # Skip windows with NaN values
            diffs = np.diff(window)
            if np.allclose(diffs, diffs[0], rtol=tolerance):  # Check if differences are constant
                ax.axvspan(df['Time'].iloc[i], df['Time'].iloc[i+linear_window], 
                          color='orange', alpha=0.2, label='Linear Pattern')
    
    ax.set_ylabel('VST Raw Value')
    ax.set_title(f'Data from {folder}')
    
    # Add legend (will only show unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

# Adjust layout and add overall title
plt.xlabel('Time')
fig.suptitle('VST Raw Data Over Time\n' + 
             'Red: Values < 0 or > 2000\n' +
             'Yellow: Missing Data\n' +
             'Orange: Linear Patterns', y=1.02)
plt.tight_layout()
plt.show()
