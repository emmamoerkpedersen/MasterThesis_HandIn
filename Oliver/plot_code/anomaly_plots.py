import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator

def create_detailed_plot(df, vinge_df, time_windows, folder):
    """Create the main detailed plot with subplots showing anomaly examples."""
    # Create figure with subplots - 2 rows, 6 columns for bottom row
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 6, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Define colors using a gradient from red through orange and green to dark blues
    colors = ['#FF0000', '#FF8000', '#00B000', '#00A0FF', '#0040FF', '#000080']

    # Main plot spanning all columns
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(df['Date'], df['Value'], 'b-', label='Sensor Water Level')
    ax_main.scatter(vinge_df['Date'], vinge_df['W.L [cm]'], 
                   color='red', s=20, label='Manual Board Measurements')
    ax_main.set_title(f'Dataset: {folder}')
    ax_main.grid(True)
    ax_main.legend()

    # Add dynamic date formatter that changes based on zoom level
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_main.xaxis.set_major_locator(locator)
    ax_main.xaxis.set_major_formatter(formatter)
    ax_main.tick_params(axis='x', rotation=45)
    ax_main.set_ylabel('Water level (mm)')

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
        
        # Set a fixed number of ticks (e.g., 4 ticks)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Always use the same date format (YYYY-MM)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
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
                y_range_global = ax_main.get_ylim()[1] - ax_main.get_ylim()[0]
                x_range_global = mdates.date2num(df['Date'].max()) - mdates.date2num(df['Date'].min())
                scaling_factor = y_range_global / x_range_global
                y_height = x_width * scaling_factor
                
                # Apply a scale factor only to the last 4 subplots (index 2, 3, 4, 5)
                if i >= 2 and y_range is not None:
                    scale_factor = 2.0
                    y_height = (y_max - y_min) * scale_factor
                
                # Draw rectangle in main plot
                rect = plt.Rectangle(
                    (mdates.date2num(start_date), y_min),
                    x_width,
                    y_height,
                    fill=False, 
                    color=colors[i],
                    linewidth=1.5,
                    transform=ax_main.transData,
                    zorder=5
                )
                ax_main.add_patch(rect)
                
                # Get the center point of the data section
                rect_center_x = mdates.date2num(start_date + (end_date - start_date) / 2)
                rect_center_y = y_min + (y_height / 2)
                
                # Create dotted connection lines with higher endpoint
                con = ConnectionPatch(
                    xyA=(rect_center_x, rect_center_y),
                    coordsA=ax_main.transData,
                    xyB=(0.5, 1.15),
                    coordsB=ax.transAxes,
                    arrowstyle='->',
                    color=colors[i],
                    linewidth=0.8,
                    linestyle=':',
                    axesA=ax_main,
                    axesB=ax
                )
                fig.add_artist(con) 