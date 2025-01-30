import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
from pathlib import Path
from analysis import ErrorAnalyzer
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

from matplotlib.ticker import MaxNLocator
from analysis import ErrorAnalyzer

def create_detailed_plot(data, time_windows, folder):
    """Create the main detailed plot with subplots showing anomaly examples."""
    if data['vst_raw'] is None:
        print(f"Missing VST_RAW data for detailed plot in folder {folder}")
        return
        
    # Create figure with subplots - 2 rows, 6 columns for bottom row
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 6, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Define colors using a gradient from red through orange and green to dark blues
    colors = ['#FF0000', '#FF8000', '#00B000', '#00A0FF', '#0040FF', '#000080']

    # Main plot spanning all columns
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(data['vst_raw']['Date'], data['vst_raw']['Value'], 
                'b-', label='Sensor Water Level')
    
    if data['vinge'] is not None:
        ax_main.scatter(data['vinge']['Date'], data['vinge']['W.L [cm]'], 
                       color='red', s=20, label='Manual Board Measurements')
    
    ax_main.set_title(f'Detailed Analysis - {folder}')
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
        mask = (data['vst_raw']['Date'] >= start_date) & (data['vst_raw']['Date'] <= end_date)
        window_data = data['vst_raw'][mask]
        
        # Get board data for the time window if available
        if data['vinge'] is not None:
            board_mask = (data['vinge']['Date'] >= start_date) & (data['vinge']['Date'] <= end_date)
            board_window_data = data['vinge'][board_mask]
            
            ax.scatter(board_window_data['Date'], board_window_data['W.L [cm]'], 
                      color='red', s=30)
        
        if len(window_data) > 0:
            ax.plot(window_data['Date'], window_data['Value'], color=colors[i])
            
            # Set custom y-axis range if specified
            if y_range is not None:
                ax.set_ylim(y_range[0], y_range[1])
        
        # Remove gridlines
        ax.grid(False)
        
        # Always set the x-axis limits to the requested time window
        ax.set_xlim(start_date, end_date)
        
        # Set a fixed number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Always use the same date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Customize subplot
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        if i == 0:
            ax.set_ylabel('Water level (mm)')

    # Adjust layout first
    plt.tight_layout()
    
    # Add rectangles and connection lines
    for i in range(6):
        ax = fig.axes[i+1]
        window = time_windows[i]
        start_date = window["start_date"]
        end_date = window["end_date"]
        y_range = window["y_range"]
        
        # Get data for the time window
        mask = (data['vst_raw']['Date'] >= start_date) & (data['vst_raw']['Date'] <= end_date)
        window_data = data['vst_raw'][mask]
        
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
                x_range_global = mdates.date2num(data['vst_raw']['Date'].max()) - mdates.date2num(data['vst_raw']['Date'].min())
                scaling_factor = y_range_global / x_range_global
                y_height = x_width * scaling_factor
                
                # Apply a scale factor only to the last 4 subplots
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
                
                # Create dotted connection lines
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
    
    # Save the plot
    plot_dir = Path(r"C:\Users\olive\OneDrive\GitHub\MasterThesis\plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f'detailed_analysis_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_errors(data, folder):
    """Create a comprehensive plot showing all detected error types using Plotly."""
    # Initialize error analyzer
    analyzer = ErrorAnalyzer(data['vst_raw'])
    
    # Create figure with plotly-resampler
    fig = FigureResampler(go.Figure())
    
    # Add main water level data
    fig.add_trace(
        go.Scattergl(
            name='Water Level',
            showlegend=True,
            line=dict(color='black', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
        ),
        hf_x=data['vst_raw']['Date'],
        hf_y=data['vst_raw']['Value']
    )
    
    # Add manual measurements if available
    if data['vinge'] is not None:
        fig.add_trace(
            go.Scatter(
                name='Manual Measurements',
                mode='markers',
                marker=dict(color='red', size=8),
                x=data['vinge']['Date'],
                y=data['vinge']['W.L [cm]'],
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            )
        )
    
    # Detect and add gaps
    gaps = analyzer.detect_gaps(max_gap_hours=3.0)
    gap_dates = data['vst_raw'].loc[gaps, 'Date']
    gap_values = data['vst_raw'].loc[gaps, 'Value']
    fig.add_trace(
        go.Scatter(
            name='Gaps',
            mode='markers',
            marker=dict(color='red', size=5, symbol='x'),
            x=gap_dates,
            y=gap_values,
            hovertemplate='Gap at: %{x}<extra></extra>'
        )
    )
    
    # Detect and add noise
    noise = analyzer.detect_noise()
    noise_dates = data['vst_raw'].loc[noise, 'Date']
    noise_values = data['vst_raw'].loc[noise, 'Value']
    fig.add_trace(
        go.Scatter(
            name='Noise',
            mode='markers',
            marker=dict(color='yellow', size=5),
            x=noise_dates,
            y=noise_values,
            hovertemplate='Noise at: %{x}<extra></extra>'
        )
    )
    
    # Detect and add frozen values
    frozen = analyzer.detect_frozen_values(min_consecutive=10)
    frozen_dates = data['vst_raw'].loc[frozen, 'Date']
    frozen_values = data['vst_raw'].loc[frozen, 'Value']
    fig.add_trace(
        go.Scatter(
            name='Frozen Values',
            mode='markers',
            marker=dict(color='blue', size=5),
            x=frozen_dates,
            y=frozen_values,
            hovertemplate='Frozen at: %{x}<extra></extra>'
        )
    )
    
    # Detect and add point anomalies
    anomaly_segments = analyzer.detect_point_anomalies()
    first_anomaly = True
    for start_idx, end_idx in anomaly_segments:
        # Add some context before and after the anomaly
        context_start = max(0, start_idx - int(analyzer.SAMPLES_PER_HOUR))
        context_end = min(len(data['vst_raw']), end_idx + int(analyzer.SAMPLES_PER_HOUR))
        
        # Plot the context period
        fig.add_trace(
            go.Scatter(
                x=data['vst_raw']['Date'].iloc[context_start:context_end],
                y=data['vst_raw']['Value'].iloc[context_start:context_end],
                mode='lines',
                line=dict(color='orange', width=2),
                name='Point Anomaly',
                showlegend=first_anomaly,
                hovertemplate='Anomaly at: %{x}<br>Value: %{y:.1f}mm<extra></extra>'
            )
        )
        first_anomaly = False
    
    # Detect and add offsets
    offsets = analyzer.detect_offsets()
    for start_idx, end_idx, magnitude in offsets:
        fig.add_trace(
            go.Scatter(
                name=f'Offset ({magnitude:.1f}mm)',
                mode='markers',
                marker=dict(color='green', size=5),
                x=data['vst_raw']['Date'].iloc[start_idx:end_idx],
                y=data['vst_raw']['Value'].iloc[start_idx:end_idx],
                hovertemplate='Offset: %{y:.1f}mm<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'Error Analysis Overview - Station {folder}',
        yaxis_title='Water level (mm)',
        xaxis_title='Date',
        template='plotly_white',
        height=800,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Show the figure
    fig.show_dash(mode='inline', port=8050 + int(folder[-1])) 