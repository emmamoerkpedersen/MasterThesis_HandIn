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
    
    # Add main water level data with reduced opacity
    fig.add_trace(
        go.Scattergl(
            name='Water Level',
            showlegend=True,
            opacity=0.5,
            line=dict(color='black', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
        ),
        hf_x=analyzer.df.index,
        hf_y=analyzer.df[analyzer.value_column]
    )
    
    # Add manual measurements if available with smaller markers
    if data['vinge'] is not None:
        fig.add_trace(
            go.Scatter(
                name='Manual Measurements',
                mode='markers',
                marker=dict(
                    color='red',
                    size=6,  # Reduced from 10
                    symbol='diamond'
                ),
                x=data['vinge']['Date'],
                y=data['vinge']['W.L [cm]'],
                hovertemplate='Manual Reading<br>Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            )
        )
    
    # Detect and add gaps with vertical lines
    gaps = analyzer.detect_gaps(min_gap_hours=48.0)
    gap_dates = analyzer.df[gaps].index
    gap_values = analyzer.df[gaps][analyzer.value_column]
    
    for date, value in zip(gap_dates, gap_values):
        fig.add_trace(
            go.Scatter(
                name='Data Gap',
                mode='lines',
                line=dict(color='red', width=2, dash='dot'),
                x=[date, date],
                y=[value - 200, value + 200],  # Vertical line ±200mm
                showlegend=False,
                hovertemplate='Data Gap<br>At: %{x}<extra></extra>'
            )
        )
    
    # Detect and add frozen values with highlighted regions
    frozen = analyzer.detect_frozen_values(min_consecutive=10)
    frozen_dates = analyzer.df[frozen].index
    frozen_values = analyzer.df[frozen][analyzer.value_column]
    
    if len(frozen_dates) > 0:
        fig.add_trace(
            go.Scatter(
                name='Frozen Values',
                mode='markers+lines',
                opacity=0.3,
                marker=dict(color='blue', size=4, symbol='square'),
                line=dict(color='lightblue', width=10),
                x=frozen_dates,
                y=frozen_values,
                hovertemplate='Frozen Values<br>Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            )
        )
    
    # Detect and add point anomalies with single markers at their center
    anomaly_segments = analyzer.detect_point_anomalies()
    if anomaly_segments:
        anomaly_x = []
        anomaly_y = []
        for start_idx, end_idx in anomaly_segments:
            # Find the middle point of the anomaly
            mid_idx = start_idx + (end_idx - start_idx) // 2
            anomaly_x.append(analyzer.df.index[mid_idx])
            anomaly_y.append(analyzer.df[analyzer.value_column].iloc[mid_idx])
        
        fig.add_trace(
            go.Scatter(
                name='Point Anomaly',
                mode='markers',
                marker=dict(
                    color='orange',
                    size=8,
                    symbol='x',
                    line=dict(width=2)
                ),
                x=anomaly_x,
                y=anomaly_y,
                hovertemplate='Point Anomaly<br>Date: %{x}<br>Value: %{y:.1f}mm<extra></extra>'
            )
        )
    
    # Detect and add offsets with highlighted regions and annotations
    offsets = analyzer.detect_offsets()
    first_offset = True
    for start_idx, end_idx, magnitude in offsets:
        offset_dates = analyzer.df.index[start_idx:end_idx]
        offset_values = analyzer.df[analyzer.value_column].iloc[start_idx:end_idx]
        
        # Add the offset period with highlighting
        fig.add_trace(
            go.Scatter(
                x=offset_dates,
                y=offset_values,
                mode='lines',
                line=dict(color='green', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.1)',
                name='Offset Error',
                showlegend=first_offset,
                hovertemplate=(
                    'Offset Error<br>'
                    'Start: %{x}<br>'
                    'Value: %{y:.1f}mm<br>'
                    f'Magnitude: {magnitude:.1f}mm'
                    '<extra></extra>'
                )
            )
        )
        
        # Calculate middle of offset period
        mid_point = offset_dates[0] + pd.Timedelta((offset_dates[-1] - offset_dates[0]) / 2)
        
        # Add a dotted vertical line for the offset
        fig.add_vline(
            x=mid_point,
            line_dash="dot",
            line_color="green",
            line_width=1,
            opacity=0.7
        )
        
        # Add annotation for offset magnitude
        fig.add_trace(
            go.Scatter(
                x=[mid_point],
                y=[offset_values.iloc[0]],
                mode='text',
                text=[f'↕ {magnitude:.0f}mm'],
                textposition='middle right',
                showlegend=False,
                hoverinfo='skip'
            )
        )
        first_offset = False
    
    # Add drift periods if vinge data is available
    if data['vinge'] is not None:
        drift_stats = analyzer.detect_drift(data['vinge'])
        
        # Add a dummy trace for the legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(width=10, color='rgba(255, 0, 0, 0.1)'),
                name='Drift Period'
            )
        )
        
        # Add colored rectangles for drift periods
        for _, drift in drift_stats.iterrows():
            fig.add_vrect(
                x0=drift['start_date'],
                x1=drift['end_date'],
                fillcolor="rgba(255, 0, 0, 0.1)",  # Light red
                opacity=0.5,
                layer="below",  # Place the rectangle behind the data
                line_width=0,  # No border
                name="Drift Period",
                showlegend=False  # Don't show individual rectangles in legend
            )
            
            # Add annotation for drift magnitude
            mid_point = drift['start_date'] + pd.Timedelta((drift['end_date'] - drift['start_date']) / 2)
            fig.add_trace(
                go.Scatter(
                    x=[mid_point],
                    y=[analyzer.df[analyzer.value_column].loc[drift['start_date']:drift['end_date']].mean()],
                    mode='text',
                    text=[f'Drift: {drift["mean_difference"]:.0f}mm'],
                    textposition='top center',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=f'Error Analysis Overview - Station {folder}',
            font=dict(size=24)
        ),
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
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            zeroline=False
        )
    )
    
    # Show the figure
    fig.show_dash(mode='inline', port=8050 + int(folder[-1])) 