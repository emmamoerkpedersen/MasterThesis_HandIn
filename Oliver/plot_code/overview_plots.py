import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from plotly_resampler import FigureResampler, register_plotly_resampler, FigureWidgetResampler
import plotly.graph_objects as go
from dash import Dash, html, dcc
import dash

# Register the resampler to enable dynamic updates
register_plotly_resampler(mode='auto')

def plot_datasets_overview(data, folder, rain_data=None):
    """Create overview plot showing VST_RAW, VINGE and rainfall data.
    
    Args:
        data (dict): Dictionary containing different datasets
        folder (str): Station folder name/ID
        rain_data (pd.DataFrame, optional): Rainfall data with datetime and precipitation columns
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Create second y-axis for rainfall
    ax2 = ax1.twinx()
    
    # Plot VST_RAW data and get mean water level
    mean_water_level = None
    if data['vst_raw'] is not None:
        mean_water_level = data['vst_raw']['Value'].mean()
        ax1.plot(data['vst_raw']['Date'], data['vst_raw']['Value'],
                label='Raw sensor data', alpha=0.8, color='#1f77b4')
        # Add horizontal line at mean water level
        ax1.axhline(y=mean_water_level, color='#1f77b4', linestyle='--', alpha=0.3, label='Mean water level')
    
    # Plot VINGE data
    if data['vinge'] is not None:
        ax1.scatter(data['vinge']['Date'], data['vinge']['W.L [cm]'],
                   color='red', s=30, alpha=0.7, label='Manual measurements')
    
    # Get the water level axis limits
    ax1_min, ax1_max = ax1.get_ylim()
    ax1_range = ax1_max - ax1_min
    
    # Plot rainfall data on secondary y-axis
    if rain_data is not None and mean_water_level is not None:
        # Filter out negative values
        rain_data = rain_data[rain_data['precipitation (mm)'] >= 0]
        
        # Plot rainfall data as bars
        ax2.bar(rain_data['datetime'], rain_data['precipitation (mm)'],
                color='gray', alpha=0.3, label='Rainfall', width=1)
        
        # Set rainfall axis limits
        max_rain = rain_data['precipitation (mm)'].max()
        ax2.set_ylim(max_rain, 0)  # Invert the axis
        
        # Set rainfall axis label and color
        ax2.set_ylabel('Rainfall (mm)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.spines['right'].set_color('gray')
        
        # Align zero of rainfall with mean water level
        ax2_pos = ax2.get_position()
        ax1_pos = ax1.get_position()
        ax2.set_position([ax2_pos.x0, ax2_pos.y0,
                         ax2_pos.width, ax2_pos.height * (mean_water_level - ax1_min)/(ax1_max - ax1_min)])
    
    # Configure primary axis
    ax1.set_title(f'Dataset Overview - Station {folder}')
    ax1.set_ylabel('Water level (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(ax1_min, ax1_max)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()

def plot_vst_raw_overview(all_data):
    """Create overview plots showing raw VST data using plotly-resampler.
    
    Args:
        all_data (dict): Dictionary containing datasets for all folders
    """
    # Create a figure for each folder
    for folder, data in all_data.items():
        if data['vst_raw'] is None:
            continue
            
        # Create a FigureResampler object
        fig = FigureResampler(go.Figure())
        
        # Add VST_RAW data
        fig.add_trace(
            go.Scattergl(
                name='VST_RAW',
                showlegend=True,
                line=dict(color='#1f77b4', width=1),  # Blue line
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            ),
            hf_x=data['vst_raw']['Date'],
            hf_y=data['vst_raw']['Value']
        )
        
        # Add VST_EDT data if available
        if data['vst_edt'] is not None:
            fig.add_trace(
                go.Scattergl(
                    name='VST_EDT',
                    showlegend=True,
                    line=dict(color='red', width=1),  # Red line
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                ),
                hf_x=data['vst_edt']['Date'],
                hf_y=data['vst_edt']['Value']
            )
        
        # Add VINGE data if available
        if data['vinge'] is not None:
            fig.add_trace(
                go.Scatter(  # Using Scatter instead of Scattergl for points
                    name='Manual Measurements',
                    mode='markers',
                    marker=dict(color='black', size=6),  # Black dots
                    showlegend=True,
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>',
                    x=data['vinge']['Date'],
                    y=data['vinge']['W.L [cm]']
                )
            )

        # Update layout
        fig.update_layout(
            title=f'Water Level Data Overview - Station {folder}',
            yaxis_title='Water level (mm)',
            xaxis_title='Date',
            template='plotly_white',
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Show the figure in a Dash app
        fig.show_dash(mode='inline', port=8050 + int(folder[-1])) 