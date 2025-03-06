"""Diagnostic tools for data preprocessing step."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec

def set_plot_style():
    """Set consistent plot style with larger fonts and professional appearance."""
    plt.style.use('seaborn-v0_8-whitegrid')  # Start with a clean base style
    plt.rcParams.update({
        # Typography
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        
        # Figure
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        
        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 8,
        
        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 10,
        'axes.titlepad': 20,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '#cccccc',
        
        # Layout
        'figure.constrained_layout.use': True
    })

def plot_preprocessing_comparison(original_data: dict, preprocessed_data: dict, output_dir: Path, periods_to_remove: dict = None):
    """
    Create comparison plots between original and preprocessed data for each station.
    
    Args:
        original_data: Dictionary containing original station data
        preprocessed_data: Dictionary containing preprocessed station data
        output_dir: Output directory path
        periods_to_remove: Dictionary with station names as keys and lists of (start, end) tuples for freezing periods
    """
    # Update plot style with larger fonts
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 22
    })
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with minimal spacing
            fig = plt.figure(figsize=(15, 15))  # Square figure
            gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.2], hspace=0.25)  # Reduced spacing
            
            orig = original_data[station_name]['vst_raw']
            proc = preprocessed_data[station_name]['vst_raw']
            
            # Calculate IQR bounds
            Q1 = orig['vst_raw'].quantile(0.25)
            Q3 = orig['vst_raw'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Calculate removed points mask here, before using it
            removed_mask = (orig['vst_raw'] < lower_bound) | (orig['vst_raw'] > upper_bound)
            
            # Original data with bounds
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(orig.index, orig['vst_raw'], color='#1f77b4', alpha=0.7, 
                    linewidth=1, label='Original')
            ax1.axhline(y=lower_bound, color='#d62728', linestyle='--', alpha=0.7,
                       label='Lower bound (Q1 - 1.5*IQR)')
            ax1.axhline(y=upper_bound, color='#d62728', linestyle='--', alpha=0.7,
                       label='Upper bound (Q3 + 4*IQR)')
            
            # Add statistics box with freezing period info
            stats_text = (f'Q1: {Q1:.1f}\nQ3: {Q3:.1f}\n'
                        f'IQR: {IQR:.1f}\n'
                        f'Outliers: {sum(removed_mask)}')
            
            # Add freezing period info if available
            if periods_to_remove is not None and station_name in periods_to_remove:
                station_periods = periods_to_remove[station_name]
                freezing_points = sum(~((orig["Date"] >= start) & (orig["Date"] <= end)) 
                                    for start, end in station_periods)
                stats_text += f'\nFreezing periods: {len(station_periods)}'
                stats_text += f'\nPoints removed (freezing): {freezing_points}'
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            alpha=0.8,
                            edgecolor='#cccccc'))
            
            # Preprocessed data
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(proc.index, proc['vst_raw'], color='#2ca02c', alpha=0.7, 
                    linewidth=1, label='Preprocessed')
            
            # Removed points highlighted
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(orig.index, orig['vst_raw'], color='#1f77b4', alpha=0.3, 
                    linewidth=1, label='Original')
            removed_points = orig[removed_mask]
            ax3.scatter(removed_points.index, removed_points['vst_raw'], 
                       color='#d62728', s=50, alpha=0.5, label='Removed Points')
            
            # Set titles with adjusted padding
            #ax1.set_title(f'Original Data with IQR Bounds - {station_name}', 
             #            pad=10, y=1.0)
            #ax2.set_title('Preprocessed Data', pad=10, y=1.0)
            #ax3.set_title('Removed Data Points', pad=10, y=1.0)
            
            # Adjust label padding
            ax1.set_ylabel('Water Level (mm)', labelpad=10)
            ax2.set_ylabel('Water Level (mm)', labelpad=10)
            ax3.set_xlabel('Date', labelpad=10)
            ax3.set_ylabel('Water Level (mm)', labelpad=10)
            
            # Add individual legends in lower right corner with larger font
            ax1.legend(loc='lower right', frameon=True, framealpha=0.8, 
                      edgecolor='#cccccc', fontsize=16)
            ax2.legend(loc='lower right', frameon=True, framealpha=0.8, 
                      edgecolor='#cccccc', fontsize=16)
            ax3.legend(loc='lower right', frameon=True, framealpha=0.8, 
                      edgecolor='#cccccc', fontsize=16)
            
            # Remove grid lines and add subtle background color
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor('#f8f9fa')
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                    spine.set_linewidth(0.5)
                ax.tick_params(axis='both', which='major', pad=8)  # Reduced tick padding
                ax.tick_params(axis='x', rotation=45)
            
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", 
                       dpi=300,
                       bbox_inches='tight',
                       facecolor='white',
                       pad_inches=0.3)  # Reduced padding
            plt.close()

def generate_preprocessing_report(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Generate a report summarizing the preprocessing results."""
    report_dir = output_dir / "diagnostics" / "preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "preprocessing_report.txt", "w") as f:
        f.write("Preprocessing Diagnostics Report\n")
        f.write("==============================\n\n")
        
        for station_name, station_data in preprocessed_data.items():
            f.write(f"\nStation: {station_name}\n")
            f.write("-" * (len(station_name) + 9) + "\n")
            
            # VST_RAW details with IQR statistics
            if station_data['vst_raw'] is not None:
                vst = station_data['vst_raw']
                
                f.write("\nVST_RAW Data:\n")
                f.write(f"  - Measurements: {len(vst)}\n")
                f.write(f"  - Time range: {pd.to_datetime(vst.index.min())} to {pd.to_datetime(vst.index.max())}\n")
                f.write(f"  - Value range: {vst['vst_raw'].min():.2f} to {vst['vst_raw'].max():.2f}\n")
                
                # Only include IQR statistics if original_data is provided
                if original_data is not None and station_name in original_data and original_data[station_name]['vst_raw'] is not None:
                    orig_vst = original_data[station_name]['vst_raw']
                    
                    # Calculate IQR statistics
                    Q1 = orig_vst['vst_raw'].quantile(0.25)
                    Q3 = orig_vst['vst_raw'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 4 * IQR
                    
                    f.write("\nIQR Statistics:\n")
                    f.write(f"  - Q1: {Q1:.2f}\n")
                    f.write(f"  - Q3: {Q3:.2f}\n")
                    f.write(f"  - IQR: {IQR:.2f}\n")
                    f.write(f"  - Lower bound: {lower_bound:.2f}\n")
                    f.write(f"  - Upper bound: {upper_bound:.2f}\n")
                    
                    points_below = len(orig_vst[orig_vst['vst_raw'] < lower_bound])
                    points_above = len(orig_vst[orig_vst['vst_raw'] > upper_bound])
                    f.write("\nPoints removed by bounds:\n")
                    f.write(f"  - Below lower bound: {points_below}\n")
                    f.write(f"  - Above upper bound: {points_above}\n")
                    f.write(f"  - Total points removed: {points_below + points_above}\n")
            
            # VINGE details
            if station_data['vinge'] is not None:
                vinge = station_data['vinge']
                vinge_dates = pd.to_datetime(vinge.index)
                f.write("\nVINGE Measurements:\n")
                f.write(f"  - Number of measurements: {len(vinge)}\n")
                f.write(f"  - Average measurements per year: {len(vinge)/len(vinge_dates.year.unique()):.1f}\n")
                f.write(f"  - Years covered: {vinge_dates.year.min()} to {vinge_dates.year.max()}\n")
            
            # Rainfall details
            if station_data['rainfall'] is not None:
                rain = station_data['rainfall']
                f.write("\nRainfall Data:\n")
                f.write(f"  - Measurements: {len(rain)}\n")
                f.write(f"  - Time period: {rain.index.min().strftime('%Y-%m-%d')} to {rain.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"  - Total rainfall: {rain['rainfall'].sum():.1f} mm\n")
                f.write(f"  - Years covered: {rain.index.year.min()} to {rain.index.year.max()}\n")
            
            # Temperature details
            if station_data['temperature'] is not None:
                temp = station_data['temperature']
                f.write("\nTemperature Data:\n")
                f.write(f"  - Measurements: {len(temp)}\n")
                f.write(f"  - Time period: {temp.index.min().strftime('%Y-%m-%d')} to {temp.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"  - Years covered: {temp.index.year.min()} to {temp.index.year.max()}\n")
                f.write(f"  - Range: {temp['temperature'].min():.1f}°C to {temp['temperature'].max():.1f}°C\n")
            
            f.write("\n" + "="*50 + "\n")

def plot_station_data_overview(original_data: dict, preprocessed_data: dict, output_dir: Path):
    """Create comprehensive visualization of all data types for each station."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name in preprocessed_data.keys():
        if preprocessed_data[station_name]['vst_raw'] is not None:
            # Create figure with GridSpec for better control
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
            
            # 1. Water level measurements (VST_RAW, VST_EDT, VINGE)
            ax1 = fig.add_subplot(gs[0])
            data = preprocessed_data[station_name]
            
            # Plot VST Raw with custom styling
            ax1.plot(data['vst_raw'].index, data['vst_raw']['vst_raw'],
                    color='#1f77b4', alpha=0.7, linewidth=1, label='VST Raw')
            
            if data['vst_edt'] is not None:
                ax1.plot(data['vst_edt'].index, data['vst_edt']['vst_raw'],
                        color='#2ca02c', alpha=0.7, linewidth=1, label='VST EDT')
            
            if data['vinge'] is not None:
                ax1.scatter(data['vinge'].index, data['vinge']['W.L [cm]'],
                          color='#d62728', alpha=0.7, s=50, label='VINGE')
            
            # Add statistics box for water levels
            stats_text = (f'VST Raw Points: {len(data["vst_raw"])}\n'
                        f'Range: {data["vst_raw"]["vst_raw"].min():.1f} to '
                        f'{data["vst_raw"]["vst_raw"].max():.1f}')
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            alpha=0.8,
                            edgecolor='#cccccc'))
            
            # 2. Rainfall data
            ax2 = fig.add_subplot(gs[1])
            if data['rainfall'] is not None:
                rain_data = data['rainfall']
                ax2.plot(rain_data.index, rain_data['rainfall'],
                        color='#1f77b4', alpha=0.7, linewidth=1, label='Rainfall')
                
                # Add rainfall statistics
                total_rain = rain_data['rainfall'].sum()
                rain_stats = f'Total Rainfall: {total_rain:.1f}mm'
                ax2.text(0.02, 0.98, rain_stats, transform=ax2.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white',
                                alpha=0.8,
                                edgecolor='#cccccc'))
            
            # 3. Temperature data
            ax3 = fig.add_subplot(gs[2])
            if data['temperature'] is not None:
                temp_data = data['temperature']
                ax3.plot(temp_data.index, temp_data['temperature'],
                        color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
                
                # Add temperature statistics
                temp_stats = (f'Range: {temp_data["temperature"].min():.1f}°C to '
                            f'{temp_data["temperature"].max():.1f}°C')
                ax3.text(0.02, 0.98, temp_stats, transform=ax3.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white',
                                alpha=0.8,
                                edgecolor='#cccccc'))
            
            # 4. VINGE measurements detail
            ax4 = fig.add_subplot(gs[3])
            if data['vinge'] is not None:
                vinge_data = data['vinge']
                ax4.scatter(vinge_data.index, vinge_data['W.L [cm]'],
                          color='#d62728', alpha=0.7, s=50, label='VINGE')
                ax4.plot(vinge_data.index, vinge_data['W.L [cm]'],
                        color='#d62728', alpha=0.3, linewidth=1)
                
                # Add VINGE statistics
                vinge_stats = f'Manual Measurements: {len(vinge_data)}'
                ax4.text(0.02, 0.98, vinge_stats, transform=ax4.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white',
                                alpha=0.8,
                                edgecolor='#cccccc'))
            
            # Set titles and labels with consistent styling
            axes = [ax1, ax2, ax3, ax4]
            titles = ['Water Level Measurements', 'Rainfall Data',
                     'Temperature Data', 'Manual Board Measurements (VINGE)']
            ylabels = ['Water Level (mm)', 'rainfall',
                      'Temperature (°C)', 'Water Level (mm)']
            
            for ax, title, ylabel in zip(axes, titles, ylabels):
                ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, labelpad=10)
                ax.legend(frameon=True, framealpha=0.8, edgecolor='#cccccc')
                ax.set_facecolor('#f8f9fa')
                
                # Style the grid
                ax.grid(True, alpha=0.3, color='#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
            
            # Set the main title
            fig.suptitle(f'Station Data Overview - {station_name}',
                        fontsize=16, fontweight='bold', y=1.02)
            
            # Save the figure
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

def plot_additional_data(preprocessed_data: dict, output_dir: Path):
    """Create additional plots for VINGE, rainfall and temperature data."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None:
            # VST and VINGE comparison
            fig = plt.figure(figsize=(15, 8))
            gs = GridSpec(1, 1)
            ax = fig.add_subplot(gs[0])
            
            # Make sure we're using datetime for x-axis
            vst_data = station_data['vst_raw'].copy()
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            # Plot VST data with custom styling
            ax.plot(vst_data.index, vst_data['vst_raw'],
                   color='#1f77b4', alpha=0.7, linewidth=1, label='VST Raw')
            
            if station_data['vinge'] is not None:
                vinge_data = station_data['vinge'].copy()
                if not isinstance(vinge_data.index, pd.DatetimeIndex):
                    vinge_data.set_index('Date', inplace=True)
                    
                # Plot VINGE data with custom styling
                ax.scatter(vinge_data.index, vinge_data['W.L [cm]'],
                         color='#d62728', alpha=0.7, s=50, label='VINGE')
                
                # Add statistics box
                stats_text = (
                    f'VST Points: {len(vst_data):,}\n'
                    f'VINGE Points: {len(vinge_data)}\n'
                    f'Time Range: {vst_data.index.min().strftime("%Y-%m-%d")} to '
                    f'{vst_data.index.max().strftime("%Y-%m-%d")}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='white',
                               alpha=0.8,
                               edgecolor='#cccccc'))
            
            # Style the plot
            ax.set_title(f'VST Raw and VINGE Measurements - {station_name}',
                        pad=20, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel('Water Level (mm)', labelpad=10)
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3, color='#cccccc')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Style the legend
            ax.legend(frameon=True, framealpha=0.8, edgecolor='#cccccc',
                     loc='upper right')
            
            plt.savefig(diagnostic_dir / f"{station_name}_vst_vinge.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Rainfall data plot
            if station_data['rainfall'] is not None:
                fig = plt.figure(figsize=(15, 8))
                gs = GridSpec(1, 1)
                ax = fig.add_subplot(gs[0])
                
                rain_data = station_data['rainfall']
                ax.plot(rain_data.index, rain_data['rainfall'],
                       color='#1f77b4', alpha=0.7, linewidth=1, label='Rainfall')
                
                # Add rainfall statistics
                total_rain = rain_data['rainfall'].sum()
                rain_stats = (
                    f'Total Rainfall: {total_rain:.1f}mm\n'
                    f'Average: {rain_data["rainfall"].mean():.2f}mm/day\n'
                    f'Max: {rain_data["rainfall"].max():.1f}mm'
                )
                ax.text(0.02, 0.98, rain_stats, transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='white',
                               alpha=0.8,
                               edgecolor='#cccccc'))
                
                # Style the plot
                ax.set_title(f'Rainfall Data - {station_name}',
                           pad=20, fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', labelpad=10)
                ax.set_ylabel('rainfall', labelpad=10)
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, alpha=0.3, color='#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.legend(frameon=True, framealpha=0.8, edgecolor='#cccccc')
                
                plt.savefig(diagnostic_dir / f"{station_name}_rainfall.png",
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
            
            # Temperature data plot
            if station_data['temperature'] is not None:
                fig = plt.figure(figsize=(15, 8))
                gs = GridSpec(1, 1)
                ax = fig.add_subplot(gs[0])
                
                temp_data = station_data['temperature']
                ax.plot(temp_data.index, temp_data['temperature'],
                       color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
                
                # Add freezing line
                ax.axhline(y=0, color='#2ca02c', linestyle='--', alpha=0.7,
                          label='Freezing Point')
                
                # Add temperature statistics
                temp_stats = (
                    f'Range: {temp_data["temperature"].min():.1f}°C to '
                    f'{temp_data["temperature"].max():.1f}°C\n'
                    f'Average: {temp_data["temperature"].mean():.1f}°C\n'
                    f'Days Below 0°C: {sum(temp_data["temperature"] < 0)}'
                )
                ax.text(0.02, 0.98, temp_stats, transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='white',
                               alpha=0.8,
                               edgecolor='#cccccc'))
                
                # Style the plot
                ax.set_title(f'Temperature Data - {station_name}',
                           pad=20, fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', labelpad=10)
                ax.set_ylabel('Temperature (°C)', labelpad=10)
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, alpha=0.3, color='#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.legend(frameon=True, framealpha=0.8, edgecolor='#cccccc')
                
                plt.savefig(diagnostic_dir / f"{station_name}_temperature.png",
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
    
    # Reset font sizes to default at the end
    plt.rcParams.update(plt.rcParamsDefault) 

def create_interactive_temperature_plot(preprocessed_data: dict, output_dir: Path):
    """Create interactive plotly plot showing VST_RAW data with temperature overlay."""
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Define consistent colors matching matplotlib style
    colors = {
        'temperature': '#d62728',  # Red
        'water_level': '#1f77b4',  # Blue
        'freezing': '#2ca02c',     # Green
        'background': '#f8f9fa',   # Light gray
        'grid': '#cccccc'          # Light gray for grid
    }
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None and station_data['temperature'] is not None:
            # Prepare data
            vst_data = station_data['vst_raw'].copy()
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
                
            temp_data = station_data['temperature'].copy()
            if not isinstance(temp_data.index, pd.DatetimeIndex):
                temp_data.set_index('Date', inplace=True)
            
            # Create figure with two subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=(
                    "<b>Temperature Data</b> (Red background indicates freezing temperatures)",
                    "<b>Water Level Data</b> (Red background indicates freezing temperatures)"
                ),
                vertical_spacing=0.15,
                row_heights=[0.4, 0.6],
                shared_xaxes=True
            )
            
            # Add temperature data
            fig.add_trace(
                go.Scatter(
                    x=temp_data.index,
                    y=temp_data['temperature'],
                    name="Temperature",
                    line=dict(color=colors['temperature'], width=1.5),
                    hovertemplate="Date: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add freezing line
            fig.add_trace(
                go.Scatter(
                    x=temp_data.index,
                    y=[0] * len(temp_data),
                    name="Freezing Point",
                    line=dict(color=colors['freezing'], width=1.5, dash='dash'),
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Find freezing periods
            temp_series = temp_data['temperature']
            freezing = temp_series < 0
            state_changes = freezing.ne(freezing.shift()).fillna(True)
            change_points = temp_series.index[state_changes]
            
            if freezing.iloc[-1]:
                change_points = change_points.append(temp_series.index[-1:])
            
            # Add freezing period rectangles
            for i in range(0, len(change_points)-1, 2):
                if i+1 < len(change_points) and freezing.loc[change_points[i]]:
                    for row in [1, 2]:
                        fig.add_vrect(
                            x0=change_points[i],
                            x1=change_points[i+1],
                            fillcolor="rgba(255, 0, 0, 0.1)",
                            layer="below",
                            line_width=0,
                            row=row, col=1
                        )
            
            # Add VST_RAW data
            fig.add_trace(
                go.Scatter(
                    x=vst_data.index,
                    y=vst_data['vst_raw'],
                    name="Water Level",
                    line=dict(color=colors['water_level'], width=1.5),
                    hovertemplate="Date: %{x}<br>Water Level: %{y:.1f}mm<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Add statistics as annotations
            temp_stats = (
                f"Temperature Range: {temp_data['temperature'].min():.1f}°C to "
                f"{temp_data['temperature'].max():.1f}°C<br>"
                f"Average: {temp_data['temperature'].mean():.1f}°C<br>"
                f"Days Below 0°C: {sum(freezing)}"
            )
            
            water_stats = (
                f"Water Level Range: {vst_data['vst_raw'].min():.1f}mm to "
                f"{vst_data['vst_raw'].max():.1f}mm<br>"
                f"Average: {vst_data['vst_raw'].mean():.1f}mm"
            )
            
            # Update layout with professional styling
            fig.update_layout(
                height=800,
                width=1200,
                title=dict(
                    text=f"<b>{station_name}</b> - Water Level and Temperature Analysis",
                    x=0.5,
                    y=0.98,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20)
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=colors['grid']
                ),
                plot_bgcolor=colors['background'],
                paper_bgcolor='white',
                hovermode='x unified',
                font=dict(size=12)
            )
            
            # Update axes styling
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=colors['grid'],
                zeroline=False,
                title_text="Date",
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=colors['grid'],
                zeroline=False,
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )
            
            # Update y-axis titles
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Water Level (mm)", row=2, col=1)
            
            # Add statistics annotations
            fig.add_annotation(
                text=temp_stats,
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=colors['grid'],
                borderwidth=1,
                align="left"
            )
            
            fig.add_annotation(
                text=water_stats,
                xref="paper", yref="paper",
                x=0.01, y=0.45,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=colors['grid'],
                borderwidth=1,
                align="left"
            )
            
            # Save interactive plot
            fig.write_html(diagnostic_dir / f"{station_name}_temperature_analysis.html") 