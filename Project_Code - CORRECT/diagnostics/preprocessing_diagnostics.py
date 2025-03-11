"""Diagnostic tools for data preprocessing step."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

def set_plot_style():
    """Set a consistent, professional plot style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set the layout engine explicitly to avoid warnings
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = False
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Set colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # Set grid style
    plt.rcParams['grid.alpha'] = 0.2  # Reduced from 0.3 for subtler grid
    plt.rcParams['grid.linestyle'] = '--'
    
    # Set figure background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'  # Changed from '#f8f9fa' to pure white
    
    # Set spine colors
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 1.0

def plot_preprocessing_comparison(original_data: dict, preprocessed_data: dict, output_dir: Path, periods_to_remove: dict = None):
    """
    Create comparison plots between original and preprocessed data for each station.
    
    Args:
        original_data: Dictionary containing original station data
        preprocessed_data: Dictionary containing preprocessed station data
        output_dir: Output directory path
        periods_to_remove: Dictionary with station names as keys and lists of (start, end) tuples for freezing periods
    """
    # Set professional plot style with larger fonts
    set_plot_style()
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame to store statistics for all stations
    stats_data = []
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with GridSpec for better layout control - now with just 2 subplots
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1], hspace=0.3)
            
            orig = original_data[station_name]['vst_raw']
            proc = preprocessed_data[station_name]['vst_raw']
            
            # Calculate IQR bounds
            Q1 = orig['vst_raw'].quantile(0.25)
            Q3 = orig['vst_raw'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Calculate removed points mask here, before using it
            removed_mask = (orig['Value'] < lower_bound) | (orig['Value'] > upper_bound)
            
            # Original data with bounds
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(orig['Date'], orig['Value'], color='#1f77b4', alpha=0.7, 
                    linewidth=1, label='Original')
            ax1.axhline(y=lower_bound, color='#d62728', linestyle='--', alpha=0.7,
                       label='Lower bound (Q1 - 1.5*IQR)')
            ax1.axhline(y=upper_bound, color='#d62728', linestyle='--', alpha=0.7,
                       label='Upper bound (Q3 + 4*IQR)')
            
            # Plot processed data with improved styling
            ax1.plot(proc['Date'], proc['Value'], color='#2ca02c', alpha=0.8, 
                   linewidth=1.2, label='Preprocessed Data', zorder=4)
            
            # Highlight removed points with improved styling
            removed_points = orig[outlier_mask & ~freezing_mask]  # Outliers only
            if len(removed_points) > 0:
                ax1.scatter(removed_points['Date'], removed_points['Value'], 
                           color='#ff7f0e', s=25, alpha=0.6, label='Outliers', zorder=5)
            
            frozen_points = orig[freezing_mask]  # All freezing points
            if not frozen_points.empty:
                ax1.scatter(frozen_points['Date'], frozen_points['Value'],
                          color='#1f77b4', s=25, alpha=0.4, label='Freezing Period', zorder=5)
            
            # Preprocessed data - second subplot with improved styling
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(proc['Date'], proc['Value'], color='#2ca02c', alpha=0.7, 
                    linewidth=1, label='Preprocessed')
            
            # Removed points highlighted
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(orig['Date'], orig['Value'], color='#1f77b4', alpha=0.3, 
                    linewidth=1, label='Original')
            removed_points = orig[removed_mask]
            ax3.scatter(removed_points['Date'], removed_points['Value'], 
                       color='#d62728', s=50, alpha=0.5, label='Removed Points')
            
            # Set titles with adjusted padding
            #ax1.set_title(f'Original Data with IQR Bounds - {station_name}', 
             #            pad=10, y=1.0)
            #ax2.set_title('Preprocessed Data', pad=10, y=1.0)
            #ax3.set_title('Removed Data Points', pad=10, y=1.0)
            
            # Set consistent axis labels with proper padding
            ax1.set_ylabel('Water Level (mm)', fontsize=12, labelpad=10)
            ax2.set_ylabel('Water Level (mm)', fontsize=12, labelpad=10)
            ax2.set_xlabel('Date', fontsize=12, labelpad=10)
            
            # Handle potentially duplicate labels in the first subplot
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), 
                      loc='lower right', frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', fontsize=10)
            
            # Add legend for second subplot
            ax2.legend(loc='lower right', frameon=True, framealpha=0.9, 
                      edgecolor='#cccccc', fontsize=10)
            
            # Set a professional background style for all subplots
            for ax in [ax1, ax2]:
                ax.set_facecolor('#ffffff')  # White background
                ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=10, pad=8)
                ax.tick_params(axis='x', rotation=45)
            
            # Save the figure with high resolution
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", 
                       dpi=300,
                       bbox_inches='tight',
                       facecolor='white')
            plt.close()
    
    # Create and save statistics table
    stats_df = pd.DataFrame(stats_data)
    
    # Format the statistics table
    stats_df = stats_df.round(2)
    stats_df['Removal %'] = stats_df['Removal %'].round(1).astype(str) + '%'
    
    # Save statistics to CSV
    stats_df.to_csv(diagnostic_dir / "preprocessing_statistics.csv", index=False)
    
    # Create LaTeX table
    with open(diagnostic_dir / "preprocessing_statistics.tex", "w") as f:
        # Write LaTeX table header
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Preprocessing Statistics Summary for Stream Water Level Data}\n")
        f.write("\\label{tab:preprocessing-stats}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Station & Total Points & Points Removed & Removal (\\%) & Outliers & Freezing & Flatlines \\\\\n")
        f.write("\\midrule\n")
        
        # Write data rows
        for _, row in stats_df.iterrows():
            station = row['Station'].replace("_", "\\_")  # Escape underscores for LaTeX
            f.write(f"{station} & ")
            f.write(f"{row['Total Points']:,.0f} & ")
            f.write(f"{row['Points Removed']:,.0f} & ")
            f.write(f"{float(row['Removal %'].strip('%')):.1f} & ")
            f.write(f"{row['Outliers']:,.0f} & ")
            f.write(f"{row['Freezing']:,.0f} & ")
            f.write(f"{row['Flatlines']:,.0f} \\\\\n")
        
        # Write table footer
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write("\\item Note: Points Removed represents the total number of data points excluded during preprocessing. ")
        f.write("Outliers are identified using IQR method (1.5×IQR below Q1 or 4×IQR above Q3). ")
        f.write("Freezing points are removed during periods of sub-zero temperatures. ")
        f.write("Flatlines represent sequences of 20 or more identical consecutive values.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    
    # Create a more readable text version
    with open(diagnostic_dir / "preprocessing_statistics.txt", "w") as f:
        f.write("Preprocessing Statistics Summary\n")
        f.write("==============================\n\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"Station: {row['Station']}\n")
            f.write("-" * (len(row['Station']) + 9) + "\n")
            f.write(f"Total Points: {row['Total Points']:,}\n")
            f.write(f"Points Removed: {row['Points Removed']:,} ({row['Removal %']})\n")
            f.write("Breakdown:\n")
            f.write(f"  • Outliers: {row['Outliers']:,}\n")
            f.write(f"  • Freezing: {row['Freezing']:,}\n")
            f.write(f"  • Flatlines: {row['Flatlines']:,}\n")
            f.write("\nIQR Statistics:\n")
            f.write(f"  • Q1: {row['Q1']:.1f}\n")
            f.write(f"  • Q3: {row['Q3']:.1f}\n")
            f.write(f"  • IQR: {row['IQR']:.1f}\n")
            f.write("\n" + "="*30 + "\n\n")

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
        if station_name in original_data and original_data[station_name]['vst_raw'] is not None:
            # Create figure with GridSpec for better control - increased figure height
            fig = plt.figure(figsize=(15, 12))
            # Adjusted height ratios to make plots bigger
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.4)
            
            # 1. Water level measurements (VST_RAW only)
            ax1 = fig.add_subplot(gs[0])
            data = preprocessed_data[station_name]
            
            # Plot VST Raw with custom styling
            ax1.plot(data['vst_raw'].index, data['vst_raw']['Value'],
                    color='#1f77b4', alpha=0.7, linewidth=1, label='VST Raw')
            
            if data['vst_edt'] is not None:
                ax1.plot(data['vst_edt'].index, data['vst_edt']['Value'],
                        color='#2ca02c', alpha=0.7, linewidth=1, label='VST EDT')
            
            if data['vinge'] is not None:
                ax1.scatter(data['vinge'].index, data['vinge']['W.L [cm]'],
                          color='#d62728', alpha=0.7, s=50, label='VINGE')
            
            # Add statistics box for water levels
            stats_text = (f'VST Raw Points: {len(data["vst_raw"])}\n'
                        f'Range: {data["vst_raw"]["Value"].min():.1f} to '
                        f'{data["vst_raw"]["Value"].max():.1f}')
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
                ax2.plot(rain_data.index, rain_data['precipitation (mm)'],
                        color='#1f77b4', alpha=0.7, linewidth=1, label='Rainfall')
                
                # Add rainfall statistics
                total_rain = rain_data['precipitation (mm)'].sum()
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
                ax3.plot(temp_data.index, temp_data['temperature (C)'],
                        color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
                
                # Add temperature statistics
                temp_stats = (f'Range: {temp_data["temperature (C)"].min():.1f}°C to '
                            f'{temp_data["temperature (C)"].max():.1f}°C')
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
            ylabels = ['Water Level (mm)', 'Precipitation (mm)',
                      'Temperature (°C)', 'Water Level (mm)']
            
            for ax, title, ylabel in zip(axes, titles, ylabels):
                ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, labelpad=15)  # Increased y-axis label padding
                ax.legend(frameon=True, framealpha=0.8, edgecolor='#cccccc')
                ax.set_facecolor('#f8f9fa')
                
                # Style the grid
                ax.grid(True, alpha=0.3, color='#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
                
                # Add more padding to y-axis
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            # Save the figure
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

def plot_additional_data(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Create thesis-quality visualizations for VINGE, rainfall and temperature data."""
    # Call the three separate visualization functions
    plot_vst_vinge_comparison(preprocessed_data, output_dir, original_data)
    plot_climate_water_level(preprocessed_data, output_dir)
    plot_seasonal_analysis(preprocessed_data, output_dir)

def plot_vst_vinge_comparison(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Create visualization comparing VST_RAW, VST_EDT, and VINGE measurements."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if (station_name in original_data and
            original_data[station_name]['vst_raw'] is not None and 
            station_data['vst_edt'] is not None and 
            station_data['vinge'] is not None):
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.3)
            
            # Main plot with VST_RAW, VST_EDT, and VINGE
            ax1 = fig.add_subplot(gs[0])
            
            # Plot VST data with custom styling
            ax.plot(vst_data.index, vst_data['Value'],
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
            
            # --- Monthly temperature subplot ---
            ax2 = fig.add_subplot(gs[1])
            # Calculate monthly average temperatures
            monthly_temp = temp_data.resample('M').mean()
            
            # Create a bar plot of monthly temperatures
            months = monthly_temp.index
            ax2.bar(months, monthly_temp['temperature (C)'], 
                   color='#d62728', alpha=0.7, width=25)
            
            # Add freezing line
            ax2.axhline(y=0, color='#2ca02c', linestyle='--', linewidth=1.5, 
                      alpha=0.8, label='Freezing Point (0°C)')
            
            # --- Rainfall subplot ---
            ax3 = fig.add_subplot(gs[2])
            rain_data = station_data['rainfall'].copy()
            if not isinstance(rain_data.index, pd.DatetimeIndex):
                rain_data.set_index('Date', inplace=True)
            
            # Rainfall data plot
            if station_data['rainfall'] is not None:
                fig = plt.figure(figsize=(15, 8))
                gs = GridSpec(1, 1)
                ax = fig.add_subplot(gs[0])
                
                rain_data = station_data['rainfall']
                ax.plot(rain_data.index, rain_data['precipitation (mm)'],
                       color='#1f77b4', alpha=0.7, linewidth=1, label='Rainfall')
                
                # Add rainfall statistics
                total_rain = rain_data['precipitation (mm)'].sum()
                rain_stats = (
                    f'Total Rainfall: {total_rain:.1f}mm\n'
                    f'Average: {rain_data["precipitation (mm)"].mean():.2f}mm/day\n'
                    f'Max: {rain_data["precipitation (mm)"].max():.1f}mm'
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
                ax.set_ylabel('Precipitation (mm)', labelpad=10)
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
                ax.plot(temp_data.index, temp_data['temperature (C)'],
                       color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
                
                # Add freezing line
                ax.axhline(y=0, color='#2ca02c', linestyle='--', alpha=0.7,
                          label='Freezing Point')
                
                # Add temperature statistics
                temp_stats = (
                    f'Range: {temp_data["temperature (C)"].min():.1f}°C to '
                    f'{temp_data["temperature (C)"].max():.1f}°C\n'
                    f'Average: {temp_data["temperature (C)"].mean():.1f}°C\n'
                    f'Days Below 0°C: {sum(temp_data["temperature (C)"] < 0)}'
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
                ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
            
            # Add main title for the entire figure
            fig.suptitle(f'{station_name} - Climate and Water Level Analysis',
                       fontsize=18, fontweight='bold', y=0.98)
            
            plt.savefig(diagnostic_dir / f"{station_name}_climate_water_level.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()


def plot_seasonal_analysis(preprocessed_data: dict, output_dir: Path):
    """Create thesis-quality visualization for seasonal water level patterns."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None:
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            vst_data = station_data['vst_raw'].copy()
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            # 3.1 Monthly boxplot analysis (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Add month column for grouping
            vst_data_copy = vst_data.copy()
            vst_data_copy['month'] = vst_data_copy.index.month
            
            # Create boxplot by month
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            monthly_data = [vst_data_copy[vst_data_copy['month'] == month]['Value'] 
                          for month in range(1, 13)]
            
            ax1.boxplot(monthly_data, labels=month_names)
            ax1.set_title('Monthly Water Level Distribution', 
                         fontsize=16, fontweight='bold', pad=15)
            ax1.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
            ax1.set_xlabel('Month', fontsize=14, labelpad=10)
            
            # 3.2 Yearly boxplot analysis (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Add year column for grouping
            vst_data_copy['year'] = vst_data_copy.index.year
            
            # Get unique years and create boxplot
            years = sorted(vst_data_copy['year'].unique())
            yearly_data = [vst_data_copy[vst_data_copy['year'] == year]['Value'] 
                         for year in years]
            
            ax2.boxplot(yearly_data, labels=years)
            ax2.set_title('Yearly Water Level Distribution', 
                         fontsize=16, fontweight='bold', pad=15)
            ax2.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
            ax2.set_xlabel('Year', fontsize=14, labelpad=10)
            ax2.tick_params(axis='x', rotation=45)
            
            # 3.3 Calculate and plot monthly averages (bottom left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Group by month and calculate mean
            monthly_means = vst_data_copy.groupby('month')['Value'].mean()
            monthly_std = vst_data_copy.groupby('month')['Value'].std()
            
            ax3.bar(month_names, monthly_means, 
                   yerr=monthly_std, alpha=0.7, 
                   capsize=5, color='#1f77b4')
            
            ax3.set_title('Monthly Average Water Levels', 
                         fontsize=16, fontweight='bold', pad=15)
            ax3.set_ylabel('Average Water Level (mm)', fontsize=14, labelpad=10)
            ax3.set_xlabel('Month', fontsize=14, labelpad=10)
            
            # 3.4 Multi-year overlay (bottom right)
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Create a relative time axis (day of year) and plot each year separately
            colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
            
            for i, year in enumerate(years):
                year_data = vst_data_copy[vst_data_copy['year'] == year]
                year_data['day_of_year'] = year_data.index.dayofyear
                
                # To handle leap years uniformly, restrict to first 365 days
                year_data = year_data[year_data['day_of_year'] <= 365]
                
                # Create 15-day rolling average for smoother visualization
                if len(year_data) > 30:  # Only apply smoothing if enough data
                    rolling_avg = year_data['Value'].rolling(window=15, center=True).mean()
                    ax4.plot(year_data['day_of_year'], rolling_avg, 
                           color=colors[i], label=str(year), linewidth=1.5)
            
            ax4.set_title('Yearly Patterns Overlay (15-day rolling avg)', 
                         fontsize=16, fontweight='bold', pad=15)
            ax4.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
            ax4.set_xlabel('Day of Year', fontsize=14, labelpad=10)
            
            # Add month names on x-axis for better reference
            month_positions = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
            ax4.set_xticks(month_positions)
            ax4.set_xticklabels(month_names)
            
            # Add legend with year labels
            ax4.legend(ncol=min(3, len(years)), loc='best', 
                      frameon=True, framealpha=0.9, fontsize=10)
            
            # Set a professional background style for all subplots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
            
            # Add main title for the entire figure
            fig.suptitle(f'{station_name} - Seasonal Water Level Patterns',
                       fontsize=18, fontweight='bold', y=0.98)
            
            plt.savefig(diagnostic_dir / f"{station_name}_seasonal_analysis.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # Reset font sizes to default at the end
    plt.rcParams.update(plt.rcParamsDefault)

def create_spectral_analysis(preprocessed_data: dict, output_dir: Path):
    """
    Create spectral analysis plots to identify cyclic patterns in stream water levels.
    
    This visualization helps identify dominant cyclical patterns (daily, weekly, seasonal, annual)
    in surface water level data using FFT (Fast Fourier Transform).
    
    Args:
        preprocessed_data: Dictionary containing preprocessed station data
        output_dir: Output directory path
    """
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None:
            try:
                # Prepare data
                vst_data = station_data['vst_raw'].copy()
                if not isinstance(vst_data.index, pd.DatetimeIndex):
                    vst_data.set_index('Date', inplace=True)
                
                # Resample to regular daily intervals
                daily_data = vst_data['Value'].resample('D').mean()
                
                # Fill gaps with interpolation (required for FFT)
                daily_data = daily_data.interpolate(method='linear', limit=7)
                
                # Drop NaN values if any remain
                daily_data = daily_data.dropna()
                
                # Skip if insufficient data
                if len(daily_data) < 365:  # Need at least a year of data
                    print(f"Skipping spectral analysis for {station_name}: insufficient data")
                    continue
                
                # Create multi-panel figure
                fig = plt.figure(figsize=(15, 14))
                gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.5, 1], hspace=0.3)
                
                # 1. Original time series
                ax1 = fig.add_subplot(gs[0])
                ax1.plot(daily_data.index, daily_data.values, color='#1f77b4', linewidth=1.5, alpha=0.8)
                ax1.set_title('Stream Water Level Time Series', fontsize=16, fontweight='bold', pad=15)
                ax1.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
                
                # Mark the years on the x-axis
                years = pd.date_range(start=daily_data.index.min(), end=daily_data.index.max(), freq='YS')
                ax1.set_xticks(years)
                ax1.set_xticklabels([d.strftime('%Y') for d in years], rotation=45)
                
                # 2. FFT Periodogram
                ax2 = fig.add_subplot(gs[1])
                
                # Calculate FFT
                values = daily_data.values
                N = len(values)
                # Remove mean (detrend)
                values = values - np.mean(values)
                
                # Get sample spacing (in days)
                sample_spacing = (daily_data.index[-1] - daily_data.index[0]).total_seconds() / (N-1) / 86400
                
                # Calculate FFT
                fft_values = np.fft.rfft(values)
                fft_freqs = np.fft.rfftfreq(N, sample_spacing)
                
                # Convert to periods in days
                periods = 1 / fft_freqs[1:]  # Skip the first value (DC component)
                amplitudes = np.abs(fft_values)[1:]  # Skip the first value
                
                # Plot power spectrum
                ax2.plot(periods, amplitudes, color='#2ca02c', linewidth=1.5, alpha=0.8)
                ax2.set_xlabel('Period (days)', fontsize=14, labelpad=10)
                ax2.set_ylabel('Amplitude', fontsize=14, labelpad=10)
                ax2.set_title('Spectral Power of Stream Water Level Fluctuations', 
                           fontsize=16, fontweight='bold', pad=15)
                
                # Set log scale for better visualization
                ax2.set_xscale('log')
                ax2.set_xlim(1, 1000)  # Display periods from 1 day to ~3 years
                
                # Add vertical lines for key periods
                key_periods = {
                    7: 'Weekly',
                    30.44: 'Monthly',
                    91.31: 'Seasonal (3 months)',
                    182.62: 'Semi-annual',
                    365.25: 'Annual'
                }
                
                for period, label in key_periods.items():
                    ax2.axvline(x=period, color='#d62728', alpha=0.5, 
                              linestyle='--', linewidth=1)
                    ax2.text(period, ax2.get_ylim()[1] * 0.9, label, 
                           rotation=90, verticalalignment='top', 
                           fontsize=10, color='#d62728')
                
                # 3. Dominant cycles - find and highlight the 5 strongest periods
                ax3 = fig.add_subplot(gs[2])
                
                # Find peaks in the FFT spectrum
                from scipy.signal import find_peaks
                # Only look at periods between 2 days and 500 days
                valid_idx = (periods >= 2) & (periods <= 500)
                
                if sum(valid_idx) > 0:
                    peaks, _ = find_peaks(amplitudes[valid_idx], distance=5)
                    
                    if len(peaks) > 0:
                        # Get the indices of the top 5 peaks
                        peak_amplitudes = amplitudes[valid_idx][peaks]
                        sorted_peaks = np.argsort(peak_amplitudes)[::-1]  # Descending order
                        top_peaks = sorted_peaks[:min(5, len(sorted_peaks))]
                        top_periods = periods[valid_idx][peaks][top_peaks]
                        top_amplitudes = amplitudes[valid_idx][peaks][top_peaks]
                        
                        # Create bar plot for dominant cycles
                        bars = ax3.bar(range(len(top_periods)), top_amplitudes, 
                                    color='#ff7f0e', alpha=0.7, width=0.6)
                        
                        # Label the bars with the period values
                        for i, (period, amp) in enumerate(zip(top_periods, top_amplitudes)):
                            # Convert to more intuitive units
                            if period >= 365:
                                label = f"{period/365.25:.1f} years"
                            elif period >= 30:
                                label = f"{period/30.44:.1f} months"
                            else:
                                label = f"{period:.1f} days"
                                
                            ax3.text(i, amp/2, label, ha='center', va='center', 
                                  rotation=0, color='black', fontsize=10, fontweight='bold')
                
                ax3.set_title('Dominant Cycles in Stream Water Level', 
                           fontsize=16, fontweight='bold', pad=15)
                ax3.set_ylabel('Amplitude', fontsize=14, labelpad=10)
                ax3.set_xticks([])  # Hide x-axis ticks
                ax3.set_xlim(-0.5, len(top_periods) - 0.5)
                
                # Add a key statistics box
                mean_level = daily_data.mean()
                std_level = daily_data.std()
                min_level = daily_data.min()
                max_level = daily_data.max()
                data_range = (daily_data.index.max() - daily_data.index.min()).days
                
                stats_text = (
                    f"Data Statistics:\n"
                    f"• Time span: {data_range} days ({data_range/365.25:.1f} years)\n"
                    f"• Mean water level: {mean_level:.1f} mm\n"
                    f"• Standard deviation: {std_level:.1f} mm\n"
                    f"• Range: {min_level:.1f} to {max_level:.1f} mm"
                )
                
                ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes,
                       verticalalignment='bottom', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                               alpha=0.9, edgecolor='#cccccc'),
                       fontsize=11)
                
                # Set a professional background style for all subplots
                for ax in [ax1, ax2, ax3]:
                    ax.set_facecolor('#f8f9fa')
                    ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                    for spine in ax.spines.values():
                        spine.set_color('#cccccc')
                    ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
                
                # Add main title for the entire figure
                fig.suptitle(f'{station_name} - Spectral Analysis of Stream Water Level Fluctuations',
                           fontsize=18, fontweight='bold', y=0.98)
                
                # Don't use layout adjustments - rely on bbox_inches='tight' when saving
                # plt.subplots_adjust(top=0.97, bottom=0.06, left=0.08, right=0.92, hspace=0.3)
                
                # Save the figure
                plt.savefig(diagnostic_dir / f"{station_name}_spectral_analysis.png",
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
            except Exception as e:
                print(f"Error creating spectral analysis for {station_name}: {str(e)}")
                import traceback
                traceback.print_exc()

def create_change_point_detection(preprocessed_data: dict, output_dir: Path):
    """
    Create visualizations that detect and highlight significant shifts in stream water level patterns.
    
    This function uses statistical methods to identify periods where fundamental changes 
    occurred in the water level data, which could indicate natural regime shifts,
    anthropogenic influences, or data collection changes.
    
    Args:
        preprocessed_data: Dictionary containing preprocessed station data
        output_dir: Output directory path
    """
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import ruptures as rpt
    except ImportError:
        print("Warning: ruptures package not installed. To use change point detection, install with:")
        print("pip install ruptures")
        return
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None:
            try:
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
                    y=temp_data['temperature (C)'],
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
            temp_series = temp_data['temperature (C)']
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
                    y=vst_data['Value'],
                    name="Water Level",
                    line=dict(color=colors['water_level'], width=1.5),
                    hovertemplate="Date: %{x}<br>Water Level: %{y:.1f}mm<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Add statistics as annotations
            temp_stats = (
                f"Temperature Range: {temp_data['temperature (C)'].min():.1f}°C to "
                f"{temp_data['temperature (C)'].max():.1f}°C<br>"
                f"Average: {temp_data['temperature (C)'].mean():.1f}°C<br>"
                f"Days Below 0°C: {sum(freezing)}"
            )
            
            water_stats = (
                f"Water Level Range: {vst_data['Value'].min():.1f}mm to "
                f"{vst_data['Value'].max():.1f}mm<br>"
                f"Average: {vst_data['Value'].mean():.1f}mm"
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