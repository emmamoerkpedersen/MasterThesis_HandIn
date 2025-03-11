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

def plot_preprocessing_comparison(original_data: dict, preprocessed_data: dict, output_dir: Path, frost_periods: list = None):
    """Create comparison plots between original and preprocessed data for each station."""
    # Set professional plot style
    set_plot_style()
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame to store statistics for all stations
    stats_data = []
    
    # Set start date to January 1, 2010
    start_date = pd.to_datetime('2010-02-01')
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with GridSpec
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
            
            # Get the original and processed data
            orig = original_data[station_name]['vst_raw'].copy()  # Make a copy to avoid modifying original
            proc = preprocessed_data[station_name]['vst_raw'].copy()
            
            # Get the value column name from the original data
            orig_value_col = [col for col in orig.columns if col != 'Date'][0]
            
            # Ensure both DataFrames have datetime index
            if not isinstance(orig.index, pd.DatetimeIndex):
                orig.index = pd.to_datetime(orig.index)
            if not isinstance(proc.index, pd.DatetimeIndex):
                proc.index = pd.to_datetime(proc.index)
            
            # Filter data to start from 2010
            orig = orig[orig.index >= start_date]
            
            # Calculate IQR bounds using the original data
            Q1 = orig[orig_value_col].quantile(0.25)
            Q3 = orig[orig_value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Calculate removed points mask for outliers (only points outside IQR bounds)
            outlier_mask = (orig[orig_value_col] < lower_bound) | (orig[orig_value_col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            # Calculate freezing period points if available
            freezing_points = 0
            freezing_mask = pd.Series(False, index=orig.index)
            
            if frost_periods:
                for start, end in frost_periods:
                    period_mask = (orig.index >= start) & (orig.index <= end)
                    freezing_mask = freezing_mask | period_mask
                freezing_points = freezing_mask.sum()
            
            # Calculate total points removed (actual difference between orig and proc)
            total_points_removed = len(orig) - len(proc)
            
            # Calculate flatline points (not accounted for by outliers or freezing)
            flatline_count = total_points_removed - outlier_count - freezing_points
            
            # Store statistics for this station
            stats_data.append({
                'Station': station_name,
                'Total Points': len(orig),
                'Points Removed': total_points_removed,
                'Removal %': (total_points_removed/len(orig))*100,
                'Outliers': outlier_count,
                'Freezing': freezing_points,
                'Flatlines': flatline_count,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            })
            
            # Top subplot: Original data with IQR bounds and removed points
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(orig.index, orig[orig_value_col], color='#1f77b4', alpha=0.7, 
                    linewidth=1.0, label='Original Data', zorder=2)
            
            # Add IQR bounds with improved styling
            ax1.axhline(y=lower_bound, color='#ff7f0e', linestyle='--', alpha=0.6,
                       linewidth=1.0, label='IQR Bounds', zorder=3)
            ax1.axhline(y=upper_bound, color='#ff7f0e', linestyle='--', alpha=0.6,
                       linewidth=1.0, zorder=3)
            
            # Add frost periods if available
            if frost_periods:
                for start, end in frost_periods:
                    if start >= start_date:  # Only show frost periods after 2010
                        ax1.axvspan(start, end, color='#E3F2FD', alpha=0.5, 
                                  label='Frost Period' if start == frost_periods[0][0] else "", zorder=1)
            
            # Highlight only points outside IQR bounds
            outlier_points = orig[outlier_mask]
            if len(outlier_points) > 0:
                ax1.scatter(outlier_points.index, outlier_points[orig_value_col],
                          color='#ff7f0e', s=25, alpha=0.6, label='Removed Points', zorder=5)
            
            ax1.set_title('Original Data with IQR Bounds', fontsize=14, fontweight='bold', pad=15)
            ax1.set_ylabel('Water Level (mm)', fontsize=12, labelpad=10)
            ax1.legend(loc='best', frameon=True, framealpha=0.9)
            
            # Bottom subplot: Preprocessed data only
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(proc.index, proc['vst_raw'], color='#2ca02c', alpha=0.8, 
                    linewidth=1.2, label='Preprocessed Data', zorder=2)
            
            ax2.set_title('Preprocessed Data', fontsize=14, fontweight='bold', pad=15)
            ax2.set_ylabel('Water Level (mm)', fontsize=12, labelpad=10)
            ax2.set_xlabel('Date', fontsize=12, labelpad=10)
            ax2.legend(loc='best', frameon=True, framealpha=0.9)
            
            # Add main title
            fig.suptitle(f'Data Comparison - Station {station_name}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save the figure
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    # Create and save statistics table
    stats_df = pd.DataFrame(stats_data)
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
            f.write(f"{row['Removal %']:.1f} & ")  # Already a float, no need to strip
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
            f.write(f"Points Removed: {row['Points Removed']:,} ({row['Removal %']:.1f}%)\n")
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
                f.write(f"  - Total rainfall: {rain['precipitation'].sum():.1f} mm\n")
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
    
    # Set start date to February 1, 2010
    start_date = pd.to_datetime('2010-02-01')
    
    for station_name in preprocessed_data.keys():
        if station_name in original_data and original_data[station_name]['vst_raw'] is not None:
            # Create figure with GridSpec for better control
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.4)
            
            # Get data for this station
            orig_data = original_data[station_name]
            proc_data = preprocessed_data[station_name]
            
            # 1. Water level measurements (VST_RAW)
            ax1 = fig.add_subplot(gs[0])
            
            if orig_data['vst_raw'] is not None:
                vst_data = orig_data['vst_raw'].copy()
                if not isinstance(vst_data.index, pd.DatetimeIndex):
                    vst_data.set_index('Date', inplace=True)
                
                # Get the VST column name
                vst_col = [col for col in vst_data.columns if col != 'Date'][0]
                
                # Filter data from 2010
                vst_data = vst_data[vst_data.index >= start_date]
                
                ax1.plot(vst_data.index, vst_data[vst_col],
                        color='#1f77b4', alpha=0.7, linewidth=1, label='VST Raw')
                
                # Add statistics text box
                stats_text = (
                    f"Statistics:\n"
                    f"Mean: {vst_data[vst_col].mean():.1f} mm\n"
                    f"Std: {vst_data[vst_col].std():.1f} mm\n"
                    f"Range: [{vst_data[vst_col].min():.1f}, {vst_data[vst_col].max():.1f}] mm"
                )
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white', alpha=0.9, edgecolor='#cccccc'))
            
            # 2. Rainfall data
            ax2 = fig.add_subplot(gs[1])
            if proc_data['rainfall'] is not None:
                rain_data = proc_data['rainfall'].copy()
                if not isinstance(rain_data.index, pd.DatetimeIndex):
                    rain_data.set_index('Date', inplace=True)
                
                # Get rainfall column name
                rain_col = [col for col in rain_data.columns if col != 'Date'][0]
                
                # Filter data from 2010
                rain_data = rain_data[rain_data.index >= start_date]
                
                # Calculate daily sum for better visualization
                daily_rain = rain_data[rain_col].resample('D').sum()
                
                ax2.bar(daily_rain.index, daily_rain.values,
                       color='#1f77b4', alpha=0.7, width=1, label='Daily Rainfall')
                
                # Add statistics text box
                stats_text = (
                    f"Statistics:\n"
                    f"Total: {daily_rain.sum():.1f} mm\n"
                    f"Daily Max: {daily_rain.max():.1f} mm\n"
                    f"Rainy Days: {(daily_rain > 0).sum()}"
                )
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white', alpha=0.9, edgecolor='#cccccc'))
            
            # 3. Temperature data
            ax3 = fig.add_subplot(gs[2])
            if proc_data['temperature'] is not None:
                temp_data = proc_data['temperature'].copy()
                if not isinstance(temp_data.index, pd.DatetimeIndex):
                    temp_data.set_index('Date', inplace=True)
                
                # Get temperature column name
                temp_col = [col for col in temp_data.columns if col != 'Date'][0]
                
                # Filter data from 2010
                temp_data = temp_data[temp_data.index >= start_date]
                
                ax3.plot(temp_data.index, temp_data[temp_col],
                        color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
                
                # Add freezing line
                ax3.axhline(y=0, color='#2ca02c', linestyle='--', alpha=0.8,
                           label='Freezing Point')
                
                # Add statistics text box
                stats_text = (
                    f"Statistics:\n"
                    f"Mean: {temp_data[temp_col].mean():.1f}°C\n"
                    f"Range: [{temp_data[temp_col].min():.1f}, {temp_data[temp_col].max():.1f}]°C\n"
                    f"Days Below 0°C: {(temp_data[temp_col] < 0).sum()}"
                )
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white', alpha=0.9, edgecolor='#cccccc'))
            
            # Set titles and labels with consistent styling
            axes = [ax1, ax2, ax3]
            titles = ['Water Level Measurements', 'Daily Rainfall', 'Temperature']
            ylabels = ['Water Level (mm)', 'Precipitation (mm)', 'Temperature (°C)']
            
            # Format x-axis to show years properly
            for ax in axes:
                # Set date formatter for x-axis
                years_fmt = mdates.DateFormatter('%Y')
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
                ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for months
                
                # Style the grid
                ax.grid(True, which='major', alpha=0.3, color='#cccccc')
                ax.grid(True, which='minor', alpha=0.1, color='#cccccc')
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
            
            for ax, title, ylabel in zip(axes, titles, ylabels):
                ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, labelpad=15)
                ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
                ax.set_facecolor('#f8f9fa')
                
                # Add more padding to y-axis
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            # Add main title
            fig.suptitle(f'Station {station_name} - Data Overview',
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save the figure
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

def plot_additional_data(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Create thesis-quality visualizations for VINGE, rainfall and temperature data."""
    # Only call functions that don't require original_data if it's not provided
    if original_data is None:
        plot_climate_water_level(preprocessed_data, output_dir)
        plot_seasonal_analysis(preprocessed_data, output_dir)
    else:
        plot_vst_vinge_comparison(preprocessed_data, output_dir, original_data)
        plot_climate_water_level(preprocessed_data, output_dir)
        plot_seasonal_analysis(preprocessed_data, output_dir)

def plot_vst_vinge_comparison(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Create visualization comparing VST_RAW, VST_EDT, and VINGE measurements."""
    if original_data is None:
        print("Skipping VST-VINGE comparison plot: original data not provided")
        return
        
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if (station_name in original_data and
            original_data[station_name]['vst_raw'] is not None and 
            station_data.get('vst_edt') is not None and 
            station_data.get('vinge') is not None):
            
            print(f"\nProcessing station {station_name}:")
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.3)
            
            # Main plot with VST_RAW, VST_EDT, and VINGE
            ax1 = fig.add_subplot(gs[0])
            
            # Ensure data is properly indexed
            vst_raw = original_data[station_name]['vst_raw'].copy()
            vst_edt = station_data['vst_edt'].copy()
            vinge_data = station_data['vinge'].copy()
            
            print(f"VINGE data shape before processing: {vinge_data.shape}")
            print(f"VINGE data columns: {vinge_data.columns}")
            print(f"First few rows of VINGE data:\n{vinge_data.head()}")
            
            if not isinstance(vst_raw.index, pd.DatetimeIndex):
                vst_raw.set_index('Date', inplace=True)
            if not isinstance(vst_edt.index, pd.DatetimeIndex):
                vst_edt.set_index('Date', inplace=True)
            if not isinstance(vinge_data.index, pd.DatetimeIndex):
                vinge_data.set_index('Date', inplace=True)
            
            # Focus on last 4 years
            end_date = vst_raw.index.max()
            start_date = end_date - pd.DateOffset(years=4)
            
            print(f"Date range: {start_date} to {end_date}")
            
            # Filter data for last 4 years
            vst_raw = vst_raw[vst_raw.index >= start_date]
            vst_edt = vst_edt[vst_edt.index >= start_date]
            vinge_data = vinge_data[vinge_data.index >= start_date]
            
            print(f"VINGE data points after date filtering: {len(vinge_data)}")
            
            # Get the value column name from the original data
            vst_raw_col = [col for col in vst_raw.columns if col != 'Date'][0]
            vst_edt_col = [col for col in vst_edt.columns if col != 'Date'][0]
            
            print(f"Column names - VST Raw: {vst_raw_col}, VST EDT: {vst_edt_col}")
            print("VINGE data columns:", vinge_data.columns)
            
            # Convert VINGE water level from cm to mm
            vinge_data['water_level_mm'] = vinge_data['W.L [cm]']
            
            # Convert data to numeric type
            vst_raw[vst_raw_col] = pd.to_numeric(vst_raw[vst_raw_col], errors='coerce')
            vst_edt[vst_edt_col] = pd.to_numeric(vst_edt[vst_edt_col], errors='coerce')
            
            # Print value ranges
            print(f"VST Raw range: {vst_raw[vst_raw_col].min():.1f} to {vst_raw[vst_raw_col].max():.1f}")
            print(f"VST EDT range: {vst_edt[vst_edt_col].min():.1f} to {vst_edt[vst_edt_col].max():.1f}")
            print(f"VINGE range: {vinge_data['water_level_mm'].min():.1f} to {vinge_data['water_level_mm'].max():.1f}")
            
            # Plot raw VST data
            ax1.plot(vst_raw.index, vst_raw[vst_raw_col],
                    color='#1f77b4', alpha=0.7, linewidth=1.0, 
                    label='VST Raw')
            
            # Plot EDT corrected data
            ax1.plot(vst_edt.index, vst_edt[vst_edt_col],
                    color='#2ca02c', alpha=0.8, linewidth=1.5, 
                    label='VST EDT')
            
            # Plot VINGE measurements with larger markers and higher zorder
            ax1.scatter(vinge_data.index, vinge_data['water_level_mm'],
                       color='#d62728', alpha=0.8, s=50, 
                       label='Manual Board (VINGE)', zorder=5)
            
            print(f"Number of VINGE points plotted: {len(vinge_data)}")
            
            # Bottom subplot for VINGE vs VST_RAW differences
            ax2 = fig.add_subplot(gs[1])
            
            # Calculate differences between VINGE and closest VST_RAW readings
            differences = []
            vinge_dates = []
            closest_vst_values = []
            
            for date, value in vinge_data.iterrows():
                # Find closest VST reading within 12 hours
                window_start = date - pd.Timedelta(hours=12)
                window_end = date + pd.Timedelta(hours=12)
                closest_vst = vst_raw.loc[(vst_raw.index >= window_start) & 
                                        (vst_raw.index <= window_end)]
                
                if not closest_vst.empty:
                    # Find the closest date
                    closest_date = closest_vst.index[abs(closest_vst.index - date).argmin()]
                    closest_value = vst_raw.loc[closest_date, vst_raw_col]
                    vinge_value = value['water_level_mm']
                    
                    # Only calculate difference if both values are numeric and not NaN
                    if pd.notnull(closest_value) and pd.notnull(vinge_value):
                        # Calculate difference between VINGE and closest VST
                        diff = float(vinge_value) - float(closest_value)
                        differences.append(diff)
                        vinge_dates.append(date)
                        closest_vst_values.append(closest_value)
            
            # Plot differences
            if differences:  # Only plot if we have valid differences
                ax2.scatter(vinge_dates, differences, color='#1f77b4', alpha=0.8, s=40,
                           label='VINGE - VST_RAW difference')
            
            # Add horizontal line at 0 for reference
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add threshold lines at ±20mm
            ax2.axhline(y=20, color='#d62728', linestyle='--', alpha=0.8,
                       label='Correction Threshold (±20mm)')
            ax2.axhline(y=-20, color='#d62728', linestyle='--', alpha=0.8)
            
            # Style the plots
            ax1.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
            ax2.set_ylabel('Difference (mm)', fontsize=14, labelpad=10)
            ax2.set_xlabel('Date', fontsize=14, labelpad=10)
            
            # Create consistent date range and ticks for both plots
            date_range = pd.date_range(start=start_date, end=end_date, freq='6M')
            
            for ax in [ax1, ax2]:
                # Set the same x-axis limits for both plots
                ax.set_xlim(start_date, end_date)
                
                # Set major ticks at 6-month intervals
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                
                # Set minor ticks at 1-month intervals
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
                
                # Rotate labels
                ax.tick_params(axis='x', rotation=45)
                
                # Ensure the grid aligns with major ticks
                ax.grid(True, which='major', linestyle='--', alpha=0.7, color='#cccccc')
                ax.grid(True, which='minor', linestyle=':', alpha=0.3, color='#cccccc')
            
            # Style the legends
            ax1.legend(loc='upper right', frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', fontsize=12)
            ax2.legend(loc='lower right', frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', fontsize=12)
            
            # Set a professional background style for all subplots
            for ax in [ax1, ax2]:
                ax.set_facecolor('#f8f9fa')
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
                
                # Add more padding to y-axis
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            # Adjust layout to prevent label cutoff
            plt.subplots_adjust(bottom=0.2)
            
            plt.savefig(diagnostic_dir / f"{station_name}_vst_vinge_comparison.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()


def plot_climate_water_level(preprocessed_data: dict, output_dir: Path):
    """Create thesis-quality visualization for climate and water level data."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None and station_data['rainfall'] is not None and station_data['temperature'] is not None:
            fig = plt.figure(figsize=(15, 16))
            gs = GridSpec(4, 1, figure=fig, height_ratios=[1.5, 1, 1, 2], hspace=0.3)
            
            # --- Temperature subplot ---
            ax1 = fig.add_subplot(gs[0])
            temp_data = station_data['temperature'].copy()
            if not isinstance(temp_data.index, pd.DatetimeIndex):
                temp_data.set_index('Date', inplace=True)
            
            # Get temperature column name
            temp_col = [col for col in temp_data.columns if col != 'Date'][0]
            
            ax1.plot(temp_data.index, temp_data[temp_col],
                   color='#d62728', alpha=0.8, linewidth=1.5, label='Temperature')
            
            # Add freezing line
            ax1.axhline(y=0, color='#2ca02c', linestyle='--', linewidth=1.5, 
                      alpha=0.8, label='Freezing Point (0°C)')
            
            # Find freezing periods
            freezing = temp_data[temp_col] < 0
            freezing_groups = []
            in_freezing = False
            start_date = None
            
            for date, is_freezing in zip(temp_data.index, freezing):
                if is_freezing and not in_freezing:
                    # Start of freezing period
                    start_date = date
                    in_freezing = True
                elif not is_freezing and in_freezing:
                    # End of freezing period
                    freezing_groups.append((start_date, date))
                    in_freezing = False
            
            # Add last period if still freezing at the end
            if in_freezing:
                freezing_groups.append((start_date, temp_data.index[-1]))
            
            # Highlight significant freezing periods
            for start, end in freezing_groups:
                if (end - start).days >= 3:  # Only highlight significant freezing periods
                    ax1.axvspan(start, end, alpha=0.2, color='#9467bd',
                             label='Freezing Period' if start == freezing_groups[0][0] else "")
            
            # Add temperature statistics
            temp_stats = (
                f'Range: {temp_data[temp_col].min():.1f}°C to '
                f'{temp_data[temp_col].max():.1f}°C\n'
                f'Average: {temp_data[temp_col].mean():.1f}°C\n'
                f'Days Below 0°C: {sum(freezing)}\n'
                f'Freezing Periods: {len(freezing_groups)}'
            )
            
            ax1.text(0.02, 0.98, temp_stats, transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='white',
                            alpha=0.9,
                            edgecolor='#cccccc'))
            
            # --- Monthly temperature subplot ---
            ax2 = fig.add_subplot(gs[1])
            # Calculate monthly average temperatures
            monthly_temp = temp_data.resample('M').mean()
            
            # Create a bar plot of monthly temperatures
            months = monthly_temp.index
            ax2.bar(months, monthly_temp[temp_col], 
                   color='#d62728', alpha=0.7, width=25)
            
            # Add freezing line
            ax2.axhline(y=0, color='#2ca02c', linestyle='--', linewidth=1.5, 
                      alpha=0.8, label='Freezing Point (0°C)')
            
            # --- Rainfall subplot ---
            ax3 = fig.add_subplot(gs[2])
            rain_data = station_data['rainfall'].copy()
            if not isinstance(rain_data.index, pd.DatetimeIndex):
                rain_data.set_index('Date', inplace=True)
            
            # Get rainfall column name
            rain_col = [col for col in rain_data.columns if col != 'Date'][0]
            
            # Calculate and plot moving average for smoother visualization
            rain_rolling = rain_data[rain_col].rolling(window=30, min_periods=1).mean()
            ax3.plot(rain_data.index, rain_rolling, 
                   color='#1f77b4', alpha=0.9, linewidth=2, label='30-day Moving Average')
            
            # Add light bar plot in background for daily values
            ax3.bar(rain_data.index, rain_data[rain_col], 
                   color='#1f77b4', alpha=0.3, width=1, label='Daily Precipitation')
            
            # Add rainfall statistics
            rain_stats = (
                f'Total Rainfall: {rain_data[rain_col].sum():.1f} mm\n'
                f'Average: {rain_data[rain_col].mean():.2f} mm/day\n'
                f'Max Daily: {rain_data[rain_col].max():.1f} mm\n'
                f'Rainy Days: {sum(rain_data[rain_col] > 0)}'
            )
            
            ax3.text(0.02, 0.98, rain_stats, transform=ax3.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                            alpha=0.9,
                           edgecolor='#cccccc'))
            
            # --- Water Level subplot ---
            ax4 = fig.add_subplot(gs[3])
            vst_data = station_data['vst_raw'].copy()
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            # Get VST column name
            vst_col = [col for col in vst_data.columns if col != 'Date'][0]
            
            ax4.plot(vst_data.index, vst_data[vst_col],
                   color='#2ca02c', alpha=0.8, linewidth=1.5, label='Water Level')
            
            # Highlight freezing periods in water level subplot too
            for start, end in freezing_groups:
                if (end - start).days >= 3:  # Only highlight significant freezing periods
                    ax4.axvspan(start, end, alpha=0.2, color='#9467bd',
                             label='Freezing Period' if start == freezing_groups[0][0] else "")
            
            # Add water level statistics
            water_stats = (
                f'Range: {vst_data[vst_col].min():.1f} to {vst_data[vst_col].max():.1f} mm\n'
                f'Average: {vst_data[vst_col].mean():.1f} mm\n'
                f'Standard Dev: {vst_data[vst_col].std():.1f} mm\n'
                f'Measurements: {len(vst_data)}'
            )
            
            ax4.text(0.02, 0.98, water_stats, transform=ax4.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                            alpha=0.9,
                           edgecolor='#cccccc'))
            
            # Set titles and labels
            ax1.set_title('Temperature Data', fontsize=16, fontweight='bold', pad=15)
            ax2.set_title('Monthly Average Temperature', fontsize=16, fontweight='bold', pad=15)
            ax3.set_title('Precipitation Data', fontsize=16, fontweight='bold', pad=15)
            ax4.set_title('Water Level Measurements', fontsize=16, fontweight='bold', pad=15)
            
            ax1.set_ylabel('Temperature (°C)', fontsize=14, labelpad=10)
            ax2.set_ylabel('Temperature (°C)', fontsize=14, labelpad=10)
            ax3.set_ylabel('Precipitation (mm)', fontsize=14, labelpad=10)
            ax4.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
            ax4.set_xlabel('Date', fontsize=14, labelpad=10)
            
            # Add legends
            for i, ax in enumerate([ax1, ax2, ax3, ax4]):
                # Handle potentially duplicate labels
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), 
                         loc='best', frameon=True, framealpha=0.9,
                         edgecolor='#cccccc', fontsize=12)
            
            # Set a professional background style for all subplots
            for ax in [ax1, ax2, ax3, ax4]:
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
            
            monthly_data = [vst_data_copy[vst_data_copy['month'] == month]['vst_raw'] 
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
            yearly_data = [vst_data_copy[vst_data_copy['year'] == year]['vst_raw'] 
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
            monthly_means = vst_data_copy.groupby('month')['vst_raw'].mean()
            monthly_std = vst_data_copy.groupby('month')['vst_raw'].std()
            
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
                    rolling_avg = year_data['vst_raw'].rolling(window=15, center=True).mean()
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
                daily_data = vst_data['vst_raw'].resample('D').mean()
                
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
                
                # Resample to daily data for more manageable processing
                daily_data = vst_data['vst_raw'].resample('D').mean()
                
                # Fill small gaps in daily data (up to 7 days)
                daily_data = daily_data.interpolate(method='linear', limit=7)
                
                # Drop NaN values if any remain
                daily_data = daily_data.dropna()
                
                # Skip if not enough data
                if len(daily_data) < 365:  # Need at least a year of data
                    print(f"Skipping change point detection for {station_name}: insufficient data")
                    continue
                
                # Create figure
                fig = plt.figure(figsize=(15, 15))
                gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.3)
                
                # 1. Original time series with change points
                ax1 = fig.add_subplot(gs[0])
                
                # Plot water level data
                ax1.plot(daily_data.index, daily_data.values, color='#1f77b4', 
                       linewidth=1, alpha=0.7, label='Daily water level')
                
                # Apply change point detection
                # Convert to numpy array for the algorithm
                signal = daily_data.values
                
                # Use Pelt method for change point detection with robust cost function
                # BIC penalty helps avoid over-segmentation
                algo = rpt.Pelt(model="rbf").fit(signal)
                change_points = algo.predict(pen=10)
                
                # Convert indices back to dates
                change_point_dates = [daily_data.index[cp-1] if cp < len(daily_data) else daily_data.index[-1] 
                                    for cp in change_points[:-1]]  # Exclude the last point which is len(signal)
                
                # Create segments
                segments = []
                segment_starts = [0] + change_points[:-1]
                segment_ends = change_points
                
                for start, end in zip(segment_starts, segment_ends):
                    if end > start:  # Valid segment
                        segment = signal[start:end]
                        segments.append((start, end, np.mean(segment)))
                
                # Plot each segment with different colors and add mean line
                colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
                
                for i, (start, end, mean_val) in enumerate(segments):
                    # Plot segment with unique color
                    start_date = daily_data.index[start]
                    end_date = daily_data.index[min(end-1, len(daily_data)-1)]
                    
                    segment_dates = daily_data.index[start:end]
                    segment_values = signal[start:end]
                    
                    ax1.plot(segment_dates, segment_values, color=colors[i], linewidth=1.5, alpha=0.8,
                           label=f'Segment {i+1}' if i < 5 else "")
                    
                    # Add horizontal line for segment mean
                    ax1.hlines(y=mean_val, xmin=start_date, xmax=end_date, 
                             color=colors[i], linestyle='--', linewidth=2, alpha=0.9)
                
                # Add vertical lines for change points
                for cp_date in change_point_dates:
                    ax1.axvline(x=cp_date, color='#d62728', linestyle='-', linewidth=1.5, alpha=0.8)
                    # Add date annotation
                    ax1.annotate(cp_date.strftime('%Y-%m-%d'), 
                               xy=(cp_date, ax1.get_ylim()[0]),
                               xytext=(cp_date, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0])*0.05),
                               rotation=90, fontsize=9, ha='right', va='top',
                               color='#d62728')
                
                ax1.set_title('Stream Water Level with Detected Change Points',
                            fontsize=16, fontweight='bold', pad=15)
                ax1.set_ylabel('Water Level (mm)', fontsize=14, labelpad=10)
                
                # Add a custom legend for the first few segments
                handles, labels = ax1.get_legend_handles_labels()
                # Remove duplicate daily water level entry if it exists
                if labels and labels[0] == 'Daily water level':
                    handles = handles[1:6]  # Just keep first 5 segments
                    labels = labels[1:6]
                ax1.legend(handles, labels, loc='upper right', frameon=True,
                         framealpha=0.9, edgecolor='#cccccc', fontsize=10)
                
                # 2. Annual statistics by segment
                ax2 = fig.add_subplot(gs[1])
                
                # Calculate annual statistics for each segment
                annual_stats = []
                
                for i, (start, end, _) in enumerate(segments):
                    segment_dates = daily_data.index[start:end]
                    segment_values = signal[start:end]
                    
                    # Create temporary pandas Series for easy resampling
                    segment_series = pd.Series(segment_values, index=segment_dates)
                    
                    # Calculate annual statistics
                    if len(segment_series) > 30:  # Only if segment has meaningful length
                        annual_mean = segment_series.resample('Y').mean()
                        annual_min = segment_series.resample('Y').min()
                        annual_max = segment_series.resample('Y').max()
                        
                        # Plot annual means with error bars for min/max
                        years = [d.year for d in annual_mean.index]
                        ax2.errorbar(years, annual_mean.values, 
                                   yerr=[annual_mean.values - annual_min.values, 
                                         annual_max.values - annual_mean.values],
                                   fmt='o-', color=colors[i], alpha=0.8, 
                                   capsize=5, label=f'Segment {i+1}')
                        
                        # Store segment info for reporting
                        for year, mean_val, min_val, max_val in zip(
                            years, annual_mean.values, annual_min.values, annual_max.values):
                            annual_stats.append({
                                'segment': i+1,
                                'year': year,
                                'mean': mean_val,
                                'min': min_val,
                                'max': max_val
                            })
                
                ax2.set_xlabel('Year', fontsize=14, labelpad=10)
                ax2.set_ylabel('Annual Water Level (mm)', fontsize=14, labelpad=10)
                ax2.set_title('Annual Water Level Statistics by Segment', 
                           fontsize=16, fontweight='bold', pad=15)
                ax2.legend(loc='best', frameon=True, framealpha=0.9,
                         edgecolor='#cccccc', fontsize=10)
                
                # 3. Segment statistics comparison
                ax3 = fig.add_subplot(gs[2])
                
                # Calculate key statistics for each segment
                segment_means = []
                segment_stds = []
                segment_ranges = []
                segment_durations = []
                segment_labels = []
                
                for i, (start, end, _) in enumerate(segments):
                    segment_dates = daily_data.index[start:end]
                    segment_values = signal[start:end]
                    
                    if len(segment_values) > 0:
                        segment_means.append(np.mean(segment_values))
                        segment_stds.append(np.std(segment_values))
                        segment_ranges.append(np.max(segment_values) - np.min(segment_values))
                        
                        # Calculate duration in days
                        duration = (segment_dates[-1] - segment_dates[0]).days + 1
                        segment_durations.append(duration)
                        
                        # Create segment label with duration
                        if duration > 365:
                            label = f"{i+1}\n{duration/365:.1f} yrs"
                        else:
                            label = f"{i+1}\n{duration} days"
                        segment_labels.append(label)
                
                # Width for the bars
                bar_width = 0.25
                r1 = np.arange(len(segment_means))
                r2 = [x + bar_width for x in r1]
                r3 = [x + bar_width for x in r2]
                
                # Create grouped bar chart
                ax3.bar(r1, segment_means, width=bar_width, color='#1f77b4', alpha=0.7, label='Mean')
                ax3.bar(r2, segment_stds, width=bar_width, color='#ff7f0e', alpha=0.7, label='St. Dev.')
                ax3.bar(r3, segment_ranges, width=bar_width, color='#2ca02c', alpha=0.7, label='Range')
                
                # Add segment duration as text
                for i, (mean, label) in enumerate(zip(segment_means, segment_labels)):
                    ax3.text(i, mean/2, label, ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=10)
                
                # Set x-axis ticks
                ax3.set_xticks([r + bar_width for r in range(len(segment_means))])
                ax3.set_xticklabels([f"Segment {i+1}" for i in range(len(segment_means))])
                
                ax3.set_ylabel('vst_raw (mm)', fontsize=14, labelpad=10)
                ax3.set_title('Statistical Comparison Across Segments', 
                           fontsize=16, fontweight='bold', pad=15)
                ax3.legend(loc='upper right', frameon=True, framealpha=0.9,
                         edgecolor='#cccccc', fontsize=10)
                
                # Add a summary box for change points
                if change_point_dates:
                    cp_summary = "Detected Change Points:\n"
                    for i, cp_date in enumerate(change_point_dates):
                        # Calculate water level change at this point
                        if i < len(segments) - 1:
                            level_before = segments[i][2]  # Mean of segment before
                            level_after = segments[i+1][2]  # Mean of segment after
                            change = level_after - level_before
                            change_pct = (change / level_before) * 100 if level_before != 0 else float('inf')
                            
                            cp_summary += (f"• {cp_date.strftime('%Y-%m-%d')}: "
                                         f"{change:+.1f}mm ({change_pct:+.1f}%)\n")
                    
                    ax1.text(0.02, 0.05, cp_summary, transform=ax1.transAxes,
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
                fig.suptitle(f'{station_name} - Change Point Detection Analysis',
                           fontsize=18, fontweight='bold', y=0.98)
                
                # Don't use layout adjustments - rely on bbox_inches='tight' when saving
                # plt.subplots_adjust(top=0.97, bottom=0.06, left=0.08, right=0.92, hspace=0.3)
                
                # Save the figure
                plt.savefig(diagnostic_dir / f"{station_name}_change_point_analysis.png",
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
            except Exception as e:
                print(f"Error creating change point analysis for {station_name}: {str(e)}")
                import traceback
                traceback.print_exc()
