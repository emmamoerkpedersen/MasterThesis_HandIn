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
            f.write(f"{row['Removal %']:.1f} & ")
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
    """Create simple visualization of VST, temperature, and rainfall data."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name in preprocessed_data.keys():
        if station_name in original_data and original_data[station_name]['vst_raw'] is not None:
            # Create figure with three subplots
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
            
            # Get data for this station
            orig_data = original_data[station_name]
            proc_data = preprocessed_data[station_name]
            
            # 1. Water level measurements (VST_RAW)
            ax1 = fig.add_subplot(gs[0])
            if orig_data['vst_raw'] is not None:
                vst_data = orig_data['vst_raw'].copy()
                if not isinstance(vst_data.index, pd.DatetimeIndex):
                    vst_data.set_index('Date', inplace=True)
                
                # Get the VST column name (no date filtering)
                vst_col = [col for col in vst_data.columns if col != 'Date'][0]
                
                ax1.plot(vst_data.index, vst_data[vst_col],
                        color='#1f77b4', alpha=0.7, linewidth=1, label='VST Raw')
            
            # 2. Temperature data
            ax2 = fig.add_subplot(gs[1])
            if proc_data['temperature'] is not None:
                temp_data = proc_data['temperature'].copy()
                if not isinstance(temp_data.index, pd.DatetimeIndex):
                    temp_data.set_index('Date', inplace=True)
                
                # Get temperature column name
                temp_col = [col for col in temp_data.columns if col != 'Date'][0]
                
                ax2.plot(temp_data.index, temp_data[temp_col],
                        color='#d62728', alpha=0.7, linewidth=1, label='Temperature')
            
            # 3. Rainfall data
            ax3 = fig.add_subplot(gs[2])
            if proc_data['rainfall'] is not None:
                rain_data = proc_data['rainfall'].copy()
                if not isinstance(rain_data.index, pd.DatetimeIndex):
                    rain_data.set_index('Date', inplace=True)
                
                # Get rainfall column name
                rain_col = [col for col in rain_data.columns if col != 'Date'][0]
                
                # Plot rainfall directly without resampling
                ax3.bar(rain_data.index, rain_data[rain_col],
                       color='#1f77b4', alpha=0.7, width=0.8, label='Rainfall')
            
            # Set titles and labels
            axes = [ax1, ax2, ax3]
            titles = ['Water Level Measurements', 'Temperature', 'Rainfall']
            ylabels = ['Water Level (mm)', 'Temperature (°C)', 'Precipitation (mm)']
            
            # Format x-axis and style plots
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            for ax, title, ylabel in zip(axes, titles, ylabels):
                ax.set_title(title, pad=20, fontsize=14)
                ax.set_ylabel(ylabel, labelpad=10)
                ax.legend(loc='upper right')
            
            # Add main title
            fig.suptitle(f'Station {station_name} - Data Overview',
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save the figure
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

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
            print(f"VINGE data columns:", vinge_data.columns)
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
            
            # Remove NaN values from VINGE data since these are manual measurements
            vinge_data = vinge_data.dropna()
            
            print(f"VINGE data points after date filtering and NaN removal: {len(vinge_data)}")
            
            # Get the value column name from the original data
            vst_raw_col = [col for col in vst_raw.columns if col != 'Date'][0]
            vst_edt_col = [col for col in vst_edt.columns if col != 'Date'][0]
            
            print(f"Column names - VST Raw: {vst_raw_col}, VST EDT: {vst_edt_col}")
            print("VINGE data columns:", vinge_data.columns)
            
            # Convert VINGE water level from cm to mm
            if 'W.L [cm]' in vinge_data.columns:
                vinge_data['water_level_mm'] = vinge_data['W.L [cm]']
            elif 'vinge' in vinge_data.columns:
                vinge_data['water_level_mm'] = vinge_data['vinge']
            else:
                print(f"Warning: No recognized VINGE column found. Available columns: {vinge_data.columns}")
                return
            
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
