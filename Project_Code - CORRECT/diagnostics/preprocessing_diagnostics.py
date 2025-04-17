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
    
    # Set publication-quality styling similar to create_water_level_plot_png
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame to store statistics for all stations
    stats_data = []
    
    # Set start date to January 1, 2010
    start_date = pd.to_datetime('2010-01-01')
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with GridSpec
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.1)
            
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
            proc = proc[proc.index >= start_date]
            
            # PERFORMANCE OPTIMIZATION: Downsample data if too large (more than 10,000 points)
            if len(orig) > 10000:
                # Use efficient resampling instead of random sampling
                orig_plot = orig.resample('6H').mean().dropna()
            else:
                orig_plot = orig
                
            if len(proc) > 10000:
                # Use efficient resampling for processed data
                proc_plot = proc.resample('6H').mean().dropna()
            else:
                proc_plot = proc
            
            # Calculate IQR bounds using the original data (use full dataset for calculations)
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
            
            # Plot the original data
            ax1.plot(orig_plot.index, orig_plot[orig_value_col], color='#1f77b4', alpha=0.8, 
                    linewidth=0.8, label='Original Data', zorder=2)
            
            # Add IQR bounds with improved styling
            ax1.axhline(y=lower_bound, color='#ff7f0e', linestyle='--', alpha=0.6,
                       linewidth=0.8, label='IQR Bounds', zorder=1)
            ax1.axhline(y=upper_bound, color='#ff7f0e', linestyle='--', alpha=0.6,
                       linewidth=0.8, zorder=1)
            
            # Add frost periods if available
            if frost_periods:
                for start, end in frost_periods:
                    if start >= start_date:  # Only show frost periods after 2010
                        ax1.axvspan(start, end, color='#E3F2FD', alpha=0.5, 
                                  label='Frost Period' if start == frost_periods[0][0] else "", zorder=1)
            
            # Find actual outliers (points removed during preprocessing)
            # We do this by comparing original data with preprocessed data
            # To get the missing points:
            outlier_points = orig[outlier_mask].copy()
            
            print(f"Found {len(outlier_points)} outlier points for station {station_name}")
            print(f"Outlier value range: {outlier_points[orig_value_col].min()} to {outlier_points[orig_value_col].max()}")
            
            # PERFORMANCE OPTIMIZATION: Only plot a sample of outlier points if there are too many
            if len(outlier_points) > 0:
                if len(outlier_points) > 1000:
                    # Sample to get at most 1000 outlier points
                    outlier_sample = outlier_points.sample(n=min(1000, len(outlier_points)), random_state=42)
                    ax1.scatter(outlier_sample.index, outlier_sample[orig_value_col],
                              color='#d62728', s=25, alpha=0.7, label='Removed Points (Sample)', zorder=3)
                    print(f"Plotting {len(outlier_sample)} sample outlier points")
                else:
                    # Plot all outliers
                    ax1.scatter(outlier_points.index, outlier_points[orig_value_col],
                              color='#d62728', s=25, alpha=0.7, label='Removed Points', zorder=3)
            
            #ax1.set_title('Original Data with Quality Control Bounds', fontweight='bold', pad=15)
            ax1.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
            
            # Clean styling similar to create_water_level_plot_png
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(False)
            ax1.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            
            # Bottom subplot: Preprocessed data only
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(proc_plot.index, proc_plot['vst_raw'], color='#2ca02c', alpha=0.8, 
                    linewidth=0.8, label='Preprocessed Data', zorder=2)
            
            #ax2.set_title('Preprocessed Data (2010 onwards)', fontweight='bold', pad=15)
            ax2.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
            ax2.set_xlabel('Date', fontweight='bold', labelpad=10)
            
            # Clean styling similar to create_water_level_plot_png
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(False)
            ax2.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            
            # Set consistent x-axis limits
            x_min = start_date
            x_max = pd.to_datetime('2022-01-01')  # Set a consistent end date or use data max
            for ax in [ax1, ax2]:
                ax.set_xlim(x_min, x_max)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
                ax.tick_params(axis='x', rotation=45)
            
            # Set consistent y-axis range for the preprocessed data plot
            ax2.set_ylim(0, max(proc_plot['vst_raw'].max() * 1.1, upper_bound * 1.1))
            
            # Add main title
            #fig.suptitle(f'Data Preprocessing - Station {station_name}', 
            #            fontweight='bold', y=0.95)
            
            # Format the figure for nice display
            fig.autofmt_xdate()
            plt.tight_layout()
            
            # PERFORMANCE OPTIMIZATION: Use a lower DPI for faster rendering, but still good quality
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", 
                       dpi=200, bbox_inches='tight', facecolor='white')
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
    """Create visualization of temperature, rainfall, and VST data showing full available date ranges."""
    set_plot_style()
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication-quality styling similar to create_water_level_plot_png
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Define start date for precipitation data
    precip_start_date = pd.to_datetime('2010-01-01')
    
    for station_name in original_data.keys():
        if original_data[station_name]['vst_raw'] is not None:
            # Create figure with GridSpec for better control of subplot heights - now with 4 rows
            fig = plt.figure(figsize=(15, 15))  # Increased figure height for 4 subplots
            gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 3], hspace=0.2)
            
            # Get data for this station (use original data for all plots)
            orig_data = original_data[station_name]
            
            # Track min and max dates to show data availability
            min_dates = {}
            max_dates = {}
            
            # 1. Temperature data (first subplot) - Using ORIGINAL data
            ax1 = fig.add_subplot(gs[0])
            if orig_data['temperature'] is not None:
                temp_data = orig_data['temperature'].copy()
                if not isinstance(temp_data.index, pd.DatetimeIndex):
                    temp_data.set_index('Date', inplace=True)
                
                # Get temperature column name
                temp_col = [col for col in temp_data.columns if col != 'Date'][0]
                
                # PERFORMANCE OPTIMIZATION: Downsample temperature data if needed
                if len(temp_data) > 10000:
                    temp_data = temp_data.resample('6H').mean().dropna()
                
                # Track date range
                min_dates['temperature'] = temp_data.index.min()
                max_dates['temperature'] = temp_data.index.max()
                
                ax1.plot(temp_data.index, temp_data[temp_col],
                        color='#d62728', alpha=0.8, linewidth=0.8, label='Temperature')
                
                ax1.set_ylabel('Temperature (°C)', fontweight='bold', labelpad=10)
                ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best')
                
                # Clean style similar to water_level_plot
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.grid(False)
                
                # Set x-axis limits for temperature subplot - use its own data range
                ax1.set_xlim(min_dates['temperature'], max_dates['temperature'])
                
                # Ensure x-axis is visible
                ax1.xaxis.set_visible(True)
            else:
                ax1.text(0.5, 0.5, 'No temperature data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=14)
            
            # 2. Rainfall data (second subplot) - from 2010 onwards - Using ORIGINAL data
            ax2 = fig.add_subplot(gs[1])
            if orig_data['rainfall'] is not None:
                rain_data = orig_data['rainfall'].copy()
                if not isinstance(rain_data.index, pd.DatetimeIndex):
                    rain_data.set_index('Date', inplace=True)
                
                # Handle timezone issues - ensure both are timezone-naive or timezone-aware
                if rain_data.index.tz is not None:
                    # Make precip_start_date timezone-aware to match the DataFrame
                    precip_start_date_adj = precip_start_date.tz_localize(rain_data.index.tz)
                    rain_data_filtered = rain_data[rain_data.index >= precip_start_date_adj]
                else:
                    # Both are timezone-naive
                    rain_data_filtered = rain_data[rain_data.index >= precip_start_date]
                
                # Get rainfall column name
                rain_col = [col for col in rain_data.columns if col != 'Date'][0]
                
                # PERFORMANCE OPTIMIZATION: For rainfall, resample to daily sum for better visualization
                if len(rain_data_filtered) > 1000:
                    rain_data_plot = rain_data_filtered.resample('D').sum().dropna()
                else:
                    rain_data_plot = rain_data_filtered
                
                # Track date range of filtered data
                if not rain_data_filtered.empty:
                    min_dates['rainfall'] = rain_data_filtered.index.min()
                    max_dates['rainfall'] = rain_data_filtered.index.max()
                    
                    # Plot rainfall as bars
                    ax2.bar(rain_data_plot.index, rain_data_plot[rain_col],
                           color='#1f77b4', alpha=0.7, width=1, label='Rainfall')
                    
                    ax2.set_ylabel('Precipitation (mm)', fontweight='bold', labelpad=10)
                    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best')
                    
                    # Clean style similar to water_level_plot
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.grid(False)
                    
                    # Set x-axis limits specifically for rainfall subplot - use its own data range
                    ax2.set_xlim(min_dates['rainfall'], max_dates['rainfall'])
                    
                    # Ensure x-axis is visible
                    ax2.xaxis.set_visible(True)
                else:
                    ax2.text(0.5, 0.5, 'No precipitation data available from 2010 onwards', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No rainfall data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=14)
            
            # 3. VINGE data subplot (manual measurements) - Using ORIGINAL data
            ax3 = fig.add_subplot(gs[2])
            if orig_data.get('vinge') is not None:
                vinge_data = orig_data['vinge'].copy()
                
                # Just use the VINGE data as is, without any filtering or processing
                if not isinstance(vinge_data.index, pd.DatetimeIndex):
                    if 'Date' in vinge_data.columns:
                        vinge_data.set_index('Date', inplace=True)
                    else:
                        # Try to convert the index to datetime
                        try:
                            vinge_data.index = pd.to_datetime(vinge_data.index)
                        except:
                            print(f"Warning: Could not convert VINGE index to datetime for station {station_name}.")
                
                # Identify the column with VINGE data
                if 'W.L [cm]' in vinge_data.columns:
                    vinge_col = 'W.L [cm]'
                    # Assume VINGE data is already in mm
                    vinge_data['water_level_mm'] = vinge_data[vinge_col]
                elif 'vinge' in vinge_data.columns:
                    vinge_col = 'vinge'
                    vinge_data['water_level_mm'] = vinge_data[vinge_col]
                else:
                    # Try to use the first non-index column
                    value_cols = [col for col in vinge_data.columns if col != 'Date']
                    if value_cols:
                        vinge_col = value_cols[0]
                        vinge_data['water_level_mm'] = vinge_data[vinge_col]
                    else:
                        print(f"Warning: No suitable column found in VINGE data for station {station_name}.")
                        vinge_col = None
                
                if vinge_col is not None and isinstance(vinge_data.index, pd.DatetimeIndex):
                    # Remove NaN values
                    vinge_data = vinge_data.dropna(subset=['water_level_mm'])
                    
                    # Track date range
                    if not vinge_data.empty:
                        min_dates['vinge'] = vinge_data.index.min()
                        max_dates['vinge'] = vinge_data.index.max()
                        
                        # Plot as scatter (don't connect points)
                        ax3.scatter(vinge_data.index, vinge_data['water_level_mm'],
                                color='#d62728', alpha=0.8, s=20, 
                                label='Vinge', zorder=5)
                        
                        ax3.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
                        ax3.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best')
                        
                        # Clean style
                        ax3.spines['top'].set_visible(False)
                        ax3.spines['right'].set_visible(False)
                        ax3.grid(False)
                        
                        # Set x-axis limits using its own data range
                        ax3.set_xlim(min_dates['vinge'], max_dates['vinge'])
                        
                        # Ensure x-axis is visible
                        ax3.xaxis.set_visible(True)
                        
                        print(f"Plotted {len(vinge_data)} VINGE measurements for station {station_name}.")
                    else:
                        ax3.text(0.5, 0.5, 'No valid VINGE measurements after filtering', 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax3.transAxes, fontsize=14)
                else:
                    ax3.text(0.5, 0.5, 'VINGE data could not be properly processed', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes, fontsize=14)
            else:
                ax3.text(0.5, 0.5, 'No VINGE data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=14)
            
            # 4. Water level measurements (VST_RAW) - larger subplot at bottom - Using ORIGINAL data
            ax4 = fig.add_subplot(gs[3])
            if orig_data['vst_raw'] is not None:
                vst_data = orig_data['vst_raw'].copy()
                if not isinstance(vst_data.index, pd.DatetimeIndex):
                    vst_data.set_index('Date', inplace=True)
                
                # Get the VST column name
                vst_col = [col for col in vst_data.columns if col != 'Date'][0]
                
                # PERFORMANCE OPTIMIZATION: Downsample data if too large
                if len(vst_data) > 10000:
                    vst_data = vst_data.resample('6H').mean().dropna()
                
                # Track date range
                min_dates['vst_raw'] = vst_data.index.min()
                max_dates['vst_raw'] = vst_data.index.max()
                
                ax4.plot(vst_data.index, vst_data[vst_col],
                        color='#1f77b4', alpha=0.8, linewidth=0.8, label='VST Raw')
                
                ax4.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
                ax4.set_xlabel('Date', fontweight='bold', labelpad=10)
                ax4.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best')
                
                # Clean style similar to water_level_plot
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
                ax4.grid(False)
                
                # Set x-axis limits for VST subplot - use its own data range
                ax4.set_xlim(min_dates['vst_raw'], max_dates['vst_raw'])
            else:
                ax4.text(0.5, 0.5, 'No VST data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=14)
            
            # Format x-axis dates for all subplots - with independent x-axis ranges
            for ax, data_type in zip([ax1, ax2, ax3, ax4], ['temperature', 'rainfall', 'vinge', 'vst_raw']):
                if data_type in min_dates:
                    # Use appropriate date formatter based on the span of data
                    date_span = max_dates[data_type] - min_dates[data_type]
                    
                    if date_span.days > 365 * 10:  # More than 10 years
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
                    elif date_span.days > 365 * 2:  # More than 2 years
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        ax.xaxis.set_major_locator(mdates.YearLocator())  # Every year
                    else:  # Less than 2 years
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
                    
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Make sure x-axis ticks and labels are shown for each subplot
                    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            
            # Generate data availability text for title
            availability_text = []
            for data_type in ['temperature', 'rainfall', 'vinge', 'vst_raw']:
                if data_type in min_dates:
                    start_year = min_dates[data_type].year
                    end_year = max_dates[data_type].year
                    availability_text.append(f"{data_type.capitalize()}: {start_year}-{end_year}")
            
            # Add main title with data availability
           # fig.suptitle(f'Station {station_name} - Data Overview\n({", ".join(availability_text)})',
           #             fontweight='bold', y=0.95)
            
            # Format the figure for nice display - don't use autofmt_xdate as it can hide x-axis labels
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
            
            # PERFORMANCE OPTIMIZATION: Reduce DPI for faster rendering
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

def plot_vst_vinge_comparison(preprocessed_data: dict, output_dir: Path, original_data: dict = None):
    """Create visualization comparing VST_RAW, VST_EDT, and VINGE measurements."""
    if original_data is None:
        print("Skipping VST-VINGE comparison plot: original data not provided")
        return
        
    set_plot_style()
    
    # Set publication-quality styling similar to create_water_level_plot_png
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if (station_name in original_data and
            original_data[station_name]['vst_raw'] is not None and 
            station_data.get('vst_edt') is not None and 
            original_data[station_name].get('vinge') is not None):  # Use original VINGE data
            
            print(f"\nProcessing station {station_name}:")
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.1)
            
            # Main plot with VST_RAW, VST_EDT, and VINGE
            ax1 = fig.add_subplot(gs[0])
            
            # Ensure data is properly indexed
            vst_raw = original_data[station_name]['vst_raw'].copy()
            vst_edt = station_data['vst_edt'].copy()
            vinge_data = original_data[station_name]['vinge'].copy()  # Use original VINGE data
            
            # Print information about the VINGE data
            print(f"VINGE data shape before processing: {vinge_data.shape}")
            print(f"VINGE data columns:", vinge_data.columns)
            print(f"First few rows of VINGE data:\n{vinge_data.head()}")
            
            # Convert indices to DatetimeIndex if needed
            if not isinstance(vst_raw.index, pd.DatetimeIndex):
                vst_raw.set_index('Date', inplace=True)
            if not isinstance(vst_edt.index, pd.DatetimeIndex):
                vst_edt.set_index('Date', inplace=True)
            if not isinstance(vinge_data.index, pd.DatetimeIndex):
                if 'Date' in vinge_data.columns:
                    vinge_data.set_index('Date', inplace=True)
                else:
                    try:
                        vinge_data.index = pd.to_datetime(vinge_data.index)
                    except:
                        print(f"Warning: Could not convert VINGE index to datetime for station {station_name}.")
            
            # Focus on 2022-01-01 to 2024-01-01
            start_date = pd.to_datetime('2022-01-01')
            end_date = pd.to_datetime('2024-01-01')
            
            print(f"Date range: {start_date} to {end_date}")
            
            # Filter data for specified date range
            vst_raw_filtered = vst_raw[(vst_raw.index >= start_date) & (vst_raw.index <= end_date)]
            vst_edt_filtered = vst_edt[(vst_edt.index >= start_date) & (vst_edt.index <= end_date)]
            
            # Filter VINGE data if it has a datetime index
            if isinstance(vinge_data.index, pd.DatetimeIndex):
                vinge_filtered = vinge_data[(vinge_data.index >= start_date) & (vinge_data.index <= end_date)]
            else:
                print("Warning: VINGE data index is not a DatetimeIndex. Cannot filter by date.")
                vinge_filtered = vinge_data.copy()  # Use as is
                
            # Remove NaN values from VINGE data
            vinge_filtered = vinge_filtered.dropna()
            
            print(f"VINGE data points after date filtering and NaN removal: {len(vinge_filtered)}")
            
            # Get the value column names
            vst_raw_col = [col for col in vst_raw.columns if col != 'Date'][0]
            vst_edt_col = [col for col in vst_edt.columns if col != 'Date'][0]
            
            print(f"Column names - VST Raw: {vst_raw_col}, VST EDT: {vst_edt_col}")
            print("VINGE data columns:", vinge_filtered.columns)
            
            # Get or create the VINGE water level column - assume it's already in mm
            if 'W.L [cm]' in vinge_filtered.columns:
                vinge_filtered['water_level_mm'] = vinge_filtered['W.L [cm]']  # Assume already in mm
            elif 'vinge' in vinge_filtered.columns:
                vinge_filtered['water_level_mm'] = vinge_filtered['vinge']
            else:
                # Try to find any numeric column
                numeric_cols = [col for col in vinge_filtered.columns 
                              if pd.api.types.is_numeric_dtype(vinge_filtered[col])]
                
                if numeric_cols:
                    vinge_col = numeric_cols[0]
                    vinge_filtered['water_level_mm'] = vinge_filtered[vinge_col]
                    print(f"Using column '{vinge_col}' for VINGE water level data")
                else:
                    print(f"Warning: No recognized VINGE column found. Available columns: {vinge_filtered.columns}")
                    return
            
            # Convert data to numeric type
            vst_raw_filtered[vst_raw_col] = pd.to_numeric(vst_raw_filtered[vst_raw_col], errors='coerce')
            vst_edt_filtered[vst_edt_col] = pd.to_numeric(vst_edt_filtered[vst_edt_col], errors='coerce')
            vinge_filtered['water_level_mm'] = pd.to_numeric(vinge_filtered['water_level_mm'], errors='coerce')
            
            # Print value ranges
            if not vst_raw_filtered.empty:
                print(f"VST Raw range: {vst_raw_filtered[vst_raw_col].min():.1f} to {vst_raw_filtered[vst_raw_col].max():.1f}")
            
            if not vst_edt_filtered.empty:
                print(f"VST EDT range: {vst_edt_filtered[vst_edt_col].min():.1f} to {vst_edt_filtered[vst_edt_col].max():.1f}")
            
            if not vinge_filtered.empty:
                print(f"VINGE range: {vinge_filtered['water_level_mm'].min():.1f} to {vinge_filtered['water_level_mm'].max():.1f}")
            
            # PERFORMANCE OPTIMIZATION: Downsample large datasets for plotting
            if len(vst_raw_filtered) > 10000:
                # Use efficient resampling for VST data (average hourly data points)
                vst_raw_plot = vst_raw_filtered.resample('1H').mean().dropna()
            else:
                vst_raw_plot = vst_raw_filtered
                
            if len(vst_edt_filtered) > 10000:
                vst_edt_plot = vst_edt_filtered.resample('1H').mean().dropna()
            else:
                vst_edt_plot = vst_edt_filtered
            
            # Plot raw VST data
            ax1.plot(vst_raw_plot.index, vst_raw_plot[vst_raw_col],
                    color='#1f77b4', alpha=0.8, linewidth=0.8, 
                    label='VST Raw')
            
            # Plot EDT corrected data
            ax1.plot(vst_edt_plot.index, vst_edt_plot[vst_edt_col],
                    color='#2ca02c', alpha=0.8, linewidth=0.8, 
                    label='VST EDT')
            
            # Plot VINGE measurements with larger markers and higher zorder
            ax1.scatter(vinge_filtered.index, vinge_filtered['water_level_mm'],
                       color='#d62728', alpha=0.8, s=20, 
                       label='Vinge', zorder=5)
            
            print(f"Number of VINGE points plotted: {len(vinge_filtered)}")
            
            # Bottom subplot for VINGE vs VST_RAW differences
            ax2 = fig.add_subplot(gs[1])
            
            # Calculate differences between VINGE and closest VST_RAW readings
            differences = []
            vinge_dates = []
            closest_vst_values = []
            
            # Calculate differences for all VINGE measurements
            for date, row in vinge_filtered.iterrows():
                # Find closest VST reading within 12 hours
                window_start = date - pd.Timedelta(hours=12)
                window_end = date + pd.Timedelta(hours=12)
                closest_vst = vst_raw_filtered.loc[(vst_raw_filtered.index >= window_start) & 
                                        (vst_raw_filtered.index <= window_end)]
                
                if not closest_vst.empty:
                    # Find the closest date
                    closest_date = closest_vst.index[abs(closest_vst.index - date).argmin()]
                    closest_value = vst_raw_filtered.loc[closest_date, vst_raw_col]
                    vinge_value = row['water_level_mm']
                    
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
            #ax1.set_title('Water Level Measurements Comparison', fontweight='bold', pad=15)
            ax1.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
            
            #ax2.set_title('Difference Between Manual and Automated Measurements', fontweight='bold', pad=15)
            ax2.set_ylabel('Difference (mm)', fontweight='bold', labelpad=10)
            ax2.set_xlabel('Date', fontweight='bold', labelpad=10)
            
            # Apply consistent styling for both subplots
            for ax in [ax1, ax2]:
                # Clean styling similar to create_water_level_plot_png
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(False)
                
                # Set the same x-axis limits for both plots
                ax.set_xlim(start_date, end_date)
                
                # Set major ticks at 2-month intervals for the 2022-2024 period
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
                ax.tick_params(axis='x', rotation=45)
            
            # Style the legends
            ax1.legend(loc='upper right', frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', fontsize=12)
            ax2.legend(loc='lower right', frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', fontsize=12)
            
            # Add main title
            #fig.suptitle(f'Manual vs Automated Water Level Measurements - Station {station_name} (2022-2024)',
            #            fontweight='bold', y=0.95)
            
            # Format the figure for nice display
            fig.autofmt_xdate()
            plt.tight_layout()
            
            # Save with better quality but reasonable rendering speed
            plt.savefig(diagnostic_dir / f"{station_name}_vst_vinge_comparison.png",
                       dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
