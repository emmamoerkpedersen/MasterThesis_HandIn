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
    
    # Define consistent colors for better visualization
    COLORS = {
        'vst_original': '#1f77b4',      # Blue for original water level
        'vst_processed': '#2ca02c',     # Green for processed water level
        'temp_original': '#d62728',     # Red for original temperature
        'temp_processed': '#ff7f0e',    # Orange for processed temperature
        'rain_original': '#1f77b4',     # Blue for original rainfall
        'rain_processed': '#2ca02c',    # Green for processed rainfall
        'outliers': '#d62728',          # Red for outliers
        'bounds': '#ff7f0e',            # Orange for bounds
        'training': '#E8F5E9',          # Light green for training
        'validation': '#FFF3E0',        # Light orange for validation
        'testing': '#E3F2FD',           # Light blue for testing
        'frost': '#BBDEFB'              # Light blue for frost periods
    }
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame to store statistics for all stations
    stats_data = []
    
    # Set start date to January 1, 2010 - make it timezone-aware
    start_date = pd.to_datetime('2010-01-01').tz_localize('UTC')
    
    # Define split dates - make them timezone-aware
    train_end = pd.to_datetime('2022-01-01').tz_localize('UTC')
    val_end = pd.to_datetime('2024-01-01').tz_localize('UTC')
    
    # Try to load frost periods if not provided
    if frost_periods is None or len(frost_periods) == 0:
        try:
            import pickle
            frost_path = Path(__file__).parent.parent / "data_utils" / "Sample data" / "frost_periods.pkl"
            if frost_path.exists():
                with open(frost_path, 'rb') as f:
                    frost_periods = pickle.load(f)
                print(f"Loaded {len(frost_periods)} frost periods from {frost_path}")
            else:
                print(f"No frost periods file found at {frost_path}")
                frost_periods = []
        except Exception as e:
            print(f"Error loading frost periods: {e}")
            frost_periods = []
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with GridSpec - now with 3 rows for water level, temperature, and rainfall
            fig = plt.figure(figsize=(15, 18))
            gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.1)
            
            # Get the original and processed data for all variables
            orig_vst = original_data[station_name]['vst_raw'].copy()
            proc_vst = preprocessed_data[station_name]['vst_raw'].copy()
            orig_temp = original_data[station_name]['temperature'].copy() if 'temperature' in original_data[station_name] else None
            proc_temp = preprocessed_data[station_name]['temperature'].copy() if 'temperature' in preprocessed_data[station_name] else None
            orig_rain = original_data[station_name]['rainfall'].copy() if 'rainfall' in original_data[station_name] else None
            proc_rain = preprocessed_data[station_name]['rainfall'].copy() if 'rainfall' in preprocessed_data[station_name] else None
            
            # Get the value column names
            vst_col = [col for col in orig_vst.columns if col != 'Date'][0]
            temp_col = [col for col in orig_temp.columns if col != 'Date'][0] if orig_temp is not None else None
            rain_col = [col for col in orig_rain.columns if col != 'Date'][0] if orig_rain is not None else None
            
            # For processed data, check if column names have been standardized
            if 'vst_raw' in proc_vst.columns:
                proc_vst_col = 'vst_raw'
            else:
                proc_vst_col = [col for col in proc_vst.columns if col != 'Date'][0]
            
            # Function to ensure datetime index is timezone-aware
            def ensure_tz_aware(df):
                if df is not None:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz.zone != 'UTC':
                        df.index = df.index.tz_convert('UTC')
                return df
            
            # Ensure all DataFrames have timezone-aware datetime index
            orig_vst = ensure_tz_aware(orig_vst)
            proc_vst = ensure_tz_aware(proc_vst)
            orig_temp = ensure_tz_aware(orig_temp)
            proc_temp = ensure_tz_aware(proc_temp)
            orig_rain = ensure_tz_aware(orig_rain)
            proc_rain = ensure_tz_aware(proc_rain)
            
            # Filter data to start from 2010
            orig_vst = orig_vst[orig_vst.index >= start_date]
            proc_vst = proc_vst[proc_vst.index >= start_date]
            if orig_temp is not None:
                orig_temp = orig_temp[orig_temp.index >= start_date]
            if proc_temp is not None:
                proc_temp = proc_temp[proc_temp.index >= start_date]
            if orig_rain is not None:
                orig_rain = orig_rain[orig_rain.index >= start_date]
            if proc_rain is not None:
                proc_rain = proc_rain[proc_rain.index >= start_date]
            
            # PERFORMANCE OPTIMIZATION: Downsample data if too large
            if len(orig_vst) > 10000:
                orig_vst_plot = orig_vst.resample('6H').mean().dropna()
                proc_vst_plot = proc_vst.resample('6H').mean().dropna()
            else:
                orig_vst_plot = orig_vst
                proc_vst_plot = proc_vst
                
            if orig_temp is not None and len(orig_temp) > 10000:
                orig_temp_plot = orig_temp.resample('6H').mean().dropna()
                proc_temp_plot = proc_temp.resample('6H').mean().dropna()
            else:
                orig_temp_plot = orig_temp
                proc_temp_plot = proc_temp
                
            if orig_rain is not None and len(orig_rain) > 10000:
                orig_rain_plot = orig_rain.resample('6H').sum().dropna()
                proc_rain_plot = proc_rain.resample('6H').sum().dropna()
            else:
                orig_rain_plot = orig_rain
                proc_rain_plot = proc_rain
            
            # Calculate IQR bounds using the original data (use full dataset for calculations)
            Q1 = orig_vst[vst_col].quantile(0.25)
            Q3 = orig_vst[vst_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Calculate removed points mask for outliers (only points outside IQR bounds)
            outlier_mask = (orig_vst[vst_col] < lower_bound) | (orig_vst[vst_col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            # Calculate freezing period points if available
            freezing_points = 0
            freezing_mask = pd.Series(False, index=orig_vst.index)
            
            if frost_periods:
                for start, end in frost_periods:
                    # Ensure start and end are timezone-aware
                    if hasattr(start, 'tzinfo') and start.tzinfo is None:
                        start = pd.to_datetime(start).tz_localize('UTC')
                    if hasattr(end, 'tzinfo') and end.tzinfo is None:
                        end = pd.to_datetime(end).tz_localize('UTC')
                    
                    period_mask = (orig_vst.index >= start) & (orig_vst.index <= end)
                    freezing_mask = freezing_mask | period_mask
                freezing_points = freezing_mask.sum()
            
            # Calculate total points removed (actual difference between orig and proc)
            total_points_removed = len(orig_vst) - len(proc_vst)
            
            # Calculate flatline points (not accounted for by outliers or freezing)
            flatline_count = total_points_removed - outlier_count - freezing_points
            
            # Store statistics for this station
            stats_data.append({
                'Station': station_name,
                'Total Points': len(orig_vst),
                'Points Removed': total_points_removed,
                'Removal %': (total_points_removed/len(orig_vst))*100,
                'Outliers': outlier_count,
                'Freezing': freezing_points,
                'Flatlines': flatline_count,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            })
            
            # Function to add split backgrounds to subplot (without adding to legend)
            def add_split_backgrounds(ax):
                # Add colored backgrounds for train/val/test splits without adding to legend
                # Make sure all timestamps are timezone-aware with UTC
                end_date = pd.to_datetime('2025-01-01').tz_localize('UTC')
                ax.axvspan(start_date, train_end, color=COLORS['training'], alpha=0.3, label='_nolegend_')
                ax.axvspan(train_end, val_end, color=COLORS['validation'], alpha=0.3, label='_nolegend_')
                ax.axvspan(val_end, end_date, color=COLORS['testing'], alpha=0.3, label='_nolegend_')
                
                # Add frost periods if available
                if frost_periods:
                    for start, end in frost_periods:
                        # Ensure start and end are timezone-aware
                        if hasattr(start, 'tzinfo') and start.tzinfo is None:
                            start = pd.to_datetime(start).tz_localize('UTC')
                        if hasattr(end, 'tzinfo') and end.tzinfo is None:
                            end = pd.to_datetime(end).tz_localize('UTC')
                            
                        if start >= start_date:  # Only show frost periods after 2010
                            ax.axvspan(start, end, color=COLORS['frost'], alpha=0.5, 
                                     label='_nolegend_', zorder=1, hatch='///')
            
            # Top subplot: Water Level
            ax1 = fig.add_subplot(gs[0])
            add_split_backgrounds(ax1)
            
            # Plot the original and preprocessed water level data
            ax1.plot(orig_vst_plot.index, orig_vst_plot[vst_col], color=COLORS['vst_original'], alpha=0.8, 
                    linewidth=0.8, label='Original VST', zorder=2)
            ax1.plot(proc_vst_plot.index, proc_vst_plot[proc_vst_col], color=COLORS['vst_processed'], alpha=0.8,
                    linewidth=0.8, label='Preprocessed VST', zorder=2)
            
            # Add IQR bounds
            ax1.axhline(y=lower_bound, color=COLORS['bounds'], linestyle='--', alpha=0.6,
                       linewidth=0.8, label='IQR Bounds', zorder=1)
            ax1.axhline(y=upper_bound, color=COLORS['bounds'], linestyle='--', alpha=0.6,
                       linewidth=0.8, zorder=1)
            
            # Find and plot outliers
            outlier_points = orig_vst[outlier_mask].copy()
            if len(outlier_points) > 0:
                if len(outlier_points) > 1000:
                    outlier_sample = outlier_points.sample(n=min(1000, len(outlier_points)), random_state=42)
                    ax1.scatter(outlier_sample.index, outlier_sample[vst_col],
                              color=COLORS['outliers'], s=25, alpha=0.7, label='Removed Points (Sample)', zorder=3)
                else:
                    ax1.scatter(outlier_points.index, outlier_points[vst_col],
                              color=COLORS['outliers'], s=25, alpha=0.7, label='Removed Points', zorder=3)
            
            ax1.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
            ax1.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(False)
            
            # Middle subplot: Temperature
            ax2 = fig.add_subplot(gs[1])
            add_split_backgrounds(ax2)
            
            if orig_temp is not None and proc_temp is not None:
                # Get the temperature column names - be more flexible
                orig_temp_col = temp_col  # Already determined earlier
                
                # For processed data, the column might be renamed to 'temperature'
                if temp_col in proc_temp.columns:
                    proc_temp_col = temp_col
                elif 'temperature' in proc_temp.columns:
                    proc_temp_col = 'temperature'
                else:
                    # Try to find any column that might contain temperature data
                    proc_temp_col = proc_temp.columns[0]
                
                ax2.plot(orig_temp_plot.index, orig_temp_plot[orig_temp_col], color=COLORS['temp_original'], alpha=0.8,
                        linewidth=0.8, label='Original Temperature', zorder=2)
                ax2.plot(proc_temp_plot.index, proc_temp_plot[proc_temp_col], color=COLORS['temp_processed'], alpha=0.8,
                        linewidth=0.8, label='Preprocessed Temperature', zorder=2)
                
                # Add freezing threshold line at 0°C
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8, label='Freezing Point (0°C)', zorder=1)
                
                ax2.set_ylabel('Temperature (°C)', fontweight='bold', labelpad=10)
                ax2.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            else:
                ax2.text(0.5, 0.5, 'No temperature data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=14)
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(False)
            
            # Bottom subplot: Rainfall
            ax3 = fig.add_subplot(gs[2])
            add_split_backgrounds(ax3)
            
            if orig_rain is not None and proc_rain is not None:
                # Get the rainfall column names - be more flexible
                orig_rain_col = rain_col  # Already determined earlier
                
                # For processed data, the column might be renamed to 'rainfall'
                if rain_col in proc_rain.columns:
                    proc_rain_col = rain_col
                elif 'rainfall' in proc_rain.columns:
                    proc_rain_col = 'rainfall'
                else:
                    # Try to find any column that might contain rainfall data
                    proc_rain_col = proc_rain.columns[0]
                
                # Plot rainfall as bars
                bar_width = 1  # Width of bars in days
                ax3.bar(orig_rain_plot.index, orig_rain_plot[orig_rain_col], width=bar_width,
                       color=COLORS['rain_original'], alpha=0.5, label='Original Rainfall', zorder=2)
                ax3.bar(proc_rain_plot.index, proc_rain_plot[proc_rain_col], width=bar_width,
                       color=COLORS['rain_processed'], alpha=0.5, label='Preprocessed Rainfall', zorder=2)
                ax3.set_ylabel('Rainfall (mm)', fontweight='bold', labelpad=10)
                ax3.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            else:
                ax3.text(0.5, 0.5, 'No rainfall data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=14)
            
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.grid(False)
            ax3.set_xlabel('Date', fontweight='bold', labelpad=10)
            
            # Set consistent x-axis limits and format for all subplots
            x_min = start_date
            x_max = pd.to_datetime('2025-01-01').tz_localize('UTC')  # Use timezone-aware date
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(x_min, x_max)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
                ax.tick_params(axis='x', rotation=45)
            
            # Add a single legend for the data splits at the bottom of the figure
            split_legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['training'], alpha=0.3, label='Training'),
                plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['validation'], alpha=0.3, label='Validation'),
                plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['testing'], alpha=0.3, label='Test'),
                plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['frost'], alpha=0.5, label='Frost Period', hatch='///')
            ]
            fig.legend(handles=split_legend_elements, loc='lower center', ncol=4, frameon=True, 
                      framealpha=0.9, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.01))
            
            # Format the figure for nice display
            plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust bottom to make room for the split legend
            
            # Save the figure
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
    
    # Define consistent colors for better visualization
    COLORS = {
        'vst_original': '#1f77b4',      # Blue for water level
        'temp_original': '#d62728',     # Red for temperature
        'rain_original': '#1f77b4',     # Blue for rainfall
        'vinge': '#d62728',             # Red for VINGE measurements
    }
    
    # Define start date for precipitation data - make it timezone-aware
    precip_start_date = pd.to_datetime('2010-01-01').tz_localize('UTC')
    
    # Function to ensure datetime index is timezone-aware
    def ensure_tz_aware(df):
        if df is not None:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz.zone != 'UTC':
                df.index = df.index.tz_convert('UTC')
        return df
    
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
                temp_data = ensure_tz_aware(temp_data)
                
                # Get temperature column name
                temp_col = [col for col in temp_data.columns if col != 'Date'][0]
                
                # PERFORMANCE OPTIMIZATION: Downsample temperature data if needed
                if len(temp_data) > 10000:
                    temp_data = temp_data.resample('6H').mean().dropna()
                
                # Track date range
                min_dates['temperature'] = temp_data.index.min()
                max_dates['temperature'] = temp_data.index.max()
                
                ax1.plot(temp_data.index, temp_data[temp_col],
                        color=COLORS['temp_original'], alpha=0.8, linewidth=0.8, label='Temperature')
                
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
                rain_data = ensure_tz_aware(rain_data)
                
                # Filter by start date - now both are timezone-aware
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
                           color=COLORS['rain_original'], alpha=0.7, width=1, label='Rainfall')
                    
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
                vinge_data = ensure_tz_aware(vinge_data)
                
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
                
                if vinge_col is not None:
                    # Remove NaN values
                    vinge_data = vinge_data.dropna(subset=['water_level_mm'])
                    
                    # Track date range
                    if not vinge_data.empty:
                        min_dates['vinge'] = vinge_data.index.min()
                        max_dates['vinge'] = vinge_data.index.max()
                        
                        # Plot as scatter (don't connect points)
                        ax3.scatter(vinge_data.index, vinge_data['water_level_mm'],
                                color=COLORS['vinge'], alpha=0.8, s=20, 
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
                vst_data = ensure_tz_aware(vst_data)
                
                # Get the VST column name
                vst_col = [col for col in vst_data.columns if col != 'Date'][0]
                
                # PERFORMANCE OPTIMIZATION: Downsample data if too large
                if len(vst_data) > 10000:
                    vst_data = vst_data.resample('6H').mean().dropna()
                
                # Track date range
                min_dates['vst_raw'] = vst_data.index.min()
                max_dates['vst_raw'] = vst_data.index.max()
                
                ax4.plot(vst_data.index, vst_data[vst_col],
                        color=COLORS['vst_original'], alpha=0.8, linewidth=0.8, label='VST Raw')
                
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
    
    # Define consistent colors for better visualization
    COLORS = {
        'vst_raw': '#1f77b4',         # Blue for VST Raw
        'vst_edt': '#2ca02c',         # Green for VST EDT
        'vinge': '#d62728',           # Red for VINGE measurements
        'difference': '#1f77b4',      # Blue for difference
        'threshold': '#d62728',       # Red for threshold
    }
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to ensure datetime index is timezone-aware
    def ensure_tz_aware(df):
        if df is not None:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz.zone != 'UTC':
                df.index = df.index.tz_convert('UTC')
        return df
    
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
            
            # Apply timezone handling
            vst_raw = ensure_tz_aware(vst_raw)
            vst_edt = ensure_tz_aware(vst_edt)
            vinge_data = ensure_tz_aware(vinge_data)
            
            # Print information about the VINGE data
            print(f"VINGE data shape before processing: {vinge_data.shape}")
            print(f"VINGE data columns:", vinge_data.columns)
            print(f"First few rows of VINGE data:\n{vinge_data.head()}")
            
            # Focus on 2022-01-01 to 2024-01-01
            start_date = pd.to_datetime('2022-01-01').tz_localize('UTC')
            end_date = pd.to_datetime('2024-01-01').tz_localize('UTC')
            
            print(f"Date range: {start_date} to {end_date}")
            
            # Filter data for specified date range
            vst_raw_filtered = vst_raw[(vst_raw.index >= start_date) & (vst_raw.index <= end_date)]
            vst_edt_filtered = vst_edt[(vst_edt.index >= start_date) & (vst_edt.index <= end_date)]
            vinge_filtered = vinge_data[(vinge_data.index >= start_date) & (vinge_data.index <= end_date)]
            
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
                    color=COLORS['vst_raw'], alpha=0.8, linewidth=0.8, 
                    label='VST Raw')
            
            # Plot EDT corrected data
            ax1.plot(vst_edt_plot.index, vst_edt_plot[vst_edt_col],
                    color=COLORS['vst_edt'], alpha=0.8, linewidth=0.8, 
                    label='VST EDT')
            
            # Plot VINGE measurements with larger markers and higher zorder
            ax1.scatter(vinge_filtered.index, vinge_filtered['water_level_mm'],
                       color=COLORS['vinge'], alpha=0.8, s=20, 
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
                
                # Make sure window dates have the same timezone as the data
                if window_start.tz is None:
                    window_start = window_start.tz_localize('UTC')
                if window_end.tz is None:
                    window_end = window_end.tz_localize('UTC')
                    
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
                ax2.scatter(vinge_dates, differences, color=COLORS['difference'], alpha=0.8, s=40,
                           label='VINGE - VST_RAW difference')
            
            # Add horizontal line at 0 for reference
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add threshold lines at ±20mm
            ax2.axhline(y=20, color=COLORS['threshold'], linestyle='--', alpha=0.8,
                       label='Correction Threshold (±20mm)')
            ax2.axhline(y=-20, color=COLORS['threshold'], linestyle='--', alpha=0.8)
            
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
