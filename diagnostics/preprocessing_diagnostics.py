"""Diagnostic tools for data preprocessing step."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm # Added for colormaps



def set_plot_style():
    """Set a consistent, professional plot style for all visualizations suitable for a thesis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set the layout engine explicitly to avoid warnings
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = False
    
    # Set font family and sizes (increased for thesis)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 20,
        'axes.titlesize': 1,  # No titles
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'figure.titlesize': 1
    })
    
    # Define consistent colors for the thesis
    thesis_colors = [
        '#1f77b4',  # Blue
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=thesis_colors)
    
    # Remove grid completely for cleaner plots
    plt.rcParams['axes.grid'] = False
    
    # Set figure background to pure white
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Style spines
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Configure tick parameters
    plt.rcParams['xtick.major.size'] = 7.0
    plt.rcParams['ytick.major.size'] = 7.0
    plt.rcParams['xtick.minor.size'] = 4.0
    plt.rcParams['ytick.minor.size'] = 4.0
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.minor.width'] = 1.0
    plt.rcParams['ytick.minor.width'] = 1.0
    
    # Legend settings
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#cccccc'
    plt.rcParams['legend.fancybox'] = True
    
    # Figure quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def plot_preprocessing_comparison(original_data: dict, preprocessed_data: dict, output_dir: Path, frost_periods: list = None):
    """Create comparison plots between original and preprocessed data for each station."""
    # Set professional plot style
    set_plot_style()
    
    # Define consistent colors for thesis-worthy visualization
    COLORS = {
        'vst_original': '#1f77b4',   # Blue for original water level
        'vst_processed': '#2ca02c',  # Green for processed water level
        'outliers': '#d62728',       # Red for outliers
        'bounds': '#ff7f0e',         # Orange for bounds
        'training': '#C8E6C9',       # Light green for training
        'validation': '#FFE0B2',     # Light orange for validation
        'testing': '#BBDEFB',        # Light blue for testing
        'frost': '#90CAF9'           # Light blue for frost periods
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
            
            # Create figure for water level only (removed temperature and rainfall plots)
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            # Get the original and processed data for water level
            orig_vst = original_data[station_name]['vst_raw'].copy()
            proc_vst = preprocessed_data[station_name]['vst_raw'].copy()
            
            # Get the value column names
            vst_col = [col for col in orig_vst.columns if col != 'Date'][0]
            
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
                    elif hasattr(df.index.tz, 'zone') and df.index.tz.zone != 'UTC':
                        df.index = df.index.tz_convert('UTC')
                    elif str(df.index.tz) != 'UTC':
                        # Handle datetime.timezone objects without zone attribute
                        df.index = df.index.tz_convert('UTC')
                return df
            
            # Ensure all DataFrames have timezone-aware datetime index
            orig_vst = ensure_tz_aware(orig_vst)
            proc_vst = ensure_tz_aware(proc_vst)
            
            # Filter data to start from 2010
            orig_vst = orig_vst[orig_vst.index >= start_date]
            proc_vst = proc_vst[proc_vst.index >= start_date]
            
            # PERFORMANCE OPTIMIZATION: Downsample data if too large
            if len(orig_vst) > 10000:
                orig_vst_plot = orig_vst.resample('6H').mean().dropna()
            else:
                orig_vst_plot = orig_vst
            
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
            
            # Plot the original VST data
            ax1.plot(orig_vst_plot.index, orig_vst_plot[vst_col], color=COLORS['vst_original'], alpha=0.8, 
                    linewidth=1.2, label='Original VST', zorder=2)
            
            # Add IQR bounds
            ax1.axhline(y=lower_bound, color=COLORS['bounds'], linestyle='--', alpha=0.6,
                       linewidth=1.0, label='IQR Bounds', zorder=1)
            ax1.axhline(y=upper_bound, color=COLORS['bounds'], linestyle='--', alpha=0.6,
                       linewidth=1.0, zorder=1)
            
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
            
            # Add frost periods if available
            if frost_periods:
                frost_added_to_legend = False
                for start, end in frost_periods:
                    # Ensure start and end are timezone-aware
                    if hasattr(start, 'tzinfo') and start.tzinfo is None:
                        start = pd.to_datetime(start).tz_localize('UTC')
                    if hasattr(end, 'tzinfo') and end.tzinfo is None:
                        end = pd.to_datetime(end).tz_localize('UTC')
                        
                    if start >= start_date:  # Only show frost periods after start date
                        if not frost_added_to_legend:
                            ax1.axvspan(start, end, color=COLORS['frost'], alpha=0.3, 
                                     label='Frost Period', zorder=1, hatch='///')
                            frost_added_to_legend = True
                        else:
                            ax1.axvspan(start, end, color=COLORS['frost'], alpha=0.3, 
                                     label='_nolegend_', zorder=1, hatch='///')
            
            ax1.set_ylabel('Water Level (mm)', fontsize=24, fontweight='bold', labelpad=30)
            # Position legend in the upper right corner to avoid data
            ax1.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
            
            # Remove grid lines and spines for cleaner look
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_xlabel('Date', fontsize=24, fontweight='bold', labelpad=10)
            
            # Set consistent x-axis limits and format
            x_min = start_date
            x_max = pd.to_datetime('2025-01-01').tz_localize('UTC')  # Use timezone-aware date
            ax1.set_xlim(x_min, x_max)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax1.xaxis.set_major_locator(mdates.YearLocator(1))
            ax1.tick_params(axis='x', rotation=45)
            
            # Format the figure with tight layout
            plt.tight_layout()
            
            # Save the figure with higher DPI for thesis quality
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Create a separate plot for the data splits
            plot_data_splits(preprocessed_data[station_name], station_name, diagnostic_dir, 
                            start_date, train_end, val_end)
    
    # Create and save statistics table
    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
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
            f.write("Flatlines represent sequences of 30 or more identical consecutive values.\n")
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

def plot_data_splits(station_data, station_name, output_dir, start_date, train_end, val_end):
    """Create plot showing data splits (training/validation/testing) and environmental variables."""
    # Use the consistent plot style
    set_plot_style()
    
    # Create figure with GridSpec for better subplot control
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.2)
    
    # Get the data
    vst_data = station_data['vst_raw'].copy()
    temp_data = station_data['temperature'].copy() if 'temperature' in station_data else None
    rain_data = station_data['rainfall'].copy() if 'rainfall' in station_data else None
    
    # Function to ensure datetime index is timezone-aware
    def ensure_tz_aware(df):
        if df is not None:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif hasattr(df.index.tz, 'zone') and df.index.tz.zone != 'UTC':
                df.index = df.index.tz_convert('UTC')
            elif str(df.index.tz) != 'UTC':
                # Handle datetime.timezone objects without zone attribute
                df.index = df.index.tz_convert('UTC')
        return df
    
    # Ensure DataFrames have timezone-aware datetime index
    vst_data = ensure_tz_aware(vst_data)
    temp_data = ensure_tz_aware(temp_data)
    rain_data = ensure_tz_aware(rain_data)
    
    # Filter data to start from specified date
    vst_data = vst_data[vst_data.index >= start_date]
    if temp_data is not None:
        temp_data = temp_data[temp_data.index >= start_date]
    if rain_data is not None:
        rain_data = rain_data[rain_data.index >= start_date]
    
    # PERFORMANCE OPTIMIZATION: Downsample data if too large
    if len(vst_data) > 10000:
        vst_plot = vst_data.resample('6H').mean().dropna()
    else:
        vst_plot = vst_data
    
    if temp_data is not None and len(temp_data) > 10000:
        temp_plot = temp_data.resample('6H').mean().dropna()
    else:
        temp_plot = temp_data
    
    if rain_data is not None:
        rain_plot = rain_data.copy()
        
        # Resample to 6H to match preprocessing_comparison method
        if len(rain_plot) > 10000:
            rain_plot = rain_plot.resample('6H').sum().dropna()
    else:
        rain_plot = rain_data
    
    # Get the VST column name
    if 'vst_raw' in vst_plot.columns:
        vst_col = 'vst_raw'
    else:
        vst_col = [col for col in vst_plot.columns if col != 'Date'][0]
    
    # Define colors for consistent thesis appearance
    COLORS = {
        'water_level': '#1f77b4',     # Blue for water level
        'temperature': '#d62728',     # Red for temperature
        'rainfall': '#1f77b4',        # Blue for rainfall (matching preprocessing_comparison)
        'training': '#C8E6C9',        # Light green for training
        'validation': '#FFE0B2',      # Light orange for validation
        'testing': '#BBDEFB'          # Light blue for testing
    }
    
    # Set date range
    end_date = pd.to_datetime('2025-01-01').tz_localize('UTC')
    
    # 1. Water Level Plot (top subplot)
    ax1 = fig.add_subplot(gs[0])
    
    # Add background colors for different periods with appropriate alpha
    # Add them from back to front (testing, validation, training)
    ax1.axvspan(val_end, end_date, color=COLORS['testing'], alpha=0.3, label='Testing')
    ax1.axvspan(train_end, val_end, color=COLORS['validation'], alpha=0.3, label='Validation')
    ax1.axvspan(start_date, train_end, color=COLORS['training'], alpha=0.3, label='Training')
    
    # Plot the water level data
    ax1.plot(vst_plot.index, vst_plot[vst_col], color=COLORS['water_level'], 
           linewidth=1.5, label='Water Level', zorder=5)
    
    # Customize the water level plot
    ax1.set_ylabel('Water Level (mm)', fontsize=24, fontweight='bold', labelpad=30)
    
    # Remove top and right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Set x-axis limits
    ax1.set_xlim(start_date, end_date)
    
    # Improve y-axis scaling with buffer
    y_min, y_max = vst_plot[vst_col].min(), vst_plot[vst_col].max()
    y_buffer = (y_max - y_min) * 0.1  # Add 10% buffer
    ax1.set_ylim(y_min - y_buffer, y_max + y_buffer)
    
    # Add legend to the first plot
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, 
             edgecolor='#cccccc', fontsize=12)
    
    # 2. Temperature Plot (middle subplot)
    ax2 = fig.add_subplot(gs[1])
    
    # Add the same background colors for splits (without adding to legend)
    ax2.axvspan(val_end, end_date, color=COLORS['testing'], alpha=0.3, label='_nolegend_')
    ax2.axvspan(train_end, val_end, color=COLORS['validation'], alpha=0.3, label='_nolegend_')
    ax2.axvspan(start_date, train_end, color=COLORS['training'], alpha=0.3, label='_nolegend_')
    
    # Plot temperature if available
    if temp_plot is not None and not temp_plot.empty:
        # Get temperature column name
        temp_col = 'temperature' if 'temperature' in temp_plot.columns else temp_plot.columns[0]
        
        ax2.plot(temp_plot.index, temp_plot[temp_col], color=COLORS['temperature'], 
                linewidth=1.5, label='Temperature', zorder=5)
        
        # Customize the temperature plot
        ax2.set_ylabel('Temperature (°C)', fontsize=24, fontweight='bold', labelpad=30)
        
        # Improve y-axis scaling with buffer
        t_min, t_max = temp_plot[temp_col].min(), temp_plot[temp_col].max()
        t_buffer = (t_max - t_min) * 0.1  # Add 10% buffer
        ax2.set_ylim(t_min - t_buffer, t_max + t_buffer)
    else:
        ax2.text(0.5, 0.5, 'No temperature data available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
    
    # Remove top and right spines for cleaner look
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Set x-axis limits
    ax2.set_xlim(start_date, end_date)
    
    # 3. Rainfall Plot (bottom subplot)
    ax3 = fig.add_subplot(gs[2])
    
    # Add the same background colors for splits (without adding to legend)
    ax3.axvspan(val_end, end_date, color=COLORS['testing'], alpha=0.3, label='_nolegend_')
    ax3.axvspan(train_end, val_end, color=COLORS['validation'], alpha=0.3, label='_nolegend_')
    ax3.axvspan(start_date, train_end, color=COLORS['training'], alpha=0.3, label='_nolegend_')
    
    # Plot rainfall if available
    if rain_plot is not None and not rain_plot.empty:
        # Get rainfall column name
        rain_col = 'rainfall' if 'rainfall' in rain_plot.columns else rain_plot.columns[0]
        
        # Replace negative values with 0 for rainfall (they're likely fill values)
        rain_plot[rain_col] = rain_plot[rain_col].clip(lower=0)
        
        # Resample to 6H to match exactly what's in plot_preprocessing_comparison
        rain_plot = rain_plot.resample('6H').sum().dropna()
        
        # Plot rainfall as bars with exact same parameters as in plot_preprocessing_comparison
        bar_width = 4
        ax3.bar(rain_plot.index, rain_plot[rain_col], width=bar_width,
               color='#1f77b4', alpha=0.85, label='Rainfall', zorder=2, linewidth=0.2)
        
        # Customize the rainfall plot
        ax3.set_ylabel('Rainfall (mm)', fontsize=24, fontweight='bold', labelpad=30)
    else:
        ax3.text(0.5, 0.5, 'No rainfall data available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=14)
    
    # Remove top and right spines for cleaner look
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Set x-axis limits
    ax3.set_xlim(start_date, end_date)
    
    # Add x-axis label to the bottom subplot only
    ax3.set_xlabel('Date', fontsize=24, fontweight='bold', labelpad=10)
    
    # Format x-axis with clean ticks on all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        # Make rotated tick labels aligned properly
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high resolution for thesis
    output_path = output_dir / f"data_splits_{station_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_station_data_overview(original_data: dict, preprocessed_data: dict, output_dir: Path):
    """Create visualization of temperature, rainfall, and VST data showing full available date ranges."""
    # Apply consistent plot styling
    set_plot_style()
    
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Define consistent colors for better visualization that match thesis style
    COLORS = {
        'vst_original': '#1f77b4',      # Blue for water level
        'temp_original': '#d62728',     # Red for temperature
        'rain_original': '#7090FF',     # Lighter blue for rainfall
        'vinge': '#ff7f0e',             # Orange for VINGE measurements to contrast with water level
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
            elif hasattr(df.index.tz, 'zone') and df.index.tz.zone != 'UTC':
                df.index = df.index.tz_convert('UTC')
            elif str(df.index.tz) != 'UTC':
                # Handle datetime.timezone objects without zone attribute
                df.index = df.index.tz_convert('UTC')
        return df
    
    for station_name in original_data.keys():
        if original_data[station_name]['vst_raw'] is not None:
            # Create figure with GridSpec for better control of subplot heights - now with 4 rows
            fig = plt.figure(figsize=(15, 16))  # Increased figure height for better spacing
            gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 3], hspace=0.3)
            
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
                
                # Plot temperature with thicker line
                ax1.plot(temp_data.index, temp_data[temp_col],
                        color=COLORS['temp_original'], alpha=0.9, linewidth=1.2, label='Temperature')
                
                ax1.set_ylabel('Temperature (°C)', fontsize=18, fontweight='bold', labelpad=30)
                ax1.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
                
                # Clean style similar to water_level_plot
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                
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
                
                # Replace negative values with 0 for rainfall (they're likely fill values)
                rain_data_filtered[rain_col] = rain_data_filtered[rain_col].clip(lower=0)
                
                # PERFORMANCE OPTIMIZATION: For rainfall, resample to daily sum for better visualization
                if len(rain_data_filtered) > 1000:
                    rain_data_plot = rain_data_filtered.resample('6H').sum().dropna()
                else:
                    rain_data_plot = rain_data_filtered
                
                # Track date range of filtered data
                if not rain_data_filtered.empty:
                    min_dates['rainfall'] = rain_data_filtered.index.min()
                    max_dates['rainfall'] = rain_data_filtered.index.max()
                    
                    # Plot rainfall as bars with alpha for better appearance
                    ax2.bar(rain_data_plot.index, rain_data_plot[rain_col],
                           color='#1f77b4', alpha=0.85, width=2, label='Rainfall', linewidth=0.2)
                    
                    ax2.set_ylabel('Precipitation (mm)', fontsize=18, fontweight='bold', labelpad=30)
                    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
                    
                    # Clean style
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    
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
                        
                        # Plot as scatter with larger marker size
                        ax3.scatter(vinge_data.index, vinge_data['water_level_mm'],
                                   color='#d62728', s=30, 
                                   label='VINGE', zorder=5, edgecolors='white', linewidth=0.5)
                        
                        ax3.set_ylabel('Water Level (mm)', fontsize=18, fontweight='bold', labelpad=30)
                        ax3.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
                        
                        # Clean style
                        ax3.spines['top'].set_visible(False)
                        ax3.spines['right'].set_visible(False)
                        
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
                
                # Plot water level with thicker line
                ax4.plot(vst_data.index, vst_data[vst_col],
                        color=COLORS['vst_original'], alpha=0.9, linewidth=1.2, label='Water Level')
                
                ax4.set_ylabel('Water Level (mm)', fontsize=18, fontweight='bold', labelpad=30)
                ax4.set_xlabel('Date', fontsize=24, fontweight='bold', labelpad=10)
                ax4.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
                
                # Clean style
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
                
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
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        ax.xaxis.set_major_locator(mdates.YearLocator())  # Every year
                    else:  # Less than 2 years
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
                    
                    # Make tick labels more readable with proper rotation and alignment
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # Make sure x-axis ticks and labels are shown for each subplot
                    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            
            # Generate data availability text for title
            availability_text = []
            for data_type in ['temperature', 'rainfall', 'vinge', 'vst_raw']:
                if data_type in min_dates:
                    start_year = min_dates[data_type].year
                    end_year = max_dates[data_type].year
                    availability_text.append(f"{data_type.capitalize()}: {start_year}-{end_year}")
            
            # Format the figure for nice display
            plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave room for the titles
            
            # Save with high resolution for thesis quality
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            # In plot_station_data_overview, after plt.tight_layout(rect=[0, 0, 1, 0.92]), add fig.subplots_adjust(left=0.18)
            fig.subplots_adjust(left=0.18)

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
        'font.size': 16,  # was 12
        'axes.titlesize': 21,  # was 16
        'axes.labelsize': 18,  # was 14
        'xtick.labelsize': 16,  # was 12
        'ytick.labelsize': 16,  # was 12
        'legend.fontsize': 16,  # was 12
        'figure.titlesize': 23  # was 18
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
            elif hasattr(df.index.tz, 'zone') and df.index.tz.zone != 'UTC':
                df.index = df.index.tz_convert('UTC')
            elif str(df.index.tz) != 'UTC':
                # Handle datetime.timezone objects without zone attribute
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
                       color='#d62728', alpha=0.8, s=20, 
                       label='VINGE', zorder=5)
            
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
            ax1.set_ylabel('Water Level (mm)', fontsize=24, fontweight='bold', labelpad=30)
            
            ax2.set_ylabel('Difference (mm)', fontsize=24, fontweight='bold', labelpad=30)
            ax2.set_xlabel('Date', fontsize=24, fontweight='bold', labelpad=10)
            
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
            
            # Format the figure for nice display
            fig.autofmt_xdate()
            plt.tight_layout()
            
            # Save with better quality but reasonable rendering speed
            plt.savefig(diagnostic_dir / f"{station_name}_vst_vinge_comparison.png",
                       dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

def get_time_windows():
    """Define time windows for anomaly analysis for a single station based on actual_anomalies.
    
    Returns:
        list: A list of dictionaries, each defining a time window for analysis.
              Dates are timezone-aware (UTC).
    """
    actual_anomalies = {
        'spike': {
            'period': ('2022-11-20', '2022-11-26'), 
            'source_station': '21006846'
        },
        'drift': {
            'period': ('2022-03-01 00:00:00', '2022-06-01 00:00:00'),
            'source_station': '21006846'
        },
        'offset': {
            'period': ('2008-09-01', '2008-10-15'),
            'source_station': '21006846'
        },
        'baseline shift': {
            'period': ('2021-07-05', '2021-07-13'),
            'source_station': '21006846'
        },
        'missing_data': { 
            'period': ('2007-03-01', '2008-01-17'), 
            'source_station': '21006846' 
        },
        'noise': { 
            'period': ('2009-06-03', '2009-06-12'),
            'source_station': '21006847'
        }
    }

    processed_anomalies = []
    for error_type, details in actual_anomalies.items():
        if error_type == 'noise':
            continue
        start_date_str, end_date_str = details['period']
        processed_anomalies.append({
            'error_type': error_type,
            'start_date_str': start_date_str,
            'end_date_str': end_date_str,
            'details': details
        })

    # Sort anomalies by start date
    processed_anomalies.sort(key=lambda x: pd.to_datetime(x['start_date_str']))

    time_windows = []
    for anomaly in processed_anomalies:
        error_type = anomaly['error_type']
        start_date_str = anomaly['start_date_str']
        end_date_str = anomaly['end_date_str']
        
        title = f"{error_type.replace('_', ' ').title()}"
        if error_type == 'baseline shift': 
            title = "Baseline Shift"
        elif error_type == 'missing_data': 
            title = "Missing Data"

        time_windows.append({
            "title": title,
            "start_date": pd.to_datetime(start_date_str).tz_localize('UTC'),
            "end_date": pd.to_datetime(end_date_str).tz_localize('UTC'),
            "y_range": None
        })
    
    return time_windows

def create_detailed_plot(station_data: dict, time_windows: list, station_id: str, output_dir: Path):
    """
    Create the main detailed plot with subplots showing anomaly examples for a single station.
    Adapted from Oliver/plotting/error_plots.py and integrated with thesis styling.

    Args:
        station_data: Dictionary with data for the station, expecting 'vst_raw' and optionally 'vinge' DataFrames.
                      DataFrames should have a DateTimeIndex or a 'Date' column.
        time_windows: List of time window dictionaries from get_time_windows().
        station_id: Identifier for the station, used in title and filename.
        output_dir: Path object for the base output directory.

    Returns:
        Path: The path to the saved plot file, or None if plotting failed.
    """
    set_plot_style()

    # Define 5 distinct colors from Turbo colormap for subplots
    # Using linspace from 0.1 to 0.9 to avoid the very dark/light ends for better visibility
    turbo_subplot_colors = plt.cm.turbo(np.linspace(0.1, 0.9, 5))

    # Prepare VST data
    if station_data.get('vst_raw') is None or station_data['vst_raw'].empty:
        print(f"Missing or empty VST_RAW data for detailed plot for station {station_id}")
        return None
    
    vst_df = station_data['vst_raw'].copy()
    if not isinstance(vst_df.index, pd.DatetimeIndex):
        if 'Date' in vst_df.columns:
            try:
                vst_df['Date'] = pd.to_datetime(vst_df['Date'])
                vst_df = vst_df.set_index('Date')
            except Exception as e:
                print(f"Could not set 'Date' column as index for VST data: {e}")
                return None
        else: # Try to use the first datetime-like column as index or fail
            print(f"VST_RAW data for station {station_id} does not have a DatetimeIndex or 'Date' column.")
            return None
            
    if vst_df.index.tz is None:
        vst_df.index = vst_df.index.tz_localize('UTC')
    elif vst_df.index.tz.zone != 'UTC':
        vst_df.index = vst_df.index.tz_convert('UTC')

    vst_val_col = 'Value' # Default from error_plots.py
    if vst_val_col not in vst_df.columns:
        numeric_cols = [col for col in vst_df.columns if pd.api.types.is_numeric_dtype(vst_df[col])]
        if not numeric_cols:
            print(f"No numeric value column found for VST_RAW data for station {station_id}")
            return None
        vst_val_col = numeric_cols[0]
        print(f"Using column '{vst_val_col}' for VST_RAW values for station {station_id}.")

    # Prepare VINGE data (optional)
    vinge_df = None
    vinge_val_col = None
    if station_data.get('vinge') is not None and not station_data['vinge'].empty:
        vinge_df = station_data['vinge'].copy()
        if not isinstance(vinge_df.index, pd.DatetimeIndex):
            if 'Date' in vinge_df.columns:
                try:
                    vinge_df['Date'] = pd.to_datetime(vinge_df['Date'])
                    vinge_df = vinge_df.set_index('Date')
                except Exception as e:
                    print(f"Could not set 'Date' column as index for VINGE data: {e}")
                    vinge_df = None # Continue without VINGE
            # else: assume index is datetime if 'Date' column not present and not DatetimeIndex
        
        if vinge_df is not None and isinstance(vinge_df.index, pd.DatetimeIndex):
            if vinge_df.index.tz is None:
                vinge_df.index = vinge_df.index.tz_localize('UTC')
            elif vinge_df.index.tz.zone != 'UTC':
                vinge_df.index = vinge_df.index.tz_convert('UTC')

            vinge_val_col_options = ['W.L [cm]', 'vinge', 'Value']
            for col_option in vinge_val_col_options:
                if col_option in vinge_df.columns and pd.api.types.is_numeric_dtype(vinge_df[col_option]):
                    vinge_val_col = col_option
                    break
            
            if vinge_val_col is None: # Fallback to first numeric
                numeric_cols_vinge = [col for col in vinge_df.columns if pd.api.types.is_numeric_dtype(vinge_df[col])]
                if numeric_cols_vinge:
                    vinge_val_col = numeric_cols_vinge[0]
                else:
                    print(f"No suitable numeric value column found for VINGE data for station {station_id}.")
                    vinge_df = None 
            if vinge_df is not None and vinge_val_col:
                 print(f"Using column '{vinge_val_col}' for VINGE values for station {station_id}.")
        else: # If index could not be set or is not DatetimeIndex
            vinge_df = None


    num_windows = len(time_windows)
    if num_windows == 0:
        print(f"No time windows provided for station {station_id}. Skipping detailed plot.")
        return None

    fig = plt.figure(figsize=(max(24, num_windows * 4), 12)) # Adjust width based on num_windows
    gs = fig.add_gridspec(2, num_windows, height_ratios=[2, 1], hspace=0.4, wspace=0.3) # Increased hspace

    prop_cycler = plt.rcParams['axes.prop_cycle']
    thesis_colors = prop_cycler.by_key()['color']

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(vst_df.index, vst_df[vst_val_col], 
                color=thesis_colors[0], label='Sensor Water Level', linewidth=1.2)
    
    ax_main.legend(fontsize=plt.rcParams['legend.fontsize'], loc='upper right') # Use styled fontsize

    locator = mdates.AutoDateLocator(maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_main.xaxis.set_major_locator(locator)
    ax_main.xaxis.set_major_formatter(formatter)
    ax_main.tick_params(axis='x', rotation=45)
    ax_main.set_ylabel('Water level (mm)') # Fontsize and weight from style

    for i in range(num_windows):
        window = time_windows[i]
        start_date = window["start_date"] # Assumed tz-aware from get_time_windows
        end_date = window["end_date"]   # Assumed tz-aware
        title = window["title"]
        y_range_cfg = window["y_range"]
        
        ax = fig.add_subplot(gs[1, i])
        
        mask_vst = (vst_df.index >= start_date) & (vst_df.index <= end_date)
        window_vst_data = vst_df[mask_vst]
        
        vinge_plotted_on_subplot = False # Flag to track if VINGE is plotted for legend
        if window["title"] == "Drift": # Conditional plotting for VINGE data
            if vinge_df is not None and not vinge_df.empty and vinge_val_col in vinge_df.columns:
                mask_vinge = (vinge_df.index >= start_date) & (vinge_df.index <= end_date)
                board_window_data = vinge_df[mask_vinge]
                if not board_window_data.empty:
                     ax.scatter(board_window_data.index, board_window_data[vinge_val_col], 
                               color=thesis_colors[1], s=40, alpha=0.9, 
                               label='VINGE', # Add label for legend
                               edgecolors='white', linewidth=0.5, zorder=5)
                     vinge_plotted_on_subplot = True
        
        if not window_vst_data.empty:
            ax.plot(window_vst_data.index, window_vst_data[vst_val_col], 
                   color=turbo_subplot_colors[i % 5], # Use Turbo colors
                   linewidth=1.2, alpha=0.9)
            if y_range_cfg is not None:
                ax.set_ylim(y_range_cfg[0], y_range_cfg[1])
        else:
             ax.text(0.5, 0.5, 'No VST data in window', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_xlim(start_date, end_date)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) # Ensure nbins is not too large for small windows
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        ax.set_title(title, fontsize=plt.rcParams['axes.labelsize']) # Use styled fontsize
        ax.tick_params(axis='x', labelsize=16, rotation=30) # Increased size and added rotation
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor') # Set horizontal alignment for rotated labels
            
        if i == 0:
            ax.set_ylabel('Water level (mm)')
        
        if vinge_plotted_on_subplot: # Add legend if VINGE was plotted
            ax.legend(fontsize=plt.rcParams['legend.fontsize'] * 0.8, loc='upper right') # Smaller legend for subplot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leave space for suptitle and bottom labels

    for i in range(num_windows):
        ax_sub = fig.axes[i+1] 
        window = time_windows[i]
        start_date = window["start_date"]
        end_date = window["end_date"]
        y_range_cfg = window["y_range"]

        mask_vst = (vst_df.index >= start_date) & (vst_df.index <= end_date)
        window_vst_data = vst_df[mask_vst]
        
        valid_vst_values = window_vst_data[vst_val_col].dropna()
        
        y_min_eff, y_max_eff = np.nan, np.nan # Effective y_min/y_max for rectangle calculation

        if not valid_vst_values.empty:
            y_min_data_for_rect = valid_vst_values.min()
            y_max_data_for_rect = valid_vst_values.max()

            y_min_eff = y_min_data_for_rect
            y_max_eff = y_max_data_for_rect

            if title in ["Offset", "Baseline Shift", "Spike"]:
                data_span = y_max_data_for_rect - y_min_data_for_rect
                buffer = 0.0
                main_y_min_ax_val, main_y_max_ax_val = ax_main.get_ylim()
                
                if np.isfinite(data_span) and data_span > 1e-6: # If there's a discernible span
                    buffer = 0.1 * data_span
                elif np.isfinite(main_y_min_ax_val) and np.isfinite(main_y_max_ax_val): # Flat data, use main plot relative buffer
                    buffer = 0.025 * (main_y_max_ax_val - main_y_min_ax_val)
                else: # Fallback if main plot ylim also not finite (should be rare)
                    buffer = 0.1 # Small absolute buffer as last resort
                
                y_min_eff -= buffer
                y_max_eff += buffer
        
        # y_range_cfg is always None due to get_time_windows modification
        # So y_min_rect and y_max_rect are based on y_min_eff, y_max_eff
        y_min_rect = y_min_eff 
        y_max_rect = y_max_eff

        if np.isfinite(y_min_rect) and np.isfinite(y_max_rect) and y_max_rect > y_min_rect:
            rect_height = y_max_rect - y_min_rect
        else: # Fallback for empty, flat, or non-finite data after buffering
            main_y_min_ax, main_y_max_ax = ax_main.get_ylim()
            rect_height = (main_y_max_ax - main_y_min_ax) * 0.05 
            
            # Attempt to center the fallback rectangle
            mid_val = np.nanmean(valid_vst_values) if not valid_vst_values.empty else (main_y_min_ax + main_y_max_ax) / 2.0
            if not np.isfinite(mid_val): # Further fallback if mean is not finite
                mid_val = (main_y_min_ax + main_y_max_ax) / 2.0
            y_min_rect = mid_val - rect_height / 2.0


        x_start_num = mdates.date2num(start_date)
        x_end_num = mdates.date2num(end_date)
        x_width_num = x_end_num - x_start_num
        
        if x_width_num > 0 and rect_height > 0 :
            rect = plt.Rectangle(
                (x_start_num, y_min_rect), x_width_num, rect_height,
                fill=False, color=turbo_subplot_colors[i % 5], # Use Turbo colors
                linewidth=2.0, transform=ax_main.transData, zorder=10)
            ax_main.add_patch(rect)
            
            rect_center_x = x_start_num + x_width_num / 2
            rect_center_y = y_min_rect + rect_height / 2
            
            con = ConnectionPatch(
                xyA=(rect_center_x, rect_center_y), coordsA=ax_main.transData,
                xyB=(0.5, 1.15), coordsB=ax_sub.transAxes, # Point slightly above the subplot
                arrowstyle="->", color=turbo_subplot_colors[i % 5], # Use Turbo colors
                linewidth=1.5, linestyle=':', connectionstyle="arc3,rad=0.2", zorder=9)
            fig.add_artist(con)


    diagnostic_plot_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f'detailed_anomaly_analysis_{station_id}.png'
    full_plot_path = diagnostic_plot_dir / plot_filename
    
    try:
        plt.savefig(full_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved detailed anomaly plot to {full_plot_path}")
    except Exception as e:
        print(f"Error saving plot {full_plot_path}: {e}")
        plt.close(fig) # Ensure figure is closed even if save fails
        return None
    finally:
        plt.close(fig) 
    
    return full_plot_path
