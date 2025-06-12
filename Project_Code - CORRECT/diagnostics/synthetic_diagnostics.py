"""Diagnostic tools for synthetic error injection."""

import sys
from pathlib import Path
import matplotlib.ticker as mticker
import scipy.stats as stats # Add for linear regression

# Assuming synthetic_diagnostics.py is in Project_Code - CORRECT/diagnostics/
# Adjust sys.path to allow imports from the project root ('Project_Code - CORRECT')
_current_script_path = Path(__file__).resolve()
_project_root = _current_script_path.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared.synthetic.synthetic_errors import ErrorPeriod
import seaborn as sns
from synthetic_error_config import SYNTHETIC_ERROR_PARAMS
from models.lstm_traditional.config import LSTM_CONFIG
import numpy as np
from data_utils.data_loading import load_all_station_data

def plot_synthetic_errors(original_data: pd.DataFrame, 
                         modified_data: pd.DataFrame,
                         error_periods: List[ErrorPeriod],
                         station_name: str,
                         output_dir: Path) -> str:
    """Create visualization of synthetic errors."""
    diagnostic_dir = output_dir / "synthetic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Water Level Time Series with Synthetic Errors - {station_name}', fontsize=14)
    
    # Get the overall min and max values for the "Value" column,
    # using dropna() to avoid NaN values.
    value_col = "vst_raw"
    orig_vals = original_data[value_col].dropna()
    mod_vals = modified_data[value_col].dropna()
    
    if orig_vals.empty and mod_vals.empty:
        y_min, y_max = 0, 1
    elif orig_vals.empty:
        y_min, y_max = mod_vals.min(), mod_vals.max()
    elif mod_vals.empty:
        y_min, y_max = orig_vals.min(), orig_vals.max()
    else:
        y_min = min(orig_vals.min(), mod_vals.min())
        y_max = max(orig_vals.max(), mod_vals.max())
    
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding
    
    # Set the same y-axis limits for both plots
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # Define colors for different error types (match with create_interactive_plot)
    ERROR_COLORS = {
        'base': '#1f77b4',      # Default blue
        'spike': '#ff7f0e',     # Orange
        'flatline': '#2ca02c',  # Green
        'offset': '#d62728',    # Red
        'drift': '#9467bd',     # Purple
        'baseline shift': '#8c564b',  # Brown
        'noise': '#e377c2'      # Pink
    }
    
    # Plot original data in first subplot
    ax1.plot(original_data, label='Original Data', color=ERROR_COLORS['base'], alpha=0.7)
    ax1.set_title('Original Test Data')
    ax1.grid(False)
    ax1.legend()
    
    # Plot base data in second subplot
    ax2.plot(modified_data, color=ERROR_COLORS['base'], alpha=0.7, label='Base Data', zorder=1)
    
    # Track which error types we've seen (for legend)
    seen_error_types = {'base'}
    error_lines = {}
    
    # Plot each error type
    for period in error_periods:
        error_type = period.error_type
        if period.error_type not in ERROR_COLORS:
            print(f"Warning: Unknown error type {period.error_type}")
            continue
            
        period_data = modified_data.loc[period.start_time:period.end_time]
        
        # Skip if frequency is 0 in config
        error_config = SYNTHETIC_ERROR_PARAMS.get(error_type, {})
        if error_config.get('frequency', 1) == 0:
            print(f"Skipping {error_type} visualization (frequency = 0)")
            continue
            
        seen_error_types.add(error_type)
        
        # Special handling for each error type
        if error_type == 'spike':
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type], 
                          alpha=1.0, linewidth=2, marker='o', zorder=3)[0]
        
        elif error_type == 'flatline':
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type], 
                          alpha=0.9, linewidth=2)[0]
            # Add vertical indicator
            middle_time = period.start_time + (period.end_time - period.start_time) / 2
            ax2.axvline(x=middle_time, color=ERROR_COLORS[error_type], alpha=0.3, linewidth=8)
        
        elif error_type == 'drift':
            ax2.axvspan(period.start_time, period.end_time, 
                       color='yellow', alpha=0.2, zorder=1)
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type],
                          alpha=0.9, linewidth=2, zorder=3)[0]
        
        elif error_type == 'baseline shift':
            ax2.axvline(x=period.start_time, color=ERROR_COLORS[error_type],
                       alpha=0.8, linewidth=2, linestyle='--')
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type],
                          alpha=0.9, linewidth=2, zorder=3)[0]
        
        elif error_type == 'offset':
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type],
                          alpha=0.9, linewidth=2)[0]
            middle_time = period.start_time + (period.end_time - period.start_time) / 2
            ax2.axvline(x=middle_time, color=ERROR_COLORS[error_type], alpha=0.3, linewidth=8)
        
        elif error_type == 'noise':
            line = ax2.plot(period_data, color=ERROR_COLORS[error_type],
                          alpha=0.9, linewidth=1)[0]
        
        if error_lines.get(error_type) is None:
            error_lines[error_type] = line
    
    # Update legend to only show error types that were actually used
    legend_lines = [line for error_type, line in error_lines.items() 
                   if error_type in seen_error_types]
    legend_labels = [error_type.capitalize() for error_type, line in error_lines.items() 
                    if error_type in seen_error_types]
    ax2.legend(legend_lines, legend_labels)
    
    ax2.set_title('Test Data with Injected Errors')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Water Level (mm)')
    ax2.grid(False)
    
    plt.tight_layout()
    output_path = diagnostic_dir / f"{station_name}_synthetic_errors.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return str(output_path)

def create_interactive_plot(original_data: pd.DataFrame, 
                          modified_data: pd.DataFrame,
                          error_periods: List[ErrorPeriod],
                          station_name: str,
                          output_dir: Path) -> str:
    """Create interactive plotly visualization of synthetic errors."""
    diagnostic_dir = output_dir / "synthetic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       subplot_titles=('Original Test Data', 'Test Data with Injected Errors'))
    
    # Define colors for different error types
    ERROR_COLORS = {
        'base': '#1f77b4',      # Default blue
        'spike': '#ff7f0e',     # Orange
        'flatline': '#2ca02c',  # Green
        'offset': '#d62728',    # Red
        'drift': '#9467bd',     # Purple
        'baseline shift': '#8c564b',  # Brown
        'noise': '#e377c2'      # Pink
    }
    
    # Plot original data
    fig.add_trace(
        go.Scatter(x=original_data.index, y=original_data['vst_raw'],
                  name='Original Data',
                  line=dict(color=ERROR_COLORS['base'], width=1),
                  opacity=0.7),
        row=1, col=1
    )
    
    # Plot base modified data
    fig.add_trace(
        go.Scatter(x=modified_data.index, y=modified_data['vst_raw'],
                  name='Base Data',
                  line=dict(color=ERROR_COLORS['base'], width=1),
                  opacity=0.7),
        row=2, col=1
    )
    
    # Plot each error type
    for period in error_periods:
        error_type = period.error_type
        period_data = modified_data.loc[period.start_time:period.end_time]
        
        # Add hover text with error details
        hover_text = [
            f"Error Type: {error_type}<br>" +
            f"Start: {period.start_time}<br>" +
            f"End: {period.end_time}<br>" +
            f"Original Value: {float(orig[0]):.2f}<br>" +  # Extract float from array
            f"Modified Value: {float(mod[0]):.2f}<br>" +   # Extract float from array
            f"Parameters: {period.parameters}"
            for orig, mod in zip(period.original_values.reshape(-1, 1), 
                               period.modified_values.reshape(-1, 1))
        ]
        
        fig.add_trace(
            go.Scatter(
                x=period_data.index,
                y=period_data['vst_raw'],
                name=error_type.capitalize(),
                line=dict(color=ERROR_COLORS[error_type], width=2),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add visual indicators for specific error types
        if error_type == 'drift':
            fig.add_vrect(
                x0=period.start_time, x1=period.end_time,
                fillcolor='yellow', opacity=0.2,
                layer='below', line_width=0,
                row=2, col=1
            )
        elif error_type in ['flatline', 'offset']:
            middle_time = period.start_time + (period.end_time - period.start_time) / 2
            fig.add_vline(
                x=middle_time, line_color=ERROR_COLORS[error_type],
                opacity=0.3, line_width=8,
                row=2, col=1
            )
        elif error_type == 'baseline shift':
            fig.add_vline(
                x=period.start_time, line_color=ERROR_COLORS[error_type],
                opacity=0.8, line_width=2, line_dash='dash',
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'Water Level Time Series with Synthetic Errors - {station_name}',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Water Level (mm)', row=1, col=1)
    fig.update_yaxes(title_text='Water Level (mm)', row=2, col=1)
    
    # Save interactive plot
    output_path = diagnostic_dir / f"{station_name}_synthetic_errors_interactive.html"
    fig.write_html(output_path)
    
    return str(output_path)

def generate_synthetic_report(stations_results: dict, output_dir: Path) -> Path:
    """
    Generate report of synthetic error injection results.
    
    Args:
        stations_results: Dictionary of station results with synthetic data
        output_dir: Output directory for saving report
        
    Returns:
        Path to the generated report
    """
    report_dir = output_dir / "synthetic"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "synthetic_error_report.txt"
    
    with open(report_path, "w") as f:
        f.write("Synthetic Error Injection Report\n")
        f.write("==============================\n\n")
        
        for station_key, results in stations_results.items():
            f.write(f"\nStation: {station_key}\n")
            f.write("-" * (len(station_key) + 9) + "\n")
            
            # Group errors by type
            errors_by_type = {}
            for period in results['error_periods']:
                if period.error_type not in errors_by_type:
                    errors_by_type[period.error_type] = []
                errors_by_type[period.error_type].append(period)
            
            f.write("\nInjected Errors:\n")
            for error_type, periods in errors_by_type.items():
                f.write(f"\n  {error_type.capitalize()}:\n")
                f.write(f"    - Count: {len(periods)} instances\n")
                f.write(f"    - Time ranges:\n")
                for period in periods:
                    start_time = period.start_time.strftime('%Y-%m-%d %H:%M')
                    end_time = period.end_time.strftime('%Y-%m-%d %H:%M')
                    duration = (period.end_time - period.start_time).total_seconds() / 3600  # hours
                    f.write(f"      * {start_time} to {end_time} (duration: {duration:.1f} hours)\n")
                    if hasattr(period, 'parameters') and period.parameters:
                        f.write(f"        Parameters: {period.parameters}\n")
            
            f.write("\n" + "="*50 + "\n")
    
    return report_path

def plot_synthetic_vs_actual(all_error_data_to_plot: List[tuple], 
                               station_id_for_filename: str,
                               all_stations_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                               output_dir: Path) -> str:
    """
    Create a single stacked comparison plot showing synthetic vs. actual anomalies for multiple error types.

    Args:
        all_error_data_to_plot: A list of tuples. Each tuple contains:
            (original_df_segment_for_synthetic, modified_df_segment_for_synthetic, error_period_obj, error_type_name)
            where original_df_segment_for_synthetic is the *segment* used for synthetic injection.
        station_id_for_filename: Base station ID for the output filename (e.g., '21006846').
        all_stations_data_dict: Dict containing all raw data for all stations, keyed by station ID, then by data type (e.g., 'vst_raw').
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot or None if not applicable.
    """
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 24,
        'axes.titlesize': 29,
        'axes.labelsize': 27,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 30, # Increased further
        'figure.titlesize': 29
    })
    
    # Actual anomaly data (timestamps for station 21006846, except where noted)
    actual_anomalies = {
        'spike': {
            'period': ('2019-05-08 00:00', '2019-05-08 01:00'), # Short duration for a spike
            'source_station': '21006846'
        },
        'drift': {
            'period': ('2022-03-01 00:00:00', '2022-06-01 00:00:00'),  # Updated period for drift
            'source_station': '21006846' # Assuming this is the typical station for comparison
        },
        'offset': {
            'period': ('2008-09-06', '2008-10-10'),
            'source_station': '21006846'
        },
        'baseline shift': {
            'period': ('2021-07-05', '2021-07-13'),
            'source_station': '21006846'
        },
        'flatline': {
            'period': ('2023-01-10', '2023-01-17'), # Placeholder, update if real one found for 21006846
            'source_station': '21006846' 
        },
        'noise': {
            'period': ('2009-06-03', '2009-06-12'), #2009, june 3rd to june 10th
            'source_station': '21006847' # Note: from a different station, used as example
        }
    }
    
    num_error_types_to_plot = len(all_error_data_to_plot)
    if num_error_types_to_plot == 0:
        print("No data provided to plot_synthetic_vs_actual. Skipping plot generation.")
        return None

    fig, axes = plt.subplots(num_error_types_to_plot, 2, 
                           figsize=(20, 5 * num_error_types_to_plot), 
                           dpi=300, squeeze=False)#, sharex=True) # squeeze=False ensures axes is always 2D
    
    #fig.suptitle(f'Synthetic vs. Actual Anomaly Comparison for Station {station_id_for_filename}', fontsize=20, y=1.00)

    zoom_windows = {
        'spike': pd.Timedelta(days=2),    # Shorter window for spikes
        'drift': pd.Timedelta(days=30),   # Longer for drift to show context
        'offset': pd.Timedelta(days=5),
        'baseline shift': pd.Timedelta(days=20),
        'flatline': pd.Timedelta(days=7),
        'noise': pd.Timedelta(days=8) 
    }
    
    for idx, data_tuple in enumerate(all_error_data_to_plot):
        original_data_for_type, modified_data_for_type, synthetic_period_obj, error_type_name = data_tuple
        
        # --- Plot synthetic anomaly (left column) --- 
        ax_synthetic = axes[idx, 0]
        
        # Ensure a minimum visible duration for axvspan, especially for point anomalies
        min_span_duration = pd.Timedelta(hours=12) # Define a minimum visual span, e.g., 12 hours
        synthetic_start_for_span = synthetic_period_obj.start_time
        synthetic_end_for_span = synthetic_period_obj.end_time
        if (synthetic_end_for_span - synthetic_start_for_span) < min_span_duration:
            center_time = synthetic_start_for_span + (synthetic_end_for_span - synthetic_start_for_span) / 2
            synthetic_start_for_span = center_time - min_span_duration / 2
            synthetic_end_for_span = center_time + min_span_duration / 2

        ax_synthetic.axvspan(synthetic_start_for_span, synthetic_end_for_span, 
                           color='#ffffcc', alpha=0.5, label='Synthetic Anomaly Period')
        
        window_size_synthetic = zoom_windows.get(error_type_name, pd.Timedelta(days=7))
        window_start_synthetic = synthetic_period_obj.start_time - window_size_synthetic
        window_end_synthetic = synthetic_period_obj.end_time + window_size_synthetic
        
        mask_synthetic = (original_data_for_type.index >= window_start_synthetic) & (original_data_for_type.index <= window_end_synthetic)
        window_original_synthetic = original_data_for_type[mask_synthetic]
        window_modified_synthetic = modified_data_for_type[mask_synthetic]
        
        ax_synthetic.plot(window_original_synthetic.index, window_original_synthetic['vst_raw'], 
                        label='Original Segment', color='#1f77b4', linewidth=1.5)
        ax_synthetic.plot(window_modified_synthetic.index, window_modified_synthetic['vst_raw'], 
                        label='With Synthetic Error', color='#d62728', linewidth=1.5, linestyle='--')
        
        title_suffix_synthetic = f" (Duration: {(synthetic_period_obj.end_time - synthetic_period_obj.start_time).days}d)" \
                                 if (synthetic_period_obj.end_time - synthetic_period_obj.start_time).days > 0 else ""
        ax_synthetic.set_title(f'Synthetic {error_type_name.title()} {title_suffix_synthetic}', fontweight='bold')
        ax_synthetic.set_xlabel('') # Remove x-label
        ax_synthetic.tick_params(axis='x', labelbottom=False, bottom=False) # Remove x-ticks and labels
        ax_synthetic.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=15)
        ax_synthetic.spines['top'].set_visible(False)
        ax_synthetic.spines['right'].set_visible(False) # Ensure right spine is also invisible
        ax_synthetic.grid(False) # Ensure grid is off

        # --- Plot actual anomaly example (right column) --- 
        ax_actual = axes[idx, 1]
        actual_anomaly_info = actual_anomalies.get(error_type_name)
        
        if actual_anomaly_info and actual_anomaly_info['period']:
            start_date_actual, end_date_actual = actual_anomaly_info['period']
            source_station_id_actual = actual_anomaly_info['source_station']
            start_ts_actual = pd.Timestamp(start_date_actual)
            end_ts_actual = pd.Timestamp(end_date_actual)

            context_data_for_actual_plot = None
            actual_station_df_dict = all_stations_data_dict.get(source_station_id_actual)
            if actual_station_df_dict:
                # Attempt to get 'vst_raw' or a fallback like 'Value' if renamed, or first numerical
                if 'vst_raw' in actual_station_df_dict and actual_station_df_dict['vst_raw'] is not None:
                    context_data_for_actual_plot = actual_station_df_dict['vst_raw']
                elif 'Value' in actual_station_df_dict and actual_station_df_dict['Value'] is not None: # Common original name
                     context_data_for_actual_plot = actual_station_df_dict['Value']
                     if 'vst_raw' not in context_data_for_actual_plot.columns and 'Value' in context_data_for_actual_plot.columns:
                         context_data_for_actual_plot = context_data_for_actual_plot.rename(columns={'Value': 'vst_raw'})
                else: # Fallback to first numerical column if specific keys are missing
                    for key, df_val in actual_station_df_dict.items():
                        if df_val is not None and not df_val.empty and np.issubdtype(df_val.iloc[:, 0].dtype, np.number):
                            context_data_for_actual_plot = df_val.copy()
                            # Ensure it has a 'vst_raw' column for consistency in plotting logic
                            if context_data_for_actual_plot.columns[0] != 'vst_raw':
                                context_data_for_actual_plot = context_data_for_actual_plot.rename(
                                    columns={context_data_for_actual_plot.columns[0]: 'vst_raw'}
                                )
                            print(f"Used first numerical column '{context_data_for_actual_plot.columns[0]}' as 'vst_raw' for station {source_station_id_actual}")
                            break
            
            if context_data_for_actual_plot is None or context_data_for_actual_plot.empty:
                ax_actual.text(0.5, 0.5, f'Data for station {source_station_id_actual} not found or empty.\nCannot display actual {error_type_name} example.',
                               horizontalalignment='center', verticalalignment='center', color='red', transform=ax_actual.transAxes)
                ax_actual.set_title(f'Real {error_type_name.title()} Example (Data Missing)', fontweight='bold')
            else:
                 # Ensure 'vst_raw' column exists in the selected DataFrame for actual anomaly plotting
                if 'vst_raw' not in context_data_for_actual_plot.columns:
                    # This case should be rare if above logic works, but as a fallback:
                    valid_cols = [col for col in context_data_for_actual_plot.columns if col != 'Date']
                    if valid_cols:
                        context_data_for_actual_plot = context_data_for_actual_plot.rename(columns={valid_cols[0]: 'vst_raw'})
                    else:
                        ax_actual.text(0.5, 0.5, f"No suitable data column for 'vst_raw' in station {source_station_id_actual}",
                                       horizontalalignment='center', verticalalignment='center', color='red', transform=ax_actual.transAxes)
                        ax_actual.set_title(f'Real {error_type_name.title()} Example (Data Column Missing)', fontweight='bold')
                        ax_actual.set_xlabel('') # Remove x-label
                        ax_actual.tick_params(axis='x', labelbottom=False, bottom=False) # Remove x-ticks and labels
                        ax_actual.set_ylabel('') # Remove y-label for real plots
                        ax_actual.grid(False); ax_actual.spines["top"].set_visible(False); ax_actual.spines["right"].set_visible(False)
                        continue # to next error type in the main loop


                window_size_actual = zoom_windows.get(error_type_name, pd.Timedelta(days=7))
                actual_period_midpoint = start_ts_actual + (end_ts_actual - start_ts_actual) / 2
                context_start_actual = actual_period_midpoint - window_size_actual
                context_end_actual = actual_period_midpoint + window_size_actual
                if (end_ts_actual - start_ts_actual) > (window_size_actual * 1.5):
                    context_start_actual = start_ts_actual - pd.Timedelta(days=max(1, int(window_size_actual.days * 0.25)))
                    context_end_actual = end_ts_actual + pd.Timedelta(days=max(1, int(window_size_actual.days * 0.25)))
                
                min_available_date = context_data_for_actual_plot.index.min()
                max_available_date = context_data_for_actual_plot.index.max()
                
                # Ensure valid_start is not after valid_end
                valid_start_actual = max(context_start_actual, min_available_date)
                valid_end_actual = min(context_end_actual, max_available_date)
                if valid_start_actual > valid_end_actual:
                    # This can happen if the anomaly period is completely outside the available data range
                    # or if the zoom window logic pushes it out. Fallback to just the anomaly period itself for xlim.
                    valid_start_actual = start_ts_actual - pd.Timedelta(days=1)
                    valid_end_actual = end_ts_actual + pd.Timedelta(days=1)
                    # And ensure these are within overall bounds of available data if possible
                    valid_start_actual = max(valid_start_actual, min_available_date)
                    valid_end_actual = min(valid_end_actual, max_available_date)
                    if valid_start_actual > valid_end_actual: # Still problematic, just use anomaly period
                        valid_start_actual = start_ts_actual
                        valid_end_actual = end_ts_actual

                mask_actual_context = (context_data_for_actual_plot.index >= valid_start_actual) & (context_data_for_actual_plot.index <= valid_end_actual)
                window_data_actual_context = context_data_for_actual_plot[mask_actual_context]
                
                if not window_data_actual_context.empty:
                    ax_actual.plot(window_data_actual_context.index, window_data_actual_context['vst_raw'], 
                                 color='#1f77b4', linewidth=1.5, label=f'Actual Data (Station {source_station_id_actual})')
                else:
                    ax_actual.text(0.5, 0.4, f'No data for Station {source_station_id_actual}\\nin range {valid_start_actual.date()} to {valid_end_actual.date()}', 
                                    horizontalalignment='center', verticalalignment='center', color='red', transform=ax_actual.transAxes)

                mask_actual_anomaly_in_plot_data = (context_data_for_actual_plot.index >= start_ts_actual) & (context_data_for_actual_plot.index <= end_ts_actual)
                actual_anomaly_segment = context_data_for_actual_plot[mask_actual_anomaly_in_plot_data]
                # if not actual_anomaly_segment.empty:  # REMOVED ORANGE LINE
                #     ax_actual.plot(actual_anomaly_segment.index, actual_anomaly_segment['vst_raw'], 
                #                  color='#ff7f0e', linewidth=2.0, linestyle='-', label=f'Actual Anomaly Detail (Sta {source_station_id_actual})')

                # --- Add VINGE data plotting specifically for drift error type ---
                if error_type_name == 'drift':
                    vinge_df_for_station = all_stations_data_dict.get(source_station_id_actual, {}).get('vinge')
                    if vinge_df_for_station is not None and not vinge_df_for_station.empty:
                        # Ensure VINGE index is datetime
                        if not isinstance(vinge_df_for_station.index, pd.DatetimeIndex):
                            vinge_df_for_station.index = pd.to_datetime(vinge_df_for_station.index)
                        
                        vinge_data_in_window = vinge_df_for_station[
                            (vinge_df_for_station.index >= valid_start_actual) & 
                            (vinge_df_for_station.index <= valid_end_actual)
                        ].copy()

                        if not vinge_data_in_window.empty and 'W.L [cm]' in vinge_data_in_window.columns:
                            vinge_data_in_window['W.L [cm]'] = pd.to_numeric(vinge_data_in_window['W.L [cm]'], errors='coerce')
                            vinge_data_in_window.dropna(subset=['W.L [cm]'], inplace=True)
                            if not vinge_data_in_window.empty:
                                ax_actual.scatter(vinge_data_in_window.index, vinge_data_in_window['W.L [cm]'],
                                                  color='green', # Distinct color for VINGE
                                                  label=f'VINGE(Sta {source_station_id_actual})',
                                                  s=60, zorder=6, alpha=0.7) 
                    
                    # Add VST EDT data for drift plot
                    vst_edt_df_for_station = all_stations_data_dict.get(source_station_id_actual, {}).get('vst_edt')
                    if vst_edt_df_for_station is not None and not vst_edt_df_for_station.empty:
                        if not isinstance(vst_edt_df_for_station.index, pd.DatetimeIndex):
                            vst_edt_df_for_station.index = pd.to_datetime(vst_edt_df_for_station.index)
                        
                        vst_edt_in_window = vst_edt_df_for_station[
                            (vst_edt_df_for_station.index >= valid_start_actual) & 
                            (vst_edt_df_for_station.index <= valid_end_actual)
                        ].copy()

                        # Assuming the VST EDT data column is also 'vst_raw' or needs to be identified
                        # For consistency, let's assume it might be named 'Value' like in some raw files, or just take the first numeric col
                        edt_col_name = 'vst_raw' # Default assumption
                        if 'Value' in vst_edt_in_window.columns and 'vst_raw' not in vst_edt_in_window.columns:
                            vst_edt_in_window = vst_edt_in_window.rename(columns={'Value': 'vst_raw'})
                        elif 'vst_raw' not in vst_edt_in_window.columns:
                            numeric_cols_edt = [col for col in vst_edt_in_window.columns if pd.api.types.is_numeric_dtype(vst_edt_in_window[col])]
                            if numeric_cols_edt:
                                edt_col_name = numeric_cols_edt[0]
                                if edt_col_name != 'vst_raw': # rename if it's not already vst_raw
                                     vst_edt_in_window = vst_edt_in_window.rename(columns={edt_col_name: 'vst_raw'})
                            else:
                                edt_col_name = None # No suitable column found
                        
                        if edt_col_name and not vst_edt_in_window.empty and 'vst_raw' in vst_edt_in_window.columns:
                            vst_edt_in_window['vst_raw'] = pd.to_numeric(vst_edt_in_window['vst_raw'], errors='coerce')
                            vst_edt_in_window.dropna(subset=['vst_raw'], inplace=True)
                            if not vst_edt_in_window.empty:
                                ax_actual.plot(vst_edt_in_window.index, vst_edt_in_window['vst_raw'],
                                               color='#2ca02c', # Matplotlib green, different from VINGE
                                               label=f'VST EDT (Sta {source_station_id_actual})',
                                               linewidth=1.5, linestyle=':', zorder=4)

                # --- End of VINGE data plotting for drift ---

                title_actual = f'Real {error_type_name.title()}'
                # No need to add (Period from Station X) if source_station_id_actual is same as station_id_for_filename
                # The legend for data line already clarifies station. If truly different, it will be clear.
                ax_actual.set_title(title_actual, fontweight='bold')
                ax_actual.set_xlim(valid_start_actual, valid_end_actual)

        else:
            ax_actual.text(0.5, 0.5, f'Placeholder for real {error_type_name.title()} error example\n(Update `actual_anomalies` in script if found)', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_actual.transAxes, fontsize=14, color='#555555')
            ax_actual.set_title(f'Real {error_type_name.title()} Error Example', fontweight='bold')
        
        ax_actual.set_xlabel('') # Remove x-label
        ax_actual.tick_params(axis='x', labelbottom=False, bottom=False) # Remove x-ticks and labels

        ax_actual.set_ylabel('') # Remove y-label for real plots
        ax_actual.spines['top'].set_visible(False)
        ax_actual.spines['right'].set_visible(False) # Ensure right spine is also invisible
        ax_actual.grid(False) # Ensure grid is off
        
        # Auto-adjust y-limits for each row independently for better visualization
        for ax_col in [ax_synthetic, ax_actual]:
            if len(ax_col.lines) > 0:
                # Collect y_data from all lines that actually have data
                y_data_list = [line.get_ydata(orig=False) for line in ax_col.lines if len(line.get_ydata(orig=False)) > 0]
                if y_data_list: # Only proceed if we have some data arrays
                    y_data_all_lines = np.concatenate(y_data_list)
                    y_data_all_lines = y_data_all_lines[~np.isnan(y_data_all_lines)]
                    if len(y_data_all_lines) > 0:
                        y_min_ax, y_max_ax = np.nanmin(y_data_all_lines), np.nanmax(y_data_all_lines)
                        y_range_ax = y_max_ax - y_min_ax if y_max_ax > y_min_ax else 1.0 # Ensure range is not zero
                        margin_ax = 0.1 * y_range_ax
                        ax_col.set_ylim(y_min_ax - margin_ax, y_max_ax + margin_ax)
    
    # --- Shared Legend --- 
    # Desired legend entries
    desired_labels_map = {
        "VST RAW": None, 
        "Error modified": None,
        "Anomaly Period": None,
        "VINGE": None, # Changed from VINGE Data
        "VST EDT": None  # Added VST EDT
    }
    found_handles_for_labels = {label: None for label in desired_labels_map}

    for ax_row in axes:
        for ax_col in ax_row:
            handles, labels = ax_col.get_legend_handles_labels()
            for handle, label_text in zip(handles, labels):
                if "Original Segment" in label_text or "Actual Data (Station" in label_text:
                    if not found_handles_for_labels["VST RAW"]:
                        found_handles_for_labels["VST RAW"] = handle
                elif "With Synthetic Error" in label_text:
                    if not found_handles_for_labels["Error modified"]:
                        found_handles_for_labels["Error modified"] = handle
                elif "Anomaly Period" in label_text:
                    if not found_handles_for_labels["Anomaly Period"]:
                        found_handles_for_labels["Anomaly Period"] = handle
                elif label_text.startswith("VINGE(Sta") : # Match user's new label format for VINGE
                    if not found_handles_for_labels["VINGE"]:
                        found_handles_for_labels["VINGE"] = handle
                elif label_text.startswith("VST EDT (Sta") : # Match VST EDT label
                    if not found_handles_for_labels["VST EDT"]:
                        found_handles_for_labels["VST EDT"] = handle
    
    # Filter out None handles and create the legend
    final_handles = []
    final_labels = []
    for label, handle in found_handles_for_labels.items():
        if handle:
            final_handles.append(handle)
            final_labels.append(label)

    if final_handles:
        fig.legend(final_handles, final_labels,
                   loc='lower center', ncol=len(final_handles),  # Force all items in one row
                   bbox_to_anchor=(0.5, 0.01))  # Move legend lower

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.subplots_adjust(hspace=0.5)
    
    output_filename = f'stacked_synthetic_vs_actual_comparison_{station_id_for_filename}.png'
    output_path_stacked = output_dir / "synthetic" / output_filename # Save inside synthetic subdir
    (output_dir / "synthetic").mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    plt.savefig(output_path_stacked, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved stacked synthetic vs actual comparison plot to {output_path_stacked}")
    return str(output_path_stacked)

def run_all_synthetic_diagnostics(
    split_datasets: dict, 
    stations_results: dict, 
    output_dir: Path
) -> dict:
    """
    Run simplified synthetic error diagnostics with focus on key plots:
    1. A basic plot showing data with injected anomalies
    2. Synthetic vs actual comparison (if available)
    
    Args:
        split_datasets: Dictionary of split datasets (can be in different formats)
        stations_results: Dictionary of station results with synthetic data
        output_dir: Output directory for saving plots
        
    Returns:
        Dictionary with diagnostic results
    """
    print("Generating simplified synthetic error diagnostics...")
    diagnostic_results = {}
    
    # Process each station in the results (limited to just one station to save time)
    processed_stations = 0
    
    for station_key, result_data in stations_results.items():
        if not result_data.get('error_periods') or processed_stations >= 1:
            continue
            
        try:
            # Find the original data (either in split_datasets structure)
            original_data = None
            station_id = station_key.split('_')[0]  # Extract station ID from the key
            
            # Handle different split_datasets structures
            if 'windows' in split_datasets:
                # Original structure with years and stations
                for split_type, split_data in split_datasets['windows'].items():
                    if station_id in split_data:
                        original_data = split_data[station_id]
                        break
            else:
                # New structure with direct test data
                original_data = split_datasets.get(station_id, None)
                
            # If we couldn't find it in expected locations, try to use a direct reference
            if original_data is None and isinstance(split_datasets, pd.DataFrame):
                original_data = split_datasets
                
            if original_data is None:
                print(f"Warning: Could not find original data for {station_key}")
                continue
                
            # For DataFrame, we need to create a DataFrame with just vst_raw
            if isinstance(original_data, pd.DataFrame):
                original_data = pd.DataFrame({'vst_raw': original_data['vst_raw']})
            
            # Create basic plot showing data with injected anomalies
            diagnostic_dir = output_dir / "synthetic"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(original_data.index, original_data['vst_raw'], label='Original Data', alpha=0.7, color='blue')
            ax.plot(result_data['modified_data'].index, result_data['modified_data']['vst_raw'], 
                   label='Data with Injected Errors', alpha=0.7, color='red')
            
            # Highlight error periods with different colors by type
            error_types = set(period.error_type for period in result_data['error_periods'])
            color_map = {
                error_type: plt.cm.tab10(i % 10) 
                for i, error_type in enumerate(error_types)
            }
            
            for period in result_data['error_periods']:
                ax.axvspan(period.start_time, period.end_time, 
                          color=color_map[period.error_type], 
                          alpha=0.2, 
                          label=f"{period.error_type.capitalize()}")
            
            # Remove duplicate labels
            handles, labels = [], []
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            ax.legend(handles, labels)
            
            ax.set_title(f'Data with Injected Synthetic Errors - {station_key}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level')
            ax.grid(True)
            
            # Save plot
            output_path = diagnostic_dir / f'synthetic_errors_{station_key}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create synthetic vs actual plot if available
            comparison_plot = plot_synthetic_vs_actual(
                all_error_data_to_plot=[(original_data, result_data['modified_data'], period, period.error_type) for period in result_data['error_periods']],
                station_id_for_filename=station_key,
                all_stations_data_dict=load_all_station_data(),
                output_dir=output_dir
            )
            
            # Store results
            diagnostic_results[station_key] = {
                'data_plot': str(output_path),
                'comparison_plot': comparison_plot
            }
            
            processed_stations += 1
            print(f"Processed station {station_key}")
        
        except Exception as e:
            print(f"Error generating diagnostics for {station_key}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed simplified synthetic diagnostics ({processed_stations} stations processed)")
    return diagnostic_results 

if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    import numpy as np # Added for numerical operations if needed by preprocessor or data selection
    from copy import deepcopy
    import sys
    import traceback # Import traceback here

    # Assuming synthetic_diagnostics.py is in Project_Code - CORRECT/diagnostics/
    # current_script_path = Path(__file__).resolve() # No longer needed here
    # project_root = current_script_path.parents[1] # No longer needed here

    # Add project root to sys.path to allow imports from other modules
    # sys.path.append(str(project_root)) # This is now done at the top of the script

    from synthetic_error_config import SYNTHETIC_ERROR_PARAMS
    from models.lstm_traditional.config import LSTM_CONFIG # config should be found due to top-level sys.path modification
    from _2_synthetic.synthetic_errors import SyntheticErrorGenerator, ErrorPeriod # _2_synthetic should be found
    from _3_lstm_model.preprocessing_LSTM import DataPreprocessor # _3_lstm_model should be found

    # PIL import is no longer needed here as we are plotting directly to stacked subplots
    # try:
    #     from PIL import Image
    #     PIL_AVAILABLE = True
    # except ImportError:
    #     PIL_AVAILABLE = False
    #     print("Warning: Pillow (PIL) library not found. Individual plots will be generated, but not stitched together. To enable stitching, please install Pillow: pip install Pillow")

    station_id_to_plot = '21006846'
    thesis_plots_output_dir = _project_root / "results" / "thesis_error_visualizations"
    thesis_plots_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Initializing generation of stacked error comparison plot for station {station_id_to_plot} ---")
    print(f"Output will be saved in: {thesis_plots_output_dir / 'synthetic'}")

    base_data_for_injection = None
    raw_station_data_concatenated = None # Will hold concatenated data for the primary station
    all_stations_loaded_data = {} # Will hold data for ALL stations

    try:
        print("\nLoading data for ALL stations...")
        all_stations_loaded_data = load_all_station_data()
        if not all_stations_loaded_data:
            print("Critical: load_all_station_data returned empty. Exiting.")
            sys.exit(1)
        
        # Normalize column names in the loaded data (e.g. 'Value' to 'vst_raw') for consistency
        # This step is important if plot_synthetic_vs_actual expects 'vst_raw'
        for st_id, station_data_dict in all_stations_loaded_data.items():
            if 'vst_raw' in station_data_dict and station_data_dict['vst_raw'] is not None:
                if 'Value' in station_data_dict['vst_raw'].columns and 'vst_raw' not in station_data_dict['vst_raw'].columns:
                    station_data_dict['vst_raw'] = station_data_dict['vst_raw'].rename(columns={'Value': 'vst_raw'})
            # Add similar logic for other potential data types if needed, e.g. if 'VINGE' is main column for vinge type


        print(f"\nAttempting to load and prepare data for primary station: {station_id_to_plot} for synthetic injection context...")
        # This part remains for getting the train/val/test for the *primary* station, from which we select a segment
        model_cfg = LSTM_CONFIG.copy() # Revert to using LSTM_CONFIG for the preprocessor
        preprocessor = DataPreprocessor(config=model_cfg) 
        
        print(f"Calling preprocessor.load_and_split_data for station {station_id_to_plot}...")
        train_df, val_df, test_df, _ = preprocessor.load_and_split_data(_project_root, station_id_to_plot)
        
        print("Data splits loaded. Concatenating train, val, and test sets to form base data for station.")
        
        parts_to_concat = []
        if isinstance(train_df, pd.DataFrame) and not train_df.empty:
            parts_to_concat.append(train_df)
        if isinstance(val_df, pd.DataFrame) and not val_df.empty:
            parts_to_concat.append(val_df)
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            parts_to_concat.append(test_df)

        if not parts_to_concat:
            print(f"All data splits (train, val, test) are empty or invalid for station {station_id_to_plot}. Exiting.")
            sys.exit(1)
            
        raw_station_data_concatenated = pd.concat(parts_to_concat).sort_index()
        raw_station_data_concatenated = raw_station_data_concatenated[~raw_station_data_concatenated.index.duplicated(keep='first')]
        print(f"Concatenated data for primary station {station_id_to_plot}, final shape: {raw_station_data_concatenated.shape}")

        if raw_station_data_concatenated.empty:
            print(f"Failed to load or assemble raw data for primary station {station_id_to_plot}. Exiting.")
            sys.exit(1)
        
        if 'vst_raw' not in raw_station_data_concatenated.columns:
            vst_raw_col_candidates = [col for col in raw_station_data_concatenated.columns if col.endswith('_vst_raw') or col == 'Value']
            if vst_raw_col_candidates:
                raw_station_data_concatenated = raw_station_data_concatenated.rename(columns={vst_raw_col_candidates[0]: 'vst_raw'})
            else:
                numerical_cols = raw_station_data_concatenated.select_dtypes(include=np.number).columns
                if not numerical_cols.empty:
                    raw_station_data_concatenated = raw_station_data_concatenated.rename(columns={numerical_cols[0]: 'vst_raw'})
                else:
                    print("Critical: 'vst_raw' column not found and no numerical alternative. Exiting.")
                    sys.exit(1)
        
        if not isinstance(raw_station_data_concatenated.index, pd.DatetimeIndex):
            try:
                raw_station_data_concatenated.index = pd.to_datetime(raw_station_data_concatenated.index)
            except Exception as e:
                print(f"Could not convert index to DatetimeIndex: {e}. Exiting.")
                sys.exit(1)
        
        available_years = sorted(raw_station_data_concatenated.index.year.unique())
        selected_year_data = None
        MAX_NAN_PERCENTAGE_PREFERRED = 0.05
        MAX_NAN_PERCENTAGE_PROBLEMATIC = 0.02

        preferred_years_list = [y for y in available_years if not (2010 <= y <= 2016)]
        problematic_years_list = [y for y in available_years if 2010 <= y <= 2016]

        print(f"Searching for suitable data segment...")
        for year in sorted(preferred_years_list, reverse=True):
            data_year = raw_station_data_concatenated[raw_station_data_concatenated.index.year == year]
            if len(data_year) > 7000:
                nan_percentage = data_year['vst_raw'].isnull().sum() / len(data_year)
                if nan_percentage <= MAX_NAN_PERCENTAGE_PREFERRED:
                    selected_year_data = data_year
                    print(f"Selected data from preferred year {year} ({len(data_year)} points, {nan_percentage*100:.2f}% NaNs).")
                    break
        
        if selected_year_data is None:
            print("No ideal year found in preferred years. Checking 2010-2016 range with stricter NaN criteria.")
            for year in sorted(problematic_years_list, reverse=True):
                data_year = raw_station_data_concatenated[raw_station_data_concatenated.index.year == year]
                if len(data_year) > 7000:
                    nan_percentage = data_year['vst_raw'].isnull().sum() / len(data_year)
                    if nan_percentage <= MAX_NAN_PERCENTAGE_PROBLEMATIC:
                        selected_year_data = data_year
                        print(f"Selected data from problematic year {year} ({len(data_year)} points, {nan_percentage*100:.2f}% NaNs) as it met stricter NaN criteria.")
                        break
                    else:
                        print(f"Skipping year {year} from 2010-2016 range due to high NaN percentage: {nan_percentage*100:.2f}% (threshold: {MAX_NAN_PERCENTAGE_PROBLEMATIC*100:.2f}%).")
        
        if selected_year_data is not None:
            base_data_for_injection = selected_year_data.copy()
        else:
            print("No single year found meeting quality criteria. Applying general fallback logic for data segment.")
            if len(raw_station_data_concatenated) > 7000:
                 potential_segment = raw_station_data_concatenated.tail(8760).copy()
                 if len(potential_segment) < 7000 and len(raw_station_data_concatenated) >= 7000:
                     potential_segment = raw_station_data_concatenated.head(8760).copy()
                 elif len(potential_segment) < 7000:
                     potential_segment = raw_station_data_concatenated.copy()
                 base_data_for_injection = potential_segment
                 segment_start_year = base_data_for_injection.index.min().year if not base_data_for_injection.empty else 'N/A'
                 segment_end_year = base_data_for_injection.index.max().year if not base_data_for_injection.empty else 'N/A'
                 segment_nan_percentage = (base_data_for_injection['vst_raw'].isnull().sum() / len(base_data_for_injection)) if not base_data_for_injection.empty else 0
                 print(f"Using fallback segment of {len(base_data_for_injection)} points, from years {segment_start_year}-{segment_end_year}, with {segment_nan_percentage*100:.2f}% NaNs.")
                 if not base_data_for_injection.empty and any(2010 <= y <= 2016 for y in range(base_data_for_injection.index.min().year, base_data_for_injection.index.max().year + 1)):
                     print("Warning: This fallback segment might include data from the 2010-2016 range. Visuals may be affected if NaN content is high in that portion.")
            else:
                base_data_for_injection = raw_station_data_concatenated.copy()
                print(f"Using all available data ({len(base_data_for_injection)} points).")

        if base_data_for_injection is None or base_data_for_injection.empty:
            print(f"Critical: No suitable data segment could be selected. Exiting.")
            sys.exit(1)
        
        if 'vst_raw' in base_data_for_injection.columns:
            base_data_for_injection = base_data_for_injection[['vst_raw']].copy()
        else:
            print("Critical: 'vst_raw' column is missing after data preparation. Exiting.")
            sys.exit(1)

    except Exception as e:
        print(f"Error during data loading/preparation for {station_id_to_plot}: {e}")
        traceback.print_exc()
        sys.exit(1)

    error_types_to_visualize = ['spike', 'offset', 'drift', 'noise', 'baseline shift', 'flatline']
    all_plot_data_collected = [] # List to store tuples of (original, modified, error_period_obj, error_type_name)

    for error_name in error_types_to_visualize:
        print(f"\n--- Processing data for {error_name} error plot ---")
        
        current_data_pristine = base_data_for_injection.copy()
        error_config_single = deepcopy(SYNTHETIC_ERROR_PARAMS)
        
        for err_key_cfg in error_config_single:
            if isinstance(error_config_single[err_key_cfg], dict) and 'count_per_year' in error_config_single[err_key_cfg]:
                error_config_single[err_key_cfg]['count_per_year'] = 0
        
        if error_name in error_config_single and isinstance(error_config_single[error_name], dict) and 'count_per_year' in error_config_single[error_name]:
            error_config_single[error_name]['count_per_year'] = 1 # Inject one instance
            print(f"Configured to inject 1 instance of {error_name}.")
        else:
            print(f"Warning: Config for '{error_name}' not found/malformed. Skipping.")
            continue

        error_generator_single = SyntheticErrorGenerator(config=error_config_single)
        data_to_modify = current_data_pristine.copy()
        modified_data_with_single_error = data_to_modify 

        print(f"Injecting {error_name} into a clean copy of the data segment...")
        try:
            if error_name == 'spike':
                modified_data_with_single_error = error_generator_single.inject_spike_errors(data_to_modify)
            elif error_name == 'offset':
                modified_data_with_single_error, _ = error_generator_single.inject_offset_errors(data_to_modify)
            elif error_name == 'drift':
                modified_data_with_single_error = error_generator_single.inject_drift_errors(data_to_modify)
            elif error_name == 'noise':
                modified_data_with_single_error = error_generator_single.inject_noise_errors(data_to_modify)
            elif error_name == 'baseline shift':
                modified_data_with_single_error = error_generator_single.inject_baseline_shift_errors(data_to_modify)
            elif error_name == 'flatline':
                modified_data_with_single_error, _ = error_generator_single.inject_flatline_errors(data_to_modify)
            else:
                print(f"Unknown error type '{error_name}' for injection. Skipping.")
                continue
        except Exception as e:
            print(f"Error during injection of {error_name}: {e}"); traceback.print_exc(); continue

        injected_error_details = [p for p in error_generator_single.error_periods if p.error_type == error_name]
        
        if not injected_error_details:
            print(f"No {error_name} error was injected. Skipping this type for the plot.")
            continue
        
        first_injected_error = injected_error_details[0]
        print(f"Successfully injected 1 instance of {error_name}.")
        
        if not isinstance(modified_data_with_single_error, pd.DataFrame):
             modified_data_with_single_error = pd.DataFrame(modified_data_with_single_error, 
                                                            index=current_data_pristine.index, 
                                                            columns=['vst_raw'])
        
        all_plot_data_collected.append((
            current_data_pristine.copy(), 
            modified_data_with_single_error.copy(), 
            first_injected_error, 
            error_name
        ))

    if all_plot_data_collected:
        print(f"\n--- Generating single stacked comparison plot for {len(all_plot_data_collected)} error types... ---")
        try:
            print(f"all_plot_data_collected: {all_plot_data_collected}")
            stacked_plot_path = plot_synthetic_vs_actual(
                all_error_data_to_plot=all_plot_data_collected,
                station_id_for_filename=station_id_to_plot,
                all_stations_data_dict=all_stations_loaded_data,
                output_dir=thesis_plots_output_dir
            )
            if stacked_plot_path:
                print(f"Stacked plot generation complete. Path: {stacked_plot_path}")
            else:
                print("Stacked plot generation did not return a valid path.")
        except Exception as e:
            print(f"Error during generation of stacked plot: {e}")
            traceback.print_exc()
    else:
        print("\nNo error types were successfully processed and injected. Skipping stacked plot generation.")

    print(f"\n--- Thesis plot generation process finished. ---")
    print(f"Note: The 'actual anomaly' side of the plots uses placeholder data (or provided examples) and may need further refinement.") 
