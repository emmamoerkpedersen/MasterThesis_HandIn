"""Diagnostic tools for synthetic error injection."""

from typing import List, Dict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _2_synthetic.synthetic_errors import ErrorPeriod
import seaborn as sns
from config import SYNTHETIC_ERROR_PARAMS
import numpy as np

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
        'baseline_shift': '#8c564b',  # Brown
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
        
        elif error_type == 'baseline_shift':
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
        'baseline_shift': '#8c564b',  # Brown
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
        elif error_type == 'baseline_shift':
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

def plot_synthetic_vs_actual(original_data, modified_data, error_periods, station_name, output_dir) -> str:
    """
    Create comparison plots between synthetic and actual anomalies for each error type.
    Shows zoomed views of synthetic errors alongside real-world examples of similar anomalies.
    
    Returns:
        Path to the saved plot or None if not applicable
    """
    # Set publication-quality styling
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
    
    # Create a dictionary for actual anomalies (from real-world observations)
    # In a real application, this should come from labeled data
    actual_anomalies = {
        'spike': [
            ('2023-04-01', '2023-04-02'),  # Replace with actual anomalies when found
        ],
        'drift': [
            ('2023-02-01', '2023-02-15')   # Replace with actual anomalies when found
        ],
        'offset': [
            ('2022-06-01', '2022-06-10')   # Replace with actual anomalies when found
        ],
        'baseline_shift': [
            ('2022-09-01', '2022-09-30')   # Replace with actual anomalies when found
        ],
        'flatline': [
            ('2023-01-10', '2023-01-17')   # Replace with actual anomalies when found
        ],
        'noise': [
            ('2023-05-01', '2023-05-07')   # Replace with actual anomalies when found
        ]
    }
    
    # Get unique error types from error_periods
    error_types = set(period.error_type for period in error_periods)
    
    # If no error types found, return None
    if not error_types:
        print(f"No error types found for station {station_name}")
        return None
    
    # Create a subplot for each error type, with 2 columns (synthetic vs actual)
    n_types = len(error_types)
    fig, axes = plt.subplots(n_types, 2, figsize=(20, 5*n_types), dpi=300)
    
    if n_types == 1:
        axes = axes.reshape(1, -1)  # Make single axis row into 2D array for consistent indexing
    
    # Define zoom windows for each error type - using balanced windows
    zoom_windows = {
        'spike': pd.Timedelta(days=5),       # 5 days for spike
        'drift': pd.Timedelta(days=21),      # 21 days for drift
        'offset': pd.Timedelta(days=14),     # 14 days for offset
        'baseline_shift': pd.Timedelta(days=21),  # 21 days for baseline shift
        'flatline': pd.Timedelta(days=7),    # 7 days for flatline
        'noise': pd.Timedelta(days=7)        # 7 days for noise
    }
    
    for idx, error_type in enumerate(sorted(error_types)):
        # Filter error periods for current type
        type_periods = [p for p in error_periods if p.error_type == error_type]
        
        if not type_periods:
            continue
            
        # Find the most representative error period based on error type
        if error_type == 'spike':
            # Choose spike with largest magnitude
            period = max(type_periods, 
                       key=lambda p: abs(p.parameters.get('magnitude', 0)))
        elif error_type == 'drift':
            # Choose drift with longest duration
            period = max(type_periods, 
                       key=lambda p: p.parameters.get('duration', 0))
        elif error_type == 'baseline_shift':
            # Choose shift with largest magnitude
            period = max(type_periods, 
                       key=lambda p: abs(p.parameters.get('magnitude', 0)))
        else:
            # Default to first period
            period = type_periods[0]
        
        # Plot synthetic anomaly (left column)
        ax_synthetic = axes[idx, 0]
        
        # Calculate zoom window
        window_size = zoom_windows.get(error_type, pd.Timedelta(days=7))
        window_start = period.start_time - window_size
        window_end = period.end_time + window_size
        
        # Filter data for the window
        mask = (original_data.index >= window_start) & (original_data.index <= window_end)
        window_original = original_data[mask]
        window_modified = modified_data[mask]
        
        # Plot zoomed data
        ax_synthetic.plot(window_original.index, window_original['vst_raw'], 
                        label='Original', color='#1f77b4', linewidth=1.0)
        ax_synthetic.plot(window_modified.index, window_modified['vst_raw'], 
                        label='With Synthetic Error', color='#d62728', linewidth=1.0, linestyle='--')
        
        # Highlight synthetic error period with light yellow background
        ax_synthetic.axvspan(period.start_time, period.end_time, 
                           color='#ffffcc', alpha=0.5)
        
        # Add error parameters to title
        params_str = ', '.join(f"{k}: {v:.2f}" if isinstance(v, float) 
                             else f"{k}: {v}" 
                             for k, v in period.parameters.items()
                             if k not in ['original_values', 'modified_values'])
        
        ax_synthetic.set_title(f'Synthetic {error_type.title()} Error', fontweight='bold')
        
        # Format synthetic plot
        ax_synthetic.set_xlabel('Date', fontweight='bold', labelpad=10)
        ax_synthetic.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
        ax_synthetic.tick_params(axis='x', rotation=45)
        ax_synthetic.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
        
        # Remove top and right spines for cleaner look
        ax_synthetic.spines['top'].set_visible(False)
        ax_synthetic.spines['right'].set_visible(False)
        ax_synthetic.grid(False)
        
        # Plot actual anomaly example (right column)
        ax_actual = axes[idx, 1]
        
        # If we have examples of this anomaly type, highlight them
        if error_type in actual_anomalies and actual_anomalies[error_type]:
            # Get the first example
            start_date, end_date = actual_anomalies[error_type][0]
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # Create zoom window similar to synthetic plot
            context_window = zoom_windows.get(error_type, pd.Timedelta(days=7))
            context_start = start_ts - context_window
            context_end = end_ts + context_window
            
            # Only use dates that are within our data range
            valid_start = max(context_start, original_data.index.min())
            valid_end = min(context_end, original_data.index.max())
            
            # Filter data for the real anomaly window
            mask = (original_data.index >= valid_start) & (original_data.index <= valid_end)
            window_data = original_data[mask]
            
            # Plot the window data
            ax_actual.plot(window_data.index, window_data['vst_raw'], 
                         color='#1f77b4', linewidth=1.0, label='Actual Data')
            
            # Highlight the actual anomaly period
            ax_actual.axvspan(start_ts, end_ts, color='#ffffcc', alpha=0.5)
            
            # Plot the anomaly period with different style
            anomaly_mask = (original_data.index >= start_ts) & (original_data.index <= end_ts)
            anomaly_data = original_data[anomaly_mask]
            ax_actual.plot(anomaly_data.index, anomaly_data['vst_raw'], 
                         color='#ff7f0e', linewidth=1.5, 
                         label='Anomaly Period')
            
            ax_actual.set_title(f'Real {error_type.title()} Error Example', fontweight='bold')
            ax_actual.set_xlim(valid_start, valid_end)
        else:
            # If no real examples yet, show a message that this is a placeholder
            ax_actual.text(0.5, 0.5, f'Placeholder for real {error_type} error example\n(to be replaced with actual data)',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_actual.transAxes, fontsize=14, color='#555555')
            ax_actual.set_title(f'Real {error_type.title()} Error Example', fontweight='bold')
        
        # Format actual plot
        ax_actual.set_xlabel('Date', fontweight='bold', labelpad=10)
        ax_actual.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
        ax_actual.tick_params(axis='x', rotation=45)
        ax_actual.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
        
        # Remove top and right spines for cleaner look
        ax_actual.spines['top'].set_visible(False)
        ax_actual.spines['right'].set_visible(False)
        ax_actual.grid(False)
        
        # Auto-adjust y-limits based on data in view with a small margin
        for ax in [ax_synthetic, ax_actual]:
            if len(ax.lines) > 0:
                y_data = np.concatenate([line.get_ydata() for line in ax.lines])
                y_data = y_data[~np.isnan(y_data)]  # Remove NaNs
                if len(y_data) > 0:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    y_range = y_max - y_min
                    margin = 0.1 * y_range  # 10% margin
                    ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    
    # Create diagnostics directory if it doesn't exist
    diagnostic_dir = output_dir / "synthetic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    output_path = diagnostic_dir / f'synthetic_vs_actual_comparison_{station_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved synthetic vs actual comparison plot for {station_name} to {output_path}")
    
    return str(output_path)

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
                original_data=original_data,
                modified_data=result_data['modified_data'],
                error_periods=result_data['error_periods'],
                station_name=station_key,
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