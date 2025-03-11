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

def plot_synthetic_errors(original_data: pd.DataFrame, 
                         modified_data: pd.DataFrame,
                         error_periods: List[ErrorPeriod],
                         station_name: str,
                         output_dir: Path):
    """Create visualization of synthetic errors."""
    diagnostic_dir = output_dir / "diagnostics" / "synthetic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Water Level Time Series with Synthetic Errors - {station_name}', fontsize=14)
    
    # Get the overall min and max values for the "Value" column,
    # using dropna() to avoid NaN values.
    value_col = "Value"
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
    plt.savefig(diagnostic_dir / f"{station_name}_synthetic_errors.png", dpi=300)
    plt.close()

def create_interactive_plot(original_data: pd.DataFrame, 
                          modified_data: pd.DataFrame,
                          error_periods: List[ErrorPeriod],
                          station_name: str,
                          output_dir: Path):
    """Create interactive plotly visualization of synthetic errors."""
    diagnostic_dir = output_dir / "diagnostics" / "synthetic"
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
        go.Scatter(x=original_data.index, y=original_data['Value'],
                  name='Original Data',
                  line=dict(color=ERROR_COLORS['base'], width=1),
                  opacity=0.7),
        row=1, col=1
    )
    
    # Plot base modified data
    fig.add_trace(
        go.Scatter(x=modified_data.index, y=modified_data['Value'],
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
                y=period_data['Value'],
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
    fig.write_html(diagnostic_dir / f"{station_name}_synthetic_errors_interactive.html")

def generate_synthetic_report(stations_results: dict, output_dir: Path):
    """Generate report of synthetic error injection results."""
    report_dir = output_dir / "diagnostics" / "synthetic"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "synthetic_error_report.txt", "w") as f:
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

def plot_synthetic_vs_actual(original_data, modified_data, error_periods, station_name, output_dir):
    """
    Create comparison plots between synthetic and actual anomalies for each error type.
    
    Args:
        original_data (pd.DataFrame): Original station data
        modified_data (pd.DataFrame): Data with synthetic errors
        error_periods (list): List of ErrorPeriod objects containing error information
        station_name (str): Name of the station
        output_dir (Path): Directory to save the plots
    """
    # Only process station 21006845
    if station_name != '21006845':
        return

    # Known periods of actual anomalies for station 21006845
    actual_anomalies = {
        'spike': [
            ('2019-06-15', '2019-06-16'),
            ('2019-08-20', '2019-08-21')
        ],
        'drift': [
            ('2019-03-01', '2019-03-15')
        ],
        'offset': [
            ('2019-07-01', '2019-07-10')
        ],
        'baseline_shift': [
            ('2019-09-01', '2019-09-30')
        ]
    }
    
    # Get unique error types from error_periods
    error_types = set(period.error_type for period in error_periods)
    
    # Create a subplot for each error type
    n_types = len(error_types)
    fig, axes = plt.subplots(n_types, 2, figsize=(20, 6*n_types))
    
    if n_types == 1:
        axes = axes.reshape(1, -1)
    
    # Define zoom windows for each error type
    zoom_windows = {
        'spike': pd.Timedelta(days=2),      # 2 days for spike
        'drift': pd.Timedelta(days=30),     # 30 days for drift
        'offset': pd.Timedelta(days=14),    # 14 days for offset
        'baseline_shift': pd.Timedelta(days=30)  # 30 days for baseline shift
    }
    
    for idx, error_type in enumerate(error_types):
        # Filter error periods for current type
        type_periods = [p for p in error_periods if p.error_type == error_type]
        
        if type_periods:
            # Find the most representative error period
            # For example, one with the largest magnitude or longest duration
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
            
            # Plot synthetic anomaly (zoomed)
            ax_synthetic = axes[idx, 0]
            
            # Calculate zoom window
            window_size = zoom_windows.get(error_type, pd.Timedelta(days=14))
            window_start = period.start_time - window_size/2
            window_end = period.end_time + window_size/2
            
            # Filter data for the window
            mask = (original_data.index >= window_start) & (original_data.index <= window_end)
            window_original = original_data[mask]
            window_modified = modified_data[mask]
            
            # Plot zoomed data
            ax_synthetic.plot(window_original.index, window_original['Value'], 
                            label='Original', color='blue', alpha=0.6)
            ax_synthetic.plot(window_modified.index, window_modified['Value'], 
                            label='With Synthetic Error', color='red', alpha=0.6)
            
            # Highlight synthetic error period
            ax_synthetic.axvspan(period.start_time, period.end_time, 
                               color='yellow', alpha=0.3)
            
            # Add error parameters to title
            params_str = ', '.join(f"{k}: {v:.2f}" if isinstance(v, float) 
                                 else f"{k}: {v}" 
                                 for k, v in period.parameters.items())
            ax_synthetic.set_title(f'Synthetic {error_type.capitalize()} Anomaly\n{params_str}')
        
        # Plot actual anomaly (full range)
        ax_actual = axes[idx, 1]
        ax_actual.plot(original_data.index, original_data['Value'], 
                      label='Original Data', color='blue')
        
        # Highlight actual anomaly periods
        if error_type in actual_anomalies:
            for start, end in actual_anomalies[error_type]:
                ax_actual.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                                color='red', alpha=0.3)
        
        ax_actual.set_title(f'Actual {error_type.capitalize()} Anomaly')
        
        # Format axes
        for ax in [ax_synthetic, ax_actual]:
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level (mm)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True)
            
            # Increase y-axis range by 20%
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.tight_layout()
    
    # Create diagnostics directory if it doesn't exist
    diagnostic_dir = output_dir / "diagnostics" / "synthetic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    output_path = diagnostic_dir / f'synthetic_vs_actual_comparison_{station_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() 