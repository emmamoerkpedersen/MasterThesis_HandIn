"""Diagnostic tools for synthetic error injection."""

from typing import List, Dict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _2_synthetic.synthetic_errors import ErrorPeriod

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
    
    # Get the overall min and max values
    y_min = min(original_data.min().min(), modified_data.min().min())
    y_max = max(original_data.max().max(), modified_data.max().max())
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding
    
    # Set the same y-axis limits for both plots
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
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
    
    # Plot original data in first subplot
    ax1.plot(original_data, label='Original Data', color=ERROR_COLORS['base'], alpha=0.7)
    ax1.set_title('Original Test Data')
    ax1.grid(False)
    ax1.legend()
    
    # Plot base data in second subplot
    ax2.plot(modified_data, color=ERROR_COLORS['base'], alpha=0.7, label='Base Data', zorder=1)
    
    # Track lines for legend
    error_lines = {error_type: None for error_type in ERROR_COLORS.keys() if error_type != 'base'}
    
    # Plot each error type
    for period in error_periods:
        error_type = period.error_type
        period_data = modified_data.loc[period.start_time:period.end_time]
        
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
        
        if error_lines[error_type] is None:
            error_lines[error_type] = line
    
    # Add legend with only the error types that were used
    legend_lines = [line for error_type, line in error_lines.items() 
                   if line is not None]
    legend_labels = [error_type.capitalize() for error_type, line in error_lines.items() 
                    if line is not None]
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

def generate_synthetic_report(stations_results: Dict, output_dir: Path):
    """Generate report of synthetic error injection results."""
    report_dir = output_dir / "diagnostics" / "synthetic"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "synthetic_error_report.txt", "w") as f:
        f.write("Synthetic Error Injection Report\n")
        f.write("==============================\n\n")
        
        for station_name, results in stations_results.items():
            f.write(f"\nStation: {station_name}\n")
            f.write("-" * (len(station_name) + 9) + "\n")
            
            # Count errors by type
            error_counts = {}
            for period in results['error_periods']:
                error_counts[period.error_type] = error_counts.get(period.error_type, 0) + 1
            
            f.write("\nInjected Errors:\n")
            for error_type, count in error_counts.items():
                f.write(f"  - {error_type.capitalize()}: {count} instances\n")
            
            # Add time range information
            if results['error_periods']:
                start_time = min(period.start_time for period in results['error_periods'])
                end_time = max(period.end_time for period in results['error_periods'])
                f.write(f"\nTime Range Affected:\n")
                f.write(f"  From: {start_time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"  To: {end_time.strftime('%Y-%m-%d %H:%M')}\n")
            
            f.write("\n" + "="*50 + "\n") 