import os
import webbrowser
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to the Project_Code - CORRECT folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

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

def create_full_plot(test_data, test_predictions, station_id, model_config=None, best_val_loss=None, create_html=True, open_browser=True, metrics=None, title_suffix=None, show_config=False, synthetic_data=None, vinge_data=None, output_dir=None):
    """
    Create an interactive plot with aligned datetime indices, rainfall data, and model configuration.
    
    Args:
        test_data: DataFrame containing test data with datetime index
        test_predictions: DataFrame or Series containing predictions
        station_id: ID of the station being analyzed
        model_config: Optional dictionary containing model configuration parameters
        best_val_loss: Optional best validation loss achieved during training
        create_html: Whether to create HTML plot (default: True)
        open_browser: Whether to open the plot in browser (default: True)
        metrics: Optional dictionary with additional performance metrics
        title_suffix: Optional suffix to add to the plot title
        show_config: Whether to show model configuration (default: False)
        synthetic_data: Optional DataFrame containing data with synthetic errors
        vinge_data: Optional DataFrame containing manual board (VINGE) measurements
        output_dir: Optional output directory. If None, uses default location.
    """
    # Use provided output_dir or default location
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    station_id = str(station_id)
    
    # # Create the title with optional suffix
    # title = f'Prediction Analysis for Station {station_id}'
    # if title_suffix:
    #     title = f'{title} - {title_suffix}'
    
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']
    
    # Get rainfall data without resampling
    rainfall_data = None
    if 'rainfall' in test_data.columns:
        rainfall_data = test_data['rainfall']
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Extract predictions values - handle both DataFrame and Series inputs
    if isinstance(test_predictions, pd.DataFrame):
        predictions_values = test_predictions['vst_raw'].values
    else:
        predictions_values = test_predictions.values if isinstance(test_predictions, pd.Series) else test_predictions
    
    # Create a Series of NaN values with the same length as test_actual
    predictions_series = pd.Series(
        index=test_actual.index,
        data=np.nan
    )

    # Get sequence length from model config, default to 50 if not specified
    sequence_length = 1

    if model_config and model_config.get('model_type') == 'iterative':
        prediction_window = 1
        
        # Place predictions at their correct future positions
        if len(predictions_values) > 0:
            # Calculate the start index for predictions (sequence_length steps from the start)
            start_idx = sequence_length
            # Place each prediction at its correct future position
            for i in range(0, len(predictions_values), prediction_window):
                if start_idx + i + prediction_window <= len(test_actual):
                    predictions_series.iloc[start_idx + i:start_idx + i + prediction_window] = predictions_values[i:i + prediction_window]
    else:
        # For standard model, still account for sequence_length offset
        if len(predictions_values) > 0:
            # Place predictions starting from sequence_length
            end_idx = min(sequence_length + len(predictions_values), len(test_actual))
            predictions_series.iloc[sequence_length:end_idx] = predictions_values[:end_idx-sequence_length]
    
    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create error periods directory
    error_periods_dir = output_dir / 'error_periods'
    error_periods_dir.mkdir(exist_ok=True)
    
    # Extract error periods and synthetic data DataFrame from synthetic_data dict if available
    error_periods = []
    synthetic_df = None
    if synthetic_data is not None:
        if isinstance(synthetic_data, dict):
            error_periods = synthetic_data.get('error_periods', [])
            synthetic_df = synthetic_data.get('data', None)
        elif hasattr(synthetic_data, 'error_periods'):
            error_periods = synthetic_data.error_periods
            synthetic_df = synthetic_data
        elif isinstance(synthetic_data, pd.DataFrame):
            synthetic_df = synthetic_data
    
    # Align vinge_data index with test_data
    if vinge_data is not None:
        vinge_data = vinge_data.reindex(test_data.index)

    # Always create the PNG version with config text at the bottom
    png_path = create_water_level_plot_png(
        test_actual, 
        predictions_series, 
        station_id, 
        timestamp, 
        model_config, 
        output_dir,
        best_val_loss,
        metrics=metrics,
        show_config=show_config,
        title_suffix=title_suffix,
        synthetic_data=synthetic_data,  # Pass the full structure with both data and error_periods
        vinge_data=vinge_data  # Pass vinge_data to the PNG creation function
    )
    
    # Create the HTML version only if requested
    if create_html:
        # Create subplots - rainfall on top, water level on bottom
        subplot_rows = 2 if rainfall_data is not None else 1
        
        if rainfall_data is not None:
            specs = [[{"secondary_y": False}] for _ in range(subplot_rows)]
            subplot_titles = [
                'Rainfall',
                f'Water Level - Station {station_id}'
            ]
            
            fig = make_subplots(
                rows=subplot_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=subplot_titles,
                specs=specs,
                row_heights=[0.3, 0.7]  # Give more space to rainfall
            )
            
            # Add rainfall data to first subplot (top)
            fig.add_trace(
                go.Bar(
                    x=rainfall_data.index,
                    y=rainfall_data.values,
                    name="Rainfall",
                    marker_color='rgba(0, 0, 255, 0.6)',  # More visible blue
                    opacity=0.8,
                    width=60*60*1000,  # 1-hour width in milliseconds
                    yaxis="y1"
                ),
                row=1, col=1
            )
            
            # Add water level data to second subplot (bottom)
            fig.add_trace(
                go.Scatter(
                    x=test_actual.index,
                    y=test_actual.values,
                    name="VST RAW (Clean)",
                    line=dict(color='#1f77b4', width=1)
                ),
                row=2, col=1
            )
            
            # Add VINGE data if available
            if vinge_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=vinge_data.index,
                        y=vinge_data['vinge'].values,
                        name="VINGE Data",
                        mode='markers',
                        marker=dict(size=8, color='#ff7f0e'),
                    ),
                    row=2, col=1
                )
            if vinge_data is None:
                print(f"VINGE data is None for station {station_id}")
            
            # Add synthetic error data if available
            if synthetic_df is not None:
                # Add colored background regions for different error types
                error_colors = {
                    'spike': 'rgba(255, 0, 0, 0.1)',      # Light red
                    'flatline': 'rgba(0, 255, 0, 0.1)',   # Light green
                    'drift': 'rgba(0, 0, 255, 0.1)',      # Light blue
                    'offset': 'rgba(255, 165, 0, 0.1)',   # Light orange
                    'baseline_shift': 'rgba(128, 0, 128, 0.1)',  # Light purple
                    'noise': 'rgba(128, 128, 128, 0.1)'   # Light gray
                }
                
                # Add error type backgrounds
                for error_type, color in error_colors.items():
                    # Find periods with this error type
                    type_periods = [p for p in error_periods if p.error_type == error_type]
                    
                    for period in type_periods:
                        fig.add_shape(
                            type="rect",
                            x0=period.start_time,
                            x1=period.end_time,
                            y0=test_actual.min(),
                            y1=test_actual.max(),
                            fillcolor=color,
                            opacity=0.3,
                            layer="below",
                            line_width=0,
                            row=2, col=1
                        )
                
                # Add synthetic data line
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_df.index,
                        y=synthetic_df['vst_raw'].values,
                        name="VST RAW (with Synthetic Errors)",
                        line=dict(color='#d62728', width=1, dash='dot')
                    ),
                    row=2, col=1
                )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predictions",
                    line=dict(color='#2ca02c', width=1)
                ),
                row=2, col=1
            )
            
            # Update y-axes labels and ranges
            fig.update_yaxes(
                title_text="Rainfall (mm)",
                row=1, col=1,
                autorange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangemode='nonnegative'
            )
            
            fig.update_yaxes(
                title_text="Water Level (mm)",
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            
            # Update x-axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangeslider_visible=False,
                row=1, col=1
            )
            
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangeslider_visible=True,
                row=2, col=1
            )
            
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=test_actual.index,
                    y=test_actual.values,
                    name="VST RAW (Clean)",
                    line=dict(color='#1f77b4', width=1)
                )
            )
            
            # Add VINGE data if available
            if vinge_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=vinge_data.index,
                        y=vinge_data['vinge'].values,
                        name="VINGE Data",
                        # Should be shown as dots
                        mode='markers',
                        marker=dict(size=8, color='#ff7f0e'),
                    ),
                    row=2, col=1
                )
            
            # Add synthetic error data if available
            if synthetic_df is not None:
                # Add colored background regions for different error types
                error_colors = {
                    'spike': 'rgba(255, 0, 0, 0.1)',      # Light red
                    'flatline': 'rgba(0, 255, 0, 0.1)',   # Light green
                    'drift': 'rgba(0, 0, 255, 0.1)',      # Light blue
                    'offset': 'rgba(255, 165, 0, 0.1)',   # Light orange
                    'baseline_shift': 'rgba(128, 0, 128, 0.1)',  # Light purple
                    'noise': 'rgba(128, 128, 128, 0.1)'   # Light gray
                }
                
                # Add error type backgrounds
                for error_type, color in error_colors.items():
                    # Find periods with this error type
                    type_periods = [p for p in error_periods if p.error_type == error_type]
                    
                    for period in type_periods:
                        fig.add_shape(
                            type="rect",
                            x0=period.start_time,
                            x1=period.end_time,
                            y0=test_actual.min(),
                            y1=test_actual.max(),
                            fillcolor=color,
                            opacity=0.3,
                            layer="below",
                            line_width=0
                        )
                
                # Add synthetic data line
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_df.index,
                        y=synthetic_df['vst_raw'].values,
                        name="VST RAW (with Synthetic Errors)",
                        line=dict(color='#d62728', width=1, dash='dot')
                    )
                )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predictions",
                    line=dict(color='#2ca02c', width=1)
                )
            )
        
        # Update layout
        fig.update_layout(
            width=1500,  # Back to reasonable width
            height=1000,  # Keep height
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=80, t=180, b=80),  # Increased top margin for config box
            dragmode='zoom'  # Enable box zooming in both directions
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        
        # Create model configuration text
        config_text = ""
        if model_config:
            # Create a comprehensive config text with all important parameters
            config_lines = [
                f"Model Configuration:<br>",
                f"Architecture: LSTM<br>",
                f"Hidden Size: {model_config.get('hidden_size', 'N/A')}<br>",
                f"Layers: {model_config.get('num_layers', 'N/A')}<br>",
                f"Dropout: {model_config.get('dropout', 'N/A')}<br>",
                f"Batch Size: {model_config.get('batch_size', 'N/A')}<br>",
                f"Sequence Length: {model_config.get('sequence_length', 'N/A')}<br>",
                f"Learning Rate: {model_config.get('learning_rate', 0.001):.6f}<br>",
                f"Epochs: {model_config.get('epochs', 'N/A')}<br>",
                f"Patience: {model_config.get('patience', 'N/A')}<br>",
                f"Loss Function: {model_config.get('objective_function', 'N/A')}<br>"
            ]
            
            # Add best validation loss if provided
            if best_val_loss is not None:
                config_lines.append(f"Best Val Loss: {best_val_loss:.6f}<br>")
                
            # Add optional configuration parameters if they exist
            if 'peak_weight' in model_config:
                config_lines.append(f"Peak Weight: {model_config.get('peak_weight', 'N/A')}<br>")
            
            if 'grad_clip_value' in model_config:
                config_lines.append(f"Gradient Clip: {model_config.get('grad_clip_value', 'N/A')}<br>")
                
            if 'use_smoothing' in model_config:
                config_lines.append(f"Use Smoothing: {model_config.get('use_smoothing', False)}<br>")
                
            if model_config.get('use_smoothing', False) and 'smoothing_alpha' in model_config:
                config_lines.append(f"Smoothing Alpha: {model_config.get('smoothing_alpha', 'N/A')}<br>")
                
            config_lines.append(f"Time Features: {model_config.get('use_time_features', False)}<br>")
            config_lines.append(f"Cumulative Features: {model_config.get('use_cumulative_features', False)}<br>")
            
            # Feature columns
            if 'feature_cols' in model_config:
                features_str = ', '.join(model_config.get('feature_cols', []))
                config_lines.append(f"Features: {features_str}<br>")
            
            # Create final text
            config_text = ''.join(config_lines)
        
        # Add model configuration text in a box at the top of the plot
        if model_config:
            # Create a more compact version of the config text for the top box
            compact_lines = []
            
            # First row: Architecture, Hidden Size, Layers, Batch Size
            row1 = f"Architecture: LSTM | Hidden Size: {model_config.get('hidden_size', 'N/A')} | "
            row1 += f"Layers: {model_config.get('num_layers', 'N/A')} | Dropout: {model_config.get('dropout', 'N/A')} | "
            row1 += f"Batch: {model_config.get('batch_size', 'N/A')}"
            compact_lines.append(row1)
            
            # Second row: Learning Rate, Sequence Length, Loss Function, Best Val Loss
            row2 = f"Learning Rate: {model_config.get('learning_rate', 0.001):.6f} | "
            row2 += f"Sequence Length: {model_config.get('sequence_length', 'N/A')} | "
            row2 += f"Loss Function: {model_config.get('objective_function', 'N/A')}"
            if best_val_loss is not None:
                row2 += f" | Best Val Loss: {best_val_loss:.6f}"
            compact_lines.append(row2)
            
            # Third row: Peak Weight, Gradient Clip, Features
            row3 = ""
            if 'peak_weight' in model_config:
                row3 += f"Peak Weight: {model_config.get('peak_weight', 'N/A')} | "
            if 'grad_clip_value' in model_config:
                row3 += f"Grad Clip: {model_config.get('grad_clip_value', 'N/A')} | "
            row3 += f"Time Features: {model_config.get('use_time_features', False)} | "
            row3 += f"Cumulative Features: {model_config.get('use_cumulative_features', False)}"
            compact_lines.append(row3)
            
            # Create compact text
            compact_text = "<br>".join(compact_lines)
            
            # Add as annotation at the top of the plot
            fig.add_annotation(
                x=0.5,  # Center of the plot
                y=1.50,  # Just above the plot title
                xref="paper", 
                yref="paper",
                text=compact_text,
                showarrow=False,
                font=dict(size=12),
                align="center",
                bgcolor="rgba(240, 240, 240, 0.9)",
                bordercolor="#000000",
                borderwidth=1,
                borderpad=8,
                width=1200  # Wide enough for the text
            )
        
        # Add metrics annotation if metrics are provided
        if metrics and any(not np.isnan(v) for v in metrics.values()):
            metrics_lines = []
            
            # General metrics section
            general_metrics = []
            if 'rmse' in metrics and not np.isnan(metrics['rmse']):
                general_metrics.append(f"RMSE: {metrics['rmse']:.4f} mm")
            if 'mae' in metrics and not np.isnan(metrics['mae']):
                general_metrics.append(f"MAE: {metrics['mae']:.4f} mm")
            if 'r2' in metrics and not np.isnan(metrics['r2']):
                general_metrics.append(f"R²: {metrics['r2']:.4f}")
            
            # Peak metrics section
            peak_metrics = []
            if 'peak_rmse' in metrics and not np.isnan(metrics['peak_rmse']):
                peak_metrics.append(f"Peak RMSE: {metrics['peak_rmse']:.4f} mm")
            if 'peak_mae' in metrics and not np.isnan(metrics['peak_mae']):
                peak_metrics.append(f"Peak MAE: {metrics['peak_mae']:.4f} mm")
            
            # Create combined text
            if general_metrics and peak_metrics:
                metrics_text = (
                    "<b>Performance Metrics:</b> " + 
                    ", ".join(general_metrics) + 
                    " | <b>Peak Performance:</b> " + 
                    ", ".join(peak_metrics)
                )
            elif general_metrics:
                metrics_text = "<b>Performance Metrics:</b> " + ", ".join(general_metrics)
            elif peak_metrics:
                metrics_text = "<b>Peak Performance:</b> " + ", ".join(peak_metrics)
            else:
                metrics_text = None
            
            if metrics_text:
                # Add as annotation below the config text
                fig.add_annotation(
                    x=0.5,  # Center of the plot
                    y=0.99,  # Just below the config box
                    xref="paper", 
                    yref="paper",
                    text=metrics_text,
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                    bgcolor="rgba(217, 237, 247, 0.9)",  # Light blue background
                    bordercolor="#31708f",  # Blue border
                    borderwidth=1,
                    borderpad=8,
                    width=1200  # Wide enough for the text
                )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="all")
                ]),
                bgcolor='rgba(150, 200, 250, 0.4)',
                activecolor='rgba(100, 150, 200, 0.8)'
            ),
            row=subplot_rows, col=1
        )
        
        # Save HTML plot
        html_path = output_dir / f'predictions_station_{station_id}_{timestamp}.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True, config={
            'displayModeBar': True,
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'station_{station_id}_prediction_{timestamp}',
                'height': 1000,
                'width': 1500,
                'scale': 2
            }
        })
        
        # Create focused error period plots as PNGs, max 2 per error type
        if error_periods:
            print(f"Creating focused error period PNG plots for up to 2 of each error type...")
            error_type_counts = defaultdict(int)
            for i, period in enumerate(error_periods):
                etype = period.error_type
                if error_type_counts[etype] >= 5:
                    continue
                error_type_counts[etype] += 1

                # Prepare zoomed-in range (add margin before/after)
                margin = pd.Timedelta(hours=24)
                x_start = period.start_time - margin
                x_end = period.end_time + margin

                # Slice data to the zoomed-in window
                mask = (test_actual.index >= x_start) & (test_actual.index <= x_end)
                y_clean = test_actual[mask].values
                y_pred = predictions_series[mask].values
                y_synth = synthetic_df['vst_raw'][mask].values if synthetic_df is not None else None

                # Compute y-limits based on all available data in window
                y_all = [y_clean, y_pred]
                if y_synth is not None:
                    y_all.append(y_synth)
                y_concat = np.concatenate([arr[~np.isnan(arr)] for arr in y_all if arr is not None and len(arr) > 0])
                if len(y_concat) > 0:
                    ymin, ymax = np.min(y_concat), np.max(y_concat)
                    yrange = ymax - ymin
                    ypad = yrange * 0.05 if yrange > 0 else 1.0
                    ylims = (ymin - ypad, ymax + ypad)
                else:
                    ylims = None

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

                # Plot clean data
                ax.plot(test_actual.index, test_actual.values, color='#1f77b4', linewidth=0.8, label='VST RAW')
                # Plot synthetic data if available
                if synthetic_df is not None:
                    ax.plot(synthetic_df.index, synthetic_df['vst_raw'].values, color='#d62728', linewidth=0.8, linestyle='--', label='VST RAW (with Synthetic Errors)')
                # Plot predictions
                ax.plot(predictions_series.index, predictions_series.values, color='#2ca02c', linewidth=0.8, label='Predicted')

                # Set x-limits to focus on error period
                ax.set_xlim([x_start, x_end])
                # Set y-limits for better visibility
                if ylims is not None:
                    ax.set_ylim(ylims)
                ax.set_xlabel('Date', fontweight='bold')
                ax.set_ylabel('Water Level (mm)', fontweight='bold')
                ax.set_title(f'Error Period: {etype.title()} ({period.start_time.date()} to {period.end_time.date()})', fontweight='bold')
                ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
                fig.autofmt_xdate()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()

                # Save PNG
                error_plot_path = error_periods_dir / f'error_period_{etype}_{error_type_counts[etype]}_{timestamp}.png'
                plt.savefig(error_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
        
        # Open HTML in browser if requested
        if open_browser:
            absolute_path = os.path.abspath(html_path)
            print(f"Opening plot in browser: {absolute_path}")
            webbrowser.open('file://' + str(absolute_path))
        
    return png_path

def create_water_level_plot_png(actual, predictions, station_id, timestamp, model_config=None, output_dir=None, best_val_loss=None, metrics=None, vinge_data=None, show_config=False, title_suffix=None, synthetic_data=None, anomalies=None):
    """
    Create a publication-quality matplotlib plot with just water level data and save as PNG.
    Designed for thesis report with consistent colors and clean styling.
    
    Args:
        actual: Series containing actual values
        predictions: Series containing predicted values
        station_id: ID of the station
        timestamp: Timestamp for filename
        model_config: Optional model configuration dictionary
        output_dir: Optional output directory path
        best_val_loss: Optional best validation loss achieved during training
        metrics: Optional dictionary with additional performance metrics
        vinge_data: Optional DataFrame containing manual board (VINGE) measurements
        title_suffix: Optional suffix to add to the plot title
        synthetic_data: Optional DataFrame containing data with synthetic errors
        anomalies: Optional Boolean array indicating anomaly locations
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
        output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create figure with good dimensions for publication
    # Adjusted height for better space utilization
    fig = plt.figure(figsize=(12, 3), dpi=300)  # Changed from (12, 8.5) to (12, 3)
    
    # Create a larger figure to accommodate the config text and metrics
    # Add a third row for metrics if available
    if metrics and any(not np.isnan(v) for v in metrics.values()):
        gs = plt.GridSpec(3, 1, height_ratios=[6, 0.8, 0.8], hspace=0.15)  # Reduced spacing
    else:
        gs = plt.GridSpec(2, 1, height_ratios=[6, 0.8], hspace=0.15)  # Reduced spacing
    
    # Main plot in top section
    ax = fig.add_subplot(gs[0])
    
    # Extract error periods from synthetic data if available
    error_periods = []
    synthetic_df = None
    if synthetic_data is not None:
        if isinstance(synthetic_data, dict):
            error_periods = synthetic_data.get('error_periods', [])
            synthetic_df = synthetic_data.get('data', None)
        elif hasattr(synthetic_data, 'error_periods'):
            error_periods = synthetic_data.error_periods
            synthetic_df = synthetic_data
        elif isinstance(synthetic_data, pd.DataFrame):
            synthetic_df = synthetic_data
    
    # Add colored background regions for different error types if synthetic data is available
    if error_periods:
        error_colors = {
            'spike': '#ffcccc',      # Light red
            'flatline': '#ccffcc',   # Light green
            'drift': '#ccccff',      # Light blue
            'offset': '#ffd699',     # Light orange
            'baseline_shift': '#e6ccff',  # Light purple
            'noise': '#e6e6e6'       # Light gray
        }
        
        # Create a mapping to track which error types have been added to legend
        error_legend_added = {}
        
        # Add error type backgrounds
        for error_type, color in error_colors.items():
            # Find periods with this error type
            type_periods = [p for p in error_periods if p.error_type == error_type]
            
            for period in type_periods:
                # Add to legend only for the first occurrence of each type
                label = f'{error_type.title()} Error' if error_type not in error_legend_added else None
                error_legend_added[error_type] = True
                
                ax.axvspan(
                    period.start_time,
                    period.end_time,
                    color=color,
                    alpha=0.3,
                    label=label
                )
    
    # Plot with consistent colors - blue for actual, red for predicted
    ax.plot(actual.index, actual.values, color='#1f77b4', linewidth=0.8, label='VST RAW')
    
    # Add synthetic data if available
    if synthetic_df is not None:
        ax.plot(synthetic_df.index, synthetic_df['vst_raw'].values, color='#d62728', linewidth=0.8, linestyle='--', label='VST RAW (with Synthetic Errors)')
    
    ax.plot(predictions.index, predictions.values, color='#2ca02c', linewidth=0.8, label='Predicted')
    
    # Add VINGE data if available
    if vinge_data is not None:
        ax.plot(vinge_data.index, vinge_data['vinge'].values, 'o', color='#ff7f0e', label='VINGE Data')
    
    # Clean styling
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
    
    # No grid lines as requested
    ax.grid(False)
    
    
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', 
             loc='upper center', bbox_to_anchor=(0.5, 1.15), 
             ncol=4, handletextpad=0.5, columnspacing=1.0)
    
    # Format the date axis
    fig.autofmt_xdate(bottom=0.2)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add configuration text in the middle section if model_config is provided
    if model_config and show_config is True:
        # Create a new axes for the config text
        ax_config = fig.add_subplot(gs[1])
        ax_config.axis('off')  # Hide axes
        
        # Prepare configuration text
        config_lines = []
        
        # First row: Architecture, Hidden Size, Layers, Batch Size
        config_lines.append(f"Architecture: LSTM | Hidden Size: {model_config.get('hidden_size', 'N/A')} | Layers: {model_config.get('num_layers', 'N/A')} | Dropout: {model_config.get('dropout', 'N/A')} | Batch: {model_config.get('batch_size', 'N/A')}")
        
        # Second row: Learning Rate, Sequence Length, Loss Function, Best Val Loss
        row2 = f"Learning Rate: {model_config.get('learning_rate', 0.001):.6f} | Sequence Length: {model_config.get('sequence_length', 'N/A')} | Loss Function: {model_config.get('objective_function', 'N/A')}"
        if best_val_loss is not None:
            row2 += f" | Best Val Loss: {best_val_loss:.6f}"
        config_lines.append(row2)
        
        # Third row: Peak Weight, Gradient Clip, Features
        row3 = ""
        if 'peak_weight' in model_config:
            row3 += f"Peak Weight: {model_config.get('peak_weight', 'N/A')} | "
        if 'grad_clip_value' in model_config:
            row3 += f"Grad Clip: {model_config.get('grad_clip_value', 'N/A')} | "
        row3 += f"Time Features: {model_config.get('use_time_features', False)} | Cumulative Features: {model_config.get('use_cumulative_features', False)}"
        config_lines.append(row3)
        
        # Join the lines with newlines and add to the plot
        config_text = '\n'.join(config_lines)
        ax_config.text(0.5, 0.5, config_text, 
                     ha='center', va='center', 
                     fontsize=10,
                     transform=ax_config.transAxes,
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='#f0f0f0', 
                              edgecolor='#cccccc',
                              alpha=0.9))
    
    # Add metrics text in the bottom section if metrics are provided and have values
    if metrics and any(not np.isnan(v) for v in metrics.values()):
        # Create a new axes for the metrics text
        ax_metrics = fig.add_subplot(gs[2])
        ax_metrics.axis('off')  # Hide axes
        
        # Create two columns for different types of metrics
        general_metrics = []
        peak_metrics = []
        
        # Fix extreme R² value
        if 'r2' in metrics and not np.isnan(metrics['r2']):
            # Constrain R² to a reasonable range for display
            r2_value = metrics['r2']
            if r2_value < -1:
                # For very negative values, display as -1 with a note
                displayed_r2 = -1.0
                general_metrics.append(f"R²: {displayed_r2:.3f} (very poor fit)")
            else:
                general_metrics.append(f"R²: {r2_value:.3f}")
        
        # Add standard metrics to general column
        if 'rmse' in metrics and not np.isnan(metrics['rmse']):
            general_metrics.append(f"RMSE: {metrics['rmse']:.4f} mm")
        if 'mae' in metrics and not np.isnan(metrics['mae']):
            general_metrics.append(f"MAE: {metrics['mae']:.4f} mm")
        if 'mean_error' in metrics and not np.isnan(metrics['mean_error']):
            general_metrics.append(f"Mean Error: {metrics['mean_error']:.4f} mm")
        
        # Add peak metrics to the peak column
        if 'peak_rmse' in metrics and not np.isnan(metrics['peak_rmse']):
            peak_metrics.append(f"Peak RMSE: {metrics['peak_rmse']:.4f} mm")
        if 'peak_mae' in metrics and not np.isnan(metrics['peak_mae']):
            peak_metrics.append(f"Peak MAE: {metrics['peak_mae']:.4f} mm")
        
        # Create metrics headings
        general_header = "Performance Metrics:"
        peak_header = "Peak Performance Metrics:"
        
        # Display the metrics
        if general_metrics and peak_metrics:
            # Two column layout with both general and peak metrics
            general_text = general_header + "\n" + "\n".join(general_metrics)
            peak_text = peak_header + "\n" + "\n".join(peak_metrics)
            
            # Left column - general metrics
            ax_metrics.text(0.25, 0.5, general_text, 
                         ha='center', va='center', 
                         fontsize=10,
                         transform=ax_metrics.transAxes,
                         bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='#e6f2ff', 
                                  edgecolor='#3399ff',
                                  alpha=0.9))
            
            # Right column - peak metrics
            ax_metrics.text(0.75, 0.5, peak_text, 
                         ha='center', va='center', 
                         fontsize=10,
                         transform=ax_metrics.transAxes,
                         bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='#ffe6e6', 
                                  edgecolor='#ff6666',
                                  alpha=0.9))
        elif general_metrics:
            # Single column with just general metrics
            general_text = general_header + "\n" + "\n".join(general_metrics)
            ax_metrics.text(0.5, 0.5, general_text, 
                         ha='center', va='center', 
                         fontsize=10,
                         transform=ax_metrics.transAxes,
                         bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='#e6f2ff', 
                                  edgecolor='#3399ff',
                                  alpha=0.9))
        elif peak_metrics:
            # Single column with just peak metrics
            peak_text = peak_header + "\n" + "\n".join(peak_metrics)
            ax_metrics.text(0.5, 0.5, peak_text, 
                         ha='center', va='center', 
                         fontsize=10,
                         transform=ax_metrics.transAxes,
                         bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor='#ffe6e6', 
                                  edgecolor='#ff6666',
                                  alpha=0.9))
    
    # Tight layout and save with high quality
    plt.tight_layout()
    output_path = output_dir / f'water_level_station_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved water level PNG plot to: {output_path}")
    plt.close()
    
    return output_path

def plot_convergence(history, station_id, title=None, output_dir=None):
    """
    Plot training and validation loss over epochs, with learning rate changes.
    
    Args:
        history: Dictionary containing training history
        station_id: ID of the station
        title: Optional plot title
        output_dir: Optional output directory. If None, uses default location.
    """
    # Use provided output_dir or default location
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots - one for loss, one for learning rate
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    
    # Add smoothed validation loss if available
    if 'smoothed_val_loss' in history:
        ax1.plot(history['smoothed_val_loss'], 
                label='Smoothed Val Loss', 
                color='purple', 
                linestyle='--',
                linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(title if title else f'Training and Validation Loss - Station {station_id}')
    ax1.grid(True)
    ax1.legend()
    
    # Add learning rate plot if available
    if 'learning_rates' in history:
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(history['learning_rates'], 'g-', label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')  # Log scale for better visualization
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    output_path = output_dir / f'convergence_plot_{station_id}.png'
    plt.savefig(output_path)
    print(f"Saved convergence plot to: {output_path}")
    plt.close()

def plot_features_stacked_plots(data, feature_cols, output_dir=None, years_to_show=3):
    """
    Create a publication-quality plot of engineered features, organized by station.
    Each station has its own subplot showing its rainfall-related features.
    Temperature and water level data are shown in separate subplots.
    
    Args:
        data: DataFrame containing feature data
        feature_cols: List of feature column names
        output_dir: Optional output directory path (default: saves to results/feature_plots)
        years_to_show: Number of most recent years to display (default: 3)
    
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/feature_plots"))
        output_dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter data to only show the most recent years
    if years_to_show > 0 and isinstance(data.index, pd.DatetimeIndex):
        end_date = data.index.max()
        start_date = end_date - pd.DateOffset(years=years_to_show)
        filtered_data = data[data.index >= start_date].copy()
        
        # Only use filtered data if it's not empty
        if not filtered_data.empty:
            print(f"Limiting feature plot to the most recent {years_to_show} years: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            data = filtered_data
        else:
            print(f"Warning: No data in the last {years_to_show} years. Using all available data.")
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

    # Define colors for different feature types
    short_window_colors = {
        '90hour': '#fde725',    # Yellow for longest window (90h)
        '48hour': '#22a884',    # Green for second longest (48h)
        '7hour': '#414487',     # Blue for 7h
        '1hour': '#440154'      # Purple for shortest window (1h)
    }
    
    long_window_colors = {
        '1year': '#fde725',     # Yellow for longest window (1y)
        '6months': '#22a884',   # Green for second longest (6m)
        '3months': '#414487',   # Blue for 3m
        '1month': '#440154'     # Purple for shortest window (1m)
    }
    
    temp_colors = {
        '45': '#ff1a1a',  # Bright red
        '47': '#990000'   # Dark red
    }

    # Create figure with 8 subplots stacked vertically
    fig = plt.figure(figsize=(14, 24))  # Increased height to accommodate 8 plots
    gs = GridSpec(8, 1, height_ratios=[1]*8, hspace=0.2)  # Reduced hspace for tighter spacing

    # 1-3. Short-window precipitation features (3 plots, one per station)
    stations = ['46', '45', '47']
    windows = ['1hour', '7hour', '48hour', '90hour']
    
    for i, station in enumerate(stations):
        ax = fig.add_subplot(gs[i])
        for window in windows:
            column = f'station_{station}_rain_{window}'
            if column in data.columns:
                ax.plot(data.index, data[column],
                       label=f'{window} cumulative',
                       linewidth=0.8,
                       alpha=0.9,
                       color=short_window_colors[window])
        
        ax.set_ylabel('Precipitation (mm)', fontweight='bold', labelpad=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        plt.setp(ax.get_xticklabels(), visible=False)  # Hide x-axis labels

    # 4-6. Long-window precipitation features (3 plots, one per station)
    windows = ['1month', '3months', '6months', '1year']
    
    for i, station in enumerate(stations):
        ax = fig.add_subplot(gs[i+3])
        for window in windows:
            column = f'station_{station}_rain_{window}'
            if column in data.columns:
                ax.plot(data.index, data[column],
                       label=f'{window} cumulative',
                       linewidth=0.8,
                       alpha=0.9,
                       drawstyle='steps-post',
                       color=long_window_colors[window])
        
        ax.set_ylabel('Precipitation (mm)', fontweight='bold', labelpad=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        plt.setp(ax.get_xticklabels(), visible=False)  # Hide x-axis labels

    # 7. Temperature features
    ax = fig.add_subplot(gs[6])
    temp_columns = [col for col in data.columns if 'temperature' in col.lower()]
    for col in temp_columns:
        if "45" in col:
            station = "'45"
            color = temp_colors['45']
        else:
            station = "'47"
            color = temp_colors['47']
        
        ax.plot(data.index, data[col],
               label=f'Temperature Station {station}',
               linewidth=0.8,
               alpha=0.9,
               color=color)
    
    ax.set_ylabel('Temperature (°C)', fontweight='bold', labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), visible=False)  # Hide x-axis labels

    # 8. Time-based features (bottom subplot)
    ax = fig.add_subplot(gs[7])
    
    # Plot month features
    if 'month_sin' in data.columns:
        ax.plot(data.index, data['month_sin'],
               label='Month (sin)',
               linewidth=0.8,
               alpha=0.9,
               color='#fde725')  # Yellow from Viridis
    if 'month_cos' in data.columns:
        ax.plot(data.index, data['month_cos'],
               label='Month (cos)',
               linewidth=0.8,
               alpha=0.9,
               linestyle=':',  # Stippled line
               color='#fde725')  # Yellow from Viridis
    
    # Plot day of year features
    if 'day_of_year_sin' in data.columns:
        ax.plot(data.index, data['day_of_year_sin'],
               label='Day of Year (sin)',
               linewidth=0.8,
               alpha=0.9,
               color='#22a884')  # Green from Viridis
    if 'day_of_year_cos' in data.columns:
        ax.plot(data.index, data['day_of_year_cos'],
               label='Day of Year (cos)',
               linewidth=0.8,
               alpha=0.9,
               linestyle=':',  # Stippled line
               color='#22a884')  # Green from Viridis
    
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)  # Only show x-label on bottom subplot
    ax.set_ylabel('Value', fontweight='bold', labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', ncol=2, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Add horizontal line at 0 for reference in time features plot
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Format x-axis dates only for the bottom subplot
    if years_to_show <= 4:
        # For shorter time ranges, show more detailed x-ticks (monthly)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quarterly
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        # For longer time ranges, use quarterly ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Semi-annually
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05)  # Reduce vertical space between subplots

    # Save the figure
    output_path = output_dir / f'station_features_stacked_{timestamp}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"Saved stacked features plot to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

def plot_anomalies(test_data, test_predictions, anomalies, station_id, model_config=None, best_val_loss=None, create_html=True, open_browser=True, metrics=None, title_suffix=None, show_config=False, synthetic_data=None):
    """
    Create an interactive plot with aligned datetime indices, rainfall data, model configuration, and anomalies.
    
    Args:
        test_data: DataFrame containing test data with datetime index
        test_predictions: DataFrame or Series containing predictions
        anomalies: Boolean array or Series indicating anomaly locations
        station_id: ID of the station being analyzed
        model_config: Optional dictionary containing model configuration parameters
        best_val_loss: Optional best validation loss achieved during training
        create_html: Whether to create HTML plot (default: True)
        open_browser: Whether to open the plot in browser (default: True)
        metrics: Optional dictionary with additional performance metrics
        title_suffix: Optional suffix to add to the plot title
        show_config: Whether to show model configuration (default: False)
        synthetic_data: Optional DataFrame containing data with synthetic errors
    """
    # Ensure output directory exists using relative path
    output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    station_id = str(station_id)
    
    # Create the title with optional suffix
    title = f'Prediction Analysis with Anomalies for Station {station_id}'
    if title_suffix:
        title = f'{title} - {title_suffix}'
    
    # Get the actual test data with its datetime index (for reference only)
    test_actual = test_data['vst_raw']
    
    # Get rainfall data without resampling
    rainfall_data = None
    if 'rainfall' in test_data.columns:
        rainfall_data = test_data['rainfall']
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Extract predictions values - handle both DataFrame and Series inputs
    if isinstance(test_predictions, pd.DataFrame):
        predictions_values = test_predictions['vst_raw'].values
    else:
        predictions_values = test_predictions.values if isinstance(test_predictions, pd.Series) else test_predictions
    
    # Create a Series of NaN values with the same length as test_actual
    predictions_series = pd.Series(
        index=test_actual.index,
        data=np.nan
    )

    # Get sequence length from model config, default to 50 if not specified
    sequence_length = model_config.get('sequence_length', 50) if model_config else 50

    if model_config and model_config.get('model_type') == 'iterative':
        prediction_window = model_config.get('prediction_window', 10)
        
        # Place predictions at their correct future positions
        if len(predictions_values) > 0:
            # Calculate the start index for predictions (sequence_length steps from the start)
            start_idx = sequence_length
            # Place each prediction at its correct future position
            for i in range(0, len(predictions_values), prediction_window):
                if start_idx + i + prediction_window <= len(test_actual):
                    predictions_series.iloc[start_idx + i:start_idx + i + prediction_window] = predictions_values[i:i + prediction_window]
    else:
        # For standard model, still account for sequence_length offset
        if len(predictions_values) > 0:
            # Place predictions starting from sequence_length
            end_idx = min(sequence_length + len(predictions_values), len(test_actual))
            predictions_series.iloc[sequence_length:end_idx] = predictions_values[:end_idx-sequence_length]
    
    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create error periods directory
    error_periods_dir = output_dir / 'error_periods'
    error_periods_dir.mkdir(exist_ok=True)
    
    # Extract error periods and synthetic data DataFrame from synthetic_data dict if available
    error_periods = []
    synthetic_df = None
    if synthetic_data is not None:
        if isinstance(synthetic_data, dict):
            error_periods = synthetic_data.get('error_periods', [])
            synthetic_df = synthetic_data.get('data', None)
        elif hasattr(synthetic_data, 'error_periods'):
            error_periods = synthetic_data.error_periods
            synthetic_df = synthetic_data
        elif isinstance(synthetic_data, pd.DataFrame):
            synthetic_df = synthetic_data
    
    # Always create the PNG version with config text at the bottom
    png_path = create_water_level_plot_png(
        test_actual, 
        predictions_series, 
        station_id, 
        timestamp, 
        model_config, 
        output_dir,
        best_val_loss,
        metrics=metrics,
        show_config=show_config,
        title_suffix=title_suffix,
        synthetic_data=synthetic_data, 
        anomalies=anomalies,  # Pass the full structure with both data and error_periods
        vinge_data=None  # Pass vinge_data as None since it's not provided in the new function
    )
    
    # Create the HTML version only if requested
    if create_html:
        # Create subplots - rainfall on top, water level on bottom
        subplot_rows = 2 if rainfall_data is not None else 1
        
        if rainfall_data is not None:
            specs = [[{"secondary_y": False}] for _ in range(subplot_rows)]
            subplot_titles = [
                'Rainfall',
                f'Water Level - Station {station_id}'
            ]
            
            fig = make_subplots(
                rows=subplot_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=subplot_titles,
                specs=specs,
                row_heights=[0.3, 0.7]  # Give more space to rainfall
            )
            
            # Add rainfall data to first subplot (top)
            fig.add_trace(
                go.Bar(
                    x=rainfall_data.index,
                    y=rainfall_data.values,
                    name="Rainfall",
                    marker_color='rgba(0, 0, 255, 0.6)',  # More visible blue
                    opacity=0.8,
                    width=60*60*1000,  # 1-hour width in milliseconds
                    yaxis="y1"
                ),
                row=1, col=1
            )
            
            # Add water level data to second subplot (bottom)
            # First add original data as a light gray line for reference
            fig.add_trace(
                go.Scatter(
                    x=test_actual.index,
                    y=test_actual.values,
                    name="VST RAW (Reference)",
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=1)
                ),
                row=2, col=1
            )
            
            # Add synthetic error data if available
            if synthetic_df is not None:
                # Add colored background regions for different error types
                error_colors = {
                    'spike': 'rgba(255, 0, 0, 0.1)',      # Light red
                    'flatline': 'rgba(0, 255, 0, 0.1)',   # Light green
                    'drift': 'rgba(0, 0, 255, 0.1)',      # Light blue
                    'offset': 'rgba(255, 165, 0, 0.1)',   # Light orange
                    'baseline_shift': 'rgba(128, 0, 128, 0.1)',  # Light purple
                    'noise': 'rgba(128, 128, 128, 0.1)'   # Light gray
                }
                
                # Add error type backgrounds
                for error_type, color in error_colors.items():
                    # Find periods with this error type
                    type_periods = [p for p in error_periods if p.error_type == error_type]
                    
                    for period in type_periods:
                        fig.add_shape(
                            type="rect",
                            x0=period.start_time,
                            x1=period.end_time,
                            y0=test_actual.min(),
                            y1=test_actual.max(),
                            fillcolor=color,
                            opacity=0.3,
                            layer="below",
                            line_width=0,
                            row=2, col=1
                        )
                
                # Add synthetic data line
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_df.index,
                        y=synthetic_df['vst_raw'].values,
                        name="VST RAW (with Synthetic Errors)",
                        line=dict(color='#d62728', width=1, dash='dot')
                    ),
                    row=2, col=1
                )
            
            # Add predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predictions",
                    line=dict(color='#2ca02c', width=1)
                ),
                row=2, col=1
            )
            
            # Add anomalies as scatter points on the synthetic data
            if isinstance(anomalies, pd.Series):
                anomaly_indices = anomalies[anomalies].index
            else:
                anomaly_indices = test_actual.index[anomalies]
            
            if synthetic_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_indices,
                        y=synthetic_df.loc[anomaly_indices, 'vst_raw'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='x',
                            line=dict(width=2)
                        )
                    ),
                    row=2, col=1
                )
            
            # Update y-axes labels and ranges
            fig.update_yaxes(
                title_text="Rainfall (mm)",
                row=1, col=1,
                autorange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangemode='nonnegative'
            )
            
            fig.update_yaxes(
                title_text="Water Level (mm)",
                row=2, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
            
            # Update x-axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangeslider_visible=False,
                row=1, col=1
            )
            
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                rangeslider_visible=True,
                row=2, col=1
            )
            
        else:
            fig = go.Figure()
            
            # Add original data as a light gray line for reference
            fig.add_trace(
                go.Scatter(
                    x=test_actual.index,
                    y=test_actual.values,
                    name="VST RAW (Reference)",
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=1)
                )
            )
            
            # Add synthetic error data if available
            if synthetic_df is not None:
                # Add colored background regions for different error types
                error_colors = {
                    'spike': 'rgba(255, 0, 0, 0.1)',      # Light red
                    'flatline': 'rgba(0, 255, 0, 0.1)',   # Light green
                    'drift': 'rgba(0, 0, 255, 0.1)',      # Light blue
                    'offset': 'rgba(255, 165, 0, 0.1)',   # Light orange
                    'baseline_shift': 'rgba(128, 0, 128, 0.1)',  # Light purple
                    'noise': 'rgba(128, 128, 128, 0.1)'   # Light gray
                }
                
                # Add error type backgrounds
                for error_type, color in error_colors.items():
                    # Find periods with this error type
                    type_periods = [p for p in error_periods if p.error_type == error_type]
                    
                    for period in type_periods:
                        fig.add_shape(
                            type="rect",
                            x0=period.start_time,
                            x1=period.end_time,
                            y0=test_actual.min(),
                            y1=test_actual.max(),
                            fillcolor=color,
                            opacity=0.3,
                            layer="below",
                            line_width=0
                        )
                
                # Add synthetic data line
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_df.index,
                        y=synthetic_df['vst_raw'].values,
                        name="VST RAW (with Synthetic Errors)",
                        line=dict(color='#d62728', width=1, dash='dot')
                    )
                )
            
            # Add predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predictions",
                    line=dict(color='#2ca02c', width=1)
                )
            )
            
            # Add anomalies as scatter points on the synthetic data
            if isinstance(anomalies, pd.Series):
                anomaly_indices = anomalies[anomalies].index
            else:
                anomaly_indices = test_actual.index[anomalies]
            
            if synthetic_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_indices,
                        y=synthetic_df.loc[anomaly_indices, 'vst_raw'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='x',
                            line=dict(width=2)
                        )
                    )
                )
        
        # Update layout
        fig.update_layout(
            width=1500,
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=80, t=180, b=80),
            dragmode='zoom'
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

        # Add range selector buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="all")
                ]),
                bgcolor='rgba(150, 200, 250, 0.4)',
                activecolor='rgba(100, 150, 200, 0.8)'
            ),
            row=subplot_rows, col=1
        )
        
        # Save HTML plot
        html_path = output_dir / f'anomalies_station_{station_id}_{timestamp}.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True, config={
            'displayModeBar': True,
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'station_{station_id}_anomalies_{timestamp}',
                'height': 1000,
                'width': 1500,
                'scale': 2
            }
        })
        
        # Open HTML in browser if requested
        if open_browser:
            absolute_path = os.path.abspath(html_path)
            print(f"Opening plot in browser: {absolute_path}")
            webbrowser.open('file://' + str(absolute_path))
        
    return png_path



def plot_feature_importance(feature_names, importance_scores, station_id, output_dir=None, title_suffix=None):
    """
    Create a publication-quality plot of feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: List or array of importance scores corresponding to features
        station_id: ID of the station being analyzed
        output_dir: Optional output directory path
        title_suffix: Optional suffix to add to the plot title
    
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
        output_dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Create figure with extra space at the bottom for legend
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # Reduced figure size
    
    # Ensure importance_scores is a 1D numpy array
    importance_scores = np.asarray(importance_scores).flatten()
    feature_names = np.asarray(feature_names)
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)
    sorted_scores = importance_scores[sorted_indices]
    sorted_names = feature_names[sorted_indices]
    
    # Filter features based on importance threshold and limit to top 10
    importance_threshold = 0.1  # Increased threshold
    mask = sorted_scores >= importance_threshold
    sorted_scores = sorted_scores[mask]
    sorted_names = sorted_names[mask]
    
    # Take only top 10 features
    if len(sorted_scores) > 10:
        sorted_scores = sorted_scores[-10:]
        sorted_names = sorted_names[-10:]
    
    positions = np.arange(len(sorted_scores))
    
    # Create color palette using blues
    colors = {
        # Blues from light to dark
        'rainfall': '#08519c',     # Dark blue
        'water_level': '#6baed6',  # Medium blue
        'temperature': '#020406',  # Dark
        'time': '#f0f5fa',        # Very light blue
        'default': '#c6dbef'      # Light blue
    }
    
    # Create a legend mapping for feature types
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors['water_level'], label='Water Level'),
        plt.Rectangle((0,0),1,1, facecolor=colors['rainfall'], label='Precipitation'),
        plt.Rectangle((0,0),1,1, facecolor=colors['temperature'], label='Temperature'),
        plt.Rectangle((0,0),1,1, facecolor=colors['time'], label='Time')
    ]
    
    # Assign colors based on feature names
    feature_colors = []
    for feature in sorted_names:
        feature_lower = str(feature).lower()
        if 'rain' in feature_lower or 'precipitation' in feature_lower:
            feature_colors.append(colors['rainfall'])
        elif any(x in feature_lower for x in ['temperature']):
            feature_colors.append(colors['temperature'])
        elif any(x in feature_lower for x in ['water', 'level', 'vst']):
            feature_colors.append(colors['water_level'])
        elif any(x in feature_lower for x in ['sin', 'cos', 'month', 'day']):
            feature_colors.append(colors['time'])
        else:
            feature_colors.append(colors['default'])
    
    # Create horizontal bar plot
    bars = ax.barh(positions, sorted_scores, align='center', color=feature_colors)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Add legend at the bottom right
    ax.legend(handles=legend_elements, 
             loc='lower right',
             bbox_to_anchor=(1.0, 0.02),
             frameon=True,
             facecolor='white',
             edgecolor='#cccccc',
             ncol=2)
    
    plt.subplots_adjust(bottom=0.2, top=0.95, right=0.85)  # Adjusted margins
    
    ax.set_xlabel('Importance Score', fontweight='bold', labelpad=5)
    ax.set_ylabel('Features', fontweight='bold', labelpad=5)
    
    # Format feature names for better readability
    formatted_features = []
    for feature in sorted_names:
        feature_str = str(feature)
        
        # Handle time features
        if 'month_sin' in feature_str:
            formatted = 'Month sine'
        elif 'month_cos' in feature_str:
            formatted = 'Month cos'
        elif 'day_of_year_sin' in feature_str:
            formatted = 'DoY sine'
        elif 'day_of_year_cos' in feature_str:
            formatted = 'DoY cos'
        else:
            # Always try to extract station number for station-specific features
            station = ""
            if 'station_45' in feature_str or "21006845" in feature_str:
                station = "'45"
            elif 'station_46' in feature_str or "21006846" in feature_str:
                station = "'46"
            elif 'station_47' in feature_str or "21006847" in feature_str:
                station = "'47"
            
            # Handle different feature types
            if 'vst_raw' in feature_str:
                formatted = f"vst raw {station}"
            elif 'temperature' in feature_str:
                formatted = f"Temp {station}"
            elif 'rain' in feature_str or 'precipitation' in feature_str:
                # Handle derived rainfall features
                if any(window in feature_str for window in ['1hour', '7hour', '48hour', '90hour', '1month', '3months', '6months', '1year']):
                    time_window = ''
                    if '1hour' in feature_str:
                        time_window = '1h'
                    elif '7hour' in feature_str:
                        time_window = '7h'
                    elif '48hour' in feature_str:
                        time_window = '48h'
                    elif '90hour' in feature_str:
                        time_window = '90h'
                    elif '1month' in feature_str:
                        time_window = '1M'
                    elif '3months' in feature_str:
                        time_window = '3M'
                    elif '6months' in feature_str:
                        time_window = '6M'
                    elif '1year' in feature_str:
                        time_window = '1Y'
                    formatted = f"Prec {time_window} {station}"
                else:
                    formatted = f"Prec {station}"
            else:
                formatted = feature_str.replace('_', ' ').title()
        
        formatted_features.append(formatted)
    
    # Set y-tick labels with formatted feature names
    ax.set_yticks(positions)
    ax.set_yticklabels(formatted_features)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)  # Reduced font size
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    output_path = output_dir / f'feature_importance_station_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved feature importance plot to: {output_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return output_path

def create_individual_feature_plots(data, output_dir=None):
    """
    Create individual plots for different feature groups that can be stacked later.
    
    Args:
        data: DataFrame containing feature data
        output_dir: Optional output directory path
    
    Returns:
        Dictionary of paths to the saved PNG files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/feature_plots"))
        output_dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Define Viridis colors for precipitation plots
    short_window_colors = {
        '90hour': '#fde725',    # Yellow for longest window (90h)
        '48hour': '#22a884',    # Green for second longest (48h)
        '7hour': '#414487',     # Blue for 7h
        '1hour': '#440154'      # Purple for shortest window (1h)
    }
    
    long_window_colors = {
        '1year': '#fde725',     # Yellow for longest window (1y)
        '6months': '#22a884',   # Green for second longest (6m)
        '3months': '#414487',   # Blue for 3m
        '1month': '#440154'     # Purple for shortest window (1m)
    }
    
    # Define temperature colors (two shades of red)
    temp_colors = {
        '45': '#ff1a1a',  # Bright red
        '47': '#990000'   # Dark red
    }
    
    output_paths = {}
    
    # 1. Short-window precipitation features for each station
    stations = ['46', '45', '47']
    windows = ['1hour', '7hour', '48hour', '90hour']
    
    for station in stations:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
        
        for window in windows:
            column = f'station_{station}_rain_{window}'
            if column in data.columns:
                ax.plot(data.index, data[column], 
                       label=f'{window} cumulative',
                       linewidth=0.8,
                       alpha=0.9,
                       color=short_window_colors[window])
        
        ax.set_xlabel('Date', fontweight='bold', labelpad=10)
        ax.set_ylabel('Precipitation (mm)', fontweight='bold', labelpad=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        path = output_dir / f'short_precip_station_{station}_{timestamp}.png'
        plt.savefig(path, bbox_inches='tight', facecolor='white')
        output_paths[f'short_precip_{station}'] = path
        plt.close()
    
    # 2. Long-window precipitation features for each station
    windows = ['1month', '3months', '6months', '1year']
    
    for station in stations:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
        
        for window in windows:
            column = f'station_{station}_rain_{window}'
            if column in data.columns:
                ax.plot(data.index, data[column], 
                       label=f'{window} cumulative',
                       linewidth=0.8,
                       alpha=0.9,
                       drawstyle='steps-post',
                       color=long_window_colors[window])
        
        ax.set_xlabel('Date', fontweight='bold', labelpad=10)
        ax.set_ylabel('Precipitation (mm)', fontweight='bold', labelpad=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        path = output_dir / f'long_precip_station_{station}_{timestamp}.png'
        plt.savefig(path, bbox_inches='tight', facecolor='white')
        output_paths[f'long_precip_{station}'] = path
        plt.close()
    
    # 3. Temperature features for all stations
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    
    # Plot temperature for each station if available
    temp_columns = [col for col in data.columns if 'temperature' in col.lower()]
    for col in temp_columns:
        if "45" in col:
            station = "'45"
            color = temp_colors['45']
        else:
            station = "'47"
            color = temp_colors['47']
            
        ax.plot(data.index, data[col], 
               label=f'Temperature Station {station}',
               linewidth=0.8,
               alpha=0.9,
               color=color)
    
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax.set_ylabel('Temperature (°C)', fontweight='bold', labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = output_dir / f'temperature_{timestamp}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    output_paths['temperature'] = path
    plt.close()
    
    # 4. Time-based features (sine and cosine curves)
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    
    # Plot month features
    if 'month_sin' in data.columns:
        ax.plot(data.index, data['month_sin'], 
               label='Month (sin)',
               linewidth=0.8,
               alpha=0.9,
               color='#fde725')  # Yellow from Viridis
    if 'month_cos' in data.columns:
        ax.plot(data.index, data['month_cos'], 
               label='Month (cos)',
               linewidth=0.8,
               alpha=0.9,
               linestyle=':',  # Stippled line
               color='#fde725')  # Yellow from Viridis
    
    # Plot day of year features
    if 'day_of_year_sin' in data.columns:
        ax.plot(data.index, data['day_of_year_sin'], 
               label='Day of Year (sin)',
               linewidth=0.8,
               alpha=0.9,
               color='#22a884')  # Green from Viridis
    if 'day_of_year_cos' in data.columns:
        ax.plot(data.index, data['day_of_year_cos'], 
               label='Day of Year (cos)',
               linewidth=0.8,
               alpha=0.9,
               linestyle=':',  # Stippled line
               color='#22a884')  # Green from Viridis
    
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax.set_ylabel('Value', fontweight='bold', labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', ncol=2, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    path = output_dir / f'time_features_{timestamp}.png'
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    output_paths['time_features'] = path
    plt.close()
    
    return output_paths

def plot_feature_correlation(data, output_dir=None):
    """
    Create a correlation plot for specified features.
    
    Args:
        data: DataFrame containing the features
        output_dir: Optional output directory path
    
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
        output_dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 16,  # Increased from 12
        'axes.titlesize': 24,  # Increased from 16
        'axes.labelsize': 20,  # Increased from 14
        'xtick.labelsize': 16,  # Increased from 12
        'ytick.labelsize': 16,  # Increased from 12
        'legend.fontsize': 16,  # Increased from 12
        'figure.titlesize': 24  # Increased from 18
    })
    
    # Select features for correlation analysis
    features = [
        'vst_raw',
        'feature_station_21006845_vst_raw',
        'feature_station_21006847_vst_raw',
        'station_47_rain_1hour',
        'station_47_rain_7hour',
        'station_47_rain_48hour',
        'station_47_rain_90hour',
        'station_47_rain_1month',
        'station_47_rain_3months',
        'station_47_rain_6months',
        'station_47_rain_1year'
    ]
    
    # Create correlation matrix
    corr_matrix = data[features].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)  # Increased figure size
    
    # Create custom diverging colormap from red (1) through white (0) to blue (-1)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#00008B', '#ffffff', '#8B0000'])
    
    # Plot correlation matrix
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va='bottom', fontsize=20)  # Increased font size
    
    # Format feature names for better readability
    feature_names = [
        'VST Raw',
        'VST Raw 45',
        'VST Raw 47',
        'Precip 1h',
        'Precip 7h',
        'Precip 48h',
        'Precip 90h',
        'Precip 1M',
        'Precip 3M',
        'Precip 6M',
        'Precip 1Y'
    ]
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=16)  # Increased font size
    ax.set_yticklabels(feature_names, fontsize=16)  # Increased font size
    
    # Add correlation values in each cell
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=16)  # Increased font size
    
    # Add title
    plt.title('Feature Correlation Matrix', pad=20, fontsize=24)  # Increased font size
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'feature_correlation_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved correlation plot to: {output_path}")
    
    # Close figure
    plt.close()
    
    return output_path