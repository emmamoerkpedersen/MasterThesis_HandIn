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

def create_full_plot(test_data, test_predictions, station_id, model_config=None, best_val_loss=None, create_html=True, open_browser=True, metrics=None, title_suffix=None, show_config=False, synthetic_data=None):
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
    """
    # Ensure output directory exists using relative path
    output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    station_id = str(station_id)
    
    # Create the title with optional suffix
    title = f'Prediction Analysis for Station {station_id}'
    if title_suffix:
        title = f'{title} - {title_suffix}'
    
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']
    
    # Get rainfall data without resampling
    rainfall_data = None
    if 'rainfall' in test_data.columns:
        rainfall_data = test_data['rainfall']
    
    # Get vinge data if available in test_data
    vinge_data = None
    if 'vinge' in test_data.columns:
        vinge_data = test_data[['vinge']].copy()
        print(f"Found vinge data with {len(vinge_data[~vinge_data['vinge'].isna()])} non-null values")
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Extract predictions values - handle both DataFrame and Series inputs
    if isinstance(test_predictions, pd.DataFrame):
        predictions_values = test_predictions['vst_raw'].values
    else:
        predictions_values = test_predictions.values if isinstance(test_predictions, pd.Series) else test_predictions
    
    # For iterative forecasting, shift predictions to align with their actual prediction times
    if model_config and model_config.get('model_type') == 'iterative':
        sequence_length = model_config.get('sequence_length', 50)
        prediction_window = model_config.get('prediction_window', 10)
        
        # Create a Series of NaN values with the same length as test_actual
        predictions_series = pd.Series(
            index=test_actual.index,
            data=np.nan
        )
        
        # Place predictions at their correct future positions
        if len(predictions_values) > 0:
            # Calculate the start index for predictions (sequence_length steps from the start)
            start_idx = sequence_length
            # Place each prediction at its correct future position
            for i in range(0, len(predictions_values), prediction_window):
                if start_idx + i + prediction_window <= len(test_actual):
                    predictions_series.iloc[start_idx + i:start_idx + i + prediction_window] = predictions_values[i:i + prediction_window]
    else:
        # For standard model, just align with actual data
        predictions_series = pd.Series(
            data=predictions_values,
            index=test_actual.index[:len(predictions_values)],
            name='Predictions'
        )
    
    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
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
        vinge_data=vinge_data,
        show_config=show_config,
        title_suffix=title_suffix
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
            
            # Add synthetic error data if available
            if synthetic_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_data.index,
                        y=synthetic_data['vst_raw'].values,
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
            
            # Add vinge (manual board) data to the water level subplot if available
            if vinge_data is not None:
                try:
                    # Get the vinge column (could be 'vinge', 'water_level_mm', or 'W.L [cm]')
                    vinge_column = None
                    
                    # Try to identify the correct column to use
                    possible_columns = ['vinge', 'water_level_mm', 'W.L [cm]']
                    for col in possible_columns:
                        if col in vinge_data.columns:
                            vinge_column = col
                            break
                    
                    # If we couldn't find a recognized column but have only one column, use that
                    if vinge_column is None and len(vinge_data.columns) == 1:
                        vinge_column = vinge_data.columns[0]
                        print(f"Using unrecognized column '{vinge_column}' for vinge data in HTML plot")
                    
                    if vinge_column is not None:
                        # Filter out NaN values
                        vinge_no_nan = vinge_data.dropna(subset=[vinge_column])
                        
                        if not vinge_no_nan.empty:
                            # Convert to numeric if needed
                            if not pd.api.types.is_numeric_dtype(vinge_no_nan[vinge_column]):
                                vinge_no_nan[vinge_column] = pd.to_numeric(vinge_no_nan[vinge_column], errors='coerce')
                                vinge_no_nan = vinge_no_nan.dropna(subset=[vinge_column])
                            
                            # Add vinge data to the water level subplot (row 2)
                            fig.add_trace(
                                go.Scatter(
                                    x=vinge_no_nan.index,
                                    y=vinge_no_nan[vinge_column].values,
                                    name="Vinge",
                                    mode='markers',
                                    marker=dict(
                                        color='#2ca02c',  # Green color
                                        size=8,
                                        symbol='circle'
                                    )
                                ),
                                row=2, col=1  # Add to water level subplot (row 2)
                            )
                            print(f"Added {len(vinge_no_nan)} manual board measurements to the rainfall+waterlevel subplot plot")
                        else:
                            print("No valid vinge data points for HTML plot")
                    else:
                        print(f"Could not find a usable column in vinge data for HTML plot. Available columns: {vinge_data.columns}")
                except Exception as e:
                    print(f"Error adding vinge data to single-plot HTML: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
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
            
            # Add synthetic error data if available
            if synthetic_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=synthetic_data.index,
                        y=synthetic_data['vst_raw'].values,
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
            
            # Add vinge (manual board) data to the plot if available
            if vinge_data is not None:
                try:
                    # Get the vinge column (could be 'vinge', 'water_level_mm', or 'W.L [cm]')
                    vinge_column = None
                    
                    # Try to identify the correct column to use
                    possible_columns = ['vinge', 'water_level_mm', 'W.L [cm]']
                    for col in possible_columns:
                        if col in vinge_data.columns:
                            vinge_column = col
                            break
                    
                    # If we couldn't find a recognized column but have only one column, use that
                    if vinge_column is None and len(vinge_data.columns) == 1:
                        vinge_column = vinge_data.columns[0]
                        print(f"Using unrecognized column '{vinge_column}' for vinge data in HTML plot")
                    
                    if vinge_column is not None:
                        # Filter out NaN values
                        vinge_no_nan = vinge_data.dropna(subset=[vinge_column])
                        
                        if not vinge_no_nan.empty:
                            # Convert to numeric if needed
                            if not pd.api.types.is_numeric_dtype(vinge_no_nan[vinge_column]):
                                vinge_no_nan[vinge_column] = pd.to_numeric(vinge_no_nan[vinge_column], errors='coerce')
                                vinge_no_nan = vinge_no_nan.dropna(subset=[vinge_column])
                            
                            # In the simple figure case (no rainfall)
                            fig.add_trace(
                                go.Scatter(
                                    x=vinge_no_nan.index,
                                    y=vinge_no_nan[vinge_column].values,
                                    name="Vinge",
                                    mode='markers',
                                    marker=dict(
                                        color='#2ca02c',  # Green color
                                        size=8,
                                        symbol='circle'
                                    )
                                )
                            )
                            print(f"Added {len(vinge_no_nan)} manual board measurements to the simple HTML plot")
                        else:
                            print("No valid vinge data points for HTML plot")
                    else:
                        print(f"Could not find a usable column in vinge data for HTML plot. Available columns: {vinge_data.columns}")
                except Exception as e:
                    print(f"Error adding vinge data to single-plot HTML: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,  # Moved down slightly to make room for config
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
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
            margin=dict(l=80, r=80, t=180, b=80)  # Increased top margin for config box
        )
        
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
        
        # Open HTML in browser if requested
        if open_browser:
            absolute_path = os.path.abspath(html_path)
            print(f"Opening plot in browser: {absolute_path}")
            webbrowser.open('file://' + str(absolute_path))
        
    return png_path

def create_water_level_plot_png(actual, predictions, station_id, timestamp, model_config=None, output_dir=None, best_val_loss=None, metrics=None, vinge_data=None, show_config=False, title_suffix=None):
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
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # If vinge data is not provided or empty, try to load it from the preprocessed data
    if vinge_data is None or (isinstance(vinge_data, pd.DataFrame) and vinge_data.empty):
        try:
            print(f"Vinge data not provided, attempting to load from preprocessed data...")
            
            # Define possible directories to look for the data using relative paths
            possible_dirs = [
                Path(os.path.join(PROJECT_ROOT, "data_utils/Sample data")),
                Path(os.path.join(PROJECT_ROOT, "../data_utils/Sample data")),
                Path(os.path.join(PROJECT_ROOT, "data_utils")),
                Path(os.path.join(PROJECT_ROOT, "../data_utils"))
            ]
            
            preprocessed_data = None
            # Try to load from possible directories
            for data_dir in possible_dirs:
                try:
                    if (data_dir / "preprocessed_data.pkl").exists():
                        print(f"Found preprocessed_data.pkl in {data_dir}")
                        preprocessed_data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
                        break
                    elif (data_dir / "original_data.pkl").exists():
                        print(f"Found original_data.pkl in {data_dir}")
                        preprocessed_data = pd.read_pickle(data_dir / "original_data.pkl")
                        break
                except Exception as e:
                    print(f"Could not load from {data_dir}: {str(e)}")
            
            if preprocessed_data is None:
                print("Could not find preprocessed data in any of the expected locations.")
            else:
                # If we found the data, try to extract vinge data
                if station_id in preprocessed_data:
                    station_data = preprocessed_data[station_id]
                    
                    # Check if vinge data exists for this station
                    if 'vinge' in station_data:
                        # Handle different structures - could be Series or DataFrame
                        vinge_raw = station_data['vinge']
                        if isinstance(vinge_raw, pd.DataFrame):
                            vinge_data = vinge_raw
                            print(f"Loaded vinge data as DataFrame with columns: {vinge_data.columns}")
                        else:
                            # Convert to DataFrame if it's a Series
                            vinge_data = pd.DataFrame({'vinge': vinge_raw})
                            print(f"Converted vinge data Series to DataFrame")
                        
                        print(f"Loaded vinge data for station {station_id}: {len(vinge_data)} records")
                        
                        # Check if we have any non-null vinge values
                        if 'vinge' in vinge_data.columns:
                            non_null_count = vinge_data['vinge'].count()
                            print(f"Found {non_null_count} non-null vinge measurements")
                    else:
                        print(f"No vinge data found for station {station_id}")
                else:
                    print(f"Station {station_id} not found in preprocessed data")
        except Exception as e:
            print(f"Error loading vinge data: {str(e)}")
            import traceback
            traceback.print_exc()
            vinge_data = None
    
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
    fig = plt.figure(figsize=(12, 8.5), dpi=300)
    
    # Create a larger figure to accommodate the config text and metrics
    # Add a third row for metrics if available
    if metrics and any(not np.isnan(v) for v in metrics.values()):
        gs = plt.GridSpec(3, 1, height_ratios=[6, 0.8, 0.8], hspace=0.15)  # Reduced spacing
    else:
        gs = plt.GridSpec(2, 1, height_ratios=[6, 0.8], hspace=0.15)  # Reduced spacing
    
    # Main plot in top section
    ax = fig.add_subplot(gs[0])
    
    # Plot with consistent colors - blue for actual, red for predicted
    ax.plot(actual.index, actual.values, color='#1f77b4', linewidth=0.8, label='VST RAW')
    ax.plot(predictions.index, predictions.values, color='#d62728', linewidth=0.8, label='Predicted')
    
    # Add VINGE measurements if provided (similar to preprocessing_diagnostics.py)
    if vinge_data is not None and (isinstance(vinge_data, pd.DataFrame) and not vinge_data.empty):
        try:
            # Check if vinge data is properly formatted
            # Get the vinge column (could be 'vinge', 'water_level_mm', or 'W.L [cm]')
            vinge_column = None
            
            # Try to identify the correct column to use
            possible_columns = ['vinge', 'water_level_mm', 'W.L [cm]']
            for col in possible_columns:
                if col in vinge_data.columns:
                    vinge_column = col
                    print(f"Using '{col}' column for vinge data")
                    break
            
            # If we couldn't find a recognized column but have only one column, use that
            if vinge_column is None and len(vinge_data.columns) == 1:
                vinge_column = vinge_data.columns[0]
                print(f"Using unrecognized column '{vinge_column}' for vinge data")
                
            if vinge_column is not None:
                # Ensure data is properly indexed with datetime
                if not isinstance(vinge_data.index, pd.DatetimeIndex):
                    if 'Date' in vinge_data.columns:
                        try:
                            vinge_data.set_index('Date', inplace=True)
                            print("Set vinge data index to 'Date' column")
                        except Exception as e:
                            print(f"Error setting index to Date column: {str(e)}")
                
                # Only proceed if we have a proper DatetimeIndex now
                if isinstance(vinge_data.index, pd.DatetimeIndex):
                    # Filter to match the time range of actual data
                    try:
                        vinge_filtered = vinge_data[
                            (vinge_data.index >= actual.index.min()) & 
                            (vinge_data.index <= actual.index.max())
                        ]
                        print(f"Filtered vinge data to time range: {len(vinge_filtered)} records remain")
                        
                        # Drop NaN values
                        vinge_filtered = vinge_filtered.dropna(subset=[vinge_column])
                        print(f"After dropping NaNs: {len(vinge_filtered)} records remain")
                        
                        if not vinge_filtered.empty:
                            try:
                                # Convert to numeric if needed
                                if not pd.api.types.is_numeric_dtype(vinge_filtered[vinge_column]):
                                    vinge_filtered[vinge_column] = pd.to_numeric(vinge_filtered[vinge_column], errors='coerce')
                                    vinge_filtered = vinge_filtered.dropna(subset=[vinge_column])
                                    print(f"Converted vinge data to numeric, {len(vinge_filtered)} records remain")
                                
                                # Plot VINGE measurements with larger markers and higher zorder
                                ax.scatter(
                                    vinge_filtered.index, 
                                    vinge_filtered[vinge_column],
                                    color='#2ca02c',  # Green color
                                    alpha=0.8, 
                                    s=50,  # Larger point size
                                    label='Vinge',
                                    zorder=5,  # Ensure points are drawn on top
                                    marker='o'
                                )
                                print(f"Added {len(vinge_filtered)} manual board measurements to the plot")
                            except Exception as e:
                                print(f"Error plotting vinge data points: {str(e)}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("No vinge data points within the plot time range")
                    except Exception as e:
                        print(f"Error filtering vinge data: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Vinge data does not have a DatetimeIndex. Current index type: {type(vinge_data.index)}")
            else:
                print(f"Could not find a usable column in vinge data. Available columns: {vinge_data.columns}")
        except Exception as e:
            print(f"Error processing vinge data for plotting: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("No vinge data available for plotting")
    
    # Clean styling
    if title_suffix:
        title_text = f'Water Level Predictions - Station {station_id} - {title_suffix}'
    else:
        title_text = f'Water Level Predictions - Station {station_id}'
        
    if isinstance(timestamp, str):
        ax.set_title(title_text, fontweight='bold', pad=15)
    else:
        if title_suffix:
            ax.set_title(f'{title_text}\n{timestamp.strftime("%Y-%m-%d")}', fontweight='bold', pad=15)
        else:
            ax.set_title(f'Water Level Predictions - Station {station_id}\n{timestamp.strftime("%Y-%m-%d")}', fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
    
    # No grid lines as requested
    ax.grid(False)
    
    # Add clean legend
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    
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


def plot_scaled_predictions(predictions, targets, test_data=None, title="Scaled Predictions vs Targets"):
    
        """
        Plot scaled predictions and targets before inverse transformation.
        
        Args:
            predictions: Scaled predictions array
            targets: Scaled targets array
            test_data: Original DataFrame with datetime index (optional)
            title: Plot title
        """
        # Create figure
        fig = go.Figure()
        
        # Flatten predictions and targets for plotting
        flat_predictions = predictions.reshape(-1)
        flat_targets = targets.reshape(-1)
        
        # Create x-axis points - either use dates from test_data or timesteps
        if test_data is not None and hasattr(test_data, 'index'):
            x_points = test_data.index[:len(flat_predictions)]
            x_label = 'Date'
        else:
            x_points = np.arange(len(flat_predictions))
            x_label = 'Timestep'
        
        # Add targets
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=flat_targets,
                name="Scaled Targets",
                line=dict(color='blue', width=1)
            )
        )

        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=flat_predictions,
                name="Scaled Predictions",
                line=dict(color='red', width=1)
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title='Scaled Value',
            width=1200,
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add rangeslider if using dates
        if test_data is not None and hasattr(test_data, 'index'):
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"  # This will format the x-axis as dates
                )
            )
        
        # Save and open in browser using relative path
        output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / 'scaled_predictions.html'
        fig.write_html(str(html_path))
        print(f"Opening scaled predictions plot in browser...")
        webbrowser.open('file://' + os.path.abspath(html_path))

def plot_convergence(history, station_id, title=None):
    """
    Plot training and validation loss over epochs, with learning rate changes.
    
    Args:
        history: Dictionary containing training history
        station_id: ID of the station
        title: Optional plot title
    """
    # Ensure output directory exists using relative path
    output_dir = Path(os.path.join(PROJECT_ROOT, "results/lstm"))
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
            print(f"Limiting plot to the most recent {years_to_show} years: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            data = filtered_data
        else:
            print(f"Warning: No data in the last {years_to_show} years. Using all available data.")
    
    # Organize features by station and type
    station_features = {
        'Station 21006846': [],
        'Station 21006845': [],
        'Station 21006847': []
    }
    
    # Special groups for separate plotting
    temperature_features = []
    water_level_features = []
    time_features = []
    
    # Group features by station and type
    for feature in feature_cols:
        # Check for special feature types
        if any(x in feature for x in ['sin', 'cos']):
            time_features.append(feature)
            continue
        elif 'temperature' in feature.lower() or 'temp' in feature.lower():
            temperature_features.append(feature)
            continue
        elif 'vst_raw' in feature.lower() or 'water_level' in feature.lower():
            water_level_features.append(feature)
            continue
            
        # Identify station-specific rainfall features
        if 'feature_station_21006845' in feature or 'feature1' in feature:
            station_features['Station 21006845'].append(feature)
        elif 'feature_station_21006847' in feature or 'feature2' in feature:
            station_features['Station 21006847'].append(feature)
        else:
            station_features['Station 21006846'].append(feature)
    
    # Add special feature groups if they have features
    special_groups = {}
    if time_features:
        special_groups['Time Features'] = time_features
    if temperature_features:
        special_groups['Temperature'] = temperature_features
    if water_level_features:
        special_groups['Water Level'] = water_level_features
    
    # Remove empty stations
    station_features = {k: v for k, v in station_features.items() if v}
    
    # Combine all groups for plotting
    all_plot_groups = {**station_features, **special_groups}
    
    # If no features, show message and return
    if not all_plot_groups:
        print("No features found for plotting.")
        return None
    
    # Set high-quality styling for matplotlib
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Number of subplots needed
    n_plots = len(all_plot_groups)
    
    # Adjust height based on number of plots
    fig_height = min(4 * n_plots, 16)  # Cap height at 16 inches
    
    # Create figure with subplots (one per group)
    fig, axes = plt.subplots(
        n_plots, 
        1, 
        figsize=(12, fig_height), 
        dpi=300,
        gridspec_kw={'height_ratios': [1] * n_plots}
    )
    
    # Handle the case where there's only one group
    if n_plots == 1:
        axes = [axes]
    
    # Define a color palette for different feature types
    colors = {
        # Viridis-inspired colors for rainfall features
        '30day': '#440154',    # Deep purple for 30-day features
        '180day': '#21918c',   # Teal for 180-day features
        '365day': '#fde725',   # Yellow for 365-day features
        'rainfall': '#5ec962', # Green for direct rainfall
        
        # Keep other colors the same
        'temperature': '#d62728',  # Red for temperature
        'water_level': '#1f77b4',  # Blue for water level
        'month_sin': '#ffb703', # Yellow for month sin
        'month_cos': '#fd9e02', # Gold for month cos
        'day_sin': '#06d6a0',   # Green for day sin
        'day_cos': '#118ab2',   # Blue-green for day cos
        'default': '#073b4c'    # Dark teal for other features
    }
    
    # Plot each group's features
    for i, (group_name, features) in enumerate(all_plot_groups.items()):
        ax = axes[i]
        
        # Handle different types of feature groups
        if group_name == 'Time Features':
            # Group time features by type
            sin_features = sorted([f for f in features if 'sin' in f])
            cos_features = sorted([f for f in features if 'cos' in f])
            
            # Define line styles
            line_styles = {
                'sin': '-',
                'cos': '--'
            }
            
            # Plot each sin/cos pair
            for sin_f, cos_f in zip(sin_features, cos_features):
                feature_type = 'month' if 'month' in sin_f else 'day'
                
                # Format labels nicely
                sin_label = f"{'Month' if 'month' in sin_f else 'Day of Year'} (sin)"
                cos_label = f"{'Month' if 'month' in cos_f else 'Day of Year'} (cos)"
                
                # Plot sin curve
                ax.plot(
                    data.index, 
                    data[sin_f], 
                    label=sin_label,
                    color=colors[f'{feature_type}_sin'],
                    linestyle=line_styles['sin'],
                    linewidth=0.8,
                    alpha=0.9
                )
                
                # Plot corresponding cos curve
                ax.plot(
                    data.index, 
                    data[cos_f], 
                    label=cos_label,
                    color=colors[f'{feature_type}_cos'],
                    linestyle=line_styles['cos'],
                    linewidth=0.8,
                    alpha=0.8
                )
            
            # Add horizontal line at 0 for reference in time features
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
        elif group_name == 'Temperature':
            # Plot each temperature feature
            for feature in features:
                ax.plot(
                    data.index, 
                    data[feature], 
                    label='Temperature',
                    color=colors['temperature'],
                    linewidth=0.8,
                    alpha=0.9
                )
                
        elif group_name == 'Water Level':
            # Plot each water level feature with the main station in black
            for j, feature in enumerate(features):
                station_label = 'Main Station'
                if 'feature1' in feature or 'feature_station_21006845' in feature:
                    station_label = 'Station 21006845'
                    line_style = '--'
                    line_color = '#1f77b4'  # Blue 
                elif 'feature2' in feature or 'feature_station_21006847' in feature:
                    station_label = 'Station 21006847'
                    line_style = ':'
                    line_color = 'black'  # Red
                else:
                    station_label = 'Main Station'
                    line_style = '-'
                    line_color = 'black'  # Main station in black
                
                ax.plot(
                    data.index, 
                    data[feature], 
                    label=f'Water Level ({station_label})',
                    color=line_color,
                    linewidth=0.8 if station_label != 'Main Station' else 1.2,  # Make main station slightly thicker
                    alpha=0.9,
                    linestyle=line_style
                )
                
        else:  # Regular station features (rainfall and cumulative rainfall)
            # Group features by type for better organization
            feature_groups = {}
            
            for feature in features:
                # Determine feature type
                if '30day' in feature:
                    feature_type = '30day'
                elif '180day' in feature:
                    feature_type = '180day'
                elif '365day' in feature:
                    feature_type = '365day'
                elif 'rainfall' in feature.lower():
                    feature_type = 'rainfall'
                else:
                    feature_type = 'default'
                
                
                if feature_type not in feature_groups:
                    feature_groups[feature_type] = []
                feature_groups[feature_type].append(feature)
            
            # Plot each feature group - but only add legend for the first station to avoid duplicates
            for feature_type, feats in feature_groups.items():
                # Create friendly label based on feature type
                if feature_type == '30day':
                    label = "30-Day Cumulative Rainfall"
                elif feature_type == '180day':
                    label = "180-Day Cumulative Rainfall"
                elif feature_type == '365day':
                    label = "365-Day Cumulative Rainfall"
                elif feature_type == 'rainfall':
                    label = "Rainfall"
                else:
                    # For unknown feature types, use the raw name
                    label = feats[0].replace('_', ' ').replace('feature station', '').title()
                
                # Determine if this is the main station (21006846) to decide whether to add legend
                is_main_station = group_name == 'Station 21006846'
                
                for feat in feats:
                    ax.plot(
                        data.index, 
                        data[feat], 
                        label=label if is_main_station else "_nolegend_",  # Only add to legend for main station
                        color=colors[feature_type],
                        linewidth=0.8,
                        alpha=0.9
                    )
        
        # Set title and format axes
        ax.set_title(group_name, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        # Format x-axis dates
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
        
        # Only show x-axis label on bottom subplot
        if i == n_plots - 1:
            ax.set_xlabel('Date', fontweight='bold', labelpad=10)
        else:
            # Hide x-tick labels for all but the bottom subplot
            plt.setp(ax.get_xticklabels(), visible=False)
        
        # Set y-label based on group/feature types
        if group_name == 'Time Features':
            ax.set_ylabel('Value', fontweight='bold', labelpad=10)
        elif group_name == 'Temperature':
            ax.set_ylabel('Temperature (°C)', fontweight='bold', labelpad=10)
        elif group_name == 'Water Level':
            ax.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
        elif any('rainfall' in f.lower() for f in features):
            ax.set_ylabel('Rainfall (mm)', fontweight='bold', labelpad=10)
        else:
            ax.set_ylabel('Value', fontweight='bold', labelpad=10)
        
        # Add proper legends with distinct entries - only for certain plot types
        handles, labels = ax.get_legend_handles_labels()
        
        # Only add legend if there are actual legend entries
        if handles:
            by_label = dict(zip(labels, handles))
            # Do not show the "30-Day Cumulative Rainfall" type labels for non-main stations
            if "Station 21006845" in group_name or "Station 21006847" in group_name:
                # Skip legend for non-main rainfall stations
                pass
            else:
                ax.legend(
                    by_label.values(), 
                    by_label.keys(),
                    loc='upper right', 
                    frameon=True, 
                    framealpha=0.9, 
                    edgecolor='#cccccc',
                    ncol=2 if group_name == 'Time Features' else 1
                )
    
    # Adjust spacing between subplots
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    
    # Removed the main title as requested
    
    # Save the figure with high resolution
    output_path = output_dir / f'station_features_{timestamp}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"Saved station features plot to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path
