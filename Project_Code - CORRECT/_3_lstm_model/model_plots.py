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


def create_full_plot(test_data, test_predictions, station_id, model_config=None, best_val_loss=None, create_html=True, open_browser=True, metrics=None):
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
    """
    # Ensure output directory exists
    output_dir = Path("Project_Code - CORRECT/results/lstm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    station_id = str(station_id)
    
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
    
    # If no vinge data in test_data, try to load from preprocessed data
    if vinge_data is None or (isinstance(vinge_data, pd.DataFrame) and vinge_data['vinge'].count() == 0):
        try:
            print("Vinge data not found in test_data or has no valid entries, loading from original data...")
            data_dir = Path("Project_Code - CORRECT/data_utils/Sample data")
            preprocessed_data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
            
            if station_id in preprocessed_data:
                station_data = preprocessed_data[station_id]
                
                # Check if vinge data exists for this station
                if 'vinge' in station_data:
                    # Handle different structures - could be Series or DataFrame
                    vinge_raw = station_data['vinge']
                    if isinstance(vinge_raw, pd.DataFrame):
                        vinge_data = vinge_raw
                        print(f"Vinge data is already a DataFrame with columns: {vinge_data.columns}")
                    else:
                        # Convert to DataFrame if it's a Series
                        vinge_data = pd.DataFrame({'vinge': vinge_raw})
                    
                    print(f"Loaded vinge data for station {station_id}: {len(vinge_data)} records")
                    
                    # Filter vinge data to match test data time range if test_actual has dates
                    if len(test_actual) > 0:
                        start_date = test_actual.index.min()
                        end_date = test_actual.index.max()
                        vinge_data = vinge_data[
                            (vinge_data.index >= start_date) & 
                            (vinge_data.index <= end_date)
                        ]
                        print(f"Filtered vinge data to match test period: {len(vinge_data)} records")
                    
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
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Extract predictions values - handle both DataFrame and Series inputs
    if isinstance(test_predictions, pd.DataFrame):
        predictions_values = test_predictions['vst_raw'].values
    else:
        predictions_values = test_predictions.values if isinstance(test_predictions, pd.Series) else test_predictions
    
    # Trim the predictions to match actual data length if needed
    if len(predictions_values) > len(test_actual):
        print("Trimming predictions to match actual data length")
        predictions_values = predictions_values[:len(test_actual)]
    else:
        print("Using full predictions")
    
    # Create a pandas Series for predictions with the matching datetime index
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
        vinge_data=vinge_data
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
                    name="Actual",
                    line=dict(color='#1f77b4', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predicted",
                    line=dict(color='#d62728', width=1)
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
                                    name="Manual Board",
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
                    name="Actual",
                    line=dict(color='#1f77b4', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions_series.index,
                    y=predictions_series.values,
                    name="Predicted",
                    line=dict(color='#d62728', width=1)
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
                                    name="Manual Board",
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
                'text': f'Prediction Analysis for Station {station_id}',
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
                y=1.05,  # Just above the plot title
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

def create_water_level_plot_png(actual, predictions, station_id, timestamp, model_config=None, output_dir=None, best_val_loss=None, metrics=None, vinge_data=None):
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
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/lstm")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # If vinge data is not provided, try to load it from the preprocessed data
    if vinge_data is None or vinge_data.empty:
        try:
            print(f"Vinge data not provided, attempting to load from preprocessed data...")
            data_dir = Path("Project_Code - CORRECT/data_utils/Sample data")
            preprocessed_data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
            
            if station_id in preprocessed_data:
                station_data = preprocessed_data[station_id]
                
                # Check if vinge data exists for this station
                if 'vinge' in station_data:
                    # Handle different structures - could be Series or DataFrame
                    vinge_raw = station_data['vinge']
                    if isinstance(vinge_raw, pd.DataFrame):
                        vinge_data = vinge_raw
                        print(f"Vinge data is already a DataFrame with columns: {vinge_data.columns}")
                    else:
                        # Convert to DataFrame if it's a Series
                        vinge_data = pd.DataFrame({'vinge': vinge_raw})
                    
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
    ax.plot(actual.index, actual.values, color='#1f77b4', linewidth=0.8, label='Actual')
    ax.plot(predictions.index, predictions.values, color='#d62728', linewidth=0.8, label='Predicted')
    
    # Add VINGE measurements if provided (similar to preprocessing_diagnostics.py)
    if vinge_data is not None and not vinge_data.empty:
        try:
            # Check if vinge data is properly formatted
            if isinstance(vinge_data, pd.DataFrame):
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
                    print(f"Using unrecognized column '{vinge_column}' for vinge data")
                    
                if vinge_column is not None:
                    # Ensure data is properly indexed with datetime
                    if not isinstance(vinge_data.index, pd.DatetimeIndex):
                        if 'Date' in vinge_data.columns:
                            vinge_data.set_index('Date', inplace=True)
                    
                    # Filter to match the time range of actual data
                    vinge_filtered = vinge_data[
                        (vinge_data.index >= actual.index.min()) & 
                        (vinge_data.index <= actual.index.max())
                    ]
                    
                    # Drop NaN values
                    vinge_filtered = vinge_filtered.dropna(subset=[vinge_column])
                    
                    if not vinge_filtered.empty:
                        try:
                            # Convert to numeric if needed
                            if not pd.api.types.is_numeric_dtype(vinge_filtered[vinge_column]):
                                vinge_filtered[vinge_column] = pd.to_numeric(vinge_filtered[vinge_column], errors='coerce')
                                vinge_filtered = vinge_filtered.dropna(subset=[vinge_column])
                            
                            # Plot VINGE measurements with larger markers and higher zorder
                            ax.scatter(
                                vinge_filtered.index, 
                                vinge_filtered[vinge_column],
                                color='#2ca02c',  # Green color
                                alpha=0.8, 
                                s=50,  # Larger point size
                                label='Manual Board',
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
                else:
                    print(f"Could not find a usable column in vinge data. Available columns: {vinge_data.columns}")
            else:
                print(f"Vinge data is not a DataFrame, it's a {type(vinge_data)}")
        except Exception as e:
            print(f"Error processing vinge data for plotting: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Clean styling
    if isinstance(timestamp, str):
        ax.set_title(f'Water Level Predictions - Station {station_id}', fontweight='bold', pad=15)
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
    if model_config:
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
        
        # Save and open in browser
        html_path = 'scaled_predictions.html'
        fig.write_html(html_path)
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
    # Ensure output directory exists
    output_dir = Path("Project_Code - CORRECT/results/lstm")
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


def create_performance_analysis_plot(actual, predictions, station_id, model_config=None, output_dir=None):

    """
    Create a comprehensive performance analysis plot with multiple subplots:
    1. Time series comparison during peak events
    2. Scatter plot of predicted vs actual values with metrics
    3. Error distribution histogram
    4. Residuals over time

    Args:
        actual: Series containing actual values with datetime index
        predictions: Series containing model predictions with datetime index
        station_id: ID of the station
        model_config: Optional model configuration
        output_dir: Optional output directory path
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/lstm")
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
    
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)
    
    # Calculate performance metrics
    # Remove NaN values for metric calculation
    valid_mask = (~np.isnan(actual.values)) & (~np.isnan(predictions.values))
    valid_actual = actual.values[valid_mask]
    valid_predictions = predictions.values[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(valid_actual, valid_predictions))
    mae = mean_absolute_error(valid_actual, valid_predictions)
    r2 = r2_score(valid_actual, valid_predictions)
    
    # Create a DataFrame for easier analysis
    analysis_df = pd.DataFrame({
        'Actual': actual.values,
        'Predicted': predictions.values,
        'Error': predictions.values - actual.values,
        'Date': actual.index
    })
    analysis_df['AbsError'] = np.abs(analysis_df['Error'])
    
    # 1. Time Series Plot - Focus on peaks
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Identify peaks (top 90th percentile)
    peak_threshold = np.nanpercentile(actual.values, 90)
    peak_mask = actual.values >= peak_threshold
    peak_indices = np.where(peak_mask)[0]
    
    # Expand indices to show some context around peaks
    context_window = 20  # days around each peak
    peak_regions = []
    
    for idx in peak_indices:
        start_idx = max(0, idx - context_window)
        end_idx = min(len(actual), idx + context_window)
        peak_regions.extend(range(start_idx, end_idx))
    
    # Remove duplicates and sort
    peak_regions = sorted(set(peak_regions))
    
    # If we have peaks, plot them with context
    if peak_regions:
        # Get the dates and values for the peak regions
        peak_dates = actual.index[peak_regions]
        peak_actual = actual.values[peak_regions]
        peak_predicted = predictions.values[peak_regions]
        
        # Plot the actual and predicted values for peak regions
        ax1.plot(peak_dates, peak_actual, color='#1f77b4', linewidth=1.8, label='Actual')
        ax1.plot(peak_dates, peak_predicted, color='#d62728', linewidth=1.8, label='Predicted')
        
        # Highlight the actual peaks
        peak_dates_only = actual.index[peak_indices]
        peak_values_only = actual.values[peak_indices]
        ax1.scatter(peak_dates_only, peak_values_only, color='blue', s=40, alpha=0.7, label='Peak Events')
    else:
        # If no peaks found, just plot the first portion of the data
        sample_size = min(100, len(actual))
        ax1.plot(actual.index[:sample_size], actual.values[:sample_size], color='#1f77b4', linewidth=1.8, label='Actual')
        ax1.plot(predictions.index[:sample_size], predictions.values[:sample_size], color='#d62728', linewidth=1.8, label='Predicted')
    
    ax1.set_title('Performance During Peak Events', fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Water Level (mm)', fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Format the date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter Plot of Predicted vs Actual
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create custom colormap for density visualization
    colors = [(0.95, 0.95, 0.95), (0.2, 0.6, 0.8)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Plot scatter with density coloring
    sc = ax2.scatter(valid_actual, valid_predictions, alpha=0.6, s=15, 
                   c=np.ones(len(valid_actual)), cmap=cmap, edgecolors='none')
    
    # Add perfect prediction line
    min_val = min(np.min(valid_actual), np.min(valid_predictions))
    max_val = max(np.max(valid_actual), np.max(valid_predictions))
    margin = (max_val - min_val) * 0.05  # 5% margin
    ax2.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
           'k--', alpha=0.8, linewidth=1, label='Perfect Prediction')
    
    # Add metrics text
    metrics_text = "Model Performance:<br>"
    if metrics:
        for key, value in metrics.items():
            # Handle extreme R² values with a special case
            if key == 'r2' and value == -1 and 'r2_original' in metrics:
                metrics_text += f"{key}: -1 (capped from {metrics['r2_original']:.4f})<br>"
            elif key == 'r2_original':
                # Skip this as we're displaying it with r2
                continue
            elif isinstance(value, (int, float)) and not np.isnan(value):
                metrics_text += f"{key}: {value:.4f}<br>"
            else:
                metrics_text += f"{key}: {value}<br>"
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
           verticalalignment='top', 
           fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', 
                    edgecolor='#cccccc',
                    alpha=0.9))
    
    ax2.set_title('Predicted vs Actual Values', fontweight='bold')
    ax2.set_xlabel('Actual Water Level (mm)', fontweight='bold')
    ax2.set_ylabel('Predicted Water Level (mm)', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_aspect('equal')
    
    # 3. Error Distribution Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate error statistics
    error_mean = np.nanmean(analysis_df['Error'])
    error_std = np.nanstd(analysis_df['Error'])
    
    # Plot histogram with KDE-like smooth curve
    n, bins, patches = ax3.hist(analysis_df['Error'].dropna(), bins=50, 
                               alpha=0.7, color='#1f77b4', density=True)
    
    # Add vertical line at mean
    ax3.axvline(error_mean, color='#d62728', linestyle='--', linewidth=1.5, 
               label=f'Mean Error: {error_mean:.2f} mm')
    
    # Add error statistics text box
    error_text = (
        f'Mean Error: {error_mean:.2f} mm\n'
        f'Error Std: {error_std:.2f} mm\n'
        f'95% Error Range: [{np.nanpercentile(analysis_df["Error"], 2.5):.1f}, '
        f'{np.nanpercentile(analysis_df["Error"], 97.5):.1f}] mm'
    )
    ax3.text(0.95, 0.95, error_text, transform=ax3.transAxes,
           verticalalignment='top', horizontalalignment='right',
           fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', 
                    edgecolor='#cccccc',
                    alpha=0.9))
    
    ax3.set_title('Error Distribution', fontweight='bold')
    ax3.set_xlabel('Prediction Error (mm)', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Residuals Over Time (Error vs Time)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get absolute error over time, dropping NaN values
    valid_dates = analysis_df.dropna(subset=['Error', 'Date'])['Date']
    valid_errors = analysis_df.dropna(subset=['Error', 'Date'])['Error']
    
    # Ensure we have proper datetime objects
    if len(valid_dates) == len(valid_errors):
        # Check that dates are in the correct range (not showing 1970)
        min_year = pd.to_datetime(valid_dates.min()).year
        if min_year < 2000:  # Likely a date conversion issue
            print(f"Warning: Date conversion issue detected - min year: {min_year}")
            # Try to convert timestamps properly if they're in epoch format
            try:
                # Create a proper time-indexed Series for plotting
                error_series = pd.Series(valid_errors.values, index=pd.to_datetime(valid_dates))
                # Plot using the Series' datetime index
                ax4.plot(error_series.index, error_series.values, 'o', color='#1f77b4', alpha=0.3, markersize=3)
                ax4.axhline(y=0, color='#d62728', linestyle='-', linewidth=1.5)
            except Exception as e:
                print(f"Error converting dates: {e}")
                # Fallback to just plotting errors without dates
                ax4.plot(np.arange(len(valid_errors)), valid_errors, 'o', color='#1f77b4', alpha=0.3, markersize=3)
                ax4.axhline(y=0, color='#d62728', linestyle='-', linewidth=1.5)
                ax4.set_xlabel('Sample Index', fontweight='bold')  # Change label if using indices
        else:
            # Dates look correct, proceed normally
            ax4.plot(valid_dates, valid_errors, 'o', color='#1f77b4', alpha=0.3, markersize=3)
            ax4.axhline(y=0, color='#d62728', linestyle='-', linewidth=1.5)
    
    # Add trend line if we have enough data
    if len(valid_dates) > 10:
        try:
            from scipy.signal import savgol_filter
            
            # Create proper time series for trend analysis
            error_series = pd.Series(valid_errors.values, index=pd.to_datetime(valid_dates))
            
            # Calculate smooth trend line using Savitzky-Golay filter
            y_trend = savgol_filter(error_series.values, min(51, len(error_series) // 10 * 2 + 1), 3)
            
            # Plot trend using the proper datetime index
            ax4.plot(error_series.index, y_trend, color='#d62728', linewidth=2, label='Error Trend')
            ax4.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
        except Exception as e:
            print(f"Could not create trend line: {e}")
            # Skip trend line if it fails
            pass
    
    ax4.set_title('Residual Analysis Over Time', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('Error (mm)', fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Format the date axis with proper locators for better display
    if min_year >= 2000:  # Only if dates look correct
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Set proper locators based on date range
        date_range = (pd.to_datetime(valid_dates.max()) - pd.to_datetime(valid_dates.min())).days
        if date_range > 365*2:
            ax4.xaxis.set_major_locator(mdates.YearLocator())
        elif date_range > 180:
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax4.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax4.tick_params(axis='x', rotation=45)
    
    # Main title with model information
    title_parts = [f'Model Performance Analysis - Station {station_id}']
    if model_config:
        config_summary = (f'Model: LSTM (Hidden: {model_config.get("hidden_size", "N/A")}, '
                         f'Layers: {model_config.get("num_layers", "N/A")}, '
                         f'LR: {model_config.get("learning_rate", 0.001):.5f})')
        title_parts.append(config_summary)
    
    fig.suptitle('\n'.join(title_parts), fontweight='bold', fontsize=18, y=0.98)
    
    # Save the figure
    output_path = output_dir / f'performance_analysis_station_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved performance analysis plot to: {output_path}")
    plt.close()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_error': error_mean,
        'std_error': error_std
    }

def plot_scaled_vs_unscaled_features(data, scaled_data, feature_cols, output_dir=None):

    """
    Create an interactive HTML plot comparing scaled and unscaled versions of all features and targets.
    
    Args:
        data: DataFrame containing unscaled data with datetime index
        scaled_data: DataFrame containing scaled data with datetime index
        feature_cols: List of feature column names to plot
        output_dir: Optional output directory path
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/lstm")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots - one for each feature
    n_features = len(feature_cols)
    
    # Create a subplot figure with Plotly
    fig = make_subplots(
        rows=n_features, 
        cols=2, 
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # Plot each feature
    for i, feature in enumerate(feature_cols):
        # Unscaled plot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[feature],
                name=f'{feature} (Unscaled)',
                line=dict(color='#1f77b4', width=1),
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Scaled plot
        fig.add_trace(
            go.Scatter(
                x=scaled_data.index,
                y=scaled_data[feature],
                name=f'{feature} (Scaled)',
                line=dict(color='#d62728', width=1),
                showlegend=False
            ),
            row=i+1, col=2
        )
        
        # Add statistics text as annotations
        unscaled_mean = np.nanmean(data[feature])
        unscaled_std = np.nanstd(data[feature])
        scaled_mean = np.nanmean(scaled_data[feature])
        scaled_std = np.nanstd(scaled_data[feature])
        
        stats_text = (
            f'Unscaled: Mean={unscaled_mean:.2f}, Std={unscaled_std:.2f}<br>'
            f'Scaled: Mean={scaled_mean:.2f}, Std={scaled_std:.2f}'
        )
        
        # Add text annotation for statistics
        fig.add_annotation(
            x=0.05, y=0.95,
            xref=f'x{i*2+1}', yref=f'y{i*2+1}',
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#cccccc',
            borderwidth=1,
            borderpad=4
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Date', row=i+1, col=1)
        fig.update_xaxes(title_text='Date', row=i+1, col=2)
        fig.update_yaxes(title_text='Value', row=i+1, col=1)
        fig.update_yaxes(title_text='Scaled Value', row=i+1, col=2)
    
    # Update layout
    fig.update_layout(
        title='Scaled vs Unscaled Features Comparison',
        height=300 * n_features,  # Adjust height based on number of features
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    # Set x-axis range for all subplots
    start_date = pd.Timestamp('2010-01-04')
    end_date = pd.Timestamp('2020-12-31')
    
    # Update the x-axis range for all subplots
    for i in range(n_features):
        # Update x-axis range for unscaled plots
        fig.update_xaxes(range=[start_date, end_date], row=i+1, col=1)
        # Update x-axis range for scaled plots
        fig.update_xaxes(range=[start_date, end_date], row=i+1, col=2)
    

    # Display the plot in the browser without saving
    fig.show()
    
    # Return None since we're not saving the file
    return None


def plot_features_stacked_plots(data, feature_cols, output_dir=None, years_to_show=3):
    """
    Create a publication-quality plot of engineered features, organized by station.
    Each station has its own subplot showing its rainfall-related features.
    Temperature and water level data are shown in separate subplots.
    
    Args:
        data: DataFrame containing feature data
        feature_cols: List of feature column names
        output_dir: Optional output directory path (default: saves to Project_Code - CORRECT/results/feature_plots)
        years_to_show: Number of most recent years to display (default: 3)
    
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/feature_plots")
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
        'Main Station': [],
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
            station_features['Main Station'].append(feature)
    
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
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
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
        '30day': '#219ebc',    # Light blue for 30-day features
        '180day': '#023047',   # Dark blue for 180-day features
        '365day': '#8ecae6',   # Medium blue for 365-day features
        'rainfall': '#2a9d8f', # Teal for direct rainfall
        'temperature': '#fb8500',  # Orange for temperature
        'water_level': '#e63946',  # Red for water level
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
                    linewidth=1.8,
                    alpha=0.9
                )
                
                # Plot corresponding cos curve
                ax.plot(
                    data.index, 
                    data[cos_f], 
                    label=cos_label,
                    color=colors[f'{feature_type}_cos'],
                    linestyle=line_styles['cos'],
                    linewidth=1.5,
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
                    linewidth=1.8,
                    alpha=0.9
                )
                
        elif group_name == 'Water Level':
            # Plot each water level feature
            for feature in features:
                station_label = 'Main Station'
                if 'feature1' in feature or 'feature_station_21006845' in feature:
                    station_label = 'Station 21006845'
                elif 'feature2' in feature or 'feature_station_21006847' in feature:
                    station_label = 'Station 21006847'
                
                ax.plot(
                    data.index, 
                    data[feature], 
                    label=f'Water Level ({station_label})',
                    color=colors['water_level'],
                    linewidth=1.8,
                    alpha=0.9
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
            
            # Plot each feature group
            for feature_type, feats in feature_groups.items():
                # Create friendly label
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
                
                for feat in feats:
                    ax.plot(
                        data.index, 
                        data[feat], 
                        label=label,
                        color=colors[feature_type],
                        linewidth=1.8,
                        alpha=0.9
                    )
        
        # Set title and format axes
        ax.set_title(group_name, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
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
        
        # Add proper legends with distinct entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
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
    
    # Add a main title with date range information
    period_str = ""
    if isinstance(data.index, pd.DatetimeIndex) and not data.empty:
        period_str = f" ({data.index.min().strftime('%b %Y')} to {data.index.max().strftime('%b %Y')})"
    
    fig.suptitle(f'Station Features Overview{period_str}', fontweight='bold', y=1.02)
    
    # Save the figure with high resolution
    output_path = output_dir / f'station_features_{timestamp}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"Saved station features plot to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

def create_interactive_feature_plot(data, feature_groups, output_dir, timestamp):
    """Helper function to create an interactive plotly version of the feature plot"""
    # Create a subplot figure with Plotly
    fig_html = make_subplots(
        rows=len(feature_groups), 
        cols=1, 
        vertical_spacing=0.1,
        subplot_titles=list(feature_groups.keys())
    )
    
    # Define a color palette for consistency
    plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot each feature group
    for i, (group_name, features) in enumerate(feature_groups.items()):
        # Group features by station for better organization
        if 'Time Features' in group_name:
            # For time features, organize by month/day rather than station
            sin_features = sorted([f for f in features if 'sin' in f])
            cos_features = sorted([f for f in features if 'cos' in f])
            
            # Plot sin features
            for j, feature in enumerate(sin_features):
                name = feature.replace('_sin', '').replace('month', 'Month').replace('day_of_year', 'Day of Year')
                fig_html.add_trace(
            go.Scatter(
                x=data.index,
                y=data[feature],
                        name=f"{name} (sin)",
                        line=dict(width=1.5, color=plotly_colors[j % len(plotly_colors)]),
                        legendgroup=name,
                    ),  
                    row=i+1, col=1
                )
            
            # Plot cos features
            for j, feature in enumerate(cos_features):
                name = feature.replace('_cos', '').replace('month', 'Month').replace('day_of_year', 'Day of Year')
                fig_html.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[feature],
                        name=f"{name} (cos)",
                        line=dict(width=1.5, color=plotly_colors[j % len(plotly_colors)], dash='dash'),
                        legendgroup=name,
                    ),  
                    row=i+1, col=1
                )
        else:
            # For cumulative features, group by station
            station_features = {}
            for feature in features:
                # Extract station name
                if 'feature1' in feature:
                    station = 'Station 21006845'
                elif 'feature2' in feature:
                    station = 'Station 21006847'
                else:
                    station = 'Main Station'
                
                if station not in station_features:
                    station_features[station] = []
                station_features[station].append(feature)
            
            # Plot each station's features
            for j, (station, feats) in enumerate(station_features.items()):
                for feat in feats:
                    fig_html.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[feat],
                            name=station,
                            line=dict(width=1.5, color=plotly_colors[j % len(plotly_colors)]),
                            legendgroup=station,
            ),  
            row=i+1, col=1
        )
    
    # Update layout
    fig_html.update_layout(
        title='Engineered Features Overview',
        height=275 * len(feature_groups),  # Adjust height based on number of groups
        width=1000,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update all y-axes
    for i in range(1, len(feature_groups) + 1):
        fig_html.update_yaxes(
            title_text=list(feature_groups.keys())[i-1],
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    
    # Update x-axes
    for i in range(1, len(feature_groups) + 1):
        if i == len(feature_groups):
            fig_html.update_xaxes(
                title_text="Date",
                row=i, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
        else:
            fig_html.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
    
    # Save the HTML file
    html_path = output_dir / f'engineered_features_{timestamp}.html'
    fig_html.write_html(str(html_path), include_plotlyjs='cdn')
    print(f"Saved interactive HTML plots to: {html_path}")
    
    return fig_html
    
    

