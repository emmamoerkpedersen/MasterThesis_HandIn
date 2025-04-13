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
        metrics=metrics
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


def create_water_level_plot_png(actual, predictions, station_id, timestamp, model_config=None, output_dir=None, best_val_loss=None, metrics=None):
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
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/lstm")
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
    ax.plot(actual.index, actual.values, color='#1f77b4', linewidth=1.2, label='Actual')
    ax.plot(predictions.index, predictions.values, color='#d62728', linewidth=1.2, label='Predicted')
    
    # Clean styling
    ax.set_title(f'Water Level Predictions - Station {station_id}', fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax.set_ylabel('Water Level (mm)', fontweight='bold', labelpad=10)
    
    # No grid lines as requested
    ax.grid(False)
    
    # Add clean legend
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best')
    
    # Format x-axis dates with better spacing
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


def plot_features_stacked_plots(data, feature_cols):
    """
    Create a stacked plot of all feature engineering rainfall data.
    
    Args:
        data: DataFrame containing rainfall data
        feature_cols: List of feature column names to plot
        output_dir: Optional output directory path
    """

    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots - one for each feature
    n_features = len(feature_cols)
    
    # Create subplot titles from feature names
    subplot_titles = [f"{feature}" for feature in feature_cols]
    
    # Create a subplot figure with Plotly
    fig = make_subplots(
        rows=n_features, 
        cols=1, 
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles
    )
    
    # Plot each feature
    for i, feature in enumerate(feature_cols):
        # Create a line plot for the feature
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[feature],
                name=feature,
                line=dict(color='#1f77b4', width=1),
               
            ),  
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Stacked Features Comparison',
        height=200 * n_features,  # Adjust height based on number of features
        width=1200,
        showlegend=False,
        template='plotly_white',
    )   
    
    # Display the plot in the browser without saving
    fig.show()
    
    # Return None since we're not saving the file
    return None
    
    

