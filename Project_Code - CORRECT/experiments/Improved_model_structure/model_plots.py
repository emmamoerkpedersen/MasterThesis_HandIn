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


def create_full_plot(test_data, test_predictions, station_id, model_config=None):
    """
    Create an interactive plot with aligned datetime indices, rainfall data, and model configuration.
    
    Args:
        test_data: DataFrame containing test data with datetime index
        test_predictions: DataFrame or Series containing predictions
        station_id: ID of the station being analyzed
        model_config: Optional dictionary containing model configuration parameters
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
    
    # Create model configuration text
    config_text = ""
    if model_config:
        config_text = (
            f"Model Configuration:<br>"
            f"Hidden Size: {model_config.get('hidden_size', 'N/A')}<br>"
            f"Layers: {model_config.get('num_layers', 'N/A')}<br>"
            f"Learning Rate: {model_config.get('learning_rate', 0.001):.6f}<br>"
            f"Batch Size: {model_config.get('batch_size', 'N/A')}<br>"
            f"Time Features: {model_config.get('use_time_features', False)}<br>"
            f"Peak Weighted Loss: {model_config.get('use_peak_weighted_loss', False)}"
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Prediction Analysis for Station {station_id}',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        width=1500,  # Increased width
        height=1000,  # Increased height
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
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add configuration text as a more visible annotation
    if config_text:
        fig.add_annotation(
            x=1.15,  # Place it to the right of the plot
            y=0.5,
                xref="paper", 
                yref="paper",
                text=config_text,
                showarrow=False,
            font=dict(size=14),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#000000",
                borderwidth=1,
            borderpad=10
            )
        # Adjust margins to make room for config text
        fig.update_layout(margin=dict(r=250))
    
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
    
    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Create and save PNG version with just water level
    create_water_level_plot_png(test_actual, predictions_series, station_id, timestamp, model_config, output_dir)
    
    # Open HTML in browser
    absolute_path = os.path.abspath(html_path)
    print(f"Opening plot in browser: {absolute_path}")
    webbrowser.open('file://' + str(absolute_path))


def create_water_level_plot_png(actual, predictions, station_id, timestamp, model_config=None, output_dir=None):
    """
    Create a publication-quality matplotlib plot with just water level data and save as PNG.
    Designed for thesis report with consistent colors and clean styling.
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
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
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
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout and save with high quality
    plt.tight_layout()
    output_path = output_dir / f'water_level_station_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved water level PNG plot to: {output_path}")
    plt.close()


def plot_scaled_predictions(predictions, targets, station_id=None, title="Scaled Predictions vs Targets"):
        """
        Plot scaled predictions and targets before inverse transformation.
        
        Args:
            predictions: Numpy array of scaled predictions
            targets: Numpy array of scaled targets
            station_id: Optional station ID for filename
            title: Plot title
        """
        # Ensure output directory exists
        output_dir = Path("Project_Code - CORRECT/results/lstm")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig = go.Figure()
        
        # Flatten predictions and targets for plotting
        flat_predictions = predictions.reshape(-1)
        flat_targets = targets.reshape(-1)
        
        # Create x-axis points
        x_points = np.arange(len(flat_predictions))
        
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
            xaxis_title='Timestep',
            yaxis_title='Scaled Value',
            width=1200,
            height=600,
            showlegend=True
        )
        
        # Save and open in browser
        station_suffix = f"_station_{station_id}" if station_id else ""
        html_path = output_dir / f'scaled_predictions{station_suffix}.html'
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
    
    # Add metrics text box
    metrics_text = (
        f'RMSE: {rmse:.2f} mm\n'
        f'MAE: {mae:.2f} mm\n'
        f'R²: {r2:.3f}'
    )
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