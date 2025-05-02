import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

class ForecastVisualizer:
    """
    Class for visualizing water level forecasts and anomalies.
    """
    def __init__(self, config=None):
        """
        Initialize the visualizer with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.anomaly_threshold = config.get('z_score_threshold', 5) if config else 5
    
    def plot_forecast_with_anomalies(self, results, title="Water Level Forecasting", save_path=None):
        """
        Plot the forecast results with detected anomalies and confidence bands.
        
        Args:
            results: Dictionary with forecasts and anomalies from the predict method
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        
        # Check for additional data 
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot water levels and forecasts
        ax1.plot(original_data.index, original_data.values, 'b-', label='Original Water Levels', linewidth=1)
        
        # If we have error-injected data, plot it
        if error_injected_data is not None:
            ax1.plot(error_injected_data.index, error_injected_data.values, 'r-', 
                    label='Water Levels (with injected errors)', linewidth=1)
        
        # Calculate and plot confidence bands - NEW CODE
        
        # First, collect all forecast columns
        forecast_columns = [col for col in forecast_data.columns if col.startswith('step_')]
        
        if len(forecast_columns) > 1:
            # We have multiple predictions per timestep
            
            # Build a forward-looking table where each row represents a specific timestamp
            # and each column represents a prediction made N steps before
            prediction_table = pd.DataFrame(index=forecast_data.index)
            
            for col in forecast_columns:
                # Extract the step number (assumes format is 'step_N')
                step = int(col.replace('step_', ''))
                
                # Shift forecasts forward by the step number to align with the actual timestamp
                # they're predicting
                if step > 0:
                    shifted_col = forecast_data[col].shift(-step)
                    prediction_table[f'pred_{step}_ahead'] = shifted_col
            
            # Remove rows with all NaNs or insufficient predictions
            prediction_table = prediction_table.dropna(how='all')
            
            # Calculate statistics
            pred_mean = prediction_table.mean(axis=1)
            pred_std = prediction_table.std(axis=1)
            
            # Plot the mean prediction
            ax1.plot(pred_mean.index, pred_mean.values, 'g-', 
                    label='Mean Forecast', linewidth=2)
            
            # Plot confidence bands (mean ± 1 std deviation)
            ax1.fill_between(pred_mean.index, 
                            pred_mean - pred_std, 
                            pred_mean + pred_std, 
                            color='g', alpha=0.2, 
                            label='Forecast Confidence Band (±1σ)')
            
            # Optionally plot a wider band for ±2 std deviations
            ax1.fill_between(pred_mean.index, 
                            pred_mean - 2*pred_std, 
                            pred_mean + 2*pred_std, 
                            color='g', alpha=0.1, 
                            label='Forecast Confidence Band (±2σ)')
        else:
            # Only one prediction per timestep, use the original code
            if 'step_24' in forecast_data.columns:
                ax1.plot(forecast_data.index, forecast_data['step_24'].values, 'g-', 
                        label='24-Step Ahead Forecast', linewidth=1.5)
            else:
                # Use the last column as the forecast
                last_step = forecast_data.columns[-1]
                ax1.plot(forecast_data.index, forecast_data[last_step].values, 'g-', 
                        label=f'{last_step.replace("step_", "")}-Step Ahead Forecast', linewidth=1.5)
        
        # Plot the clean forecast if available
        if clean_forecast is not None:
            if 'step_24' in clean_forecast.columns:
                ax1.plot(clean_forecast.index, clean_forecast['step_24'].values, '--', color='purple',
                        label='24-Step Ahead Forecast (clean data)', linewidth=1.5)
            else:
                last_step = clean_forecast.columns[-1]
                ax1.plot(clean_forecast.index, clean_forecast[last_step].values, '--', color='purple',
                        label=f'{last_step.replace("step_", "")}-Step Ahead Forecast (clean data)', linewidth=1.5)
        
        # Highlight anomalies by confidence level
        anomaly_points = detected_anomalies[detected_anomalies['is_anomaly']]
        if len(anomaly_points) > 0:
            # Choose the source data for anomaly points
            source_data = error_injected_data if error_injected_data is not None else original_data
            
            # Check if we have confidence information
            if 'confidence' in anomaly_points.columns:
                # Plot different confidence levels with different colors
                for confidence, color, marker in [
                    ('low', 'green', 'o'),      # Green circles for low confidence anomalies
                    ('medium', 'orange', 'o'),  # Orange circles for medium confidence anomalies
                    ('high', 'red', 'o')        # Red circles for high confidence anomalies
                ]:
                    conf_points = anomaly_points[anomaly_points['confidence'] == confidence]
                    if len(conf_points) > 0:
                        ax1.scatter(conf_points.index, source_data.loc[conf_points.index].values, 
                                color=color, marker=marker, s=50, 
                                label=f'{confidence.capitalize()} Confidence Anomalies ({len(conf_points)})')
            else:
                # Fall back to original behavior if no confidence info
                ax1.scatter(anomaly_points.index, source_data.loc[anomaly_points.index].values, 
                          color='orange', marker='o', s=50, 
                          label=f'Detected Anomalies ({len(anomaly_points)})')
        
        # If we have true anomalies, mark them
        true_anomalies = results.get('true_anomalies')
        if true_anomalies is not None:
            true_anomaly_points = true_anomalies[true_anomalies['is_anomaly']]
            if len(true_anomaly_points) > 0:
                ax1.scatter(true_anomaly_points.index, true_anomaly_points['actual'], 
                           color='black', marker='x', s=50, 
                           label=f'True Anomalies ({len(true_anomaly_points)})')
        
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel('Water Level', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot anomaly scores
        # If we have different types of z-scores, plot them all
        if 'z_score_abs' in detected_anomalies.columns and 'z_score_pct' in detected_anomalies.columns:
            ax2.plot(detected_anomalies.index, detected_anomalies['z_score_abs'], 'g-', 
                    label='Absolute Error Score', linewidth=1)
            ax2.plot(detected_anomalies.index, detected_anomalies['z_score_pct'], 'b-', 
                    label='Percentage Error Score', linewidth=1)
            ax2.plot(detected_anomalies.index, detected_anomalies['z_score'], 'r-', 
                    label='Combined Score', linewidth=1.5)
        else:
            # Fall back to original behavior
            ax2.plot(detected_anomalies.index, detected_anomalies['z_score'], 'g-', 
                    label='Anomaly Scores', linewidth=1.5)
            
        # Plot threshold lines for different confidence levels
        low_threshold = self.anomaly_threshold
        medium_threshold = self.anomaly_threshold * 1.5
        high_threshold = self.anomaly_threshold * 2.0
        
        ax2.axhline(y=low_threshold, color='green', linestyle='--', 
                   label=f'Low Confidence Threshold ({low_threshold:.1f})')
        ax2.axhline(y=medium_threshold, color='orange', linestyle='--', 
                   label=f'Medium Confidence Threshold ({medium_threshold:.1f})')
        ax2.axhline(y=high_threshold, color='red', linestyle='--', 
                   label=f'High Confidence Threshold ({high_threshold:.1f})')
        
        # Highlight anomalous regions
        for idx, row in anomaly_points.iterrows():
            # Set color based on confidence level if available
            highlight_color = 'yellow'  # Default
            if 'confidence' in row:
                if row['confidence'] == 'low':
                    highlight_color = 'lightgreen'
                elif row['confidence'] == 'medium':
                    highlight_color = 'lightsalmon'
                elif row['confidence'] == 'high':
                    highlight_color = 'lightcoral'
            
            # Highlight a small region around each anomaly (±3 points if possible)
            if hasattr(detected_anomalies.index, 'get_loc'):
                try:
                    idx_loc = detected_anomalies.index.get_loc(idx)
                    start_idx = max(0, idx_loc - 3)
                    end_idx = min(len(detected_anomalies) - 1, idx_loc + 3)  # Ensure end_idx is within bounds
                    start_date = detected_anomalies.index[start_idx]
                    end_date = detected_anomalies.index[end_idx]
                    ax2.axvspan(start_date, end_date, color=highlight_color, alpha=0.3)
                except (KeyError, TypeError):
                    # If index lookup fails, just highlight the point
                    ax2.axvline(x=idx, color=highlight_color, alpha=0.3)
            else:
                # For non-loc compatible indices
                ax2.axvline(x=idx, color=highlight_color, alpha=0.3)
        
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('Score', fontsize=14)
        ax2.set_title('Anomaly Scores', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Format dates on x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast_with_anomalies_plotly(self, results, title="Water Level Forecasting", save_path=None):
        """
        Create a simplified interactive Plotly visualization of the forecast results.
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Water Level Forecast", "Anomaly Scores"),
            row_heights=[0.7, 0.3]
        )
        
        # Plot water levels and forecasts
        fig.add_trace(
            go.Scatter(
                x=original_data.index,
                y=original_data.values.flatten(),
                name="Original Water Levels",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        if error_injected_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=error_injected_data.index,
                    y=error_injected_data.values.flatten(),
                    name="Water Levels with Errors",
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # Plot forecast
        forecast_col = 'step_24' if 'step_24' in forecast_data.columns else forecast_data.columns[-1]
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data[forecast_col].values,
                name=f"{forecast_col.replace('step_', '')}-Step Ahead Forecast",
                line=dict(color='green', width=1.5)
            ),
            row=1, col=1
        )
        
        # If we have clean forecast, plot it as well
        if clean_forecast is not None:
            clean_forecast_col = 'step_24' if 'step_24' in clean_forecast.columns else clean_forecast.columns[-1]
            fig.add_trace(
                go.Scatter(
                    x=clean_forecast.index,
                    y=clean_forecast[clean_forecast_col].values,
                    name=f"{clean_forecast_col.replace('step_', '')}-Step Ahead Clean Forecast",
                    line=dict(color='purple', width=1.5, dash='dash')
                ),
                row=1, col=1
            )
        
        # Plot anomalies by confidence levels
        anomaly_points = detected_anomalies[detected_anomalies['is_anomaly']]
        if len(anomaly_points) > 0 and 'confidence' in anomaly_points.columns:
            source_data = error_injected_data if error_injected_data is not None else original_data
            
            # Plot each confidence level with different colors
            for confidence, color in [
                ('low', 'green'),
                ('medium', 'orange'),
                ('high', 'red')
            ]:
                conf_points = anomaly_points[anomaly_points['confidence'] == confidence]
                if len(conf_points) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=conf_points.index,
                            y=source_data.loc[conf_points.index].values.flatten(),
                            mode='markers',
                            marker=dict(color=color, size=8),
                            name=f"{confidence.capitalize()} Confidence Anomalies ({len(conf_points)})"
                        ),
                        row=1, col=1
                    )
        
        # Plot anomaly scores
        fig.add_trace(
            go.Scatter(
                x=detected_anomalies.index,
                y=detected_anomalies['z_score'],
                name="Anomaly Score",
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # Add threshold lines for different confidence levels
        low_threshold = self.anomaly_threshold
        medium_threshold = self.anomaly_threshold * 1.5
        high_threshold = self.anomaly_threshold * 2.0
        
        # Low confidence threshold
        fig.add_trace(
            go.Scatter(
                x=[detected_anomalies.index[0], detected_anomalies.index[-1]],
                y=[low_threshold, low_threshold],
                name="Low Confidence Threshold",
                line=dict(color='green', dash='dash', width=1)
            ),
            row=2, col=1
        )
        
        # Medium confidence threshold
        fig.add_trace(
            go.Scatter(
                x=[detected_anomalies.index[0], detected_anomalies.index[-1]],
                y=[medium_threshold, medium_threshold],
                name="Medium Confidence Threshold",
                line=dict(color='orange', dash='dash', width=1)
            ),
            row=2, col=1
        )
        
        # High confidence threshold
        fig.add_trace(
            go.Scatter(
                x=[detected_anomalies.index[0], detected_anomalies.index[-1]],
                y=[high_threshold, high_threshold],
                name="High Confidence Threshold",
                line=dict(color='red', dash='dash', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="all")
                ])
            ),
            row=2, col=1
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_multi_horizon_forecasts(self, results, horizons=None, title="Multi-Horizon Water Level Forecasting", save_path=None):
        """
        Plot forecasts at different horizons and compare their performance.
        
        Args:
            results: Dictionary with test data forecasting results
            horizons: List of forecast horizons to plot (e.g., [1, 12, 24, 48, 72])
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        
        # Default horizons if not provided
        if horizons is None:
            # Try to find available horizons in the data
            horizons = []
            for col in forecast_data.columns:
                if col.startswith('step_'):
                    try:
                        horizon = int(col.split('_')[1])
                        horizons.append(horizon)
                    except ValueError:
                        continue
            
            # Sort horizons
            horizons.sort()
            
            # If still empty, use default horizons
            if not horizons:
                horizons = [1, 24]
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        # Plot actual water levels
        plt.plot(original_data.index, original_data.values, 'b-', 
                label='Actual Water Levels', linewidth=2)
        
        # Plot each forecast horizon with a different color
        colors = ['g', 'c', 'm', 'y', 'r']  # Colors for different horizons
        
        for i, horizon in enumerate(horizons):
            horizon_col = f'step_{horizon}'
            if horizon_col in forecast_data.columns:
                color_idx = i % len(colors)  # Cycle through colors if more horizons than colors
                plt.plot(forecast_data.index, forecast_data[horizon_col].values, f'{colors[color_idx]}-', 
                        label=f'{horizon}-Step Ahead Forecast', linewidth=1.5)
        
        # Mark detected anomalies
        anomaly_points = detected_anomalies[detected_anomalies['is_anomaly']]
        if len(anomaly_points) > 0:
            plt.scatter(anomaly_points.index, original_data.loc[anomaly_points.index].values, 
                      color='red', marker='o', s=50, 
                      label=f'Detected Anomalies ({len(anomaly_points)})')
        
        # Highlight anomalous regions
        for idx, row in anomaly_points.iterrows():
            # Highlight a small region around each anomaly
            if hasattr(detected_anomalies.index, 'get_loc'):
                try:
                    idx_loc = detected_anomalies.index.get_loc(idx)
                    start_idx = max(0, idx_loc - 3)
                    end_idx = min(len(detected_anomalies) - 1, idx_loc + 3)  # Ensure end_idx is within bounds
                    start_date = detected_anomalies.index[start_idx]
                    end_date = detected_anomalies.index[end_idx]
                    plt.axvspan(start_date, end_date, color='lightgray', alpha=0.3)
                except (KeyError, TypeError):
                    # If index lookup fails, just highlight the point
                    plt.axvline(x=idx, color='lightgray', alpha=0.3)
            else:
                # For non-loc compatible indices
                plt.axvline(x=idx, color='lightgray', alpha=0.3)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Water Level', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast_accuracy(self, results, horizons=None, save_path=None):
        """
        Plot forecast accuracy at different time horizons.
        
        Args:
            results: Dictionary with test data forecasting results
            horizons: List of forecast horizons to plot
            save_path: Path to save the plot
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        
        # If horizons not specified, use all available in the data
        if horizons is None:
            horizons = []
            for col in forecast_data.columns:
                if col.startswith('step_'):
                    try:
                        horizon = int(col.split('_')[1])
                        horizons.append(horizon)
                    except ValueError:
                        continue
            horizons.sort()
        
        # Calculate MAE for each forecast horizon
        mae_by_horizon = []
        forecast_horizons = []
        
        for horizon in horizons:
            horizon_col = f'step_{horizon}'
            if horizon_col in forecast_data.columns:
                # Calculate MAE for this horizon
                step_mae = np.mean(np.abs(original_data.values - forecast_data[horizon_col].values))
                mae_by_horizon.append(step_mae)
                forecast_horizons.append(horizon)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_horizons, mae_by_horizon, 'b-o', linewidth=2)
        plt.title('Forecast Accuracy by Horizon', fontsize=16)
        plt.xlabel('Forecast Horizon (hours)', fontsize=14)
        plt.ylabel('Mean Absolute Error', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add data labels
        for i, txt in enumerate(mae_by_horizon):
            plt.annotate(f'{txt:.2f}', 
                       (forecast_horizons[i], mae_by_horizon[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_impact(self, results, period, save_path=None, window_days=10):
        """Plot the impact of injected errors on a specific period with confidence bands"""
        # Extract data
        clean_data = results['clean_data']
        error_data = results['error_injected_data']
        forecast_data = results['forecasts']
        clean_forecast = results.get('clean_forecast', None)
        
        # Get first step predictions
        forecast_step1 = forecast_data['step_1'] if isinstance(forecast_data, pd.DataFrame) else forecast_data
        clean_forecast_step1 = clean_forecast['step_1'] if clean_forecast is not None and isinstance(clean_forecast, pd.DataFrame) else clean_forecast
        
        # Convert period timestamps to pandas datetime
        start = pd.Timestamp(period['start'])
        end = pd.Timestamp(period['end'])
        
        # Create window around error period
        window_start = start - pd.Timedelta(days=window_days)
        window_end = end + pd.Timedelta(days=window_days)
        
        # Ensure all data is aligned to the same index before filtering
        # First, convert clean_data to DataFrame if it's a Series
        if isinstance(clean_data, pd.Series):
            clean_data = pd.DataFrame(clean_data)
        
        # Get the common index from clean_data
        common_index = clean_data.index
        
        # Convert error_data to DataFrame if needed
        if isinstance(error_data, pd.Series):
            error_data = pd.DataFrame(error_data)
        
        # Align all data to the common index
        error_data = error_data.reindex(common_index)
        forecast_step1 = pd.Series(forecast_step1, index=common_index)
        if clean_forecast_step1 is not None:
            clean_forecast_step1 = pd.Series(clean_forecast_step1, index=common_index)
        
        # Create the window mask
        window_mask = (common_index >= window_start) & (common_index <= window_end)
        
        # Filter data to window
        clean_data_window = clean_data[window_mask]
        error_data_window = error_data[window_mask]
        forecast_step1_window = forecast_step1[window_mask]
        if clean_forecast_step1 is not None:
            clean_forecast_step1_window = clean_forecast_step1[window_mask]
        
        # Create mask for the error period within the window
        error_period_mask = (clean_data_window.index >= start) & (clean_data_window.index <= end)
        
        # Calculate metrics
        if clean_forecast_step1 is not None:
            mae_normal = np.mean(np.abs(clean_data_window.values - clean_forecast_step1_window.values))
        else:
            mae_normal = None
        
        mae_during_error = np.mean(np.abs(clean_data_window[error_period_mask].values - forecast_step1_window[error_period_mask].values))
        
        # Create the plot with a white background
        plt.figure(figsize=(15, 8), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Plot actual vs predicted values with enhanced styling
        plt.plot(clean_data_window.index, clean_data_window.values.flatten(), 
                label='Clean Data', color='blue', alpha=0.9, linewidth=2)
        plt.plot(error_data_window.index, error_data_window.values.flatten(), 
                label='Error Injected Data', color='red', alpha=0.9, linewidth=2)
                
        # Create confidence bands if we have multiple prediction columns
        forecast_columns = [col for col in forecast_data.columns if col.startswith('step_')]
        
        if len(forecast_columns) > 1:
            # We have multiple predictions per timestep
            prediction_table = pd.DataFrame(index=forecast_data.index)
            
            for col in forecast_columns:
                step = int(col.replace('step_', ''))
                if step > 0:
                    shifted_col = forecast_data[col].shift(-step)
                    prediction_table[f'pred_{step}_ahead'] = shifted_col
            
            # Filter to window
            prediction_table_window = prediction_table[window_mask].copy()
            prediction_table_window = prediction_table_window.dropna(how='all')
            
            # Calculate mean and std for confidence bands
            pred_mean = prediction_table_window.mean(axis=1)
            pred_std = prediction_table_window.std(axis=1)
            
            # Plot the mean prediction
            plt.plot(pred_mean.index, pred_mean.values, 'g-', 
                    label='Mean Forecast', linewidth=2)
            
            # Plot confidence bands (±1 std deviation)
            plt.fill_between(pred_mean.index, 
                            pred_mean - pred_std, 
                            pred_mean + pred_std, 
                            color='g', alpha=0.2, 
                            label='Forecast Confidence Band (±1σ)')
            
            # Plot wider confidence band (±2 std deviations)
            plt.fill_between(pred_mean.index, 
                            pred_mean - 2*pred_std, 
                            pred_mean + 2*pred_std, 
                            color='g', alpha=0.1, 
                            label='Forecast Confidence Band (±2σ)')
        else:
            # Just plot the single forecast line
            plt.plot(forecast_step1_window.index, forecast_step1_window.values, 
                    label='Predictions', color='green', alpha=0.9, linewidth=2)
        
        # Plot clean forecast if available
        if clean_forecast_step1 is not None:
            plt.plot(clean_forecast_step1_window.index, clean_forecast_step1_window.values, 
                    label='Clean Predictions', color='cyan', alpha=0.9, linewidth=2, 
                    linestyle='--', dashes=(5, 5))
        
        # Highlight the error period with better styling
        plt.axvspan(start, end, color='red', alpha=0.1, label='Error Period')
        
        # Customize the plot with enhanced styling
        plt.title(f"Impact of {period['type'].capitalize()} Error\n{period['description']}", 
                 fontsize=14, pad=20)
        plt.xlabel('Time', fontsize=12, labelpad=10)
        plt.ylabel('Water Level', fontsize=12, labelpad=10)
        
        # Enhanced legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fancybox=True, shadow=True)
        
        # Enhanced grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis for better readability of dates
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_impact_plotly(self, results, period, save_path=None, window_days=10):
        """Create an interactive plot showing the impact of injected errors with confidence bands"""
        # Extract data and print shapes for debugging
        clean_data = results['clean_data']
        error_data = results['error_injected_data']
        forecast_data = results['forecasts']
        clean_forecast = results.get('clean_forecast', None)
        
        print("\nDiagnostic Information:")
        print(f"Clean data shape: {clean_data.shape}")
        print(f"Error data shape: {error_data.shape}")
        print(f"Forecast data shape: {forecast_data.shape}")
        if clean_forecast is not None:
            print(f"Clean forecast shape: {clean_forecast.shape}")
        
        # Get first step predictions
        forecast_step1 = forecast_data['step_1'] if isinstance(forecast_data, pd.DataFrame) else forecast_data
        clean_forecast_step1 = clean_forecast['step_1'] if clean_forecast is not None and isinstance(clean_forecast, pd.DataFrame) else clean_forecast
        
        print(f"\nFirst step predictions:")
        print(f"Forecast step1 length: {len(forecast_step1)}")
        if clean_forecast_step1 is not None:
            print(f"Clean forecast step1 length: {len(clean_forecast_step1)}")
        
        # Convert period timestamps to pandas datetime
        start = pd.Timestamp(period['start'])
        end = pd.Timestamp(period['end'])
        
        print(f"\nPeriod: {start} to {end}")
        
        # Create window around error period
        window_start = start - pd.Timedelta(days=window_days)
        window_end = end + pd.Timedelta(days=window_days)
        
        # Ensure all data is aligned to the same index before filtering
        # First, convert clean_data to DataFrame if it's a Series
        if isinstance(clean_data, pd.Series):
            clean_data = pd.DataFrame(clean_data)
        
        # Get the common index from clean_data
        common_index = clean_data.index
        print(f"\nCommon index length: {len(common_index)}")
        
        # Convert error_data to DataFrame if needed
        if isinstance(error_data, pd.Series):
            error_data = pd.DataFrame(error_data)
        
        # Align all data to the common index
        error_data = error_data.reindex(common_index)
        forecast_step1 = pd.Series(forecast_step1, index=common_index)
        if clean_forecast_step1 is not None:
            clean_forecast_step1 = pd.Series(clean_forecast_step1, index=common_index)
        
        # Create the window mask
        window_mask = (common_index >= window_start) & (common_index <= window_end)
        print(f"Window mask length: {len(window_mask)}")
        
        # Filter data to window
        clean_data_window = clean_data[window_mask]
        error_data_window = error_data[window_mask]
        forecast_step1_window = forecast_step1[window_mask]
        if clean_forecast_step1 is not None:
            clean_forecast_step1_window = clean_forecast_step1[window_mask]
        
        print(f"\nWindowed data lengths:")
        print(f"Clean data window: {len(clean_data_window)}")
        print(f"Error data window: {len(error_data_window)}")
        print(f"Forecast window: {len(forecast_step1_window)}")
        
        # Create mask for the error period within the window
        error_period_mask = (clean_data_window.index >= start) & (clean_data_window.index <= end)
        
        # Calculate metrics
        if clean_forecast_step1 is not None:
            mae_normal = np.mean(np.abs(clean_data_window.values - clean_forecast_step1_window.values))
        else:
            mae_normal = None
        
        mae_during_error = np.mean(np.abs(clean_data_window[error_period_mask].values - forecast_step1_window[error_period_mask].values))
        
        # Create figure
        fig = go.Figure()
        
        # Add traces with enhanced styling
        fig.add_trace(
            go.Scatter(x=clean_data_window.index, y=clean_data_window.values.flatten(),
                      name='Clean Data', 
                      line=dict(color='blue', width=2.5))
        )
        
        fig.add_trace(
            go.Scatter(x=error_data_window.index, y=error_data_window.values.flatten(),
                      name='Error Injected Data', 
                      line=dict(color='red', width=2.5))
        )
        
        # Check if we should create confidence bands
        forecast_columns = [col for col in forecast_data.columns if col.startswith('step_')]
        
        if len(forecast_columns) > 1:
            # We have multiple predictions per timestep
            prediction_table = pd.DataFrame(index=forecast_data.index)
            
            for col in forecast_columns:
                step = int(col.replace('step_', ''))
                if step > 0:
                    shifted_col = forecast_data[col].shift(-step)
                    prediction_table[f'pred_{step}_ahead'] = shifted_col
            
            # Filter to window
            prediction_table_window = prediction_table[window_mask].copy()
            prediction_table_window = prediction_table_window.dropna(how='all')
            
            # Calculate mean and std for confidence bands
            pred_mean = prediction_table_window.mean(axis=1)
            pred_std = prediction_table_window.std(axis=1)
            
            # Plot the mean prediction
            fig.add_trace(
                go.Scatter(
                    x=pred_mean.index, 
                    y=pred_mean.values,
                    name='Mean Forecast',
                    line=dict(color='green', width=2.5)
                )
            )
            
            # Add confidence bands using upper/lower bound traces
            # ±1 sigma band
            fig.add_trace(
                go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean + pred_std,
                    name='Upper 1σ',
                    line=dict(color='rgba(0, 128, 0, 0)'),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean - pred_std,
                    line=dict(color='rgba(0, 128, 0, 0)'),
                    fill='tonexty',
                    fillcolor='rgba(0, 128, 0, 0.1)',
                    showlegend=False
                )
            )
            
            # ±2 sigma band (wider)
            fig.add_trace(
                go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean + 2*pred_std,
                    name='Upper 2σ',
                    line=dict(color='rgba(0, 128, 0, 0)'),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean - 2*pred_std,
                    line=dict(color='rgba(0, 128, 0, 0)'),
                    fill='tonexty',
                    fillcolor='rgba(0, 128, 0, 0.1)',
                    showlegend=True,
                    name='Forecast Confidence Bands (±2σ)'
                )
            )
        else:
            # Just plot the single forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_step1_window.index, 
                    y=forecast_step1_window.values,
                    name='Predictions',
                    line=dict(color='green', width=2.5)
                )
            )
        
        if clean_forecast_step1 is not None:
            fig.add_trace(
                go.Scatter(x=clean_forecast_step1_window.index, y=clean_forecast_step1_window.values,
                          name='Clean Predictions', 
                          line=dict(color='cyan', width=2.5, dash='dash'))  # Make clean predictions dashed
            )
        
        # Add error period highlight
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
            name="Error Period"
        )
        
        # Update layout with enhanced styling
        title_text = f"Impact of {period['type'].capitalize()} Error<br>{period['description']}"
        if mae_normal is not None:
            title_text += f"<br>Normal MAE: {mae_normal:.2f}"
        title_text += f"<br>Error Period MAE: {mae_during_error:.2f}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title=dict(text="Time", font=dict(size=14)),
            yaxis_title=dict(text="Water Level", font=dict(size=14)),
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(
                tickformat='%Y-%m-%d',
                tickangle=45,
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showgrid=True
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='lightgray',
                borderwidth=1
            )
        )
        
        # Save or show the plot
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_anomaly_investigation(self, results, anomaly_indices=None, num_anomalies=5, save_path=None):
        """
        Create detailed plots for specific anomalies to investigate them further.
        
        Args:
            results: Dictionary with test data forecasting results
            anomaly_indices: List of indices to anomalies to investigate
            num_anomalies: Number of top anomalies to investigate if anomaly_indices not provided
            save_path: Path to save the plot directory
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        
        # If no specific indices, select top anomalies by z-score
        if anomaly_indices is None:
            # Get top anomalies
            top_anomalies = detected_anomalies[detected_anomalies['is_anomaly']].sort_values('z_score', ascending=False)
            
            if len(top_anomalies) == 0:
                print("No anomalies detected to investigate.")
                return
            
            # Limit to specified number or all if fewer
            num_to_plot = min(num_anomalies, len(top_anomalies))
            top_anomalies = top_anomalies.iloc[:num_to_plot]
            
            # Use indices from top anomalies
            anomaly_indices = top_anomalies.index
        
        # Create directory for anomaly plots if saving
        if save_path:
            save_dir = Path(save_path).parent / "anomaly_investigation"
            save_dir.mkdir(exist_ok=True)
        
        # Plot each anomaly with context
        for i, idx in enumerate(anomaly_indices):
            # Skip if anomaly not in our data
            if idx not in detected_anomalies.index:
                continue
                
            anomaly = detected_anomalies.loc[idx]
            
            # Get window of data around anomaly (7 days before and after)
            window_start = idx - pd.Timedelta(days=7)
            window_end = idx + pd.Timedelta(days=7)
            
            # Filter data to window
            window_mask = (original_data.index >= window_start) & (original_data.index <= window_end)
            window_data = original_data[window_mask]
            
            # Skip if not enough data in window
            if len(window_data) < 5:
                continue
            
            # Get forecasts for different horizons
            window_forecasts = {}
            for col in forecast_data.columns:
                if col.startswith('step_'):
                    try:
                        horizon = int(col.split('_')[1])
                        window_forecasts[f'{horizon}-step'] = forecast_data.loc[window_data.index, col]
                    except (ValueError, KeyError):
                        continue
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual data
            ax.plot(window_data.index, window_data.values, 'b-', label='Actual Water Levels', linewidth=2)
            
            # Plot forecasts (limit to 3 for clarity)
            forecast_keys = sorted(list(window_forecasts.keys()), key=lambda x: int(x.split('-')[0]))[:3]
            for j, key in enumerate(forecast_keys):
                forecast = window_forecasts[key]
                ax.plot(forecast.index, forecast.values, '-', label=f'{key} Forecast', linewidth=1.5)
            
            # Mark the anomaly point with appropriate marker based on type
            anomaly_value = original_data.loc[idx]
            if hasattr(anomaly_value, 'values'):
                # DataFrame or Series with values attribute
                anomaly_value = anomaly_value.values[0]
            elif hasattr(anomaly_value, 'iloc'):
                # Series-like with iloc
                anomaly_value = anomaly_value.iloc[0]
            # Otherwise assume it's already a scalar value
            
            # Determine marker and color based on anomaly type
            marker = 'o'
            color = 'red'
            anomaly_label = f'Anomaly (z-score: {anomaly["z_score"]:.2f})'
            
            if 'anomaly_type' in anomaly:
                if anomaly['anomaly_type'] == 'scaling':
                    marker = 's'  # square
                    color = 'fuchsia'
                    anomaly_label = f'Scaling Anomaly (z-score: {anomaly["z_score"]:.2f})'
                elif anomaly['anomaly_type'] == 'offset':
                    marker = 'o'  # circle
                    color = 'darkorange'
                    anomaly_label = f'Offset Anomaly (z-score: {anomaly["z_score"]:.2f})'
            
            ax.scatter([idx], [anomaly_value], color=color, marker=marker, s=100, label=anomaly_label)
            
            # Add vertical line at anomaly
            ax.axvline(x=idx, color=color, linestyle='--', alpha=0.5)
            
            # Add additional information about anomaly metrics if available
            if 'abs_error' in anomaly and 'percent_error' in anomaly:
                title_text = (f'Anomaly Investigation: {idx.strftime("%Y-%m-%d %H:%M")}\n'
                             f'Abs Error: {anomaly["abs_error"]:.2f}, '
                             f'Percent Error: {anomaly["percent_error"]:.2f}%')
            else:
                title_text = f'Anomaly Investigation: {idx.strftime("%Y-%m-%d %H:%M")}'
                
            plt.title(title_text, fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Water Level', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                anomaly_type = anomaly.get('anomaly_type', 'unknown')
                plt.savefig(save_dir / f"anomaly_{i+1}_{anomaly_type}_{idx.strftime('%Y%m%d_%H%M')}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_horizon_metrics(self, metrics_df, save_path=None):
        """
        Plot comparison of metrics across different forecast horizons.
        
        Args:
            metrics_df: DataFrame with metrics for each horizon
            save_path: Path to save the plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
        
        # Plot MAE by horizon
        horizons = metrics_df['Horizon'].values
        
        # Plot lines for different metrics if they exist
        metrics_to_plot = [
            ('Overall_MAE', 'Overall MAE', 'b-o'),
            ('Normal_MAE', 'MAE During Normal Periods', 'g-o'),
            ('Anomaly_MAE', 'MAE During Anomalies', 'r-o')
        ]
        
        for col, label, style in metrics_to_plot:
            if col in metrics_df.columns:
                ax1.plot(horizons, metrics_df[col].values, style, label=label, linewidth=2)
        
        ax1.set_title('Forecast Error by Horizon', fontsize=16)
        ax1.set_xlabel('Forecast Horizon (hours)', fontsize=14)
        ax1.set_ylabel('Mean Absolute Error', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot impact percentage by horizon if it exists
        if 'MAE_Impact_Percentage' in metrics_df.columns:
            ax2.bar(horizons, metrics_df['MAE_Impact_Percentage'].values, color='orange', alpha=0.7)
            ax2.plot(horizons, metrics_df['MAE_Impact_Percentage'].values, 'r-o', linewidth=2)
            
            ax2.set_title('Anomaly Impact on Forecast Error by Horizon', fontsize=16)
            ax2.set_xlabel('Forecast Horizon (hours)', fontsize=14)
            ax2.set_ylabel('Impact Percentage (%)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i, v in enumerate(metrics_df['MAE_Impact_Percentage'].values):
                ax2.text(horizons[i], v + 5, f'{v:.1f}%', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 
    
    def create_interactive_plot(self, results, title="Water Level Forecast Results", show_anomalies=True):
        """
        Create an interactive plot showing water level forecasts and anomalies.
        
        Args:
            results (dict): Dictionary containing forecast results
            title (str): Plot title
            show_anomalies (bool): Whether to show anomaly indicators
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True, 
                           vertical_spacing=0.1,
                           subplot_titles=(title, "Prediction Convergence"))
        
        # Get timestamps for x-axis
        timestamps = results['timestamps']
        
        # Plot original data
        if 'original_data' in results:
            fig.add_trace(
                go.Scatter(x=timestamps, y=results['original_data']['water_level'],
                          name='Original Data', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Plot error-injected data if available
        if 'error_injected_data' in results:
            fig.add_trace(
                go.Scatter(x=timestamps, y=results['error_injected_data']['water_level'],
                          name='Error-Injected Data', line=dict(color='red', dash='dot')),
                row=1, col=1
            )
        
        # Plot predictions from each iteration
        if 'predictions_by_iteration' in results:
            predictions_df = results['predictions_by_iteration']
            num_iterations = predictions_df.shape[1]
            colors = px.colors.sequential.Viridis[::max(1, len(px.colors.sequential.Viridis)//num_iterations)]
            
            for i, col in enumerate(predictions_df.columns):
                fig.add_trace(
                    go.Scatter(x=timestamps, y=predictions_df[col],
                              name=f'Prediction (Iteration {i+1})',
                              line=dict(color=colors[i], width=1 + i*0.5)),
                    row=1, col=1
                )
            
            # Plot prediction convergence
            if num_iterations > 1:
                changes = []
                for i in range(1, num_iterations):
                    change = np.abs(predictions_df[f'iteration_{i+1}'] - predictions_df[f'iteration_{i}']).mean()
                    changes.append(change)
                
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(changes) + 1)), 
                              y=changes,
                              name='Prediction Change',
                              line=dict(color='purple'),
                              mode='lines+markers'),
                    row=2, col=1
                )
        
        # Plot anomalies if available and requested
        if show_anomalies and 'detected_anomalies' in results:
            anomalies = results['detected_anomalies']
            
            # Plot positive anomalies
            pos_anomalies = anomalies[anomalies['anomaly_type'] == 'positive_spike']
            if not pos_anomalies.empty:
                fig.add_trace(
                    go.Scatter(x=pos_anomalies.index, 
                              y=results['original_data'].loc[pos_anomalies.index, 'water_level'],
                              name='Positive Anomalies',
                              mode='markers',
                              marker=dict(color='red', size=10, symbol='triangle-up')),
                    row=1, col=1
                )
            
            # Plot negative anomalies
            neg_anomalies = anomalies[anomalies['anomaly_type'] == 'negative_spike']
            if not neg_anomalies.empty:
                fig.add_trace(
                    go.Scatter(x=neg_anomalies.index, 
                              y=results['original_data'].loc[neg_anomalies.index, 'water_level'],
                              name='Negative Anomalies',
                              mode='markers',
                              marker=dict(color='orange', size=10, symbol='triangle-down')),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            margin=dict(r=150)  # Make room for legend
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Water Level", row=1, col=1)
        fig.update_yaxes(title_text="Mean Absolute Change", row=2, col=1)
        
        return fig
    
    
    def plot_forecast_simple(self, results, title="Water Level Forecasting", save_path=None, forecast_step=None):
        """
        Create a simplified plot showing water levels, forecasts, and anomalies.
        
        Args:
            results: Dictionary with forecasts and anomalies from the predict method
            title: Plot title
            save_path: Path to save the plot
            forecast_step: Specific forecast step to display (e.g. 'step_24')
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        
        # Check for additional data 
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
        # Create figure
        plt.figure(figsize=(16, 8))
        
        # Plot water levels and forecasts
        plt.plot(original_data.index, original_data.values, 'b-', label='Original Water Levels', linewidth=1)
        
        if error_injected_data is not None:
            plt.plot(error_injected_data.index, error_injected_data.values, 'r-', 
                    label='Water Levels (with injected errors)', linewidth=1)
        
        # First, collect all forecast columns
        forecast_columns = [col for col in forecast_data.columns if col.startswith('step_')]
        
        if len(forecast_columns) > 1:
            # We have multiple predictions per timestep - create confidence bands
            
            # If a specific forecast step was requested, still show it separately
            if forecast_step is not None and forecast_step in forecast_data.columns:
                plt.plot(forecast_data.index, forecast_data[forecast_step].values, 'g-', 
                        label=f'{forecast_step.replace("step_", "")}-Step Ahead Forecast', linewidth=1.5)
            
            # Build a forward-looking table where each row represents a specific timestamp
            # and each column represents a prediction made N steps before
            prediction_table = pd.DataFrame(index=forecast_data.index)
            
            for col in forecast_columns:
                # Extract the step number (assumes format is 'step_N')
                step = int(col.replace('step_', ''))
                
                # Shift forecasts forward by the step number to align with the actual timestamp
                # they're predicting
                if step > 0:
                    shifted_col = forecast_data[col].shift(-step)
                    prediction_table[f'pred_{step}_ahead'] = shifted_col
            
            # Remove rows with all NaNs or insufficient predictions
            prediction_table = prediction_table.dropna(how='all')
            
            # Calculate statistics
            pred_mean = prediction_table.mean(axis=1)
            pred_std = prediction_table.std(axis=1)
            
            # Only plot the mean prediction if we didn't already plot a specific step
            if forecast_step is None:
                plt.plot(pred_mean.index, pred_mean.values, 'g-', 
                        label='Mean Forecast', linewidth=2)
            
            # Plot confidence bands (mean ± 1 std deviation)
            plt.fill_between(pred_mean.index, 
                            pred_mean - pred_std, 
                            pred_mean + pred_std, 
                            color='g', alpha=0.2, 
                            label='Forecast Confidence Band (±1σ)')
            
            # Optionally plot a wider band for ±2 std deviations
            plt.fill_between(pred_mean.index, 
                            pred_mean - 2*pred_std, 
                            pred_mean + 2*pred_std, 
                            color='g', alpha=0.1, 
                            label='Forecast Confidence Band (±2σ)')
        else:
            # Only one prediction per timestep, use the original code
            if forecast_step is not None and forecast_step in forecast_data.columns:
                forecast_col = forecast_step
            elif 'step_24' in forecast_data.columns:
                forecast_col = 'step_24'
            else:
                forecast_col = forecast_data.columns[-1]
                
            plt.plot(forecast_data.index, forecast_data[forecast_col].values, 'g-', 
                    label=f'{forecast_col.replace("step_", "")}-Step Ahead Forecast', linewidth=1.5)
        
        if clean_forecast is not None:
            if forecast_step is not None and forecast_step in clean_forecast.columns:
                clean_forecast_col = forecast_step
            elif 'step_24' in clean_forecast.columns:
                clean_forecast_col = 'step_24'
            else:
                clean_forecast_col = clean_forecast.columns[-1]
                
            plt.plot(clean_forecast.index, clean_forecast[clean_forecast_col].values, '--', color='purple',
                    label=f'{clean_forecast_col.replace("step_", "")}-Step Ahead Forecast (clean data)', linewidth=1.5)
        
        # Highlight anomalies by confidence level
        anomaly_points = detected_anomalies[detected_anomalies['is_anomaly']]
        if len(anomaly_points) > 0:
            # Choose the source data for anomaly points
            source_data = error_injected_data if error_injected_data is not None else original_data
            
            # Check if we have confidence information
            if 'confidence' in anomaly_points.columns:
                # Plot different confidence levels with different colors
                for confidence, color, marker in [
                    ('low', 'green', 'o'),      # Green circles for low confidence anomalies
                    ('medium', 'orange', 'o'),  # Orange circles for medium confidence anomalies
                    ('high', 'red', 'o')        # Red circles for high confidence anomalies
                ]:
                    conf_points = anomaly_points[anomaly_points['confidence'] == confidence]
                    if len(conf_points) > 0:
                        plt.scatter(conf_points.index, source_data.loc[conf_points.index].values, 
                                color=color, marker=marker, s=50, 
                                label=f'{confidence.capitalize()} Confidence Anomalies ({len(conf_points)})')
            else:
                # Fall back to original behavior if no confidence info
                plt.scatter(anomaly_points.index, source_data.loc[anomaly_points.index].values, 
                          color='orange', marker='o', s=50, 
                          label=f'Detected Anomalies ({len(anomaly_points)})')
        
        plt.title(title, fontsize=16)
        plt.ylabel('Water Level', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast_simple_plotly(self, results, title="Water Level Forecasting", save_path=None, forecast_step=None):
        """
        Create a simplified interactive Plotly visualization of the forecast results.
        
        Args:
            results: Dictionary with forecasts and anomalies from the predict method
            title: Plot title
            save_path: Path to save the plot as HTML
            forecast_step: Specific forecast step to display (e.g. 'step_24')
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        detected_anomalies = results['detected_anomalies']
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
        # Determine which forecast column to use
        if forecast_step is not None and forecast_step in forecast_data.columns:
            forecast_col = forecast_step
        elif 'step_24' in forecast_data.columns:
            forecast_col = 'step_24'
        else:
            forecast_col = forecast_data.columns[-1]
        
        # Create figure
        fig = go.Figure()
        
        # Plot water levels
        fig.add_trace(
            go.Scatter(
                x=original_data.index,
                y=original_data.values.flatten(),
                name="Original Water Levels",
                line=dict(color='blue', width=1)
            )
        )
        
        if error_injected_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=error_injected_data.index,
                    y=error_injected_data.values.flatten(),
                    name="Water Levels with Errors",
                    line=dict(color='red', width=1)
                )
            )
        
        # Plot forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data[forecast_col].values,
                name=f"{forecast_col.replace('step_', '')}-Step Ahead Forecast",
                line=dict(color='green', width=1.5)
            )
        )
        
        if clean_forecast is not None:
            fig.add_trace(
                go.Scatter(
                    x=clean_forecast.index,
                    y=clean_forecast[forecast_col].values,
                    name=f"{forecast_col.replace('step_', '')}-Step Ahead Forecast (clean)",
                    line=dict(color='purple', width=1.5, dash='dash')
                )
            )
        
        # Plot anomalies by confidence levels
        anomaly_points = detected_anomalies[detected_anomalies['is_anomaly']]
        if len(anomaly_points) > 0 and 'confidence' in anomaly_points.columns:
            source_data = error_injected_data if error_injected_data is not None else original_data
            
            # Plot each confidence level with different colors
            for confidence, color in [
                ('low', 'green'),
                ('medium', 'orange'),
                ('high', 'red')
            ]:
                conf_points = anomaly_points[anomaly_points['confidence'] == confidence]
                if len(conf_points) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=conf_points.index,
                            y=source_data.loc[conf_points.index].values.flatten(),
                            mode='markers',
                            marker=dict(color=color, size=8),
                            name=f"{confidence.capitalize()} Confidence Anomalies ({len(conf_points)})"
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Water Level",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="all")
                ])
            )
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
        
        return fig