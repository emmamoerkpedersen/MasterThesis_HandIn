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
    Class for visualizing water level forecasts.
    """
    def __init__(self, config=None):
        """
        Initialize the visualizer with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def plot_forecast(self, results, title="Water Level Forecasting", save_path=None):
        """
        Plot the forecast results.
        
        Args:
            results: Dictionary with forecasts from the predict method
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract and ensure proper data format
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        
        # Check for additional data 
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
        # Convert numpy arrays to pandas Series/DataFrame with proper timestamps
        if isinstance(original_data, np.ndarray):
            # Get the actual timestamps from the data if available
            if 'timestamps' in results:
                index = pd.to_datetime(results['timestamps'])
                original_data = pd.DataFrame(original_data, index=index)
            else:
                # If no timestamps provided, use the original data's index
                print("Warning: No timestamps provided. Using default index.")
                original_data = pd.DataFrame(original_data)
        
        # Get the actual date range from the data
        date_range = original_data.index
        print(f"Data range: {date_range[0]} to {date_range[-1]}")
        
        if isinstance(forecast_data, np.ndarray):
            # Create DataFrame with same index as original data
            forecast_df = pd.DataFrame(
                np.zeros(len(date_range)) * np.nan,
                index=date_range
            )
            # Fill in the forecast values at the appropriate indices
            forecast_df.iloc[-len(forecast_data):] = forecast_data
            forecast_data = forecast_df
        
        if error_injected_data is not None and isinstance(error_injected_data, np.ndarray):
            # Create DataFrame with same index as original data
            error_df = pd.DataFrame(
                np.zeros(len(date_range)) * np.nan,
                index=date_range
            )
            # Fill in the error-injected values
            error_df.iloc[:len(error_injected_data)] = error_injected_data
            error_injected_data = error_df
        
        if clean_forecast is not None and isinstance(clean_forecast, np.ndarray):
            # Create DataFrame with same index as original data
            clean_df = pd.DataFrame(
                np.zeros(len(date_range)) * np.nan,
                index=date_range
            )
            # Fill in the clean forecast values
            clean_df.iloc[-len(clean_forecast):] = clean_forecast
            clean_forecast = clean_df
        
        # Create figure
        plt.figure(figsize=(16, 8))
        
        # Plot water levels and forecasts
        plt.plot(original_data.index, original_data.values.flatten(), 'b-', label='Original Water Levels', linewidth=1)
        
        # If we have error-injected data, plot it
        if error_injected_data is not None:
            plt.plot(error_injected_data.index, error_injected_data.values.flatten(), 'r-', 
                    label='Water Levels (with injected errors)', linewidth=1)
        
        # Plot the forecast
        plt.plot(forecast_data.index, forecast_data.values.flatten(), 'g-', 
                label='Forecast', linewidth=1.5)
        
        # Plot the clean forecast if available
        if clean_forecast is not None:
            plt.plot(clean_forecast.index, clean_forecast.values.flatten(), '--', color='purple',
                    label='Forecast (clean data)', linewidth=1.5)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Water Level', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Format dates on x-axis if we have datetime index
        if isinstance(original_data.index, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_forecast_plotly(self, results, title="Water Level Forecasting", save_path=None):
        """
        Create an interactive Plotly visualization of the forecast results.
        """
        # Extract data
        original_data = results['clean_data']
        forecast_data = results['forecasts']
        error_injected_data = results.get('error_injected_data')
        clean_forecast = results.get('clean_forecast')
        
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
        forecast_col = 'step_24' if 'step_24' in forecast_data.columns else forecast_data.columns[-1]
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data[forecast_col].values,
                name=f"{forecast_col.replace('step_', '')}-Step Ahead Forecast",
                line=dict(color='green', width=1.5)
            )
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
        """Plot the impact of injected errors on a specific period"""
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
        plt.plot(forecast_step1_window.index, forecast_step1_window.values, 
                label='Predictions', color='green', alpha=0.9, linewidth=2)
        if clean_forecast_step1 is not None:
            plt.plot(clean_forecast_step1_window.index, clean_forecast_step1_window.values, 
                    label='Clean Predictions', color='cyan', alpha=0.9, linewidth=2, 
                    linestyle='--', dashes=(5, 5))  # Make clean predictions dashed
        
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
        """Create an interactive plot showing the impact of injected errors"""
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
        
        fig.add_trace(
            go.Scatter(x=forecast_step1_window.index, y=forecast_step1_window.values,
                      name='Predictions', 
                      line=dict(color='green', width=2.5))
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
            ('Normal_MAE', 'MAE During Normal Periods', 'g-o')
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
            
            ax2.set_title('Error Impact on Forecast Error by Horizon', fontsize=16)
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
    
    def plot_feature_importance_analysis(self, feature_importance, title="Feature Importance Analysis", save_path=None, min_importance_threshold=0.01):
        """
        Create a comprehensive horizontal bar plot of feature importances.
        
        Args:
            feature_importance: List of tuples (feature_name, importance_value) or dictionary
            title: Plot title
            save_path: Path to save the plot
            min_importance_threshold: Minimum importance value to highlight as potentially removable
        """
        # Convert input to list of tuples if it's a dictionary
        if isinstance(feature_importance, dict):
            feature_importance = list(feature_importance.items())
        
        # Sort by importance value
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Separate features and values
        features = [x[0] for x in feature_importance]
        importance_values = [x[1] for x in feature_importance]
        
        # Create figure with appropriate height
        plt.figure(figsize=(12, max(8, len(features) * 0.3)))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(features)), importance_values)
        
        # Customize the plot
        plt.xlabel('Importance Score')
        plt.title(title)
        
        # Set y-axis ticks and labels
        plt.yticks(range(len(features)), features)
        
        # Add value labels at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}',
                    ha='left', va='center', fontweight='bold')
            
            # Color bars based on importance
            if width < min_importance_threshold:
                bar.set_color('lightcoral')  # Red for potentially removable features
            else:
                bar.set_color('lightblue')   # Blue for important features
        
        # Add a vertical line for the threshold
        plt.axvline(x=min_importance_threshold, color='red', linestyle='--', alpha=0.5)
        plt.text(min_importance_threshold, len(features), 
                f'Threshold: {min_importance_threshold}',
                ha='right', va='top', color='red', alpha=0.7)
        
        # Add grid for better readability
        plt.grid(True, axis='x', alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        total_features = len(features)
        removable_features = sum(1 for v in importance_values if v < min_importance_threshold)
        
        print("\nFeature Importance Summary:")
        print(f"Total features: {total_features}")
        print(f"Features below threshold ({min_importance_threshold}): {removable_features}")
        print(f"Potential feature reduction: {removable_features/total_features*100:.1f}%")
        
        # Group features by importance ranges
        ranges = [(0.1, float('inf'), 'Very High'),
                 (0.05, 0.1, 'High'),
                 (0.01, 0.05, 'Medium'),
                 (0.001, 0.01, 'Low'),
                 (0, 0.001, 'Very Low')]
        
        print("\nFeature Importance Distribution:")
        for min_val, max_val, label in ranges:
            count = sum(1 for v in importance_values if min_val <= v < max_val)
            print(f"{label} importance ({min_val:.3f} - {max_val if max_val != float('inf') else '∞'}): {count} features")
            if count > 0:
                features_in_range = [f for f, v in feature_importance if min_val <= v < max_val]
                print(f"  Features: {', '.join(features_in_range)}")