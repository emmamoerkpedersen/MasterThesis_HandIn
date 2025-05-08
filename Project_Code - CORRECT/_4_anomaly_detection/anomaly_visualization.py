import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def align_predictions_series(test_actual, predictions_values, model_config):
    """
    Align predictions to the correct placement in the time series, matching create_full_plot logic.
    Returns a pandas Series with the same index as test_actual.
    """
    predictions_series = pd.Series(index=test_actual.index, data=np.nan)
    sequence_length = model_config.get('sequence_length', 50) if model_config else 50
    if model_config and model_config.get('model_type') == 'iterative':
        prediction_window = model_config.get('prediction_window', 10)
        start_idx = sequence_length
        for i in range(0, len(predictions_values), prediction_window):
            if start_idx + i + prediction_window <= len(test_actual):
                predictions_series.iloc[start_idx + i:start_idx + i + prediction_window] = predictions_values[i:i + prediction_window]
    else:
        if len(predictions_values) > 0:
            end_idx = min(sequence_length + len(predictions_values), len(test_actual))
            predictions_series.iloc[sequence_length:end_idx] = predictions_values[:end_idx-sequence_length]
    return predictions_series

def plot_water_level_anomalies(
    test_data,
    predictions,
    z_scores,
    anomalies,
    title="Water Level Forecasting with Anomalies",
    output_dir=None,
    save_png=True,
    save_html=True,
    show_plot=True,
    model_config=None
):
    """
    Creates a plot showing water level data, predictions, z-scores, and detected anomalies.
    Aligns predictions/z_scores/anomalies using the same logic as create_full_plot.
    
    Args:
        test_data (pd.DataFrame): DataFrame containing water level data with datetime index.
                                 Should have a 'vst_raw' column for water levels.
        predictions (pd.Series or np.array): Predicted water level values.
        z_scores (np.array): Z-scores calculated from residuals.
        anomalies (np.array): Boolean array indicating anomaly points.
        title (str): Plot title.
        output_dir (str or Path): Directory to save output files.
        save_png (bool): Whether to save plot as PNG.
        save_html (bool): Whether to save interactive plot as HTML.
        show_plot (bool): Whether to display the plot.
        model_config (dict or None): Model configuration dictionary.
        
    Returns:
        tuple: Paths to saved PNG and HTML files (if applicable).
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path("results/anomaly_detection")
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get timestamp for file naming
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure data is in the right format
    if isinstance(test_data, pd.DataFrame):
        actual_values = test_data['vst_raw']
    else:
        actual_values = test_data
    
    n = len(actual_values)

    # Align predictions using the helper
    predictions_series = align_predictions_series(actual_values, predictions, model_config)

    # Align z_scores and anomalies in the same way as predictions
    z_scores_series = pd.Series(index=actual_values.index, data=np.nan)
    anomalies_series = pd.Series(index=actual_values.index, data=False)
    # Find where predictions are not nan (i.e., where we placed predictions)
    valid_pred_mask = ~np.isnan(predictions_series.values)
    # Place z_scores and anomalies at the same locations
    z_scores_series[valid_pred_mask] = z_scores[:np.sum(valid_pred_mask)]
    anomalies_series[valid_pred_mask] = anomalies[:np.sum(valid_pred_mask)]

    # Plot using matplotlib (for PNG)
    if save_png or show_plot:
        # Set up figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Top plot: Water levels and predictions
        ax1 = axes[0]
        
        # Plot original water levels
        ax1.plot(actual_values.index, actual_values.values, color='blue', linewidth=1, label='Original Water Levels')
        
        # Plot predictions
        ax1.plot(predictions_series.index, predictions_series.values, color='green', linewidth=1, label='Mean Forecast')
        
        # Mark anomalies only where predictions exist
        valid_anomaly_indices = actual_values.index[anomalies_series & valid_pred_mask]
        if len(valid_anomaly_indices) > 0:
            ax1.scatter(valid_anomaly_indices, actual_values.loc[valid_anomaly_indices], 
                       color='red', s=50, marker='o', label='Detected Anomalies')
        
        # Format top plot
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel('Water Level', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Anomaly scores (z-scores)
        ax2 = axes[1]
        
        # Plot the z-scores
        ax2.plot(actual_values.index, np.abs(z_scores_series), color='blue', linewidth=1, label='Absolute Z-Score')
        
        # Add threshold line
        threshold = 5.0  # Default threshold value
        ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        
        # Format bottom plot
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('|Z-Score|', fontsize=14)
        ax2.set_title('Anomaly Z-Scores', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Format date axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the figure if requested
        png_path = None
        if save_png:
            png_path = output_dir / f"water_level_anomalies_{timestamp}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {png_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Create interactive Plotly visualization (for HTML)
    html_path = None
    if save_html:
        # Create figure with 2 subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Water Level Forecasting with Anomalies', 'Anomaly Z-Scores'),
            row_heights=[0.7, 0.3]
        )
        
        # Top plot: Water levels and predictions
        # Original water levels
        fig.add_trace(
            go.Scatter(
                x=actual_values.index,
                y=actual_values.values,
                name="Original Water Levels",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=predictions_series.index,
                y=predictions_series.values,
                name="Predictions",
                line=dict(color='green', width=1)
            ),
            row=1, col=1
        )
        
        # Add anomalies as scatter points only where predictions exist
        valid_anomaly_indices = actual_values.index[anomalies_series & valid_pred_mask]
        if len(valid_anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_anomaly_indices,
                    y=actual_values.loc[valid_anomaly_indices],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='circle'),
                    name=f"Detected Anomalies ({len(valid_anomaly_indices)})"
                ),
                row=1, col=1
            )
        
        # Bottom plot: Z-scores
        fig.add_trace(
            go.Scatter(
                x=actual_values.index,
                y=np.abs(z_scores_series),
                name="Absolute Z-Score",
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        # Add threshold line
        threshold = 5.0  # Default threshold value
        fig.add_trace(
            go.Scatter(
                x=[actual_values.index[0], actual_values.index[-1]],
                y=[threshold, threshold],
                name=f"Threshold ({threshold})",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Water Level",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="|Z-Score|",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            row=2, col=1
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
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
            row=1, col=1
        )
        
        # Save to HTML
        html_path = output_dir / f"water_level_anomalies_{timestamp}.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True)
        print(f"Interactive plot saved as {html_path}")
    
    return png_path, html_path 