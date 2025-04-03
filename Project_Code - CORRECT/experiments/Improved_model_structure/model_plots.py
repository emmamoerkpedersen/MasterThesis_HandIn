import os
import webbrowser
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_full_plot(test_data, test_predictions, station_id, model_config=None):
    """
    Create an interactive plot with aligned datetime indices, rainfall data, and model configuration.
    
    Args:
        test_data: DataFrame containing test data with datetime index
        test_predictions: DataFrame or Series containing predictions
        station_id: ID of the station being analyzed
        model_config: Optional dictionary containing model configuration parameters
    """
    # Convert station_id to string if it's not already
    station_id = str(station_id)
    
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']
    
    # Get rainfall data
    rainfall_data = None
    if 'rainfall' in test_data.columns:
        # Resample rainfall to hourly sums for better visibility
        rainfall_data = test_data['rainfall'].resample('1H').sum()
        
        # Replace -1 values with 0 for plotting only (keep original data for model)
        # This is just for visualization, not changing the actual model data
        plotting_rainfall = rainfall_data.copy()
        #plotting_rainfall[plotting_rainfall < 0] = 0
    
    # Resample water level data to hourly means for better performance
    test_actual = test_actual.resample('1H').mean()
    test_predictions = test_predictions['vst_raw'].resample('1H').mean()
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Trim the actual data to match predictions
    if len(test_predictions) > len(test_actual):
        print("Trimming predictions to match actual data length")
        test_predictions = test_predictions[:len(test_actual)]
    else:
        print("Using full predictions")
    
    # Create a pandas Series for predictions with the matching datetime index
    predictions_series = pd.Series(
        data=test_predictions,
        index=test_actual.index[:len(test_predictions)],
        name='Predictions'
    )
    
    # Print final shapes for verification
    print(f"Final test data shape: {test_actual.shape}")
    print(f"Final predictions shape: {predictions_series.shape}")
    
    # Create subplots - rainfall on top, water level on bottom
    subplot_rows = 2 if rainfall_data is not None else 1
    
    if rainfall_data is not None:
        # Create specs for subplots - must be a 2D list
        specs = [[{"secondary_y": False}] for _ in range(subplot_rows)]
        
        # Create subplot titles
        subplot_titles = [
            'Rainfall',
            f'Water Level - Actual vs Predicted (Station {station_id})'
        ]
        
        fig = make_subplots(
            rows=subplot_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,  # Reduced spacing between subplots
            subplot_titles=subplot_titles,
            specs=specs,
            row_heights=[0.2, 0.8]  # Rainfall gets less space than water level
        )
        
        # Add rainfall data to first subplot (top)
        fig.add_trace(
            go.Bar(
                x=plotting_rainfall.index,
                y=plotting_rainfall.values,
                name="Rainfall",
                marker_color='rgba(65, 105, 225, 0.7)',  # Royal blue with transparency
                opacity=0.8,
                width=60*60*1000*10,  # Adjust bar width for hourly data (80% of hour in milliseconds)
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
                line=dict(color='#1f77b4', width=1.5)  # Blue line, slightly thicker
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions_series.index,
                y=predictions_series.values,
                name="Predicted",
                line=dict(color='#d62728', width=1.5)  # Red line, slightly thicker
            ),
            row=2, col=1
        )
        
        # Calculate error band
        error = np.abs(test_actual.values - predictions_series.values)
        upper_bound = test_actual.values + error
        lower_bound = test_actual.values - error
        
        # Add transparent error band
        fig.add_trace(
            go.Scatter(
                x=test_actual.index,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_actual.index,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor='rgba(231, 107, 243, 0.15)',
                name='Error Band',
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Update y-axes labels and ranges
        fig.update_yaxes(
            title_text="Rainfall (mm)",
            row=1, col=1,
            autorange=True,  # Normal orientation for rainfall
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            rangemode='nonnegative'  # Ensures y-axis starts at 0 or higher
        )
        
        fig.update_yaxes(
            title_text="Water Level (mm)",
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)'
        )
        
        # Update x-axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            rangeslider_visible=False,  # Add range slider for bottom plot only
            row=1, col=1
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            rangeslider_visible=True,
            row=2, col=1
        )
        
    else:
        # Just create a single plot for water level
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=test_actual.index,
                y=test_actual.values,
                name="Actual",
                line=dict(color='#1f77b4', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions_series.index,
                y=predictions_series.values,
                name="Predicted",
                line=dict(color='#d62728', width=1.5)
            )
        )
        
        # Calculate error band
        error = np.abs(test_actual.values - predictions_series.values)
        upper_bound = test_actual.values + error
        lower_bound = test_actual.values - error
        
        # Add transparent error band
        fig.add_trace(
            go.Scatter(
                x=test_actual.index,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_actual.index,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor='rgba(231, 107, 243, 0.15)',
                name='Error Band'
            )
        )
        
        # Update axes
        fig.update_yaxes(
            title_text="Water Level (mm)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            rangeslider_visible=True
        )
    
    # Create model configuration text
    config_text = ""
    if model_config:
        config_text = (
            f"<b>Model Configuration:</b><br>"
            f"Hidden Size: {model_config.get('hidden_size', 'N/A')}<br>"
            f"Num Layers: {model_config.get('num_layers', 'N/A')}<br>"
            f"Learning Rate: {model_config.get('learning_rate', 0.001)}<br>"
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
            'font': {'size': 24, 'color': '#111111', 'family': 'Arial, sans-serif'}
        },
        width=1200,
        height=800 if rainfall_data is not None else 600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        ),
        plot_bgcolor='rgb(250, 250, 255)',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=100, b=80),
        annotations=[
            dict(
                x=0.01, 
                y=0.98, 
                xref="paper", 
                yref="paper",
                text=config_text,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                borderpad=6
            )
        ] if config_text else [],
        hovermode='x unified'
    )
    
    # Add range selector to bottom x-axis
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='rgba(150, 200, 250, 0.4)',
            activecolor='rgba(100, 150, 200, 0.8)'
        ),
        row=subplot_rows, col=1  # Only add to bottom subplot
    )
    
    # Save and open in browser
    html_path = f'predictions_station_{station_id}.html'
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True, config={
        'displayModeBar': True,
        'responsive': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'station_{station_id}_prediction',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    })
    
    # Open in browser
    absolute_path = os.path.abspath(html_path)
    print(f"Opening plot in browser: {absolute_path}")
    webbrowser.open('file://' + absolute_path)


def plot_scaled_predictions(predictions, targets, station_id=None, title="Scaled Predictions vs Targets"):
        """
        Plot scaled predictions and targets before inverse transformation.
        
        Args:
            predictions: Numpy array of scaled predictions
            targets: Numpy array of scaled targets
            station_id: Optional station ID for filename
            title: Plot title
        """
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
        html_path = f'scaled_predictions{station_suffix}.html'
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
    plt.savefig(f'convergence_plot_{station_id}.png')
    plt.close()