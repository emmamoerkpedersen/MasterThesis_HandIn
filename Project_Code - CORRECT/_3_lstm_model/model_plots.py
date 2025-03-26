import os
import webbrowser
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def create_full_plot(test_data, test_predictions, station_id):
    """
    Create an interactive plot with aligned datetime indices.
    """
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']
    
    # Reshape predictions from (sequences, sequence_length, 1) to 1D array
    test_predictions = test_predictions['vst_raw']  # Flatten the predictions
    
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
    
    # Create figure
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(
            x=test_actual.index,
            y=test_actual.values,
            name="Actual",
            line=dict(color='blue', width=1)
        )
    )

    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=predictions_series.index,
            y=predictions_series.values,
            name="Predicted",
            line=dict(color='red', width=1)
        )
    )

    # Update layout
    fig.update_layout(
        title=f'Water Level - Actual vs Predicted (Station {station_id[0]})',
        xaxis_title='Time',
        yaxis_title='Water Level',
        width=1200,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"  # This will format the x-axis as dates
        )
    )

    # Save and open in browser
    html_path = 'predictions_with_dates.html'
    fig.write_html(html_path)
    
    # Open in browser
    absolute_path = os.path.abspath(html_path)
    print(f"Opening plot in browser: {absolute_path}")
    webbrowser.open('file://' + absolute_path)


def plot_scaled_predictions(predictions, targets, title="Scaled Predictions vs Targets"):
        """
        Plot scaled predictions and targets before inverse transformation.
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
        html_path = 'scaled_predictions.html'
        fig.write_html(html_path)
        print(f"Opening scaled predictions plot in browser...")
        webbrowser.open('file://' + os.path.abspath(html_path))


def plot_convergence(history, title="Training and Validation Loss"):
    """
    Plot training and validation loss over epochs to visualize convergence.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
    """
    # Create figure
    fig = go.Figure()
    
    # Get epochs array
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Add training loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_loss'],
            name="Training Loss",
            line=dict(color='blue', width=1)
        )
    )

    # Add validation loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_loss'],
            name="Validation Loss",
            line=dict(color='red', width=1)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        width=1200,
        height=600,
        showlegend=True,
        yaxis_type="log"  # Use log scale for better visualization
    )
    
    # Save and open in browser
    html_path = 'convergence_plot.html'
    fig.write_html(html_path)
    print(f"Opening convergence plot in browser...")
    webbrowser.open('file://' + os.path.abspath(html_path))