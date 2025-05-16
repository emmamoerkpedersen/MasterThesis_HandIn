"""
Utility functions for the water level prediction pipeline.
This module contains helper functions extracted from main.py to make it cleaner.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to calculate NSE (Nash-Sutcliffe Efficiency)
def calculate_nse(observed, predicted):
    """Calculate Nash-Sutcliffe Efficiency coefficient."""
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

def prepare_prediction_dataframe(predictions, data_index, data_length, flatten=True):
    """
    Prepare a DataFrame with predictions properly aligned with the data index.
    
    Args:
        predictions: The prediction array
        data_index: The index from the original data
        data_length: Length of the original data
        flatten: Whether to flatten the predictions array
        
    Returns:
        DataFrame with aligned predictions
    """
    # Ensure predictions are in the right shape
    if flatten:
        predictions_reshaped = predictions.flatten() 
    else:
        predictions_reshaped = predictions
    
    # Ensure predictions match data length
    if len(predictions_reshaped) > data_length:
        # If predictions are longer, truncate to match data length
        predictions_reshaped = predictions_reshaped[:data_length]
    elif len(predictions_reshaped) < data_length:
        # If predictions are shorter, pad with NaN values
        padding = np.full(data_length - len(predictions_reshaped), np.nan)
        predictions_reshaped = np.concatenate([predictions_reshaped, padding])
        if flatten:
            predictions_reshaped = predictions_reshaped.flatten() # Ensure 1D after padding

    # Create DataFrame with aligned predictions
    return pd.DataFrame(
        predictions_reshaped,
        index=data_index,
        columns=['vst_raw']
    )

def calculate_performance_metrics(actual_data, predictions, valid_mask=None):
    """
    Calculate performance metrics for model predictions.
    
    Args:
        actual_data: Series or array containing actual values
        predictions: Series or array containing predicted values
        valid_mask: Boolean mask for valid data points (optional)
        
    Returns:
        Dictionary with performance metrics
    """
    # Ensure actual_data and predictions are numpy arrays
    actual_data = np.array(actual_data)
    predictions = np.array(predictions)
    
    if valid_mask is None:
        # Create mask for non-NaN values in both actual and predictions
        actual_nan_mask = ~np.isnan(actual_data)
        predictions_nan_mask = ~np.isnan(predictions)
        valid_mask = actual_nan_mask & predictions_nan_mask
    
    # Handle dimensionality - ensure arrays are 1D and of the same shape
    # First check if arrays are same shape
    if actual_data.shape != predictions.shape:
        # If not, try to match them
        min_length = min(len(actual_data), len(predictions))
        actual_data = actual_data[:min_length]
        predictions = predictions[:min_length]
        valid_mask = valid_mask[:min_length]
    
    # Make sure everything is 1D
    actual_data = actual_data.flatten()
    predictions = predictions.flatten()
    valid_mask = valid_mask.flatten()
    
    # Ensure we have enough valid data points
    if np.sum(valid_mask) < 2:
        print("Warning: Not enough valid data points for metric calculation")
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'nse': np.nan
        }
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(
        actual_data[valid_mask], 
        predictions[valid_mask]
    ))
    mae = mean_absolute_error(
        actual_data[valid_mask], 
        predictions[valid_mask]
    )
    nse = calculate_nse(
        actual_data[valid_mask], 
        predictions[valid_mask]
    )
    
    return {
        'rmse': rmse,
        'mae': mae,
        'nse': nse
    }

def save_comparison_metrics(output_path, error_frequency, clean_metrics, error_metrics):
    """
    Save comparison metrics between clean and error-injected models.
    
    Args:
        output_path: Path to save the metrics
        error_frequency: Frequency of injected errors
        clean_metrics: Dictionary with clean model metrics
        error_metrics: Dictionary with error model metrics
        
    Returns:
        Path to the saved metrics file
    """
    # Create dataframe with metrics
    metrics_df = pd.DataFrame({
        'error_frequency': [error_frequency],
        'clean_val_loss': [clean_metrics.get('val_loss', np.nan)],
        'error_val_loss': [error_metrics.get('val_loss', np.nan)],
        'clean_rmse': [clean_metrics.get('rmse', np.nan)],
        'error_rmse': [error_metrics.get('rmse', np.nan)],
        'clean_mae': [clean_metrics.get('mae', np.nan)],
        'error_mae': [error_metrics.get('mae', np.nan)],
        'clean_nse': [clean_metrics.get('nse', np.nan)],
        'error_nse': [error_metrics.get('nse', np.nan)]
    })

    # Include timestamp in the metrics filename to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = Path(output_path) / f"error_comparison_metrics_{timestamp}.csv"

    # Also save to the standard filename for the visualization code to find
    standard_metrics_file = Path(output_path) / "error_comparison_metrics.csv"

    # Save to timestamped file
    metrics_df.to_csv(metrics_file, index=False)
    
    # If standard file exists, append to it, otherwise create new file
    if standard_metrics_file.exists():
        existing_df = pd.read_csv(standard_metrics_file)
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        combined_df.to_csv(standard_metrics_file, index=False)
    else:
        metrics_df.to_csv(standard_metrics_file, index=False)

    return metrics_file, standard_metrics_file

def print_metrics_table(metrics, title="Model Performance Metrics"):
    """
    Print a formatted table of model metrics.
    
    Args:
        metrics: Dictionary with model metrics
        title: Title for the metrics table
    """
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Value':<15}")
    print("-" * 50)
    
    for metric_name, metric_value in metrics.items():
        if metric_name == 'val_loss':
            print(f"{'Val Loss':<15} {metric_value:<15.6f}")
        elif metric_name in ['rmse', 'mae']:
            print(f"{metric_name.upper():<15} {metric_value:<15.2f}")
        else:
            print(f"{metric_name.upper():<15} {metric_value:<15.4f}")

def print_comparison_table(clean_metrics, error_metrics, error_frequency):
    """
    Print a comparison table between clean and error model metrics.
    
    Args:
        clean_metrics: Dictionary with clean model metrics
        error_metrics: Dictionary with error model metrics
        error_frequency: Frequency of injected errors
    """
    print("\nPerformance Metrics Comparison:")
    print("-" * 80)
    print(f"Error Frequency: {error_frequency * 100:.1f}%")
    print("-" * 80)
    print(f"{'Metric':<15} {'Clean-Trained':<15} {'Error-Trained':<15} {'Difference':<15} {'% Change':<15}")
    print("-" * 80)
    
    # Calculate difference and percent change for each metric
    metrics_to_compare = [
        ('val_loss', 'Val Loss', '{:<15.6f}'),
        ('rmse', 'RMSE', '{:<15.2f}'),
        ('mae', 'MAE', '{:<15.2f}'),
        ('nse', 'NSE', '{:<15.4f}')
    ]
    
    for metric_key, metric_name, format_str in metrics_to_compare:
        clean_value = clean_metrics.get(metric_key, np.nan)
        error_value = error_metrics.get(metric_key, np.nan)
        
        # Calculate difference
        diff = error_value - clean_value
        
        # Calculate percent change
        if not np.isnan(clean_value) and clean_value != 0:
            pct_change = (diff / clean_value) * 100
        else:
            pct_change = float('inf')
        
        # Print row
        print(f"{metric_name:<15} {format_str.format(clean_value)} {format_str.format(error_value)} {format_str.format(diff)} {pct_change:<15.2f}")

def prepare_features_df(test_data):
    """
    Prepare a features DataFrame for residual analysis.
    
    Args:
        test_data: DataFrame containing test data
        
    Returns:
        DataFrame with features for residual analysis
    """
    # Create features DataFrame
    features_df = pd.DataFrame(index=test_data.index)
    features_df['vst_raw'] = test_data['vst_raw']
    
    # Add rainfall if available
    if 'rainfall' in test_data.columns:
        features_df['rainfall'] = test_data['rainfall']
    
    # Add temperature if available
    if 'temperature' in test_data.columns:
        features_df['temperature'] = test_data['temperature']
    
    # Check for any temperature-related columns if direct 'temperature' is not available
    elif any('temperature' in col.lower() for col in test_data.columns):
        temp_cols = [col for col in test_data.columns if 'temperature' in col.lower()]
        # Use the first temperature column found
        if temp_cols:
            features_df['temperature'] = test_data[temp_cols[0]]
            print(f"Using {temp_cols[0]} for temperature analysis")
    
    return features_df 