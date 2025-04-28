import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse
import json
from tqdm import tqdm
import itertools

# Add necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import our components
from water_forecast_model import WaterLevelForecaster
from visualization import ForecastVisualizer
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

"""
Hyperparameter tuning script for the water level forecaster model.
This script performs a grid search over specified hyperparameters to find the best configuration
for minimizing forecast error and maximizing anomaly detection performance.
"""

# Default base configuration
BASE_CONFIG = {
    # Model parameters
    'hidden_size': 128,         
    'num_layers': 2,            
    'dropout': 0.2,             
    'batch_size': 32,
    'sequence_length': 10,     
    'prediction_window': 24,   
    'sequence_stride': 20,      
    'epochs': 50,               
    'patience': 10,             
    'z_score_threshold': 6.7,   
    'learning_rate': 0.001,     
    
    # Feature engineering
    'use_time_features': True,  
    'use_cumulative_features': True, 
    'use_lagged_features': True,  
    'lag_hours': [1],
    
    # Features to use
    'feature_cols': [
        'rainfall'
    ],
    'output_features': ['vst_raw'],

    # Other stations to use as features
    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['vst_raw', 'rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['vst_raw', 'rainfall']
        }
    ]
}

# Simple error periods for testing across configurations
TUNING_ERROR_PERIODS = [
    {'start': '2022-02-15', 'end': '2022-02-16', 'type': 'offset', 'magnitude': 100},  # Short offset
    {'start': '2022-05-01', 'end': '2022-05-02', 'type': 'scaling', 'magnitude': 1.5},  # Short scaling
    {'start': '2022-06-20', 'end': '2022-06-21', 'type': 'dropout', 'magnitude': -500}  # Short large dropout
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for water level forecasting')
    
    parser.add_argument('--station', type=str, default='21006846',
                        help='Station ID to use')
    
    parser.add_argument('--output_dir', type=str, default='hyperparameter_tuning',
                        help='Directory to save results')
    
    parser.add_argument('--reduced', action='store_true',
                        help='Run with a reduced parameter grid (faster)')
    
    return parser.parse_args()

def get_parameter_grid(reduced=False):
    """
    Define the hyperparameter grid to search over.
    Use the reduced flag for a smaller grid during testing.
    """
    if reduced:
        # Smaller grid for quicker testing
        param_grid = {
            # Model architecture
            'hidden_size': [64, 128],
            'num_layers': [1, 2],
            'dropout': [0.2],
            
            # Training
            'batch_size': [64, 128],  # Increased batch sizes
            'sequence_length': [96, 192],  # 24h and 48h worth of data
            
            # Anomaly detection
            'z_score_threshold': [5.0, 7.0],
            
            # Outlier preprocessing
            'preprocess_outliers': [True, False],
            'outlier_window_size': [12],  # 3 hours worth of data
            'outlier_threshold': [8, 12]
        }
    else:
        # Full grid for comprehensive search
        param_grid = {
            # Model architecture
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            
            # Training
            'batch_size': [64, 96, 128],  # Larger batch sizes
            'sequence_length': [96, 144, 192],  # 24h, 36h, 48h worth of data
            
            # Anomaly detection
            'z_score_threshold': [4.0, 5.5, 7.0, 8.5],
            
            # Outlier preprocessing
            'preprocess_outliers': [True, False],
            'outlier_window_size': [8, 12, 16],  # 2h, 3h, 4h worth of data
            'outlier_threshold': [5, 8, 12, 15]
        }
    
    return param_grid

def evaluate_model(forecaster, val_data, test_data, config, error_periods):
    """
    Evaluate a model configuration on validation and test data with 
    injected errors. Return metrics for comparison.
    """
    metrics = {}
    
    # Check if outlier preprocessing is enabled
    preprocess_outliers = config.get('preprocess_outliers', True)
    
    # Set outlier preprocessing parameters if enabled
    if preprocess_outliers:
        outlier_window_size = config.get('outlier_window_size', 5)
        outlier_threshold = config.get('outlier_threshold', 10)
    
    # 1. Evaluate on clean validation data
    val_results_clean = forecaster.predict(
        val_data, 
        preprocess_outliers=preprocess_outliers
    )
    
    # Calculate MAE on clean validation data
    clean_mae = np.mean(np.abs(
        val_results_clean['clean_data'].values - 
        val_results_clean['forecasts']['step_1'].values
    ))
    
    metrics['clean_val_mae'] = clean_mae
    
    # 2. Evaluate on validation data with injected errors
    val_results_with_errors = forecaster.predict(
        val_data, 
        inject_errors=True,
        error_periods=error_periods,
        preprocess_outliers=preprocess_outliers
    )
    
    # Calculate MAE during normal and error periods
    error_masks = []
    error_maes = []
    error_anomaly_detection_rates = []
    
    # Calculate MAE and anomaly detection rate for each error period
    for i, period in enumerate(error_periods):
        # Get mask for this error period
        period_mask = (val_results_with_errors['clean_data'].index >= period['start']) & \
                     (val_results_with_errors['clean_data'].index <= period['end'])
        
        error_masks.append(period_mask)
        
        # Calculate MAE during this error period
        period_mae = np.mean(np.abs(
            val_results_with_errors['clean_data'][period_mask].values - 
            val_results_with_errors['forecasts']['step_1'][period_mask].values
        ))
        
        error_maes.append(period_mae)
        
        # Calculate anomaly detection rate during this error period
        anomalies = val_results_with_errors['detected_anomalies']
        anomaly_flags = anomalies['is_anomaly']
        
        # Count correct anomaly detections in this period
        period_detections = (anomaly_flags & period_mask).sum() / period_mask.sum() if period_mask.sum() > 0 else 0
        error_anomaly_detection_rates.append(period_detections)
    
    # Combine all error period masks
    all_error_mask = np.logical_or.reduce(error_masks) if error_masks else np.array([False] * len(val_results_with_errors['clean_data']))
    normal_mask = ~all_error_mask
    
    # Calculate MAE for error periods vs normal periods
    if normal_mask.any():
        normal_mae = np.mean(np.abs(
            val_results_with_errors['clean_data'][normal_mask].values - 
            val_results_with_errors['forecasts']['step_1'][normal_mask].values
        ))
        metrics['normal_periods_mae'] = normal_mae
    else:
        metrics['normal_periods_mae'] = None
    
    if all_error_mask.any():
        error_mae = np.mean(np.abs(
            val_results_with_errors['clean_data'][all_error_mask].values - 
            val_results_with_errors['forecasts']['step_1'][all_error_mask].values
        ))
        metrics['error_periods_mae'] = error_mae
    else:
        metrics['error_periods_mae'] = None
    
    # Calculate average MAE across all error periods
    metrics['avg_error_period_mae'] = np.mean(error_maes) if error_maes else None
    
    # Calculate average anomaly detection rate
    metrics['avg_anomaly_detection_rate'] = np.mean(error_anomaly_detection_rates) if error_anomaly_detection_rates else None
    
    # Calculate false positive rate (anomalies detected during normal periods)
    if normal_mask.any():
        false_positives = (val_results_with_errors['detected_anomalies']['is_anomaly'] & normal_mask).sum()
        total_normal = normal_mask.sum()
        metrics['false_positive_rate'] = false_positives / total_normal if total_normal > 0 else 0
    else:
        metrics['false_positive_rate'] = None
    
    # Calculate combined score (lower is better)
    # This score balances prediction accuracy with anomaly detection performance
    if metrics['avg_error_period_mae'] is not None and metrics['avg_anomaly_detection_rate'] is not None:
        # We want high anomaly detection rate and low MAE
        # Invert anomaly detection rate (1 - rate) so lower is better for both
        metrics['combined_score'] = metrics['avg_error_period_mae'] + (1 - metrics['avg_anomaly_detection_rate']) * 100
    else:
        metrics['combined_score'] = float('inf')
    
    return metrics

def run_grid_search(train_data, val_data, test_data, param_grid, output_dir, station_id):
    """
    Run a grid search over the parameter grid and evaluate each configuration.
    """
    # Get all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Calculate total number of combinations
    total_combinations = np.prod([len(values) for values in param_values])
    print(f"Running grid search with {total_combinations} parameter combinations")
    
    # Create list to store results
    results = []
    
    # Try each combination
    for i, values in enumerate(tqdm(itertools.product(*param_values), total=total_combinations)):
        # Create configuration for this run
        config = BASE_CONFIG.copy()
        
        # Update with grid search parameters
        run_params = dict(zip(param_keys, values))
        config.update({k: v for k, v in run_params.items() 
                       if k not in ['preprocess_outliers', 'outlier_window_size', 'outlier_threshold']})
        
        # Extract preprocessing parameters to pass to predict method
        preprocess_outliers = run_params.get('preprocess_outliers', True)
        outlier_window_size = run_params.get('outlier_window_size', 5)
        outlier_threshold = run_params.get('outlier_threshold', 10)
        
        # Initialize model with this configuration
        forecaster = WaterLevelForecaster(config)
        
        # Initialize preprocessor and load data
        preprocessor = DataPreprocessor(config)
        forecaster.preprocessor = preprocessor
        
        # Train the model
        try:
            model = forecaster.train(train_data, val_data, project_dir, station_id)
            
            # Update outlier preprocessing parameters
            forecaster.outlier_window_size = outlier_window_size
            forecaster.outlier_threshold = outlier_threshold
            
            # Evaluate the model
            metrics = evaluate_model(forecaster, val_data, test_data, run_params, TUNING_ERROR_PERIODS)
            
            # Add parameters to metrics
            for param_name, param_value in run_params.items():
                metrics[param_name] = param_value
            
            # Add to results
            results.append(metrics)
            
            # Print current best
            print(f"Run {i+1}/{total_combinations} - Combined Score: {metrics['combined_score']:.2f}")
            
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by combined score (lower is better)
    results_df = results_df.sort_values('combined_score')
    
    # Save results
    results_path = output_dir / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save best configuration
    if len(results_df) > 0:
        best_config = results_df.iloc[0].to_dict()
        best_config_path = output_dir / "best_config.json"
        
        # Remove metrics from config file (keep only hyperparameters)
        metric_keys = ['clean_val_mae', 'normal_periods_mae', 'error_periods_mae', 
                      'avg_error_period_mae', 'avg_anomaly_detection_rate',
                      'false_positive_rate', 'combined_score']
        
        best_params = {k: v for k, v in best_config.items() if k not in metric_keys}
        
        with open(best_config_path, 'w') as f:
            json.dump(best_params, f, indent=2)
    
    return results_df

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup paths
    station_id = args.station
    output_path = project_dir / args.output_dir
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get parameter grid
    param_grid = get_parameter_grid(reduced=args.reduced)
    
    print(f"Project root: {project_dir}")
    print(f"Using station: {station_id}")
    print(f"Output directory: {output_path}")
    
    # Initialize preprocessor with base config
    base_preprocessor = DataPreprocessor(BASE_CONFIG)
    
    # Load data once to avoid reloading for each parameter combination
    print("Loading and splitting data...")
    train_data, val_data, test_data = base_preprocessor.load_and_split_data(project_dir, station_id)
    
    # Run grid search
    results_df = run_grid_search(train_data, val_data, test_data, param_grid, output_path, station_id)
    
    # Print best configuration
    if len(results_df) > 0:
        best_config = results_df.iloc[0]
        print("\nBest Configuration:")
        for param_name in param_grid.keys():
            print(f"  {param_name}: {best_config[param_name]}")
        
        print("\nBest Metrics:")
        print(f"  Combined Score: {best_config['combined_score']:.2f}")
        print(f"  Clean Validation MAE: {best_config['clean_val_mae']:.2f}")
        print(f"  Avg Error Period MAE: {best_config['avg_error_period_mae']:.2f}")
        print(f"  Anomaly Detection Rate: {best_config['avg_anomaly_detection_rate']:.2f}")
        print(f"  False Positive Rate: {best_config['false_positive_rate']:.4f}")
    
    print("\nHyperparameter tuning completed. Results saved to:", output_path)

if __name__ == "__main__":
    main() 