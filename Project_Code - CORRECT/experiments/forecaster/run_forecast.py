import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse

# Add necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import our components
from water_forecast_model import WaterLevelForecaster
from visualization import ForecastVisualizer
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

# Default configuration
DEFAULT_CONFIG = {
    # Model parameters
    'hidden_size': 64,          # Reduced from 128 to save memory
    'num_layers': 2,            
    'dropout': 0.2,             
    'batch_size': 96,          # Reduced from 64 to save memory
    'sequence_length': 400,  # 5 days of data (increased from 396)
    'prediction_window': 1,    # Predict up to 12 steps ahead
    'sequence_stride':15,      # Stride for creating sequences
    'epochs': 100,           # Maximum epochs
    'patience': 10,             # Early stopping patience
    'z_score_threshold': 6.7,   # Anomaly detection threshold
    'learning_rate': 0.001,     
    
    # Model features
    'use_feature_importance': False,  # Disable feature importance calculation by default
    
    # Feature engineering
    'use_time_features': True,  
    'use_cumulative_features': True, 
    'use_lagged_features': True,  
    'lag_hours': [2, 6, 12, 24, 48, 72],  # Added 6h lag for better intermediate context
    
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
    ],
    
    # Low importance features to filter out (importance < 0.01)
    'low_importance_features': [
        # Very short-term features
        'water_level_roc_15min',
        'water_level_roc_30min',
        'water_level_acc_15min',
        'water_level_acc_1h',
        
        # Redundant lag features
        'water_level_lag_12h',
        'water_level_lag_24h',
        
        # Time features
        'month_sin',
        'month_cos',
        'day_of_year_sin',
        'day_of_year_cos',
        
        # Long-term rainfall features
        'rainfall_180day',
        'feature1_rain_180day',
        'feature2_rain_180day',
        'feature1_rain_365day',
        
        # Low importance station features
        'feature_station_21006847_vst_raw',
        
        # Redundant MA features
        'water_level_ma_18h',
        'water_level_ma_36h',
        'water_level_ma_48h'
    ]
}

# Default error periods for testing
DEFAULT_ERROR_PERIODS = [
    # Basic error types over longer periods
    {
        'start': '2022-02-15', 'end': '2022-03-01', 
        'type': 'offset', 'magnitude': 100,  # Add constant value
        'description': 'Long period constant offset'
    },
    {
        'start': '2022-05-01', 'end': '2022-05-15', 
        'type': 'scaling', 'magnitude': 1.5,  # Multiply by factor
        'description': 'Long period scaling factor'
    },
    {
        'start': '2022-09-01', 'end': '2022-09-15', 
        'type': 'noise', 'magnitude': 50,  # Random noise amplitude
        'description': 'Long period random noise'
    },
    {
        'start': '2022-11-01', 'end': '2022-11-15', 
        'type': 'missing', 'magnitude': 0,  # Data completely missing (NaN)
        'description': 'Long period of missing data'
    },
    
    # Short, intense error periods
    {
        'start': '2022-01-15', 'end': '2022-01-15 23:59:59', 
        'type': 'offset', 'magnitude': 500,  # Large sudden spike
        'description': 'Single day large spike'
    },
    {
        'start': '2022-06-20', 'end': '2022-06-22', 
        'type': 'offset', 'magnitude': -500,  # Large sudden drop
        'description': 'Short period large drop'
    }
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run water level forecasting and anomaly detection')
    
    parser.add_argument('--station', type=str, default='21006846',
                        help='Station ID to use')
    
    parser.add_argument('--output_dir', type=str, default='forecast_results',
                        help='Directory to save results')
    
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'test_error', 'multi_horizon'], 
                        default='test_error', help='Mode to run in')
    
    parser.add_argument('--prediction_mode', type=str, choices=['standard', 'iterative'],
                        default='standard', help='Prediction mode to use')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a trained model from (for predict mode)')
    
    parser.add_argument('--horizons', type=str, default='1,12,24,48,72',
                        help='Comma-separated list of forecast horizons to evaluate')
    
    return parser.parse_args()

def train_and_save_model(forecaster, train_data, val_data, project_root, station_id, output_dir):
    """Train a model and save it"""
    print("\nTraining model...")
    model = forecaster.train(train_data, val_data, project_root, station_id)
    
    # Save the trained model
    model_path = output_dir / "trained_model.pth"
    forecaster.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def run_validation_with_errors(forecaster, visualizer, val_data, error_periods, output_dir, prediction_mode='standard'):
    """Run and visualize predictions on validation data with injected errors"""
    print("\nPredicting on validation data with injected errors...")
    
    # First get predictions on clean data
    print("Getting predictions on clean data...")
    if prediction_mode == 'iterative':
        val_results_clean = forecaster.predict_iteratively(val_data)
    else:
        val_results_clean = forecaster.predict(val_data)
    
    # Then get predictions with injected errors
    print("Getting predictions with injected errors...")
    val_data_with_errors = forecaster._inject_errors(val_data.copy(), error_periods)
    if prediction_mode == 'iterative':
        val_results_with_errors = forecaster.predict_iteratively(val_data_with_errors)
    else:
        val_results_with_errors = forecaster.predict(val_data_with_errors)
    
    # Combine results for visualization
    combined_results = {
        'clean_data': val_results_clean['clean_data'],
        'error_injected_data': val_data_with_errors[forecaster.preprocessor.output_features],
        'forecasts': val_results_with_errors['forecasts'],
        'clean_forecast': val_results_clean['forecasts'],
        'detected_anomalies': val_results_with_errors['detected_anomalies']
    }
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(exist_ok=True)
    
    # Create feature importance visualization if enabled and available
    if forecaster.config.get('use_feature_importance', False) and 'feature_importance' in val_results_with_errors:
        feature_importance = val_results_with_errors['feature_importance']
        if feature_importance is not None:  # Additional check to ensure we have data
            print("\nCreating feature importance visualization...")
            
            # Handle different feature importance formats
            if isinstance(feature_importance, list):
                # For iterative predictions, use the last iteration
                if isinstance(feature_importance[-1], dict) and 'importances' in feature_importance[-1]:
                    feature_importance = feature_importance[-1]['importances']
                else:
                    # If it's a list of tuples, use as is
                    feature_importance = feature_importance
            elif isinstance(feature_importance, dict):
                # If it's already a dictionary, use as is
                feature_importance = feature_importance
            elif isinstance(feature_importance, tuple):
                # If it's a tuple of (feature, importance), convert to dict
                feature_importance = dict(zip(forecaster.preprocessor.feature_cols, feature_importance))
                
            visualizer.plot_feature_importance_analysis(
                feature_importance,
                title="Feature Importance Analysis",
                save_path=plots_dir / "feature_importance_analysis.png",
                min_importance_threshold=0.01  # Features below this might be candidates for removal
            )
    
    # Plot validation results
    visualizer.plot_forecast_with_anomalies(
        combined_results,
        title="Water Level Forecasting with Injected Errors",
        save_path=plots_dir / "water_forecast_with_errors.png"
    )
    
    # Interactive Plotly plot
    visualizer.plot_forecast_with_anomalies_plotly(
        combined_results,
        title="Water Level Forecasting with Injected Errors - Interactive",
        save_path=interactive_dir / "water_forecast_with_errors.html"
    )
    
    # Plot focused views for each error period
    print("\nCreating focused error period plots...")
    for i, period in enumerate(error_periods):
        # Static matplotlib plot
        visualizer.plot_error_impact(
            combined_results,
            period,
            save_path=plots_dir / f"error_impact_{period['type']}_{i+1}.png"
        )
        
        # Interactive Plotly plot
        visualizer.plot_error_impact_plotly(
            combined_results,
            period,
            save_path=interactive_dir / f"error_impact_{period['type']}_{i+1}.html"
        )
    
    # Summarize anomalies
    anomalies = combined_results['detected_anomalies']
    anomaly_count = anomalies['is_anomaly'].sum()
    total_points = len(anomalies)
    
    print(f"\nAnomaly Detection Summary (With Errors Injected):")
    print(f"Total data points: {total_points}")
    print(f"Anomalies detected: {anomaly_count} ({anomaly_count/total_points*100:.2f}%)")
    
    if 'anomaly_type' in anomalies.columns:
        # Count by type
        type_counts = anomalies[anomalies['is_anomaly']]['anomaly_type'].value_counts()
        for anomaly_type, count in type_counts.items():
            print(f"  - {anomaly_type.capitalize()} anomalies: {count} ({count/anomaly_count*100:.2f}%)")
    
    return combined_results

def run_test_predictions(forecaster, visualizer, test_data, output_dir):
    """Run and visualize predictions on test data"""
    print("\nPredicting on test data...")
    test_results = forecaster.predict(test_data)
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(exist_ok=True)
    
    # Plot test results
    print("\nPlotting test results...")
    # Static matplotlib plot
    visualizer.plot_forecast_with_anomalies(
        test_results,
        title="Water Level Forecasting for Test Data",
        save_path=plots_dir / "water_forecast_test.png"
    )
    
    # Interactive Plotly plot
    print("\nCreating interactive visualizations...")
    visualizer.plot_forecast_with_anomalies_plotly(
        test_results,
        title="Water Level Forecasting for Test Data - Interactive",
        save_path=interactive_dir / "water_forecast_test.html"
    )
    
    # Summarize anomalies in test data
    test_anomalies = test_results['detected_anomalies'] 
    test_anomaly_count = test_anomalies['is_anomaly'].sum()
    test_total_points = len(test_anomalies)
    
    print(f"\nTest Data Anomaly Detection Summary:")
    print(f"Total test data points: {test_total_points}")
    print(f"Anomalies detected in test data: {test_anomaly_count} ({test_anomaly_count/test_total_points*100:.2f}%)")
    
    if 'anomaly_type' in test_anomalies.columns:
        # Count by type
        type_counts = test_anomalies[test_anomalies['is_anomaly']]['anomaly_type'].value_counts()
        for anomaly_type, count in type_counts.items():
            print(f"  - {anomaly_type.capitalize()} anomalies: {count} ({count/test_anomaly_count*100:.2f}%)")
    
    return test_results

def run_multi_horizon_analysis(forecaster, visualizer, test_data, horizons, output_dir):
    """Run and visualize predictions with multiple forecast horizons"""
    print(f"\nRunning multi-horizon analysis with horizons: {horizons}")
    
    # Ensure config has enough output steps
    max_horizon = max(horizons)
    if forecaster.prediction_window < max_horizon:
        print(f"Adjusting prediction window from {forecaster.prediction_window} to {max_horizon}")
        forecaster.prediction_window = max_horizon
        forecaster.config['prediction_window'] = max_horizon
    
    # Make predictions on test data
    test_results = forecaster.predict(test_data)
    
    # Create multi-horizon directory
    horizon_dir = output_dir / "multi_horizon"
    horizon_dir.mkdir(exist_ok=True)
    
    # Plot multi-horizon forecast
    print("\nPlotting multi-horizon forecast...")
    visualizer.plot_multi_horizon_forecasts(
        test_results,
        horizons=horizons,
        title="Multi-Horizon Water Level Forecasting",
        save_path=horizon_dir / "multi_horizon_forecast.png"
    )
    
    # Plot forecast accuracy by horizon
    print("\nPlotting forecast accuracy by horizon...")
    visualizer.plot_forecast_accuracy(
        test_results,
        horizons=horizons,
        save_path=horizon_dir / "forecast_accuracy.png"
    )
    
    # Calculate metrics for anomalies
    detected_anomalies = test_results['detected_anomalies']
    anomaly_indices = detected_anomalies[detected_anomalies['is_anomaly']].index
    
    # Calculate metrics for each horizon
    metrics = []
    
    anomaly_mask = detected_anomalies['is_anomaly']
    normal_mask = ~anomaly_mask
    
    for horizon in horizons:
        horizon_col = f'step_{horizon}'
        if horizon_col in test_results['forecasts'].columns:
            horizon_metrics = {
                'Horizon': horizon,
                'Overall_MAE': None,
                'Normal_MAE': None,
                'Anomaly_MAE': None,
                'MAE_Difference': None,
                'MAE_Impact_Percentage': None
            }
            
            # Calculate overall MAE
            horizon_metrics['Overall_MAE'] = np.mean(np.abs(
                test_results['clean_data'].values - test_results['forecasts'][horizon_col].values
            ))
            
            # Calculate MAE during normal periods
            if normal_mask.any():
                horizon_metrics['Normal_MAE'] = np.mean(np.abs(
                    test_results['clean_data'].loc[normal_mask].values - 
                    test_results['forecasts'].loc[normal_mask, horizon_col].values
                ))
            
            # Calculate MAE during anomalies
            if anomaly_mask.any():
                horizon_metrics['Anomaly_MAE'] = np.mean(np.abs(
                    test_results['clean_data'].loc[anomaly_mask].values - 
                    test_results['forecasts'].loc[anomaly_mask, horizon_col].values
                ))
            
            # Calculate difference and impact percentage
            if horizon_metrics['Normal_MAE'] is not None and horizon_metrics['Anomaly_MAE'] is not None:
                horizon_metrics['MAE_Difference'] = horizon_metrics['Anomaly_MAE'] - horizon_metrics['Normal_MAE']
                horizon_metrics['MAE_Impact_Percentage'] = (
                    horizon_metrics['MAE_Difference'] / horizon_metrics['Normal_MAE'] * 100
                )
            
            metrics.append(horizon_metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics to CSV
    metrics_df.to_csv(horizon_dir / "horizon_metrics.csv", index=False)
    
    # Plot metrics
    visualizer.plot_horizon_metrics(
        metrics_df,
        save_path=horizon_dir / "horizon_metrics_comparison.png"
    )
    
    print("\nHorizon Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    return test_results, metrics_df

def load_and_filter_data(preprocessor, project_root, station_id, remove_low_importance_features=False):
    """Load data and filter out low importance features"""
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Get list of features to remove
    features_to_remove = DEFAULT_CONFIG['low_importance_features']
    
    # Remove low importance features
    if remove_low_importance_features:
        for feature in features_to_remove:
            if feature in train_data.columns:
                print(f"Removing low importance feature: {feature}")
                train_data = train_data.drop(columns=[feature])
                val_data = val_data.drop(columns=[feature])
                test_data = test_data.drop(columns=[feature])
    
        # Update feature columns in preprocessor
        preprocessor.feature_cols = [col for col in preprocessor.feature_cols if col not in features_to_remove]
        
    # Reinitialize the feature scaler with updated columns
    preprocessor.feature_scaler = preprocessor.feature_scaler.__class__(
        feature_cols=preprocessor.feature_cols,
        output_features=preprocessor.output_features,
        device=preprocessor.device
    )
    
    print(f"\nRemaining features ({len(preprocessor.feature_cols)}):")
    for feature in preprocessor.feature_cols:
        print(f"  - {feature}")
    
    return train_data, val_data, test_data

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup paths
    station_id = args.station
    base_output_path = project_dir / args.output_dir
    
    # Create prediction mode-specific output directory
    output_path = base_output_path / f"prediction_mode_{args.prediction_mode}"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(',')]
    
    print(f"Project root: {project_dir}")
    print(f"Using station: {station_id}")
    print(f"Output directory: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Prediction mode: {args.prediction_mode}")
    
    # Initialize forecasting model and visualizer
    config = DEFAULT_CONFIG.copy()
    
    # Update config to accommodate the largest horizon
    config['prediction_window'] = max(horizons)
    
    forecaster = WaterLevelForecaster(config)
    visualizer = ForecastVisualizer(config)
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(config)
    forecaster.preprocessor = preprocessor
    
    print("Loading and filtering data...")
    train_data, val_data, test_data = load_and_filter_data(preprocessor, project_dir, station_id)
    
    #Print the features and model config used
    print(f"Features used: {preprocessor.feature_cols}")
    print(f"Model config: {config}")
    
    # Run in the specified mode
    if args.mode == 'train':
        # Train mode - train and save model, then test on validation data
        train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
        # Clean validation run
        print("\nPredicting on clean validation data...")
        if args.prediction_mode == 'iterative':
            val_results_clean = forecaster.predict_iteratively(val_data)
        else:
            val_results_clean = forecaster.predict(val_data)
        
        visualizer.plot_forecast_with_anomalies(
            val_results_clean, 
            title=f"Water Level Forecasting with Clean Validation Data ({args.prediction_mode} mode)",
            save_path=output_path / "water_forecast_clean_val.png"
        )
        
        # Also test on test data
        run_test_predictions(forecaster, visualizer, test_data, output_path)
        
    elif args.mode == 'predict':
        # Prediction mode - load model and run on test data
        if args.model_path:
            model_loaded = forecaster.load_model(args.model_path)
            if not model_loaded:
                print(f"Could not load model from {args.model_path}. Training new model.")
                train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        else:
            print("No model path provided. Training new model.")
            train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
        # Run predictions on test data
        run_test_predictions(forecaster, visualizer, test_data, output_path)
        
    elif args.mode == 'test_error':
        # Error testing mode - train model and run with injected errors
        train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
        # Create separate directories for validation and test results
        val_dir = output_path / "validation_with_errors"
        test_dir = output_path / "test_predictions"
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Validation Data with Injected Errors ({args.prediction_mode} mode) ===")
        # Run with injected errors on validation data
        run_validation_with_errors(forecaster, visualizer, val_data, DEFAULT_ERROR_PERIODS, val_dir, args.prediction_mode)
        
        print(f"\n=== Test Data Predictions ({args.prediction_mode} mode) ===")
        # Run predictions on test data
        run_test_predictions(forecaster, visualizer, test_data, test_dir)
        
    elif args.mode == 'multi_horizon':
        # Multi-horizon mode - train and test with multiple forecast horizons
        train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
        # Run multi-horizon analysis
        run_multi_horizon_analysis(forecaster, visualizer, test_data, horizons, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 

'''
Arg options 

python run_forecast.py --station 21006846 --output_dir forecast_results --mode predict --model_path trained_model.pth
cd "Project_Code - CORRECT"; python experiments/forecaster/run_forecast.py --mode test_error --prediction_mode standard
'''