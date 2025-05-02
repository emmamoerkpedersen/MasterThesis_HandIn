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
from iterative_forecast_model import WaterLevelForecaster
from iterative_visualization import ForecastVisualizer
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

# Default configuration
DEFAULT_CONFIG = {
    # Model parameters
    'hidden_size': 14,          # Increased for more model capacity
    'num_layers': 1,            # Using three layers for better modeling
    'dropout': 0.2,             
    'batch_size': 16,          # Reduced batch size for better training
    'sequence_length': 100,  # Extended sequence length to capture more history
    'prediction_window': 10,    # Predict one step ahead
    'sequence_stride': 10,      # Stride for creating sequences
    'epochs': 10,           # Increased for better convergence
    'patience': 5,             # Early stopping patience
    'z_score_threshold': 5,   # Anomaly detection threshold
    'learning_rate': 0.001,     
    
    # Model architecture configuration

    # Feature engineering
    'use_time_features': True,  
    'use_cumulative_features': True, 
    'use_lagged_features': True,  
    'lag_hours': [72, 96, 168, 336, 720, 1440],  # Lag hours (1d, 2d, 4d, 7d, 14d, 30d, 60d)
    
    # Custom features to add - empty by default
    'custom_features': [],
    
    # Features to use
    'feature_cols': [
        'rainfall',
        'temperature',
        'vst_raw_feature'
    ],
    'output_features': ['vst_raw'],

    # Other stations to use as features
    'feature_stations': [
        {
            'station_id': '21006845',
            'features': ['rainfall']
        },
        {
            'station_id': '21006847',
            'features': ['rainfall']
        }
    ],
    
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
    
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'test_error'], 
                        default='test_error', help='Mode to run in')
    
    parser.add_argument('--prediction_mode', type=str, choices=['standard'],
                        default='standard', help='Prediction mode to use')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a trained model from (for predict mode)')
    
    
    # Add new arguments for feature engineering control
    parser.add_argument('--add_ma_features', action='store_true',
                        help='Add moving average features (use --ma_hours to specify)')
    
    parser.add_argument('--ma_hours', type=str, default='6,12,24',
                       help='Comma-separated list of hours for moving average features')
    
    parser.add_argument('--sequence_length', type=int, default=500,
                        help='Sequence length to use for the model (number of timesteps)')
    
                       
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers')
    
    return parser.parse_args()

def train_and_save_model(forecaster, train_data, val_data, project_root, station_id, output_dir):
    """Train a model and save it"""
    print("\nTraining model...")
    model = forecaster.train(train_data, val_data, project_root, station_id)
    
    # Save the trained model
    model_path = output_dir / "trained_model.pth"
    forecaster.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Print feature summary report
    print("\nFeature Usage Report:")
    total_features = len(forecaster.preprocessor.feature_cols)
    print(f"Total features used in model: {total_features}")
    
    # Group features by type
    feature_types = {
        "Base": [],
        "Station": [],
        "Time": [],
        "Cumulative": [],
        "Lag": [],
        "MA": [],
        "ROC": [],
        "Other": []
    }
    
    for feature in forecaster.preprocessor.feature_cols:
        if feature in forecaster.config['feature_cols']:
            feature_types["Base"].append(feature)
        elif feature.startswith('feature_station_'):
            feature_types["Station"].append(feature)
        elif any(time_feat in feature for time_feat in ["month", "day_of_year"]):
            feature_types["Time"].append(feature)
        elif any(cum_feat in feature for cum_feat in ["30day", "180day", "365day"]):
            feature_types["Cumulative"].append(feature)
        elif "lag" in feature:
            feature_types["Lag"].append(feature)
        elif "ma_" in feature and not "roc" in feature:
            feature_types["MA"].append(feature)
        elif "roc" in feature or "acc" in feature:
            feature_types["ROC"].append(feature)
        else:
            feature_types["Other"].append(feature)
    
    # Print summary counts
    for feature_type, features in feature_types.items():
        count = len(features)
        if count > 0:
            percentage = (count / total_features) * 100
            print(f"  - {feature_type} Features: {count} ({percentage:.1f}%)")
    
    # Print top 5 lag features as a sample
    if feature_types["Lag"]:
        print("\nSample Lag Features:")
        for feature in sorted(feature_types["Lag"])[:5]:
            print(f"  - {feature}")
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"  - Hidden Size: {forecaster.config['hidden_size']}")
    print(f"  - Layers: {forecaster.config['num_layers']}")
    print(f"  - Dropout: {forecaster.config['dropout']}")
    print(f"  - Sequence Length: {forecaster.config['sequence_length']}")
    
    return model

def reconstruct_features_with_errors(forecaster, original_data, error_periods):
    """
    Reconstruct all features after injecting errors into raw data.
    This ensures that all derived features are calculated from error-injected data.
    
    Args:
        forecaster: WaterLevelForecaster instance
        original_data: Original DataFrame with clean data
        error_periods: List of error period dictionaries
        
    Returns:
        DataFrame with errors injected and features recalculated
    """
    # Get the output feature (target column)
    output_feature = forecaster.preprocessor.output_features[0] if isinstance(
        forecaster.preprocessor.output_features, list) else forecaster.preprocessor.output_features
    
    # Create a copy of just the essential columns (raw data before feature engineering)
    # First, determine which columns are raw data vs derived features
    raw_columns = ['temperature', 'rainfall', output_feature]
    
    # Add any feature station columns
    for station in forecaster.config['feature_stations']:
        for feature in station['features']:
            raw_columns.append(f"feature_station_{station['station_id']}_{feature}")
    
    # Create a clean base dataframe with only raw columns
    base_df = pd.DataFrame(index=original_data.index)
    for col in raw_columns:
        if col in original_data.columns:
            base_df[col] = original_data[col]
    
    # Now inject errors into the target column
    print(f"Injecting errors into raw data column: {output_feature}")
    for period in error_periods:
        start = pd.Timestamp(period['start'])
        end = pd.Timestamp(period['end'])
        error_type = period['type']
        magnitude = period['magnitude']
        
        # Get mask for the period
        mask = (base_df.index >= start) & (base_df.index <= end)
        
        # Skip if no data points in this period
        if not mask.any():
            continue
            
        # Apply the error to the target column only
        if error_type == 'noise':
            # Generate random noise with the correct shape
            noise = np.random.normal(0, magnitude, size=sum(mask))
            # Apply noise directly
            base_df.loc[mask, output_feature] += noise
        
        elif error_type == 'offset':
            # Apply offset
            base_df.loc[mask, output_feature] += magnitude
        
        elif error_type == 'scaling':
            # Apply scaling
            base_df.loc[mask, output_feature] *= magnitude
        
        elif error_type == 'missing':
            # Set values to NaN (simulating missing data)
            base_df.loc[mask, output_feature] = np.nan
    
    # Now rebuild all features from scratch
    print("Reconstructing all features from error-injected data...")
    
    # Add cumulative features if enabled
    if forecaster.config.get('use_cumulative_features', False):
        base_df = forecaster.preprocessor._add_cumulative_features(base_df)
    
    # Add time features if enabled
    if forecaster.config.get('use_time_features', False):
        base_df = forecaster.preprocessor._add_time_features(base_df)
    
    # Add lagged features if enabled
    if forecaster.config.get('use_lagged_features', False):
        lags = forecaster.config.get('lag_hours', [1, 2, 3, 6, 12, 24])
        
        # Log what feature types are being included/excluded
        print(f"  - Including lag features: {lags}")
        
        base_df = forecaster.preprocessor.feature_engineer.add_lagged_features(
            base_df, 
            target_col=output_feature, 
            lags=lags
        )
    
    # Add any custom features
    if forecaster.config.get('custom_features'):
        print(f"  - Adding custom features")
        base_df = forecaster.preprocessor.feature_engineer.add_custom_features(
            base_df,
            target_col=output_feature,
            feature_specs=forecaster.config['custom_features']
        )
    
    print(f"Feature reconstruction complete. Data shape: {base_df.shape}")
    
    return base_df

def run_validation_with_errors(forecaster, visualizer, val_data, error_periods, output_dir, prediction_mode='standard'):
    """Run and visualize predictions on validation data with injected errors"""
    print("\nPredicting on validation data with injected errors...")
    
    # First get predictions on clean data
    print("Getting predictions on clean data...")
    val_results_clean = forecaster.predict(val_data)
    
    # Get the output feature name
    output_feature = forecaster.preprocessor.output_features[0] if isinstance(
        forecaster.preprocessor.output_features, list) else forecaster.preprocessor.output_features
    
    # Create a version with errors injected and features recalculated
    print("Reconstructing data with errors...")
    val_data_with_errors = reconstruct_features_with_errors(forecaster, val_data, error_periods)
    
    # Make predictions with the reconstructed error-injected data
    print("Getting predictions with injected errors...")
    val_results_with_errors = forecaster.predict(val_data_with_errors)
    
    # Combine results for visualization
    combined_results = {
        'clean_data': val_results_clean['clean_data'],
        'error_injected_data': val_data_with_errors[output_feature],
        'forecasts': val_results_with_errors['forecasts'],
        'clean_forecast': val_results_clean['forecasts'],
        'detected_anomalies': val_results_with_errors['detected_anomalies']
    }
    
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
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
    
    # Create simplified plots without anomalies
    print("\nCreating simplified visualizations (predictions vs actual only)...")
    # Static simplified plot
    visualizer.plot_forecast_simple(
        combined_results,
        title="Water Level Forecast vs Actual (Simplified) - Validation",
        save_path=plots_dir / "water_forecast_validation_simple.png"
    )
    
    # Interactive simplified plot
    visualizer.plot_forecast_simple_plotly(
        combined_results,
        title="Water Level Forecast vs Actual - Interactive (Simplified) - Validation",
        save_path=interactive_dir / "water_forecast_validation_simple.html"
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
    
    # Create simplified plots without anomalies
    print("\nCreating simplified visualizations (predictions vs actual only)...")
    # Static simplified plot
    visualizer.plot_forecast_simple(
        test_results,
        title="Water Level Forecast vs Actual (Simplified)",
        save_path=plots_dir / "water_forecast_simple.png"
    )
    
    # Interactive simplified plot
    visualizer.plot_forecast_simple_plotly(
        test_results,
        title="Water Level Forecast vs Actual - Interactive (Simplified)",
        save_path=interactive_dir / "water_forecast_simple.html"
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


def load_and_filter_data(preprocessor, project_root, station_id):
    """Load data and filter out low importance features"""
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
        
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

def format_config_value(key, value):
    """Format configuration values as descriptive strings for console output"""
    if key.startswith('use_') and isinstance(value, bool):
        return 'Enabled' if value else 'Disabled'
    return str(value)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup paths
    station_id = args.station
    base_output_path = project_dir / args.output_dir
    
    # Create prediction mode-specific output directory
    output_path = base_output_path / f"prediction_mode_{args.prediction_mode}"
    output_path.mkdir(exist_ok=True, parents=True)
    
    
    print(f"Project root: {project_dir}")
    print(f"Using station: {station_id}")
    print(f"Output directory: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Prediction mode: {args.prediction_mode}")
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    
    # Set model architecture and feature engineering parameters
    config.update({
        'sequence_length': args.sequence_length,
        'num_layers': args.num_layers
    })
    
    # Handle custom features
    custom_features = []
    
    # Add MA features if requested
    if args.add_ma_features:
        ma_hours = [int(h) for h in args.ma_hours.split(',')]
        for hours in ma_hours:
            custom_features.append({
                'type': 'ma',
                'params': {'hours': hours}
            })
        print(f"Adding {len(ma_hours)} custom MA features with hours: {ma_hours}")
    
    # Add custom features to config
    if custom_features:
        config['custom_features'] = custom_features
    
    # Print key configuration values
    print("\nConfiguration:")
    for key, value in {
        'sequence_length': config['sequence_length'],
        'num_layers': config['num_layers'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'hidden_size': config['hidden_size']
    }.items():
        print(f"  - {key.replace('_', ' ').title()}: {format_config_value(key, value)}")
    
    # Print custom features if any
    if custom_features:
        print("\nCustom Features:")
        for feature in custom_features:
            print(f"  - {feature['type'].upper()} feature with params: {feature['params']}")
    
    # Initialize forecasting model and visualizer
    forecaster = WaterLevelForecaster(config)
    visualizer = ForecastVisualizer(config)
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(config)
    forecaster.preprocessor = preprocessor
    
    print("Loading and filtering data...")
    train_data, val_data, test_data = load_and_filter_data(preprocessor, project_dir, station_id)
    
    # Run in the specified mode
    if args.mode == 'train':
        # Train mode - train and save model, then test on validation data
        train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
        # Clean validation run
        print("\nPredicting on clean validation data...")
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

        
        run_validation_with_errors(forecaster, visualizer, val_data, DEFAULT_ERROR_PERIODS, val_dir, args.prediction_mode)
        
        print(f"\n=== Test Data Predictions ({args.prediction_mode} mode) ===")
        # Run predictions on test data
        run_test_predictions(forecaster, visualizer, test_data, test_dir)
    
    
    print("\nDone!")

if __name__ == "__main__":
    main() 

'''
Arg options 

python run_forecast.py --station 21006846 --output_dir forecast_results --mode predict --model_path trained_model.pth
cd "Project_Code - CORRECT"; python experiments/forecaster/run_forecast.py --mode test_error --prediction_mode standard
'''