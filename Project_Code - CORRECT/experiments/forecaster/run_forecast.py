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
    'hidden_size': 128,           # Reduced for faster training
    'num_layers': 3,              # Using three layers for better modeling
    'dropout': 0.2,               # Reduced dropout for simpler training
    'batch_size': 64,             # Batch size for training
    'sequence_length': 700,       # Increased sequence length for better context
    'prediction_window': 24,      # Predict 24 steps ahead (6 hours)
    'sequence_stride': 25,        # Stride for creating sequences
    'epochs': 20,                 # Moderate number of epochs
    'patience': 7,                # Early stopping patience
    'z_score_threshold': 2.0,     # Base threshold for dynamic anomaly detection
    'learning_rate': 0.001,       # Learning rate
    'use_attention': True,        # Use attention mechanism
    'num_attention_heads': 2,     # Number of attention heads
    'target_column': 'vst_raw',   # Target column for prediction
    
    # Model architecture configuration - retained for model building
    'use_dual_branch': True,     # Used in build_model to print architecture info
    'use_residual': True,        # Used in build_model to print architecture info
    'long_term_emphasis': 0.9,   # Weight for long-term patterns (70% long-term, 30% short-term)
    
    # Feature engineering
    'use_time_features': True,  
    'use_cumulative_features': True, 
    'use_lagged_features': True,  
    'lag_hours': [336, 672, 1344], # Lag hours for features
    
    # Custom features - simplified set for faster processing
    'custom_features': [
        # Long-term moving averages resistant to anomalies
        {'type': 'ma', 'params': {'hours': 368}},   # 7-day MA
        
        # Exponential moving averages (adapt gradually to changes)
        {'type': 'ema', 'params': {'hours': 768}},   # 12-hour EMA (48 15-min periods)
        
        # Rate of change features (help detect pattern shifts)
        {'type': 'roc', 'params': {'hours': 768}},   # 24-hour rate of change
    ],
    
    # Features to use
    'feature_cols': [
        'rainfall',
        'temperature'
    ],
    'output_features': ['vst_raw'],

    # Other stations to use as features - simplified to just one station
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

    # Error injection parameters for testing
    'error_magnitude': 2.5,      # Magnitude of error injection
    'error_window': 12,          # Window for error injection
}

# Default error periods for validation with injected errors
DEFAULT_ERROR_PERIODS = [
    # Longer anomaly periods for testing model robustness
    {
        'start': '2022-02-15', 'end': '2022-03-15', 
        'type': 'offset', 'magnitude': 100,  # Add constant value for 1 month
        'description': 'Extended period constant offset (1 month)'
    },
    {
        'start': '2022-05-01', 'end': '2022-05-21', 
        'type': 'scaling', 'magnitude': 1.5,  # Multiply by factor for 3 weeks
        'description': 'Extended period scaling factor (3 weeks)'
    },
    {
        'start': '2022-09-01', 'end': '2022-09-15', 
        'type': 'noise', 'magnitude': 50,  # Random noise amplitude
        'description': 'Long period random noise (2 weeks)'
    },
    {
        'start': '2022-11-01', 'end': '2022-11-15', 
        'type': 'missing', 'magnitude': 0,  # Data completely missing (NaN)
        'description': 'Long period of missing data (2 weeks)'
    },
    
    # Combined/complex error patterns
    {
        'start': '2022-07-05', 'end': '2022-07-25', 
        'type': 'scaling', 'magnitude': 0.7,  # Scaling down (data appears lower than it should be)
        'description': 'Long period of scaled-down values'
    },
    {
        'start': '2023-01-10', 'end': '2023-01-30', 
        'type': 'offset', 'magnitude': -200,  # Negative offset (drops in the data)
        'description': 'Extended period with negative offset'
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
    },
    
    # Gradual drift errors (harder to detect)
    {
        'start': '2023-03-01', 'end': '2023-03-31', 
        'type': 'drift', 'magnitude': 300,  # Gradually increasing error 
        'description': 'Gradual upward drift over a month'
    }
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run water level forecasting with error injection for testing anomaly detection')
    
    parser.add_argument('--station', type=str, default='21006846',
                        help='Station ID to use')
    
    parser.add_argument('--output_dir', type=str, default='forecast_results',
                        help='Directory to save results')
    
    # Removed all other modes
    parser.add_argument('--mode', type=str, choices=['test_error'],
                        default='test_error', help='Mode is fixed to test_error for simplicity')
    
    # Remove prediction mode options since we're using only iterative
    parser.add_argument('--prediction_mode', type=str, default='iterative',
                        help='Using iterative prediction mode by default')
    
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
    print(f"  - Sequence Lerngth: {forecaster.config['sequence_length']}")
    print(f"  - Attention: {forecaster.config['use_attention']}")
    
    return model

def run_validation_with_errors(forecaster, visualizer, val_data, error_periods, output_dir):
    """Run and visualize predictions on validation data with injected errors"""
    print("\nPredicting on validation data with injected errors...")
    
    # Get the preprocessor for feature engineering
    preprocessor = forecaster.preprocessor
    target_col = forecaster.config['target_column']
    
    # ----- STEP 1: Create raw data with injected errors -----
    # First, identify which columns are derived from the target column
    derived_features = []
    for feature in preprocessor.feature_cols:
        if target_col in feature:  # This is a feature derived from the target column
            derived_features.append(feature)
    
    print(f"Found {len(derived_features)} features derived from the target column '{target_col}'")
    
    # Make a copy of the validation data for clean predictions
    val_data_clean = val_data.copy()
    
    # Make a copy for error injection - we'll need to regenerate features
    # Start with the core columns before feature engineering
    core_columns = [col for col in val_data.columns if not any(f in col for f in ['lag_', 'ma_', 'ema_', 'roc_', 'cum_'])]
    val_data_with_errors = val_data[core_columns].copy()
    
    # Inject errors into the target column
    print(f"Injecting errors into data column: {target_col}")
    
    # Apply each error period to the data
    for error_period in error_periods:
        start_time = pd.to_datetime(error_period['start'])
        end_time = pd.to_datetime(error_period['end'])
        magnitude = error_period['magnitude']
        error_type = error_period.get('type', 'offset')
        description = error_period.get('description', '')
        
        # Create a mask for the current error period
        mask = (val_data_with_errors.index >= start_time) & (val_data_with_errors.index <= end_time)
        
        if mask.sum() == 0:
            print(f"Warning: No data points found in period {start_time} to {end_time}")
            continue
            
        # Apply the error to the target column
        original_values = val_data_with_errors.loc[mask, target_col].copy()
        
        # Apply different error types
        if error_type == 'offset':
            # Simple additive error
            modified_values = original_values + magnitude
            print(f"Added offset of {magnitude} to {mask.sum()} points from {start_time} to {end_time}")
        elif error_type == 'scaling':
            # Multiplicative error
            modified_values = original_values * magnitude
            print(f"Scaled by {magnitude} for {mask.sum()} points from {start_time} to {end_time}")
        elif error_type == 'noise':
            # Random noise
            noise = np.random.normal(0, magnitude, size=len(original_values))
            modified_values = original_values + noise
            print(f"Added noise with std={magnitude} to {mask.sum()} points from {start_time} to {end_time}")
        elif error_type == 'missing':
            # Set to NaN
            modified_values = np.full_like(original_values, np.nan)
            print(f"Set {mask.sum()} points to NaN from {start_time} to {end_time}")
        elif error_type == 'drift':
            # Generate a linear increase over time
            steps = np.linspace(0, magnitude, len(original_values))
            modified_values = original_values + steps
            print(f"Added drift up to {magnitude} to {mask.sum()} points from {start_time} to {end_time}")
        else:
            # Default to offset
            modified_values = original_values + magnitude
            print(f"Added default offset of {magnitude} to {mask.sum()} points from {start_time} to {end_time}")
        
        val_data_with_errors.loc[mask, target_col] = modified_values
    
    # Verify error injection worked
    print("\nVerifying error injection...")
    for error_period in error_periods:
        start_time = pd.to_datetime(error_period['start'])
        end_time = pd.to_datetime(error_period['end'])
        mask = (val_data_with_errors.index >= start_time) & (val_data_with_errors.index <= end_time)
        if mask.sum() > 0:
            # Calculate average difference between clean and error data
            clean_vals = val_data_clean[target_col].loc[mask]
            error_vals = val_data_with_errors[target_col].loc[mask]
            avg_diff = (error_vals - clean_vals).mean()
            print(f"Verified error injection for period {start_time} to {end_time}: Average difference = {avg_diff:.2f}")
    
    # ----- STEP 2: Regenerate features for the error-injected data -----
    print("\nRegenerating features for error-injected data...")
    
    # Add lagged features using the error-injected data
    if preprocessor.config.get('use_lagged_features', False):
        lags = preprocessor.config.get('lag_hours', [1, 2, 3, 6, 12, 24])
        print(f"  - Regenerating lag features with lags: {lags}")
        
        # Reset the feature engineer's feature list to core features
        preprocessor.feature_engineer.feature_cols = preprocessor.feature_cols.copy()
        
        # Add lagged features of the target column with errors
        val_data_with_errors = preprocessor.feature_engineer.add_lagged_features(
            val_data_with_errors,
            target_col=target_col,
            lags=lags
        )
        
        # Add custom features if specified
        if preprocessor.config.get('custom_features', None):
            print(f"  - Regenerating custom features")
            val_data_with_errors = preprocessor.feature_engineer.add_custom_features(
                val_data_with_errors,
                target_col=target_col,
                feature_specs=preprocessor.config['custom_features']
            )
    
    # Make sure all necessary columns exist
    for col in preprocessor.feature_cols:
        if col not in val_data_with_errors.columns:
            print(f"Warning: Feature column '{col}' is missing in error-injected data. Adding from clean data.")
            val_data_with_errors[col] = val_data_clean[col]
    
    # Debug output - show differences in a few feature columns
    print("\nFeature comparison between clean and error-injected data:")
    sample_idx = val_data_clean.index[100]  # Pick an arbitrary index
    print(f"Sample at index: {sample_idx}")
    
    # Show differences in target column
    clean_target = val_data_clean.loc[sample_idx, target_col]
    error_target = val_data_with_errors.loc[sample_idx, target_col]
    print(f"  Target '{target_col}': Clean={clean_target:.2f}, Error={error_target:.2f}, Diff={error_target-clean_target:.2f}")
    
    # Show differences in a few derived features
    for feature in derived_features[:5]:  # Show first 5 derived features
        if feature in val_data_with_errors.columns:
            clean_feat = val_data_clean.loc[sample_idx, feature]
            error_feat = val_data_with_errors.loc[sample_idx, feature]
            print(f"  Feature '{feature}': Clean={clean_feat:.2f}, Error={error_feat:.2f}, Diff={error_feat-clean_feat:.2f}")
    
    # ----- STEP 3: Run predictions on clean data for reference -----
    print("\nPredicting on clean validation data...")
    clean_results = forecaster.predict(val_data_clean)
    clean_predictions = clean_results['forecasts']['step_1']
    
    # ----- STEP 4: Run anomaly-aware forecasting on error-injected data -----
    print("\nRunning anomaly-aware forecasting on error-injected data...")
    anomaly_results = anomaly_aware_forecasting(
        forecaster,
        val_data_with_errors,
        val_data_clean
    )
    
    # Extract the relevant series for visualization
    corrected_values = anomaly_results['corrected_values']
    error_predictions = anomaly_results['predictions']
    hybrid_predictions = anomaly_results.get('hybrid_predictions', None)  # New field
    z_scores = anomaly_results['z_scores']
    anomaly_mask = anomaly_results['anomaly_mask']
    dynamic_threshold = anomaly_results.get('dynamic_threshold', None)  # Get dynamic threshold
    
    # ----- STEP 5: Align indices for visualization -----
    # Extract the data series with proper alignment
    clean_series = val_data_clean[target_col]
    error_series = val_data_with_errors[target_col]
    
    # Find common indices where all data is available
    common_indices = clean_series.index.intersection(
        error_series.index.intersection(
            clean_predictions.index.intersection(
                corrected_values.index
            )
        )
    )
    
    print(f"\nData dimensions after alignment:")
    print(f"  - Original clean data: {len(clean_series)}")
    print(f"  - Original error data: {len(error_series)}")
    print(f"  - Clean predictions: {len(clean_predictions)}")
    print(f"  - Error predictions: {len(error_predictions)}")
    if hybrid_predictions is not None:
        print(f"  - Hybrid predictions: {len(hybrid_predictions)}")
    print(f"  - Corrected values: {len(corrected_values)}")
    print(f"  - Common indices after alignment: {len(common_indices)}")
    
    # Align all data to common indices
    clean_series = clean_series.loc[common_indices]
    error_series = error_series.loc[common_indices]
    clean_predictions = clean_predictions.loc[common_indices]
    error_predictions = error_predictions.loc[common_indices]
    if hybrid_predictions is not None:
        hybrid_predictions = hybrid_predictions.loc[common_indices]
    corrected_values = corrected_values.loc[common_indices]
    z_scores = z_scores.loc[common_indices]
    anomaly_mask = anomaly_mask.loc[common_indices]
    if dynamic_threshold is not None:
        dynamic_threshold = dynamic_threshold.loc[common_indices]
    
    # Print accuracy metrics if available
    if 'improvement_stats' in anomaly_results and anomaly_results['improvement_stats']:
        stats = anomaly_results['improvement_stats']
        print(f"\nError correction metrics:")
        print(f"  - Original MAE (error vs clean): {stats['original_mae']:.2f}")
        print(f"  - Corrected MAE (corrected vs clean): {stats['corrected_mae']:.2f}")
        print(f"  - Improvement: {stats['mae_improvement']:.2f}%")
        print(f"  - Total anomalies detected: {stats['num_anomalies']} ({stats['percent_anomalies']:.2f}% of data)")
    
    # ----- STEP 6: Create visualizations -----
    print("\nCreating visualizations...")
    
    # 1. Overview plot showing all data
    plt.figure(figsize=(16, 8))
    
    # Plot data
    plt.plot(clean_series.index, clean_series.values, 'b-', 
             label='Original Clean Data', linewidth=1.5, alpha=0.7)
    plt.plot(error_series.index, error_series.values, 'r-', 
             label='Error-Injected Data', linewidth=1.5)
    
    # Plot predictions
    plt.plot(clean_predictions.index, clean_predictions.values, 'g-', 
             label='Predictions on Clean Data', linewidth=1.5, alpha=0.7)
             
    # Add hybrid predictions if available
    if hybrid_predictions is not None:
        plt.plot(hybrid_predictions.index, hybrid_predictions.values, 'c--', 
                label='Hybrid Predictions', linewidth=1.5, alpha=0.6)
                
    plt.plot(corrected_values.index, corrected_values.values, 'm-', 
             label='Anomaly-Corrected Values', linewidth=2.0)
    
    # Highlight error periods
    for i, period in enumerate(error_periods):
        start_time = pd.to_datetime(period['start'])
        end_time = pd.to_datetime(period['end'])
        plt.axvspan(start_time, end_time, color='yellow', alpha=0.2,
                   label=f"{period['type']} error" if i == 0 else "")
    
    # Highlight detected anomalies
    if anomaly_mask.sum() > 0:
        anomaly_points = error_series[anomaly_mask]
        plt.scatter(anomaly_points.index, anomaly_points.values, 
                   color='red', s=30, alpha=0.7, label='Detected Anomalies')
    
    plt.title("Water Level Forecasting with Anomaly Detection and Correction", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Water Level", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overview_with_correction.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create focused plots for specific error periods
    for i, error_period in enumerate(error_periods[:5]):  # Limit to first 5 error periods
        # Get error period details
        start_time = pd.to_datetime(error_period['start'])
        end_time = pd.to_datetime(error_period['end'])
        error_type = error_period['type']
        description = error_period.get('description', f"{error_type} error")
        
        # Create window around error period
        padding_days = 3
        padding = pd.Timedelta(days=padding_days)
        window_start = start_time - padding
        window_end = end_time + padding
        
        # Filter data to window
        window_mask = (clean_series.index >= window_start) & (clean_series.index <= window_end)
        
        if window_mask.sum() == 0:
            continue
            
        # Get windowed data
        clean_window = clean_series[window_mask]
        error_window = error_series[window_mask]
        clean_pred_window = clean_predictions[window_mask]
        error_pred_window = error_predictions[window_mask]
        if hybrid_predictions is not None:
            hybrid_pred_window = hybrid_predictions[window_mask]
        corrected_window = corrected_values[window_mask]
        z_scores_window = z_scores[window_mask]
        anomaly_mask_window = anomaly_mask[window_mask]
        
        # Count anomalies in this window
        anomaly_count = anomaly_mask_window.sum()
        
        # Calculate metrics
        clean_mae = np.mean(np.abs(clean_window - clean_pred_window))
        error_mae = np.mean(np.abs(clean_window - error_window))
        corrected_mae = np.mean(np.abs(clean_window - corrected_window))
        correction_improvement = ((error_mae - corrected_mae) / error_mae) * 100 if error_mae > 0 else 0
        
        # Create a 2-panel plot: upper for data, lower for z-scores
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Upper panel: Plot data and predictions
        ax1.plot(clean_window.index, clean_window.values, 'b-', 
                label='Original Clean Data', linewidth=2)
        ax1.plot(error_window.index, error_window.values, 'r-', 
                label='Error-Injected Data', linewidth=2)
        ax1.plot(clean_pred_window.index, clean_pred_window.values, 'g-', 
                label='Predictions on Clean Data', linewidth=1.5)
        
        # Add hybrid predictions if available
        if hybrid_predictions is not None:
            ax1.plot(hybrid_pred_window.index, hybrid_pred_window.values, 'c--', 
                    label='Hybrid Predictions', linewidth=1.5, alpha=0.7)
        
        ax1.plot(corrected_window.index, corrected_window.values, 'm-', 
                label='Anomaly-Corrected Values', linewidth=2.5)
        
        # Highlight the error period
        ax1.axvspan(start_time, end_time, color='yellow', alpha=0.2, label='Error Period')
        
        # Highlight detected anomalies
        if anomaly_mask_window.sum() > 0:
            anomaly_points = error_window[anomaly_mask_window]
            ax1.scatter(anomaly_points.index, anomaly_points.values, 
                      color='red', s=50, alpha=0.8, marker='x', label='Detected Anomalies')
        
        # Title with metrics
        title = (f"{description}\n"
                f"Clean MAE: {clean_mae:.2f}, Error MAE: {error_mae:.2f}, Corrected MAE: {corrected_mae:.2f}\n"
                f"Correction Improvement: {correction_improvement:.1f}%, Anomalies Detected: {anomaly_count}")
        
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel("Water Level", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Lower panel: Plot z-scores
        ax2.plot(z_scores_window.index, z_scores_window.values, 'k-', label='Z-Score')
        
        # Show dynamic threshold if available
        if dynamic_threshold is not None:
            dynamic_threshold_window = dynamic_threshold[window_mask]
            ax2.plot(dynamic_threshold_window.index, dynamic_threshold_window.values, 'r--', 
                   label=f"Dynamic Threshold")
        else:
            # Show static threshold
            ax2.axhline(y=forecaster.config['z_score_threshold'], color='r', linestyle='--', 
                       label=f"Threshold ({forecaster.config['z_score_threshold']})")
        
        # Highlight anomalies in z-score plot
        if anomaly_mask_window.sum() > 0:
            anomaly_z_scores = z_scores_window[anomaly_mask_window]
            ax2.scatter(anomaly_z_scores.index, anomaly_z_scores.values, 
                       color='red', s=50, alpha=0.8, marker='x')
        
        ax2.set_ylabel("Z-Score", fontsize=14)
        ax2.set_xlabel("Date", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"error_period_{i+1}_{error_period['type']}_with_correction.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nValidation with error injection and anomaly correction complete.")

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

def predict_with_self_correction(forecaster, val_data_with_errors, val_data_clean, max_iterations=3, z_score_threshold=5.0):
    """
    Make predictions with a self-correction feedback loop to handle anomalies.
    
    Args:
        forecaster: The WaterLevelForecaster model
        val_data_with_errors: Validation data with injected errors
        val_data_clean: Clean validation data (used for evaluation only, not for predictions)
        max_iterations: Maximum number of correction iterations
        z_score_threshold: Z-score threshold for detecting anomalies
        
    Returns:
        Dictionary with all prediction results and correction information
    """
    print(f"\nStarting self-correction prediction process (max {max_iterations} iterations)...")
    
    preprocessor = forecaster.preprocessor
    target_col = forecaster.config['target_column']
    
    # Initialize working data for corrections
    working_data = val_data_with_errors.copy()
    
    # Track the corrected values and predictions across iterations
    all_predictions = {}
    all_corrections = {}
    
    # Keep track of indices that were corrected
    corrected_indices = set()
    
    # Perform iterative correction
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}/{max_iterations}:")
        
        # Make predictions with current working data
        results = forecaster.predict(working_data)
        predictions = results['forecasts']['step_1']
        all_predictions[f"iteration_{iteration+1}"] = predictions
        
        # Calculate differences between predictions and working data
        working_values = working_data[target_col].reindex(predictions.index)
        diffs = np.abs(working_values - predictions)
        
        # Calculate robust statistics for anomaly detection
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff)) * 1.4826  # Scale factor for normal distribution
        
        # Avoid division by zero
        if mad < 1e-10:
            mad = 1e-10
            
        # Calculate z-scores
        z_scores = (diffs - median_diff) / mad
        
        # Identify anomalies based on z-score threshold
        anomaly_mask = z_scores > z_score_threshold
        anomaly_indices = working_values.index[anomaly_mask]
        
        # Count anomalies
        num_anomalies = len(anomaly_indices)
        print(f"  - Detected {num_anomalies} anomalies with z-score > {z_score_threshold}")
        
        # If no anomalies found, stop iterating
        if num_anomalies == 0:
            print("  - No anomalies detected, stopping correction process")
            break
            
        # Create a copy of working data for this iteration's corrections
        corrected_data = working_data.copy()
        
        # Replace anomalous values with predictions
        for idx in anomaly_indices:
            if idx in predictions.index:
                old_value = corrected_data.loc[idx, target_col]
                new_value = predictions.loc[idx]
                corrected_data.loc[idx, target_col] = new_value
                
                # Add to the set of corrected indices
                corrected_indices.add(idx)
                
                # Debug output for first few corrections
                if len(all_corrections) < 5 or iteration == 0:
                    print(f"  - Corrected point at {idx}: {old_value:.2f} → {new_value:.2f}, z-score: {z_scores[anomaly_mask].iloc[list(anomaly_indices).index(idx)]:.2f}")
        
        # Store corrections for this iteration
        corrections = corrected_data[target_col].copy()
        all_corrections[f"iteration_{iteration+1}"] = corrections
        
        # Regenerate features based on corrected data
        print(f"  - Regenerating features with corrected data...")
        
        # Add lagged features using the corrected data
        if preprocessor.config.get('use_lagged_features', False):
            lags = preprocessor.config.get('lag_hours', [1, 2, 3, 6, 12, 24])
            
            # Create a clean copy of the corrected data with only core columns
            core_columns = [col for col in corrected_data.columns if not any(f in col for f in ['lag_', 'ma_', 'ema_', 'roc_', 'cum_'])]
            corrected_data_core = corrected_data[core_columns].copy()
            
            # Reset the feature engineer's feature list
            preprocessor.feature_engineer.feature_cols = preprocessor.feature_cols.copy()
            
            # Add lagged features of the corrected target column
            corrected_data_with_features = preprocessor.feature_engineer.add_lagged_features(
                corrected_data_core,
                target_col=target_col,
                lags=lags
            )
            
            # Add custom features if specified
            if preprocessor.config.get('custom_features', None):
                corrected_data_with_features = preprocessor.feature_engineer.add_custom_features(
                    corrected_data_with_features,
                    target_col=target_col,
                    feature_specs=preprocessor.config['custom_features']
                )
            
            # Make sure all necessary columns exist
            for col in preprocessor.feature_cols:
                if col not in corrected_data_with_features.columns:
                    corrected_data_with_features[col] = working_data[col]
            
            # Update working data for next iteration
            working_data = corrected_data_with_features.copy()
    
    # Make a final prediction with the fully corrected data
    final_results = forecaster.predict(working_data)
    
    # Prepare return data structure
    results = {
        'initial_predictions': all_predictions.get('iteration_1', None),
        'final_predictions': final_results['forecasts']['step_1'],
        'all_predictions': all_predictions,
        'all_corrections': all_corrections,
        'corrected_indices': list(corrected_indices),
        'num_corrected': len(corrected_indices),
        'iterations_performed': len(all_predictions)
    }
    
    print(f"\nSelf-correction process completed after {len(all_predictions)} iterations")
    print(f"Total points corrected: {len(corrected_indices)}")
    
    return results

def anomaly_aware_forecasting(forecaster, val_data_with_errors, val_data_clean=None):
    """
    Enhanced anomaly-aware forecasting that uses clean data as a reference point.
    This approach:
    
    1. Creates a "clean features" version of the data based on historical patterns
    2. Makes predictions on potentially anomalous data
    3. Makes reference predictions using clean features
    4. Detects anomalies by comparing actual values with both prediction sets
    5. For points flagged as anomalies, uses the clean-based prediction instead
    
    Args:
        forecaster: The WaterLevelForecaster model
        val_data_with_errors: Data that may contain anomalies
        val_data_clean: Optional clean data for evaluation/comparison
        
    Returns:
        Dictionary with results including the corrected values
    """
    print("\nRunning enhanced anomaly-aware forecasting...")
    
    target_col = forecaster.config['target_column']
    base_z_score_threshold = forecaster.config.get('z_score_threshold', 3.0)
    
    # Create a working copy of the data
    working_data = val_data_with_errors.copy()
    
    # Step 1: Create a predicted baseline for the target column
    # This provides a reference point not influenced by recent anomalies
    print("Creating baseline prediction from historical patterns...")
    
    # Make initial predictions on the potentially anomalous data
    results = forecaster.predict(working_data)
    initial_predictions = results['forecasts']['step_1']
    
    # Extract the target values and predictions
    actual_values = working_data[target_col].reindex(initial_predictions.index)
    
    # Step 2: Generate a robust reference by creating a hybrid dataset
    # Use predictions where substantial deviations exist, otherwise use actual values
    # This creates a "semi-clean" dataset as a starting point
    print("Creating robust reference data...")
    
    # Calculate absolute differences and z-scores for initial anomaly detection
    diffs = np.abs(actual_values - initial_predictions)
    
    # Calculate robust statistics for anomaly detection
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff)) * 1.4826  # Scale factor for normal distribution
    
    # Avoid division by zero
    if mad < 1e-10:
        mad = 1e-10
        
    # Calculate z-scores for deviation detection
    z_scores = (diffs - median_diff) / mad
    
    # Create a hybrid target column using predictions for high-deviation points
    hybrid_data = working_data.copy()
    # Use adaptive threshold: higher during stable periods, lower during volatile periods
    local_volatility = diffs.rolling(window=96, min_periods=24).std() / diffs.rolling(window=96, min_periods=24).mean()
    local_volatility = local_volatility.fillna(0)
    
    # Normalize volatility to range [0.7, 1.3] to adjust threshold
    norm_volatility = 1.0 - 0.3 * (local_volatility / local_volatility.max())
    norm_volatility = norm_volatility.clip(0.7, 1.3)
    
    # Dynamic z-score threshold based on local volatility
    dynamic_threshold = base_z_score_threshold * norm_volatility
    
    # Initial high deviation mask using dynamic threshold
    high_deviation_mask = z_scores > dynamic_threshold
    
    # Replace high-deviation values with predictions as a starting point
    if high_deviation_mask.sum() > 0:
        # Apply only to the target column
        hybrid_values = actual_values.copy()
        hybrid_values[high_deviation_mask] = initial_predictions[high_deviation_mask]
        hybrid_data.loc[hybrid_values.index, target_col] = hybrid_values
        print(f"Replaced {high_deviation_mask.sum()} high-deviation points with predictions")
    
    # Step 3: Regenerate derived features using the hybrid data
    # This helps break contamination in lagged and derived features
    print("Regenerating features based on hybrid data...")
    
    # Extract core columns (non-derived features)
    core_columns = [col for col in working_data.columns if not any(f in col for f in ['lag_', 'ma_', 'ema_', 'roc_', 'cum_'])]
    hybrid_core = hybrid_data[core_columns].copy()
    
    # Regenerate features
    preprocessor = forecaster.preprocessor
    
    # Regenerate lagged features
    if preprocessor.config.get('use_lagged_features', False):
        lags = preprocessor.config.get('lag_hours', [1, 2, 3, 6, 12, 24])
        
        # Reset feature engineer feature list
        preprocessor.feature_engineer.feature_cols = preprocessor.feature_cols.copy()
        
        # Add lagged features using hybrid data
        hybrid_with_features = preprocessor.feature_engineer.add_lagged_features(
            hybrid_core,
            target_col=target_col,
            lags=lags
        )
        
        # Add custom features if specified
        if preprocessor.config.get('custom_features', None):
            hybrid_with_features = preprocessor.feature_engineer.add_custom_features(
                hybrid_with_features,
                target_col=target_col,
                feature_specs=preprocessor.config['custom_features']
            )
        
        # Make sure all required columns exist
        for col in preprocessor.feature_cols:
            if col not in hybrid_with_features.columns:
                hybrid_with_features[col] = working_data[col]
    else:
        hybrid_with_features = hybrid_data.copy()
    
    # Step 4: Make "cleaner" predictions using the hybrid data with regenerated features
    print("Making predictions using hybrid data...")
    hybrid_results = forecaster.predict(hybrid_with_features)
    hybrid_predictions = hybrid_results['forecasts']['step_1']
    
    # Step 5: Use both prediction sets to detect anomalies more robustly
    print("Performing robust anomaly detection...")
    
    # Calculate deviations from hybrid-based predictions
    hybrid_diffs = np.abs(actual_values - hybrid_predictions)
    
    # Calculate robust statistics
    hybrid_median_diff = np.median(hybrid_diffs)
    hybrid_mad = np.median(np.abs(hybrid_diffs - hybrid_median_diff)) * 1.4826
    
    # Avoid division by zero
    if hybrid_mad < 1e-10:
        hybrid_mad = 1e-10
    
    # Calculate z-scores for hybrid-based detection
    hybrid_z_scores = (hybrid_diffs - hybrid_median_diff) / hybrid_mad
    
    # Calculate percentage differences for scaling detection
    percent_diff = np.abs((actual_values - hybrid_predictions) / (hybrid_predictions + 1e-10)) * 100
    percent_diff = percent_diff.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate context-aware dynamic thresholds
    # 1. Local volatility-based threshold 
    hybrid_local_volatility = hybrid_diffs.rolling(window=96, min_periods=24).std() / hybrid_diffs.rolling(window=96, min_periods=24).mean()
    hybrid_local_volatility = hybrid_local_volatility.fillna(0)
    
    # Normalize volatility
    hybrid_norm_volatility = 1.0 - 0.3 * (hybrid_local_volatility / hybrid_local_volatility.max())
    hybrid_norm_volatility = hybrid_norm_volatility.clip(0.7, 1.3)
    
    # 2. Pattern-based threshold adjustment
    # Increase sensitivity during sudden changes in trend
    trend_change = hybrid_predictions.diff().rolling(window=12).std()
    trend_change = trend_change / trend_change.rolling(window=96, min_periods=24).mean()
    trend_change = trend_change.fillna(1)
    
    # Normalize trend change factor (lower threshold during trend changes)
    trend_factor = 1.0 - 0.4 * (trend_change / trend_change.max())
    trend_factor = trend_factor.clip(0.6, 1.0)
    
    # 3. Combined dynamic threshold
    dynamic_z_threshold = base_z_score_threshold * hybrid_norm_volatility * trend_factor
    dynamic_percent_threshold = 25.0 * hybrid_norm_volatility  # Base 25% deviation, adjusted by volatility
    
    # Calculate anomaly masks with dynamic thresholds
    z_score_anomalies = hybrid_z_scores > dynamic_z_threshold
    percent_anomalies = percent_diff > dynamic_percent_threshold
    
    # Additional mask for extreme values (always flag extremely large deviations)
    extreme_anomalies = hybrid_z_scores > (base_z_score_threshold * 3)
    
    # Add logic for consecutive anomalies (more likely to be true anomalies)
    rolling_anomaly_count = (z_score_anomalies | percent_anomalies).rolling(window=3).sum()
    consecutive_anomalies = rolling_anomaly_count >= 2
    
    # Combine all anomaly detection methods
    anomaly_mask = z_score_anomalies | percent_anomalies | extreme_anomalies | consecutive_anomalies
    
    # Step 6: Create corrected values
    # Create a corrected dataset by replacing anomalies with hybrid predictions
    corrected_values = actual_values.copy()
    corrected_values[anomaly_mask] = hybrid_predictions[anomaly_mask]
    
    # Count anomalies
    anomaly_indices = actual_values.index[anomaly_mask]
    num_anomalies = len(anomaly_indices)
    
    print(f"Detected {num_anomalies} anomalies using enhanced approach")
    print(f"Z-score anomalies: {z_score_anomalies.sum()}")
    print(f"Percentage anomalies: {percent_anomalies.sum()}")
    print(f"Extreme anomalies: {extreme_anomalies.sum()}")
    print(f"Consecutive anomalies: {consecutive_anomalies.sum()}")
    
    # Create a DataFrame for the corrected data
    corrected_data = working_data.copy()
    corrected_data.loc[corrected_values.index, target_col] = corrected_values
    
    # Calculate improvement statistics if clean data is provided
    improvement_stats = {}
    if val_data_clean is not None:
        clean_values = val_data_clean[target_col].reindex(corrected_values.index)
        
        # Calculate error metrics against clean data
        original_mae = np.mean(np.abs(clean_values - actual_values))
        corrected_mae = np.mean(np.abs(clean_values - corrected_values))
        initial_pred_mae = np.mean(np.abs(clean_values - initial_predictions))
        hybrid_pred_mae = np.mean(np.abs(clean_values - hybrid_predictions))
        
        # Calculate improvement percentage
        mae_improvement = ((original_mae - corrected_mae) / original_mae) * 100
        
        improvement_stats = {
            'original_mae': original_mae,
            'corrected_mae': corrected_mae,
            'initial_pred_mae': initial_pred_mae,
            'hybrid_pred_mae': hybrid_pred_mae,
            'mae_improvement': mae_improvement,
            'num_anomalies': num_anomalies,
            'percent_anomalies': (num_anomalies / len(actual_values)) * 100
        }
        
        print(f"Original MAE (vs clean): {original_mae:.4f}")
        print(f"Corrected MAE (vs clean): {corrected_mae:.4f}")
        print(f"Initial prediction MAE (vs clean): {initial_pred_mae:.4f}")
        print(f"Hybrid prediction MAE (vs clean): {hybrid_pred_mae:.4f}")
        print(f"MAE improvement: {mae_improvement:.2f}%")
    
    # Return results
    return {
        'actual_values': actual_values,
        'predictions': initial_predictions,
        'hybrid_predictions': hybrid_predictions,
        'z_scores': hybrid_z_scores,
        'dynamic_threshold': dynamic_z_threshold,  # Return the dynamic threshold for visualization
        'anomaly_mask': anomaly_mask,
        'anomaly_indices': list(anomaly_indices),
        'corrected_values': corrected_values,
        'corrected_data': corrected_data,
        'improvement_stats': improvement_stats
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup paths
    station_id = args.station
    base_output_path = project_dir / args.output_dir
    
    # Create output directory
    output_path = base_output_path / "error_injection_test"
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Project root: {project_dir}")
    print(f"Using station: {station_id}")
    print(f"Output directory: {output_path}")
    
    # Get configuration
    config = DEFAULT_CONFIG.copy()
    
    # Print key configuration values
    print("\nConfiguration:")
    for key, value in {
        'sequence_length': config['sequence_length'],
        'use_attention': config['use_attention'],
        'num_layers': config['num_layers'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'hidden_size': config['hidden_size']
    }.items():
        print(f"  - {key.replace('_', ' ').title()}: {format_config_value(key, value)}")
    
    # Initialize forecasting model and visualizer
    forecaster = WaterLevelForecaster(config)
    visualizer = ForecastVisualizer(config)
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(config)
    forecaster.preprocessor = preprocessor
    
    print("Loading and filtering data...")
    train_data, val_data, test_data = load_and_filter_data(preprocessor, project_dir, station_id)
    
    # Train the model
    train_and_save_model(forecaster, train_data, val_data, project_dir, station_id, output_path)
        
    # Create directory for validation with errors
    val_dir = output_path / "validation_with_errors"
    val_dir.mkdir(exist_ok=True)
    
    print(f"\n=== Validation Data with Injected Errors ===")
    
    # Run with injected errors - removed prediction_mode parameter
    run_validation_with_errors(forecaster, visualizer, val_data, DEFAULT_ERROR_PERIODS, val_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 

'''
Arg options 

python run_forecast.py --station 21006846 --output_dir forecast_results --mode predict --model_path trained_model.pth
cd "Project_Code - CORRECT"; python experiments/forecaster/run_forecast.py --mode test_error --prediction_mode standard
'''