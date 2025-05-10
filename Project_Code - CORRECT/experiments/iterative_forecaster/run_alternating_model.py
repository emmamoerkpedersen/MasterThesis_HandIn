import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import local modules
from experiments.iterative_forecaster.alternating_config import ALTERNATING_CONFIG
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _4_anomaly_detection.z_score import calculate_z_scores_mad
from _4_anomaly_detection.anomaly_visualization import plot_water_level_anomalies
from experiments.iterative_forecaster.alternating_trainer import AlternatingTrainer
from experiments.iterative_forecaster.alternating_forecast_model import AlternatingForecastModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run alternating LSTM model for water level forecasting')
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--error_multiplier', type=float, default=None, 
                      help='Error multiplier for synthetic errors. If not provided, no errors are injected.')
    parser.add_argument('--quick_mode', action='store_true', help='Enable quick mode with reduced data (3 years training, 1 year validation)')
    parser.add_argument('--error_type', type=str, default='both', choices=['both', 'train', 'validation', 'none'],
                      help='Which datasets to inject errors into (both, train, validation, or none)')
    return parser.parse_args()

def run_alternating_model(args):
    """Run the alternating LSTM model with the specified parameters."""
    # Update configuration with command line arguments
    config = ALTERNATING_CONFIG.copy()
    config.update({
        'quick_mode': args.quick_mode,
    })
    
    # Ensure the config has feature_stations entry (needed by DataPreprocessor)
    if 'feature_stations' not in config:
        config['feature_stations'] = []
    
    # Print configuration
    print("\nModel Configuration:")
    for key, value in config.items():
        if not isinstance(value, (list, dict)):
            print(f"  {key}: {value}")
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(config)
    print(f"\nInitializing trainer...")
    trainer = AlternatingTrainer(config, preprocessor)
    
    # Use our custom data loading method instead of the preprocessor's method
    print(f"\nLoading data for station {args.station_id}...")
    train_data, val_data, test_data = trainer.load_data(project_dir, args.station_id)
    
    # Store original data for visualization
    original_train_data = train_data.copy()
    original_val_data = val_data.copy()
    
    # Check if we need to inject synthetic errors
    if args.error_multiplier is not None and args.error_type != 'none':
        from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
        from utils.error_utils import configure_error_params, inject_errors_into_dataset
        from config import SYNTHETIC_ERROR_PARAMS
        
        print(f"\nInjecting synthetic errors with multiplier {args.error_multiplier:.1f}x...")
        print(f"Error injection mode: {args.error_type}")
        error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
        
        # Identify target columns for error injection
        water_level_cols = ['vst_raw','vst_raw_feature']
        print(f"Injecting errors into columns: {water_level_cols}")
        
        # Process training data if needed
        if args.error_type in ['both', 'train']:
            print("\nProcessing TRAINING data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            train_data_with_errors, train_error_report = inject_errors_into_dataset(
                original_train_data, error_generator, f"{args.station_id}_train", water_level_cols
            )
            train_data = train_data_with_errors
            
            # Handle error reporting based on actual returned format
            if isinstance(train_error_report, dict) and 'total_errors' in train_error_report:
                print(f"Training errors injected: {train_error_report['total_errors']} errors")
                if 'error_counts' in train_error_report:
                    for error_type, count in train_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Training errors injected successfully")
        
        # Process validation data if needed
        if args.error_type in ['both', 'validation']:
            print("\nProcessing VALIDATION data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            val_data_with_errors, val_error_report = inject_errors_into_dataset(
                original_val_data, error_generator, f"{args.station_id}_val", water_level_cols
            )
            val_data = val_data_with_errors
            
            # Handle error reporting based on actual returned format
            if isinstance(val_error_report, dict) and 'total_errors' in val_error_report:
                print(f"Validation errors injected: {val_error_report['total_errors']} errors")
                if 'error_counts' in val_error_report:
                    for error_type, count in val_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Validation errors injected successfully")
    else:
        print("\nNo synthetic errors injected.")
    
    # Make sure the model's week_steps matches our config
    trainer.model.week_steps = config['week_steps']
    
    print("\nTraining model...")
    history, val_predictions, val_targets = trainer.train(
        train_data, val_data, config['epochs'], config['batch_size']
    )
    
    # Skip test predictions and focus only on validation results
    print("\nSkipping test predictions, focusing on validation results only...")
    
    # Inverse transform the validation predictions to original scale
    print("\nConverting predictions back to original scale...")
    val_predictions_np = val_predictions.numpy()
    
    # Check dimensionality and reshape if necessary
    if val_predictions_np.ndim == 3:
        val_predictions_np = val_predictions_np.reshape(val_predictions_np.shape[0] * val_predictions_np.shape[1], -1)
    elif val_predictions_np.ndim == 2:
        # If predictions are already 2D but need to ensure it's [samples, features]
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    elif val_predictions_np.ndim == 1:
        # If predictions are 1D, reshape to [samples, 1]
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    
    # Apply inverse transform to get back to original scale
    val_predictions_original = trainer.target_scaler.inverse_transform(val_predictions_np)
    
    # Flatten for DataFrame creation
    val_predictions_original = val_predictions_original.flatten()
    
    # Create prediction DataFrame for visualization
    val_pred_df = pd.DataFrame({
        'vst_raw': val_predictions_original,
    }, index=val_data.index[:len(val_predictions_original)])
    
    print(f"Predictions shape: {val_predictions_original.shape}")
    print(f"Min prediction value: {np.min(val_predictions_original)}")
    print(f"Max prediction value: {np.max(val_predictions_original)}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Training convergence plot
    plot_convergence(history, str(args.station_id), 
                     title=f"Training and Validation Loss - Station {args.station_id}")
    
    # Define visualization title including error info if applicable
    viz_title_suffix = "Validation Predictions"
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        viz_title_suffix += f" (Error Mult: {args.error_multiplier}x)"
    
    # Validation predictions plot - if errors were injected, we want to show both original and corrupted data
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        # Prepare synthetic data for create_full_plot
        synthetic_data = {
            'data': val_data.copy(),
            'error_periods': []  # We don't have explicit error period info, but still can show corrupted data
        }
        
        # Create visualization showing both original and corrupted data
        create_full_plot(
            original_val_data,  # Original data as the reference
            val_pred_df,        # Model predictions
            str(args.station_id), 
            config, 
            min(history['val_loss']), 
            title_suffix=viz_title_suffix,
            synthetic_data=synthetic_data  # Pass synthetic data to show corrupted values
        )
    else:
        # Standard plot with just the validation data and predictions
        create_full_plot(
            original_val_data, 
            val_pred_df, 
            str(args.station_id), 
            config, 
            min(history['val_loss']), 
            title_suffix=viz_title_suffix
        )
    
    # Calculate anomalies for the validation set
    print("\nCalculating anomalies...")
    
    # If we're using synthetic errors, compare against the corrupted data to detect anomalies
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        # Calculate anomalies between corrupted data and predictions
        z_scores, anomalies = calculate_z_scores_mad(
            val_data['vst_raw'].values, 
            val_pred_df['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected in corrupted data: {np.sum(anomalies)}")
        
        # Generate anomaly visualization with reference to corrupted data
        plot_title = f"Water Level Forecasting with Anomaly Detection - Station {args.station_id}"
        plot_title += f" (Data with Synthetic Errors {args.error_multiplier}x)"
        
        # Create output directory
        anomaly_viz_dir = Path(project_dir) / "results" / "anomaly_detection"
        anomaly_viz_dir.mkdir(parents=True, exist_ok=True)
        
        png_path, html_path = plot_water_level_anomalies(
            test_data=val_data,
            predictions=val_pred_df['vst_raw'],
            z_scores=z_scores,
            anomalies=anomalies,
            title=plot_title,
            output_dir=anomaly_viz_dir,
            save_png=True,
            save_html=True,
            show_plot=False
        )
        
        print(f"Anomaly visualization (on corrupted data) saved to:")
        print(f"PNG: {png_path}")
        print(f"HTML: {html_path}")
        
        # Also calculate anomalies between original data and corrupted data to verify synthetic errors
        original_vs_corrupted_zscores, original_vs_corrupted_anomalies = calculate_z_scores_mad(
            original_val_data['vst_raw'].values, 
            val_data['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected between original and corrupted data: {np.sum(original_vs_corrupted_anomalies)}")
        
        # Plot these anomalies to verify synthetic error injection
        synthetic_error_plot_title = f"Synthetic Errors Detected - Station {args.station_id} (Error Multiplier: {args.error_multiplier}x)"
        
        syn_png_path, syn_html_path = plot_water_level_anomalies(
            test_data=original_val_data,
            predictions=val_data['vst_raw'],
            z_scores=original_vs_corrupted_zscores,
            anomalies=original_vs_corrupted_anomalies,
            title=synthetic_error_plot_title,
            output_dir=anomaly_viz_dir,
            save_png=True,
            save_html=True,
            show_plot=False,
            filename_prefix="synthetic_errors_"
        )
        
        print(f"Synthetic errors visualization saved to:")
        print(f"PNG: {syn_png_path}")
        print(f"HTML: {syn_html_path}")
    else:
        # Standard anomaly detection against original data
        z_scores, anomalies = calculate_z_scores_mad(
            val_data['vst_raw'].values, 
            val_pred_df['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected: {np.sum(anomalies)}")
        
        # Generate anomaly visualization
        plot_title = f"Water Level Forecasting with Anomaly Detection - Station {args.station_id}"
        
        # Create output directory
        anomaly_viz_dir = Path(project_dir) / "results" / "anomaly_detection"
        anomaly_viz_dir.mkdir(parents=True, exist_ok=True)
        
        png_path, html_path = plot_water_level_anomalies(
            test_data=val_data,
            predictions=val_pred_df['vst_raw'],
            z_scores=z_scores,
            anomalies=anomalies,
            title=plot_title,
            output_dir=anomaly_viz_dir,
            save_png=True,
            save_html=True,
            show_plot=False
        )
        
        print(f"Anomaly visualization saved to:")
        print(f"PNG: {png_path}")
        print(f"HTML: {html_path}")
    
    # Calculate metrics on validation data instead of test data
    from utils.pipeline_utils import calculate_performance_metrics
    valid_mask = ~np.isnan(original_val_data['vst_raw'].values)
    pred_mask = ~np.isnan(val_predictions_original)
    combined_mask = valid_mask[:len(pred_mask)] & pred_mask
    
    # Only calculate metrics if we have sufficient valid data points
    if np.sum(combined_mask) > 0:
        metrics = calculate_performance_metrics(
            original_val_data['vst_raw'].values[:len(pred_mask)], 
            val_predictions_original, 
            combined_mask
        )
        
        # Print metrics
        print("\nValidation Metrics (against original data):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        # If we injected errors, also calculate metrics against the corrupted data
        if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
            try:
                # Calculate error correction improvement
                corrupted_metrics = calculate_performance_metrics(
                    val_data['vst_raw'].values[:len(pred_mask)], 
                    val_predictions_original, 
                    combined_mask
                )
                
                print("\nValidation Metrics (against corrupted data):")
                for metric, value in corrupted_metrics.items():
                    print(f"  {metric}: {value:.6f}")
                    
                print("\nError Correction Analysis:")
                
                # Original vs Corrupted (how bad are the errors)
                error_metrics = calculate_performance_metrics(
                    original_val_data['vst_raw'].values[:len(pred_mask)],
                    val_data['vst_raw'].values[:len(pred_mask)],
                    combined_mask
                )
                
                print("  Error Impact (original vs corrupted):")
                for metric, value in error_metrics.items():
                    print(f"    {metric}: {value:.6f}")
                
                # Calculate correction percentage
                if error_metrics['rmse'] > 0:
                    correction_rmse = (1 - metrics['rmse'] / error_metrics['rmse']) * 100
                    print(f"    RMSE improvement: {correction_rmse:.2f}%")
                
                if error_metrics['mae'] > 0:
                    correction_mae = (1 - metrics['mae'] / error_metrics['mae']) * 100
                    print(f"    MAE improvement: {correction_mae:.2f}%")
                
                r2_improvement = metrics['r2'] - error_metrics['r2']
                print(f"    R² improvement: {r2_improvement:.4f}")
                
            except Exception as e:
                print(f"\nError during error correction analysis: {str(e)}")
                print("Continuing with model evaluation...")
    else:
        print("\nNot enough valid data points to calculate metrics")
        metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan')}
    
    return metrics

if __name__ == "__main__":
    args = parse_arguments()
    metrics = run_alternating_model(args)
    print("\nModel run completed successfully!") 