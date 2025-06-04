import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import local modules
from experiments.iterative_forecaster.alternating_config import ALTERNATING_CONFIG
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence, create_synthetic_error_zoom_plots
from _4_anomaly_detection.z_score import calculate_z_scores_mad
from _4_anomaly_detection.anomaly_visualization import (
    plot_water_level_anomalies, 
    calculate_anomaly_confidence, 
    create_anomaly_zoom_plots,
)
from experiments.iterative_forecaster.alternating_visualization import plot_zoom_comparison
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
    parser.add_argument('--experiment', type=str, default='baseline', help='Experiment number/name for organizing results (e.g., 0, 1, baseline, etc.)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results (default: 42)')
    return parser.parse_args()

def setup_experiment_directories(project_dir, experiment_name):
    """
    Create experiment-specific directory structure.
    
    Args:
        project_dir: Project root directory
        experiment_name: Name/number of the experiment
        
    Returns:
        dict: Dictionary with paths to experiment directories
    """
    # Base experiment directory under Iterative model results
    exp_dir = Path(project_dir) / "results" / "Iterative model results" / f"experiment_{experiment_name}"
    
    # Create subdirectories
    directories = {
        'base': exp_dir,
        'diagnostics': exp_dir / "diagnostics",
        'visualizations': exp_dir / "visualizations",
        'anomaly_detection': exp_dir / "anomaly_detection",
        'behavior_analysis': exp_dir / "visualizations" / "alternating_behavior"
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment directories created under: {exp_dir}")
    return directories

def set_random_seeds(seed):
    """
    Set random seeds for reproducible results across all libraries.
    
    Args:
        seed: Integer seed value
    """
    print(f"\n🎲 Setting random seeds to {seed} for reproducible results")
    
    # Set Python built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed (important for error injection)
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CUDA operations deterministic (may slightly reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"   ✅ Python random seed: {seed}")
    print(f"   ✅ NumPy random seed: {seed}")
    print(f"   ✅ PyTorch random seed: {seed}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA random seeds: {seed}")
        print(f"   ✅ CUDA deterministic mode: enabled")
    print("="*50)

def run_alternating_model(args):
    """Run the alternating LSTM model with the specified parameters."""
    # Set random seeds for reproducibility FIRST
    set_random_seeds(args.seed)
    
    # Setup experiment directories
    exp_dirs = setup_experiment_directories(project_dir, args.experiment)
    
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
    saved_error_generator = None  # Store error generator for later use
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
            validation_error_generator = SyntheticErrorGenerator(error_config)
            val_data_with_errors, val_error_report = inject_errors_into_dataset(
                original_val_data, validation_error_generator, f"{args.station_id}_val", water_level_cols
            )
            val_data = val_data_with_errors
            saved_error_generator = validation_error_generator  # Save for zoom plots
            
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
    print(f"All plots will be saved to experiment directory: {exp_dirs['base']}")
    
    # Training convergence plot
    plot_convergence(history, str(args.station_id), 
                     title=f"Training and Validation Loss - Station {args.station_id}",
                     output_dir=exp_dirs['visualizations'])
    
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
            synthetic_data=synthetic_data,  # Pass synthetic data to show corrupted values
            output_dir=exp_dirs['visualizations']
        )
    else:
        # Standard plot with just the validation data and predictions
        create_full_plot(
            original_val_data, 
            val_pred_df, 
            str(args.station_id), 
            config, 
            min(history['val_loss']), 
            title_suffix=viz_title_suffix,
            output_dir=exp_dirs['visualizations']
        )
    

    plot_zoom_comparison(
        actual_data=val_data['vst_raw'],
        predictions=val_pred_df['vst_raw'],
        output_dir=exp_dirs['visualizations'],
        station_id=args.station_id,
        corrupted_data=None,
        zoom_start=pd.Timestamp('2022-03-01'),
        zoom_end=pd.Timestamp('2022-05-30'),
        title_suffix="Validation Predictions - 2022",
    )

    plot_zoom_comparison(
        actual_data=val_data['vst_raw'],
        predictions=val_pred_df['vst_raw'],
        output_dir=exp_dirs['visualizations'],
        station_id=args.station_id,
        corrupted_data=None,
        zoom_start=pd.Timestamp('2023-04-01'),
        zoom_end=pd.Timestamp('2023-05-30'),
        title_suffix="Validation Predictions -2023",
    )
    # Calculate anomalies for the validation set
    print("\nGenerating anomaly detection analysis...")
    
    # Calculate anomalies between model predictions and "observed" data (which may contain synthetic errors)
    z_scores, anomalies = calculate_z_scores_mad(
        val_data['vst_raw'].values,  # "Observed" data (potentially with synthetic errors)
        val_pred_df['vst_raw'].values,  # Model predictions
        window_size=config['window_size'],
        threshold=config['threshold']  
    )
    
    # Calculate confidence levels for detected anomalies
    confidence = calculate_anomaly_confidence(z_scores, config['threshold'])
    
    # Count anomalies by confidence level
    high_conf_count = np.sum((anomalies) & (confidence == 'High'))
    med_conf_count = np.sum((anomalies) & (confidence == 'Medium'))
    low_conf_count = np.sum((anomalies) & (confidence == 'Low'))
    total_anomalies = np.sum(anomalies)
    
    print(f"Anomalies detected by model: {total_anomalies} total")
    print(f"  - High confidence: {high_conf_count}")
    print(f"  - Medium confidence: {med_conf_count}")
    print(f"  - Low confidence: {low_conf_count}")
    
    # Create output directory
    anomaly_viz_dir = exp_dirs['anomaly_detection']
    
    # PLOT 1: Anomaly detection plot (predictions vs observed data)
    anomaly_plot_title = f"Anomaly Detection - Station {args.station_id}"
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        anomaly_plot_title += f" (Synthetic Errors {args.error_multiplier}x)"
    anomaly_plot_title += f"\nDetected: {high_conf_count} High, {med_conf_count} Medium, {low_conf_count} Low confidence"
    
    anomaly_png_path, anomaly_html_path = plot_water_level_anomalies(
        test_data=val_data,  # Use modified data as "observed"
        predictions=val_pred_df['vst_raw'],
        z_scores=z_scores,
        anomalies=anomalies,
        threshold=config['threshold'],
        title=anomaly_plot_title,
        output_dir=anomaly_viz_dir,
        save_png=True,
        save_html=False,
        show_plot=False,
        filename_prefix="anomaly_detection_",
        confidence=confidence  # Add confidence levels
    )
    
    print(f"Anomaly detection plot saved to:")
    print(f"  PNG: {anomaly_png_path}")
    print(f"  HTML: {anomaly_html_path}")
    
    # PLOT 2: Error injection verification (original vs modified data) - only if errors were injected
    error_generator = None  # Will be set if errors were injected
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        print("\nGenerating error injection verification plot...")
        
        # Calculate anomalies between original and modified data to show injected errors
        error_z_scores, error_anomalies = calculate_z_scores_mad(
            original_val_data['vst_raw'].values,  # Original clean data
            val_data['vst_raw'].values,  # Modified data with synthetic errors
            window_size=config['window_size'],
            threshold=config['threshold']
        )
        
        error_confidence = calculate_anomaly_confidence(error_z_scores, config['threshold'])
        
        # Count synthetic errors by confidence
        synth_high = np.sum((error_anomalies) & (error_confidence == 'High'))
        synth_med = np.sum((error_anomalies) & (error_confidence == 'Medium'))
        synth_low = np.sum((error_anomalies) & (error_confidence == 'Low'))
        
        print(f"Synthetic errors detected: {np.sum(error_anomalies)} total")
        print(f"  - High confidence: {synth_high}")
        print(f"  - Medium confidence: {synth_med}")
        print(f"  - Low confidence: {synth_low}")
        
        # Create zoom plots for each error type if we have the error generator
        if saved_error_generator is not None:
            print("\nCreating zoom plots for individual error types...")
            create_anomaly_zoom_plots(
                val_data=val_data,
                predictions=val_pred_df['vst_raw'].values,
                z_scores=z_scores,
                anomalies=anomalies,
                confidence=confidence,
                error_generator=saved_error_generator,
                station_id=args.station_id,
                config=config,
                output_dir=anomaly_viz_dir,
                original_val_data=original_val_data  # Pass original validation data
            )
            
            # Create synthetic error zoom plots (without anomaly detection elements)
            print("\nCreating synthetic error zoom plots...")
            create_synthetic_error_zoom_plots(
                val_data=val_data,
                predictions=val_pred_df['vst_raw'].values,
                error_generator=saved_error_generator,
                station_id=args.station_id,
                output_dir=anomaly_viz_dir,
                original_val_data=original_val_data,
                model_config=config
            )
        else:
            print("\nNo error generator available for zoom plots")
    
    # COMPREHENSIVE ANOMALY DETECTION EVALUATION - only if synthetic errors were injected
    if args.error_multiplier is not None and args.error_type in ['both', 'validation'] and saved_error_generator is not None:
        from _4_anomaly_detection.comprehensive_evaluation import run_comprehensive_evaluation
        
        # Run comprehensive evaluation with all visualizations
        evaluation_results = run_comprehensive_evaluation(
            val_data=val_data,
            predictions=val_pred_df['vst_raw'].values,
            error_generator=saved_error_generator,
            station_id=args.station_id,
            config=config,
            output_dir=exp_dirs['anomaly_detection'],
            error_multiplier=args.error_multiplier,
            original_val_data=original_val_data  # Pass original validation data
        )
    else:
        print(f"\nSkipping comprehensive evaluation (no synthetic errors injected or error generator unavailable)")

    print(f"\nAll anomaly detection analysis completed!")
    
    # CREATE SYNTHETIC ERROR ZOOM PLOTS (INDEPENDENT OF ANOMALY DETECTION)
    if args.error_multiplier is not None and args.error_type in ['both', 'validation'] and saved_error_generator is not None:
        print("\nCreating synthetic error zoom plots (independent analysis)...")
        create_synthetic_error_zoom_plots(
            val_data=val_data,
            predictions=val_pred_df['vst_raw'].values,
            error_generator=saved_error_generator,
            station_id=args.station_id,
            output_dir=exp_dirs['visualizations'],  # Save to visualizations directory instead
            original_val_data=original_val_data,
            model_config=config
        )
    else:
        print("\nNo synthetic errors injected - skipping synthetic error zoom plots")

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
        print("\nValidation Metrics (against original data) - If errors injected, this is not relevant:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        '''
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
            '''
    else:
        print("\nNot enough valid data points to calculate metrics")
        metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan')}

    # Generate residual plots and other diagnostics
    from _3_lstm_model.model_diagnostics import generate_all_diagnostics
    
    # Create features DataFrame for residual analysis
    features_df = pd.DataFrame({
        'temperature': val_data['temperature'],
        'rainfall': val_data['rainfall']
    })
    
    # Generate all diagnostic plots
    diagnostic_vis_paths = generate_all_diagnostics(
        actual=val_data['vst_raw'],
        predictions=val_pred_df['vst_raw'],
        output_dir=exp_dirs['diagnostics'],
        station_id=args.station_id,
        features_df=features_df
    )
    
    return metrics

if __name__ == "__main__":
    args = parse_arguments()
    metrics = run_alternating_model(args)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {args.experiment} COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Results saved to: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}'}")
    print(f"  - Diagnostics: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'diagnostics'}")
    print(f"  - Anomaly Detection: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'anomaly_detection'}")
    print(f"  - Visualizations: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'visualizations'}")
    print(f"{'='*60}") 