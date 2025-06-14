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

from sklearn.metrics import precision_score, recall_score, f1_score

# Import local modules
from models.lstm_flagging.alternating_config import ALTERNATING_CONFIG
from shared.preprocessing.preprocessing_LSTM import DataPreprocessor
from experiments.BCEmodel.iterative_trainer2 import AlternatingTrainer
from shared.diagnostics.model_diagnostics import generate_all_diagnostics

from shared.diagnostics.model_plots import create_full_plot, plot_convergence, create_synthetic_error_zoom_plots
from shared.anomaly_detection.anomaly_visualization import (
    plot_water_level_anomalies, 
    calculate_anomaly_confidence, 
    create_anomaly_zoom_plots
)
from shared.anomaly_detection.z_score import calculate_z_scores_mad
from shared.anomaly_detection.mad_outlier import mad_outlier_flags


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run alternating LSTM model for water level forecasting')
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--error_multiplier', type=float, default=None, 
                      help='Error multiplier for synthetic errors. If not provided, no errors are injected.')
    parser.add_argument('--quick_mode', action='store_true', help='Enable quick mode with reduced data (3 years training, 1 year validation)')
    parser.add_argument('--error_type', type=str, default='both', choices=['both', 'train', 'validation', 'none'],
                      help='Which datasets to inject errors into (both, train, validation, or none)')
    parser.add_argument('--experiment', type=str, default='Iterative_forecaster2_0', help='Experiment number/name for organizing results (e.g., 0, 1, baseline, etc.)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--ground_truth_strategy', type=str, default='synthetic', choices=['synthetic', 'mad'],
                      help='Strategy for ground truth anomaly flags: synthetic (use synthetic error locations) or mad (use MAD-calculated flags). Default: synthetic')
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
    
    # Handle default error_multiplier when using synthetic flags
    if args.error_multiplier is None and args.ground_truth_strategy == 'synthetic':
        args.error_multiplier = 1.0
        print(f"\n🔧 Using synthetic flags as ground truth - setting default error_multiplier to {args.error_multiplier}")
        print("   💡 Note: When using synthetic flags, errors will always be injected for ground truth generation")
    elif args.ground_truth_strategy == 'synthetic' and args.error_multiplier is not None:
        print(f"\n🔧 Using synthetic flags as ground truth with error_multiplier {args.error_multiplier}")
    elif args.ground_truth_strategy == 'mad':
        print(f"\n📊 Using MAD-calculated flags as ground truth")
        if args.error_multiplier is not None:
            print(f"   📝 Error injection will still occur with multiplier {args.error_multiplier} for data corruption")
        else:
            print(f"   📝 No error injection configured")
    
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
    print(f"\nGround Truth Strategy: {args.ground_truth_strategy}")
    
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
    
    # Initialize variables for storing synthetic error ground truth
    train_synthetic_flags = None
    val_synthetic_flags = None
    
    # --- Inject synthetic errors BEFORE calculating anomaly flags ---
    saved_error_generator = None  # Store error generator for later use
    if args.error_multiplier is not None and args.error_type != 'none':
        from shared.synthetic.synthetic_errors import SyntheticErrorGenerator
        from shared.utils.error_utils import configure_error_params, inject_errors_into_dataset
        from config import SYNTHETIC_ERROR_PARAMS
        print(f"\nInjecting synthetic errors with multiplier {args.error_multiplier:.1f}x...")
        print(f"Error injection mode: {args.error_type}")
        error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
        water_level_cols = ['vst_raw','vst_raw_feature']
        print(f"Injecting errors into columns: {water_level_cols}")
        if args.error_type in ['both', 'train']:
            print("\nProcessing TRAINING data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            train_data_with_errors, train_error_report = inject_errors_into_dataset(
                original_train_data, error_generator, f"{args.station_id}_train", water_level_cols
            )
            train_data = train_data_with_errors
            
            # Extract synthetic error flags from the error report for training data
            if args.ground_truth_strategy == 'synthetic' and isinstance(train_error_report, dict):
                # Find the ground truth for vst_raw column
                vst_key = f"{args.station_id}_train_vst_raw"
                if vst_key in train_error_report and 'ground_truth' in train_error_report[vst_key]:
                    ground_truth_df = train_error_report[vst_key]['ground_truth']
                    if 'error' in ground_truth_df.columns:
                        train_synthetic_flags = ground_truth_df['error'].values.astype(bool)
                        print(f"Extracted {np.sum(train_synthetic_flags)} synthetic error flags from training data")
            
            if isinstance(train_error_report, dict) and 'total_errors' in train_error_report:
                print(f"Training errors injected: {train_error_report['total_errors']} errors")
                if 'error_counts' in train_error_report:
                    for error_type, count in train_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Training errors injected successfully")
        if args.error_type in ['both', 'validation']:
            print("\nProcessing VALIDATION data - injecting errors...")
            validation_error_generator = SyntheticErrorGenerator(error_config)
            val_data_with_errors, val_error_report = inject_errors_into_dataset(
                original_val_data, validation_error_generator, f"{args.station_id}_val", water_level_cols
            )
            val_data = val_data_with_errors
            saved_error_generator = validation_error_generator  # Save for zoom plots
            
            # Extract synthetic error flags from the error report for validation data
            if args.ground_truth_strategy == 'synthetic' and isinstance(val_error_report, dict):
                # Find the ground truth for vst_raw column
                vst_key = f"{args.station_id}_val_vst_raw"
                if vst_key in val_error_report and 'ground_truth' in val_error_report[vst_key]:
                    ground_truth_df = val_error_report[vst_key]['ground_truth']
                    if 'error' in ground_truth_df.columns:
                        val_synthetic_flags = ground_truth_df['error'].values.astype(bool)
                        print(f"Extracted {np.sum(val_synthetic_flags)} synthetic error flags from validation data")
            
            if isinstance(val_error_report, dict) and 'total_errors' in val_error_report:
                print(f"Validation errors injected: {val_error_report['total_errors']} errors")
                if 'error_counts' in val_error_report:
                    for error_type, count in val_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Validation errors injected successfully")
    else:
        print("\nNo synthetic errors injected.")

    # --- Generate anomaly flags based on the chosen strategy ---
    if args.ground_truth_strategy == 'synthetic':
        print("\n🎯 Using synthetic error flags as ground truth...")
        # Use synthetic flags if available, otherwise create empty flags
        if train_synthetic_flags is not None:
            train_flags = train_synthetic_flags
        else:
            train_flags = np.zeros(len(train_data), dtype=bool)
            print("⚠️  No synthetic flags for training data - using empty flags")
            
        if val_synthetic_flags is not None:
            val_flags = val_synthetic_flags
        else:
            val_flags = np.zeros(len(val_data), dtype=bool)
            print("⚠️  No synthetic flags for validation data - using empty flags")
            
        print(f"Training flags: {np.sum(train_flags)} anomalies out of {len(train_flags)} points")
        print(f"Validation flags: {np.sum(val_flags)} anomalies out of {len(val_flags)} points")
    else:
        print("\n📊 Generating ground truth anomaly flags using MAD...")
        window_size = config.get('window_size', 16)
        threshold = config.get('threshold', 3.0)
        train_flags, val_flags, *_ = mad_outlier_flags(
            train_data['vst_raw'], val_data['vst_raw'], threshold=threshold, window_size=window_size
        )
        print(f"MAD flags - Training: {np.sum(train_flags)} anomalies, Validation: {np.sum(val_flags)} anomalies")
    
    # Make sure the model's week_steps matches our config
    trainer.model.week_steps = config['week_steps']
    
    print("\nTraining model...")
    training_results = trainer.train(
        train_data, val_data, config['epochs'], config['batch_size'],
        train_anomaly_flags=train_flags, val_anomaly_flags=val_flags
    )
    
    # Unpack the results - handle both old (3 values) and new (6 values) return formats
    if len(training_results) == 6:
        history, val_predictions, val_targets, val_z_scores, val_anomaly_flags, val_anomaly_probs = training_results
        print("Using integrated anomaly detection results")
    else:
        history, val_predictions, val_targets = training_results
        val_z_scores = val_anomaly_flags = val_anomaly_probs = None
        print("Using standard training results (no integrated anomaly detection)")
    
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

    # Calculate z-scores and anomalies for the validation set
    print("\nGenerating anomaly detection plot for validation set...")
    window_size = config.get('window_size', 16)
    threshold = config.get('threshold', 3.0)
    z_scores, anomalies = calculate_z_scores_mad(
        val_data['vst_raw'].values,  # Observed (potentially with errors)
        val_pred_df['vst_raw'].values,  # Model predictions
        window_size=window_size,
        threshold=threshold
    )

    # Calculate metrics for anomaly detection

    precision = precision_score(val_flags, anomalies)
    recall = recall_score(val_flags, anomalies)
    f1 = f1_score(val_flags, anomalies)

    print("\nAnomaly Detection Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save HTML and PNG plot with ground truth flags
    anomaly_viz_dir = exp_dirs['anomaly_detection']
    plot_water_level_anomalies(
        test_data=val_data,
        predictions=val_pred_df['vst_raw'].values,
        z_scores=z_scores,
        anomalies=anomalies,
        threshold=threshold,
        title=f"Anomaly Detection - Station {args.station_id}",
        output_dir=anomaly_viz_dir,
        save_png=True,
        save_html=True,
        show_plot=False,
        ground_truth_flags=val_flags
    )

    # Create synthetic error zoom plots if errors were injected
    if saved_error_generator is not None:
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
        print("\nNo error generator available for synthetic error zoom plots")

    # Calculate metrics on validation data instead of test data
    from shared.utils.pipeline_utils import calculate_performance_metrics
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
                '''
                # Calculate correction percentage
                if error_metrics['rmse'] > 0:
                    correction_rmse = (1 - metrics['rmse'] / error_metrics['rmse']) * 100
                    print(f"    RMSE improvement: {correction_rmse:.2f}%")
                
                if error_metrics['mae'] > 0:
                    correction_mae = (1 - metrics['mae'] / error_metrics['mae']) * 100
                    print(f"    MAE improvement: {correction_mae:.2f}%")
                
                r2_improvement = metrics['r2'] - error_metrics['r2']
                print(f"    R² improvement: {r2_improvement:.4f}")
                '''   
            except Exception as e:
                print(f"\nError during error correction analysis: {str(e)}")
                print("Continuing with model evaluation...")

    else:
        print("\nNot enough valid data points to calculate metrics")
        metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan')}


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