import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random
from scipy.signal import savgol_filter

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir  # Point to Project_Code - CORRECT instead of parent.parent
sys.path.append(str(project_dir))

# Import local modules
from models.lstm_flagging.alternating_config import ALTERNATING_CONFIG
from models.lstm_flagging.simple_anomaly_detector import SimpleAnomalyDetector
from models.lstm_flagging.preprocessing_LSTM2 import DataPreprocessor
from shared.diagnostics.model_plots import create_full_plot, plot_convergence
from shared.anomaly_detection.comprehensive_evaluation import (
    run_single_threshold_anomaly_detection
)
from shared.anomaly_detection.z_score import calculate_z_scores_mad
from shared.anomaly_detection.mad_outlier import mad_outlier_flags
from shared.anomaly_detection.anomaly_visualization import (
    plot_water_level_anomalies, 
    calculate_anomaly_confidence, 
    create_anomaly_zoom_plots,
    create_simple_anomaly_zoom_plots
)

from shared.anomaly_detection.mad_outlier import mad_outlier_flags
from models.lstm_flagging.alternating_trainer import AlternatingTrainer
from models.lstm_flagging.alternating_forecast_model import AlternatingForecastModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run flagging LSTM model for anomaly-resistant water level forecasting')
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--error_multiplier', type=float, default=1.0, 
                      help='Error multiplier for synthetic errors')
    parser.add_argument('--quick_mode', action='store_true', help='Enable quick mode with reduced data')
    parser.add_argument('--full_dataset', action='store_true', help='Enable full dataset mode with maximum training data')
    parser.add_argument('--experiment', type=str, default='flagging_test', help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_perfect_flags', action='store_true', default=True, help='Use perfect anomaly flags')
    parser.add_argument('--anomaly_weight', type=float, default=0.3, help='Weight for anomalous periods in loss')
    parser.add_argument('--flag_method', type=str, choices=['synthetic', 'mad'], default='synthetic',
                      help='Method for generating true anomaly flags: synthetic (from error injection) or mad (MAD outlier detection)')
    parser.add_argument('--use_test_data', action='store_true',
                      help='Use test data for predictions, plots, and metrics (training still uses train/val data)')
    return parser.parse_args()

def setup_experiment_directories(project_dir, experiment_name):
    """Create experiment-specific directory structure."""
    exp_dir = Path(project_dir) / "results" / "Iterative model results" / f"experiment_{experiment_name}"
    
    directories = {
        'base': exp_dir,
        'diagnostics': exp_dir / "diagnostics",
        'visualizations': exp_dir / "visualizations",
        'anomaly_detection': exp_dir / "anomaly_detection",
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏷️ ANOMALY FLAGGING EXPERIMENT")
    print(f"Experiment directories created under: {exp_dir}")
    return directories

def set_random_seeds(seed):
    """Set random seeds for reproducible results."""
    print(f"\n🎲 Setting random seeds to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_flagging_model(args):
    """Run the flagging LSTM model with anomaly resistance."""
    
    # Set random seeds first
    set_random_seeds(args.seed)
    
    # Setup experiment directories
    exp_dirs = setup_experiment_directories(project_dir, args.experiment)
    
    # Configuration
    config = ALTERNATING_CONFIG.copy()
    config.update({
        'quick_mode': args.quick_mode,
        'full_dataset_mode': args.full_dataset,
        'use_perfect_flags': args.use_perfect_flags,
        'anomaly_weight': args.anomaly_weight,
        'use_weighted_loss': True,
        'use_anomaly_flags': True,
        'experiment': args.experiment,  # Store experiment name in config
    })
     
    print("\n🏷️ Flagging Model Configuration:")
    print(f"  use_anomaly_flags: {config['use_anomaly_flags']}")
    print(f"  use_weighted_loss: {config['use_weighted_loss']}")
    print(f"  anomaly_weight: {config['anomaly_weight']}")
    print(f"  use_perfect_flags: {config['use_perfect_flags']}")
    print(f"  flag_method: {args.flag_method}")
    if args.flag_method == 'mad':
        print(f"  mad_threshold: {config.get('mad_threshold', 3.0)}")
        print(f"  mad_window: {config.get('mad_window', 100)}")
    print(f"  quick_mode: {config['quick_mode']}")
    print(f"  full_dataset_mode: {config['full_dataset_mode']}")
    print(f"  epochs: {config['epochs']}")
    
    print("\n📊 Evaluation Configuration:")
    print(f"Training data: train + validation sets")
    if args.use_test_data:
        print(f"Evaluation data: test set (predictions, plots, metrics)")
    else:
        print(f"Evaluation data: validation set (predictions, plots, metrics)")
    
    # Print test data usage configuration
    if args.use_test_data:
        print(f"  Using TEST data for predictions, plots, and metrics")
    else:
        print(f"  Using VALIDATION data for predictions, plots, and metrics")
    
    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor(config)
    trainer = AlternatingTrainer(config, preprocessor)
    
    # Load data
    print(f"\nLoading data for station {args.station_id}...")
    train_data, val_data, test_data = trainer.load_data(project_dir, args.station_id)
    
    # Extract EDT data for plotting reference
    print("Extracting EDT reference data...")
    data_dir = project_dir / "data_utils" / "Sample data"
    data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
    station_data = data.get(args.station_id)
    
    if station_data and 'vst_edt' in station_data:
        # Extract EDT data for the validation period
        edt_data = station_data['vst_edt']['vst_edt']
        edt_val_data = edt_data[edt_data.index.isin(val_data.index)]
        print(f"Extracted {len(edt_val_data)} EDT reference points for validation period")
    else:
        edt_val_data = None
        print("No EDT data available for this station")
    
    # Store original data
    original_train_data = train_data.copy()
    original_val_data = val_data.copy()
    original_test_data = test_data.copy()  # Add original test data
    
    # Inject synthetic errors
    print(f"\nInjecting synthetic errors with multiplier {args.error_multiplier}...")
    from shared.synthetic.synthetic_errors import SyntheticErrorGenerator
    from shared.utils.error_utils import configure_error_params, inject_errors_into_dataset
    from synthetic_error_config import SYNTHETIC_ERROR_PARAMS
    
    error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
    water_level_cols = ['vst_raw', 'vst_raw_feature']
    
    # Process training data
    train_error_generator = SyntheticErrorGenerator(error_config)
    train_data_with_errors, train_error_report = inject_errors_into_dataset(
        original_train_data, train_error_generator, f"{args.station_id}_train", water_level_cols
    )
    
    # Process validation data  
    val_error_generator = SyntheticErrorGenerator(error_config)
    val_data_with_errors, val_error_report = inject_errors_into_dataset(
        original_val_data, val_error_generator, f"{args.station_id}_val", water_level_cols
    )
    
    # Process test data for anomaly detection (inject synthetic errors into target variable only)
    print(f"\nInjecting synthetic errors into TEST data for anomaly detection...")
    test_error_generator = SyntheticErrorGenerator(error_config)
    test_data_with_errors, test_error_report = inject_errors_into_dataset(
        original_test_data, test_error_generator, f"{args.station_id}_anomaly_test", ['vst_raw']  # Only target variable
    )
    
    # Add ALL features AFTER error injection so they reflect corrupted data
    print(f"\n🔧 Adding all features to corrupted data...")
    train_data_with_features = trainer.add_all_features(train_data_with_errors)
    val_data_with_features = trainer.add_all_features(val_data_with_errors)
    test_data_with_features = trainer.add_all_features(test_data_with_errors)  # Add test data features
    
    # Create anomaly detector
    detector = SimpleAnomalyDetector(
        threshold=config.get('anomaly_detection_threshold', 3.0),
        window_size=config.get('anomaly_detection_window', 16)
    )
    
    # Add anomaly flags to data based on chosen method
    if args.flag_method == 'synthetic':
        print(f"\n🎯 Using SYNTHETIC ERROR FLAGS as true anomaly flags")
        
        if config['use_perfect_flags']:
            print("   Using PERFECT flags from known synthetic error locations")
            
            # Training flags
            train_flags = detector.create_perfect_flags(
                train_error_generator, len(train_data_with_features), train_data_with_features.index
            )
            train_data_flagged = detector.add_anomaly_flags_to_dataframe(
                train_data_with_features, train_flags, config['anomaly_flag_column']
            )
            
            # Validation flags
            val_flags = detector.create_perfect_flags(
                val_error_generator, len(val_data_with_features), val_data_with_features.index
            )
            val_data_flagged = detector.add_anomaly_flags_to_dataframe(
                val_data_with_features, val_flags, config['anomaly_flag_column']
            )
            
        else:
            print("   Using AUTOMATIC detection comparing corrupted vs original data")
            
            # Automatic detection
            train_flags, _ = detector.detect_anomalies(
                train_data_with_features['vst_raw'], original_train_data['vst_raw']
            )
            train_data_flagged = detector.add_anomaly_flags_to_dataframe(
                train_data_with_features, train_flags, config['anomaly_flag_column']
            )
            
            val_flags, _ = detector.detect_anomalies(
                val_data_with_features['vst_raw'], original_val_data['vst_raw']
            )
            val_data_flagged = detector.add_anomaly_flags_to_dataframe(
                val_data_with_features, val_flags, config['anomaly_flag_column']
            )
    
    elif args.flag_method == 'mad':
        print(f"\n🔍 Using MAD OUTLIER DETECTION as true anomaly flags")
        print(f"   MAD threshold: {config.get('mad_threshold', 3.0)}")
        print(f"   MAD window size: {config.get('mad_window', 100)}")
        
        # For MAD method, we use the original clean data to identify anomalies
        # This simulates a scenario where we have clean reference data
        
        # Apply MAD outlier detection to original data
        train_flags_mad, val_flags_mad, _, _, _, _ = mad_outlier_flags(
            train_series=original_train_data['vst_raw'],
            val_series=original_val_data['vst_raw'],
            threshold=config.get('mad_threshold', 3.0),
            window_size=config.get('mad_window', 100)
        )
        
        # Convert to boolean arrays
        train_flags = train_flags_mad.astype(bool)
        val_flags = val_flags_mad.astype(bool)
        
        # Add flags to the corrupted data (which the model will train on)
        train_data_flagged = detector.add_anomaly_flags_to_dataframe(
            train_data_with_features, train_flags, config['anomaly_flag_column']
        )
        
        val_data_flagged = detector.add_anomaly_flags_to_dataframe(
            val_data_with_features, val_flags, config['anomaly_flag_column']
        )
        
        print(f"   MAD detected {np.sum(train_flags)} anomalies in training data")
        print(f"   MAD detected {np.sum(val_flags)} anomalies in validation data")
    
    else:
        raise ValueError(f"Unknown flag_method: {args.flag_method}")
    
    print(f"\n🏷️ Training model with anomaly flags...")
    print(f"   Flag method: {args.flag_method}")
    print(f"   Training anomalies: {np.sum(train_flags)} / {len(train_flags)}")
    print(f"   Validation anomalies: {np.sum(val_flags)} / {len(val_flags)}")
    
    # Reinitialize model with correct input size for anomaly flags
    trainer.reinitialize_model_for_anomaly_flags(train_data_flagged)
    
    # Train the model
    history, val_predictions, val_targets = trainer.train(
        train_data_flagged, val_data_flagged, config['epochs'], config['batch_size']
    )
    
    # Determine which dataset to use for evaluation
    if args.use_test_data:
        print("\nUsing TEST data for predictions and evaluation...")
        eval_data = original_test_data
        eval_data_flagged = test_data_with_features  # Use test data with features but no flags for prediction
        
        # For consistent synthetic anomaly approach: use test data WITH synthetic errors for prediction
        # This ensures the model gets both the corrupted data AND the correct anomaly flags
        print("Using test data WITH synthetic errors for consistent anomaly flagging...")
        test_data_for_prediction = trainer.add_all_features(test_data_with_errors)
        
        # Use synthetic error locations as perfect anomaly flags
        if args.flag_method == 'synthetic' and 'test_error_generator' in locals():
            print("Adding perfect synthetic error flags to test data for prediction")
            test_flags = detector.create_perfect_flags(
                test_error_generator, len(test_data_for_prediction), test_data_for_prediction.index
            )
            test_data_for_prediction[config['anomaly_flag_column']] = test_flags
            print(f"Added {np.sum(test_flags)} synthetic anomaly flags to test data")
        else:
            # Fallback to MAD detection if synthetic flags not available
            print("Fallback: Using MAD detection for test anomaly flags")
            from shared.anomaly_detection.mad_outlier import mad_outlier_flags
            test_flags, _, _, _, _, _ = mad_outlier_flags(
                train_series=original_train_data['vst_raw'],
                val_series=test_data_for_prediction['vst_raw'],
                threshold=config.get('mad_threshold', 3.0),
                window_size=config.get('mad_window', 100)
            )
            test_data_for_prediction[config['anomaly_flag_column']] = test_flags.astype(bool)
            print(f"MAD detected {np.sum(test_flags)} anomalies in test data for flagging")
        
        eval_predictions, _, _ = trainer.predict(test_data_for_prediction)  # Use test data with synthetic errors and flags
        dataset_name = "Test Set"
        eval_plot_title = f"Flagging Model Results on Test Set (Weight: {args.anomaly_weight})"
    else:
        print("\nUsing VALIDATION data for predictions and evaluation...")
        eval_data = original_val_data
        eval_data_flagged = val_data_flagged
        eval_predictions = val_predictions
        dataset_name = "Validation Set"
        eval_plot_title = f"Flagging Model Results on Validation Set (Weight: {args.anomaly_weight})"
    
    # Convert predictions back to original scale
    print("\nConverting predictions back to original scale...")
    
    # eval_predictions is already a numpy array from the predict method, no need to call .numpy()
    if isinstance(eval_predictions, np.ndarray):
        val_predictions_np = eval_predictions
    else:
        val_predictions_np = eval_predictions.numpy()  # Only if it's a tensor
    
    print(f"Predictions shape: {val_predictions_np.shape}")
    
    # The predictions are already unscaled from the predict method, so we can use them directly
    if val_predictions_np.ndim == 3:
        # Shape: [batch_size, seq_len, features] -> flatten to [seq_len]
        val_predictions_original = val_predictions_np.reshape(-1)
    elif val_predictions_np.ndim == 2:
        # Shape: [seq_len, features] -> flatten to [seq_len]
        val_predictions_original = val_predictions_np.flatten()
    elif val_predictions_np.ndim == 1:
        # Already 1D
        val_predictions_original = val_predictions_np
    else:
        raise ValueError(f"Unexpected prediction shape: {val_predictions_np.shape}")
    
    print(f"Flattened predictions shape: {val_predictions_original.shape}")
    
    # EXPERIMENT 3: Postprocessing smoothing to reduce oscillations
    print("Applying postprocessing smoothing filter...")
    
    def smooth_predictions(predictions, method='savgol', **kwargs):
        """
        Apply smoothing filter to reduce oscillations in predictions.
        
        Args:
            predictions: Array of predictions to smooth
            method: 'savgol', 'moving_avg', or 'exponential'
            **kwargs: Parameters for the smoothing method
        """
        if method == 'savgol':
            window_length = kwargs.get('window_length', 7)
            polyorder = kwargs.get('polyorder', 2)
            # Ensure window_length is odd and not larger than data
            window_length = min(window_length, len(predictions))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < polyorder + 1:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
            return savgol_filter(predictions, window_length, polyorder)
        
        elif method == 'moving_avg':
            window = kwargs.get('window', 5)
            return pd.Series(predictions).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        elif method == 'exponential':
            alpha = kwargs.get('alpha', 0.3)
            return pd.Series(predictions).ewm(alpha=alpha).mean().values
        
        else:
            return predictions
    
    # Apply smoothing (you can experiment with different methods)
    val_predictions_smoothed = smooth_predictions(
        val_predictions_original, 
        method='savgol', 
        window_length=7, 
        polyorder=2
    )
    
    print(f"Original predictions range: {val_predictions_original.min():.2f} to {val_predictions_original.max():.2f}")
    print(f"Smoothed predictions range: {val_predictions_smoothed.min():.2f} to {val_predictions_smoothed.max():.2f}")
    
    # Create prediction DataFrame with both original and smoothed predictions
    eval_pred_df = pd.DataFrame({
        'vst_raw': val_predictions_smoothed,  # Use smoothed predictions
        'vst_raw_original': val_predictions_original,  # Keep original for comparison
    }, index=eval_data.index[:len(val_predictions_original)])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Training convergence
    plot_convergence(history, str(args.station_id), 
                     title=f"Flagging Model Training - Station {args.station_id}",
                     output_dir=exp_dirs['visualizations'])
    
    # Main prediction plot
    synthetic_data = {
        'data': eval_data_flagged.copy(),
        'error_periods': []
    }
    
    create_full_plot(
        eval_data,  # Original data
        eval_pred_df,        # Predictions
        str(args.station_id), 
        config, 
        min(history['val_loss']), 
        title_suffix=eval_plot_title,
        synthetic_data=synthetic_data, 
        output_dir=exp_dirs['visualizations']
    )
        # Anomaly detection analysis
    print("\nGenerating anomaly detection analysis...")
    
    z_scores, anomalies = calculate_z_scores_mad(
        eval_data_flagged['vst_raw'].values,
        eval_pred_df['vst_raw'].values,
        window_size=config['window_size'],
        threshold=config['threshold']  
    )
    
    confidence = calculate_anomaly_confidence(z_scores, config['threshold'])
    
    high_conf = np.sum((anomalies) & (confidence == 'High'))
    med_conf = np.sum((anomalies) & (confidence == 'Medium'))
    low_conf = np.sum((anomalies) & (confidence == 'Low'))
    
    print(f"Anomalies detected: {np.sum(anomalies)} total")
    print(f"  - High confidence: {high_conf}")
    print(f"  - Medium confidence: {med_conf}")
    print(f"  - Low confidence: {low_conf}")
    
    # Create anomaly plot with original data
    anomaly_plot_title = f"Flagging Model Anomaly Detection - Station {args.station_id}"
    anomaly_plot_title += f"\nWeight: {args.anomaly_weight}, Detected: {high_conf}H, {med_conf}M, {low_conf}L"
    
    plot_water_level_anomalies(
        test_data=eval_data_flagged,
        predictions=eval_pred_df['vst_raw'],
        z_scores=z_scores,
        anomalies=anomalies,
        threshold=config['threshold'],
        title=anomaly_plot_title,
        output_dir=exp_dirs['anomaly_detection'],
        save_png=True,
        save_html=True,
        show_plot=False,
        filename_prefix="flagging_anomaly_detection_",
        confidence=confidence,
        original_data=eval_data,  # Add original clean data
        edt_data=edt_val_data  # Add EDT reference data
    )
    
    # Create zoom plots for each error type
    print("\nCreating zoom plots for individual error types...")
    create_anomaly_zoom_plots(
        val_data=eval_data_flagged,
        predictions=eval_pred_df['vst_raw'].values,
        z_scores=z_scores,
        anomalies=anomalies,
        confidence=confidence,
        error_generator=val_error_generator,
        station_id=args.station_id,
        config=config,
        output_dir=exp_dirs['anomaly_detection'],
        original_val_data=eval_data  # Pass original validation data
    )
    
    # Create simplified zoom plots (without detection markers)
    print("\nCreating simplified zoom plots for model behavior analysis...")
    create_simple_anomaly_zoom_plots(
        val_data=eval_data_flagged,
        predictions=eval_pred_df['vst_raw'].values,
        error_generator=val_error_generator,
        station_id=args.station_id,
        output_dir=exp_dirs['anomaly_detection'],
        original_val_data=eval_data
    )

    # Comprehensive Anomaly Detection Framework
    print("\n" + "="*60)
    print("COMPREHENSIVE ANOMALY DETECTION")
    print("="*60)
    print("🎯 Evaluating supervised model's anomaly detection capability on TEST data")
    
    # Convert error reports to stations_results format for ground truth
    print("Converting test error reports to ground truth format...")
    stations_results = {}
    
    # Convert test error report to proper format for anomaly detection
    for key, report_data in test_error_report.items():
        if 'vst_raw' in key:  # Target variable
            stations_results[key] = {
                'modified_data': test_data_with_errors,  # Test data with synthetic errors
                'ground_truth': report_data.get('ground_truth'),  # Add missing ground_truth field
                'error_periods': report_data.get('error_periods', []),
                'original_data': original_test_data
            }
    
    # Use unscaled test predictions for anomaly detection
    test_predictions_unscaled = val_predictions_smoothed  # Already unscaled
    
    print(f"📊 Anomaly detection configuration:")
    print(f"   Threshold: {config['threshold']}")
    print(f"   Window size: {config['window_size']}")
    print(f"   Using unscaled test predictions: shape {test_predictions_unscaled.shape}")
    print(f"   Prediction range: {np.nanmin(test_predictions_unscaled):.1f} to {np.nanmax(test_predictions_unscaled):.1f} mm")
    
    # Ensure predictions match test data length
    if len(test_predictions_unscaled) != len(eval_data):
        print(f"⚠️  Length mismatch: predictions={len(test_predictions_unscaled)}, data={len(eval_data)}")
        if len(test_predictions_unscaled) > len(eval_data):
            test_predictions_unscaled = test_predictions_unscaled[:len(eval_data)]
            print(f"   Truncated predictions to {len(test_predictions_unscaled)}")
        else:
            padding = np.full(len(eval_data) - len(test_predictions_unscaled), np.nan)
            test_predictions_unscaled = np.concatenate([test_predictions_unscaled, padding])
            print(f"   Padded predictions to {len(test_predictions_unscaled)}")
    
    # DEBUG: Print detailed test set information for supervised model
    print(f"\n🔍 SUPERVISED MODEL TEST SET DEBUG INFO:")
    print(f"   Test data length: {len(eval_data)}")
    print(f"   Test data period: {eval_data.index[0]} to {eval_data.index[-1]}")
    print(f"   Predictions length: {len(test_predictions_unscaled)}")
    print(f"   Non-NaN predictions: {np.sum(~np.isnan(test_predictions_unscaled))}")
    
    # Show time span
    time_span = eval_data.index[-1] - eval_data.index[0]
    print(f"   Time span: {time_span}")
    if len(eval_data) > 0:
        points_per_day = len(eval_data) / time_span.days if time_span.days > 0 else 0
        print(f"   Approximate points per day: {points_per_day:.1f}")
    
    # Check if we have ground truth data
    if not stations_results:
        print("❌ No ground truth data available for anomaly detection.")
        anomaly_results = {'error': 'No ground truth data available'}
    else:
        # Run comprehensive anomaly detection on test data
        anomaly_results = run_single_threshold_anomaly_detection(
            val_data=test_data_with_errors,  # Use TEST data with synthetic errors
            predictions=test_predictions_unscaled,  # Use unscaled test predictions
            stations_results=stations_results,  # Contains ground truth from test data
            station_id=args.station_id,
            config={
                'threshold': config['threshold'],
                'window_size': config['window_size'],
                'model_type': 'predictions'  # Time series predictions
            },
            output_dir=exp_dirs['anomaly_detection'],  # Use anomaly detection directory
            original_val_data=eval_data,  # Pass original test data for comparison
            filename_prefix=f"supervised_anomaly_detection_{args.flag_method}_",
            dataset_type='test'  # Specify that we're using test data
        )
        
        # Check if anomaly detection succeeded
        if 'error' in anomaly_results:
            print(f"❌ Anomaly detection failed: {anomaly_results['error']}")
            anomaly_results = {}
        else:
            # Add debug information about results
            print(f"\n🔍 SUPERVISED MODEL ANOMALY DETECTION RESULTS:")
            if 'confusion_metrics' in anomaly_results:
                cm = anomaly_results['confusion_metrics']
                print(f"   F1-Score: {cm['f1_score']:.4f}")
                print(f"   Precision: {cm['precision']:.4f}")
                print(f"   Recall: {cm['recall']:.4f}")
                print(f"   Total anomalies detected: {cm['total_anomalies_pred']}")
                print(f"   Total ground truth anomalies: {cm['total_anomalies_true']}")
                print(f"   True Positives: {cm['true_positives']}")
                print(f"   False Positives: {cm['false_positives']}")
                print(f"   True Negatives: {cm['true_negatives']}")
                print(f"   False Negatives: {cm['false_negatives']}")
                
                # Check if anomaly rate seems reasonable
                if cm['total_anomalies_pred'] > len(eval_data) * 0.1:  # More than 10%
                    print(f"   ⚠️ WARNING: Detection rate seems very high ({cm['total_anomalies_pred']/len(eval_data)*100:.2f}%)")
                    print(f"   This might indicate overly sensitive detection or high error injection")
    
    # Generate residual plots and other diagnostics
    print("\nGenerating diagnostic plots...")
    from shared.diagnostics.model_diagnostics import generate_all_diagnostics
    
    # Create features DataFrame for residual analysis
    features_df = pd.DataFrame({
        'temperature': eval_data_flagged['temperature'],
        'rainfall': eval_data_flagged['rainfall']
    })
    
    # Generate all diagnostic plots
    diagnostic_vis_paths = generate_all_diagnostics(
        actual=eval_data['vst_raw'],  # Use original test data as reference
        predictions=eval_pred_df['vst_raw'],
        output_dir=exp_dirs['diagnostics'],
        station_id=args.station_id,
        features_df=features_df
    )
    
    # Calculate metrics
    from shared.utils.pipeline_utils import calculate_performance_metrics
    valid_mask = ~np.isnan(eval_data['vst_raw'].values)  # Use test data instead of validation
    pred_mask = ~np.isnan(val_predictions_original)  # Use test predictions
    combined_mask = valid_mask[:len(pred_mask)] & pred_mask
    
    if np.sum(combined_mask) > 0:
        # Metrics against original clean test data
        metrics = calculate_performance_metrics(
            eval_data['vst_raw'].values[:len(pred_mask)],  # Use test data
            val_predictions_original,  # Use test predictions
            combined_mask
        )
        
        # Metrics against corrupted test data (for comparison)
        corrupted_test_metrics = calculate_performance_metrics(
            eval_data_flagged['vst_raw'].values[:len(pred_mask)] if args.use_test_data else test_data_with_errors['vst_raw'].values[:len(pred_mask)],  # Use appropriate data with errors
            val_predictions_original,  # Use test predictions
            combined_mask
        )
        
        print(f"\n{dataset_name} Metrics (vs corrupted data):")  # Updated description
        for metric, value in corrupted_test_metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        print(f"\n📊 FLAGGING MODEL RESULTS:")
        print(f"{dataset_name} Metrics (vs original clean data):")  # Updated description
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        # Add anomaly detection metrics to results if available
        if anomaly_results and 'confusion_metrics' in anomaly_results:
            metrics['anomaly_detection'] = anomaly_results['confusion_metrics']
        
        return metrics
    else:
        print("Not enough valid data for metrics")
        return {}

if __name__ == "__main__":
    args = parse_arguments()
    metrics = run_flagging_model(args)
    
    print(f"\n{'='*70}")
    print(f"🏷️ FLAGGING MODEL EXPERIMENT {args.experiment} COMPLETED!")
    print(f"{'='*70}")
    print(f"Flag method: {args.flag_method}")
    print(f"Anomaly weight: {args.anomaly_weight}")
    print(f"Perfect flags: {args.use_perfect_flags}")
    print(f"Results saved to experiment_{args.experiment}")
    print(f"{'='*70}") 