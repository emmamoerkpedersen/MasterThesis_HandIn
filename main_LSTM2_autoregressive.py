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
        print(f"  mad_threshold: {config['mad_threshold']}")
        print(f"  mad_window: {config['mad_window']}")
    print(f"  quick_mode: {config['quick_mode']}")
    print(f"  full_dataset_mode: {config['full_dataset_mode']}")

    print(f"  epochs: {config['epochs']}")
    
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
    
    # Add ALL features AFTER error injection so they reflect corrupted data
    print(f"\n🔧 Adding all features to corrupted data...")
    train_data_with_features = trainer.add_all_features(train_data_with_errors)
    val_data_with_features = trainer.add_all_features(val_data_with_errors)
    
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
        print(f"   MAD threshold: {config['mad_threshold']}")
        print(f"   MAD window size: {config['mad_window']}")
        
        # For MAD method, we use the original clean data to identify anomalies
        # This simulates a scenario where we have clean reference data
        
        # Apply MAD outlier detection to original data
        train_flags_mad, val_flags_mad, _, _, _, _ = mad_outlier_flags(
            train_series=original_train_data['vst_raw'],
            val_series=original_val_data['vst_raw'],
            threshold=config['mad_threshold'],
            window_size=config['mad_window']
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
    
    # Convert predictions back to original scale
    print("\nConverting predictions back to original scale...")
    val_predictions_np = val_predictions.numpy()
    
    if val_predictions_np.ndim == 3:
        val_predictions_np = val_predictions_np.reshape(val_predictions_np.shape[0] * val_predictions_np.shape[1], -1)
    elif val_predictions_np.ndim == 2:
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    elif val_predictions_np.ndim == 1:
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    
    val_predictions_original = trainer.target_scaler.inverse_transform(val_predictions_np).flatten()
    
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
    val_pred_df = pd.DataFrame({
        'vst_raw': val_predictions_smoothed,  # Use smoothed predictions
        'vst_raw_original': val_predictions_original,  # Keep original for comparison
    }, index=val_data_flagged.index[:len(val_predictions_original)])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Training convergence
    plot_convergence(history, str(args.station_id), 
                     title=f"Flagging Model Training - Station {args.station_id}",
                     output_dir=exp_dirs['visualizations'])
    
    # Main prediction plot
    viz_title = f"Flagging Model Results (Weight: {args.anomaly_weight})"
    synthetic_data = {
        'data': val_data_flagged.copy(),
        'error_periods': []
    }
    
    create_full_plot(
        original_val_data,  # Original data
        val_pred_df,        # Predictions
        str(args.station_id), 
        config, 
        min(history['val_loss']), 
        title_suffix=viz_title,
        synthetic_data=synthetic_data, 
        output_dir=exp_dirs['visualizations']
    )
        # Anomaly detection analysis
    print("\nGenerating anomaly detection analysis...")
    
    z_scores, anomalies = calculate_z_scores_mad(
        val_data_flagged['vst_raw'].values,
        val_pred_df['vst_raw'].values,
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
        test_data=val_data_flagged,
        predictions=val_pred_df['vst_raw'],
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
        original_data=original_val_data,  # Add original clean data
        edt_data=edt_val_data  # Add EDT reference data
    )
    
    # Create zoom plots for each error type
    print("\nCreating zoom plots for individual error types...")
    create_anomaly_zoom_plots(
        val_data=val_data_flagged,
        predictions=val_pred_df['vst_raw'].values,
        z_scores=z_scores,
        anomalies=anomalies,
        confidence=confidence,
        error_generator=val_error_generator,
        station_id=args.station_id,
        config=config,
        output_dir=exp_dirs['anomaly_detection'],
        original_val_data=original_val_data  # Pass original validation data
    )
    
    # Create simplified zoom plots (without detection markers)
    print("\nCreating simplified zoom plots for model behavior analysis...")
    create_simple_anomaly_zoom_plots(
        val_data=val_data_flagged,
        predictions=val_pred_df['vst_raw'].values,
        error_generator=val_error_generator,
        station_id=args.station_id,
        output_dir=exp_dirs['anomaly_detection'],
        original_val_data=original_val_data
    )

    # Comprehensive Anomaly Detection Framework
    print("\n" + "="*60)
    print("COMPREHENSIVE ANOMALY DETECTION")
    print("="*60)
    print("🎯 Evaluating supervised model's anomaly detection capability")
    
    # Convert error reports to stations_results format for ground truth
    print("Converting error reports to ground truth format...")
    stations_results = {}
    
    # Convert validation error report to proper format
    for key, report_data in val_error_report.items():
        if 'vst_raw' in key:  # Target variable
            stations_results[key] = {
                'modified_data': val_data_with_errors,  # Data with synthetic errors
                'ground_truth': report_data.get('ground_truth'),  # Add missing ground_truth field
                'error_periods': report_data.get('error_periods', []),
                'original_data': original_val_data
            }
    
    # Use unscaled predictions for anomaly detection
    val_predictions_unscaled = val_predictions_smoothed  # Already unscaled
    
    print(f"📊 Anomaly detection configuration:")
    print(f"   Threshold: {config['threshold']}")
    print(f"   Window size: {config['window_size']}")
    print(f"   Using unscaled predictions: shape {val_predictions_unscaled.shape}")
    print(f"   Prediction range: {np.nanmin(val_predictions_unscaled):.1f} to {np.nanmax(val_predictions_unscaled):.1f} mm")
    
    # Ensure predictions match validation data length
    if len(val_predictions_unscaled) != len(original_val_data):
        print(f"⚠️  Length mismatch: predictions={len(val_predictions_unscaled)}, data={len(original_val_data)}")
        if len(val_predictions_unscaled) > len(original_val_data):
            val_predictions_unscaled = val_predictions_unscaled[:len(original_val_data)]
            print(f"   Truncated predictions to {len(val_predictions_unscaled)}")
        else:
            padding = np.full(len(original_val_data) - len(val_predictions_unscaled), np.nan)
            val_predictions_unscaled = np.concatenate([val_predictions_unscaled, padding])
            print(f"   Padded predictions to {len(val_predictions_unscaled)}")
    
    # DEBUG: Print detailed validation set information for supervised model
    print(f"\n🔍 SUPERVISED MODEL VALIDATION SET DEBUG INFO:")
    print(f"   Validation data length: {len(original_val_data)}")
    print(f"   Validation data period: {original_val_data.index[0]} to {original_val_data.index[-1]}")
    print(f"   Predictions length: {len(val_predictions_unscaled)}")
    print(f"   Non-NaN predictions: {np.sum(~np.isnan(val_predictions_unscaled))}")
    
    # Show time span
    time_span = original_val_data.index[-1] - original_val_data.index[0]
    print(f"   Time span: {time_span}")
    if len(original_val_data) > 0:
        points_per_day = len(original_val_data) / time_span.days if time_span.days > 0 else 0
        print(f"   Approximate points per day: {points_per_day:.1f}")
    
    # Check if we have ground truth data
    if not stations_results:
        print("❌ No ground truth data available for anomaly detection.")
        anomaly_results = {'error': 'No ground truth data available'}
    else:
        # Run comprehensive anomaly detection
        anomaly_results = run_single_threshold_anomaly_detection(
            val_data=val_data_with_errors,  # Use data with synthetic errors
            predictions=val_predictions_unscaled,  # Use unscaled predictions
            stations_results=stations_results,  # Contains ground truth
            station_id=args.station_id,
            config={
                'threshold': config['threshold'],
                'window_size': config['window_size'],
                'model_type': 'predictions'  # Time series predictions
            },
            output_dir=exp_dirs['anomaly_detection'],  # Use anomaly detection directory
            original_val_data=original_val_data,
            filename_prefix=f"supervised_anomaly_detection_{args.flag_method}_"
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
                if cm['total_anomalies_pred'] > len(original_val_data) * 0.1:  # More than 10%
                    print(f"   ⚠️ WARNING: Detection rate seems very high ({cm['total_anomalies_pred']/len(original_val_data)*100:.2f}%)")
                    print(f"   This might indicate overly sensitive detection or high error injection")
    
    # Generate residual plots and other diagnostics
    print("\nGenerating diagnostic plots...")
    from shared.diagnostics.model_diagnostics import generate_all_diagnostics
    
    # Create features DataFrame for residual analysis
    features_df = pd.DataFrame({
        'temperature': val_data_flagged['temperature'],
        'rainfall': val_data_flagged['rainfall']
    })
    
    # Generate all diagnostic plots
    diagnostic_vis_paths = generate_all_diagnostics(
        actual=original_val_data['vst_raw'],  # Use original clean data as reference
        predictions=val_pred_df['vst_raw'],
        output_dir=exp_dirs['diagnostics'],
        station_id=args.station_id,
        features_df=features_df
    )
    
    # Calculate metrics
    from shared.utils.pipeline_utils import calculate_performance_metrics
    valid_mask = ~np.isnan(original_val_data['vst_raw'].values)
    pred_mask = ~np.isnan(val_predictions_original)
    combined_mask = valid_mask[:len(pred_mask)] & pred_mask
    
    if np.sum(combined_mask) > 0:
        # Metrics against original clean data
        metrics = calculate_performance_metrics(
            original_val_data['vst_raw'].values[:len(pred_mask)], 
            val_predictions_original, 
            combined_mask
        )
        
        print(f"\n📊 FLAGGING MODEL RESULTS:")
        print(f"Validation Metrics (vs original clean data):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        # Metrics against corrupted data (model's training target)
        corrupted_metrics = calculate_performance_metrics(
            val_data_flagged['vst_raw'].values[:len(pred_mask)], 
            val_predictions_original, 
            combined_mask
        )
        
        print(f"\nValidation Metrics (vs corrupted training data):")
        for metric, value in corrupted_metrics.items():
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
