"""
Main script to run the error detection pipeline.

This script coordinates the overall pipeline for water level prediction, error detection,
and model evaluation. It uses modular utility functions from the utils package to keep
the code clean and maintainable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to Python path to allow imports from experiments
sys.path.append(str(Path(__file__).parent))

# Local imports
# Configuration
from models.lstm_traditional.config import LSTM_CONFIG
from synthetic_error_config import SYNTHETIC_ERROR_PARAMS, ANOMALY_DETECTION_CONFIG

# Pipeline utilities
from shared.utils.pipeline_utils import (
    calculate_nse, prepare_prediction_dataframe, 
    calculate_performance_metrics, save_comparison_metrics,
    print_metrics_table, print_comparison_table, prepare_features_df,
    calculate_feature_importance
)

# Error handling utilities
from shared.utils.error_utils import (
    configure_error_params, print_error_frequencies,
    inject_errors_into_dataset, identify_water_level_columns
)

# Diagnostic utilities
from shared.utils.diagnostics_utils import (
    run_preprocessing_diagnostics, run_synthetic_diagnostics,
    setup_basic_diagnostics, run_advanced_diagnostics
)

# Model utilities
from shared.utils.model_utils import (
    create_lstm_model, print_model_params, train_model,
    save_model, process_val_predictions, process_test_predictions
)

# Model infrastructure
from models.lstm_traditional.preprocessing_LSTM1 import DataPreprocessor
from shared.diagnostics.model_plots import create_full_plot, plot_convergence, plot_feature_importance, create_individual_feature_plots, plot_feature_correlation
from shared.diagnostics.model_diagnostics import generate_all_diagnostics, generate_comparative_diagnostics

# Experiment modules
from experiments.error_frequency import run_error_frequency_experiments

from models.lstm_traditional.train_model import LSTM_Trainer
from models.lstm_traditional.model import LSTMModel

# from experiments.Improved_model_structure.train_model import LSTM_Trainer
#from experiments.Improved_model_structure.improved_model import LSTMModel

# Anomaly detection utilities
from shared.anomaly_detection.z_score import calculate_z_scores_mad
from shared.anomaly_detection.anomaly_visualization import (
    calculate_anomaly_confidence, plot_water_level_anomalies
)
from shared.anomaly_detection.comprehensive_evaluation import (
    run_single_threshold_anomaly_detection
)

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    inject_synthetic_errors: bool = False,
    model_diagnostics: bool = False,
    advanced_diagnostics: bool = False,
    run_anomaly_detection: bool = False,
    error_multiplier: float = 1.0,
    use_test_data: bool = False,
) -> dict:
    """
    Run the complete error detection and imputation pipeline using yearly windows.
    
    Args:
        project_root (Path): Path to the project root directory
        data_path (str): Path to the data file
        output_path (str): Path to save outputs
        preprocess_diagnostics (bool): Whether to generate preprocessing diagnostics
        synthetic_diagnostics (bool): Whether to generate synthetic error diagnostics
        inject_synthetic_errors (bool): Whether to inject synthetic errors
        model_diagnostics (bool): Whether to generate basic model plots (prediction plots)
        advanced_diagnostics (bool): Whether to generate advanced model diagnostics
        run_anomaly_detection (bool): Whether to run anomaly detection using z_score_MAD
        error_multiplier (float): Multiplier for error counts per year (1.0 = base counts)
        use_test_data (bool): Whether to use test data for predictions, plots, and metrics
    
    Returns:
        dict: Dictionary containing performance metrics and anomaly detection results
    """
    # Initialize configuration and preprocessor
    model_config = LSTM_CONFIG.copy()
    preprocessor = DataPreprocessor(model_config)
    
    # Initialize anomaly results variable
    anomaly_results = {}
    
    #########################################################
    #                Step 1: Data Preparation                #
    #########################################################
    station_id = '21006846'
    print(f"\nProcessing data for station {station_id}...")
    train_data, val_data, test_data, vinge_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Store original data for later use
    original_train_data = train_data.copy()
    original_val_data = val_data.copy()
    original_test_data = test_data.copy()
    
    # Generate diagnostics if enabled
    if model_diagnostics:
        setup_basic_diagnostics(original_train_data, preprocessor.feature_cols, output_path, years_to_show=10)
        # Create correlation plot
        # print("\nGenerating correlation plot...")
        # correlation_plot_path = plot_feature_correlation(original_train_data)
        # print(f"Correlation plot saved to: {correlation_plot_path}")
        
    if preprocess_diagnostics:
        run_preprocessing_diagnostics(project_root, output_path, station_id)
    else:
        print("Skipping preprocessing diagnostics generation...")
    
    #########################################################
    #            Step 2: Synthetic Error Generation          #
    #########################################################
    print("\nStep 2: Generating synthetic errors...")
    stations_results = {}
    from shared.synthetic.synthetic_errors import SyntheticErrorGenerator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    train_data_with_errors = None
    val_data_with_errors = None
    
    if inject_synthetic_errors:
        print(f"\nInjecting synthetic errors with multiplier {error_multiplier:.1f}x into TRAINING and VALIDATION data...")
        try:
            # Configure and apply synthetic errors
            error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, error_multiplier)
            print_error_frequencies(error_config)
            error_generator = SyntheticErrorGenerator(error_config)
            
            # Identify target columns for error injection
            water_level_cols = identify_water_level_columns(original_train_data, station_id)
            if not water_level_cols:
                water_level_cols = ['feature_station_21006845_vst_raw', 'feature_station_21006847_vst_raw']
                water_level_cols = [col for col in water_level_cols if col in original_train_data.columns]
            
            # Filter out the target column 'vst_raw' from water_level_cols
            water_level_cols = [col for col in water_level_cols if col != 'vst_raw']
            
            print(f"Injecting errors into columns: {water_level_cols}")
            
            # Process training data
            print("\nProcessing TRAINING data...")
            train_data_with_errors_raw, train_results = inject_errors_into_dataset(
                original_train_data, error_generator, f"{station_id}_train", water_level_cols, "train"
            )
            stations_results.update(train_results)
            
            # Process validation data (reuse the same generator for consistent seeding)
            print("\nProcessing VALIDATION data...")
            val_data_with_errors_raw, val_results = inject_errors_into_dataset(
                original_val_data, error_generator, f"{station_id}_val", water_level_cols, "val"
            )
            stations_results.update(val_results)
            
            # Process test data if needed for synthetic errors
            if use_test_data:
                print("\nProcessing TEST data...")
                test_data_with_errors_raw, test_results = inject_errors_into_dataset(
                    original_test_data, error_generator, f"{station_id}_test", water_level_cols, "test"
                )
                stations_results.update(test_results)

            train_data_with_errors = train_data_with_errors_raw
            val_data_with_errors = val_data_with_errors_raw
        
            print("\nSynthetic error injection complete.")
            print(f"Created datasets with synthetic errors in {len(water_level_cols)} feature columns")
            
        except Exception as e:
            print(f"Error processing synthetic errors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Step 2b: Target Variable Error Injection (for anomaly detection)
    anomaly_stations_results = {}
    if run_anomaly_detection:
        print(f"\n🎯 Injecting synthetic errors into TARGET VARIABLE for anomaly detection...")
        try:
            # Create separate error generator for anomaly detection with default multiplier
            anomaly_error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, 1.0)  # Use base error rates
            print("📊 Using base error rates for anomaly detection:")
            print_error_frequencies(anomaly_error_config)
            anomaly_error_generator = SyntheticErrorGenerator(anomaly_error_config)
            
            # Inject errors ONLY into target variable
            target_cols = ['vst_raw']
            print(f"Injecting anomaly errors into target column: {target_cols}")
            
            # Process test data for anomaly detection (changed from validation to test)
            print("\nProcessing TEST data for anomaly detection...")
            _, anomaly_test_results = inject_errors_into_dataset(
                original_test_data, anomaly_error_generator, f"{station_id}_anomaly_test", target_cols, "test"
            )
            anomaly_stations_results.update(anomaly_test_results)
            
            print("\nTarget variable error injection for anomaly detection complete.")
            
        except Exception as e:
            print(f"Error processing anomaly detection errors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate synthetic diagnostics if enabled
    if synthetic_diagnostics:
        split_datasets = {'windows': {'train': {station_id: original_train_data}, 'val': {station_id: original_val_data}}}
        synthetic_diagnostic_results = run_synthetic_diagnostics(split_datasets, stations_results, output_path)
    else:
        print("Skipping synthetic error diagnostics generation...")
    
    #########################################################
    #            Step 3: Model Training and Prediction       #
    #########################################################
    print("\nStep 3: Training LSTM models...")
    
    # Initialize model
    print("\nInitializing model...")
    input_size = len(preprocessor.feature_cols)
    
    model = create_lstm_model(input_size, model_config, LSTMModel)
    trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
    
    print_model_params(model_config)
    
    # Select appropriate training data clean/with errors
    if inject_synthetic_errors and train_data_with_errors is not None and val_data_with_errors is not None:
        print("\nUsing data with synthetic errors for training")
        training_data = train_data_with_errors
        validation_data = val_data_with_errors
    else:
        print("\nUsing clean data for training")
        training_data = train_data
        validation_data = val_data
    
    # Train model and get predictions
    history, val_predictions, val_targets = train_model(trainer, training_data, validation_data, model_config)
    save_model(model, 'final_model.pth')
    
    # Determine which dataset to use for evaluation
    if use_test_data:
        print("\nUsing TEST data for predictions and evaluation...")
        eval_data = original_test_data
        eval_data_with_errors = test_data_with_errors_raw if inject_synthetic_errors and 'test_data_with_errors_raw' in locals() else original_test_data
        eval_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
        eval_predictions_df = process_test_predictions(eval_predictions, original_test_data, model_config)
        eval_plot_title = "Test Set - " + ("Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Trained on Clean Data")
        dataset_name = "test"
    else:
        print("\nUsing VALIDATION data for predictions and evaluation...")
        eval_data = original_val_data
        eval_data_with_errors = val_data_with_errors if inject_synthetic_errors else original_val_data
        eval_predictions = val_predictions
        eval_predictions_df = process_val_predictions(val_predictions, preprocessor, original_val_data, model_config)
        eval_plot_title = "Validation Set - " + ("Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Trained on Clean Data")
        dataset_name = "validation"
    
    # Calculate metrics on evaluation set
    metrics = calculate_performance_metrics(eval_data['vst_raw'].values, eval_predictions_df['vst_raw'].values, ~np.isnan(eval_data['vst_raw'].values))
    metrics['val_loss'] = min(history['val_loss'])
    print_metrics_table(metrics)
    
    # Generate evaluation plots
    if model_diagnostics:
        # If synthetic errors were injected, build a dictionary with both the modified data and error periods for the main water level column
        if inject_synthetic_errors:
            # Use the first water level column for visualization
            main_col = water_level_cols[0] if water_level_cols else None
            eval_key = f"{station_id}_{dataset_name}_{main_col}" if main_col else None
            # Fallback: try just f"{station_id}_{dataset_name}" if above not found
            if eval_key not in stations_results and main_col:
                eval_key = f"{station_id}_{dataset_name}"
            # Extract error periods and modified data for the main column
            if eval_key in stations_results:
                synthetic_data = {
                    'data': stations_results[eval_key]['modified_data'],
                    'error_periods': stations_results[eval_key]['error_periods']
                }
            else:
                synthetic_data = None
        else:
            synthetic_data = None
        create_full_plot(
            eval_data, 
            eval_predictions_df, 
            str(station_id), 
            model_config, 
            metrics['val_loss'], 
            create_html=False,  # Disable HTML creation
            title_suffix=eval_plot_title,
            synthetic_data=synthetic_data,
            vinge_data=vinge_data
        )
        plot_convergence(history, str(station_id), title=f"Training and Validation Loss - Station {station_id}")
    
    # Generate advanced diagnostics if enabled
    if advanced_diagnostics:
        print("\nGenerating advanced diagnostic visualizations...")
        predictions_series = pd.Series(
            eval_predictions_df['vst_raw'].values, 
            index=eval_data.index[:len(eval_predictions_df)]
        )
        all_visualization_paths = run_advanced_diagnostics(
            eval_data, predictions_series, station_id, output_path, is_comparative=False
        )
    else:
        print("Skipping advanced diagnostics visualizations")
    
    # Prepare final results
    final_results = {
        'model_metrics': metrics,
        'test_metrics': {
            'test_predictions': eval_predictions_df['vst_raw'].values,
            'test_performance': calculate_performance_metrics(
                eval_data['vst_raw'].values, 
                eval_predictions_df['vst_raw'].values, 
                ~np.isnan(eval_data['vst_raw'].values)
            )
        }
    }
    
    # Add anomaly detection results if available
    if anomaly_results:
        final_results['anomaly_detection'] = anomaly_results
        
        # Print final summary if anomaly detection was run
        if 'confusion_metrics' in anomaly_results:
            print("\n" + "="*60)
            print("PIPELINE COMPLETION SUMMARY")
            print("="*60)
            print(f"Model trained and evaluated successfully")
            print(f"Anomaly detection completed with F1-score: {anomaly_results['confusion_metrics']['f1_score']:.4f}")
            print(f"Total anomalies detected: {anomaly_results['confusion_metrics']['total_anomalies_pred']}")
            print(f"Detection precision: {anomaly_results['confusion_metrics']['precision']:.4f}")
            print(f"Detection recall: {anomaly_results['confusion_metrics']['recall']:.4f}")

    #########################################################
    #              Step 4: Anomaly Detection                 #
    #########################################################
    if run_anomaly_detection:
        print("\n" + "="*60)
        print("Step 4: Anomaly Detection using Z-Score MAD")
        print("="*60)
        print("ℹ️  Flow: Model trained on clean/corrupted features → Predictions made on test data → Anomalies detected in TARGET with synthetic errors")
        
        # Check if we have anomaly ground truth data
        if not anomaly_stations_results:
            print("❌ No anomaly target data available. Run with --anomaly_detection to enable.")
            anomaly_results = {'error': 'No anomaly target data available'}
        else:
            # Use unscaled predictions from the processed test dataframe
            test_predictions_unscaled = eval_predictions_df['vst_raw'].values
            
            print(f"📊 Using unscaled predictions from test_predictions_df: shape {test_predictions_unscaled.shape}")
            print(f"📊 Original test data shape: {eval_data['vst_raw'].values.shape}")
            print(f"📊 Prediction value range: {np.nanmin(test_predictions_unscaled):.1f} to {np.nanmax(test_predictions_unscaled):.1f} mm")
            print(f"📊 Original data value range: {np.nanmin(eval_data['vst_raw'].values):.1f} to {np.nanmax(eval_data['vst_raw'].values):.1f} mm")
            
            # Ensure lengths match
            if len(test_predictions_unscaled) != len(eval_data):
                print(f"⚠️  Length mismatch: predictions={len(test_predictions_unscaled)}, data={len(eval_data)}")
                # Truncate or pad predictions to match data length
                if len(test_predictions_unscaled) > len(eval_data):
                    test_predictions_unscaled = test_predictions_unscaled[:len(eval_data)]
                    print(f"   Truncated predictions to {len(test_predictions_unscaled)}")
                else:
                    # Pad with NaN if predictions are shorter
                    padding = np.full(len(eval_data) - len(test_predictions_unscaled), np.nan)
                    test_predictions_unscaled = np.concatenate([test_predictions_unscaled, padding])
                    print(f"   Padded predictions to {len(test_predictions_unscaled)}")
            
            print(f"✅ Final prediction shape: {test_predictions_unscaled.shape}, data shape: {eval_data['vst_raw'].values.shape}")
            
            # Check for NaN predictions
            if np.sum(~np.isnan(test_predictions_unscaled)) == 0:
                print("⚠️  All predictions are NaN! Using moving average fallback for anomaly detection testing...")
                window = 24  # 6 hours moving average
                test_predictions_unscaled = pd.Series(eval_data['vst_raw']).rolling(window=window, center=True).mean().values
                print(f"   Generated {np.sum(~np.isnan(test_predictions_unscaled))} valid moving average predictions")
            
            # Get the target data with synthetic errors for anomaly detection
            anomaly_target_key = f"{station_id}_anomaly_test_vst_raw"
            if anomaly_target_key in anomaly_stations_results:
                anomaly_target_data = anomaly_stations_results[anomaly_target_key]['modified_data']
                print(f"📍 Using anomaly target data from key: {anomaly_target_key}")
            else:
                print(f"⚠️ Anomaly target key {anomaly_target_key} not found. Available keys: {list(anomaly_stations_results.keys())}")
                anomaly_target_data = eval_data  # Fallback
            
            # Run comprehensive anomaly detection using the new function
            anomaly_results = run_single_threshold_anomaly_detection(
                val_data=anomaly_target_data,  # Use TEST data with synthetic errors for anomaly detection
                predictions=test_predictions_unscaled,  # Use unscaled test predictions
                stations_results=anomaly_stations_results,  # Contains ground truth from TARGET variable error injection
                station_id=station_id,
                config=ANOMALY_DETECTION_CONFIG,
                output_dir=output_path,  # Use same directory as other model plots
                original_val_data=eval_data,  # Pass original test data for comparison
                error_multiplier=1.0,  # Anomaly detection uses base error rates
                dataset_type='test'  # Specify that we're using test data
            )
            
            # Check if anomaly detection succeeded
            if 'error' in anomaly_results:
                print(f"❌ Anomaly detection failed: {anomaly_results['error']}")
                anomaly_results = {}
            else:
                # Add debug information about anomaly detection results
                z_scores = anomaly_results['z_scores']
                detected_anomalies = anomaly_results['detected_anomalies']
                confidence = anomaly_results['confidence']
                ground_truth_flags = anomaly_results['ground_truth']
                
                print(f"\n🔍 ANOMALY DETECTION INFO:")
                print(f"   Anomalies detected: {np.sum(detected_anomalies)}")
                print(f"   Ground truth anomalies: {np.sum(ground_truth_flags)}")
                
                # Generate visualizations (independent of model_diagnostics)
                print(f"\nGenerating anomaly detection visualizations...")
                
                # Create specific directory for anomaly plots
                anomaly_output_dir = output_path  # Use same directory as other plots
                anomaly_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Saving anomaly plots to: {anomaly_output_dir}")
                
                # Count detections by confidence level for title
                high_conf = np.sum((detected_anomalies) & (confidence == 'High'))
                med_conf = np.sum((detected_anomalies) & (confidence == 'Medium'))
                low_conf = np.sum((detected_anomalies) & (confidence == 'Low'))
                
                # Create main anomaly detection plot
                title = f"Anomaly Detection - Station {station_id} (Z-Score MAD)"
                title += f"\nDetected: {high_conf} High, {med_conf} Medium, {low_conf} Low confidence"
                
                png_path, html_path = plot_water_level_anomalies(
                    test_data=anomaly_target_data,  # Use TEST data with synthetic errors for visualization
                    predictions=test_predictions_unscaled,  # Use unscaled test predictions
                    z_scores=z_scores,
                    anomalies=detected_anomalies,
                    threshold=ANOMALY_DETECTION_CONFIG['threshold'],
                    title=title,
                    output_dir=anomaly_output_dir,
                    save_png=True,
                    save_html=False,
                    show_plot=False,
                    filename_prefix="anomaly_detection_",
                    confidence=confidence,
                    original_data=eval_data,  # Pass original clean test data for comparison
                    ground_truth_flags=ground_truth_flags
                )
                
                print(f"  Anomaly detection plot saved to: {png_path}")
    else:
        print("\nSkipping anomaly detection (not requested)")
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LSTM water level prediction model')
    parser.add_argument('--error_multiplier', type=float, default=1, 
                      help='Error multiplier for synthetic errors. If not provided, no errors are injected.')
    parser.add_argument('--run_experiments', action='store_true',
                      help='Run experiments with different error multipliers')
    parser.add_argument('--preprocess_diagnostics', action='store_true',
                      help='Generate preprocessing diagnostics')
    parser.add_argument('--synthetic_diagnostics', action='store_true',
                      help='Generate synthetic error diagnostics')
    parser.add_argument('--model_diagnostics', action='store_true',
                      help='Generate basic model plots (predictions)')
    parser.add_argument('--advanced_diagnostics', action='store_true',
                      help='Generate advanced model diagnostics')
    parser.add_argument('--no_diagnostics', action='store_true',
                      help='Disable all diagnostics plots')
    parser.add_argument('--anomaly_detection', action='store_true',
                      help='Enable anomaly detection using Z-Score MAD method')
    parser.add_argument('--use_test_data', action='store_true',
                      help='Use test data for predictions, plots, and metrics (training still uses train/val data)')
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure diagnostics
    use_model_diagnostics = not args.no_diagnostics
    use_advanced_diagnostics = args.advanced_diagnostics
    
    # Configure anomaly detection
    if args.anomaly_detection:
        print("Anomaly detection enabled (Z-Score MAD method)")
    else:
        print("Anomaly detection disabled (use --anomaly_detection to enable)")
        
    # Configure test data usage
    if args.use_test_data:
        print("Using TEST data for predictions, plots, and metrics")
    else:
        print("Using VALIDATION data for predictions, plots, and metrics")
    
    # Run pipeline
    if args.run_experiments:
        run_error_frequency_experiments(run_pipeline)
    else:
        try:
            print("\nRunning LSTM model with configuration from config.py")
            
            if args.error_multiplier is not None:
                print(f"Using error multiplier: {args.error_multiplier:.1f}x")
                print(f"(This multiplies the base error counts per year defined in config.py)")
            
            print("\n📊 Evaluation Configuration:")
            print(f"Training data: train + validation sets")
            if args.use_test_data:
                print(f"Evaluation data: test set (predictions, plots, metrics)")
            else:
                print(f"Evaluation data: validation set (predictions, plots, metrics)")
            
            # Configure diagnostics output
            if args.no_diagnostics:
                print("All diagnostics plots disabled")
            elif args.model_diagnostics:
                print("Basic model plots enabled")
                if args.advanced_diagnostics:
                    print("Advanced diagnostics also enabled")
            else:
                print("Running with default diagnostic settings")
            
            # Run main pipeline
            result = run_pipeline(
                project_root=project_root,
                data_path=data_path, 
                output_path=output_path,
                preprocess_diagnostics=args.preprocess_diagnostics,
                synthetic_diagnostics=args.synthetic_diagnostics,
                inject_synthetic_errors=args.error_multiplier is not None,
                model_diagnostics=use_model_diagnostics,
                advanced_diagnostics=use_advanced_diagnostics,
                run_anomaly_detection=args.anomaly_detection,
                error_multiplier=args.error_multiplier if args.error_multiplier is not None else 1.0,
                use_test_data=args.use_test_data,
            )

            print("\nModel run completed!")
            print(f"Results saved to: {output_path}")
            
            # Print comprehensive results summary
            if 'anomaly_detection' in result and 'confusion_metrics' in result['anomaly_detection']:
                anomaly_metrics = result['anomaly_detection']['confusion_metrics']
                print(f"\nAnomaly Detection Summary:")
                print(f"  F1-Score: {anomaly_metrics['f1_score']:.4f}")
                print(f"  Precision: {anomaly_metrics['precision']:.4f}")
                print(f"  Recall: {anomaly_metrics['recall']:.4f}")
                print(f"  Total anomalies detected: {anomaly_metrics['total_anomalies_pred']}")
                print(f"  Total anomalies in ground truth: {anomaly_metrics['total_anomalies_true']}")
                print(f"  Correctly detected: {anomaly_metrics['true_positives']}")
            
        except Exception as e:
            print(f"\nError running pipeline: {e}")
            import traceback
            traceback.print_exc()