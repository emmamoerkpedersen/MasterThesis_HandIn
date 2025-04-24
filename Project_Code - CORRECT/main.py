"""
Main script to run the error detection pipeline.

This script coordinates the overall pipeline for water level prediction, error detection,
and model evaluation. It uses modular utility functions from the utils package to keep
the code clean and maintainable.
"""
import pandas as pd
import torch
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the parent directory to Python path to allow imports from experiments
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG

# Import utility modules
from utils.pipeline_utils import (
    calculate_nse, prepare_prediction_dataframe, 
    calculate_performance_metrics, save_comparison_metrics,
    print_metrics_table, print_comparison_table, prepare_features_df
)
from utils.error_utils import (
    configure_error_params, print_error_frequencies,
    inject_errors_into_dataset, identify_water_level_columns
)
from utils.diagnostics_utils import (
    run_preprocessing_diagnostics, run_synthetic_diagnostics,
    setup_basic_diagnostics, run_advanced_diagnostics
)
from utils.model_utils import (
    create_lstm_model, print_model_params, train_model,
    save_model, process_val_predictions, process_test_predictions
)

# Import model infrastructure
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _3_lstm_model.model_diagnostics import generate_all_diagnostics, generate_comparative_diagnostics
from experiments.error_frequency import run_error_frequency_experiments
from experiments.Improved_model_structure.train_model import LSTM_Trainer
from experiments.Improved_model_structure.model import LSTMModel

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    inject_synthetic_errors: bool = False,
    model_diagnostics: bool = True,
    advanced_diagnostics: bool = False,
    error_frequency: float = 0.1,
):
    """
    Run the complete error detection and imputation pipeline using yearly windows.
    
    Args:
        project_root: Path to the project root directory
        data_path: Path to the data file
        output_path: Path to save outputs
        preprocess_diagnostics: Whether to generate preprocessing diagnostics
        synthetic_diagnostics: Whether to generate synthetic error diagnostics
        inject_synthetic_errors: Whether to inject synthetic errors
        model_diagnostics: Whether to generate basic model plots (prediction plots)
        advanced_diagnostics: Whether to generate advanced model diagnostics
        error_frequency: Frequency of synthetic errors to inject (0-1)
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Start with base configuration from config.py
    model_config = LSTM_CONFIG.copy()
    
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(model_config)

    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    station_id = '21006846'
    print(f"Loading, preprocessing and splitting station data for station {station_id}...")
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Generate basic diagnostic plots if enabled
    if model_diagnostics:
        setup_basic_diagnostics(train_data, preprocessor.feature_cols, output_path)
    
    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics:
        run_preprocessing_diagnostics(project_root, output_path, station_id)
    else:
        print("Skipping preprocessing diagnostics generation...")
    
    #########################################################
    #    Step 2: Generate synthetic errors                  #
    #########################################################
    
    print("\nStep 2: Generating synthetic errors...")
    
    # Dictionary to store results for each station/year
    stations_results = {}
    # Create synthetic error generator
    from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    # Track whether we successfully created data with errors
    train_data_with_errors = None
    val_data_with_errors = None
    
    if inject_synthetic_errors:
        print(f"\nInjecting synthetic errors with frequency {error_frequency*100:.1f}% into TRAINING and VALIDATION data...")
        try:
            # Configure error parameters based on the frequency parameter
            error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, error_frequency)
            print_error_frequencies(error_config)
            
            # Create a new error generator with the modified config
            error_generator = SyntheticErrorGenerator(error_config)
            
            # Identify water level columns for error injection
            water_level_cols = identify_water_level_columns(train_data, station_id)
            if not water_level_cols:
                # Fallback to hardcoded columns if detection fails
                water_level_cols = ['feature_station_21006845_vst_raw', 'feature_station_21006847_vst_raw']
                water_level_cols = [col for col in water_level_cols if col in train_data.columns]
            
            print(f"Injecting errors into these water level columns: {water_level_cols}")
            
            # Process training data
            print("\nProcessing TRAINING data...")
            train_data_with_errors, train_results = inject_errors_into_dataset(
                train_data, error_generator, f"{station_id}_train", water_level_cols
            )
            stations_results.update(train_results)
            
            # Reset error generator for validation data
            error_generator = SyntheticErrorGenerator(error_config)
            
            # Process validation data
            print("\nProcessing VALIDATION data...")
            val_data_with_errors, val_results = inject_errors_into_dataset(
                val_data, error_generator, f"{station_id}_val", water_level_cols
            )
            stations_results.update(val_results)
            
            print("\nSynthetic error injection complete.")
            print(f"Created training and validation datasets with synthetic errors in {len(water_level_cols)} feature columns")
            
        except Exception as e:
            print(f"Error processing synthetic errors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate synthetic error diagnostics if enabled
    if synthetic_diagnostics:
        split_datasets = {'windows': {'train': {station_id: train_data}, 'val': {station_id: val_data}}}
        synthetic_diagnostic_results = run_synthetic_diagnostics(split_datasets, stations_results, output_path)
    else:
        print("Skipping synthetic error diagnostics generation...")
    
    #########################################################
    # Step 3: LSTM training and prediction                  #
    #########################################################
    
    print("\nStep 3: Training LSTM models with Station-Specific Approach...")
    
    # Initialize model
    print("\nInitializing LSTM model...")
    
    # Get input size from feature columns
    input_size = len(preprocessor.feature_cols)

    # Create the model with the correct input size
    model = create_lstm_model(input_size, model_config, LSTMModel)
    
    # Print model hyperparameters
    print_model_params(model_config)
    
    # Initialize the trainer with the verified config
    trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
    
    # Determine which data to use for training and validation
    if inject_synthetic_errors and train_data_with_errors is not None and val_data_with_errors is not None:
        print("\nUsing TRAINING and VALIDATION data with synthetic errors for model training")
        training_data = train_data_with_errors
        validation_data = val_data_with_errors
    else:
        print("\nUsing clean training and validation data (no synthetic errors)")
        training_data = train_data
        validation_data = val_data
    
    # Train the model
    history, val_predictions, val_targets = train_model(trainer, training_data, validation_data, model_config)

    # Save the trained model
    save_model(model, 'final_model.pth')
    
    # Process validation predictions
    val_predictions_df = process_val_predictions(val_predictions, preprocessor, validation_data)
    
    # Create plot titles
    val_plot_title = "Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Trained on Clean Data"
    test_plot_title = "Model Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Model Trained on Clean Data"
    
    # Get the best validation loss for plot title
    best_val_loss = min(history['val_loss'])
    
    # Plot validation results
    if model_diagnostics:
        create_full_plot(validation_data, val_predictions_df, str(station_id), model_config, best_val_loss, title_suffix=val_plot_title)
        
        # Plot convergence
        plot_convergence(history, str(station_id), title=f"Training and Validation Loss - Station {station_id}")
    
    # Make and plot test predictions
    print("\nMaking predictions on test set (clean data)...")
    test_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
    
    # Process test predictions
    test_predictions_df = process_test_predictions(test_predictions, test_data)
    
    # Plot test results with model config
    if model_diagnostics:
        create_full_plot(test_data, test_predictions_df, str(station_id), model_config, title_suffix=test_plot_title)
    
    # Create a dictionary to store all performance metrics
    performance_metrics = {}
    
    # If we used synthetic errors, create a comparison of validation predictions with and without errors
    if inject_synthetic_errors:
        # Train a second model using the same configuration but with clean data
        print("\nFor comparison, training a second model with clean data...")
        
        # Create a new model with the same configuration
        clean_model = create_lstm_model(input_size, model_config, LSTMModel)
        
        # Initialize a new trainer
        clean_trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
        
        # Train the model with clean data
        clean_history, clean_val_predictions, clean_val_targets = train_model(
            clean_trainer, train_data, val_data, model_config
        )
        
        # Get predictions on test data from the clean-trained model
        clean_test_predictions, _, _ = clean_trainer.predict(test_data)
        
        # Process clean predictions
        clean_test_predictions_flat = np.array(clean_test_predictions).flatten()
        print(f"Clean predictions shape after flatten: {clean_test_predictions_flat.shape}")
        
        if len(clean_test_predictions_flat) > len(test_data):
            print(f"Clean predictions longer than test data. Truncating from {len(clean_test_predictions_flat)} to {len(test_data)}")
            clean_test_predictions_reshaped = clean_test_predictions_flat[:len(test_data)]
        else:
            print(f"Clean predictions shorter than test data. Padding from {len(clean_test_predictions_flat)} to {len(test_data)}")
            padding_length = len(test_data) - len(clean_test_predictions_flat)
            padding = np.full(padding_length, np.nan)
            clean_test_predictions_reshaped = np.concatenate([clean_test_predictions_flat, padding])
        
        # Process error predictions
        test_predictions_flat = np.array(test_predictions).flatten()
        print(f"Error predictions shape after flatten: {test_predictions_flat.shape}")
        
        if len(test_predictions_flat) > len(test_data):
            print(f"Error predictions longer than test data. Truncating from {len(test_predictions_flat)} to {len(test_data)}")
            test_predictions_reshaped = test_predictions_flat[:len(test_data)]
        else:
            print(f"Error predictions shorter than test data. Padding from {len(test_predictions_flat)} to {len(test_data)}")
            padding_length = len(test_data) - len(test_predictions_flat)
            padding = np.full(padding_length, np.nan)
            test_predictions_reshaped = np.concatenate([test_predictions_flat, padding])
        
        # Calculate metrics for both models
        clean_metrics = calculate_performance_metrics(test_data['vst_raw'].values, clean_test_predictions_reshaped, ~np.isnan(clean_test_predictions_reshaped))
        clean_metrics['val_loss'] = min(clean_history['val_loss'])
        
        error_metrics = calculate_performance_metrics(test_data['vst_raw'].values, test_predictions_reshaped, ~np.isnan(test_predictions_reshaped))
        error_metrics['val_loss'] = min(history['val_loss'])
        
        # Print comparison metrics table
        print_comparison_table(clean_metrics, error_metrics, error_frequency)
        
        # Save comparison metrics to CSV
        metrics_file, standard_metrics_file = save_comparison_metrics(
            output_path, error_frequency, clean_metrics, error_metrics
        )
        print(f"\nMetrics saved to: {metrics_file}")
        print(f"Metrics appended to: {standard_metrics_file}")
        
        # Generate advanced diagnostic visualizations if enabled
        if advanced_diagnostics:
            print("\nGenerating comparative diagnostic visualizations...")
            
            # Create Series for model predictions
            clean_predictions_series = pd.Series(
                clean_test_predictions_reshaped, 
                index=test_data.index[:len(clean_test_predictions_reshaped)]
            )
            error_predictions_series = pd.Series(
                test_predictions_reshaped, 
                index=test_data.index[:len(test_predictions_reshaped)]
            )
            
            # Create dictionary of predictions for comparative diagnostics
            predictions_dict = {
                'clean_trained': clean_predictions_series,
                'error_trained': error_predictions_series
            }
            
            # Run comparative diagnostics
            all_visualization_paths = run_advanced_diagnostics(
                test_data, predictions_dict, station_id, output_path, is_comparative=True
            )
        else:
            print("Skipping comparative diagnostic visualizations (advanced_diagnostics=False)")
        
        # Store performance metrics for return
        performance_metrics = {
            'error_frequency': error_frequency,
            'clean_model': clean_metrics,
            'error_model': error_metrics
        }
        
        return performance_metrics
    
    else:
        # Only process one model (trained on clean or error data based on initial flags)
        
        # Calculate mask for valid data points
        test_data_nan_mask = ~np.isnan(test_data['vst_raw']).values
        
        # Make sure test_predictions_reshaped is a 1D array of the right length
        if len(test_predictions) != len(test_data):
            # First ensure predictions are flattened to 1D
            test_predictions_flat = np.array(test_predictions).flatten()
            print(f"Flattened predictions shape: {test_predictions_flat.shape}")
            print(f"Test data length: {len(test_data)}")
            
            # Resize if necessary to match test data length
            if len(test_predictions_flat) > len(test_data):
                print(f"Predictions longer than test data. Truncating from {len(test_predictions_flat)} to {len(test_data)}")
                test_predictions_reshaped = test_predictions_flat[:len(test_data)]
            else:
                # If predictions are shorter, pad with NaN values
                print(f"Predictions shorter than test data. Padding from {len(test_predictions_flat)} to {len(test_data)}")
                padding_length = len(test_data) - len(test_predictions_flat)
                print(f"Creating padding of length: {padding_length}")
                padding = np.full(padding_length, np.nan)
                test_predictions_reshaped = np.concatenate([test_predictions_flat, padding])
        else:
            test_predictions_reshaped = test_predictions
            
        # Ensure test_predictions_reshaped is a 1D array
        test_predictions_reshaped = np.array(test_predictions_reshaped).flatten()
        print(f"Final predictions shape: {test_predictions_reshaped.shape}")
        
        # Calculate prediction mask and combine with data mask
        predictions_nan_mask = ~np.isnan(test_predictions_reshaped)
        valid_mask = test_data_nan_mask & predictions_nan_mask
        
        # Calculate metrics
        metrics = calculate_performance_metrics(test_data['vst_raw'].values, test_predictions_reshaped, valid_mask)
        metrics['val_loss'] = min(history['val_loss'])
        
        # Print metrics table
        print_metrics_table(metrics)
        
        # Generate advanced diagnostic visualizations if enabled
        if advanced_diagnostics:
            print("\nGenerating advanced diagnostic visualizations...")
            
            # Create Series for model predictions
            predictions_series = pd.Series(
                test_predictions_reshaped, 
                index=test_data.index[:len(test_predictions_reshaped)]
            )
            
            # Run diagnostics
            all_visualization_paths = run_advanced_diagnostics(
                test_data, predictions_series, station_id, output_path, is_comparative=False
            )
        else:
            print("Skipping advanced diagnostics visualizations (advanced_diagnostics=False)")
        
        # Store performance metrics for return
        performance_metrics = {'model': metrics}
        
        return performance_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LSTM water level prediction model')
    parser.add_argument('--error_frequency', type=float, default=None, 
                        help='Error frequency for synthetic errors (0-1). If not provided, no errors are injected.')
    parser.add_argument('--run_experiments', action='store_true',
                        help='Run experiments with different error frequencies')
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
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add some logic to decide whether to use model_diagnostics by default
    use_model_diagnostics = not args.no_diagnostics
    use_advanced_diagnostics = args.advanced_diagnostics
    
    # Run experiments with different error frequencies
    if args.run_experiments:
        # Pass the run_pipeline function to the experiments module
        run_error_frequency_experiments(run_pipeline)
    # Run single model with specified error frequency
    else:
        try:
            print("\nRunning LSTM model with configuration from config.py")
            
            if args.error_frequency is not None:
                print(f"Injecting synthetic errors with frequency: {args.error_frequency*100:.1f}%")
            
            # Determine if we should use diagnostics
            if args.no_diagnostics:
                print("All diagnostics plots disabled")
            elif args.model_diagnostics:
                print("Basic model plots enabled")
                if args.advanced_diagnostics:
                    print("Advanced diagnostics also enabled")
            else:
                print("Running with default diagnostic settings")
        
            # Run pipeline with simplified configuration handling
            result = run_pipeline(
                project_root=project_root,
                data_path=data_path, 
                output_path=output_path,
                preprocess_diagnostics=args.preprocess_diagnostics,
                synthetic_diagnostics=args.synthetic_diagnostics,
                inject_synthetic_errors=args.error_frequency is not None,
                model_diagnostics=use_model_diagnostics,
                advanced_diagnostics=use_advanced_diagnostics,
                error_frequency=args.error_frequency if args.error_frequency is not None else 0.1,
            )

            print("\nModel run completed!")
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"\nError running pipeline: {e}")
            import traceback
            traceback.print_exc()

'''
To run the model with different diagnostic options, you can use the following command-line arguments:

For basic prediction plots only:
python main.py --model_diagnostics

For preprocessing diagnostics only:
python main.py --preprocess_diagnostics

For both basic plots and preprocessing diagnostics:
python main.py --model_diagnostics --preprocess_diagnostics

For advanced diagnostics (includes all plots):
python main.py --model_diagnostics --advanced_diagnostics

With a specific error frequency and basic plots:
python main.py --error_frequency 0.1 --model_diagnostics

With a specific error frequency and advanced diagnostics:
python main.py --error_frequency 0.1 --model_diagnostics --advanced_diagnostics

Run experiments with error frequencies:
python main.py --run_experiments

Disable all diagnostics plots:
python main.py --no_diagnostics
'''
