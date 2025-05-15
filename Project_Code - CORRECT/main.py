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
import torch

# Add the parent directory to Python path to allow imports from experiments
sys.path.append(str(Path(__file__).parent))

# Local imports
# Configuration
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG

# Pipeline utilities
from utils.pipeline_utils import (
    calculate_nse, prepare_prediction_dataframe, 
    calculate_performance_metrics, save_comparison_metrics,
    print_metrics_table, print_comparison_table, prepare_features_df
)

# Error handling utilities
from utils.error_utils import (
    configure_error_params, print_error_frequencies,
    inject_errors_into_dataset, identify_water_level_columns
)

# Diagnostic utilities
from utils.diagnostics_utils import (
    run_preprocessing_diagnostics, run_synthetic_diagnostics,
    setup_basic_diagnostics, run_advanced_diagnostics
)

# Model utilities
from utils.model_utils import (
    create_lstm_model, print_model_params, train_model,
    save_model, process_val_predictions, process_test_predictions
)

# Model infrastructure
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _3_lstm_model.model_diagnostics import generate_all_diagnostics, generate_comparative_diagnostics

# Experiment modules
from experiments.error_frequency import run_error_frequency_experiments

from _3_lstm_model.train_model import LSTM_Trainer
from _3_lstm_model.model import LSTMModel

# # EMMA HUSK AT BRUGE GAMLE MODEL OG TRÆNIN GOGSÅ!!!
# from experiments.Improved_model_structure.train_model import LSTM_Trainer
# from experiments.Improved_model_structure.model import LSTMModel

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    inject_synthetic_errors: bool = False,
    model_diagnostics: bool = False,
    advanced_diagnostics: bool = False,
    error_multiplier: float = 1.0,
    model_type: str = 'standard'
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
        error_multiplier (float): Multiplier for error counts per year (1.0 = base counts)
        model_type (str): Type of model to use (only 'standard' supported now)
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Initialize configuration and preprocessor
    model_config = LSTM_CONFIG.copy()
    preprocessor = DataPreprocessor(model_config)

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
        setup_basic_diagnostics(original_train_data, preprocessor.feature_cols, output_path)
    
    if preprocess_diagnostics:
        run_preprocessing_diagnostics(project_root, output_path, station_id)
    else:
        print("Skipping preprocessing diagnostics generation...")
    
    #########################################################
    #            Step 2: Synthetic Error Generation          #
    #########################################################
    print("\nStep 2: Generating synthetic errors...")
    stations_results = {}
    from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
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
            
            print(f"Injecting errors into columns: {water_level_cols}")
            
            # Process training data
            print("\nProcessing TRAINING data...")
            train_data_with_errors_raw, train_results = inject_errors_into_dataset(
                original_train_data, error_generator, f"{station_id}_train", water_level_cols
            )
            stations_results.update(train_results)
            
            # Process validation data
            error_generator = SyntheticErrorGenerator(error_config)
            print("\nProcessing VALIDATION data...")
            val_data_with_errors_raw, val_results = inject_errors_into_dataset(
                original_val_data, error_generator, f"{station_id}_val", water_level_cols
            )
            stations_results.update(val_results)
        

            train_data_with_errors = train_data_with_errors_raw
            val_data_with_errors = val_data_with_errors_raw
        
            print("\nSynthetic error injection complete.")
            print(f"Created datasets with synthetic errors in {len(water_level_cols)} feature columns")
            
        except Exception as e:
            print(f"Error processing synthetic errors: {str(e)}")
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
    
    # Process validation predictions
    val_predictions_df = process_val_predictions(val_predictions, preprocessor, original_val_data, model_config)
    
    
    # Set plot titles
    val_plot_title = "Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Trained on Clean Data"
    test_plot_title = "Model Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Model Trained on Clean Data"
    best_val_loss = min(history['val_loss'])
    
    # Generate validation plots
    if model_diagnostics:
        # If synthetic errors were injected, build a dictionary with both the modified data and error periods for the main water level column
        if inject_synthetic_errors:
            # Use the first water level column for visualization
            main_col = water_level_cols[0] if water_level_cols else None
            val_key = f"{station_id}_val_{main_col}" if main_col else None
            # Fallback: try just f"{station_id}_val" if above not found
            if val_key not in stations_results and main_col:
                val_key = f"{station_id}_val"
            # Extract error periods and modified data for the main column
            if val_key in stations_results:
                synthetic_data = {
                    'data': stations_results[val_key]['modified_data'],
                    'error_periods': stations_results[val_key]['error_periods']
                }
            else:
                synthetic_data = None
        else:
            synthetic_data = None
        create_full_plot(
            original_val_data, 
            val_predictions_df, 
            str(station_id), 
            model_config, 
            best_val_loss, 
            title_suffix=val_plot_title,
            synthetic_data=synthetic_data,
            vinge_data=vinge_data
        )
        plot_convergence(history, str(station_id), title=f"Training and Validation Loss - Station {station_id}")
    # Generate test predictions
    print("\nMaking predictions on test set...")
    test_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
    test_predictions_df = process_test_predictions(test_predictions, original_test_data, model_config)
    
    # Plot test results with model config
    # if model_diagnostics:
    #     create_full_plot(
    #         original_test_data, 
    #         test_predictions_df, 
    #         str(station_id), 
    #         model_config, 
    #         title_suffix=test_plot_title,
    #         synthetic_data=None  # No synthetic errors in test data
    #     )
    
    test_data_nan_mask = ~np.isnan(test_data['vst_raw']).values
    
    # Ensure predictions are properly shaped
    test_predictions_reshaped = np.array(test_predictions).flatten()
    if len(test_predictions_reshaped) != len(original_test_data):
        if len(test_predictions_reshaped) > len(original_test_data):
            test_predictions_reshaped = test_predictions_reshaped[:len(original_test_data)]
        else:
            padding = np.full(len(original_test_data) - len(test_predictions_reshaped), np.nan)
            test_predictions_reshaped = np.concatenate([test_predictions_reshaped, padding])
    
    # Calculate final metrics
    predictions_nan_mask = ~np.isnan(test_predictions_reshaped)
    valid_mask = test_data_nan_mask & predictions_nan_mask
    metrics = calculate_performance_metrics(original_test_data['vst_raw'].values, test_predictions_reshaped, valid_mask)
    metrics['val_loss'] = min(history['val_loss'])
    print_metrics_table(metrics)
    
    # Generate advanced diagnostics if enabled
    if advanced_diagnostics:
        print("\nGenerating advanced diagnostic visualizations...")
        predictions_series = pd.Series(
            test_predictions_reshaped, 
            index=original_test_data.index[:len(test_predictions_reshaped)]
        )
        all_visualization_paths = run_advanced_diagnostics(
            original_test_data, predictions_series, station_id, output_path, is_comparative=False
        )
    else:
        print("Skipping advanced diagnostics visualizations")
    
    return {'model': metrics}


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LSTM water level prediction model')
    parser.add_argument('--error_multiplier', type=float, default=None, 
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
    parser.add_argument('--model_type', type=str, choices=['standard'], default='standard',
                      help='Type of model to use. Currently only standard LSTM is supported')
    
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
    
    # Run pipeline
    if args.run_experiments:
        run_error_frequency_experiments(run_pipeline)
    else:
        try:
            print("\nRunning LSTM model with configuration from config.py")
            
            if args.error_multiplier is not None:
                print(f"Using error multiplier: {args.error_multiplier:.1f}x")
                print(f"(This multiplies the base error counts per year defined in config.py)")
            
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
                error_multiplier=args.error_multiplier if args.error_multiplier is not None else 1.0,
                model_type=args.model_type
            )

            print("\nModel run completed!")
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"\nError running pipeline: {e}")
            import traceback
            traceback.print_exc()

'''
To run the model with different options:

For standard LSTM:
python main.py --model_type standard

For iterative forecaster:
python main.py --model_type iterative

With error injection:
python main.py --model_type iterative --error_multiplier 2.0

With diagnostics:
python main.py --model_type iterative --model_diagnostics --advanced_diagnostics
'''
