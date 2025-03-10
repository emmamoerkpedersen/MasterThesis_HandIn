"""
Main script to run the error detection pipeline.

TODO:
Implement cross validation strategies?

"""
import pandas as pd
import torch
from pathlib import Path
import sys

from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data, create_interactive_temperature_plot
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG
from _1_preprocessing.split import split_data_rolling
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from _3_lstm_model.lstm_forecaster import train_LSTM, LSTMModel, create_full_plot


lstm_config = LSTM_CONFIG.copy()

def run_pipeline(
    data_path: str, 
    output_path: str, 
    test_mode: bool = False,
    test_years: int = 1,
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    detection_diagnostics: bool = False,
):
    """
    Run the complete error detection and imputation pipeline using yearly windows.
    
    Args:
        data_path: Path to input data
        output_path: Path for results
        test_mode: Run in test mode
        test_years: Number of most recent years to use in test mode
        preprocess_diagnostics: Generate diagnostics for preprocessing step
        split_diagnostics: Generate diagnostics for data splitting
        synthetic_diagnostics: Generate diagnostics for synthetic error generation
        detection_diagnostics: Generate diagnostics for error detection
    """
    
    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    station_id =['21006846']

    print(f"Loading and preprocessing station data for station {station_id}...")
   
    
    # # Load original data first (if preprocessing diagnostics enabled)
    # if preprocess_diagnostics:
    #     original_data = pd.read_pickle('data_utils/Sample data/original_data.pkl')
    
    # Load Preprocessed data
    preprocessed_data = pd.read_pickle('data_utils/Sample data/preprocessed_data.pkl')
    freezing_periods = pd.read_pickle('data_utils/Sample data/frost_periods.pkl')
    
    # Generate dictionary keeping same structure but with the specified station_id
    preprocessed_data = {station_id: preprocessed_data[station_id] for station_id in station_id}
    
    # # Generate preprocessing diagnostics if enabled
    # if preprocess_diagnostics and original_data:
    #     print("Generating preprocessing diagnostics...")
    #     plot_preprocessing_comparison(original_data, preprocessed_data, Path(output_path), freezing_periods)
    #     generate_preprocessing_report(preprocessed_data, Path(output_path), original_data)
    #     plot_additional_data(preprocessed_data, Path(output_path))
    #     create_interactive_temperature_plot(preprocessed_data, Path(output_path))
    
    #########################################################
    #    Step 2: Split data into rolling windows           #
    #########################################################
    
    print("\nSplitting data into rolling windows of 3 years training and 1 year validation...")
    
    # Split data into yearly windows
    split_datasets = split_data_rolling(preprocessed_data)
    
    # Generate split diagnostics if enabled
    # if split_diagnostics:
    #     print("Generating split diagnostics...")
    #     plot_split_visualization(split_datasets, Path(output_path))
    #     generate_split_report(split_datasets, Path(output_path))
    
    # # If in test mode, limit to most recent years
    # if test_mode and test_years > 0:
    #     print(f"Test mode enabled. Using only the {test_years} most recent years.")
        
    #     # Get all years and sort them
    #     all_years = sorted(list(split_datasets['windows'].keys()))
        
    #     # Keep only the most recent years
    #     recent_years = all_years[-test_years:] if len(all_years) > test_years else all_years
        
    #     # Filter the split datasets
    #     split_datasets['windows'] = {year: data for year, data in split_datasets['windows'].items() if year in recent_years}
        
    #     print(f"Using years: {', '.join(recent_years)}")
    
    #########################################################
    #    Step 3: Generate synthetic errors                  #
    #########################################################
    
    print("\nStep 3: Generating synthetic errors...")
    
    # Dictionary to store results for each station/year
    stations_results = {}
    # Create synthetic error generator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    # Process only test data
    if 'test' in split_datasets:
        print("\nProcessing test data...")
        for station, station_data in split_datasets['test'].items():
            try:
                print(f"Generating synthetic errors for {station} (Test)...")
                test_data = station_data['vst_raw']
                
                if test_data is None or test_data.empty:
                    print(f"No test data available for station {station}")
                    continue
                
                # Generate synthetic errors
                modified_data, ground_truth = error_generator.inject_all_errors(test_data)
                
                # Store results
                station_key = f"{station}_test"
                stations_results[station_key] = {
                    'modified_data': modified_data,
                    'ground_truth': ground_truth,
                    'error_periods': error_generator.error_periods
                }
                
            except Exception as e:
                print(f"Error processing station {station}: {str(e)}")
                continue
    
    # # Generate synthetic error diagnostics if enabled
    # if synthetic_diagnostics:
    #     from diagnostics.synthetic_diagnostics import run_all_synthetic_diagnostics
        
    #     synthetic_diagnostic_results = run_all_synthetic_diagnostics(
    #         split_datasets=split_datasets,
    #         stations_results=stations_results,
    #         output_dir=Path(output_path)
    #     )
    
    #########################################################
    # Step 4: LSTM-based Anomaly Detection                  #
    #########################################################
    
    print("\nStep 4: Training LSTM models with Station-Specific Approach...")
    
    # Prepare train and validation data
    print("\nPreparing training and validation data...")
    
    # Get all available windows
    num_windows = len(split_datasets['windows'])
    print(f"Total number of windows: {num_windows}")

    # Use all windows - each window has its own train/val split
    print(f"Using all {num_windows} windows for training/validation")
    print(f"Each window contains:")
    print(f"- 3 years of training data")
    print(f"- 1 year of validation data")
    print(f"(Test data is stored separately in split_datasets['test'])")

    # Use the LSTM configuration from config.py
    print(f"Input feature size: {len(lstm_config.get('feature_cols'))}")

    # Initialize model and trainer
    model = LSTMModel(
        input_size=len(lstm_config['feature_cols']),
        sequence_length=lstm_config.get('sequence_length'),
        hidden_size=lstm_config.get('hidden_size'),
        output_size=1,
        num_layers=lstm_config.get('num_layers'),
        dropout=lstm_config.get('dropout')
    )
    
    trainer = train_LSTM(model, lstm_config)
    
    # Train on each window
    for window_idx, window_data in split_datasets['windows'].items():
        print(f"\nProcessing window {window_idx}")
        
        # Get training and validation data for this window
        train_data = window_data['train']
        val_data = window_data['validation']
        
        # Train the model
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=lstm_config.get('epochs'),
            batch_size=lstm_config.get('batch_size'),
            patience=lstm_config.get('patience')
        )
        
        # Optionally save the model after each window
        torch.save(model.state_dict(), f'model_window_{window_idx}.pth')
    
    # After training, use the model for predictions
    test_predictions = trainer.predict(split_datasets['test'])
    
    # Create and show plot
    create_full_plot(
        test_data=split_datasets['test'][station_id[0]], 
        test_predictions=test_predictions, 
        station_id=station_id,
        sequence_length=lstm_config.get('sequence_length', 72)
    )
    
    return test_predictions, split_datasets


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline with proper hyperparameter tuning
    test_predictions, split_datasets = run_pipeline(data_path, output_path)