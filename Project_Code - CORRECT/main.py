"""
Main script to run the error detection pipeline.

TODO:
Implement cross validation strategies?

Sequence length should be the entire training dataset, entire validation and test set.
Batch size should be 1 since we use the entire period for sequences
The output should instad of predicting the next value in the sequence, should be the entire sequence. So isntead of predicting simulate?

Remove bidirectional?
"""
import pandas as pd
import torch
from pathlib import Path
import sys
import json
import os

# Disable cuDNN to avoid contiguity issues
torch.backends.cudnn.enabled = False
print("cuDNN disabled globally to avoid contiguity issues")

from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, plot_additional_data, generate_preprocessing_report, plot_station_data_overview
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG
from _1_preprocessing.split import split_data_with_combined_windows
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from _3_lstm_model.lstm_forecaster import train_LSTM, SimpleLSTMModel, create_full_plot
from _3_lstm_model.hyperparameter_tuning import run_hyperparameter_tuning, load_best_hyperparameters
from diagnostics.hyperparameter_diagnostics import generate_hyperparameter_report, save_hyperparameter_results

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    test_mode: bool = False,
    test_years: int = 1,
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    detection_diagnostics: bool = False,
    run_hyperparameter_optimization: bool = False,
    hyperparameter_trials: int = 10,
    hyperparameter_diagnostics: bool = True,
    memory_efficient: bool = False,
    aggressive_memory_saving: bool = False
):
    """
    Run the complete error detection and imputation pipeline using yearly windows.
    """
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print("\n" + "="*80)
        print(f"CUDA IS AVAILABLE! Using GPU: {device_name}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get GPU memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        free_memory = total_memory - allocated_memory
        
        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Free GPU memory: {free_memory:.2f} GB")
        
        if free_memory < 4.0:
            print(f"Limited GPU memory detected ({free_memory:.2f} GB free).")
            aggressive_memory_saving = True
            print("Enabling aggressive memory saving mode automatically")
    
    # Start with base configuration from config.py
    model_config = LSTM_CONFIG.copy()
    
    # Adjust configuration based on memory constraints if needed
    if aggressive_memory_saving:
        print("AGGRESSIVE MEMORY SAVING MODE ACTIVE - Adjusting model configuration")
        model_config['hidden_size'] = min(128, model_config['hidden_size'])
        model_config['num_layers'] = min(2, model_config['num_layers'])
        print(f"Adjusted configuration for memory efficiency:")
        print(f"  - hidden_size: {model_config['hidden_size']}")
        print(f"  - num_layers: {model_config['num_layers']}")
    
    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    station_id = '21006846'
    print(f"Loading and preprocessing station data for station {station_id}...")
   
    # Load Preprocessed data
    data_dir = project_root / "data_utils" / "Sample data"
    preprocessed_data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
    freezing_periods = pd.read_pickle(data_dir / "frost_periods.pkl")
    
    # Generate dictionary keeping same structure but with the specified station_id
    preprocessed_data = {station_id: preprocessed_data[station_id]} if station_id in preprocessed_data else {}
    
    # Skip preprocessing diagnostics if no data found for the station
    if not preprocessed_data:
        print(f"Warning: No data found for station {station_id} in preprocessed data")
        preprocess_diagnostics = False
    
    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics:
        print("Generating preprocessing diagnostics...")
        original_data = pd.read_pickle(data_dir / "original_data.pkl")
        original_data = {station_id: original_data[station_id]} if station_id in original_data else {}
        
        if original_data:
            # Convert freezing_periods to list if it's not already
            frost_periods = freezing_periods if isinstance(freezing_periods, list) else []
            plot_preprocessing_comparison(original_data, preprocessed_data, Path(output_path), frost_periods)
            plot_additional_data(preprocessed_data, Path(output_path), original_data)
            plot_station_data_overview(original_data, preprocessed_data, Path(output_path))
        else:
            print(f"Warning: No original data found for station {station_id}")
            # Still generate plots that don't require original data
            plot_additional_data(preprocessed_data, Path(output_path))
    
    #########################################################
    #    Step 2: Split data into train/val/test sets        #
    #########################################################
    
    print("\nSplitting data into train/validation/test sets...")
    split_datasets = split_data_with_combined_windows(
        preprocessed_data,
        val_years=1,
        test_years=2
    )
    
    # Use the split data directly
    train_data = split_datasets['train']
    val_data = split_datasets['validation']
    test_data = split_datasets['test']
    
    print("\nData split summary:")
    for split_name, split_data in [('Training', train_data), ('Validation', val_data), ('Test', test_data)]:
        for station_id in split_data:
            for feature in split_data[station_id]:
                data = split_data[station_id][feature]
                print(f"{split_name} - {station_id} - {feature}:")
                print(f"  Points: {len(data)}")
                print(f"  Range: {data.index.min()} to {data.index.max()}")
    
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
                test_data_synthetic = station_data['vst_raw']
                
                if test_data_synthetic is None or test_data_synthetic.empty:
                    print(f"No test data available for station {station}")
                    continue
                
                # Generate synthetic errors
                modified_data, ground_truth = error_generator.inject_all_errors(test_data_synthetic)
                
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
    
    # Generate synthetic error diagnostics if enabled
    if synthetic_diagnostics:
        from diagnostics.synthetic_diagnostics import run_all_synthetic_diagnostics
        
        synthetic_diagnostic_results = run_all_synthetic_diagnostics(
            split_datasets=split_datasets,
            stations_results=stations_results,
            output_dir=Path(output_path)
        )
    
    #########################################################
    # Step 4: Hyperparameter tuning for LSTM                #
    #########################################################
    print("\nStep 4: Hyperparameter tuning for LSTM...")
    
    # Run hyperparameter optimization if enabled
    if run_hyperparameter_optimization:
        print(f"\nRunning hyperparameter optimization with {hyperparameter_trials} trials...")
        try:
            best_config, study = run_hyperparameter_tuning(
                split_datasets=split_datasets,
                stations_results=stations_results,
                output_path=Path(output_path),
                base_config=model_config,
                n_trials=hyperparameter_trials,
                diagnostics=hyperparameter_diagnostics,
                data_sample_ratio=0.3
            )
            # Update configuration with optimized parameters
            model_config.update(best_config)
            print("\nUsing optimized hyperparameters:")
            for param, value in best_config.items():
                print(f"  {param}: {value}")
        except Exception as e:
            print(f"\nError during hyperparameter optimization: {str(e)}")
            print("Continuing with base configuration")
            import traceback
            traceback.print_exc()
    else:
        print("\nUsing base configuration from config.py:")
        for param, value in model_config.items():
            print(f"  {param}: {value}")
    
    #########################################################
    # Step 5: LSTM training and prediction                  #
    #########################################################
    
    print("\nStep 5: Training LSTM models with Station-Specific Approach...")
    
    # Initialize model
    print("\nInitializing LSTM model...")
    
    # First create a temporary model with the original feature count
    temp_model = SimpleLSTMModel(
        input_size=len(model_config['feature_cols']),
        hidden_size=model_config['hidden_size'],
        output_size=1,
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Initialize trainer with temporary model
    temp_trainer = train_LSTM(temp_model, model_config)
    
    # Prepare data to get the actual number of features after adding lagged features
    X_train, _ = temp_trainer.prepare_data(train_data, is_training=True)
    actual_input_size = X_train.shape[2]  # Get the actual number of features
    
    print(f"Actual input size after adding lagged features: {actual_input_size}")
    
    # Now create the real model with the correct input size
    model = SimpleLSTMModel(
        input_size=actual_input_size,
        hidden_size=model_config['hidden_size'],
        output_size=1,
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Initialize the real trainer with the correct model
    trainer = train_LSTM(model, model_config)
    
    # Train model on combined data
    print("\nTraining model on combined data...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        patience=model_config['patience']
    )
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    # Make predictions on test set
    print("\nMaking predictions on test set without synthetic errors...")
    test_predictions = trainer.predict(test_data)
    
    # Create and show the plot with correct data
    create_full_plot(test_data, test_predictions, station_id)
    
    return test_predictions, split_datasets


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning LSTM model with configuration from config.py")
    
    # Print the available data files
    data_dir = project_root / "data_utils" / "Sample data"
    print("\nAvailable data files:")
    for file in data_dir.glob("*"):
        print(f"  {file.name}")
    
    # Run pipeline with simplified configuration handling
    try:
        test_predictions, split_datasets = run_pipeline(
            project_root=project_root,
            data_path=data_path, 
            output_path=output_path,
            preprocess_diagnostics=False,
            split_diagnostics=False,
            synthetic_diagnostics=False,
            detection_diagnostics=False,
            run_hyperparameter_optimization=False,  # Set to True if you want to optimize
            hyperparameter_trials=10,
            hyperparameter_diagnostics=False,
            memory_efficient=True
        )

        print("\nModel run completed!")
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
    