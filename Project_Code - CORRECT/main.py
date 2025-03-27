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
import numpy as np

from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, plot_additional_data, generate_preprocessing_report, plot_station_data_overview
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from diagnostics.hyperparameter_diagnostics import generate_hyperparameter_report, save_hyperparameter_results
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
#from _3_lstm_model.hyperparameter_tuning import run_hyperparameter_tuning, load_best_hyperparameters

from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG
from _3_lstm_model.model import LSTMModel
from _3_lstm_model.train_model import DataPreprocessor, LSTM_Trainer
from _3_lstm_model.model_plots import create_full_plot, plot_scaled_predictions, plot_convergence


def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    run_hyperparameter_optimization: bool = False,
    hyperparameter_trials: int = 10,
    hyperparameter_diagnostics: bool = False,
    
):
    """
    Run the complete error detection and imputation pipeline using yearly windows.
    """
    # Start with base configuration from config.py
    model_config = LSTM_CONFIG.copy()
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(model_config)

    
    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    station_id = '21006846'
    feature_station_id = '21006845'
    print(f"Loading, preprocessing and splitting station data for station {station_id}...")
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    

    print("\nData split summary:")
    print(f"Train data: {train_data.shape}")
    for feature in train_data.columns:
        min_val = train_data[feature].min()
        max_val = train_data[feature].max()
        print(f"{feature}: Min = {min_val}, Max = {max_val}")
    print(f"Validation data: {val_data.shape}")
    print(f'Percentage of vst_raw NaN in train target data: {np.round((train_data["vst_raw"].isna().sum() / len(train_data["vst_raw"]))*100, 2)}%')
    print(f'Percentage of vst_raw NaN in val target data: {np.round((val_data["vst_raw"].isna().sum() / len(val_data["vst_raw"]))*100, 2)}%')
    print(f'Percentage of vst_raw NaN in test target data: {np.round((test_data["vst_raw"].isna().sum() / len(test_data["vst_raw"]))*100, 2)}%')
    
    print(f"Test data: {test_data.shape}")

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
    #    Step 3: Generate synthetic errors                  #
    #########################################################
    
    print("\nStep 3: Generating synthetic errors... Does not work for now, is not update for new code")
    
    # Dictionary to store results for each station/year
    stations_results = {}
    # Create synthetic error generator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    # Process only test data
    print("\nProcessing test data...")
    for station, station_data in test_data.items():
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
            split_datasets=test_data,
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

    # Now create the real model with the correct input size
    model = LSTMModel(
        input_size=len(model_config['feature_cols']),
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        output_size=len(model_config['output_features']),
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )


    # Initialize the real trainer with the correct model
    trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
    
    # Train model on combined data
    print("\nTraining model on combined data...")
    history, val_predictions, val_targets = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        patience=model_config['patience']
    )

    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Convert validation predictions to numpy and reshape
    val_predictions = val_predictions.cpu().numpy()
    val_targets = val_targets.cpu().numpy()
    
    # Preserve temporal order during inverse transform (same as predict function)
    predictions_reshaped = val_predictions.reshape(-1, 1)
    predictions_original = preprocessor.scalers['target'].inverse_transform(predictions_reshaped)
    predictions_flattened = predictions_original.flatten()  # Ensure 1D array
    
    # Create DataFrame with aligned predictions and targets
    val_predictions_df = pd.DataFrame(
        predictions_flattened,  # Use flattened 1D array
        index=val_data.index[:len(predictions_flattened)],
        columns=['vst_raw']
    )
    # Now plot with aligned data
    create_full_plot(val_data, val_predictions_df, station_id)  # Pass the Series directly
    
    # Make and plot test predictions
    print("\nMaking predictions on test set...")
    test_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
    
    # Plot convergence
    plot_convergence(history, title=f"Training and Validation Loss - Station {station_id}")

    return test_predictions, predictions_original, history


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning LSTM model with configuration from config.py")

    
    # Run pipeline with simplified configuration handling
    try:
        test_predictions, val_predictions, history = run_pipeline(
            project_root=project_root,
            data_path=data_path, 
            output_path=output_path,
            preprocess_diagnostics=False,
            synthetic_diagnostics=False,
            run_hyperparameter_optimization=False,
            hyperparameter_trials=10,
            hyperparameter_diagnostics=False,
        )

        print("\nModel run completed!")
        print(f"Results saved to: {output_path}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history['val_loss']):.6f}")
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
    