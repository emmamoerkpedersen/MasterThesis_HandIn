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
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the parent directory to Python path to allow imports from experiments
sys.path.append(str(Path(__file__).parent))

from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_station_data_overview, plot_vst_vinge_comparison
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_scaled_predictions, plot_convergence,  plot_features_stacked_plots
from config import SYNTHETIC_ERROR_PARAMS
from config import LSTM_CONFIG
from _3_lstm_model.model_diagnostics import generate_all_diagnostics, generate_comparative_diagnostics
from experiments.error_frequency import run_error_frequency_experiments

from experiments.Improved_model_structure.train_model import LSTM_Trainer
from experiments.Improved_model_structure.model import LSTMModel

#from _3_lstm_model.model import LSTMModel
# from _3_lstm_model.train_model import DataPreprocessor, LSTM_Trainer
# from _3_lstm_model.model_plots import create_full_plot, plot_scaled_predictions, plot_convergence

# Function to calculate NSE (Nash-Sutcliffe Efficiency)
def calculate_nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    inject_synthetic_errors: bool = False,
    model_diagnostics: bool = True,
    advanced_diagnostics: bool = False,  # New parameter for advanced diagnostics
    error_frequency: float = 0.1,  # Added parameter to control error frequency
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
    
    # Debug prints to understand data structure
    #print("\nData Structure Analysis:")
    #print("Train data columns:", train_data.columns.tolist())
    #print("\nSample of train_data:")
    #print(train_data.head())
    #print("\nData types:")
    #print(train_data.dtypes)

    #print("\nData split summary:")
    #for feature in train_data.columns:
    #    min_val = train_data[feature].min()
    #    max_val = train_data[feature].max()
    #    print(f"{feature}: Min = {min_val}, Max = {max_val}")
    #print(f"Validation data: {val_data.shape}")
    #print(f'Percentage of vst_raw NaN in train target data: {np.round((train_data["vst_raw"].isna().sum() / len(train_data["vst_raw"]))*100, 2)}%')
    #print(f'Percentage of vst_raw NaN in val target data: {np.round((val_data["vst_raw"].isna().sum() / len(val_data["vst_raw"]))*100, 2)}%')
    #print(f'Percentage of vst_raw NaN in test target data: {np.round((test_data["vst_raw"].isna().sum() / len(test_data["vst_raw"]))*100, 2)}%')
    
    # Plot features
    if model_diagnostics:
        feature_plot_dir = Path(output_path) / "feature_plots"
        feature_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_features_stacked_plots(train_data, preprocessor.feature_cols, output_dir=feature_plot_dir)

    # # Plot scaled vs unscaled features for visualization
    # print("\nGenerating scaled vs unscaled features plot for control...")
    
    # # Get the original data before scaling
    # original_data = train_data.copy()
    
    # # Get the scaled data by applying the scaler
    # features = pd.concat([train_data[col] for col in preprocessor.feature_cols], axis=1)
    # target = pd.DataFrame(train_data[preprocessor.output_features])
    
    # # Scale the data
    # scaled_features, scaled_target = preprocessor.feature_scaler.fit_transform(features, target)
    
    # # Create a DataFrame with the scaled data
    # scaled_data = pd.DataFrame(scaled_features, columns=preprocessor.feature_cols, index=train_data.index)
    # scaled_data[preprocessor.output_features] = scaled_target
    
    # # Create the plot
    # plot_scaled_vs_unscaled_features(
    #     data=original_data,
    #     scaled_data=scaled_data,
    #     feature_cols=preprocessor.feature_cols + [preprocessor.output_features],
    #     output_dir=Path(output_path) / "lstm"
    # )

    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics:
        print("Generating preprocessing diagnostics...")
        data_dir = project_root / "results" / "preprocessing_diagnostics"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load the original and preprocessed data directly from pickles
            original_data = pd.read_pickle(project_root / "data_utils" / "Sample data" / "original_data.pkl")
            preprocessed_data = pd.read_pickle(project_root / "data_utils" / "Sample data" / "preprocessed_data.pkl")
            
            # Filter for just our station
            original_data = {station_id: original_data[station_id]} if station_id in original_data else {}
            preprocessed_data = {station_id: preprocessed_data[station_id]} if station_id in preprocessed_data else {}
            
            # Generate preprocessing plots
            if original_data and preprocessed_data:
                print("Generating preprocessing plots...")
                plot_preprocessing_comparison(original_data, preprocessed_data, Path(output_path), [])
                plot_station_data_overview(original_data, preprocessed_data, Path(output_path))
                plot_vst_vinge_comparison(preprocessed_data, Path(output_path), original_data)
            else:
                print(f"Warning: No data found for station {station_id}")
                
        except Exception as e:
            print(f"Error generating preprocessing diagnostics: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping preprocessing diagnostics generation...")
    
    #########################################################
    #    Step 2: Generate synthetic errors                  #
    #########################################################
    
    print("\nStep 2: Generating synthetic errors...")
    
    # Dictionary to store results for each station/year
    stations_results = {}
    # Create synthetic error generator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    # Track whether we successfully created data with errors
    train_data_with_errors = None
    val_data_with_errors = None
    
    if inject_synthetic_errors:
        print(f"\nInjecting synthetic errors with frequency {error_frequency*100:.1f}% into TRAINING and VALIDATION data...")
        try:
            # Modify the error config to use our error_frequency parameter
            error_config = SYNTHETIC_ERROR_PARAMS.copy()
            
            # Set error frequencies based on the error_frequency parameter
            # Distribute the total error frequency across different error types
            error_config['offset']['frequency'] = error_frequency * 0.3  # 30% of errors are offsets
            error_config['drift']['frequency'] = error_frequency * 0.2   # 20% of errors are drifts
            error_config['flatline']['frequency'] = error_frequency * 0.2  # 20% of errors are flatlines
            error_config['spike']['frequency'] = error_frequency * 0.15   # 15% of errors are spikes
            error_config['noise']['frequency'] = error_frequency * 0.15   # 15% of errors are noise
            
            print(f"Error frequencies by type:")
            print(f"  Offset: {error_config['offset']['frequency']:.5f}")
            print(f"  Drift: {error_config['drift']['frequency']:.5f}")
            print(f"  Flatline: {error_config['flatline']['frequency']:.5f}")
            print(f"  Spike: {error_config['spike']['frequency']:.5f}")
            print(f"  Noise: {error_config['noise']['frequency']:.5f}")
            
            # Create a new error generator with the modified config
            error_generator = SyntheticErrorGenerator(error_config)
            
            # Identify which columns contain water level data that should have errors
            water_level_cols = ['feature_station_21006845_vst_raw', 'feature_station_21006847_vst_raw']
            feature_cols_with_errors = [col for col in water_level_cols if col in train_data.columns]
            
            print(f"Injecting errors into these water level columns: {feature_cols_with_errors}")
            
            # Make copies of train and validation data to modify
            train_data_with_errors = train_data.copy()
            val_data_with_errors = val_data.copy()
            
            # First process training data
            print("\nProcessing TRAINING data...")
            for column in feature_cols_with_errors:
                print(f"\nProcessing training {column}...")
                
                # Create a single-column DataFrame for the error generator
                column_data = pd.DataFrame({
                    'vst_raw': train_data[column]  # Error generator expects 'vst_raw' column
                })
                
                # Skip if column contains only NaN values
                if column_data['vst_raw'].isna().all():
                    print(f"Column {column} contains only NaN values, skipping")
                    continue
                    
                # Generate synthetic errors
                print(f"Generating synthetic errors for training {column}...")
                modified_data, ground_truth = error_generator.inject_all_errors(column_data)
                
                # Store the error periods
                station_key = f"{station_id}_train_{column}"
                stations_results[station_key] = {
                    'modified_data': modified_data,
                    'ground_truth': ground_truth,
                    'error_periods': error_generator.error_periods.copy()  # Important to copy!
                }
                
                print(f"Injected {len(error_generator.error_periods)} error periods into training {column}")
                
                # Update the training data with errors
                train_data_with_errors[column] = modified_data['vst_raw']
            
            # Reset error generator for validation data
            error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
            
            # Then process validation data
            print("\nProcessing VALIDATION data...")
            for column in feature_cols_with_errors:
                print(f"\nProcessing validation {column}...")
                
                # Create a single-column DataFrame for the error generator
                column_data = pd.DataFrame({
                    'vst_raw': val_data[column]  # Error generator expects 'vst_raw' column
                })
                
                # Skip if column contains only NaN values
                if column_data['vst_raw'].isna().all():
                    print(f"Column {column} contains only NaN values, skipping")
                    continue
                    
                # Generate synthetic errors
                print(f"Generating synthetic errors for validation {column}...")
                modified_data, ground_truth = error_generator.inject_all_errors(column_data)
                
                # Store the error periods
                station_key = f"{station_id}_val_{column}"
                stations_results[station_key] = {
                'modified_data': modified_data,
                'ground_truth': ground_truth,
                    'error_periods': error_generator.error_periods.copy()  # Important to copy!
                }
                
                print(f"Injected {len(error_generator.error_periods)} error periods into validation {column}")
                
                # Update the validation data with errors
                val_data_with_errors[column] = modified_data['vst_raw']
            
            print("\nSynthetic error injection complete.")
            print(f"Created training and validation datasets with synthetic errors in {len(feature_cols_with_errors)} feature columns")
            
        except Exception as e:
            print(f"Error processing synthetic errors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate synthetic error diagnostics if enabled
    if synthetic_diagnostics:
        from diagnostics.synthetic_diagnostics import run_all_synthetic_diagnostics
        
        print("\nGenerating synthetic error diagnostics...")
        synthetic_diagnostic_results = run_all_synthetic_diagnostics(
            split_datasets={'windows': {'train': {station_id: train_data}, 'val': {station_id: val_data}}},
            stations_results=stations_results,
            output_dir=Path(output_path)
        )
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

    # Now create the real model with the correct input size
    model = LSTMModel(
        input_size=input_size,  # Use dynamically calculated input size
        sequence_length=model_config['sequence_length'],  
        hidden_size=model_config['hidden_size'],
        output_size=len(model_config['output_features']),
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Verify the loaded parameters match what we expect
    print("\nVerifying hyperparameters:")
    expected_params = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 
                      'batch_size', 'sequence_length', 'peak_weight', 
                      'grad_clip_value', 'smoothing_alpha']
    for param in expected_params:
        print(f"  {param}: {model_config.get(param)}")
    
    # Initialize the trainer with the verified config
    trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
    
    # Print model architecture
    #print("\nModel Architecture:")
    #print(f"Input Size: {len(preprocessor.feature_cols)}")
    #print(f"Hidden Size: {model_config['hidden_size']}")
    #print(f"Number of Layers: {model_config['num_layers']}")
    #print(f"Dropout Rate: {model_config['dropout']}")
    #print(f"Learning Rate: {model_config['learning_rate']}")
    #print(f"Batch Size: {model_config['batch_size']}")
    #print(f"Sequence Length: {model_config['sequence_length']}")
    
    # Train model with verified parameters
    #print("\nStarting training with verified parameters...")
    #print("Using identical conditions as hyperparameter tuning")
    #print(f"Epochs: {model_config['epochs']}")
    #print(f"Batch size: {model_config['batch_size']}")
    #print(f"Patience: {model_config['patience']}")
    #print(f"Learning rate: {model_config['learning_rate']}")
    
    # Determine which data to use for training and validation
    if inject_synthetic_errors and train_data_with_errors is not None and val_data_with_errors is not None:
        print("\nUsing TRAINING and VALIDATION data with synthetic errors for model training")
        training_data = train_data_with_errors
        validation_data = val_data_with_errors
    else:
        print("\nUsing clean training and validation data (no synthetic errors)")
        training_data = train_data
        validation_data = val_data
    
    history, val_predictions, val_targets = trainer.train(
        train_data=training_data,
        val_data=validation_data,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        patience=model_config['patience']
    )
    
    print("\nTraining Results:")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    if 'smoothed_val_loss' in history:
        print(f"Best smoothed validation loss: {min(history['smoothed_val_loss']):.6f}")
        print(f"Final smoothed validation loss: {history['smoothed_val_loss'][-1]:.6f}")

    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Convert validation predictions to numpy and reshape
    val_predictions = val_predictions.cpu().numpy()
    val_targets = val_targets.cpu().numpy()
    
    # Preserve temporal order during inverse transform (same as predict function)
    predictions_reshaped = val_predictions.reshape(-1, 1)
    predictions_original = preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
    predictions_flattened = predictions_original.flatten()  # Ensure 1D array
    
    # Print diagnostic information about shapes
    #print(f"\nShape diagnostics:")
    #print(f"Validation data length: {len(validation_data)}")
    #print(f"Predictions length: {len(predictions_flattened)}")
    
    # Ensure predictions match validation data length
    if len(predictions_flattened) < len(validation_data):
        # If predictions are shorter, pad with NaN values
        print(f"Padding predictions with NaN values to match validation data length")
        padding = np.full(len(validation_data) - len(predictions_flattened), np.nan)
        predictions_flattened = np.concatenate([predictions_flattened, padding])
    elif len(predictions_flattened) > len(validation_data):
        # If predictions are longer, truncate to match validation data length
        print(f"Truncating predictions to match validation data length")
        predictions_flattened = predictions_flattened[:len(validation_data)]
    
    # Create DataFrame with aligned predictions and targets
    val_predictions_df = pd.DataFrame(
        predictions_flattened,
        index=validation_data.index,
        columns=['vst_raw']
    )
    
    # Create a plot title that indicates whether training used clean or error-injected data
    val_plot_title = "Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Trained on Clean Data"
    
    # Get the best validation loss
    best_val_loss = min(history['val_loss'])
    
    # Plot validation results
    if model_diagnostics and not inject_synthetic_errors:
        create_full_plot(validation_data, val_predictions_df, str(station_id), model_config, best_val_loss, title_suffix=val_plot_title)
        
        # Plot convergence
        plot_convergence(history, str(station_id), title=f"Training and Validation Loss - Station {station_id}")
    
    # Make and plot test predictions
    print("\nMaking predictions on test set (clean data)...")
    test_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
    
    # Convert test predictions to DataFrame for plotting
    test_predictions_reshaped = test_predictions.flatten() # Ensure 1D array

    # Ensure predictions match test data length
    #print(f"Test data length: {len(test_data)}")
    if len(test_predictions_reshaped) > len(test_data):
        # If predictions are longer, truncate to match test data length
        #print(f"Truncating test predictions to match test data length")
        test_predictions_reshaped = test_predictions_reshaped[:len(test_data)]
    elif len(test_predictions_reshaped) < len(test_data):
        #print(f"Padding test predictions with NaN values to match test data length")
        padding = np.full(len(test_data) - len(test_predictions_reshaped), np.nan)
        test_predictions_reshaped = np.concatenate([test_predictions_reshaped, padding])
        test_predictions_reshaped = test_predictions_reshaped.flatten() # Ensure 1D after padding

    test_predictions_df = pd.DataFrame(
        test_predictions_reshaped,
        index=test_data.index,
        columns=['vst_raw']
    )
    
    # Create a plot title that indicates whether the model was trained on clean or error-injected data 
    test_plot_title = "Model Trained on Data with Synthetic Errors" if inject_synthetic_errors else "Model Trained on Clean Data"
    
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
        clean_model = LSTMModel(
            input_size=input_size,
            sequence_length=model_config['sequence_length'],
            hidden_size=model_config['hidden_size'],
            output_size=len(model_config['output_features']),
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        
        # Initialize a new trainer
        clean_trainer = LSTM_Trainer(model_config, preprocessor=preprocessor)
        
        # Train the model with clean data
        clean_history, clean_val_predictions, clean_val_targets = clean_trainer.train(
            train_data=train_data,  # Using clean training data
            val_data=val_data,      # Using clean validation data
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            patience=model_config['patience']
        )
        
        # Conditional block for validation data related operations
        if val_data is not None:
            # Make predictions on validation data
            clean_val_predictions_np = clean_val_predictions.cpu().numpy()

            # Process predictions
            clean_predictions_reshaped = clean_val_predictions_np.reshape(-1, 1)
            clean_predictions_original = preprocessor.feature_scaler.inverse_transform_target(clean_predictions_reshaped)
            clean_predictions_flattened = clean_predictions_original.flatten()

            # Ensure predictions match validation data length
            if len(clean_predictions_flattened) < len(val_data):
                # If predictions are shorter, pad with NaN values
                padding = np.full(len(val_data) - len(clean_predictions_flattened), np.nan)
                clean_predictions_flattened = np.concatenate([clean_predictions_flattened, padding])
            elif len(clean_predictions_flattened) > len(val_data):
                # If predictions are longer, truncate to match validation data length
                clean_predictions_flattened = clean_predictions_flattened[:len(val_data)]

            # Create DataFrame with aligned predictions and targets
            clean_val_predictions_df = pd.DataFrame(
                clean_predictions_flattened,
                index=val_data.index,
                columns=['vst_raw']
            )

            # Only create comparison plot if model_diagnostics is True
            if model_diagnostics:
                # Create comparison plot between clean and error-injected model predictions
                print("\nCreating comparison plot between models trained on clean vs error-injected data...")
                plt.figure(figsize=(15, 12))

                # Plot 1: Validation data (clean vs with errors)
                plt.subplot(4, 1, 1)
                plt.plot(val_data.index, val_data['vst_raw'], label='Clean Validation Data', alpha=0.7)
                plt.title('Clean Validation Data')
                plt.legend()
                plt.grid(True)

                plt.subplot(4, 1, 2)
                plt.plot(val_data_with_errors.index, val_data_with_errors['vst_raw'], label='Validation Data with Errors', alpha=0.7)
                plt.title('Validation Data with Synthetic Errors')
                plt.legend()
                plt.grid(True)

                # Plot 3: Validation predictions for both models
                plt.subplot(4, 1, 3)
                plt.plot(clean_val_predictions_df.index, clean_val_predictions_df['vst_raw'],
                        label='Model Trained on Clean Data', alpha=0.7)
                plt.plot(val_predictions_df.index, val_predictions_df['vst_raw'],
                        label='Model Trained on Error Data', alpha=0.7, linestyle='--')
                plt.title('Validation Predictions: Clean-Trained vs Error-Trained Model')
                plt.legend()
                plt.grid(True)

                # Plot 4: Test predictions for both models (Placeholder, actual calculation below)
                plt.subplot(4, 1, 4)
                plt.title('Test Predictions Comparison (See Metrics Table)')
                plt.grid(True)
                plt.tight_layout()
                
                # Save the comparison plot
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_plot_path = Path(output_path) / f"comparison_plot_{error_frequency:.2f}_{timestamp_str}.png"
                plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"\nSaved comparison plot to: {comparison_plot_path}")

        # End of val_data specific block

        # --- Test Set Comparison (Always run if inject_synthetic_errors is True) ---

        # Get predictions on test data from the clean-trained model
        clean_test_predictions, _, _ = clean_trainer.predict(test_data)
        clean_test_predictions_reshaped = clean_test_predictions.flatten()[:len(test_data)] # Ensure 1D and correct length

        # Ensure compatible shapes before comparison
        test_data_nan_mask = ~np.isnan(test_data['vst_raw']).values

        # Ensure prediction arrays are 1D before creating mask
        if len(clean_test_predictions_reshaped.shape) > 1:
            clean_test_predictions_reshaped = clean_test_predictions_reshaped.flatten()
        if len(test_predictions_reshaped.shape) > 1:
             test_predictions_reshaped = test_predictions_reshaped.flatten()

        predictions_nan_mask = ~np.isnan(clean_test_predictions_reshaped) # Based on clean model predictions for consistency
        valid_mask = test_data_nan_mask & predictions_nan_mask

        # Clean-trained model metrics
        clean_rmse = np.sqrt(mean_squared_error(test_data['vst_raw'].values[valid_mask], clean_test_predictions_reshaped[valid_mask]))
        clean_mae = mean_absolute_error(test_data['vst_raw'].values[valid_mask], clean_test_predictions_reshaped[valid_mask])
        clean_r2 = r2_score(test_data['vst_raw'].values[valid_mask], clean_test_predictions_reshaped[valid_mask])
        clean_nse = calculate_nse(test_data['vst_raw'].values[valid_mask], clean_test_predictions_reshaped[valid_mask])

        # Error-trained model metrics
        error_rmse = np.sqrt(mean_squared_error(test_data['vst_raw'].values[valid_mask], test_predictions_reshaped[valid_mask]))
        error_mae = mean_absolute_error(test_data['vst_raw'].values[valid_mask], test_predictions_reshaped[valid_mask])
        error_r2 = r2_score(test_data['vst_raw'].values[valid_mask], test_predictions_reshaped[valid_mask])
        error_nse = calculate_nse(test_data['vst_raw'].values[valid_mask], test_predictions_reshaped[valid_mask])

        # Store metrics in dictionary (unconditionally when inject_synthetic_errors=True)
        performance_metrics = {
            'error_frequency': error_frequency,
            'clean_model': {
                'val_loss': min(clean_history['val_loss']) if 'clean_history' in locals() else np.nan, # Handle if val_data was None
                'rmse': clean_rmse,
                'mae': clean_mae,
                'r2': clean_r2,
                'nse': clean_nse
            },
            'error_model': {
                'val_loss': min(history['val_loss']), # Assumes history is always available
                'rmse': error_rmse,
                'mae': error_mae,
                'r2': error_r2,
                'nse': error_nse
            },
            'difference': {
                'val_loss': (min(history['val_loss']) - min(clean_history['val_loss'])) if 'clean_history' in locals() else np.nan,
                'rmse': error_rmse - clean_rmse,
                'mae': error_mae - clean_mae,
                'r2': error_r2 - clean_r2,
                'nse': error_nse - clean_nse
            },
            'percent_change': {
                'val_loss': ((min(history['val_loss']) - min(clean_history['val_loss'])) / min(clean_history['val_loss']) * 100) if 'clean_history' in locals() and min(clean_history['val_loss']) != 0 else float('inf'),
                'rmse': (error_rmse - clean_rmse) / clean_rmse * 100 if clean_rmse != 0 else float('inf'),
                'mae': (error_mae - clean_mae) / clean_mae * 100 if clean_mae != 0 else float('inf'),
                'r2': (error_r2 - clean_r2) / clean_r2 * 100 if clean_r2 != 0 else float('inf'),
                'nse': (error_nse - clean_nse) / clean_nse * 100 if clean_nse != 0 else float('inf')
            }
        }

        # Print metrics comparison table (unconditionally when inject_synthetic_errors=True)
        print("\nPerformance Metrics Comparison:")
        print("-" * 80)
        print(f"Error Frequency: {error_frequency * 100:.1f}%")
        print("-" * 80)
        print(f"{'Metric':<15} {'Clean-Trained':<15} {'Error-Trained':<15} {'Difference':<15} {'% Change':<15}")
        print("-" * 80)
        clean_val_loss_disp = performance_metrics['clean_model']['val_loss']
        error_val_loss_disp = performance_metrics['error_model']['val_loss']
        diff_val_loss_disp = performance_metrics['difference']['val_loss']
        pct_val_loss_disp = performance_metrics['percent_change']['val_loss']
        print(f"{'Val Loss':<15} {clean_val_loss_disp:<15.6f} {error_val_loss_disp:<15.6f} {diff_val_loss_disp:<15.6f} {pct_val_loss_disp:<15.2f}")
        print(f"{'RMSE':<15} {clean_rmse:<15.2f} {error_rmse:<15.2f} {performance_metrics['difference']['rmse']:<15.2f} {performance_metrics['percent_change']['rmse']:<15.2f}")
        print(f"{'MAE':<15} {clean_mae:<15.2f} {error_mae:<15.2f} {performance_metrics['difference']['mae']:<15.2f} {performance_metrics['percent_change']['mae']:<15.2f}")
        print(f"{'R²':<15} {clean_r2:<15.4f} {error_r2:<15.4f} {performance_metrics['difference']['r2']:<15.4f} {performance_metrics['percent_change']['r2']:<15.2f}")
        print(f"{'NSE':<15} {clean_nse:<15.4f} {error_nse:<15.4f} {performance_metrics['difference']['nse']:<15.4f} {performance_metrics['percent_change']['nse']:<15.2f}")

        # Save metrics to CSV file (unconditionally when inject_synthetic_errors=True)
        print("\nDEBUG: Attempting to save comparison metrics...") # DEBUG
        metrics_df = pd.DataFrame({
            'error_frequency': [error_frequency],
            'clean_val_loss': [performance_metrics['clean_model']['val_loss']],
            'error_val_loss': [performance_metrics['error_model']['val_loss']],
            'clean_rmse': [clean_rmse],
            'error_rmse': [error_rmse],
            'clean_mae': [clean_mae],
            'error_mae': [error_mae],
            'clean_r2': [clean_r2],
            'error_r2': [error_r2],
            'clean_nse': [clean_nse],
            'error_nse': [error_nse]
        })

        # Include timestamp in the metrics filename to avoid overwriting previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = Path(output_path) / f"error_comparison_metrics_{timestamp}.csv"

        # Also save to the standard filename for the visualization code to find
        standard_metrics_file = Path(output_path) / "error_comparison_metrics.csv"

        print(f"DEBUG: Timestamped metrics file path: {metrics_file}") # DEBUG
        print(f"DEBUG: Standard metrics file path: {standard_metrics_file}") # DEBUG

        try:
            metrics_df.to_csv(metrics_file, index=False)
            print(f"DEBUG: Saved timestamped metrics to {metrics_file}") # DEBUG

            # If standard file exists, append to it, otherwise create new file
            if standard_metrics_file.exists():
                print(f"DEBUG: Standard metrics file {standard_metrics_file} exists. Appending...") # DEBUG
                existing_df = pd.read_csv(standard_metrics_file)
                combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
                combined_df.to_csv(standard_metrics_file, index=False)
                print(f"DEBUG: Appended metrics to {standard_metrics_file}") # DEBUG
            else:
                print(f"DEBUG: Standard metrics file {standard_metrics_file} does not exist. Creating...") # DEBUG
                metrics_df.to_csv(standard_metrics_file, index=False)
                print(f"DEBUG: Created standard metrics file {standard_metrics_file}") # DEBUG

            print(f"\nMetrics saved to: {metrics_file}")
            print(f"Metrics appended to: {standard_metrics_file}")
        except Exception as e:
            print(f"\nERROR: Failed to save metrics CSV files: {e}") # DEBUG
            import traceback
            traceback.print_exc()

        # Only generate diagnostic visualizations if model_diagnostics is True
        if model_diagnostics:
            # Create comparison plot between clean and error-injected model predictions
            print("\nCreating comparison plot between models trained on clean vs error-injected data...")
            plt.figure(figsize=(15, 12))

            # Plot 1: Validation data (clean vs with errors)
            plt.subplot(4, 1, 1)
            plt.plot(val_data.index, val_data['vst_raw'], label='Clean Validation Data', alpha=0.7)
            plt.title('Clean Validation Data')
            plt.legend()
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(val_data_with_errors.index, val_data_with_errors['vst_raw'], label='Validation Data with Errors', alpha=0.7)
            plt.title('Validation Data with Synthetic Errors')
            plt.legend()
            plt.grid(True)

            # Plot 3: Validation predictions for both models
            plt.subplot(4, 1, 3)
            plt.plot(clean_val_predictions_df.index, clean_val_predictions_df['vst_raw'],
                    label='Model Trained on Clean Data', alpha=0.7)
            plt.plot(val_predictions_df.index, val_predictions_df['vst_raw'],
                    label='Model Trained on Error Data', alpha=0.7, linestyle='--')
            plt.title('Validation Predictions: Clean-Trained vs Error-Trained Model')
            plt.legend()
            plt.grid(True)

            # Plot 4: Test predictions for both models (Placeholder, actual calculation below)
            plt.subplot(4, 1, 4)
            plt.title('Test Predictions Comparison (See Metrics Table)')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the comparison plot
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_plot_path = Path(output_path) / f"comparison_plot_{error_frequency:.2f}_{timestamp_str}.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nSaved comparison plot to: {comparison_plot_path}")
        
        # Only generate advanced diagnostic visualizations if advanced_diagnostics is True
        if advanced_diagnostics:
            print("\nGenerating comparative diagnostic visualizations...")
            
            # Create diagnostics output directory
            diagnostics_dir = Path(output_path) / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            
            # Import the consolidated diagnostics function
            try:
                from _3_lstm_model.model_diagnostics import generate_comparative_diagnostics
                
                # Create Series for easier handling in diagnostics functions
                actual_series = pd.Series(test_data['vst_raw'], index=test_data.index)
                
                # Fix the shape issue by ensuring prediction arrays are 1D
                if len(clean_test_predictions_reshaped.shape) > 1:
                    clean_predictions_series = pd.Series(clean_test_predictions_reshaped.flatten(), index=test_data.index)
                else:
                    clean_predictions_series = pd.Series(clean_test_predictions_reshaped, index=test_data.index)
                    
                if len(test_predictions_reshaped.shape) > 1:
                    error_predictions_series = pd.Series(test_predictions_reshaped.flatten(), index=test_data.index)
                else:
                    error_predictions_series = pd.Series(test_predictions_reshaped, index=test_data.index)
                
                # Prepare rainfall data if available
                rainfall_series = None
                if 'rainfall' in test_data.columns:
                    rainfall_series = pd.Series(test_data['rainfall'], index=test_data.index)
                
                # Generate all comparative diagnostics with a single function call
                predictions_dict = {
                    'clean_trained': clean_predictions_series,
                    'error_trained': error_predictions_series
                }
                
                all_visualization_paths = generate_comparative_diagnostics(
                    actual=actual_series,
                    predictions_dict=predictions_dict,
                    output_dir=diagnostics_dir,
                    station_id=station_id,
                    rainfall=rainfall_series,
                    n_event_plots=3  # Analyze top 3 water level events
                )
                
            except Exception as e:
                print(f"Error generating comparative diagnostic visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping comparative diagnostic visualizations (advanced_diagnostics=False)")
        
        return performance_metrics
    
    else:
        # Only process one model (trained on clean or error data based on initial flags)
        # Calculate metrics
        # Get valid indices (non-NaN values)
        # Ensure compatible shapes before comparison
        test_data_nan_mask = ~np.isnan(test_data['vst_raw']).values
        
        # Make sure test_predictions_reshaped is a 1D array of the right length
        if len(test_predictions_reshaped) != len(test_data):
            # Resize if necessary to match test data length
            print(f"Resizing test predictions from {len(test_predictions_reshaped)} to {len(test_data)} elements")
            test_predictions_reshaped = test_predictions_reshaped[:len(test_data)]
        
        # Now safe to combine masks
        predictions_nan_mask = ~np.isnan(test_predictions_reshaped)
        valid_mask = test_data_nan_mask & predictions_nan_mask
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(
            test_data['vst_raw'][valid_mask], 
            test_predictions_reshaped[valid_mask]
        ))
        mae = mean_absolute_error(
            test_data['vst_raw'][valid_mask], 
            test_predictions_reshaped[valid_mask]
        )
        r2 = r2_score(
            test_data['vst_raw'][valid_mask], 
            test_predictions_reshaped[valid_mask]
        )
        nse = calculate_nse(
            test_data['vst_raw'][valid_mask], 
            test_predictions_reshaped[valid_mask]
        )
        
        # Store metrics
        performance_metrics = {
            'model': {
                'val_loss': min(history['val_loss']),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'nse': nse
            }
        }
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print("-" * 50)
        print(f"{'Metric':<15} {'Value':<15}")
        print("-" * 50)
        print(f"{'Val Loss':<15} {min(history['val_loss']):<15.6f}")
        print(f"{'RMSE':<15} {rmse:<15.2f}")
        print(f"{'MAE':<15} {mae:<15.2f}")
        print(f"{'R²':<15} {r2:<15.4f}")
        print(f"{'NSE':<15} {nse:<15.4f}")
        
        # Generate diagnostic visualizations
        if advanced_diagnostics:
            print("\nGenerating advanced diagnostic visualizations...")
            
            # Create diagnostics output directory
            diagnostics_dir = Path(output_path) / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            
            # Import the consolidated diagnostics function
            try:
                from _3_lstm_model.model_diagnostics import generate_all_diagnostics
                
                # Create Series for easier handling in diagnostics functions
                actual_series = pd.Series(test_data['vst_raw'], index=test_data.index)
                
                # Fix the shape issue by ensuring prediction array is 1D
                if len(test_predictions_reshaped.shape) > 1:
                    predictions_series = pd.Series(test_predictions_reshaped.flatten(), index=test_data.index)
                else:
                    predictions_series = pd.Series(test_predictions_reshaped, index=test_data.index)
                
                # Prepare rainfall data if available
                rainfall_series = None
                if 'rainfall' in test_data.columns:
                    rainfall_series = pd.Series(test_data['rainfall'], index=test_data.index)
                
                # Generate all diagnostics with a single function call
                print(f"DEBUG: Calling generate_all_diagnostics for station {station_id}...") # DEBUG
                print(f"DEBUG: Output dir: {diagnostics_dir}") # DEBUG
                print(f"DEBUG: Actual shape: {actual_series.shape}, Prediction shape: {predictions_series.shape}") # DEBUG
                all_visualization_paths = generate_all_diagnostics(
                    actual=actual_series,
                    predictions=predictions_series,
                    output_dir=diagnostics_dir,
                    station_id=station_id,
                    rainfall=rainfall_series,
                    n_event_plots=10  # Analyze top 10 water level events
                )
                
            except Exception as e:
                print(f"Error generating diagnostic visualizations: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping advanced diagnostics visualizations (advanced_diagnostics=False)")
        
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
                synthetic_diagnostics=False,
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
