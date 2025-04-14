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

# Add the parent directory to Python path to allow imports from experiments
sys.path.append(str(Path(__file__).parent))

from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_station_data_overview, plot_vst_vinge_comparison
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from diagnostics.hyperparameter_diagnostics import generate_hyperparameter_report, save_hyperparameter_results
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_scaled_predictions, plot_convergence, create_performance_analysis_plot, plot_scaled_vs_unscaled_features, plot_features_stacked_plots
from config import SYNTHETIC_ERROR_PARAMS
from config import LSTM_CONFIG


from experiments.Improved_model_structure.hyperparameter_tuning import run_hyperparameter_tuning, load_best_hyperparameters
from experiments.Improved_model_structure.train_model import LSTM_Trainer
#from experiments.Improved_model_structure.model import LSTMModel


from _3_lstm_model.model import LSTMModel
# from _3_lstm_model.train_model import DataPreprocessor, LSTM_Trainer
# from _3_lstm_model.model_plots import create_full_plot, plot_scaled_predictions, plot_convergence

def run_pipeline(
    project_root: Path,
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    synthetic_diagnostics: bool = False,
    run_hyperparameter_optimization: bool = False,
    hyperparameter_trials: int = 20,  # Reduced to 20 for faster results
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
    print(f"Loading, preprocessing and splitting station data for station {station_id}...")
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Debug prints to understand data structure
    print("\nData Structure Analysis:")
    print("Train data columns:", train_data.columns.tolist())
    print("\nSample of train_data:")
    print(train_data.head())
    print("\nData types:")
    print(train_data.dtypes)

    print("\nData split summary:")
    for feature in train_data.columns:
        min_val = train_data[feature].min()
        max_val = train_data[feature].max()
        print(f"{feature}: Min = {min_val}, Max = {max_val}")
    print(f"Validation data: {val_data.shape}")
    print(f'Percentage of vst_raw NaN in train target data: {np.round((train_data["vst_raw"].isna().sum() / len(train_data["vst_raw"]))*100, 2)}%')
    print(f'Percentage of vst_raw NaN in val target data: {np.round((val_data["vst_raw"].isna().sum() / len(val_data["vst_raw"]))*100, 2)}%')
    print(f'Percentage of vst_raw NaN in test target data: {np.round((test_data["vst_raw"].isna().sum() / len(test_data["vst_raw"]))*100, 2)}%')
    
    
    # Plot features
    #plot_features_stacked_plots(train_data, preprocessor.feature_cols)


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
                train_data=train_data,
                val_data=val_data,
                output_path=Path(output_path),
                base_config=model_config,
                n_trials=hyperparameter_trials
            )
            # Update configuration with optimized parameters
            model_config = best_config  # Replace with entire best config
            print("\nUsing optimized hyperparameters:")
            for param, value in best_config.items():
                if param in ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']:
                    print(f"  {param}: {value}")
            
            # Generate hyperparameter visualization reports if diagnostics are enabled
            if hyperparameter_diagnostics:
                print("\nGenerating hyperparameter tuning diagnostics...")
                study_path = Path(output_path) / "hyperparameter_tuning" / "all_trials.json"
                if study_path.exists():
                    generate_hyperparameter_report(
                        study_path=study_path,
                        output_dir=Path(output_path) / "hyperparameter_diagnostics",
                        top_n=5  # Show top 5 models
                    )
                else:
                    print(f"Could not find hyperparameter trials data at {study_path}")
            
        except Exception as e:
            print(f"\nError during hyperparameter optimization: {str(e)}")
            print("Continuing with base configuration")
            import traceback
            traceback.print_exc()
    else:
        # Try to load best hyperparameters from previous runs
        try:
            loaded_config = load_best_hyperparameters(Path(output_path), model_config)
            if loaded_config != model_config:
                model_config = loaded_config
                print("\nLoaded hyperparameters from previous tuning:")
                for param, value in model_config.items():
                    if param in ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']:
                        print(f"  {param}: {value}")
            else:
                print("\nUsing base configuration from config.py")
        except Exception as e:
            print(f"\nError loading previous hyperparameters: {str(e)}")
            print("Using base configuration")
    
    #########################################################
    # Step 5: LSTM training and prediction                  #
    #########################################################
    
    print("\nStep 5: Training LSTM models with Station-Specific Approach...")
    
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
    
    # Load best hyperparameters
    best_config = load_best_hyperparameters(Path(output_path), model_config)
    
    # Verify the loaded parameters match what we expect
    print("\nVerifying hyperparameters:")
    expected_params = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 
                      'batch_size', 'sequence_length', 'peak_weight', 
                      'grad_clip_value', 'smoothing_alpha']
    for param in expected_params:
        print(f"  {param}: {best_config.get(param)}")
    
    # Initialize the trainer with the verified config
    trainer = LSTM_Trainer(LSTM_CONFIG, preprocessor=preprocessor)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(f"Input Size: {len(preprocessor.feature_cols)}")
    print(f"Hidden Size: {best_config['hidden_size']}")
    print(f"Number of Layers: {best_config['num_layers']}")
    print(f"Dropout Rate: {best_config['dropout']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Sequence Length: {best_config['sequence_length']}")
    
    # Train model with verified parameters
    print("\nStarting training with verified parameters...")
    print("Using identical conditions as hyperparameter tuning")
    print(f"Epochs: {best_config['epochs']}")
    print(f"Batch size: {best_config['batch_size']}")
    print(f"Patience: {best_config['patience']}")
    print(f"Learning rate: {best_config['learning_rate']}")
    
    history, val_predictions, val_targets = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=best_config['epochs'],  # Fixed to match tuning
        batch_size=best_config['batch_size'],
        patience=best_config['patience']  # Fixed to match tuning
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
    print(f"\nShape diagnostics:")
    print(f"Validation data length: {len(val_data)}")
    print(f"Predictions length: {len(predictions_flattened)}")
    
    # Ensure predictions match validation data length
    if len(predictions_flattened) < len(val_data):
        # If predictions are shorter, pad with NaN values
        print(f"Padding predictions with NaN values to match validation data length")
        padding = np.full(len(val_data) - len(predictions_flattened), np.nan)
        predictions_flattened = np.concatenate([predictions_flattened, padding])
    elif len(predictions_flattened) > len(val_data):
        # If predictions are longer, truncate to match validation data length
        print(f"Truncating predictions to match validation data length")
        predictions_flattened = predictions_flattened[:len(val_data)]
    
    # Create DataFrame with aligned predictions and targets
    val_predictions_df = pd.DataFrame(
        predictions_flattened,  # Use adjusted 1D array  
        index=val_data.index,
        columns=['vst_raw']
    )
    # Now plot with aligned data - make sure station_id is a string
    best_val_loss = min(history['val_loss'])
    create_full_plot(val_data, val_predictions_df, str(station_id), model_config, best_val_loss)  # Pass model config and best val loss
    

    # Plot scaled predictions to check if they are correct   
    # print("\nPlotting scaled validation predictions...")
    # val_predictions, predictions_scaled, target_scaled = trainer.predict(val_data)
    # plot_scaled_predictions(predictions_scaled, target_scaled, test_data=val_data, title="Scaled Validation Predictions vs Targets")


    # # Create comprehensive performance analysis plot
    # print("\nGenerating comprehensive performance analysis plot...")
    # test_actual = pd.Series(
    #     val_data['vst_raw'].values,
    #     index=val_data.index,
    #     name='Actual'
    # )
    # val_predictions_series = pd.Series(
    #     predictions_flattened,
    #     index=val_data.index[:len(predictions_flattened)],
    #     name='Predicted'
    #)
    # performance_metrics = create_performance_analysis_plot(
    #     test_actual, 
    #     val_predictions_series, 
    #     str(station_id), 
    #     model_config,
    #     Path(output_path) / "lstm"
    # )
    
    # Print performance metrics
    # print("\nModel Performance Metrics:")
    # print(f"RMSE: {performance_metrics['rmse']:.4f} mm")
    # print(f"MAE: {performance_metrics['mae']:.4f} mm")
    # print(f"R²: {performance_metrics['r2']:.4f}")
    # print(f"Mean Error: {performance_metrics['mean_error']:.4f} mm")
    # print(f"Std Error: {performance_metrics['std_error']:.4f} mm")
    
    # Make and plot test predictions
    print("\nMaking predictions on test set...")
    test_predictions, predictions_scaled, target_scaled = trainer.predict(test_data)
  
    
    # Convert test predictions to DataFrame for plotting
    test_predictions_reshaped = test_predictions.reshape(-1, 1) if len(test_predictions.shape) > 1 else test_predictions.reshape(-1)
    test_predictions_df = pd.DataFrame(
        test_predictions_reshaped,  # Already flattened
        index=test_data.index[:len(test_predictions_reshaped)],
        columns=['vst_raw']
    )
    
    # Plot test results with model config
    create_full_plot(test_data, test_predictions_df, str(station_id), model_config)  # Pass model config
    
    # Plot convergence
    # plot_convergence(history, str(station_id), title=f"Training and Validation Loss - Station {station_id}")
    
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
            run_hyperparameter_optimization=False,  # Set to True to run hyperparameter tuning
            hyperparameter_trials=30,  # Reasonable number for demonstration
            hyperparameter_diagnostics=False,  # Simplified approach doesn't need diagnostics
        )

        print("\nModel run completed!")
        print(f"Results saved to: {output_path}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history['val_loss']):.6f}")
        if 'smoothed_val_loss' in history:
            print(f"Final smoothed validation loss: {history['smoothed_val_loss'][-1]:.6f}")
            print(f"Best smoothed validation loss: {min(history['smoothed_val_loss']):.6f}")
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
    
