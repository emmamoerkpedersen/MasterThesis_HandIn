"""
Main script to run the error detection pipeline.

TODO:
Implement cross validation strategies?

"""
import pandas as pd
from pathlib import Path
from _1_preprocessing.Processing_data import preprocess_data
from _1_preprocessing.split import split_data, split_data_yearly
from data_utils.data_loading import load_all_station_data
from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data, create_interactive_temperature_plot
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, PHYSICAL_LIMITS, LSTM_CONFIG
from diagnostics.lstm_diagnostics import run_all_diagnostics
from _3_lstm_model.lstm_forecaster import train_forecaster, evaluate_forecaster


def run_pipeline(
    data_path: str, 
    output_path: str, 
    test_mode: bool = True,
    test_years: int = 1,
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = True,
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
    
    print("Loading and preprocessing station data...")
   
   
    
    # Load original data first (if preprocessing diagnostics enabled)
    if preprocess_diagnostics:
        original_data = pd.read_pickle('data_utils/Sample data/original_data.pkl')
    
    # Load Preprocessed data
    preprocessed_data = pd.read_pickle('data_utils/Sample data/preprocessed_data.pkl')
    freezing_periods = pd.read_pickle('data_utils/Sample data/frost_periods.pkl')

    #preprocessed_data, original_data, freezing_periods = preprocess_data()
    
    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics and original_data:
        print("Generating preprocessing diagnostics...")
        plot_preprocessing_comparison(original_data, preprocessed_data, Path(output_path), freezing_periods)
        generate_preprocessing_report(preprocessed_data, Path(output_path), original_data)
        plot_additional_data(preprocessed_data, Path(output_path))
        create_interactive_temperature_plot(preprocessed_data, Path(output_path))
    
    #########################################################
    #    Step 2: Split data into yearly windows             #
    #########################################################
    
    print("\nSplitting data into yearly windows...")
    
    # Split data into yearly windows
    split_datasets = split_data_yearly(preprocessed_data)
    
    # Generate split diagnostics if enabled
    if split_diagnostics:
        print("Generating split diagnostics...")
        plot_split_visualization(split_datasets, Path(output_path))
        generate_split_report(split_datasets, Path(output_path))
    
    # If in test mode, limit to most recent years
    if test_mode and test_years > 0:
        print(f"Test mode enabled. Using only the {test_years} most recent years.")
        
        # Get all years and sort them
        all_years = sorted(list(split_datasets['windows'].keys()))
        
        # Keep only the most recent years
        recent_years = all_years[-test_years:] if len(all_years) > test_years else all_years
        
        # Filter the split datasets
        split_datasets['windows'] = {year: data for year, data in split_datasets['windows'].items() if year in recent_years}
        
        print(f"Using years: {', '.join(recent_years)}")
    
    #########################################################
    #    Step 3: Generate synthetic errors                  #
    #########################################################
    
    print("\nStep 3: Generating synthetic errors...")
    
    # Dictionary to store results for each station/year
    stations_results = {}
    
    # Create synthetic error generator
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    
    # Ensure we have the correct data structure
    if 'windows' not in split_datasets:
        print("Error: Expected 'windows' key in split_datasets")
        print(f"Available keys: {list(split_datasets.keys())}")
        raise KeyError("Missing 'windows' key in split_datasets")
    
    # Process each year and station
    for year, stations in split_datasets['windows'].items():
        print(f"\nProcessing year {year}")
        for station, station_data in stations.items():
            try:
                print(f"Generating synthetic errors for {station} ({year})...")
                station_data = station_data['vst_raw']
                
                if station_data is None or station_data.empty:
                    print(f"No data available for station {station} in {year}")
                    continue
                
                # Generate synthetic errors
                modified_data, ground_truth = error_generator.inject_all_errors(station_data)
                
                # Store results
                station_key = f"{station}_{year}"
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
    # Step 4: LSTM-based Anomaly Detection                  #
    #########################################################
    
    print("\nStep 4: Training LSTM models with Station-Specific Approach...")
    
    # Prepare train and validation data
    print("\nPreparing training and validation data...")
    years = sorted(list(split_datasets['windows'].keys()))

    # Use earlier years for training, last year for validation
    validation_year = years[-1]
    training_years = years[:-1]
    
    print(f"Splitting across years:")
    print(f"Training years: {training_years}")
    print(f"Validation year: {validation_year}")
    
    # Use the LSTM configuration from config.py
    lstm_config = LSTM_CONFIG.copy()
    
    # Add adaptive thresholding configuration
    lstm_config.update({
        'use_z_score': True,               # Use statistical z-scores instead of percentile
        'z_score_threshold': 2.5,          # Flag points > 2.5 standard deviations from mean
        'anomaly_threshold_percentile': 95 # Fallback percentile if not using z-scores
    })
    
    # Create station-specific models
    station_ids = set()
    station_models = {}
    training_results = {}
    
    # Identify all unique stations across all years
    for year in years:
        for station in split_datasets['windows'][year].keys():
            station_ids.add(station)
    
    print(f"Found {len(station_ids)} unique stations to model")
    
    # Train a model for each station
    for station_id in sorted(station_ids):
        print(f"\nTraining model for station {station_id}...")
        
        # Collect training data for this station
        station_train_data = {}
        station_val_data = {}
        
        # Gather training data from all training years for this station
        for year in training_years:
            if station_id in split_datasets['windows'][year]:
                data = split_datasets['windows'][year][station_id]
                if 'vst_raw' in data and data['vst_raw'] is not None:
                    # Create a clean copy of the training data with only the original values
                    station_train_data[f"{station_id}_{year}"] = {
                        'vst_raw': data['vst_raw'].copy()  # Ensure we have a clean copy
                    }
        
        # Gather validation data from validation year for this station - also original data
        if validation_year in split_datasets['windows'] and station_id in split_datasets['windows'][validation_year]:
            data = split_datasets['windows'][validation_year][station_id]
            if 'vst_raw' in data and data['vst_raw'] is not None:
                station_val_data[f"{station_id}_{validation_year}"] = {
                    'vst_raw': data['vst_raw'].copy()  # Use clean data for validation too
                }
        
        # Check if we have enough data
        if len(station_train_data) == 0:
            print(f"  No training data available for station {station_id}, skipping")
            continue
            
        print(f"  Training with {len(station_train_data)} years of CLEAN data")
        print(f"  Validation with {len(station_val_data)} years of CLEAN data")
        
        # Train the model
        try:
            station_model, station_history = train_forecaster(
                train_data=station_train_data,
                validation_data=station_val_data,
                config=lstm_config
            )
            
            # Store model and history
            station_models[station_id] = station_model
            training_results[f"model_{station_id}"] = {
                'history': station_history
            }
            
            print(f"  Training complete for station {station_id}")
            
            if 'val_loss' in station_history and len(station_history['val_loss']) > 0:
                print(f"  Final validation loss: {station_history['val_loss'][-1]:.6f}")
            elif 'train_loss' in station_history and len(station_history['train_loss']) > 0:
                print(f"  Final training loss: {station_history['train_loss'][-1]:.6f}")
        except Exception as e:
            print(f"  Error training model for station {station_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Evaluate models on test data with synthetic errors
    print("\nEvaluating models on test data with synthetic errors...")
    all_results = {}
    
    # Process each year and station to evaluate
    for year, stations in split_datasets['windows'].items():
        for station_name, station_data in stations.items():
            try:
                # Create station key
                station_key = f"{station_name}_{year}"
                
                # Skip if we don't have a model for this station
                if station_name not in station_models:
                    print(f"Skipping {station_key} - no model available")
                    continue
                
                # We need both clean data and modified data for evaluation
                if 'vst_raw' not in station_data or station_data['vst_raw'] is None:
                    print(f"Skipping {station_key} - no original data available")
                    continue
                
                if station_key not in stations_results:
                    print(f"Skipping {station_key} - no synthetic errors data")
                    continue
                
                # Get the model for this station
                model = station_models[station_name]
                
                # Create test data structure
                test_data = {
                    station_key: {
                        'vst_raw': station_data['vst_raw'],           # Original clean data
                        'vst_raw_modified': stations_results[station_key]['modified_data']  # Data with errors
                    }
                }
                
                # Get ground truth
                ground_truth = {
                    station_key: stations_results[station_key]['ground_truth']
                }
                
                # Evaluate the model
                print(f"Evaluating {station_key}...")
                print(f"  Testing how well model trained on CLEAN data detects anomalies in MODIFIED data")
                station_results = evaluate_forecaster(
                    model=model,
                    test_data=test_data,
                    ground_truth=ground_truth,
                    config=lstm_config,
                    split_datasets=split_datasets
                )
                
                # Add results to combined results
                all_results.update(station_results)
                
            except Exception as e:
                print(f"Error evaluating {station_name} for year {year}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Update training results with all evaluation results
    training_results.update(all_results)
    
    # Generate detection diagnostics if enabled
    if detection_diagnostics:
        print("\nGenerating LSTM diagnostics visualizations...")
        diagnostic_results = run_all_diagnostics(
            training_results=training_results,
            combined_train_data={k: v for year in training_years 
                               for station_id, v in split_datasets['windows'][year].items()
                               for k in [f"{station_id}_{year}"]},
            training_anomalies=[],  # No anomalies from confidence detector
            split_datasets=split_datasets,
            stations_results=stations_results,
            output_dir=Path(output_path)
        )
    
    # Final summary
    station_years = [k for k in training_results.keys() 
                    if not k.startswith('model_') and k != 'history']
    print(f"\nAnomaly detection complete. Analyzed {len(station_years)} station-years.")

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "VST_RAW.txt"
    output_path = project_root / "results"
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline with proper hyperparameter tuning
    run_pipeline(
        str(data_path), 
        str(output_path),
        test_mode=True,
        test_years=3,
        preprocess_diagnostics=True,
        split_diagnostics=False,
        synthetic_diagnostics=True,
        detection_diagnostics=True,
    )

