"""
Main script to run the error detection pipeline.
"""

import pandas as pd
from pathlib import Path
#from error_detection import synthetic_errors, imputation, validation
#from error_detection.anomaly_detection import AnomalyDetector
#from plot_code import error_plots
#from error_detection.utils import export_results
#from data_loading import prepare_data_for_error_detection
#from error_detection.config import SYNTHETIC_ERROR_PARAMS, PHYSICAL_LIMITS
from _1_preprocessing.Processing_data import preprocess_data
from _1_preprocessing.split import split_data, split_data_yearly, split_data_rolling
from data_utils.data_loading import load_all_station_data
from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data, create_interactive_temperature_plot
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG
from diagnostics.synthetic_diagnostics import plot_synthetic_errors, generate_synthetic_report, create_interactive_plot, plot_synthetic_vs_actual
from diagnostics.lstm_diagnostics import plot_training_history, plot_detection_results, generate_lstm_report

# Import the LSTM model and trainer
from _3_lstm_model.lstm_forecaster import LSTMModel, train_LSTM, create_full_plot

# Update these imports to match new structure
#from _3_lstm_model.lstm_model import LSTMModel
#from _3_lstm_model.data_preparation import prepare_data
#from _3_lstm_model._3_1_anomaly_detection.detector import AnomalyDetector
#from _3_lstm_model._3_2_imputation.imputer import impute_values, get_uncertainty_periods

#import torch
import numpy as np
import sys

def run_lstm_training_pipeline(lstm_config):
    """
    Run the LSTM training and prediction pipeline.
    """
    #1 Load and preprocess data
    station_id = ['21006846']
    print(f"Loading and preprocessing station data for station {station_id}...")
    preprocessed_data = pd.read_pickle('../data_utils/Sample data/preprocessed_data.pkl')
    
    # Generate dictionary with same structure but with the specified station_id
    preprocessed_data = {station_id[0]: preprocessed_data[station_id[0]] for station_id in [station_id]}
    
    #2 Split data
    print("\nSplitting data into rolling windows")
    split_datasets = split_data_rolling(preprocessed_data)
    
    #3 Generate synthetic errors for test data
    print("\nStep 3: Generating synthetic errors for test data only...")
    stations_results = {}
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
    
    #4 Train LSTM models
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
    
    # Create and show the plot
    create_full_plot(
        test_data=split_datasets['test'][station_id[0]], 
        test_predictions=test_predictions, 
        station_id=station_id,
        sequence_length=lstm_config.get('sequence_length')
    )
    
    return model, trainer, test_predictions

def run_pipeline(
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = True,
    detection_diagnostics: bool = False,
    imputation_diagnostics: bool = False,
    validation_diagnostics: bool = False,
    plot_final_diagnostics: bool = False,
    split_mode: str = "normal",
    run_lstm: bool = True  # New parameter to control LSTM execution
    ):
    """
    Run the complete error detection and imputation pipeline.
    
    Args:
        data_path: Path to input data
        output_path: Path for results
        preprocess_diagnostics: Generate diagnostics for preprocessing step
        split_diagnostics: Generate diagnostics for data splitting
        synthetic_diagnostics: Generate diagnostics for synthetic error generation
        detection_diagnostics: Generate diagnostics for error detection
        imputation_diagnostics: Generate diagnostics for imputation
    """
    
    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    
    print("Loading and preprocessing station data...")
    
    # Load original data first (if preprocessing diagnostics enabled)
    if preprocess_diagnostics:
        original_data = load_all_station_data()
    
    # Preprocess data
    preprocessed_data = preprocess_data()
    
    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics:
        print("\nGenerating preprocessing diagnostics...")
        plot_preprocessing_comparison(
            original_data, 
            preprocessed_data, 
            Path(output_path)
        )
        plot_additional_data(
            preprocessed_data,
            Path(output_path)
        )
        create_interactive_temperature_plot(
            preprocessed_data,
            Path(output_path)
        )
        generate_preprocessing_report(
            original_data, 
            preprocessed_data, 
            Path(output_path)
        )
    
        # Print summary of processed stations
        print("\nPreprocessing complete. Summary:")
        for station_name, station_data in preprocessed_data.items():
            if station_data['vst_raw'] is not None:
                print(f"\nStation: {station_name}")
                for data_type, data in station_data.items():
                    if data is not None:
                        print(f"  - {data_type}: {len(data)} points")
                        if isinstance(data, pd.DataFrame):
                            print(f"    Time range: {data.index.min()} to {data.index.max()}")
                            
    #########################################################
    #  Step 2: Split data                                     #
    #########################################################
    
    if split_mode == "normal":
        print("\nSplitting data into train/validation/test sets (normal mode)...")
        split_datasets = split_data(preprocessed_data)
    
        # Generate split diagnostics if enabled
        if split_diagnostics:
            print("Generating split diagnostics...")
            plot_split_visualization(
                split_datasets,
                Path(output_path)
            )
            generate_split_report(
                split_datasets,
                Path(output_path)
            )
    elif split_mode == "yearly":
        print("\nSplitting data using yearly windows...")
        split_datasets = split_data_yearly(preprocessed_data)
    
        if split_diagnostics:
            print("Generating yearly split diagnostics...")
            plot_split_visualization(split_datasets, Path(output_path))
            generate_split_report(split_datasets, Path(output_path))
    else:
        raise ValueError("Invalid split_mode. Choose 'normal' or 'yearly'.")

    
    #########################################################
    # Step 3: Generate synthetic errors for testing data    #
    #########################################################
    
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    error_types = ['spike', 'drift', 'offset', 'baseline_shift']
    
    if split_mode == "normal":
        print("\nInjecting synthetic errors into test data (normal mode)...")
        stations_results = {}
        for station_name, station_data in split_datasets['test'].items():
            if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
                print(f"\nProcessing station: {station_name}")
                
                # Get test data and inject errors
                test_data = station_data['vst_raw'].copy()
                modified_data, ground_truth = error_generator.inject_all_errors(test_data, error_types=error_types)
                
                # Store modified data and results
                split_datasets['test'][station_name]['vst_raw_modified'] = modified_data
                stations_results[station_name] = {
                    'error_periods': error_generator.error_periods.copy(),  # Store a copy to avoid reference issues
                    'ground_truth': ground_truth
                }
                
                # Clear error periods for next station
                error_generator.error_periods = []
                error_generator.used_indices = set()
        
        # Generate synthetic error diagnostics if enabled
        if synthetic_diagnostics:
            print("\nGenerating synthetic error diagnostics (normal mode)...")
            for station_name, station_data in split_datasets['test'].items():
                if 'vst_raw_modified' in station_data:
                    plot_synthetic_errors(
                        original_data=station_data['vst_raw'],
                        modified_data=station_data['vst_raw_modified'],
                        error_periods=stations_results[station_name]['error_periods'],
                        station_name=station_name,
                        output_dir=Path(output_path)
                    )

                    create_interactive_plot(
                        original_data=station_data['vst_raw'],
                        modified_data=station_data['vst_raw_modified'],
                        error_periods=stations_results[station_name]['error_periods'],
                        station_name=station_name,
                        output_dir=Path(output_path)
                    )

                    plot_synthetic_vs_actual(
                        original_data=station_data['vst_raw'],
                        modified_data=station_data['vst_raw_modified'],
                        error_periods=stations_results[station_name]['error_periods'],
                        station_name=station_name,
                        output_dir=Path(output_path)
                    )
                    
            generate_synthetic_report(stations_results, Path(output_path))
    
    elif split_mode == "yearly":
        print("\nInjecting synthetic errors into yearly windows...")
        stations_results = {}
        
        for year, stations in split_datasets['windows'].items():
            for station_name, station_data in stations.items():
                if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
                    print(f"\nProcessing station: {station_name} for year: {year}")
                    
                    # Create a fresh error generator for each window
                    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
                    
                    # Get window data and inject errors
                    window_data = station_data['vst_raw'].copy()
                    modified_data, ground_truth = error_generator.inject_all_errors(window_data, error_types=error_types)
                    
                    # Store results
                    station_key = f"{station_name}_{year}"
                    stations_results[station_key] = {
                        'error_periods': error_generator.error_periods.copy(),
                        'ground_truth': ground_truth
                    }
                    station_data['vst_raw_modified'] = modified_data
        
        # Generate synthetic diagnostics for yearly split if enabled
        if synthetic_diagnostics:
            print("\nGenerating synthetic error diagnostics (yearly split)...")
            
            # Select a few representative examples to plot
            max_plots_per_station = 5  # Number of years to plot per station
            plotted_stations = set()
            
            for year, stations in split_datasets['windows'].items():
                for station, station_window in stations.items():
                    # Skip if we've already plotted enough examples for this station
                    if station in plotted_stations:
                        continue
                        
                    if 'vst_raw_modified' in station_window:
                        # Create plots for this station-year combination
                        plot_synthetic_errors(
                            original_data=station_window['vst_raw'],
                            modified_data=station_window['vst_raw_modified'],
                            error_periods=stations_results[f"{station}_{year}"]['error_periods'],
                            station_name=f"{station}_{year}",
                            output_dir=Path(output_path)
                        )
                        create_interactive_plot(
                            original_data=station_window['vst_raw'],
                            modified_data=station_window['vst_raw_modified'],
                            error_periods=stations_results[f"{station}_{year}"]['error_periods'],
                            station_name=f"{station}_{year}",
                            output_dir=Path(output_path)
                        )
                        plot_synthetic_vs_actual(
                            original_data=station_window['vst_raw'],
                            modified_data=station_window['vst_raw_modified'],
                            error_periods=stations_results[f"{station}_{year}"]['error_periods'],
                            station_name=f"{station}_{year}",
                            output_dir=Path(output_path)
                        )
                        
                        # Add station to plotted set and check if we should move to next station
                        plotted_stations.add(station)
                        if len(plotted_stations) >= max_plots_per_station:
                            break
                            
                if len(plotted_stations) >= max_plots_per_station:
                    break
            
            # Still generate the report for all results
            generate_synthetic_report(stations_results, Path(output_path))
    else:
        raise ValueError("Invalid split_mode. Choose 'normal' or 'yearly'.")

    #########################################################
    # Step 4: LSTM-based Anomaly Detection                   #
    #########################################################

    print("\nStep 4: Running LSTM-based anomaly detection...")

    for year, stations in split_datasets['windows'].items():
        print(f"\nProcessing year: {year}")
        
        # Step 4.1: Hyperparameter Tuning
        tune_data = get_tuning_subset(stations, ratio=0.2)  # Get first 20% of year
        study = tune_hyperparameters(
            train_data=tune_data,
            validation_data=tune_data,  # Use same data for validation in unsupervised setting
            n_trials=100
        )
        best_params = get_best_parameters(study)
        
        # Step 4.2: Initialize model with tuned parameters
        lstm_model = LSTMAutoencoder(**best_params['model'])
        
        # Step 4.3: Process full year's data
        # ... rest of processing

    #########################################################
    # Step 5: Imputation                                     #
    #########################################################
    
    print("\nStep 5: Performing LSTM-based imputation...")
    
    # Dictionary to store imputation results for diagnostics
    imputation_results = {}
    
    for station_name, station_data in split_datasets['test'].items():
        if station_name in detection_results:
            # Step 5.1: Impute high-confidence anomalies
            imputed_values, uncertain_regions = impute_values(
                model_output=detection_results[station_name]['predictions'],
                anomaly_mask=detection_results[station_name]['anomaly_flags'],
                confidence_scores=detection_results[station_name]['confidence_scores'],
                confidence_threshold=LSTM_CONFIG['imputation']['confidence_threshold_range'][0]
            )
            
            # Step 5.2: Get periods requiring manual review
            uncertain_periods = get_uncertainty_periods(
                anomaly_mask=detection_results[station_name]['anomaly_flags'],
                confidence_scores=detection_results[station_name]['confidence_scores'],
                confidence_threshold=LSTM_CONFIG['imputation']['confidence_threshold_range'][0]
            )
            
            # Store results for diagnostics
            imputation_results[station_name] = {
                'original_data': station_data['vst_raw_modified'],
                'imputed_values': imputed_values,
                'uncertain_regions': uncertain_regions,
                'uncertain_periods': uncertain_periods
            }
    
    # Generate imputation diagnostics if enabled
    if imputation_diagnostics:
        print("\nGenerating imputation diagnostics...")
        for station_name, results in imputation_results.items():
            # TODO: Add imputation-specific diagnostic functions
            pass

    #########################################################
    # Step 6: Evaluation on New Data                         #
    #########################################################

    print("\nStep 6: Evaluating on a New Dataset...")

    # Load new, unmodified station data (assuming a function like load_new_station_data() exists)
    new_station_data = load_new_station_data()  # This should be in preprocessed form

    # Prepare the new data for the LSTM
    new_data_loaders = prepare_data(new_station_data, LSTM_CONFIG['data_preparation'])

    # Assuming lstm_model has already been trained/fine-tuned:
    with torch.no_grad():
        new_lstm_features = lstm_model(new_data_loaders['test'])

    # Detect anomalies on the new data
    new_anomaly_flags, new_confidence_scores, new_anomaly_types = detector.detect_anomalies(
        data=new_station_data,
        model_output=new_lstm_features,
        threshold=LSTM_CONFIG['detection']['threshold_range'][0]
    )

    # Evaluate imputation decision on the new data
    new_imputed_values, new_uncertain_regions = impute_values(
        model_output=new_lstm_features,
        anomaly_mask=new_anomaly_flags,
        confidence_scores=new_confidence_scores,
        confidence_threshold=LSTM_CONFIG['imputation']['confidence_threshold_range'][0]
    )

    # Optionally, generate diagnostics or report results for the new dataset
    print("Evaluation on new data complete!")

    #########################################################
    # Step 7: Validation                                     #
    #########################################################
    
    # Compare detected anomalies with ground truth from synthetic error injection
    metrics = validation.calculate_metrics(
        ground_truth=stations_results[station_name]['ground_truth'],
        detected_anomalies=detection_results[station_name]['anomaly_flags']
    )
    
    # Store validation results for diagnostics
    validation_results = {
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'confusion_matrix': metrics['confusion_matrix']
    }

    if validation_diagnostics:
        print("\nGenerating validation diagnostics...")
        for station_name, results in validation_results.items():
            # TODO: Add validation-specific diagnostic functions
            pass

    #########################################################
    # Step 7: Generate Final Plots                          #
    #########################################################
    
    # Plot comparison of detected vs actual anomalies
    error_plots.plot_detected_vs_actual(
        original_data=station_data['vst_raw'],
        modified_data=station_data['vst_raw_modified'],
        ground_truth=stations_results[station_name]['ground_truth'],
        detected_anomalies=detection_results[station_name]['anomaly_flags'],
        station_name=station_name,
        output_dir=Path(output_path)
    )
    
    # Plot imputation results
    error_plots.plot_imputation_results(
        original_data=station_data['vst_raw'],
        modified_data=station_data['vst_raw_modified'],
        imputed_data=imputation_results[station_name]['imputed_values'],
        uncertain_regions=imputation_results[station_name]['uncertain_regions'],
        station_name=station_name,
        output_dir=Path(output_path)
    )
    if plot_final_diagnostics:
        print("\nGenerating final diagnostics...")
        for station_name, results in validation_results.items():
            # TODO: Add final-specific diagnostic functions
            pass
    
    #########################################################
    # Step 8: Export Results                                #
    #########################################################
    
    export_results({
        'station_name': station_name,
        'metrics': validation_results,
        'config': {
            'lstm': LSTM_CONFIG,
            'synthetic_errors': SYNTHETIC_ERROR_PARAMS,
            'physical_limits': PHYSICAL_LIMITS
        },
        'detection_results': {
            'num_anomalies': detection_results[station_name]['num_anomalies'],
            'avg_confidence': detection_results[station_name]['avg_confidence'],
            'confidence_distribution': {
                'high': detection_results[station_name]['high_confidence'],
                'medium': detection_results[station_name]['medium_confidence'],
                'low': detection_results[station_name]['low_confidence']
            }
        },
        'imputation_results': {
            'num_imputed': len(imputation_results[station_name]['imputed_values']),
            'num_uncertain': len(imputation_results[station_name]['uncertain_periods'])
        }
    }, output_path)

    # Add LSTM execution
    if run_lstm:
        print("\nRunning LSTM training and prediction pipeline...")
        lstm_config_copy = LSTM_CONFIG.copy()
        model, trainer, predictions = run_lstm_training_pipeline(lstm_config_copy)
        
        # You can add additional code here to use the trained model
        print("LSTM pipeline completed successfully!")

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "VST_RAW.txt"
    output_path = project_root / "results"
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run just the LSTM pipeline
    run_pipeline(
        str(data_path), 
        str(output_path),
        run_lstm=True,
        preprocess_diagnostics=False,
        split_diagnostics=False,
        synthetic_diagnostics=False,  # Set to False to focus only on LSTM
        split_mode="yearly"
    )
    
    # Alternatively, you can directly call the LSTM function
    # lstm_config_copy = LSTM_CONFIG.copy()
    # model, trainer, predictions = run_lstm_training_pipeline(lstm_config_copy)
