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
from _1_preprocessing.split import split_data, split_data_yearly
from data_utils.data_loading import load_all_station_data
from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data, create_interactive_temperature_plot
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, LSTM_CONFIG
from diagnostics.synthetic_diagnostics import plot_synthetic_errors, generate_synthetic_report, create_interactive_plot, plot_synthetic_vs_actual
from diagnostics.lstm_diagnostics import plot_training_history, plot_detection_results, generate_lstm_report

# Update these imports to match new structure
#from _3_lstm_model.lstm_model import LSTMModel
#from _3_lstm_model.data_preparation import prepare_data
#from _3_lstm_model._3_1_anomaly_detection.detector import AnomalyDetector
#from _3_lstm_model._3_2_imputation.imputer import impute_values, get_uncertainty_periods

#import torch
import numpy as np

def run_pipeline(
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = True,  # Default to True since we want to verify error injection
    detection_diagnostics: bool = False,  # For future use
    imputation_diagnostics: bool = False,  # For future use
    validation_diagnostics: bool = False,  # For future use
    plot_final_diagnostics: bool = False,  # For future use
    split_mode: str = "normal"  # Options: "normal" or "yearly"
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
    # Step 4: Anomaly Detection                              #
    #########################################################
    '''
    print("\nStep 4: Running LSTM-based anomaly detection...")
    
    # Step 4.1: Initialize LSTM model and detector
    lstm_model = LSTMModel(
        input_size=LSTM_CONFIG['model']['input_size'],
        hidden_size=LSTM_CONFIG['model']['hidden_size_range'][0],  # Start with minimum value
        num_layers=LSTM_CONFIG['model']['num_layers_range'][0]
    )
    detector = AnomalyDetector()
    
    # Dictionary to store results for diagnostics
    detection_results = {}
    
    for station_name, station_data in split_datasets['test'].items():
        if 'vst_raw_modified' in station_data:
            print(f"\nProcessing station: {station_name}")
            
            # Step 4.2: Prepare data for LSTM
            data_loaders = prepare_data(station_data['vst_raw_modified'], LSTM_CONFIG['data_preparation'])
            
            # Step 4.3: Train LSTM model
            training_history = train_model(
                model=lstm_model,
                train_data=data_loaders['train'],
                validation_data=data_loaders['validation'],
                config=LSTM_CONFIG['training']
            )
            
            # Step 4.4: Get LSTM features and detect anomalies
            with torch.no_grad():
                lstm_features = lstm_model(data_loaders['test'])
            
            anomaly_flags, confidence_scores, anomaly_types = detector.detect_anomalies(
                data=station_data['vst_raw_modified'],
                model_output=lstm_features,
                threshold=LSTM_CONFIG['detection']['threshold_range'][0]
            )
            
            # Store results for diagnostics
            detection_results[station_name] = {
                'training_history': training_history,
                'predictions': lstm_features.numpy(),
                'anomaly_flags': anomaly_flags,
                'confidence_scores': confidence_scores,
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'epochs': len(training_history['train_loss']),
                'num_anomalies': np.sum(anomaly_flags),
                'avg_confidence': np.mean(confidence_scores),
                'high_confidence': np.sum(confidence_scores > 0.9),
                'medium_confidence': np.sum((confidence_scores >= 0.7) & (confidence_scores <= 0.9)),
                'low_confidence': np.sum(confidence_scores < 0.7)
            }
    
    # Generate detection diagnostics if enabled
    if detection_diagnostics:
        print("\nGenerating anomaly detection diagnostics...")
        for station_name, station_data in split_datasets['test'].items():
            if station_name in detection_results:
                # Plot training history
                plot_training_history(
                    history=detection_results[station_name]['training_history'],
                    station_name=station_name,
                    output_dir=Path(output_path)
                )
                
                # Plot detection results
                plot_detection_results(
                    original_data=station_data['vst_raw_modified'],
                    lstm_predictions=detection_results[station_name]['predictions'],
                    anomaly_flags=detection_results[station_name]['anomaly_flags'],
                    confidence_scores=detection_results[station_name]['confidence_scores'],
                    station_name=station_name,
                    output_dir=Path(output_path)
                )
        
        # Generate overall detection report
        generate_lstm_report(detection_results, Path(output_path))

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
    '''

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "VST_RAW.txt"
    output_path = project_root / "results"
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline with preprocessing, split and synthetic diagnostics enabled
    run_pipeline(
        str(data_path), 
        str(output_path), 
        preprocess_diagnostics=False,
        split_diagnostics=False,
        synthetic_diagnostics=True,
        detection_diagnostics=False,
        imputation_diagnostics=False,
        validation_diagnostics=False,
        plot_final_diagnostics=False,
        split_mode="yearly"
    )
