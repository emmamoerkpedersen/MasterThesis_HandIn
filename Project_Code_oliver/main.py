"""
Main script to run the error detection pipeline.

The Anomaly Detection Paradox: Training with Imperfect Data
You've touched on a profound challenge in anomaly detection - the "clean data paradox." Let me explore this conceptually:
The Challenge
Imperfect Training Data: Our VST_RAW data is preprocessed to remove extreme outliers, but likely still contains subtle anomalies
Model Confusion: If we train on data containing real anomalies, the model may learn those patterns as "normal"
Blind Spots: The model could develop specific blind spots to certain anomaly types present in training data
Conceptual Approaches (Without Code)
1. Iterative Self-Improving Process
Initial Detection: Train model on raw data, find highest-confidence anomalies
Data Refinement: Remove these from training set, retrain model
Convergence: Repeat until anomaly detections stabilize
Benefit: Each iteration produces a "cleaner" normal dataset
2. Ensemble Consensus
Diverse Training Windows: Train multiple models on different time windows
Voting System: True anomalies likely trigger multiple models
Confidence Scoring: Weight detection by consensus strength
Benefit: Anomalies in one training set are diluted across the ensemble
3. Robust Statistical Training
Weighted Loss Functions: Use loss functions that de-emphasize outliers
Trimmed Statistics: Train on trimmed means rather than full distributions
RANSAC-like Approaches: Identify inliers through iterative consensus
Benefit: Automatically reduces influence of potential anomalies
4. Physical Constraints Integration
Domain Knowledge Filter: Apply physical impossibility filters before training
Physics-Informed Loss: Penalize physically implausible reconstructions more heavily
Seasonal Normalization: Account for expected seasonal patterns explicitly
Benefit: Grounds detection in water system physics
5. Transfer Learning Approach
Synthetic Normal Base: Create synthetic "perfectly normal" data with physics models
Pre-Training: Train first on purely synthetic normal patterns
Careful Fine-Tuning: Cautiously adapt to real data characteristics
Benefit: Establishes clear "normal" baseline before exposure to anomalies
Practical Next Steps
The most immediately practical approach would be a combination:
Apply Physical Constraints first to remove clearly impossible values
Use Robust Training Methods that inherently down-weight outliers
Implement Iterative Refinement to gradually improve the training set
This multi-layered strategy would help address the paradox of needing clean data to find anomalies, while using anomaly detection to create clean data.
What aspects of this challenge would you like to explore more deeply?

"""

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info, 2=warning, 3=error
logging.getLogger('tensorflow').setLevel(logging.WARNING)
import tensorflow as tf
import pandas as pd
from pathlib import Path
from _1_preprocessing.Processing_data import preprocess_data
from _1_preprocessing.split import split_data, split_data_yearly
from data_utils.data_loading import load_all_station_data
from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data, create_interactive_temperature_plot
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS, PHYSICAL_LIMITS, LSTM_CONFIG
from diagnostics.synthetic_diagnostics import plot_synthetic_errors, generate_synthetic_report, create_interactive_plot, plot_synthetic_vs_actual
from diagnostics.lstm_diagnostics import (
    # Core visualization functions
    plot_training_history,
    plot_reconstruction_results,
    plot_synthetic_anomaly_detection,
    plot_confidence_detector_iterations,
    # Summary metrics and plots
    plot_error_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_temporal_performance,
    
    # Report generation
    generate_lstm_report,
    plot_training_anomalies
)
from _3_lstm_model.statistical_model import StatisticalDetector
from _3_lstm_model.hyperparameter_tuning import run_hyperparameter_tuning, load_best_hyperparameters
from _3_lstm_model.torch_lstm_model import train_autoencoder, learn_optimal_threshold, evaluate_realistic
from _3_lstm_model.confidence_detector import ConfidenceIntervalAnomalyDetector

import numpy as np

def run_pipeline(
    data_path: str, 
    output_path: str, 
    test_mode: bool = False,
    test_years: int = 5,
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = True,
    detection_diagnostics: bool = False,
    imputation_diagnostics: bool = False,
    validation_diagnostics: bool = False,
    plot_final_diagnostics: bool = False,
    hyperparameter_tuning: bool = False,
    hyperparameter_search_type: str = "tpe",
    hyperparameter_trials: int = 20,
    hyperparameter_diagnostics: bool = False,
    optimal_threshold_tuning: bool = False
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
        imputation_diagnostics: Generate diagnostics for imputation
        validation_diagnostics: Generate diagnostics for validation
        plot_final_diagnostics: Generate final summary plots
        hyperparameter_tuning: Whether to run hyperparameter tuning
        hyperparameter_search_type: Type of hyperparameter search ('tpe', 'cmaes', or 'random')
        hyperparameter_trials: Number of trials for hyperparameter search
        hyperparameter_diagnostics: Whether to generate hyperparameter tuning diagnostics
        optimal_threshold_tuning: Whether to learn an optimal threshold for anomaly detection
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
    if preprocess_diagnostics and original_data:
        print("Generating preprocessing diagnostics...")
        plot_preprocessing_comparison(original_data, preprocessed_data, Path(output_path))
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
        print("Generating synthetic error diagnostics...")
        for year, stations in split_datasets['windows'].items():
            for station in stations:
                station_key = f"{station}_{year}"
                if station_key in stations_results:
                    # Plot synthetic errors
                    plot_synthetic_errors(
                        original_data=stations[station]['vst_raw'],
                        modified_data=stations_results[station_key]['modified_data'],
                        error_periods=stations_results[station_key]['error_periods'],
                        station_name=station_key,
                        output_dir=Path(output_path)
                    )
                    
                    # Create interactive plot
                    create_interactive_plot(
                        original_data=stations[station]['vst_raw'],
                        modified_data=stations_results[station_key]['modified_data'],
                        error_periods=stations_results[station_key]['error_periods'],
                        station_name=station_key,
                        output_dir=Path(output_path)
                    )
                    
                    # Plot synthetic vs actual errors
                    plot_synthetic_vs_actual(
                        original_data=stations[station]['vst_raw'],
                        modified_data=stations_results[station_key]['modified_data'],
                        error_periods=stations_results[station_key]['error_periods'],
                        station_name=station_key,
                        output_dir=Path(output_path)
                    )
        
        # Generate the report for all results
        generate_synthetic_report(stations_results, Path(output_path))
    
    #########################################################
    # Step 4: LSTM-based Anomaly Detection                  #
    #########################################################
    
    print("\nStep 4: Training LSTM model with Station-Specific Approach...")
    
    # Prepare train and validation data
    print("\nPreparing training and validation data...")
    combined_train_data = {}
    validation_data = {}
    years = sorted(list(split_datasets['windows'].keys()))

    if len(years) == 1:
        # If we only have one year, split stations within that year
        year = years[0]
        stations = list(split_datasets['windows'][year].keys())
        n_stations = len(stations)
        n_train = max(1, int(n_stations * 0.8))  # At least 1 station for training
        
        # Use 80% of stations for training
        train_stations = stations[:n_train]
        val_stations = stations[n_train:]
        
        print(f"Splitting single year {year}:")
        print(f"Training stations: {train_stations}")
        print(f"Validation stations: {val_stations}")
        
        # Assign stations to train and validation
        for station in train_stations:
            if 'vst_raw' in split_datasets['windows'][year][station]:
                combined_train_data[f"{station}_{year}"] = split_datasets['windows'][year][station]
        
        for station in val_stations:
            if 'vst_raw' in split_datasets['windows'][year][station]:
                validation_data[f"{station}_{year}"] = split_datasets['windows'][year][station]
    else:
        # Multiple years: use earlier years for training, last year for validation
        validation_year = years[-1]
        training_years = years[:-1]
        
        print(f"Splitting across years:")
        print(f"Training years: {training_years}")
        print(f"Validation year: {validation_year}")
        
        # Collect training data from all but the last year
        for year in training_years:
            for station, data in split_datasets['windows'][year].items():
                if 'vst_raw' in data and data['vst_raw'] is not None:
                    combined_train_data[f"{station}_{year}"] = data
        
        # Use the last year for validation
        for station, data in split_datasets['windows'][validation_year].items():
            if 'vst_raw' in data and data['vst_raw'] is not None:
                validation_data[f"{station}_{validation_year}"] = data

    print(f"\nFinal data split:")
    print(f"Training data: {len(combined_train_data)} station-years")
    print(f"Validation data: {len(validation_data)} station-years")

    if len(combined_train_data) == 0:
        raise ValueError("No training data available. Please ensure there is at least one station for training.")

    # Define model configuration
    lstm_config = LSTM_CONFIG.copy()
    lstm_config['feature_cols'] = ['Value']  # Simplified feature set
    lstm_config['sequence_length'] = 96  # 24 hours (with 15-min intervals)
    lstm_config['hidden_dim'] = 32  # Simple hidden dimension
    lstm_config['epochs'] = 10  # Allow more total training time
    lstm_config['batch_size'] = 64
    lstm_config['learning_rate'] = 0.001  # Back to original, more stable value
    lstm_config['dropout_rate'] = 0.2 # 0.2 is the default. Dropout rate is the fraction of neurons that are randomly dropped out during training.
    lstm_config['patience'] = 5  # Give it more time to find good solutions
    
    # Initialize confidence detector
    print("\nSetting up Confidence Interval Anomaly Detector...")
    confidence_detector = ConfidenceIntervalAnomalyDetector(
        config=lstm_config,
        models_dir=Path(output_path) / "models",
        confidence_level=0.99,
        max_iterations=1
    )

    # Train individual models for each station
    print("\nTraining individual station models...")
    station_models = {}
    station_results = {}

    # Group data by station
    station_data = {}
    for station_year_key, data in combined_train_data.items():
        station_id = station_year_key.split('_')[0]
        if station_id not in station_data:
            station_data[station_id] = {}
        station_data[station_id][station_year_key] = data

    # Train a model for each station
    for station_id, data in station_data.items():
        print(f"\nProcessing station {station_id}...")
        
        # Get validation data for this station
        station_val_data = {k: v for k, v in validation_data.items() if k.startswith(station_id + '_')}
        
        try:
            # Train model with iterative refinement
            model, history, training_anomalies, cleaned_data = confidence_detector.train_station_model(
                station_id=station_id,
                train_data=data,
                validation_data=station_val_data if station_val_data else None
            )
            
            # Store results
            station_models[station_id] = model
            station_results[station_id] = {
                'history': history,
                'training_anomalies': training_anomalies,
                'cleaned_data': cleaned_data
            }
            
            print(f"Successfully trained model for station {station_id}")
            
        except Exception as e:
            print(f"Error training model for station {station_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Initialize results dictionary for evaluation
    all_results = {}

    # Evaluate each station's model on its test data
    print("\nEvaluating station models...")
    for year, stations in split_datasets['windows'].items():
        for station_name, station_data in stations.items():
            try:
                # Create station key
                station_key = f"{station_name}_{year}"
                
                # Skip if we don't have the model or necessary data
                if station_name not in station_models:
                    print(f"Skipping {station_key} - no model available")
                    continue
                
                # Get the cleaned data for this station
                cleaned_data = None
                if station_name in station_results:
                    # Extract cleaned data for this specific station-year
                    station_cleaned_data = station_results[station_name]['cleaned_data']
                    if station_key in station_cleaned_data:
                        cleaned_data = station_cleaned_data[station_key].get('vst_raw')
                
                if cleaned_data is None:
                    print(f"Warning: No cleaned data available for {station_key}, falling back to original data")
                    if 'vst_raw' not in station_data or station_data['vst_raw'] is None:
                        print(f"Skipping {station_key} - no vst_raw data available")
                        continue
                    cleaned_data = station_data['vst_raw']
                
                if station_key not in stations_results:
                    print(f"Skipping {station_key} - no synthetic errors data")
                    continue
                
                # Create test data structure using cleaned data
                test_data = {
                    station_key: {
                        'vst_raw': cleaned_data,  # Now using cleaned data instead of original
                        'vst_raw_modified': stations_results[station_key]['modified_data']
                    }
                }
                
                # Get ground truth
                ground_truth = {
                    station_key: stations_results[station_key]['ground_truth']
                }
                
                print(f"\nEvaluating {station_key} using cleaned data as baseline")
                # Evaluate using the station's model
                results = evaluate_realistic(
                    model=station_models[station_name],
                    test_data=test_data,
                    ground_truth=ground_truth,
                    config=lstm_config,
                    split_datasets=split_datasets
                )
                
                # Add results
                all_results.update(results)
                
            except Exception as e:
                print(f"Error evaluating {station_name} for year {year}: {e}")
                import traceback
                traceback.print_exc()

    # Combine all results for diagnostics
    training_results = {
        'station_results': station_results,
        'evaluation_results': all_results
    }
    
    # 5. Generate detection diagnostics
    if detection_diagnostics:
        print("\nGenerating detection diagnostics...")
        
        # Set output directory for all plots
        diagnostics_dir = Path(output_path) / "diagnostics" / "lstm"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        # First, plot training history
        if 'history' in training_results:
            try:
                print("Generating training history visualization...")
                plot_training_history(
                    history=training_results['history'],
                    output_dir=Path(output_path),
                    model_name="lstm_autoencoder",
                    figsize=(12, 6),
                    dpi=300
                )
            except Exception as e:
                print(f"Error generating training history plot: {e}")
                import traceback
                traceback.print_exc()
        
        # Plot results for each station/year
        for station_key, results in training_results.items():
            # Skip metadata keys
            if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
                continue
            
            try:
                # Get required data for visualization
                if 'original_data' not in results:
                    print(f"Warning: Missing original_data for {station_key}, trying to retrieve it")
                    station_id, year = station_key.split('_')
                    if station_id in split_datasets['windows'][year]:
                        results['original_data'] = split_datasets['windows'][year][station_id]['vst_raw'].copy()
                
                if 'modified_data' not in results:
                    print(f"Warning: Missing modified_data for {station_key}, trying to retrieve it")
                    station_id, year = station_key.split('_')
                    if station_key in stations_results:
                        results['modified_data'] = stations_results[station_key]['modified_data'].copy()
                
                # Use our enhanced plotting function
                print(f"Generating visualization for {station_key}...")
                plot_reconstruction_results(
                    original_data=results.get('original_data'),
                    modified_data=results.get('modified_data'),
                    reconstruction_errors=results.get('reconstruction_errors'),
                    anomaly_flags=results.get('anomaly_flags'),
                    timestamps=results.get('timestamps'),
                    threshold=results.get('threshold'),
                    station_name=station_key,
                    output_dir=Path(output_path),
                    reconstructed_values=results.get('reconstructions'),
                    # Add optional parameters for enhanced visualization
                    prediction_intervals=results.get('prediction_intervals'),
                    z_scores=results.get('z_scores'),
                    figsize=(14, 10),
                    dpi=300
                )
            except Exception as e:
                print(f"Error generating visualization for {station_key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary diagnostics
        print("\nGenerating summary diagnostics...")
        
        # Plot error distribution with better error handling
        try:
            print("Generating error distribution plot...")
            plot_error_distribution(
                results=training_results,
                output_dir=Path(output_path),
                figsize=(12, 8),
                dpi=300
            )
        except Exception as e:
            print(f"Error generating error distribution plot: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate performance metrics if ground truth is available
        has_ground_truth = False
        for station_key, result in training_results.items():
            if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
                continue
            if isinstance(result, dict) and 'ground_truth' in result and result['ground_truth'] is not None:
                has_ground_truth = True
                break
        
        if has_ground_truth:
            print("Ground truth data available, generating performance evaluation metrics...")
            
            # Create performance metrics
            try:
                print("Generating confusion matrix...")
                plot_confusion_matrix(
                    results=training_results,
                    output_dir=Path(output_path),
                    figsize=(10, 8),
                    dpi=300
                )
                
                print("Generating ROC curve...")
                plot_roc_curve(
                    results=training_results,
                    output_dir=Path(output_path),
                    figsize=(10, 8),
                    dpi=300
                )
                
                print("Generating precision-recall curve...")
                plot_precision_recall_curve(
                    results=training_results,
                    output_dir=Path(output_path),
                    figsize=(10, 8),
                    dpi=300
                )
                
                print("Analyzing temporal performance patterns...")
                plot_temporal_performance(
                    results=training_results,
                    output_dir=Path(output_path),
                    figsize=(12, 10),
                    dpi=300
                )
                plot_synthetic_anomaly_detection(
                    results=training_results,
                    output_dir=Path(output_path),
                    figsize=(12, 10),
                    dpi=300
                )

            except Exception as e:
                print(f"Error generating performance evaluation plots: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No ground truth data available, skipping performance evaluation metrics.")
        
        # Generate comprehensive HTML report
        try:
            print("\nGenerating comprehensive LSTM report...")
            report_result = generate_lstm_report(
                detection_results=training_results,
                training_results=training_results,
                output_dir=Path(output_path)
            )
            print(f"LSTM report generated: {report_result['report_path']}")
        except Exception as e:
            print(f"Error generating LSTM report: {e}")
            import traceback
            traceback.print_exc()
        
        # Visualize training anomalies if available
        if 'training_anomalies' in training_results:
            try:
                print("Generating training anomalies visualization...")
                plot_training_anomalies(
                    anomalies_per_iteration=training_results['training_anomalies'],
                    output_dir=Path(output_path),
                    figsize=(12, 8),
                    dpi=300
                )
            except Exception as e:
                print(f"Error generating training anomalies plot: {e}")
                import traceback
                traceback.print_exc()
        
        # After confidence detector runs
        if training_anomalies:
            plot_confidence_detector_iterations(
                original_data=combined_train_data,
                anomalies_per_iteration=training_anomalies,
                output_dir=Path(output_path)
            )
        
        print("\nDiagnostics generation complete.")
    
    # Final summary
    station_years = [k for k in training_results.keys() 
                    if k not in ['initial_training', 'fine_tuning', 'history']]
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
        test_years=5,
        preprocess_diagnostics=False,
        split_diagnostics=False,
        synthetic_diagnostics=True,
        hyperparameter_diagnostics=True,
        detection_diagnostics=True,
        imputation_diagnostics=False,
        validation_diagnostics=False,
        plot_final_diagnostics=False,
        hyperparameter_tuning=False,
        hyperparameter_search_type="tpe",
        hyperparameter_trials=1,
        optimal_threshold_tuning=False
    )

"""
Training Phase:
Clean Data ──> Learn Normal Patterns
                     │
Synthetic Anomalies ─┴─> Fine-tune to Recognize Anomalies
                     │
                     v
            Trained Model

Deployment Phase:
New Raw Data ──> Trained Model ──> Detected Anomalies
(potentially                (no synthetic
 anomalous)                 anomalies needed)

 So, i need to understand the curent training process for the LSTM. We use the raw data AND the modified data with synthetic anomalies. What i envision, is that the model learns to detect anomalies and gets progressively better? How exacttly does the current implementation work? The thing th
 at worries me is that the raw data is only cleaned to remove extreme values, actual anomalies similar to the ones that we sythnetically inject might still be there. How can we actually
"""

