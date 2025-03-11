"""
Hyperparameter tuning for LSTM autoencoder model using Optuna.

CONSIDERATIONS:
1. Parameter Space:
   - Expanded parameter ranges with more granular steps
   - Added intermediate values to better explore the space
   - Sequence length: Added 72 timesteps option
   - LSTM units: Added [96, 48] architecture option
   - Learning rate: More fine-grained steps between 0.001 and 0.0001
   - Batch size: Added intermediate sizes 48 and 96

2. Optimization Strategy:
   - Using TPE (Tree-structured Parzen Estimators) sampler
   - Added pruning to stop unpromising trials early
   - Added parameter constraints to avoid inefficient combinations
   - Improved early stopping with slightly more patience

3. Quality Focus:
   - Prioritizing finding optimal configuration over speed
   - Comprehensive parameter space exploration
   - Careful validation to avoid overfitting
   - Detailed tracking of trial performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import optuna
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from _3_lstm_model.lstm_model import ConvLSTMAutoencoder, train_autoencoder, evaluate_with_synthetic, train_two_phase_autoencoder
from diagnostics.hyperparameter_diagnostics import (
    save_hyperparameter_results,
    plot_best_model_results,
    generate_hyperparameter_report
)
from diagnostics.lstm_diagnostics import plot_reconstruction_results

# Create a custom pruning callback that works with both older and newer versions of Optuna
class CustomPruningCallback(tf.keras.callbacks.Callback):
    """Custom pruning callback that reports intermediate values to Optuna."""
    
    def __init__(self, trial, monitor="val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is not None:
            self.trial.report(current_score, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

def run_hyperparameter_search(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    ground_truth: Dict,
    param_grid: Dict[str, List],
    output_dir: Path,
    n_trials: int = 100,
    search_type: str = 'tpe',
    base_config: Dict = None,
    evaluation_metric: str = 'reconstruction_error',
    diagnostics: bool = False
) -> Tuple[List[Dict], Dict]:
    """Run hyperparameter search using Optuna."""
    
    # Create output directory
    results_dir = output_dir / "diagnostics" / "hyperparameter_tuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results tracking
    all_results = []
    
    # Define the objective function for Optuna
    def objective(trial):
        trial_start_time = time.time()
        
        try:
            # Create config by sampling from parameter grid
            config = base_config.copy() if base_config else {}
            
            # Important: Disable iterative training during hyperparameter search
            if 'iterative_training' in config:
                config['iterative_training']['enabled'] = False
                config['iterative_training']['max_iterations'] = 1
            else:
                config['iterative_training'] = {
                    'enabled': False,
                    'max_iterations': 1
                }
            
            print("\nStarting new trial with following configuration:")
            
            # Sample parameters based on the param_grid
            for param, values in param_grid.items():
                if isinstance(values[0], int):
                    if param == 'sequence_length':
                        config[param] = trial.suggest_int(param, min(values), max(values))
                    elif param == 'fine_tuning_epochs':
                        config[param] = trial.suggest_int(param, min(values), max(values))
                    elif param == 'fine_tuning_batch_size':
                        config[param] = trial.suggest_int(param, min(values), max(values))
                    elif param == 'epochs':
                        config[param] = trial.suggest_int(param, min(values), max(values))
                    elif param == 'batch_size':
                        config[param] = trial.suggest_int(param, min(values), max(values))
                    else:
                        # For lists of integers, use categorical
                        param_index = f"{param}_index"
                        index = trial.suggest_int(param_index, 0, len(values) - 1)
                        config[param] = values[index]
                elif isinstance(values[0], float):
                    if param == 'learning_rate' or param == 'fine_tuning_lr':
                        config[param] = trial.suggest_float(param, min(values), max(values), log=True)
                    else:
                        config[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    # For lists or other structures, use categorical
                    param_index = f"{param}_index"
                    index = trial.suggest_int(param_index, 0, len(values) - 1)
                    config[param] = values[index]
            
            # Handle special parameters with nested structure
            if 'iterative_training.enabled' in param_grid:
                if 'iterative_training' not in config:
                    config['iterative_training'] = {}
                config['iterative_training']['enabled'] = trial.suggest_categorical(
                    'iterative_training.enabled', 
                    param_grid['iterative_training.enabled']
                )
            
            if 'iterative_training.max_iterations' in param_grid:
                if 'iterative_training' not in config:
                    config['iterative_training'] = {}
                config['iterative_training']['max_iterations'] = trial.suggest_int(
                    'iterative_training.max_iterations',
                    min(param_grid['iterative_training.max_iterations']),
                    max(param_grid['iterative_training.max_iterations'])
                )
            
            # Handle feature selection
            if 'feature_selection' in param_grid:
                features_idx = trial.suggest_int('feature_selection_idx', 0, len(param_grid['feature_selection'])-1)
                config['feature_cols'] = param_grid['feature_selection'][features_idx]
            
            # Print the configuration
            for key, value in config.items():
                print(f"{key}: {value}")
            
            print("\nTraining model with this configuration...")
            
            # Train model with this configuration
            model, training_results = train_two_phase_autoencoder(
                train_data=train_data,
                validation_data=val_data,
                config=config,
                verbose=1
            )
            
            # Evaluate model
            if evaluation_metric == 'reconstruction_error':
                # Get the final validation loss (reconstruction error)
                if isinstance(training_results['fine_tuning'], dict) and 'val_loss' in training_results['fine_tuning']:
                    # If we have validation loss from fine-tuning phase
                    metric_value = training_results['fine_tuning']['val_loss'][-1]
                elif isinstance(training_results['initial_training'], dict) and isinstance(training_results['initial_training'][-1], dict) and 'val_loss' in training_results['initial_training'][-1]:
                    # Get the validation loss from the last iteration of initial training
                    metric_value = training_results['initial_training'][-1]['val_loss'][-1]
                else:
                    # If no validation loss is available, use training loss
                    if isinstance(training_results['fine_tuning'], dict) and 'loss' in training_results['fine_tuning']:
                        metric_value = training_results['fine_tuning']['loss'][-1]
                    elif isinstance(training_results['initial_training'], dict) and isinstance(training_results['initial_training'][-1], dict) and 'loss' in training_results['initial_training'][-1]:
                        metric_value = training_results['initial_training'][-1]['loss'][-1]
                    else:
                        # If we can't find a valid loss value, use a default high value
                        print("No valid loss values found")
                        metric_value = 1.0  # Default high value instead of inf
            else:
                # For other metrics, evaluate on test data
                test_results = evaluate_with_synthetic(model, test_data, ground_truth, config)
                metric_value = calculate_overall_performance(test_results, evaluation_metric)
            
            # Handle inf or nan values
            if np.isnan(metric_value) or np.isinf(metric_value):
                print(f"Warning: Invalid metric value ({metric_value}). Using default value instead.")
                metric_value = 1.0  # Default high value instead of inf
            
            print(f"Trial completed with {evaluation_metric}: {metric_value}")
            
            # Save results for this trial
            trial_results = {
                'config': config,
                'metric': evaluation_metric,
                'value': float(metric_value),
                'training_time': time.time() - trial_start_time
            }
            all_results.append(trial_results)
            
            return metric_value
            
        except Exception as e:
            import traceback
            print(f"Trial failed with error: {e}")
            print(traceback.format_exc())
            # Return a default high value for failed trials
            return 1.0  # Default high value instead of inf
    
    # Create study with appropriate sampler
    if search_type == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler()
    elif search_type == 'random':
        sampler = optuna.samplers.RandomSampler()
    else:  # default to TPE
        sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,
            max_resource=30,
            reduction_factor=3
        ),
        direction='minimize' if evaluation_metric == 'reconstruction_error' else 'maximize'
    )
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
    
    # Get best parameters
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Convert best_params to config format
    best_config = base_config.copy() if base_config else {}
    for param, value in best_params.items():
        if param.endswith('_index'):
            original_param = param[:-6]
            best_config[original_param] = param_grid[original_param][value]
        else:
            best_config[param] = value
    
    # Generate report if diagnostics enabled
    if diagnostics:
        generate_hyperparameter_report(all_results, best_config, results_dir, evaluation_metric)
    
    return all_results, best_config

def run_hyperparameter_tuning(
    split_datasets: Dict,
    stations_results: Dict,
    output_path: Path,
    base_config: Dict = None,
    search_type: str = 'tpe',
    n_trials: int = None,
    diagnostics: bool = False,
    data_sample_ratio: float = 1.0
):
    """Run hyperparameter tuning with more conservative ranges."""
    # Create output directory
    output_dir = output_path / "diagnostics" / "hyperparameter_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define very conservative parameter grid for initial testing
    param_grid = {
        # Architecture
        'sequence_length': [48, 96, 144],
        'lstm_units': [
            [32, 16],
            [64, 32],
            [64, 32, 16]
        ],
        'dropout_rate': [0.2, 0.3, 0.4],
        'use_bidirectional': [True, False],
        'conv_filters': [32, 64],
        'kernel_size': [3, 5],
        
        # Training
        'learning_rate': [0.05, 0.01, 0.1],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'rmsprop'],
        'loss_function': ['mse', 'mae'],
        
        # Anomaly-specific
        'iterative_training.enabled': [True, False],
        'iterative_training.max_iterations': [2, 3],
        'confidence_threshold_std': [1.5, 2.0, 2.5],
        
        # Feature-related
        'feature_selection': [
            ['Value'],
            ['Value', 'roc_1h', 'roc_3h'],
            ['Value', 'roc_1h', 'roc_3h', 'rolling_mean_6h', 'rolling_std_6h']
        ]
    }
    
    print("\nUsing conservative hyperparameter ranges for initial testing")
    print("Parameter grid:", param_grid)
    
    # Validate input data
    print("\nValidating input data...")
    valid_data = {}
    
    for year, stations in split_datasets['windows'].items():
        for station_name, station_data in stations.items():
            if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
                data = station_data['vst_raw']
                
                # Basic data validation
                if data.empty:
                    print(f"Skipping empty dataset: {station_name} ({year})")
                    continue
                    
                if data.isna().any().any():
                    print(f"Warning: NaN values in {station_name} ({year})")
                    data = data.fillna(method='ffill').fillna(method='bfill')
                
                # Check data range
                data_min = data.min().min()
                data_max = data.max().max()
                print(f"\nStation {station_name} ({year}):")
                print(f"  Samples: {len(data)}")
                print(f"  Value range: [{data_min:.2f}, {data_max:.2f}]")
                
                if np.isinf(data_min) or np.isinf(data_max):
                    print("  Skipping due to infinite values")
                    continue
                
                # Store valid data
                if year not in valid_data:
                    valid_data[year] = {}
                valid_data[year][station_name] = station_data
    
    if not valid_data:
        raise ValueError("No valid data found for hyperparameter tuning")
    
    # Replace split_datasets with validated data
    split_datasets['windows'] = valid_data
    
    print(f"\nProceeding with {sum(len(stations) for stations in valid_data.values())} valid stations")
    
    # Prepare data for hyperparameter tuning
    train_data = {}
    val_data = {}
    test_data = {}
    ground_truth = {}
    
    # Collect data from all years
    for year, stations in split_datasets['windows'].items():
        for station_name, station_data in stations.items():
            if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
                # Create a unique key for each station-year combination
                station_year_key = f"{station_name}_{year}"
                
                # Split data into train/val (e.g., 80/20)
                data = station_data['vst_raw']
                train_size = int(len(data) * 0.8)
                
                # Store in dictionaries
                train_data[station_year_key] = {'vst_raw': data.iloc[:train_size].copy()}
                val_data[station_year_key] = {'vst_raw': data.iloc[train_size:].copy()}
                
                # Add test data with synthetic errors
                if 'vst_raw_modified' in station_data and station_data['vst_raw_modified'] is not None:
                    test_data[station_year_key] = {
                        'vst_raw': data.copy(),
                        'vst_raw_modified': station_data['vst_raw_modified'].copy()
                    }
                    
                    # Add ground truth
                    if station_year_key in stations_results:
                        ground_truth[station_year_key] = stations_results[station_year_key]['ground_truth']
    
    # Sample data if ratio is less than 1.0
    if data_sample_ratio < 1.0:
        print(f"Sampling data for hyperparameter tuning (ratio: {data_sample_ratio:.2f})")
        
        # Sample stations first (if we have many)
        station_keys = list(train_data.keys())
        np.random.seed(42)  # For reproducibility
        
        if len(station_keys) > 5:  # If we have many stations, sample them
            num_stations = max(2, int(len(station_keys) * data_sample_ratio))
            selected_stations = np.random.choice(station_keys, size=num_stations, replace=False)
            
            # Filter data to only include selected stations
            train_data = {k: v for k, v in train_data.items() if k in selected_stations}
            val_data = {k: v for k, v in val_data.items() if k in selected_stations}
            test_data = {k: v for k, v in test_data.items() if k in selected_stations}
            ground_truth = {k: v for k, v in ground_truth.items() if k in selected_stations}
            
            print(f"Selected {num_stations} stations for hyperparameter tuning")
        
        # Then sample data points with temporal awareness
        for station_key in list(train_data.keys()):
            if 'vst_raw' in train_data[station_key]:
                # Get the full dataframe
                df = train_data[station_key]['vst_raw']
                
                # Ensure representation from different time periods
                sampled_indices = []
                
                # Get all years in the dataset
                if hasattr(df.index, 'year'):
                    years = df.index.year.unique()
                    
                    # Group years into decades or periods
                    periods = {}
                    for year in years:
                        decade = (year // 10) * 10  # 1990s, 2000s, 2010s, 2020s
                        if decade not in periods:
                            periods[decade] = []
                        periods[decade].append(year)
                    
                    # Sample from each period
                    for decade, decade_years in periods.items():
                        # For each decade, sample some years
                        sampled_years = np.random.choice(
                            decade_years, 
                            size=min(len(decade_years), max(1, int(len(decade_years) * data_sample_ratio))),
                            replace=False
                        )
                        
                        # For each sampled year, sample some data
                        for year in sampled_years:
                            year_data = df[df.index.year == year]
                            if len(year_data) > 0:
                                # Sample across seasons (quarters)
                                for quarter in range(1, 5):
                                    quarter_data = year_data[year_data.index.quarter == quarter]
                                    if len(quarter_data) > 0:
                                        # Sample some data from this quarter
                                        n_quarter_samples = max(50, int(len(quarter_data) * data_sample_ratio))
                                        step = max(1, len(quarter_data) // n_quarter_samples)
                                        quarter_indices = np.arange(0, len(quarter_data), step)[:n_quarter_samples]
                                        sampled_indices.extend(quarter_data.iloc[quarter_indices].index)
                else:
                    # Fall back to simple systematic sampling if datetime index not available
                    n_samples = max(100, int(len(df) * data_sample_ratio))
                    step = max(1, len(df) // n_samples)
                    sampled_indices = np.arange(0, len(df), step)[:n_samples]
                    
                # Apply the sampling    
                if len(sampled_indices) > 0:
                    train_data[station_key]['vst_raw'] = df.loc[sampled_indices].copy()
                else:
                    # Fallback if sampling failed
                    n_samples = max(100, int(len(df) * data_sample_ratio))
                    train_data[station_key]['vst_raw'] = df.sample(n=n_samples).copy()
                
                # Do the same for validation data
                if station_key in val_data and 'vst_raw' in val_data[station_key]:
                    df = val_data[station_key]['vst_raw']
                    sampled_indices = []
                    
                    # Apply the same temporal sampling logic
                    if hasattr(df.index, 'year'):
                        years = df.index.year.unique()
                        periods = {}
                        for year in years:
                            decade = (year // 10) * 10
                            if decade not in periods:
                                periods[decade] = []
                            periods[decade].append(year)
                        
                        for decade, decade_years in periods.items():
                            sampled_years = np.random.choice(
                                decade_years, 
                                size=min(len(decade_years), max(1, int(len(decade_years) * data_sample_ratio))),
                                replace=False
                            )
                            
                            for year in sampled_years:
                                year_data = df[df.index.year == year]
                                if len(year_data) > 0:
                                    for quarter in range(1, 5):
                                        quarter_data = year_data[year_data.index.quarter == quarter]
                                        if len(quarter_data) > 0:
                                            n_quarter_samples = max(25, int(len(quarter_data) * data_sample_ratio))
                                            step = max(1, len(quarter_data) // n_quarter_samples)
                                            quarter_indices = np.arange(0, len(quarter_data), step)[:n_quarter_samples]
                                            sampled_indices.extend(quarter_data.iloc[quarter_indices].index)
                    else:
                        n_samples = max(50, int(len(df) * data_sample_ratio))
                        step = max(1, len(df) // n_samples)
                        sampled_indices = np.arange(0, len(df), step)[:n_samples]
                    
                    if len(sampled_indices) > 0:
                        val_data[station_key]['vst_raw'] = df.loc[sampled_indices].copy()
                    else:
                        n_samples = max(50, int(len(df) * data_sample_ratio))
                        val_data[station_key]['vst_raw'] = df.sample(n=n_samples).copy()
                
                # And for test data
                if station_key in test_data:
                    if 'vst_raw' in test_data[station_key]:
                        df = test_data[station_key]['vst_raw']
                        sampled_indices = []
                        
                        # Apply the same temporal sampling logic
                        if hasattr(df.index, 'year'):
                            # Similar temporal sampling as above
                            years = df.index.year.unique()
                            periods = {}
                            for year in years:
                                decade = (year // 10) * 10
                                if decade not in periods:
                                    periods[decade] = []
                                periods[decade].append(year)
                            
                            for decade, decade_years in periods.items():
                                sampled_years = np.random.choice(
                                    decade_years, 
                                    size=min(len(decade_years), max(1, int(len(decade_years) * data_sample_ratio))),
                                    replace=False
                                )
                                
                                for year in sampled_years:
                                    year_data = df[df.index.year == year]
                                    if len(year_data) > 0:
                                        for quarter in range(1, 5):
                                            quarter_data = year_data[year_data.index.quarter == quarter]
                                            if len(quarter_data) > 0:
                                                n_quarter_samples = max(25, int(len(quarter_data) * data_sample_ratio))
                                                step = max(1, len(quarter_data) // n_quarter_samples)
                                                quarter_indices = np.arange(0, len(quarter_data), step)[:n_quarter_samples]
                                                sampled_indices.extend(quarter_data.iloc[quarter_indices].index)
                        else:
                            n_samples = max(50, int(len(df) * data_sample_ratio))
                            step = max(1, len(df) // n_samples)
                            sampled_indices = np.arange(0, len(df), step)[:n_samples]
                        
                        if len(sampled_indices) > 0:
                            test_data[station_key]['vst_raw'] = df.loc[sampled_indices].copy()
                        else:
                            n_samples = max(50, int(len(df) * data_sample_ratio))
                            test_data[station_key]['vst_raw'] = df.sample(n=n_samples).copy()
                    
                    # Also sample the modified test data (with synthetic errors)
                    if 'vst_raw_modified' in test_data[station_key]:
                        df = test_data[station_key]['vst_raw_modified']
                        
                        # Use the same indices as for the raw test data to maintain alignment
                        if 'vst_raw' in test_data[station_key]:
                            # Get the indices from the already sampled raw data
                            raw_indices = test_data[station_key]['vst_raw'].index
                            # Use the same indices for modified data
                            test_data[station_key]['vst_raw_modified'] = df.loc[raw_indices].copy()
                        else:
                            # Fallback if raw data sampling failed
                            sampled_indices = []
                            if hasattr(df.index, 'year'):
                                # Similar temporal sampling as above
                                years = df.index.year.unique()
                                periods = {}
                                for year in years:
                                    decade = (year // 10) * 10
                                    if decade not in periods:
                                        periods[decade] = []
                                    periods[decade].append(year)
                                
                                for decade, decade_years in periods.items():
                                    sampled_years = np.random.choice(
                                        decade_years, 
                                        size=min(len(decade_years), max(1, int(len(decade_years) * data_sample_ratio))),
                                    replace=False
                                )
                                
                                for year in sampled_years:
                                    year_data = df[df.index.year == year]
                                    if len(year_data) > 0:
                                        for quarter in range(1, 5):
                                            quarter_data = year_data[year_data.index.quarter == quarter]
                                            if len(quarter_data) > 0:
                                                n_quarter_samples = max(25, int(len(quarter_data) * data_sample_ratio))
                                                step = max(1, len(quarter_data) // n_quarter_samples)
                                                quarter_indices = np.arange(0, len(quarter_data), step)[:n_quarter_samples]
                                                sampled_indices.extend(quarter_data.iloc[quarter_indices].index)
                            else:
                                n_samples = max(50, int(len(df) * data_sample_ratio))
                                step = max(1, len(df) // n_samples)
                                sampled_indices = np.arange(0, len(df), step)[:n_samples]
                            
                            if len(sampled_indices) > 0:
                                test_data[station_key]['vst_raw_modified'] = df.loc[sampled_indices].copy()
                            else:
                                n_samples = max(50, int(len(df) * data_sample_ratio))
                                test_data[station_key]['vst_raw_modified'] = df.sample(n=n_samples).copy()
    
        print(f"Data sampling complete. Using approximately {data_sample_ratio:.0%} of the original data.")
    
    # Run hyperparameter search
    results, best_config = run_hyperparameter_search(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        ground_truth=ground_truth,
        param_grid=param_grid,
        output_dir=output_dir,
        n_trials=n_trials,
        search_type=search_type,
        base_config=base_config,
        evaluation_metric='reconstruction_error',
        diagnostics=diagnostics
    )
    
    # Save the results
    print("\nSaving hyperparameter tuning results...")
    try:
        save_hyperparameter_results(results, best_config, output_dir)
        print(f"Results saved to {output_dir / 'hyperparameter_results.json'}")
    except Exception as e:
        print(f"Error saving hyperparameter results: {e}")
        import traceback
        print(traceback.format_exc())
    
    return best_config

def load_best_hyperparameters(output_dir: Path, base_config: Dict = None) -> Dict:
    """Load the best hyperparameters from a previous tuning run."""
    results_file = output_dir / "diagnostics" / "hyperparameter_tuning" / "hyperparameter_results.json"
    
    print(f"Looking for hyperparameter results at: {results_file}")
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"Loaded best hyperparameters from {results_file}")
            
            # Use the best configuration if available
            if 'best_config' in results:
                best_config = results['best_config']
                
                # Start with base config or empty dict
                config = base_config.copy() if base_config else {}
                
                # Track and show changes
                print("Found saved hyperparameters. Updating configuration:")
                for key, value in best_config.items():
                    # Skip iterative_training settings - we'll restore those separately
                    if key != 'iterative_training':
                        if key in config:
                            print(f"  {key}: {config[key]} -> {value}")
                        else:
                            print(f"  {key}: None -> {value}")
                        config[key] = value
                
                # Preserve iterative training settings from base config 
                if 'iterative_training' in base_config:
                    print(f"  Preserving iterative_training settings from base config: {base_config['iterative_training']}")
                    config['iterative_training'] = base_config['iterative_training']
                
                return config
            else:
                print("No best configuration found in results")
                return base_config
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            return base_config
    else:
        print("No previous hyperparameter tuning results found")
        return base_config 