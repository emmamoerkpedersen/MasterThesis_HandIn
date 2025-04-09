import os
import json
import numpy as np
import optuna
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from experiments.Improved_model_structure.train_model import LSTM_Trainer

def objective(trial, train_data, val_data, base_config):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        train_data: Training data
        val_data: Validation data
        base_config: Base configuration dictionary to modify
    
    Returns:
        float: Validation loss (to be minimized)
    """
    # Copy base configuration so we don't modify the original
    config = base_config.copy()
    
    # Define hyperparameters to optimize
    config['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 128])#, 256])#, 512])
    config['num_layers'] = trial.suggest_int('num_layers', 1, 2, step=1)
    config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    config['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
    config['sequence_length'] = trial.suggest_categorical('sequence_length', [1000, 1500, 2000])#, 2500, 3000])
    
    # Additional hyperparameters specific to our model
    if config.get('use_peak_weighted_loss', False):
        config['peak_weight'] = trial.suggest_float('peak_weight', 1.0, 3.0)
    
    config['grad_clip_value'] = trial.suggest_float('grad_clip_value', 0.5, 2.0)
    
    if config.get('use_smoothing', False):
        config['smoothing_alpha'] = trial.suggest_float('smoothing_alpha', 0.1, 0.5)
    
    # Print current trial configuration
    print(f"\nTrial {trial.number} Configuration:")
    for key, value in config.items():
        if key in ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 'sequence_length']:
            print(f"  {key}: {value}")
    
    try:
        # Create preprocessor and trainer with the trial configuration
        preprocessor = DataPreprocessor(config)
        trainer = LSTM_Trainer(config, preprocessor)
        
        # Progress tracking callback for reporting intermediate values
        def report_intermediate_value(epoch, train_loss, val_loss):
            if np.isfinite(val_loss):
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        # Check data validity
        if train_data is None or val_data is None or len(train_data) == 0 or len(val_data) == 0:
            print("Error: Invalid training or validation data")
            return float('inf')
            
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        
        # Train with same conditions as actual training
        history, _, _ = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=300,
            batch_size=config['batch_size'],
            patience=15,
            epoch_callback=report_intermediate_value
        )
        
        if not history or len(history.get('val_loss', [])) == 0:
            return float('inf')
        
        # Track both raw and best validation loss
        best_val_loss = min(history['val_loss'])
        final_val_loss = history['val_loss'][-1]
        
        print(f"Trial {trial.number} results:")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Final validation loss: {final_val_loss:.6f}")
        
        # Return best validation loss to match actual training metric
        return best_val_loss
        
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        print(traceback.format_exc())
        return float('inf')

def run_hyperparameter_tuning(train_data, val_data, output_path, base_config, n_trials=30):
    """
    Run hyperparameter tuning with Optuna.
    
    Args:
        train_data: Training data
        val_data: Validation data
        output_path: Path to save results
        base_config: Base configuration dictionary
        n_trials: Number of trials to run
    
    Returns:
        dict: Best configuration found
        optuna.Study: Completed study object
    """
    # Create results directory if it doesn't exist
    results_dir = output_path / "hyperparameter_tuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"lstm_hyperparameter_study_{timestamp}"
    storage_name = f"sqlite:///{results_dir}/optuna_{study_name}.db"
    
    # Check data validity
    if train_data is None or val_data is None:
        print("Error: Training or validation data is None")
        return base_config, None
        
    if len(train_data) == 0 or len(val_data) == 0:
        print("Error: Training or validation data is empty")
        return base_config, None
        
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Create the study
    try:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Run optimization
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        study.optimize(
            lambda trial: objective(trial, train_data, val_data, base_config),
            n_trials=n_trials,
            timeout=36000,  # 10 hours maximum
            show_progress_bar=True,
            catch=(Exception,)  # Catch exceptions to prevent entire study from failing
        )
        
        # Check if we have any completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("No trials completed successfully. Using base configuration.")
            return base_config, study
            
        # Get best parameters from completed trials
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        print(f"Best trial achieved validation loss: {best_value:.6f}")
        print("Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Merge best parameters with base config
        best_config = base_config.copy()
        for param, value in best_params.items():
            best_config[param] = value
        
        # Save best configuration
        save_hyperparameter_results(best_config, study, results_dir)
        
        # Visualize results only if we have valid trials
        if len(completed_trials) >= 1:
            try:
                # Create hyperparameter importance plot
                param_importance_fig = optuna.visualization.plot_param_importances(study)
                param_importance_fig.write_image(str(results_dir / "param_importances.png"))
                
                # Create optimization history plot
                history_fig = optuna.visualization.plot_optimization_history(study)
                history_fig.write_image(str(results_dir / "optimization_history.png"))
                
                # Create parallel coordinate plot if we have at least 2 trials
                if len(completed_trials) >= 2:
                    parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
                    parallel_fig.write_image(str(results_dir / "parallel_coordinate.png"))
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
        
        return best_config, study
    except Exception as e:
        print(f"Error in hyperparameter tuning: {str(e)}")
        print(traceback.format_exc())
        return base_config, None

def save_hyperparameter_results(best_config, study, output_dir):
    """
    Save the best hyperparameters to a JSON file.
    
    Args:
        best_config: Dictionary containing the best configuration
        study: Completed Optuna study object
        output_dir: Directory to save results
    """
    # Save best configuration as JSON
    best_config_path = output_dir / "best_config.json"
    
    # Convert any numpy values to regular Python types for JSON serialization
    best_config_serializable = {}
    for key, value in best_config.items():
        if isinstance(value, np.integer):
            best_config_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            best_config_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            best_config_serializable[key] = value.tolist()
        else:
            best_config_serializable[key] = value
    
    with open(best_config_path, 'w') as f:
        json.dump(best_config_serializable, f, indent=4)
    
    # Save all trials information
    trials_path = output_dir / "all_trials.json"
    trials_data = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_info = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value
            }
            trials_data.append(trial_info)
    
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=4)
    
    print(f"Saved best configuration to {best_config_path}")
    print(f"Saved all trials data to {trials_path}")

def load_best_hyperparameters(output_path, default_config):
    """
    Load best hyperparameters from a previous tuning run.
    
    Args:
        output_path: Path to the output directory
        default_config: Default configuration to use if loading fails
    
    Returns:
        dict: Configuration with best hyperparameters or default config if loading fails
    """
    try:
        best_config_path = output_path / "hyperparameter_tuning" / "best_config.json"
        print(f"Looking for best config at: {best_config_path}")
        
        if best_config_path.exists():
            with open(best_config_path, 'r') as f:
                best_config = json.load(f)
            print(f"Loaded best hyperparameters from {best_config_path}")
            return best_config
        else:
            print(f"No best config found at {best_config_path}, using default configuration")
            return default_config
    except Exception as e:
        print(f"Error loading best hyperparameters: {str(e)}")
        print(traceback.format_exc())
        return default_config
