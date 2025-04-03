"""
Simplified hyperparameter tuning for LSTM forecaster model using Optuna.
"""

import optuna
import torch
import numpy as np
import json
import os
from pathlib import Path
import pickle

from experiments.Improved_model_structure.train_model import DataPreprocessor, LSTM_Trainer


def define_model(trial, preprocessor, base_config):
    """
    Define the model with hyperparameters suggested by Optuna.
    """
    # Define the core hyperparameters to tune
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 350])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    sequence_length = trial.suggest_int('sequence_length', 1000, 5000, step=1000)
    
    # Create configuration for this trial
    config = base_config.copy()
    
    # Update with trial parameters
    config.update({
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'epochs': 100,       # Reduced for faster hyperparameter search
        'patience': 15      # Adjusted for stable early stopping
    })
    
    return config


def objective(trial, train_data, val_data, preprocessor, base_config):
    """
    Objective function for Optuna optimization.
    """
    # Get configuration for this trial
    config = define_model(trial, preprocessor, base_config)
    
    # Initialize trainer
    trainer = LSTM_Trainer(config, preprocessor=preprocessor)
    
    # Simple callback for Optuna pruning
    def epoch_callback(epoch, train_loss, val_loss):
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Train the model
    try:
        history, _, _ = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            patience=config['patience'],
            sequence_length=config['sequence_length'],
            epoch_callback=epoch_callback
        )
        
        # Return the best validation loss for optimization
        best_val_loss = min(history['val_loss'])
        return best_val_loss
    
    except optuna.exceptions.TrialPruned:
        # Handle pruned trials gracefully
        raise
        
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        # Return a high loss value to indicate failure
        return float('inf')


def run_hyperparameter_tuning(train_data, val_data, output_path, base_config, n_trials=30):
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        train_data: Training data
        val_data: Validation data
        output_path: Path to save results
        base_config: Base configuration to use
        n_trials: Number of trials to run
    
    Returns:
        Best hyperparameters, Optuna study
    """
    from experiments.Improved_model_structure.train_model import DataPreprocessor
    
    # Initialize preprocessor with base configuration
    preprocessor = DataPreprocessor(base_config)
    
    # Create a study object and optimize
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    
    try:
        # Run optimization with progress output
        study.optimize(
            lambda trial: objective(trial, train_data, val_data, preprocessor, base_config), 
            n_trials=n_trials
        )
        
        # Get best parameters
        best_params = study.best_params
        
        # Combine with base config
        best_config = base_config.copy()
        best_config.update(best_params)
        
        # Save best hyperparameters
        hp_path = output_path / 'best_hyperparameters.json'
        with open(hp_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Save the study for later analysis
        study_path = output_path / 'hyperparameter_study.pkl'
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
            
        print(f"\nBest hyperparameters saved to {hp_path}")
        print("Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
            
        return best_config, study
        
    except Exception as e:
        print(f"Error during hyperparameter optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return base_config, None


def load_best_hyperparameters(output_path, base_config):
    """
    Load best hyperparameters from a previous run if available.
    
    Returns:
        Dictionary with best hyperparameters or base_config if file doesn't exist
    """
    hp_path = output_path / 'best_hyperparameters.json'
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            return json.load(f)
    return base_config